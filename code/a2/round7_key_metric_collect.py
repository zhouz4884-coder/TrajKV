"""Calibration-only moments and actual-score diagnostics on an oldDepth student.

This is an instrumented diagnostic owner retaining original FP16 visual K.
The unchanged Round6 packed reader still produces every model attention output.
No GT answers, centered-query covariance, or deployment-budget claim is used.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/experiments'))

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def ratios(numerator, denominator):
    return [float(a / b) if float(b) > 0 else None for a, b in zip(numerator, denominator)]


def collector_class():
    from fast_wht import wht, torch_wht
    from masked_bank import unique_bytes
    from round6_depth_bank import DepthResidualBank
    from round6_depth_codec import decode_sign_stage

    class MetricCollector(DepthResidualBank):
        sink = None

        def _initialize(self, model, cache, visual, method, legacy_scripts, kivi_source, keep_reference=False):
            # Clone before PhaseState extracts/replaces the native cache owner.
            self.original_keys_diagnostic = [layer.keys.index_select(2, visual).detach().clone()
                                             for layer in cache.layers]
            if any(value.dtype != torch.float16 for value in self.original_keys_diagnostic):
                raise ValueError('Original diagnostic visual K must be FP16')
            self.measured = {}
            self.metric_diagnostics = []
            super()._initialize(model, cache, visual, method, legacy_scripts, kivi_source, False)
            sink = type(self).sink
            layers, heads = len(self.states), self.states[0].text_k.shape[1]
            if sink.get('q_outer_sum') is None:
                sink.update(q_outer_sum=torch.zeros(layers, heads, 128, 128, dtype=torch.float64),
                            q_sum=torch.zeros(layers, heads, 128, dtype=torch.float64),
                            q_count=torch.zeros(layers, heads, dtype=torch.int64),
                            phase_count={name: torch.zeros(layers, heads, dtype=torch.int64)
                                         for name in ('prefill', 'decode')})

        def owned(self):
            result = super().owned()
            refs = unique_bytes(self.original_keys_diagnostic)
            result['diagnostic_original_visual_K_bytes'] = refs
            result['diagnostic_fp16_reference'] += refs
            result['total_unique'] += refs
            result['full_visual_retained'] = True
            packed = sum(result[name] for name in ('visual_base', 'visual_residual', 'selector_metadata', 'visual_position')) + self.encoder_parameter_bytes
            result['diagnostic_complete_visual_owner_bytes'] = packed + refs
            result['diagnostic_actual_visual_bits_including_encoder'] = 16 * (packed + refs) / self.visual_full_bytes
            return result

        def _actual_rotated_key(self, layer):
            state = self.states[layer]
            decoded = (self.book.float()[state.direction_code.long()].flatten(-2) * state.direction_gain.float()).half()
            for head in range(decoded.shape[1]):
                for stage in (0, 1):
                    slot = int(state.depth_residual_lookup[0, head, stage])
                    if slot >= 0:
                        decoded[0, head] = decode_sign_stage(decoded[0, head],
                            state.depth_residual_code[slot], state.depth_residual_alpha[slot])
            return decoded

        def read(self, q, layer):
            # Every emitted model value comes directly from the frozen reader.
            actual = super().read(q, layer)
            if self.turn != 1 or self.pass_kind != 'committed':
                return actual
            phase = 'prefill' if 'prefill' in self.phase else 'decode'
            slot = (layer, phase)
            seen = self.measured.get(slot, 0)
            if seen >= (1 if phase == 'prefill' else 2):
                return actual
            self.measured[slot] = seen + 1
            self._measure(q[:, :, -1:], layer, phase, seen)
            return actual

        def _measure(self, query, layer, phase, index):
            state, original = self.states[layer], self.original_keys_diagnostic[layer]
            heads = original.shape[1]
            groups = query.shape[1] // heads
            qr = wht(query, half=True)
            z = qr.float().reshape(heads, groups, 128)
            sink = type(self).sink
            sink['q_outer_sum'][layer] += (z.transpose(-1, -2) @ z).double().cpu()
            sink['q_sum'][layer] += z.sum(1).double().cpu()
            sink['q_count'][layer] += groups
            sink['phase_count'][phase][layer] += groups
            khat_rot = self._actual_rotated_key(layer)
            qn = query.float().reshape(heads, groups, 128)
            original_h = original[0].float()
            hat_h = khat_rot[0].float()
            # Dense FP32 accumulation from the same FP16 input values as reader.
            score_hat = (z @ hat_h.transpose(-1, -2)) / math.sqrt(128)
            score_hat += (qn * state.direction_mean[0].float()).sum(-1, keepdim=True) / math.sqrt(128)
            score_original = (qn @ original_h.transpose(-1, -2)) / math.sqrt(128)
            actual_delta = score_hat - score_original
            # This materialized native reconstruction is only a bridge diagnostic.
            native_hat = (torch_wht(khat_rot.float()) + state.direction_mean.float()).half()
            native_delta = native_hat[0].float() - original_h
            delta_wht = torch_wht(native_delta)
            native_bridge_score_delta = (qn @ native_delta.transpose(-1, -2)) / math.sqrt(128)
            rounded_wht_bridge_delta = (z @ delta_wht.transpose(-1, -2)) / math.sqrt(128)
            q_energy = z.square().sum((-2, -1))
            delta_energy = delta_wht.square().sum((-2, -1))
            projected_energy = (z @ delta_wht.transpose(-1, -2)).square().sum((-2, -1))
            actual_energy = actual_delta.square().sum((-2, -1))
            bridge_energy = native_bridge_score_delta.square().sum((-2, -1))
            row = dict(layer=layer, phase=phase, sample_index=index, query_phase=self.phase,
                grouped_query_heads=groups, visual_tokens=original.shape[-2],
                actual_score_error_energy_by_head=actual_energy.cpu().tolist(),
                actual_score_error_mse_by_head=(actual_energy / (groups * original.shape[-2])).cpu().tolist(),
                actual_score_error_max_abs_by_head=actual_delta.abs().amax((-2, -1)).cpu().tolist(),
                native_bridge_score_error_energy_by_head=bridge_energy.cpu().tolist(),
                native_bridge_delta_energy_by_head=delta_energy.cpu().tolist(),
                query_uncentered_energy_by_head=q_energy.cpu().tolist(),
                rho_native_bridge_wht_by_head=ratios(128 * projected_energy, q_energy * delta_energy),
                actual_to_native_bridge_energy_ratio_by_head=ratios(actual_energy, bridge_energy),
                native_bridge_transport_remainder_energy_by_head=(actual_delta - native_bridge_score_delta).square().sum((-2, -1)).cpu().tolist(),
                rounded_wht_bridge_remainder_energy_by_head=(actual_delta - rounded_wht_bridge_delta).square().sum((-2, -1)).cpu().tolist(),
                scope='paired actual student query and same-image K error; visual logits only, no softmax/output/task guarantee')
            self.metric_diagnostics.append(row)

        def result_metadata(self):
            result = super().result_metadata()
            diagnostic = self.owned()
            result.update(task_accuracy_eligible=False, deployment_budget_eligible=False,
                diagnostic_collector=True, diagnostic_original_visual_K_retained=True,
                diagnostic_original_visual_K_bytes=diagnostic['diagnostic_original_visual_K_bytes'],
                packed_base_complete_budget_bits=self.actual_budget_bits,
                depth_residual_source_fp16_retained=True, depth_residual_budget_verified=False,
                depth_residual_actual_budget_bits=diagnostic['diagnostic_actual_visual_bits_including_encoder'],
                sparse_residual_actual_budget_bits=diagnostic['diagnostic_actual_visual_bits_including_encoder'],
                diagnostic_actual_visual_bits_including_encoder=diagnostic['diagnostic_actual_visual_bits_including_encoder'],
                query_metric_diagnostics=self.metric_diagnostics,
                query_metric_sampling='first question only; final query position in prefill and first two available committed decode reads; all GQA query heads',
                query_metric_scope='uncentered second moment of actual FP16 WHT(q) from oldDepth student; q mean never subtracted')
            return result

    return MetricCollector


def export_metrics(output, sink, manifest):
    counts = sink['q_count']
    if bool((counts <= 0).any()):
        raise RuntimeError('Every layer/KV head needs at least one actual sampled query')
    moment = sink['q_outer_sum'] / counts[..., None, None]
    moment = .5 * (moment + moment.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(moment)
    lambda1 = eigenvalues[..., -1].clamp_min(0)
    rest = eigenvalues[..., :-1].sum(-1).clamp_min(0) / 127
    # Zero isotropic scale is made explicit; no arbitrary infinite gamma artifact.
    floor = 1e-12
    gamma = (lambda1 / rest.clamp_min(floor) - 1).clamp_min(0).float()
    u = eigenvectors[..., -1].half()
    raw = dict(q_outer_sum=sink['q_outer_sum'], q_sum=sink['q_sum'], q_count=counts,
               phase_count=sink['phase_count'], uncentered_second_moment=moment,
               eigenvalues=eigenvalues, deployed_u_fp16=u, gamma_uncapped_fp32=gamma)
    raw_path = output / 'moments.pt'
    torch.save(raw, raw_path)
    raw_hash = sha(raw_path)
    files = []
    for name, cap in (('metric_cap16.json', 16.), ('metric_uncapped.json', None)):
        deployed_gamma = gamma if cap is None else gamma.clamp_max(cap)
        record = dict(schema='trajkv_round7_rank1_metric_v1', basis='centered_wht', calibration_only=True,
            u=u.float().tolist(), gamma=deployed_gamma.tolist(),
            inputHash=raw_hash, source_moments_sha256=raw_hash,
            samples_sha256=manifest['samples_sha256'], calibration_case_ids=manifest['selected_ids'],
            query_count_per_layer_head=counts.tolist(), query_phase_count={key: value.tolist() for key, value in sink['phase_count'].items()},
            rank1_approximation='M/mean_rest approximated by I+gamma*u*u^T; not the original full query/task metric',
            gamma_formula='max(lambda_top/max(mean(other 127 eigenvalues),1e-12)-1,0)',
            gamma_cap=cap, zero_rest_floor=floor, query_mean_subtracted=False,
            query_source='actual oldDepth student question/decode forwards; FP16 WHT(q), uncentered outer products; all GQA heads',
            labels_used=False, GT_answers_used=False, task_scores_fit=False,
            deployed_u_dtype='FP16', deployed_gamma_dtype='FP32', u_silent_renormalization=False,
            tensor_bytes=u.numel() * 2 + deployed_gamma.numel() * 4,
            scope='calibration-derived encoder-only approximation; serving constants require full independent byte charge')
        path = output / name
        write_json(path, record)
        files.append(dict(path=name, sha256=sha(path), gamma_cap=cap, tensor_bytes=record['tensor_bytes']))
    stats = dict(schema='trajkv_round7_actual_query_moments_v1', query_mean_subtracted=False,
        q_count=counts.tolist(), q_sum=sink['q_sum'].tolist(),
        phase_count={key: value.tolist() for key, value in sink['phase_count'].items()},
        eigvalsh=eigenvalues.tolist(), top_eigenvalue=lambda1.tolist(), mean_other_eigenvalues=rest.tolist(),
        gamma_uncapped=gamma.tolist(), deployed_u_fp16_norm=u.float().norm(dim=-1).tolist(),
        zero_rest_heads=int((rest <= floor).sum()), moments_sha256=raw_hash,
        metric_files=files, labels_used=False, deployment_budget_eligible=False,
        scope='uncentered student-query moments and native-reconstruction bridge diagnostics; rho does not gate execution or imply achievable task benefit')
    write_json(output / 'stats.json', stats)
    return files, raw_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if not torch.cuda.is_available():
        raise RuntimeError('This collector requires actual CUDA oldDepth student forwards')
    variants = config['variants']
    if len(variants) != 1 or variants[0]['method'] != 'round6_depth':
        raise ValueError('Exactly one frozen oldDepth round6_depth baseline variant is required')
    variant = variants[0]
    for key, value in config['arguments'].items():
        setattr(args, key, Path(value) if key in ('model', 'legacy_scripts', 'kivi_source', 'samples') else value)
    for key, value in dict(turns=1, max_tokens=8, max_patches=16, margin=0., max_replays=0,
                           save_states=False, read_budget_bytes=None, gain_table_data=None).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if args.turns != 1 or getattr(args, 'native_question_history', False):
        raise ValueError('Collector must use image-boundary complete first-question replay')
    args.max_patches = int(variant.get('cap', args.max_patches))
    data = json.loads(args.samples.read_text(encoding='utf-8-sig'))
    count, start = int(config.get('count', 16)), int(config.get('start', 0))
    if not (1 <= count <= 8 or count == 16) or start != 0:
        raise ValueError('Use a fixed first 1..16 cal images; prototype count<=8 and full cal16 are separate runs')
    rows = data[:count]
    if len(rows) != count or len({row['source_image_id'] for row in rows}) != count:
        raise ValueError('Need the requested number of distinct calibration images')
    if any(len(row['questions']) != 1 for row in rows):
        raise ValueError('Use first-question-only input rows')
    forbidden = {'answer', 'answers', 'reference_answers', 'ground_truth', 'gt_answer', 'targets'}
    if any(forbidden.intersection(row) for row in rows):
        raise ValueError('Calibration session rows must not contain GT answer fields')
    args.output.mkdir(parents=True, exist_ok=False)
    from bank import install_legacy
    install_legacy(args.legacy_scripts)
    import session_boundary
    from transformers import AutoProcessor, AutoModelForImageTextToText
    torch.set_num_threads(4)
    torch.manual_seed(1212)
    torch.backends.cuda.matmul.allow_tf32 = False
    selected_inputs = []
    for row in rows:
        image_paths = row.get('remote_images', row['images'])
        hashes = [sha(path) for path in image_paths]
        if row.get('hashes') and hashes != row['hashes']:
            raise ValueError('Calibration image bytes differ from source manifest')
        selected_inputs.append(dict(id=row['id'], image_hashes=hashes, source_image_id=row['source_image_id']))
    source_names = ('round7_key_metric_collect.py', 'round6_depth_bank.py', 'round6_depth_attention.py',
                    'round6_depth_codec.py', 'fast_wht.py', 'quantizers.py', 'session_boundary.py',
                    'metadata_direction_bank.py', 'learned_direction_bank.py', 'masked_bank.py')
    manifest = dict(schema='trajkv_round7_key_metric_collect_manifest_v1', config=config,
        config_sha256=sha(args.config), samples_sha256=sha(args.samples), selected_ids=[row['id'] for row in rows],
        selected_inputs=selected_inputs, calibration_count=count,
        effective_max_patches=args.max_patches,
        labels_used=False, answers_visible_to_collector=False, task_scores_fit=False,
        diagnostic_source_FP16_K_retained=True, deployment_budget_eligible=False,
        source_reader='frozen round6_depth_attention unchanged; super.read returns all model outputs',
        sampling='first image-boundary question prefill last query and first two available decode queries; all GQA heads equally weighted',
        query_mean_subtracted=False, query_metric='uncentered second moment of actual reader FP16 WHT(q)',
        moment_accumulation='per-sample GQA outer/sum FP32 with TF32 disabled; cross-sample accumulation FP64 on CPU',
        rho_scope='native FP16 reconstruction bridge, WHT basis, same-image paired sampled query/token supports; diagnostic only',
        sources={name: sha(Path(__file__).with_name(name)) for name in source_names},
        external_codec_inputs={variant[key]: sha(variant[key]) for key in ('direction_book_path', 'metadata_factor_path')},
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__)
    write_json(args.output / 'manifest.json', manifest)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, local_files_only=True,
        dtype=torch.float16, device_map='cuda:0', attn_implementation='sdpa').eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    collector = collector_class()
    collector.settings, collector.sink = variant, {}
    previous_bank = session_boundary.ProgressiveBank
    session_boundary.ProgressiveBank = collector
    try:
        with torch.inference_mode(), (args.output / 'cases.jsonl').open('w', encoding='utf-8') as stream:
            for row in rows:
                result, _ = session_boundary.run_session(model, processor, row, args, 'round6_depth', 'base', keep_reference=False)
                case = dict(session_id=row['id'], visual_tokens=result['visual_tokens'],
                    owned_initial=result['owned_initial'], owned_final=result['owned_final'],
                    diagnostic_original_visual_K_retained=True, deployment_budget_eligible=False,
                    diagnostic_actual_visual_bits_including_encoder=result['diagnostic_actual_visual_bits_including_encoder'],
                    actual_query_error_diagnostics=result['query_metric_diagnostics'],
                    generated_tokens=len(result['turns'][0]['token_ids']),
                    scope='unlabelled actual oldDepth student query collection; generated answers are not fitted or used as labels')
                stream.write(json.dumps(case, allow_nan=False) + '\n')
                stream.flush()
                print(json.dumps(dict(done=row['id'], visual_tokens=case['visual_tokens'], diagnostics=len(case['actual_query_error_diagnostics']))), flush=True)
    finally:
        session_boundary.ProgressiveBank = previous_bank
    files, moments_hash = export_metrics(args.output, collector.sink, manifest)
    for path, original_hash in manifest['external_codec_inputs'].items():
        if sha(path) != original_hash:
            raise RuntimeError('A frozen codec input changed during collection')
    write_json(args.output / 'complete.json', dict(status='COMPLETE', cases=len(rows), metric_files=files,
        moments_sha256=moments_hash, labels_used=False, deployment_budget_eligible=False))


if __name__ == '__main__':
    main()
