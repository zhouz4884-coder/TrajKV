"""Four calibration-holdout M1 suffix-TF diagnostics, no answer tokens.

F is native, L is original low43 and V2R is the actual new packed bank. The
native reference is deliberately retained for numerical comparisons. This is
QA0 and is not production memory, speed, quality, or independent-confirmation
evidence. No factors, codebooks or candidate parameters are fitted here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))


def require(value, message):
    if not value:
        raise ValueError('ROUND12_VALUE_OFFSET_DIAGNOSTIC: ' + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_logits(output, ident, branch, tensor):
    import torch
    folder = output / 'first_logits'
    folder.mkdir(exist_ok=True)
    path = folder / (hashlib.sha256(ident.encode()).hexdigest()[:20] + '_' + branch + '.pt')
    with path.open('xb') as f:
        torch.save({'id': ident, 'branch': branch, 'raw_last_suffix_logits': tensor,
                    'generated_answer_tokens': 0, 'GT_loaded': False}, f)
    return {'file': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def measured_forward(report, ident, phase, call):
    import torch
    event = {'id': ident, 'phase': phase, 'attempted': True, 'completed': False}
    report['model_forward_events'].append(event)
    started = time.perf_counter()
    try:
        result = call()
        torch.cuda.synchronize()
        event['completed'] = True
        return result
    except Exception as exc:
        event.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        event['diagnostic_wall_seconds'] = time.perf_counter() - started


def select_holdout(calibration, candidate):
    path = Path(calibration['cal_select_inputs'])
    require(sha(path) == calibration['cal_select_inputs_sha256'], 'cal_select source SHA differs')
    rows = [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    forbidden = {'answers', 'answer', 'refs', 'references', 'labels', 'scores', 'official_score', 'reference_answers'}
    require(all(not (set(row) & forbidden) for row in rows), 'answer/reference fields in calibration inputs')
    selected = []
    for domain in ('textvqa', 'infovqa', 'docvqa', 'chartqa'):
        matches = [row for row in rows if row['dataset'] == domain]
        require(bool(matches), 'missing calibration domain ' + domain)
        selected.append(matches[0])
    require(len(selected) == len({r['id'] for r in selected}) == len({r['original_image_sha256'] for r in selected}) == 4,
            'exactly four unique source/images required')
    ratio = read(candidate['ratio_path'])
    held = {r['id']: r['image_sha256'] for r in ratio['source_events'] if r['role'] == 'holdout'}
    require(held == {r['id']: r['original_image_sha256'] for r in selected}, 'not the exact ratio-asset holdout4')
    forbidden_images = set(calibration['development_image_sha256'])
    require(not forbidden_images & {r['original_image_sha256'] for r in selected}, 'diagnostic/development image overlap')
    return selected


def logits_summary(reference, actual):
    import torch
    require(reference.ndim == actual.ndim == 1 and reference.shape == actual.shape, 'logit vocabulary mismatch')
    require(bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()), 'nonfinite first suffix logits')
    lf = torch.log_softmax(reference.double(), -1)
    lc = torch.log_softmax(actual.double(), -1)
    pf = lf.exp()
    kl = float((pf * (lf - lc)).sum())
    top = torch.topk(actual.float(), 5)
    return {'vocabulary_size': actual.numel(), 'native_to_branch_logit_KL': kl,
            'native_argmax': int(reference.argmax()), 'branch_argmax': int(actual.argmax()),
            'same_argmax': bool(reference.argmax() == actual.argmax()),
            'top5_ids': top.indices.tolist(), 'top5_logits': top.values.tolist(),
            'mean_abs_logit_delta': float((reference.double() - actual.double()).abs().mean()),
            'scope': 'raw final-question-suffix-position logits; no generated or GT answer tokens'}


def value_metrics(native_cache, vp, owners, *, v2r, reference_scales=None):
    import torch
    import round12_static_codec_v1 as codec
    from round6_depth_codec import decode_sign_stage
    result, scales = [], []
    for layer, owner in enumerate(owners):
        state = owner.state
        source = native_cache.layers[layer].values.index_select(2, vp).detach()[0].float()
        levels = getattr(state, 'round12_value_levels', None) if v2r else None
        base = codec.half_value(state.v_code, state.v_scale, state.v_min,
                                bits=2 if v2r else 1, levels=levels).transpose(2, 3).contiguous()[0]
        decoded = base.clone()
        for head in range(4):
            for stage in range(2):
                slot = int(state.depth_residual_lookup[1, head, stage])
                if slot >= 0:
                    decoded[head] = decode_sign_stage(decoded[head], state.depth_residual_code[slot],
                                                     state.depth_residual_alpha[slot])
        scale = state.v_scale[0, :, :, 0].detach().float()
        scales.append(scale.cpu().contiguous().clone())
        s1 = scale if reference_scales is None else reference_scales[layer].to(scale.device)
        offset = owner.shared['encoder_factor.value_center_offsets'][layer].float()[:, None]
        mean = source.mean(1, keepdim=True)
        energy = source.square().sum((1, 2))
        centered_energy = (source - mean).square().sum((1, 2))
        sse0 = (source - base.float()).square().sum((1, 2))
        sse1 = (source - decoded.float()).square().sum((1, 2))
        actual_offset_difference = offset * (scale - s1)
        for head in range(4):
            valid = s1[head].abs() > 1e-12
            ratios = (scale[head][valid] / s1[head][valid]).cpu()
            full_ratio = (scale[head] / torch.where(valid, s1[head], torch.ones_like(s1[head]))).cpu().tolist()
            valid_cpu = valid.cpu().tolist()
            ratio_by_channel = [full_ratio[c] if valid_cpu[c] else None for c in range(128)]
            row = {'layer': layer, 'head': head, 'N': source.shape[-2],
                   'native_energy': float(energy[head]), 'native_centered_energy': float(centered_energy[head]),
                   'base_SSE': float(sse0[head]), 'after_residual_SSE': float(sse1[head]),
                   'base_relative_SSE_native_energy': float(sse0[head] / energy[head].clamp_min(1e-30)),
                   'after_residual_relative_SSE_native_energy': float(sse1[head] / energy[head].clamp_min(1e-30)),
                   'base_relative_SSE_centered_energy': float(sse0[head] / centered_energy[head].clamp_min(1e-30)),
                   'after_residual_relative_SSE_centered_energy': float(sse1[head] / centered_energy[head].clamp_min(1e-30)),
                   'base_mean_error_norm': float((base.float().mean(1) - source.mean(1))[head].norm()),
                   'after_residual_mean_error_norm': float((decoded.float().mean(1) - source.mean(1))[head].norm()),
                   'native_RMS': float(source[head].square().mean().sqrt()),
                   'base_RMS': float(base[head].float().square().mean().sqrt()),
                   'after_residual_RMS': float(decoded[head].float().square().mean().sqrt()),
                   'value_residual_depth': sum(int(state.depth_residual_lookup[1, head, depth]) >= 0 for depth in range(2)),
                   'V1_folded_scale': s1[head].cpu().tolist(), 'branch_folded_scale': scale[head].cpu().tolist(),
                   'scale2_over_scale1': ratio_by_channel,
                   'scale_ratio_valid_channels': int(valid.sum()),
                   'scale_ratio_mean': float(ratios.mean()) if ratios.numel() else None,
                   'scale_ratio_median': float(ratios.median()) if ratios.numel() else None,
                   'scale_ratio_min': float(ratios.min()) if ratios.numel() else None,
                   'scale_ratio_max': float(ratios.max()) if ratios.numel() else None,
                   'frozen_value_center_offset': float(offset[head, 0]),
                   'offset_times_scale_difference': actual_offset_difference[head].cpu().tolist(),
                   'offset_difference_RMS': float(actual_offset_difference[head].square().mean().sqrt()),
                   'interpretation': 'offset*(S2-S1) is inherited affine displacement component, not complete decoded-mean difference'}
            result.append(row)
        del source, base, decoded
    return result, scales


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True, help='An actual frozen V2R run configuration')
    parser.add_argument('--calibration-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config, calibration = read(args.config), read(args.calibration_config)
    require(config.get('candidate', {}).get('mode') == 'V2R', 'use actual V2R candidate configuration')
    require(config.get('question_suffix') == calibration.get('question_suffix'), 'calibration/question suffix differs')
    require(config.get('model_revision') == calibration.get('model_revision'), 'model revision differs')
    require(not (set(config) & {'answers', 'labels', 'references', 'reference_answers', 'scores'}), 'reference fields in config')
    import round10_runner_v1 as frozen
    import round12_runner_v1 as static
    candidate_info, candidate_locks = static.candidate_assets(config)
    variant = frozen.frozen_variant(config)
    rows = select_holdout(calibration, config['candidate'])
    output = args.output.resolve()
    require(str(output).startswith('/media/') and '/baseline_matrix_v3/results/round12' in str(output), 'project data-disk output only')
    output.mkdir(parents=True, exist_ok=False)
    locks = {str(args.config): sha(args.config), str(args.calibration_config): sha(args.calibration_config),
             calibration['cal_select_inputs']: calibration['cal_select_inputs_sha256'], **candidate_locks,
             **{variant[key]: value for key, value in config['expected_asset_sha256'].items()}}
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from model_registry import get_model_spec
    from evaluate import model_identity, tensor_sha
    from m1_protocol import prepare_m1_inputs, m1_prefix_forward, m1_forward_suffix, clone_m1_prefix_cache
    from cache_bridge import HybridVisualCache
    from attention_bridge import attention_context
    from ownership import cache_storage
    from r7_legacy_bridge_v1 import build_r7_owners
    from round12_static_bank_v1 import build_round12_owners
    require(torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 6), 'actual RTX3090 CUDA required')
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    identity = model_identity(config, get_model_spec('M1'))
    processor = AutoProcessor.from_pretrained(config['model_path'], local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(config['model_path'], dtype=torch.float16,
        device_map={'': 0}, local_files_only=True, attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    parameter_count = sum(p.numel() for p in model.parameters())
    require(parameter_count == 7944373760 and all(p.dtype == torch.float16 for p in model.parameters()), 'full original M1 FP16 required')
    sys.path.insert(0, config['legacy_a2'])
    kwargs = {key: config[key] for key in ('legacy_a2', 'legacy_experiments', 'legacy_scripts', 'kivi_source')}
    kwargs['expected_asset_sha256'] = config['expected_asset_sha256']
    report = {'schema': 'trajkv.round12.value_offset_diagnostic.v1',
              'status': 'RUNNING', 'actual_CUDA': True, 'GT_or_answers_loaded': False,
              'training_performed': False, 'candidate_or_f0_modified': False,
              'task_answer_events': 0, 'generated_answer_tokens': 0,
              'native_reference_retained_for_QA0': True,
              'production_memory_or_speed_proof': False, 'task_quality_gain_claimed': False,
              'model_parameter_count': parameter_count, 'model_identity': identity,
              'candidate_assets': candidate_info, 'frozen_variant_sha256': frozen.canonical_sha(variant),
              'locks': locks, 'expected_ids': [r['id'] for r in rows],
              'sources': [], 'model_forward_events': [], 'executed_legacy_source_sha256': {}}
    write(output / 'manifest.json', report)
    for row in rows:
        case = {'id': row['id'], 'dataset': row['dataset'], 'original_image_sha256': row['original_image_sha256'],
                'status': 'RUNNING', 'role': 'existing_cal_select_holdout4_not_independent_task_confirmation',
                'branches': {}}
        native = batch = image = prefix = fcache = native_output = None
        owners = meta = hybrid = prefix_clone = out = old_scales = scales = logits = metrics = f_logits = None
        torch.cuda.reset_peak_memory_stats()
        try:
            require(sha(row['image_path']) == row['original_image_sha256'], 'original source image SHA differs')
            with Image.open(row['image_path']) as raw:
                image = raw.convert('RGB')
            batch = prepare_m1_inputs(processor, image, row['question'] + config['question_suffix'], max_patches=8)
            case.update(full_prompt_hash=hashlib.sha256(batch.prompt.encode()).hexdigest(),
                        input_ids_sha256=tensor_sha(batch.ids), native_position_ids_sha256=tensor_sha(batch.positions),
                        compression_boundary=batch.boundary, visual_tokens=int(batch.visual_positions.numel()))
            prefix = measured_forward(report, row['id'], 'native_image_prefix',
                                       lambda: m1_prefix_forward(model, batch))
            native = prefix.past_key_values
            prefix = None
            fcache = clone_m1_prefix_cache(native, batch.boundary)
            native_output = measured_forward(report, row['id'], 'F_full_question_suffix_TF',
                                              lambda: m1_forward_suffix(model, batch, fcache))
            f_logits = native_output.logits[0, -1].detach().float().cpu().clone()
            case['branches']['F'] = {'first_logits': logits_summary(f_logits, f_logits),
                                     'raw_logit_evidence': save_logits(output, row['id'], 'F', f_logits),
                                     'native_reference': True}
            del native_output, fcache
            old_scales = None
            for name in ('L', 'V2R'):
                if name == 'L':
                    owners, meta = build_r7_owners(native, batch.visual_positions, variant, **kwargs)
                else:
                    owners, meta = build_round12_owners(native, batch.visual_positions, variant,
                                                       candidate=config['candidate'], **kwargs)
                for path, info in meta['legacy_sources'].items():
                    previous = report['executed_legacy_source_sha256'].get(path)
                    require(previous is None or previous == info['sha256'], 'legacy source changed between branches')
                    report['executed_legacy_source_sha256'][path] = info['sha256']
                metrics, scales = value_metrics(native, batch.visual_positions, owners,
                                                v2r=name == 'V2R', reference_scales=old_scales)
                if name == 'L':
                    old_scales = scales
                prefix_clone = clone_m1_prefix_cache(native, batch.boundary)
                hybrid = HybridVisualCache(prefix_clone, batch.visual_positions, owners, consume_native=True)
                frozen.share_original_visual_position_map(hybrid)
                owners = prefix_clone = None
                with attention_context(model, cache=hybrid):
                    out = measured_forward(report, row['id'], name + '_full_question_suffix_TF',
                                            lambda: m1_forward_suffix(model, batch, hybrid))
                logits = out.logits[0, -1].detach().float().cpu().clone()
                case['branches'][name] = {'first_logits': logits_summary(f_logits, logits),
                    'raw_logit_evidence': save_logits(output, row['id'], name, logits),
                    'value_layer_head_metrics': metrics, 'actual_packed_owner_bytes': meta['returned_owners_unique_tensor_capacity_bytes'],
                    'owner_after_suffix': cache_storage(hybrid, batch.visual_positions),
                    'native_reference_extra_memory_excluded_from_production_claim': True,
                    'selected_stages': meta['selected_stages']}
                del out, hybrid, logits, meta, metrics, scales
                gc.collect()
            case['status'] = 'PASS_QA0_NUMERICAL_MEASUREMENT'
            del old_scales, f_logits, image
        except Exception as exc:
            case.update(status='FAILED_QA0', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        finally:
            case['diagnostic_peak_allocated_bytes_including_native_reference'] = torch.cuda.max_memory_allocated()
            report['sources'].append(case)
            write(output / (hashlib.sha256(row['id'].encode()).hexdigest()[:20] + '.json'), case)
            print(json.dumps({'id': row['id'], 'status': case['status'], 'task_QA': 0}), flush=True)
            native = batch = image = prefix = fcache = native_output = None
            owners = meta = hybrid = prefix_clone = out = old_scales = scales = logits = metrics = f_logits = None
            gc.collect()
            torch.cuda.empty_cache()
    require(all(sha(path) == digest for path, digest in locks.items()), 'locked input/asset changed')
    require(all(sha(path) == digest for path, digest in report['executed_legacy_source_sha256'].items()),
            'executed legacy source changed')
    require(frozen.canonical_sha(variant) == report['frozen_variant_sha256'], 'variant mutated')
    torch.cuda.synchronize()
    report.update(status='PASS_ACTUAL_CUDA_VALUE_OFFSET_QA0' if all(x['status'].startswith('PASS') for x in report['sources']) else 'COMPLETE_WITH_QA0_FAILURES',
        native_prefix_model_forwards=sum(x['phase'] == 'native_image_prefix' for x in report['model_forward_events']),
        suffix_TF_model_forwards=sum(x['phase'] != 'native_image_prefix' for x in report['model_forward_events']),
        completed_model_forwards=sum(x['completed'] for x in report['model_forward_events']),
        GPU=torch.cuda.get_device_name(), torch_version=torch.__version__, elapsed_seconds=time.perf_counter() - started,
        source_sha256={name: sha(ROOT / 'code' / name) for name in
            ('round12_value_offset_diagnostic_v1.py', 'round12_static_bank_v1.py', 'round12_static_codec_v1.py',
             'round12_static_attention_v1.py', 'm1_protocol.py', 'cache_bridge.py', 'attention_bridge.py')})
    write(output / 'VALUE_OFFSET_DIAGNOSTIC_v1.json', report)
    print(json.dumps({'status': report['status'], 'output': str(output), 'new_free_QA': 0,
                      'native_prefix_forwards': report['native_prefix_model_forwards'],
                      'suffix_TF_forwards': report['suffix_TF_model_forwards']}), flush=True)
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
