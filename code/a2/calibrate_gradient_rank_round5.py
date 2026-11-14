"""RateQuant-inspired gradient ranking in TrajKV's fixed sign-residual format.

This is a strategy port, not the official RateQuant reproduction.  It uses
Full-generated DocVQA answer tokens and a pooled answer-CE gradient, whereas
the cited paper uses text causal-LM calibration and per-target-token gradient
energy. No gold answer, model update, or online gradient is used here.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/experiments'))
from bank import install_legacy
from quantizers import binary_centroids, pack1
from affine_rate_bank import fold
from fast_wht import torch_wht
from fused_progressive import decode_last_generic
import session_boundary


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_rows(rows, images, max_cases, first_question_per_image=False):
    selected = []
    for row in rows:
        if row['image_id'] not in selected:
            selected.append(row['image_id'])
    allowed = set(selected[:images])
    chosen = [row for row in rows if row['image_id'] in allowed]
    if first_question_per_image:
        seen, first = set(), []
        for row in chosen:
            if row['image_id'] not in seen:
                seen.add(row['image_id'])
                first.append(row)
        chosen = first
    return chosen[:max_cases]


def make_prefix_cache(entries, config, requires_grad):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache(config=config)
    keys, values = [], []
    for layer, (source_k, source_v) in enumerate(entries):
        key = source_k.detach().clone().requires_grad_(requires_grad)
        value = source_v.detach().clone().requires_grad_(requires_grad)
        if requires_grad and (not key.is_leaf or not value.is_leaf):
            raise RuntimeError('Native prefix K/V must remain autograd leaves')
        keys.append(key)
        values.append(value)
        cache.update(key, value, layer)
    return cache, keys, values


def unpack_sign(words, shape):
    shifts = torch.arange(32, device=words.device)
    bits = ((words.long()[..., None] >> shifts) & 1).reshape(shape)
    return 2 * bits.float() - 1


def reconstruct_errors(source_k, source_v, book, base, layer, value_group):
    """Actual frozen codes/F16 metadata; same residual arithmetic as sparse bank.

    K is encoded/refined in centered WHT coordinates and then transformed back
    to the original native-cache coordinates for pairing with native gradients.
    This is a materialized dictionary reference: it does not differentiate the
    deployed QR-half rounding or the quantizer/code-assignment operation.
    """
    k, v = source_k.float(), source_v.float()
    mean = k.mean(-2, keepdim=True).half()
    rotated = torch_wht(k - mean.float())
    norm = rotated.norm(dim=-1, keepdim=True)
    initial = (norm / (128 ** .5)).clamp_min(1e-20)
    parts, bf, indices = (rotated / initial).reshape(-1, 4), book.float(), []
    for start in range(0, len(parts), 65536):
        part = parts[start:start + 65536]
        distance = part.square().sum(-1, keepdim=True) + bf.square().sum(-1)[None] - 2 * (part @ bf.T)
        indices.append(distance.argmin(-1).byte())
    codes = torch.cat(indices).reshape(*rotated.shape[:-1], 32)
    template = bf[codes.long()].reshape(rotated.shape)
    gain = (norm / template.norm(dim=-1, keepdim=True).clamp_min(1e-20)).half()
    if base.get('key_head_factors') is not None:
        kg = torch.tensor(base['key_head_factors'][layer], device=k.device, dtype=torch.float32)[None, :, None, None]
        gain = (gain.float() * kg).half()
        if base['key_gain_mode'] == 'whole':
            mean = (mean.float() * kg).half()
    base_rotated = (template * gain.float()).half()
    k_error = rotated - base_rotated.float()
    k_alpha = k_error.abs().mean(-1).half()
    k_sign = unpack_sign(pack1(k_error >= 0), k_error.shape)
    refined_rotated = (base_rotated.float() + k_sign * k_alpha.float()[..., None]).half()
    decoded_k = (torch_wht(base_rotated.float()) + mean.float()).half()
    refined_k = (torch_wht(refined_rotated.float()) + mean.float()).half()

    record = binary_centroids(source_v.transpose(2, 3).contiguous(), group_size=value_group)
    if base.get('value_head_factors') is not None:
        vg = torch.tensor(base['value_head_factors'][layer], device=v.device, dtype=torch.float32)[None, :, None, None]
        vo = torch.tensor(base['value_center_offsets'][layer], device=v.device, dtype=torch.float32)[None, :, None, None]
        record = fold(record, 1, vg, vo)
    decoded_v = decode_last_generic(*record, value_group, 1).transpose(2, 3).contiguous()
    v_error = v - decoded_v.float()
    v_alpha = v_error.abs().mean(-1).half()
    v_sign = unpack_sign(pack1(v_error >= 0), v_error.shape)
    refined_v = (decoded_v.float() + v_sign * v_alpha.float()[..., None]).half()
    return {'k': (decoded_k.float() - k, refined_k.float() - k),
            'v': (decoded_v.float() - v, refined_v.float() - v)}


def unit_statistics(gradient, error_base, error_refined):
    """Per-head statistics, retaining K/V's common CE scale without rescaling."""
    gradient = gradient.float()
    g2 = gradient.square()
    base2, refined2 = error_base.square(), error_refined.square()
    reduction = base2 - refined2
    nv, dimension = gradient.shape[-2:]
    reduce = lambda value: value.sum(dim=(0, 2, 3)).double().cpu()
    sensitivity = reduce(g2) / nv  # mean over visual token positions, sum D.
    delta_mse = reduce(reduction) / (nv * dimension)
    return {'sensitivity_w': sensitivity,
            'mean_base_mse': reduce(base2) / (nv * dimension),
            'mean_refined_mse': reduce(refined2) / (nv * dimension),
            'mean_mse_gain': delta_mse,
            'score_case_product': sensitivity * delta_mse,
            'fine_gradient_weighted_error_gain_sum': reduce(g2 * reduction),
            'gradient_square_sum': reduce(g2),
            'base_error_sse': reduce(base2), 'refined_error_sse': reduce(refined2)}


def summarize(total, cases):
    output, scalar_ranks, fine_ranks, mse_ranks, joint = {}, {}, {}, {}, []
    for axis in ('k', 'v'):
        means = {key: value / cases for key, value in total[axis].items()}
        # Product of calibration means matches the separable w_head * DeltaD
        # strategy. The mean of per-case products is retained as a distinct proxy.
        scalar = means['sensitivity_w'] * means['mean_mse_gain']
        entries = []
        for layer in range(scalar.shape[0]):
            for head in range(scalar.shape[1]):
                entry = {'unit': [layer, head], 'axis': axis,
                    'score_head_scalar': float(scalar[layer, head]),
                    'sensitivity_w': float(means['sensitivity_w'][layer, head]),
                    'mean_mse_gain': float(means['mean_mse_gain'][layer, head]),
                    'mean_case_product': float(means['score_case_product'][layer, head]),
                    'fine_gradient_weighted_error_gain_sum': float(means['fine_gradient_weighted_error_gain_sum'][layer, head])}
                entries.append(entry)
                joint.append(entry)
        scalar_ranks[axis] = sorted(entries, key=lambda row: (-row['score_head_scalar'], row['unit']))
        fine_ranks[axis] = sorted(entries, key=lambda row: (-row['fine_gradient_weighted_error_gain_sum'], row['unit']))
        mse_ranks[axis] = sorted(entries, key=lambda row: (-row['mean_mse_gain'], row['unit']))
        output[axis] = {key: value.tolist() for key, value in means.items()}
        output[axis]['score_head_scalar'] = scalar.tolist()
    return {'per_axis': output, 'ranks': scalar_ranks, 'fine_grained_secondary_ranks': fine_ranks,
            'mse_only_ranks': mse_ranks,
            'joint_KV_ranks': sorted(joint, key=lambda row: (-row['score_head_scalar'], row['axis'], row['unit']))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    a = c['arguments']
    install_legacy(a['legacy_scripts'])
    from p180_multiturn_core import attention_bridge
    import run_p175_modern_replication as oldbase
    from transformers import AutoProcessor, AutoModelForImageTextToText

    torch.manual_seed(1212)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    target_limit = int(c.get('target_tokens', 8))
    if not 1 <= target_limit <= 8:
        raise ValueError('This strategy port uses 1..8 self-generated answer targets')
    loss_scale = float(c.get('gradient_loss_scale', 128.))
    if not loss_scale > 0:
        raise ValueError('Positive gradient loss scale required')
    rows = selected_rows(json.loads(Path(a['samples']).read_text(encoding='utf-8')),
                         int(c.get('calibration_image_count', 16)), int(c.get('max_cases', 39)),
                         bool(c.get('first_question_per_image', False)))
    if not rows:
        raise ValueError('No calibration cases selected')
    base_path, book_path = Path(c['fixed_metadata_factor_path']), Path(c['encoder_book_path'])
    base = json.loads(base_path.read_text())
    if base['encoder_book_sha256'] != sha(book_path):
        raise ValueError('Frozen metadata/book hash mismatch')
    if base['value_group_size'] != c.get('value_group_size', 'all'):
        raise ValueError('Frozen V window mismatch')
    if base['key_gain_mode'] not in ('none', 'centered', 'whole'):
        raise ValueError('Invalid frozen K factor mode')
    manifest = {'config': c, 'config_sha256': sha(args.config), 'samples_sha256': sha(a['samples']),
        'encoder_book_sha256': sha(book_path), 'fixed_metadata_sha256': sha(base_path),
        'selected_case_ids': [row['id'] for row in rows], 'selected_image_ids': list(dict.fromkeys(row['image_id'] for row in rows)),
        'first_question_per_image': bool(c.get('first_question_per_image', False)),
        'model_weights_frozen': True, 'gold_answers_loaded': False, 'labels_used': False,
        'self_generated_teacher_targets_used': True,
        'task_loss': 'mean CE over at most 8 Full argmax self-generated answer tokens, including generated stop token',
        'gradient_scope': 'one pooled answer-CE backward per independent question; native SDPA and detached prefix KV leaves',
        'reference_method': 'RateQuant gradient-sensitivity x distortion allocation strategy',
        'reference_url': 'https://arxiv.org/html/2605.06675v2#S3.SS2',
        'official_reproduction': False,
        'differences_from_official': 'Doc image/questions and Full self-targets; pooled answer CE gradient squared, not WikiText2 next-token/per-target gradient energies; fixed TrajKV sign-residual format',
        'sources': {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name('quantizers.py'), Path(__file__).with_name('fast_wht.py'), Path(__file__).with_name('affine_rate_bank.py'), Path(__file__).with_name('sparse_v_residual_bank.py'))}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    processor = AutoProcessor.from_pretrained(a['model'], local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(a['model'], local_files_only=True,
        dtype=torch.float16, device_map='cuda:0', attn_implementation='sdpa').eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    text_config = model.config.get_text_config()
    nl, nh = text_config.num_hidden_layers, text_config.num_key_value_heads
    if text_config.hidden_size // text_config.num_attention_heads != 128:
        raise ValueError('Current format requires head dimension 128')
    if any(hasattr(layer.self_attn, '_p185_read') for layer in model.model.language_model.layers):
        raise RuntimeError('A custom packed attention hook must not be active for native KV gradients')
    book = torch.tensor(json.loads(book_path.read_text())['centers'], device=model.device, dtype=torch.float16)
    if tuple(book.shape) != (256, 4):
        raise ValueError('Frozen book shape mismatch')
    total = {'k': {}, 'v': {}}
    controls, case_logs = [], []
    nonzero_gradient_elements = 0
    stops = oldbase.stop_ids(model, processor)
    with (args.output / 'cases.jsonl').open('w', encoding='utf-8') as case_file:
        for index, row in enumerate(rows):
            started = time.perf_counter()
            with torch.no_grad():
                features = {key: value.to(model.device, dtype=model.dtype if value.is_floating_point() else value.dtype)
                    for key, value in session_boundary.features(processor, row, a.get('max_patches', 8)).items()}
                visual = (features['input_ids'][0] == model.config.image_token_id).nonzero().flatten()
                if not visual.numel():
                    raise ValueError('No visual tokens in calibration case')
                boundary = int(visual[-1]) + 1
                prompt_length = features['input_ids'].shape[-1]
                suffix = features['input_ids'][:, boundary:].clone()
                if not suffix.numel():
                    raise ValueError('Visual boundary must precede the question suffix')
                with attention_bridge(model, native=True):
                    teacher = model(**features, use_cache=True, return_dict=True, logits_to_keep=1)
                teacher_cache = teacher.past_key_values
                ids, original_logits = [], []
                for step in range(target_limit):
                    original_logits.append(teacher.logits[:, -1].detach().float().cpu())
                    token = int(teacher.logits[:, -1].argmax(-1))
                    ids.append(token)
                    if token in stops or step + 1 == target_limit:
                        break
                    with attention_bridge(model, native=True):
                        teacher = model(input_ids=torch.tensor([[token]], device=model.device),
                            position_ids=torch.tensor([[prompt_length + step]], device=model.device),
                            past_key_values=teacher_cache, use_cache=True, return_dict=True, logits_to_keep=1)
                prefix = [(layer.keys[:, :, :boundary].detach().clone(), layer.values[:, :, :boundary].detach().clone())
                          for layer in teacher_cache.layers]
                del teacher, teacher_cache, features
                if len(ids) > 1:
                    suffix = torch.cat((suffix, torch.tensor([ids[:-1]], device=model.device)), -1)
                target = torch.tensor([ids], device=model.device)
                qpos = torch.arange(boundary, boundary + suffix.shape[-1], device=model.device)[None]
                control_logits = None
                if index < int(c.get('native_cache_control_cases', 1)):
                    native_cache, _, _ = make_prefix_cache(prefix, text_config, False)
                    if int(native_cache.get_seq_length()) != boundary:
                        raise RuntimeError('Native control cache has wrong prefix length')
                    with attention_bridge(model, native=True):
                        control = model(input_ids=suffix, position_ids=qpos, past_key_values=native_cache,
                            use_cache=True, return_dict=True, logits_to_keep=len(ids))
                    control_logits = control.logits.detach().float().cpu()
                    del native_cache, control
            native_cache, leaf_k, leaf_v = make_prefix_cache(prefix, text_config, True)
            if int(native_cache.get_seq_length()) != boundary:
                raise RuntimeError('Gradient cache has wrong prefix length')
            with attention_bridge(model, native=True):
                output = model(input_ids=suffix, position_ids=qpos, past_key_values=native_cache,
                    use_cache=True, return_dict=True, logits_to_keep=len(ids))
            if output.logits.shape[:2] != target.shape:
                raise RuntimeError('Self-target/logit alignment mismatch')
            loss = F.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]), target.reshape(-1))
            if not loss.requires_grad or not bool(torch.isfinite(loss)):
                raise RuntimeError('Native answer CE is detached or nonfinite')
            if control_logits is not None:
                observed = output.logits.detach().float().cpu()
                difference = observed - control_logits
                max_abs = float(difference.abs().max())
                relative = float(difference.norm() / control_logits.norm().clamp_min(1e-20))
                original = torch.stack(original_logits, 1)
                controls.append({'case_id': row['id'], 'max_abs_gradient_vs_no_grad_logits': max_abs,
                    'relative_l2_gradient_vs_no_grad_logits': relative,
                    'gradient_vs_no_grad_argmax_same': bool(torch.equal(observed.argmax(-1), control_logits.argmax(-1))),
                    'max_abs_batched_replay_vs_original_generation': float((control_logits - original).abs().max()),
                    'prefix_boundary': boundary, 'suffix_length': suffix.shape[-1], 'teacher_targets': len(ids)})
                if max_abs > float(c.get('native_cache_control_max_abs', .5)) or relative > float(c.get('native_cache_control_relative_l2', .01)):
                    raise RuntimeError('Leaf-gradient cache changed native same-input replay numerics')
            (loss * loss_scale).backward()
            statistics = {'k': {}, 'v': {}}
            with torch.no_grad():
                for layer in range(nl):
                    if leaf_k[layer].grad is None or leaf_v[layer].grad is None:
                        raise RuntimeError('A native prefix K/V leaf has no gradient')
                    gradients = {'k': leaf_k[layer].grad.index_select(2, visual).float() / loss_scale,
                                 'v': leaf_v[layer].grad.index_select(2, visual).float() / loss_scale}
                    if any(not bool(torch.isfinite(value).all()) for value in gradients.values()):
                        raise RuntimeError('Nonfinite native KV gradient')
                    source_k, source_v = [value.index_select(2, visual) for value in prefix[layer]]
                    value_group = visual.numel() if c.get('value_group_size', 'all') == 'all' else 128
                    errors = reconstruct_errors(source_k, source_v, book, base, layer, value_group)
                    for axis in ('k', 'v'):
                        nonzero_gradient_elements += int(torch.count_nonzero(gradients[axis]))
                        unit = unit_statistics(gradients[axis], *errors[axis])
                        for name, value in unit.items():
                            statistics[axis].setdefault(name, torch.zeros(nl, nh, dtype=torch.float64))[layer] = value
                for axis in ('k', 'v'):
                    for name, value in statistics[axis].items():
                        total[axis].setdefault(name, torch.zeros_like(value)).add_(value)
            torch.cuda.synchronize()
            log = {'case': index + 1, 'id': row['id'], 'image_id': row['image_id'],
                'teacher_token_ids': ids, 'teacher_target_tokens': len(ids), 'visual_tokens': visual.numel(),
                'prefix_boundary': boundary, 'answer_ce': float(loss.detach()),
                'seconds': time.perf_counter() - started,
                'statistics': {axis: {name: value.tolist() for name, value in stats.items()} for axis, stats in statistics.items()}}
            case_file.write(json.dumps(log) + '\n')
            case_file.flush()
            case_logs.append({key: value for key, value in log.items() if key != 'statistics'})
            print(json.dumps({'phase': 'gradient_ranking', 'case': index + 1, 'cases': len(rows), 'id': row['id'],
                              'answer_ce': log['answer_ce'], 'seconds': round(log['seconds'], 2)}), flush=True)
            del output, native_cache, leaf_k, leaf_v, prefix, loss, gradients, errors, source_k, source_v, suffix, target
    if nonzero_gradient_elements == 0:
        raise RuntimeError('All native visual K/V gradients are zero')
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError('Model weights were not kept frozen')
    result = summarize(total, len(rows))
    result.update(schema='trajkv_gradient_rank_strategy_port_v1', status='COMPLETE',
        cases=len(rows), images=len({row['image_id'] for row in rows}), labels_used=False,
        gold_answers_loaded=False, self_generated_teacher_targets_used=True,
        model_weights_frozen=True, task_accuracy_eligible=False, official_reproduction=False,
        primary_score='mean_case[mean_visual_token ||grad pooled_answer_CE||^2] * mean_case[MSE_base - MSE_sign_refined]',
        secondary_score='mean_case sum_visual_elements grad^2 * (error_base^2 - error_sign_refined^2)',
        K_V_gradient_sensitivities_separately_normalized=False,
        calibration_ids=[row['id'] for row in rows], cases_log=case_logs,
        native_cache_controls=controls, nonzero_gradient_elements=nonzero_gradient_elements,
        native_gradient_loss_scale=loss_scale,
        encoder_book_sha256=sha(book_path), fixed_metadata_sha256=sha(base_path), config_sha256=sha(args.config),
        residual_format='same sign1/scalar + stored FP16 MAD amplitude per 128D token as SparseVResidualBank; original codebook and frozen metadata',
        K_coordinate_scope='sign residual encoded/refined in centered WHT space, reconstructed back to original native-cache K coordinates; frozen mean preserved',
        approximations=['pooled self-answer CE squared gradient differs from official per-target text causal-LM gradient-energy averaging',
                        'diagonal gradient-squared distortion proxy omits coordinate and K/V cross terms',
                        'native-cache derivative uses materialized dictionary errors; deployed QR-half and quantizer rounding are not differentiated',
                        'ranking must be compared in the same actual packed residual byte budget; no ANLS prediction'])
    (args.output / 'ranking_scores.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (args.output / 'complete.json').write_text(json.dumps({'status': 'COMPLETE', 'cases': len(rows),
        'images': result['images'], 'official_reproduction': False}), encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'cases': len(rows), 'images': result['images'],
                      'top_K_units': result['ranks']['k'][:3], 'top_V_units': result['ranks']['v'][:3]}), flush=True)


if __name__ == '__main__':
    main()
