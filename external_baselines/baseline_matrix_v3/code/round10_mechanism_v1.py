"""CPU-only, fixed-query mechanism replay on eight predeclared fit sources.

Private tensors and WO remain on the data disk. No model is loaded, no GPU is
used and no encoder sees query weights. This is not a task-quality experiment.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
CALIBRATOR_SHA = 'abbc45f07e5899ff6018e0a22c5bba94c8764b03cd518e764de1949d9df87a8a'
PINS = {'code/round9_calibration_v1.py': '757cc51740ad2f92d8ebd9ad9adadf2719131ae634b16b24052363f8e45ba84f',
        'code/round9_value_bank_v1.py': '14d242a6643a738457c9b8593d0f0eb6e383c5e979b2e196fcca28e7d37a5f2e',
        'code/round10_value_bank_v1.py': '5c9880d9e7facdc1623473dd74759aee9064bd5ff313f1581390fe0ccd3aae1c'}
DOMAINS = ('textvqa', 'infovqa', 'docvqa', 'chartqa')


def require(x, message):
    if not x: raise ValueError('ROUND10_MECHANISM: ' + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def bit_changes(left, right):
    import torch
    value = (left.long() ^ right.long()) & 0xffffffff
    return int(((value[..., None] >> torch.arange(32)) & 1).sum())


def coefficients(a, width):
    """Exact summation-by-parts coefficients, INCLUDING every terminal term."""
    import torch
    a = a.double(); n = a.shape[-1]
    require(n % width == 0, 'complete windows')
    b = torch.empty_like(a)
    for start in range(0, n, width):
        end = start + width
        b[..., start:end-1] = a[..., start+1:end] - a[..., start:end-1]
        b[..., end-1] = -a[..., end-1]
    denominator = a.square().sum(-1)
    numerator = b.square().sum(-1)
    ratio = numerator / denominator.clamp_min(1e-300)
    return b, [float(v) if float(d) > 0 else None for v, d in zip(ratio.flatten(), denominator.flatten())]


def trace_feedback(state, source, window):
    """Reference encoder using ACTUAL sequential-Half A/B and FP32 carry."""
    import torch
    import round9_value_bank_v1 as old
    require(source.device.type == 'cpu' and source.dtype == torch.float16 and source.shape[:2] == (1, 4), 'CPU native V')
    n = source.shape[-2]; width = n if window == 'global' else window
    require(width in (1, 32, 64, 128, 256, n) and n % 256 == 0 and n % width == 0, 'native tile/window layout')
    a = old.decode_value_fields(state, n, base_bits_override=0)
    b = old.decode_value_fields(state, n, base_bits_override=1)
    decision = torch.empty_like(source, dtype=torch.bool)
    carry_all = torch.empty_like(source, dtype=torch.float32)
    remainder = torch.empty_like(source, dtype=torch.float64)
    outside_u = 0
    for start in range(0, n, width):
        carry = torch.zeros_like(source[:, :, 0], dtype=torch.float32)
        for token in range(start, start + width):
            x, lo, hi = source[:, :, token].float(), a[:, :, token].float(), b[:, :, token].float()
            previous = carry
            u = x + previous
            bit = (u - hi).abs() < (u - lo).abs()
            y = torch.where(bit, hi, lo)
            carry = u - y
            require(bool(torch.isfinite(carry).all()), 'nonfinite CPU carry')
            decision[:, :, token] = bit; carry_all[:, :, token] = carry
            remainder[:, :, token] = (y.double() - x.double()) - (previous.double() - carry.double())
            outside_u += int(((u < torch.minimum(lo, hi)) | (u > torch.maximum(lo, hi))).sum())
    result = copy(state)
    result.v_code = old._pack(decision.transpose(2, 3).contiguous())
    decoded = old.decode_value_fields(result, n)
    require(torch.equal(decoded, torch.where(decision, b, a)), 'packed code must decode the exact selected Half level')
    gap = (b.float() - a.float()).abs(); nonzero = gap > 0
    normalized = carry_all[nonzero] / gap[nonzero]
    inside = torch.arange(1, n) % width != 0
    carry_difference = carry_all[:, :, 1:] - carry_all[:, :, :-1]
    input_difference = source[:, :, 1:].float() - source[:, :, :-1].float()
    terminal = carry_all[:, :, width-1::width]
    stats = {'actual_width': width, 'global_has_no_tile_reset': window == 'global',
        'carry_mean': float(carry_all.mean()), 'carry_RMS': float(carry_all.double().square().mean().sqrt()),
        'carry_maxabs': float(carry_all.abs().max()),
        'carry_within_window_adjacent_difference_RMS': float(carry_difference[:, :, inside].double().square().mean().sqrt()) if bool(inside.any()) else None,
        'native_input_within_window_adjacent_difference_RMS': float(input_difference[:, :, inside].double().square().mean().sqrt()) if bool(inside.any()) else None,
        'carry_terminal_RMS': float(terminal.double().square().mean().sqrt()),
        'carry_terminal_maxabs': float(terminal.abs().max()),
        'actual_level_gap_RMS': float(gap.double().square().mean().sqrt()), 'zero_gap_fraction': float((gap == 0).float().mean()),
        'carry_over_actual_nonzero_gap_RMS': float(normalized.double().square().mean().sqrt()) if normalized.numel() else None,
        'native_input_outside_actual_levels_fraction': float(((source.float() < torch.minimum(a, b).float()) | (source.float() > torch.maximum(a, b).float())).float().mean()),
        'carry_adjusted_input_outside_levels_fraction': outside_u / source.numel(),
        'FP32_step_remainder_RMS': float(remainder.square().mean().sqrt()),
        'FP32_step_remainder_maxabs': float(remainder.abs().max()),
        'changed_base_bits_vs_original': bit_changes(state.v_code, result.v_code),
        'changed_code_words_vs_original': int((state.v_code != result.v_code).sum()),
        'sole_rho_or_stability_claim': False}
    return result, decoded, carry_all, remainder, stats


def weighted_geometry(weights, source, decoded, carry, remainder, width):
    """Fixed compressed-K visual weights; never passed to the encoder."""
    import torch
    a = weights[0, :, 0].double(); b, h = coefficients(a, width)
    errors = decoded.double() - source.double()
    left, right, rounding = [], [], []
    n = a.shape[-1]
    for head in range(4):
        group = slice(head * 7, (head + 1) * 7)
        left.append(a[group] @ errors[0, head])
        right.append(b[group] @ carry[0, head].double())
        rounding.append(a[group] @ remainder[0, head])
    lhs, rhs, rem = torch.cat(left), torch.cat(right), torch.cat(rounding)
    closure = lhs - rhs - rem
    require(float(closure.abs().max()) <= 1e-8 * (1 + float(lhs.abs().max())), 'weighted carry/terminal/rounding closure')
    return {'H_geom_per_query_head': h, 'weighted_error_MSE': float(lhs.square().mean()),
        'weighted_FP32_remainder_RMS': float(rem.square().mean().sqrt()),
        'weighted_identity_residual_maxabs': float(closure.abs().max()),
        'boundary_included': True, 'H_is_geometric_not_loss_bound': True}


def error_moments(source, value, weights):
    import torch
    e = value.double() - source.double()
    bias = e.mean(2)
    total, diagonal = [], []
    for h in range(4):
        a = weights[0, h*7:(h+1)*7, 0].double()
        total.extend((a @ e[0, h]).square().mean(-1).tolist())
        diagonal.extend((a.square() @ e[0, h].square().mean(-1)).tolist())
    return {'native_V_MSE': float(e.square().mean()),
        'per_channel_token_mean_error_energy': float(bias.square().mean()),
        'token_centered_error_MSE': float((e - bias[:, :, None]).square().mean()),
        'compressed_K_weighted_value_error_MSE': sum(total) / len(total),
        'deterministic_diagonal_token_term': sum(diagonal) / len(diagonal),
        'deterministic_cross_token_term': (sum(total) - sum(diagonal)) / len(total),
        'cross_term_is_not_universal_rho_or_independence_estimate': True}


def projected_output(sample, probabilities, value, wo):
    import torch
    import round9_calibration_v1 as cal
    n = value.shape[-2]
    visual = value.repeat_interleave(7, 1).float()
    text = sample['exact_values'].repeat_interleave(7, 1).float()
    heads = (probabilities[..., :n] @ visual + probabilities[..., n:] @ text).half()
    # Explicit CPU diagnostic arithmetic, not a claim of CUDA Half GEMM equality.
    return cal.project(heads, wo, ideal=True).half().double()


def analyze(config, output):
    import torch
    import round9_calibration_v1 as cal
    import round9_value_bank_v1 as old
    require(config['schema'] == 'trajkv.round10.mechanism_config.v1' and config['purpose'] == 'mechanism_only', 'mechanism-only scope')
    require(not set(config).intersection({'answers', 'references', 'labels', 'refs'}), 'no GT')
    for name, digest in PINS.items(): require(sha(ROOT / name) == digest, 'frozen CPU helper source: ' + name)
    root = Path(config['collection']); closed = read(root / 'complete.json')
    require(sha(root / 'complete.json') == config['collection_complete_sha256'] and
        closed['status'] == 'PASS_LS48_AND_ACTUAL_MODEL_SELECT16' and closed['actual_GPU'] is True and
        closed['source_hashes']['code/round10_calibrate_v1.py'] == CALIBRATOR_SHA, 'closed actual calibration only')
    require(sha(config['targets_file']) == config['targets_sha256'], 'prelocked eight sources')
    targets = read(config['targets_file'])
    require(targets['schema'] == 'trajkv.round10.mechanism_targets.v1' and
            targets['selection_rule'] == 'first_two_per_dataset_in_frozen_calFit48_input_order_without_values_or_scores' and
            Counter(x['dataset'] for x in targets['sources']) == Counter({d: 2 for d in DOMAINS}), 'fixed eight-source design')
    fit = read(root / 'fit_collection_complete.json')
    require(fit['status'] == 'PASS_REAL_MODEL_FIT48_ALL28' and len(fit['sources']) == 48, 'closed fit48 tensors')
    require(targets['fit_inputs_sha256'] == read(root / 'manifest.json')['config']['cal_fit_sha256'], 'same fit input parent')
    selected = read(root / 'window_selected.json')
    require(sha(root / 'window_selected.json') == closed['window_sha256'] and
            selected['window'] == closed['selected_window'], 'frozen calSelect window')
    window = selected['window']; require(window in (32, 64, 128, 256, 'global'), 'one selected nontrivial window')
    require(sha(fit['private_WO_file']) == fit['private_WO_sha256'] == closed['private_WO_sha256'], 'private WO SHA')
    torch.set_num_threads(int(config.get('CPU_threads', 4)))
    wos = torch.load(fit['private_WO_file'], map_location='cpu', weights_only=True)
    require(set(wos) == set(range(28)) and all(x.device.type == 'cpu' for x in wos.values()), 'CPU WO all28')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    rows = []
    with (output / 'layer_records.jsonl').open('x', encoding='utf-8') as stream:
        for target in targets['sources']:
            found = [x for x in fit['sources'] if (x['dataset'], x['id'], x['source_id']) ==
                     (target['dataset'], target['id'], target['source_id'])]
            require(len(found) == 1 and sha(found[0]['file']) == found[0]['sha256'], 'exact private source shard')
            case = torch.load(found[0]['file'], map_location='cpu', weights_only=True)
            require(case['schema'] == 'trajkv.round10.ls_fit_case.v1' and
                    case['source']['original_image_sha256'] == target['original_image_sha256'], 'case/source identity')
            events = [x for x in case['events'] if x['phase'] == 'suffix_last' and x['step'] == 0]
            require(len(events) == 28 and {e['layer_idx'] for e in events} == set(range(28)), 'all28 suffix-last only')
            for event in sorted(events, key=lambda x: x['layer_idx']):
                i = event['layer_idx']; sample = {**case['layers'][i], **event, 'book': case['book'],
                    'visual_positions': case['visual_positions'], 'selected_stages': case['selected_stages'],
                    'original_factors': case['original_factors']}
                cal.validate_sample(sample)
                require(all(x.device.type == 'cpu' for x in sample.values() if isinstance(x, torch.Tensor)), 'CPU data only')
                state = SimpleNamespace(**sample['r7_fields']); code_before = tensor_sha(state.v_code)
                v0 = old.decode_value_fields(state, sample['visual_v'].shape[-2])
                p = cal.probabilities(sample, sample['r7_fields'], sample['book'])
                pf = cal.probabilities(sample, native=True)
                n = sample['visual_v'].shape[-2]; av = p[..., :n]
                ff = projected_output(sample, pf, sample['visual_v'], wos[i])
                cc = projected_output(sample, p, v0, wos[i])
                cf = projected_output(sample, p, sample['visual_v'], wos[i])
                fc = projected_output(sample, pf, v0, wos[i])
                result = {'dataset': target['dataset'], 'id': target['id'], 'source_id': target['source_id'],
                    'layer_idx': i, 'phase': 'suffix_last', 'source_shard_sha256': found[0]['sha256'],
                    'native_V_energy': float(sample['visual_v'].double().square().mean()),
                    'local_FF_projected_energy': float(ff.square().mean()), 'local_CC_projected_energy': float(cc.square().mean()),
                    'four_arm_MSE_to_FF': {k: float((v - ff).square().mean()) for k, v in [('FF', ff), ('CF', cf), ('FC', fc), ('CC', cc)]},
                    'original': error_moments(sample['visual_v'], v0, av), 'window': window}
                output0 = None; fb0_code = None
                for label, rule in (('FB0', 1), ('FBW', window)):
                    new, value, carry, remainder, stats = trace_feedback(state, sample['visual_v'], rule)
                    projected = projected_output(sample, p, value, wos[i]); delta = projected - cc; original_error = cc - ff
                    if label == 'FB0': fb0_code = new.v_code; output0 = projected
                    geometry = weighted_geometry(av, sample['visual_v'], value, carry, remainder, stats['actual_width'])
                    moments = error_moments(sample['visual_v'], value, av)
                    moments.update(stats); moments.update(geometry)
                    moments.update(CC_projected_MSE_to_FF=float((projected - ff).square().mean()),
                        projected_delta_energy=float(delta.square().mean()),
                        projected_original_error_cross_delta=2*float((original_error*delta).mean()),
                        changed_base_bits_vs_FB0=bit_changes(fb0_code, new.v_code))
                    result[label] = moments
                    new = value = carry = remainder = projected = delta = None
                require(tensor_sha(state.v_code) == code_before, 'original packed source mutated')
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n');stream.flush()
                rows.append(result)
                print(json.dumps({'phase': 'CPU_MECHANISM_LAYER', 'id': target['id'], 'layer': i}), flush=True)
            case = None; gc.collect()
    summaries = []
    for target in targets['sources']:
        group = [r for r in rows if (r['dataset'], r['id']) == (target['dataset'], target['id'])]
        require(len(group) == 28, 'complete source layers')
        item = {k: target[k] for k in ('dataset', 'id', 'source_id')}
        for label in ('original', 'FB0', 'FBW'):
            fields = ['native_V_MSE', 'per_channel_token_mean_error_energy', 'token_centered_error_MSE',
                      'compressed_K_weighted_value_error_MSE', 'deterministic_diagonal_token_term', 'deterministic_cross_token_term']
            item[label] = {k: sum(r[label][k] for r in group)/28 for k in fields}
            item[label]['CC_projected_MSE_to_FF'] = sum(r['four_arm_MSE_to_FF']['CC'] if label == 'original' else r[label]['CC_projected_MSE_to_FF'] for r in group)/28
        summaries.append(item)
    result = {'status': 'PASS_CPU_FIXED_QUERY_EIGHT_SOURCE_MECHANISM', 'sources': summaries, 'source_count': 8,
        'independent_source_counts': {d: 2 for d in DOMAINS}, 'layer_records': len(rows), 'query_phase': 'suffix_last_only',
        'selected_window': window, 'collection_complete_sha256': config['collection_complete_sha256'],
        'targets_sha256': config['targets_sha256'], 'private_WO_sha256': fit['private_WO_sha256'],
        'module_sha256': sha(__file__), 'helper_source_hashes': PINS,
        'layer_records_sha256': sha(output/'layer_records.jsonl'), 'new_model_forwards': 0, 'new_GPU_operations': 0,
        'private_tensors_or_weights_public': False, 'task_quality_claim': False, 'no_proxy_performance_gate': True,
        'arithmetic': 'CPU FP32 attention probabilities/reads; Half V and head rounding; FP32 WO then Half output; not CUDA packed bitwise equality',
        'weights_used_only_for_posthoc_fixed_query_analysis_not_encoder_selection': True}
    write(output / 'summary.json', result)
    return result


def self_check(output):
    import torch
    import round9_value_bank_v1 as old
    import round10_value_bank_v1 as reference
    torch.manual_seed(10);n = 256
    state = SimpleNamespace(v_code=old._pack(torch.rand(1, 4, 128, n) > .5),
        v_scale=torch.full((1, 4, 128, 1), .5, dtype=torch.float16),
        v_min=torch.full((1, 4, 128, 1), -.3, dtype=torch.float16),
        depth_residual_lookup=torch.full((2, 4, 2), -1, dtype=torch.int32),
        depth_residual_code=old._pack(torch.rand(1, n, 128) > .5),
        depth_residual_alpha=torch.full((1, n), .1, dtype=torch.float16))
    state.depth_residual_lookup[1, 0, 0] = 0
    x = (torch.randn(1, 4, n, 128) * .25).half(); checks = []
    for window in (1, 32, 'global'):
        actual, decoded, carry, rem, stats = trace_feedback(state, x, window)
        expected, _ = reference.feedback_assign_window(state, x, window=window, backend='torch')
        require(torch.equal(actual.v_code, expected.v_code), 'reference FP32 recurrence packed identity')
        a = torch.rand(1, 28, 1, n);a = a / a.sum(-1, keepdim=True) * .4
        weighted_geometry(a, x, decoded, carry, rem, stats['actual_width'])
        checks.append('actual_Half_levels_and_packed_FP32_reference_' + str(window))
    a = torch.full((1, 28, 1, n), 1./n)
    _, one = coefficients(a, 1);_, thirtytwo = coefficients(a, 32)
    require(all(v == 1 for v in one) and all(abs(v - 1/32) < 1e-15 for v in thirtytwo), 'boundary coefficient controls')
    checks.append('terminal_H_W1_one_uniform_W32_one_over32')
    left = torch.zeros((1, 4, 128, n//32), dtype=torch.int32);right = left.clone();right[0, 0, 0, 0] = 3
    require(bit_changes(left, right) == 2, 'true bit count not changed words')
    checks.append('two_flipped_bits_in_one_packed_word')
    out = Path(output);out.mkdir(parents=True, exist_ok=False)
    result = {'status': 'PASS_CPU_MECHANISM_NUMERIC_CONTROLS', 'checks': checks, 'module_sha256': sha(__file__),
              'actual_GPU': False, 'actual_private_eight_source_positive': False, 'task_quality_claim': False}
    write(out/'CPU_RECEIPT.json', result);return result


def main():
    p = argparse.ArgumentParser();p.add_argument('--config', type=Path);p.add_argument('--output', type=Path, required=True)
    p.add_argument('--self-check', action='store_true');a = p.parse_args()
    result = self_check(a.output) if a.self_check else analyze(read(a.config), a.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__': main()
