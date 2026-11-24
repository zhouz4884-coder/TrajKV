"""Synthetic CUDA QA0: new A versus exact frozen original high base fields.

28 layers, four KV heads, D128, Nv256, three nonvisual prefix positions.
No model, image, task inputs, reference answers, free generation, or fitting.
Residual pool values are deliberately excluded from the base comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))


def require(value, message):
    if not value:
        raise ValueError('ROUND12_A_HIGH_BASE_DIAGNOSTIC: ' + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(tensor):
    import torch
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_comparison(left, right):
    import torch
    result = {'left_shape': list(left.shape), 'right_shape': list(right.shape),
              'left_dtype': str(left.dtype), 'right_dtype': str(right.dtype),
              'left_sha256': tensor_sha(left), 'right_sha256': tensor_sha(right)}
    compatible = left.shape == right.shape and left.dtype == right.dtype
    result['same_shape_dtype'] = compatible
    result['bit_exact'] = compatible and result['left_sha256'] == result['right_sha256']
    if compatible:
        result['differing_elements'] = int((left != right).sum())
        result['differing_bytes'] = int((left.contiguous().view(torch.uint8) != right.contiguous().view(torch.uint8)).sum())
        result['max_abs_numeric_difference'] = float((left.double() - right.double()).abs().max()) if left.numel() else 0.
    return result


def source_hashes(cache):
    return [{'layer': i, 'K_sha256': tensor_sha(layer.keys), 'V_sha256': tensor_sha(layer.values)}
            for i, layer in enumerate(cache.layers)]


def effective_parameters(config, variant, factor_data, book_sha):
    return {'effective_book_sha256': book_sha,
            'key_assignment': variant.get('key_assignment', 'l2'),
            'key_amplitude': variant.get('key_amplitude', 'norm'),
            'key_gain_mode': factor_data['key_gain_mode'],
            'V_bits': variant.get('bits_v', 1), 'V_layout': variant.get('v_layout', 'channel'),
            'V_encoder': variant.get('v_encoder', 'lloyd'),
            'V_group_size': variant.get('v_group_size', 'all'),
            'model_id': config.get('model_id'), 'head_dim': 128,
            'mean_is_source_FP32_token_mean_rounded_Half': True,
            'K_frame': 'normalized128_WHT_of_source_minus_Half_mean',
            'assignment_distance': 'x2_plus_c2_minus_2_x_matmul_cT_chunk65536',
            'gain_rounding': 'Half(norm_source/norm_template)_then_Half_times_f0',
            'V_encoding_order': 'binary_centroids_3_iterations_then_original_fold_bits1_once'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a-config', type=Path, required=True)
    p.add_argument('--high-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    aconfig = json.loads(args.a_config.read_text(encoding='utf-8-sig'))
    hconfig = json.loads(args.high_config.read_text(encoding='utf-8-sig'))
    require(aconfig.get('candidate', {}).get('mode') == 'A', 'actual A configuration required')
    require(aconfig['variant_source']['name'] == 'r7_joint_k1_gt_cond'
            and hconfig['variant_source']['name'] == 'r7_budget_k2_cond', 'exact current cond profiles required')
    require(not (set(aconfig) | set(hconfig)) & {'answers', 'labels', 'references', 'reference_answers', 'scores'},
            'unexpected reference fields in configurations')
    from round10_runner_v1 import frozen_variant
    from round12_static_bank_v1 import build_round12_owners
    from r7_legacy_bridge_v2 import build_r7_owners
    import r7_legacy_bridge_v1 as low_bridge
    avariant, hvariant = frozen_variant(aconfig), frozen_variant(hconfig)
    afactors = json.loads(Path(avariant['metadata_factor_path']).read_text(encoding='utf-8-sig'))
    hfactors = json.loads(Path(hvariant['metadata_factor_path']).read_text(encoding='utf-8-sig'))
    require(aconfig['candidate']['key_book_sha256'] == sha(aconfig['candidate']['key_book_path'])
            == sha(hvariant['direction_book_path']), 'A effective K2 book differs from frozen high')
    for name in ('legacy_a2', 'legacy_experiments', 'legacy_scripts', 'kivi_source'):
        require(aconfig[name] == hconfig[name], 'different legacy dependency: ' + name)
    locked_files = {str(args.a_config): sha(args.a_config), str(args.high_config): sha(args.high_config),
                    aconfig['candidate']['key_book_path']: aconfig['candidate']['key_book_sha256']}
    for config, variant in ((aconfig, avariant), (hconfig, hvariant)):
        locked_files[config['variant_source']['path']] = config['variant_source']['sha256']
        locked_files.update({variant[key]: digest for key, digest in config['expected_asset_sha256'].items()})
    output = args.output.resolve()
    require(str(output).startswith('/media/') and '/baseline_matrix_v3/results/round12' in str(output), '8TB project data disk required')
    output.mkdir(parents=True, exist_ok=False)
    import torch
    from types import SimpleNamespace
    require(torch.cuda.is_available() and args.device.startswith('cuda'), 'actual CUDA required')
    device = torch.device(args.device)
    require(torch.cuda.get_device_capability(device) == (8, 6), 'actual RTX3090 SM86 required')
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    gen = torch.Generator(device=device).manual_seed(1200266)
    layers = [SimpleNamespace(
        keys=(torch.randn((1, 4, 259, 128), device=device, generator=gen) * (0.4 + i / 56)).half().contiguous(),
        values=(torch.randn((1, 4, 259, 128), device=device, generator=gen) * (0.5 + i / 84) + (i - 14) / 70).half().contiguous())
        for i in range(28)]
    cache = low_bridge._CacheProxy(layers)
    vp = torch.arange(3, 259, device=device, dtype=torch.long)
    before = source_hashes(cache)
    kwargs = {name: aconfig[name] for name in ('legacy_a2', 'legacy_experiments', 'legacy_scripts', 'kivi_source')}
    aowners, ameta = build_round12_owners(cache, vp, avariant, candidate=aconfig['candidate'],
        expected_asset_sha256=aconfig['expected_asset_sha256'], **kwargs)
    after_a = source_hashes(cache)
    require(after_a == before, 'A changed synthetic source tensors')
    howners, hmeta = build_r7_owners(cache, vp, hvariant,
        expected_asset_sha256=hconfig['expected_asset_sha256'], **kwargs)
    after_h = source_hashes(cache)
    require(after_h == before, 'frozen high changed synthetic source tensors')
    fields = ('direction_code', 'direction_gain', 'direction_mean', 'v_code', 'v_scale', 'v_min')
    comparisons = []
    for layer, (left, right) in enumerate(zip(aowners, howners)):
        comparisons.append({'layer': layer,
            'fields': {name: tensor_comparison(getattr(left.state, name), getattr(right.state, name)) for name in fields}})
    numeric_factors = {}
    for name in ('key_head_factors', 'value_head_factors', 'value_center_offsets'):
        numeric_factors[name] = {
            'JSON_numeric_values_identical': afactors[name] == hfactors[name],
            'loaded_actual_FP32': tensor_comparison(aowners[0].shared['encoder_factor.' + name],
                                                   howners[0].shared['encoder_factor.' + name])}
    book_comparison = tensor_comparison(aowners[0].shared['book'], howners[0].shared['book'])
    params_a = effective_parameters(aconfig, avariant, afactors, aconfig['candidate']['key_book_sha256'])
    params_h = effective_parameters(hconfig, hvariant, hfactors, sha(hvariant['direction_book_path']))
    factors_match = all(v['JSON_numeric_values_identical'] and v['loaded_actual_FP32']['bit_exact']
                        for v in numeric_factors.values())
    fields_exact = all(value['bit_exact'] for row in comparisons for value in row['fields'].values())
    source_pins = {path: value['sha256'] for meta in (ameta, hmeta)
                   for path, value in meta['legacy_sources'].items()}
    require(all(sha(path) == digest for path, digest in source_pins.items()), 'legacy source changed')
    require(all(sha(path) == digest for path, digest in locked_files.items()), 'configuration/asset changed')
    current_sources = ('round12_a_high_base_diagnostic_v1.py', 'round12_static_bank_v1.py',
                       'round12_static_codec_v1.py', 'round12_static_attention_v1.py',
                       'r7_legacy_bridge_v1.py', 'r7_legacy_bridge_v2.py', 'round10_runner_v1.py')
    torch.cuda.synchronize(device)
    matched = fields_exact and factors_match and book_comparison['bit_exact'] and params_a == params_h
    report = {'schema': 'trajkv.round12.A_high_synthetic_base.v1',
        'status': 'PASS_SYNTHETIC_CUDA_BASE_BIT_EXACT' if matched else 'COMPLETE_SYNTHETIC_BASE_DIFFERENCES',
        'actual_CUDA': True, 'model_forward_calls': 0, 'task_answer_events': 0,
        'synthetic_tensors_are_not_actual_model_native_outputs': True,
        'image_or_task_dataset_read': False, 'question_or_answer_tokens_consumed': False,
        'candidate_or_f0_modified': False,
        'geometry': {'layers': 28, 'KV_heads': 4, 'head_dim': 128, 'visual_tokens': 256,
                     'nonvisual_prefix_tokens': 3, 'total_prefix_tokens': 259, 'dtype': 'float16',
                     'visual_positions': [3, 258], 'seed': 1200266},
        'A_config': {'path': str(args.a_config), 'sha256': sha(args.a_config)},
        'high_config': {'path': str(args.high_config), 'sha256': sha(args.high_config)},
        'effective_A_parameters': params_a, 'effective_high_parameters': params_h,
        'effective_base_parameter_values_identical': params_a == params_h,
        'f0_factor_files': {'A': {'path': avariant['metadata_factor_path'], 'sha256': sha(avariant['metadata_factor_path'])},
                            'high': {'path': hvariant['metadata_factor_path'], 'sha256': sha(hvariant['metadata_factor_path'])}},
        'numeric_factors': numeric_factors, 'effective_book': book_comparison,
        'base_fields_all_bit_exact': fields_exact, 'per_layer_base_fields': comparisons,
        'source_before': before, 'source_unchanged_after_A': after_a == before,
        'source_unchanged_after_high': after_h == before,
        'A_actual_selected_stages': ameta['selected_stages'], 'high_actual_selected_stages': hmeta['selected_stages'],
        'A_stage_axis_counts': dict(Counter(stage[0] for stage in ameta['selected_stages'])),
        'high_stage_axis_counts': dict(Counter(stage[0] for stage in hmeta['selected_stages'])),
        'residual_pool_values_compared_as_base_fields': False,
        'A_complete_packed_owner_bytes': ameta['returned_owners_unique_tensor_capacity_bytes'],
        'high_complete_packed_owner_bytes': hmeta['returned_owners_unique_tensor_capacity_bytes'],
        'interpretation': 'A positive result establishes base-field identity on this synthetic geometry only; task or all-real-source equivalence is not implied.',
        'executed_legacy_source_sha256': source_pins,
        'locked_config_and_asset_sha256': locked_files,
        'source_sha256': {name: sha(ROOT / 'code' / name) for name in current_sources},
        'GPU': torch.cuda.get_device_name(device), 'torch_version': torch.__version__,
        'elapsed_seconds': time.perf_counter() - started}
    path = output / 'A_HIGH_SYNTHETIC_BASE_v1.json'
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(path), 'sha256': sha(path),
                      'task_QA': 0, 'base_fields_bit_exact': fields_exact}), flush=True)
    return 0 if matched else 2


if __name__ == '__main__':
    raise SystemExit(main())
