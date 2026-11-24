"""Round12 true packed static endpoints, preserving frozen low43 lineage.

The original bridge builds a disposable low bank. Changed axes are encoded
again from the caller's actual native source, with unchanged f0 applied once.
No native source or encoding workspace is retained by returned owners.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import time

import torch
import r7_legacy_bridge_v1 as bridge
import round12_static_codec_v1 as codec

MODES = ('A', 'V2R', 'B', 'C', 'D', 'G8', 'R')
RUNTIME_FILES = ('round12_static_bank_v1.py', 'round12_static_codec_v1.py', 'round12_static_attention_v1.py')
SCHEMA = 'trajkv.round12.static.v1'


def require(value, message):
    if not value:
        raise ValueError('ROUND12_STATIC: ' + message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_capacity(owners):
    unique = {}
    for owner in owners:
        for value in owner.named_tensors().values():
            if value.numel():
                unique.setdefault(bridge._storage(value), value.untyped_storage().nbytes())
    return sum(unique.values())


class Round12Owner(bridge.R7LegacyOwner):
    def named_tensors(self, include_constants=True):
        result = super().named_tensors(include_constants)
        mask = getattr(self.state, 'round12_v_mask', None)
        if isinstance(mask, torch.Tensor):
            result['encoder_constants.v_channel_mask'] = mask
        # levels share the same eight-byte storage as shared['value_levels'].
        levels = getattr(self.state, 'round12_value_levels', None)
        if isinstance(levels, torch.Tensor):
            result['encoder_constants.value_levels_view'] = levels
        return result

    owner_tensors = named_tensors


def _read_asset(candidate, path_key, sha_key):
    path = Path(candidate[path_key]).resolve()
    digest = candidate.get(sha_key)
    require(path.is_file() and isinstance(digest, str) and len(digest) == 64 and sha(path) == digest,
            'exact hashed asset required: ' + path_key)
    return json.loads(path.read_text(encoding='utf-8-sig')), {'path': str(path), 'sha256': digest}


def _layer_entries(selected, layer):
    return [(str(axis), int(head), int(stage)) for axis, index, head, stage in selected if int(index) == layer]


def _reencode(state, entries, axis, source, base, encode_sign_depth):
    stats = []
    for head in sorted({head for a, head, _ in entries if a == axis}):
        depths = [stage for a, h, stage in entries if a == axis and h == head]
        require(depths == list(range(len(depths))), 'ordered residual depths')
        codes, alphas, _, distortion = encode_sign_depth(source[head], base[head], len(depths))
        for stage in depths:
            slot = entries.index((axis, head, stage))
            state.depth_residual_code[slot].copy_(codes[stage])
            state.depth_residual_alpha[slot].copy_(alphas[stage])
        stats.append({'axis': axis, 'head': head, 'depth': len(depths), 'squared_error_by_depth': distortion})
    return stats


def expected_owner_bytes(mode, n, stages=43, format_bytes=4, ratio_bytes=0):
    fixed = {'A': (6382, 91653), 'V2R': (6382, 89733),
             'B': (7278, 97797), 'C': (5486, 93445),
             'D': (6382, 99589), 'G8': (5486, 155141)}
    if mode == 'R':
        return (3816 + 18 * stages) * n + 89733 + format_bytes
    rate, header = fixed[mode]
    return rate * n + header + format_bytes + ratio_bytes


@torch.no_grad()
def build_round12_owners(native_prefix_cache, visual_positions, variant, *, candidate,
        legacy_a2, legacy_experiments, legacy_scripts, kivi_source,
        expected_asset_sha256=None, expected_legacy_source_sha256=None):
    mode = candidate.get('mode')
    require(mode in MODES, 'unknown candidate mode')
    require(variant.get('name') == 'r7_joint_k1_gt_cond', 'original low43 variant required')
    require(not torch.is_autocast_enabled() and not torch.backends.cuda.matmul.allow_tf32,
            'FP16 original sources, autocast off and TF32 off')
    started = time.perf_counter()
    owners0, metadata = bridge.build_r7_owners(native_prefix_cache, visual_positions, variant,
        legacy_a2=legacy_a2, legacy_experiments=legacy_experiments,
        legacy_scripts=legacy_scripts, kivi_source=kivi_source,
        expected_asset_sha256=expected_asset_sha256,
        expected_legacy_source_sha256=expected_legacy_source_sha256)
    original = [list(x) for x in metadata['selected_stages']]
    require(len(original) == 43 and sum(x[0] == 'k' for x in original) == 23 and
            sum(x[0] == 'v' for x in original) == 20, 'concrete original K23/V20 coverage')
    n = int(visual_positions.numel())
    require(unique_capacity(owners0) == 4590 * n + 89733, 'original complete-owner formula')
    shared = owners0[0].shared
    device = visual_positions.device
    assets = {}
    key_changed = mode in ('A', 'B', 'C', 'D', 'G8')
    value_changed = mode in ('V2R', 'C', 'D')
    if key_changed:
        data, proof = _read_asset(candidate, 'key_book_path', 'key_book_sha256')
        assets['key_book'] = proof
        book = torch.tensor(data['centers'], device=device, dtype=torch.float16).contiguous()
        shape = {'A': (256, 4), 'C': (256, 4), 'B': (1024, 4), 'D': (1024, 4), 'G8': (4096, 8)}[mode]
        require(tuple(book.shape) == shape and bool(torch.isfinite(book).all()), 'candidate book shape/finite values')
        shared['book'] = book
    else:
        book = shared['book']
    vmode = 1 if mode == 'V2R' else (2 if mode in ('C', 'D') else 0)
    levels = None
    if mode == 'V2R':
        if candidate.get('ratio_path'):
            data, proof = _read_asset(candidate, 'ratio_path', 'ratio_sha256')
            assets['value_ratio'] = proof
            values = data.get('levels', data.get('value_levels'))
            require(values is not None, 'ratio asset needs four levels')
        else:
            require(candidate.get('technical_only') is True, 'formal V2R needs hashed calibrated ratio asset')
            values = candidate.get('value_levels', [-1.5104, -.45278, .45278, 1.5104])
        levels = codec.normalized_levels(values, device)
        shared['value_levels'] = levels
    # This real four-byte descriptor charges the new global format selection.
    kbits = (int(book.shape[0]) - 1).bit_length()
    shared['round12_format'] = torch.tensor([1, int(book.shape[1]), kbits, vmode], device=device, dtype=torch.uint8)
    selected = original
    if mode == 'R':
        target = expected_owner_bytes('G8', n)
        count = (target - (3816 * n + 89733 + 4)) // (18 * n)
        ranking = [list(x) for x in variant['selected_stages']]
        require(ranking[:43] == original and count >= 43, 'R protects original43 before appending')
        selected = ranking[:min(int(count), len(ranking))]
    selected = deepcopy(selected)
    lookup = shared['global_residual_lookup']
    if mode == 'R':
        lookup.fill_(-1)
    with bridge._legacy_paths(Path(legacy_a2), Path(legacy_experiments), Path(legacy_scripts)):
        wht = importlib.import_module('fast_wht').wht
        legacy_key = importlib.import_module('round7_key_codec')
        sign_encoder = importlib.import_module('round6_depth_codec').encode_sign_depth
        reader = importlib.import_module('round12_static_attention_v1').attention
        f0 = json.loads(Path(variant['metadata_factor_path']).read_text(encoding='utf-8-sig'))
        gain_mode = f0['key_gain_mode']
        kf = shared['encoder_factor.key_head_factors']
        vf = shared['encoder_factor.value_head_factors']
        vo = shared['encoder_factor.value_center_offsets']
        stats = []
        result = []
        for layer, old in enumerate(owners0):
            state = old.state
            entries = _layer_entries(selected, layer)
            old_entries = _layer_entries(original, layer)
            if mode == 'R':
                if entries:
                    state.depth_residual_code = torch.empty((len(entries), n, 4), dtype=torch.int32, device=device)
                    state.depth_residual_alpha = torch.empty((len(entries), n), dtype=torch.float16, device=device)
                    for slot, (axis, head, depth) in enumerate(entries):
                        lookup[layer, int(axis == 'v'), head, depth] = slot
                else:
                    for name in ('depth_residual_code', 'depth_residual_alpha'):
                        if hasattr(state, name):
                            delattr(state, name)
            elif (key_changed or value_changed) and old_entries:
                # The unchanged axis pool is preserved byte-for-byte.
                state.depth_residual_code = state.depth_residual_code.clone()
                state.depth_residual_alpha = state.depth_residual_alpha.clone()
            detail = {'layer': layer, 'entries': entries, 'residual_distortion': []}
            if key_changed or mode == 'R':
                source = native_prefix_cache.layers[layer].keys.index_select(2, visual_positions).float()
                mean = source.mean(-2, keepdim=True).half().contiguous()
                rotated = wht(source - mean.float())
                if key_changed:
                    factor = kf[layer][None, :, None, None]
                    if tuple(book.shape) == (256, 4):
                        code, gain = legacy_key.encode_key(rotated, book, amplitude='norm', assignment='l2', gain_factor=factor)
                    else:
                        code, gain = codec.encode_key(rotated, book, gain_factor=factor,
                            chunk_parts=int(candidate.get('key_chunk_parts', 4096)))
                    state.direction_code, state.direction_gain = code, gain
                    state.direction_mean = ((mean.float() * factor).half().contiguous()
                                            if gain_mode == 'whole' else mean)
                base = codec.decode_key(state.direction_code, state.direction_gain, book)
                if entries:
                    detail['residual_distortion'] += _reencode(state, entries, 'k', rotated[0], base[0], sign_encoder)
                del source, mean, rotated, base
            if value_changed or mode == 'R':
                source = native_prefix_cache.layers[layer].values.index_select(2, visual_positions).detach().contiguous()
                if mode == 'V2R':
                    state.v_code, state.v_scale, state.v_min = codec.encode_v2_ratio(source, levels,
                        vf[layer][None, :, None, None], vo[layer][None, :, None, None],
                        iterations=int(candidate.get('value_fit_iterations', 12)))
                elif mode in ('C', 'D'):
                    state.v_code, state.v_scale, state.v_min, state.round12_v_mask = codec.compact_vhalf(
                        state.v_code, state.v_scale, state.v_min)
                if mode in ('C', 'D'):
                    decoded = codec.decode_vhalf(state.v_code, state.v_scale, state.v_min, state.round12_v_mask)
                else:
                    decoded = codec.half_value(state.v_code, state.v_scale, state.v_min,
                        bits=2 if mode == 'V2R' else 1, levels=levels)
                if entries:
                    detail['residual_distortion'] += _reencode(state, entries, 'v', source[0], decoded.transpose(2, 3)[0], sign_encoder)
                del source, decoded
            state.depth_residual_lookup = lookup[layer]
            state.round12_vmode = vmode
            if levels is not None:
                state.round12_value_levels = levels
            result.append(Round12Owner(layer, state, shared, reader, old.boundary))
            stats.append(detail)
    owners0.clear()
    actual = unique_capacity(result)
    expected = expected_owner_bytes(mode, n, len(selected), ratio_bytes=8 if mode == 'V2R' else 0)
    require(actual == expected, f'complete actual owner {actual} differs from formula {expected}')
    forbidden = {bridge._storage(t) for layer in native_prefix_cache.layers for t in (layer.keys, layer.values)}
    allowed = set(bridge._STATE_TENSORS) | {'round12_v_mask', 'round12_value_levels'}
    for owner in result:
        require(not any(hasattr(owner.state, name) for name in ('visual_k', 'visual_v')), 'native visual source retained')
        require(all(not isinstance(t, torch.Tensor) or not t.numel() or key in allowed
                    for key, t in vars(owner.state).items()), 'unreported state tensor')
        for name, tensor in owner.named_tensors().items():
            require(bridge._storage(tensor) not in forbidden and not tensor.requires_grad and tensor.grad_fn is None,
                    'native alias/autograd retained: ' + name)
    metadata.update(selected_stages=selected, selected_stage_count=len(selected),
        returned_owners_unique_tensor_capacity_bytes=actual, reader_unchanged=False,
        original_FP16_visual_storage_retained_by_owners=False)
    metadata['round12_static'] = {
        'schema': SCHEMA, 'mode': mode, 'encoding_applied': True,
        'technical_only': bool(candidate.get('technical_only', False)),
        'base_lineage': 'original_r7_joint_k1_gt_cond',
        'original43_selected_stages': original, 'fixed_original43': mode != 'R',
        'selected_stages': selected, 'key_words': int(book.shape[0]), 'key_block_dim': int(book.shape[1]),
        'key_index_bits': kbits, 'key_payload_bytes_per_token_head': 128 // int(book.shape[1]) * kbits // 8,
        'value_mode': ('V1', 'V2R', 'Vhalf')[vmode], 'actual_owner_bytes': actual,
        'expected_complete_owner_bytes': expected, 'format_descriptor_bytes': 4,
        'value_level_bytes': 8 if levels is not None else 0,
        'value_mask_bytes': 1792 if vmode == 2 else 0,
        'mask_scope': 'per_image_per_layer_head' if vmode == 2 else None,
        'unchanged_axis_fields_preserved': True, 'f0_applied_once': True,
        'native_source_retained': False, 'candidate_assets': assets, 'layer_stats': stats,
        'encoding_seconds_including_lineage_build_and_diagnostics': time.perf_counter() - started,
        'source_sha256': {name: sha(Path(__file__).with_name(name)) for name in RUNTIME_FILES},
        'CUDA_equivalence_established_by_factory': False,
    }
    require(all(sha(v['path']) == v['sha256'] for v in assets.values()), 'candidate asset changed during encoding')
    require(all(sha(v['path']) == v['sha256'] for v in metadata['fixed_assets'].values()), 'frozen lineage asset changed')
    return result, metadata
