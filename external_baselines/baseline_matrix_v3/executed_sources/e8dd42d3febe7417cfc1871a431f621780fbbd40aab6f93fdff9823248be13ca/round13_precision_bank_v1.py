"""Round13 all-support precision exchange and full-affine V2R transport.

Only new modules change. R is built by its frozen factory, its actual owner is
the per-request cap, and every logical V field is preserved in token-tier mode.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path
import time

import torch
import r7_legacy_bridge_v1 as bridge
import round12_static_bank_v1 as previous
import round12_static_codec_v1 as old_codec
import round13_precision_codec_v1 as codec

SCHEMA = 'trajkv.round13.precision.v1'
RUNTIME_FILES = ('round13_precision_bank_v1.py', 'round13_precision_codec_v1.py',
                 'round13_precision_attention_v1.py')


def require(value, message):
    if not value:
        raise ValueError('ROUND13_PRECISION_BANK: ' + message)


class TokenTierOwner(previous.Round12Owner):
    def named_tensors(self, include_constants=True):
        result = super().named_tensors(include_constants)
        for name in ('round13_high_code', 'round13_tier_rank', 'round13_high_book'):
            value = getattr(self.state, name, None)
            if isinstance(value, torch.Tensor):
                result[name] = value
        return result

    owner_tensors = named_tensors


def token_tier_budget(n, k_slots, cap_bytes, *, units=112, additional_book_bytes=2048,
                      map_bytes_per_token=4):
    """Largest shared high-token count under the exact original R owner cap."""
    require(n > 0 and k_slots >= 0 and units > 0, 'nonnegative source budget')
    fixed_delta = additional_book_bytes + map_bytes_per_token * n - 18 * k_slots * n
    h = min(n, max(0, (-fixed_delta) // (16 * units)))
    actual = cap_bytes + fixed_delta + 16 * units * h
    require(actual <= cap_bytes, 'even h=0 cannot fit both books/map after requested residual exchange')
    return {'visual_tokens': n, 'high_tokens': h, 'low_tokens': n - h,
            'R_cap_bytes': int(cap_bytes), 'expected_owner_bytes': int(actual),
            'K_residual_slots_refunded': int(k_slots), 'K_residual_refund_bytes': 18 * k_slots * n,
            'additional_high_book_bytes': additional_book_bytes, 'signed_rank_bytes': map_bytes_per_token * n,
            'high_index_increment_bytes': 16 * units * h, 'slack_bytes': int(cap_bytes - actual),
            'full_support': True, 'V_fields_frozen_to_request_R': True}


def _validate(owners, native_prefix_cache, extra=()):
    forbidden = {bridge._storage(t) for layer in native_prefix_cache.layers for t in (layer.keys, layer.values)}
    allowed = set(bridge._STATE_TENSORS) | {
        'round12_v_mask', 'round12_value_levels', *extra}
    for owner in owners:
        require(not any(hasattr(owner.state, name) for name in ('visual_k', 'visual_v')), 'native visual retained')
        require(all(not isinstance(value, torch.Tensor) or not value.numel() or name in allowed
                    for name, value in vars(owner.state).items()), 'unreported state tensor')
        for name, value in owner.named_tensors().items():
            require(bridge._storage(value) not in forbidden and not value.requires_grad and value.grad_fn is None,
                    'native alias/autograd: ' + name)


def _base(native_prefix_cache, visual_positions, variant, candidate, legacy_kwargs):
    return previous.build_round12_owners(native_prefix_cache, visual_positions, variant,
        candidate=candidate, **legacy_kwargs)


@torch.no_grad()
def build_token_tier_owners(native_prefix_cache, visual_positions, variant, *, candidate,
                            token_ranking, **legacy_kwargs):
    """Exchange request R K residuals for question-free high precision rows.

    token_ranking is an image-only permutation of original visual row numbers.
    All 112 units use the same high mask. Two index pools partition the rows;
    only one gain per row is retained. No K residual is retained.
    """
    started = time.perf_counter()
    owners0, metadata = _base(native_prefix_cache, visual_positions, variant, {'mode': 'R'}, legacy_kwargs)
    n = int(visual_positions.numel())
    device = visual_positions.device
    ranking = torch.as_tensor(token_ranking, device=device, dtype=torch.long)
    require(ranking.ndim == 1 and ranking.numel() == n and
            torch.equal(ranking.sort().values, torch.arange(n, device=device)), 'complete image-only row ranking')
    selected_R = deepcopy(metadata['selected_stages'])
    k_selected = [x for x in selected_R if x[0] == 'k']
    v_selected = [x for x in selected_R if x[0] == 'v']
    require(all(int(x[3]) == 0 for x in selected_R), 'frozen R depth-zero selected units')
    cap = previous.unique_capacity(owners0)
    budget = token_tier_budget(n, len(k_selected), cap)
    high_n, low_n = budget['high_tokens'], budget['low_tokens']
    high = torch.zeros(n, dtype=torch.bool, device=device)
    high[ranking[:high_n]] = True
    ranks = codec.tier_rank(high)
    high_positions = high.nonzero(as_tuple=False).flatten()
    low_positions = (~high).nonzero(as_tuple=False).flatten()
    data, asset = previous._read_asset(candidate, 'key_book_path', 'key_book_sha256')
    high_book = torch.tensor(data['centers'], device=device, dtype=torch.float16).contiguous()
    require(tuple(high_book.shape) == (256, 4) and bool(torch.isfinite(high_book).all()), 'hashed K2 G4_256 book')
    shared = owners0[0].shared
    require(tuple(shared['book'].shape) == (16, 4), 'original K1 G4_16 base retained')
    shared.pop('round12_format', None)
    shared['round13_precision_format'] = torch.tensor([13, 4, 4, 8], device=device, dtype=torch.uint8)
    shared['round13_high_book'] = high_book
    shared['round13_tier_rank'] = ranks
    lookup = shared['global_residual_lookup']
    lookup.fill_(-1)
    details, result = [], []
    with bridge._legacy_paths(Path(legacy_kwargs['legacy_a2']), Path(legacy_kwargs['legacy_experiments']),
                              Path(legacy_kwargs['legacy_scripts'])):
        wht = importlib.import_module('fast_wht').wht
        encode = importlib.import_module('round7_key_codec').encode_key
        reader = importlib.import_module('round13_precision_attention_v1').attention
        factors = shared['encoder_factor.key_head_factors']
        for layer, old in enumerate(owners0):
            state = old.state
            old_entries = previous._layer_entries(selected_R, layer)
            value_entries = [entry for entry in old_entries if entry[0] == 'v']
            if value_entries:
                slots = torch.tensor([old_entries.index(entry) for entry in value_entries], device=device)
                old_rc, old_ra = state.depth_residual_code, state.depth_residual_alpha
                state.depth_residual_code = old_rc.index_select(0, slots).contiguous()
                state.depth_residual_alpha = old_ra.index_select(0, slots).contiguous()
                for slot, (_, head, depth) in enumerate(value_entries):
                    lookup[layer, 1, head, depth] = slot
                    original_slot = old_entries.index(('v', head, depth))
                    require(torch.equal(state.depth_residual_code[slot], old_rc[original_slot]) and
                            torch.equal(state.depth_residual_alpha[slot], old_ra[original_slot]),
                            'logical V residual bytes changed')
                del old_rc, old_ra
            else:
                for name in ('depth_residual_code', 'depth_residual_alpha'):
                    if hasattr(state, name):
                        delattr(state, name)
            low_code = state.direction_code.index_select(2, low_positions).contiguous()
            gain = state.direction_gain.clone()
            if high_n:
                source = native_prefix_cache.layers[layer].keys.index_select(2, visual_positions).float()
                mean = source.mean(-2, keepdim=True).half().contiguous()
                rotated = wht(source - mean.float())
                high_code, high_gain = encode(rotated.index_select(2, high_positions), high_book,
                    amplitude='norm', assignment='l2', gain_factor=factors[layer][None, :, None, None])
                gain[..., high_positions, :] = high_gain
                del source, mean, rotated, high_gain
            else:
                high_code = low_code.new_empty((1, 4, 0, 32))
            state.direction_code = low_code
            state.direction_gain = gain.contiguous()
            state.round13_high_code = high_code
            state.round13_high_book = high_book
            state.round13_tier_rank = ranks
            state.round13_high_count, state.round13_low_count = high_n, low_n
            state.round13_key_residual_count = 0
            state.depth_residual_lookup = lookup[layer]
            require(tuple(state.v_code.shape) == (1, 4, 128, n // 32), 'V original all-row pool')
            result.append(TokenTierOwner(layer, state, shared, reader, old.boundary))
            details.append({'layer': layer, 'retained_V_entries': value_entries,
                            'logical_V_residual_fields_bit_exact': True,
                            'K_low_index_bytes': low_code.numel(), 'K_high_index_bytes': high_code.numel(),
                            'K_gain_bytes': gain.numel() * gain.element_size()})
    owners0.clear()
    actual = previous.unique_capacity(result)
    require(actual == budget['expected_owner_bytes'] and actual <= cap, 'complete actual token owner/cap mismatch')
    require(not bool((lookup[:, 0] >= 0).any()), 'a K residual survived precision exchange')
    _validate(result, native_prefix_cache, ('round13_high_code', 'round13_high_book', 'round13_tier_rank'))
    metadata['reference_R'] = metadata.pop('round12_static')
    metadata.update(selected_stages=v_selected, selected_stage_count=len(v_selected),
        returned_owners_unique_tensor_capacity_bytes=actual, reader_unchanged=False,
        original_FP16_visual_storage_retained_by_owners=False)
    metadata['round13_precision'] = {'schema': SCHEMA, 'mode': 'token_tier', **budget,
        'actual_owner_bytes': actual, 'original_R_selected_stages': selected_R,
        'same_request_R_K_residual_slots_removed': k_selected,
        'same_request_R_V_residual_slots_preserved': v_selected,
        'V_code_scale_min_preserved_without_reencoding': True,
        'V_residuals_preserved_by_original_layer_head_depth_token_position': True,
        'K_mean_preserved': True, 'all_original_visual_positions_preserved': True,
        'K_pool_contract': 'low16B/high32B disjoint compact rows; gain one selected-mode value per row',
        'tier_map': 'shared signed int32 dense rank; >=0 high, negative=-(low_rank+1)',
        'selector_name': candidate.get('selector_name'),
        'selector_question_free_required': True, 'selector_ranking_scope': 'original visual token row indices',
        'candidate_assets': {'key_book': asset}, 'layer_stats': details,
        'encoding_seconds_including_reference_R': time.perf_counter() - started,
        'CUDA_equivalence_established_by_factory': False,
        'source_sha256': {name: previous.sha(Path(__file__).with_name(name)) for name in RUNTIME_FILES}}
    require(previous.sha(asset['path']) == asset['sha256'], 'high-book asset changed during encoding')
    return result, metadata


@torch.no_grad()
def build_caltransport_owners(native_prefix_cache, visual_positions, variant, *, candidate, **legacy_kwargs):
    """Original low43 K/coverage, true re-encoded V2 with full old affine map."""
    started = time.perf_counter()
    owners0, metadata = bridge.build_r7_owners(native_prefix_cache, visual_positions, variant, **legacy_kwargs)
    n = int(visual_positions.numel())
    original = deepcopy(metadata['selected_stages'])
    require(len(original) == 43 and sum(x[0] == 'k' for x in original) == 23 and
            sum(x[0] == 'v' for x in original) == 20, 'original low43 K23/V20')
    assets = {}
    if candidate.get('ratio_path'):
        data, proof = previous._read_asset(candidate, 'ratio_path', 'ratio_sha256')
        assets['value_ratio'] = proof
        values = data.get('levels', data.get('value_levels'))
        require(values is not None, 'calibrated four ratio levels')
    else:
        require(candidate.get('technical_only') is True, 'formal full-affine V2 requires frozen ratio asset')
        values = candidate.get('value_levels', [-1.5104, -.45278, .45278, 1.5104])
    shared = owners0[0].shared
    levels = old_codec.normalized_levels(values, visual_positions.device)
    shared['value_levels'] = levels
    shared['round12_format'] = torch.tensor([1, 4, 4, 1], device=visual_positions.device, dtype=torch.uint8)
    factors = shared['encoder_factor.value_head_factors']
    result, details = [], []
    with bridge._legacy_paths(Path(legacy_kwargs['legacy_a2']), Path(legacy_kwargs['legacy_experiments']),
                              Path(legacy_kwargs['legacy_scripts'])):
        binary = importlib.import_module('quantizers').binary_centroids
        sign = importlib.import_module('round6_depth_codec').encode_sign_depth
        reader = importlib.import_module('round12_static_attention_v1').attention
        for layer, old in enumerate(owners0):
            state = old.state
            source = native_prefix_cache.layers[layer].values.index_select(2, visual_positions).detach().contiguous()
            raw_code, raw_scale, raw_min = binary(source.transpose(2, 3).contiguous(), group_size=n)
            require(torch.equal(raw_code, state.v_code), 'raw binary assignments differ from original low43 lineage')
            code, scale, minimum, diag = codec.encode_v2_transport(source, levels, raw_scale, raw_min,
                state.v_scale, state.v_min, factors[layer][None, :, None, None],
                iterations=int(candidate.get('value_fit_iterations', 12)))
            state.v_code, state.v_scale, state.v_min = code, scale, minimum
            entries = previous._layer_entries(original, layer)
            if entries:
                # K residual entries are copied without change; V entries are RE.
                state.depth_residual_code = state.depth_residual_code.clone()
                state.depth_residual_alpha = state.depth_residual_alpha.clone()
            decoded = old_codec.half_value(code, scale, minimum, bits=2, levels=levels)
            residual_stats = previous._reencode(state, entries, 'v', source[0],
                                                decoded.transpose(2, 3)[0], sign)
            state.round12_vmode = 1
            state.round12_value_levels = levels
            result.append(previous.Round12Owner(layer, state, shared, reader, old.boundary))
            details.append({'layer': layer, 'transport': diag, 'V_residual_reencoding': residual_stats})
            del source, raw_code, raw_scale, raw_min, decoded
    owners0.clear()
    actual = previous.unique_capacity(result)
    expected = previous.expected_owner_bytes('V2R', n, ratio_bytes=8)
    require(actual == expected, 'full-affine V2R actual complete owner differs from original V2R')
    _validate(result, native_prefix_cache)
    metadata.update(returned_owners_unique_tensor_capacity_bytes=actual, reader_unchanged=False,
                    original_FP16_visual_storage_retained_by_owners=False)
    metadata['round13_precision'] = {'schema': SCHEMA, 'mode': 'CalTransport',
        'actual_owner_bytes': actual, 'expected_owner_bytes': expected,
        'comparison': 'same low43 K fields/coverage and V2R bytes; change only inherited affine map, RE V residuals',
        'K_fields_preserved': True, 'selected_stages': original,
        'f0_transport': 'actual old binary raw/fold Half endpoints fit a,b, folded into existing V2 metadata',
        'extra_deployment_transport_tensor_bytes': 0, 'format_descriptor_bytes': 4,
        'ratio_bytes': 8, 'candidate_assets': assets, 'layer_stats': details,
        'encoding_seconds_including_original_low43': time.perf_counter() - started,
        'CUDA_equivalence_established_by_factory': False,
        'source_sha256': {name: previous.sha(Path(__file__).with_name(name)) for name in RUNTIME_FILES}}
    require(all(previous.sha(asset['path']) == asset['sha256'] for asset in assets.values()), 'ratio asset changed')
    return result, metadata


def build_round13_precision_owners(native_prefix_cache, visual_positions, variant, *, candidate,
                                  token_ranking=None, **legacy_kwargs):
    mode = candidate.get('mode')
    if mode == 'token_tier':
        require(token_ranking is not None, 'image-only token ranking required')
        return build_token_tier_owners(native_prefix_cache, visual_positions, variant,
            candidate=candidate, token_ranking=token_ranking, **legacy_kwargs)
    if mode in ('CalTransport', 'caltransport'):
        return build_caltransport_owners(native_prefix_cache, visual_positions, variant,
            candidate=candidate, **legacy_kwargs)
    raise ValueError('ROUND13_PRECISION_BANK: unknown precision mode ' + str(mode))
