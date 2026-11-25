"""Round13 contextualized-then-compacted real packed A visual prefix.

The complete native image prefix is encoded using the unchanged Round12 A
encoder before any packed row is removed. Retained metadata, codewords, gains,
and residuals are exact gathers of that full-N representation. Native text is
excluded with ORIGINAL visual positions by HybridVisualCache; only the packed
visual owner's positions are sparse. Original logical/RoPE positions survive.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

import torch

import r7_legacy_bridge_v1 as bridge
import round12_static_bank_v1 as full_bank
import round12_static_codec_v1 as codec
from round13_selector_v1 import SELECTORS, select_visual_rows

SCHEMA = 'trajkv.round13.contextualized_pruned_owner.v1'
RUNTIME_FILES = ('round13_pruned_bank_v1.py', 'round13_selector_v1.py',
                 *full_bank.RUNTIME_FILES)
P_DESCRIPTOR_BYTES = 16


def require(value, message):
    if not value:
        raise ValueError('ROUND13_PRUNED: ' + message)


def budget_plan(n, *, descriptor_bytes=P_DESCRIPTOR_BYTES):
    """Exact complete-owner budget, measured relative to original image rows.

    Sparse A needs its own kept positions (already in 6382*M), original VP for
    enforcing text exclusion (8*N), and a real four-int32 format descriptor.
    No uncharged retained row-ID list is needed: kept VP is the reader map.
    """
    require(type(n) is int and n > 0 and n % 32 == 0, 'original N must be divisible32')
    cap = 4590 * n + 89733
    fixed = 91657 + 8 * n + int(descriptor_bytes)
    m = min(n, max(0, (cap - fixed) // (32 * 6382) * 32))
    fallback = m < 256
    actual = cap if fallback else 6382 * m + fixed
    return {'N': n, 'M': n if fallback else m, 'proposed_M': m,
            'original_low_bytes_cap': cap, 'expected_owner_bytes': actual,
            'sparse_A_fixed_bytes': fixed, 'extra_original_position_map_bytes': 8 * n,
            'new_descriptor_bytes': int(descriptor_bytes), 'fallback_original_low': fallback,
            'strict_cap_satisfied': actual <= cap, 'alignment': 32}


def validate_sparse_causal_positions(kept_positions, original_positions, exact_positions,
                                     query_positions, boundary):
    """Strict history contract: missing rows must be precisely visual pruning.

    It is illegal to smuggle pruned image rows into the native-FP16 text owner.
    This check does not reconstruct any key/value or silently renumber a row.
    """
    require(all(t.ndim == 1 and t.dtype == torch.long for t in
                (kept_positions, original_positions, exact_positions, query_positions)), 'int64 serial position vectors required')
    n = query_positions.numel()
    require(n > 0 and exact_positions.numel() >= n, 'queries and exact text suffix required')
    logical = int(query_positions[-1]) + 1
    expected_query = torch.arange(logical - n, logical, device=query_positions.device)
    require(torch.equal(query_positions, expected_query) and int(query_positions[0]) >= boundary,
            'queries must remain original contiguous post-boundary positions')
    require(torch.equal(exact_positions[-n:], query_positions), 'exact text must include current query suffix')
    require(torch.equal(original_positions, original_positions.unique(sorted=True))
            and torch.equal(kept_positions, kept_positions.unique(sorted=True)), 'original/kept positions must be sorted unique')
    require(bool(torch.isin(kept_positions, original_positions).all()), 'kept positions outside original image')
    union = torch.cat((original_positions, exact_positions)).sort().values
    require(torch.equal(union, torch.arange(logical, device=query_positions.device)),
            'original visual plus exact text must cover causal history exactly; pruned visual cannot be text')
    return logical


class PrunedOwner(full_bank.Round12Owner):
    def __init__(self, layer_idx, state, shared, reader, boundary):
        super().__init__(layer_idx, state, shared, reader, boundary)
        self.original_positions = shared['round13_original_positions']

    def attend(self, q, *, exact_keys, exact_values, exact_positions, query_positions, scale=None):
        require(not self._active, 'concurrent layer owner use unsupported')
        require(q.ndim == 4 and tuple(q.shape[:2]) == (1, 28) and q.shape[-1] == 128,
                'M1 batch1 Hq28 D128 only')
        require(exact_keys.shape == exact_values.shape and exact_keys.ndim == 4
                and tuple(exact_keys.shape[:2]) == (1, 4) and exact_keys.shape[-1] == 128,
                'exact native text GQA geometry mismatch')
        require(q.stride(-1) == 1 and exact_keys.stride(-1) == exact_values.stride(-1) == 1
                and exact_keys.stride(-2) == exact_values.stride(-2) == 128
                and exact_keys.stride(1) == exact_values.stride(1), 'row-contiguous matching text strides required')
        require(q.shape[-2] == query_positions.numel() and exact_keys.shape[-2] == exact_positions.numel(),
                'query/text position count differs')
        logical = validate_sparse_causal_positions(self.positions, self.original_positions,
                                                   exact_positions, query_positions, self.boundary)
        require(scale is None or math.isclose(float(scale), 128 ** -.5, rel_tol=0., abs_tol=1e-12),
                'reader fixed D128 attention scale')
        require(q.is_cuda and q.dtype == exact_keys.dtype == exact_values.dtype == torch.float16
                and q.device == exact_keys.device == exact_values.device == self.positions.device,
                'same-device CUDA FP16 actual packed reader required')
        self._active = True
        try:
            self.state.text_k, self.state.text_v = exact_keys, exact_values
            self.state.text_pos, self.state.text_length, self.state.length = exact_positions, exact_positions.numel(), logical
            # Original full-N affine values are retained; the packed row stride
            # and metadata group length now both refer to compact M, not N.
            return self._reader(q, self.state, self.shared['book'], self.positions.numel())
        finally:
            self._clear_text()
            self._active = False


@torch.no_grad()
def compact_full_a_owners(owners, kept_rows, original_visual_positions, *, selector_id=0, seed=0):
    """Destructive full-A->compact-A conversion, also usable in CPU format QA.

    M=N is the exact identity operation: returns the original owners and stores
    no extra descriptor or duplicate map. A full-N identity is a technical
    control and generally does not satisfy the original-low storage budget.
    """
    n, m = int(original_visual_positions.numel()), int(kept_rows.numel())
    require(n % 32 == m % 32 == 0 and m > 0, 'whole real V words, no padding')
    require(kept_rows.dtype == torch.long and torch.equal(kept_rows, kept_rows.unique(sorted=True))
            and int(kept_rows[0]) >= 0 and int(kept_rows[-1]) < n, 'sorted unique original row IDs')
    require(owners and all(torch.equal(o.positions, original_visual_positions) for o in owners),
            'input must be complete original-N A owners')
    if m == n:
        require(torch.equal(kept_rows, torch.arange(n, device=kept_rows.device)), 'M=N identity IDs differ')
        return owners
    shared = owners[0].shared
    require(all(o.shared is shared for o in owners), 'shared full-A constant pool required')
    original = original_visual_positions.detach().clone().contiguous()
    positions = original.index_select(0, kept_rows).contiguous()
    shared['round13_original_positions'] = original
    shared['round13_pruned_format'] = torch.tensor([1, int(selector_id), int(seed), n],
                                                  device=original.device, dtype=torch.int32)
    result = []
    for old in owners:
        state = old.state
        require(int(state.round12_vmode) == 0 and state.v_code.shape[-1] * 32 == n,
                'only full-A whole-image V1 has been admitted')
        require(tuple(state.direction_code.shape) == (1, 4, n, 32)
                and tuple(state.direction_gain.shape) == (1, 4, n, 1), 'actual K2 G4_256 full-row layout required')
        state.direction_code = state.direction_code.index_select(2, kept_rows).contiguous()
        state.direction_gain = state.direction_gain.index_select(2, kept_rows).contiguous()
        # V words are packed ACROSS token rows; slicing words is not a gather.
        full_bits = codec.unpack_channel_codes(state.v_code, 1, n)
        state.v_code = codec.pack_channel_codes(full_bits.index_select(-1, kept_rows), 1)
        if hasattr(state, 'depth_residual_code'):
            require(state.depth_residual_code.shape[-2] == n and state.depth_residual_alpha.shape[-1] == n,
                    'full-N residual pools required before row compaction')
            state.depth_residual_code = state.depth_residual_code.index_select(1, kept_rows).contiguous()
            state.depth_residual_alpha = state.depth_residual_alpha.index_select(1, kept_rows).contiguous()
        # direction_mean, v_scale and v_min remain EXACT full-image values.
        state.visual_pos = positions
        result.append(PrunedOwner(old.layer_idx, state, shared, old._reader, old.boundary))
    return result


@torch.no_grad()
def build_pruned_owners(native_prefix_cache, visual_positions, variant, *, candidate,
                        legacy_a2, legacy_experiments, legacy_scripts, kivi_source,
                        expected_asset_sha256=None, expected_legacy_source_sha256=None):
    """Factory-compatible entry point returning (owners, JSON metadata).

    Caller MUST construct HybridVisualCache(native, ORIGINAL visual_positions,
    owners), then consume/release native cache before measuring persistence.
    """
    started = time.perf_counter()
    selector = candidate.get('selector', 'importance_diversity')
    require(selector in SELECTORS, 'known image-only selector required')
    n = int(visual_positions.numel())
    plan = budget_plan(n)
    technical = bool(candidate.get('technical_keep_all', False))
    require(not technical or candidate.get('technical_only') is True, 'identity overbudget is technical-only')
    if plan['fallback_original_low'] and not technical:
        owners, metadata = bridge.build_r7_owners(native_prefix_cache, visual_positions, variant,
            legacy_a2=legacy_a2, legacy_experiments=legacy_experiments,
            legacy_scripts=legacy_scripts, kivi_source=kivi_source,
            expected_asset_sha256=expected_asset_sha256,
            expected_legacy_source_sha256=expected_legacy_source_sha256)
        actual = full_bank.unique_capacity(owners)
        require(actual == plan['original_low_bytes_cap'], 'fallback frozen low owner byte mismatch')
        metadata['round13_pruned'] = {'schema': SCHEMA, 'selector': selector, 'N': n, 'M': n,
            'fallback_original_low': True, 'fallback_reason': 'complete_thumbnail_cannot_fit_A_at_original_low_cap',
            'budget_plan': plan, 'actual_owner_bytes': actual, 'original_absolute_positions_preserved': True}
        return owners, metadata
    m = n if technical else plan['M']
    if candidate.get('keep_count') is not None:
        supplied = int(candidate['keep_count'])
        require(supplied % 32 == 0 and 256 <= supplied <= m, 'supplied keep count must fit exact planned budget')
        m = supplied
    # Encode all original rows before selecting/compacting any code field.
    acandidate = {key: candidate[key] for key in ('key_book_path', 'key_book_sha256', 'key_chunk_parts')
                  if key in candidate}
    acandidate.update(mode='A', technical_only=bool(candidate.get('technical_only', False)))
    owners_a, metadata = full_bank.build_round12_owners(native_prefix_cache, visual_positions, variant,
        candidate=acandidate, legacy_a2=legacy_a2, legacy_experiments=legacy_experiments,
        legacy_scripts=legacy_scripts, kivi_source=kivi_source,
        expected_asset_sha256=expected_asset_sha256,
        expected_legacy_source_sha256=expected_legacy_source_sha256)
    rows, selection = select_visual_rows(native_prefix_cache, visual_positions, m, selector=selector,
        thumbnail_indices=candidate.get('thumbnail_indices'), thumbnail_required=True,
        seed=int(candidate.get('selector_seed', 0)), config=candidate.get('selector_config'))
    owners = compact_full_a_owners(owners_a, rows, visual_positions,
                                  selector_id=SELECTORS.index(selector), seed=int(candidate.get('selector_seed', 0)))
    if owners is not owners_a:
        owners_a.clear()
    actual = full_bank.unique_capacity(owners)
    expected = 6382 * m + 91657 + (0 if m == n else 8 * n + P_DESCRIPTOR_BYTES)
    require(actual == expected, f'actual physical unique storage {actual} differs from exact formula {expected}')
    require(technical or actual <= plan['original_low_bytes_cap'], 'complete owner exceeds original low per-image cap')
    forbidden = {bridge._storage(t) for layer in native_prefix_cache.layers for t in (layer.keys, layer.values)}
    for owner in owners:
        require(not any(hasattr(owner.state, name) for name in ('visual_k', 'visual_v')), 'native image tensors retained')
        for name, tensor in owner.named_tensors().items():
            require(bridge._storage(tensor) not in forbidden and not tensor.requires_grad and tensor.grad_fn is None,
                    'native storage/autograd retained: ' + name)
    metadata['returned_owners_unique_tensor_capacity_bytes'] = actual
    metadata['round13_pruned'] = {'schema': SCHEMA, 'selector': selector, 'N': n, 'M': m,
        'retained_fraction': m / n, 'fallback_original_low': False, 'technical_only': technical,
        'budget_plan': plan, 'actual_owner_bytes': actual, 'expected_owner_bytes': expected,
        'strict_original_low_cap_satisfied': actual <= plan['original_low_bytes_cap'],
        'effective_bits_per_original_element': actual * 8 / (n * 28 * 4 * 128 * 2),
        'bits_per_retained_element': actual * 8 / (m * 28 * 4 * 128 * 2),
        'full_native_prefill_before_pruning': True, 'full_A_encoding_before_compaction': True,
        'full_N_means_scales_gains_and_residuals_exact_gather': True,
        'M_equals_N_returns_original_A_owners_without_mutation': m == n,
        'original_absolute_positions_preserved': True, 'original_logical_prefix_length': int(owners[0].boundary),
        'cache_constructor_must_receive_original_visual_positions': True,
        'pruned_visual_rows_must_not_be_native_FP16_text': True,
        'selection': selection, 'encoding_host_elapsed_seconds': time.perf_counter() - started,
        'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in RUNTIME_FILES},
        'CUDA_equivalence_established_by_factory': False}
    return owners, metadata
