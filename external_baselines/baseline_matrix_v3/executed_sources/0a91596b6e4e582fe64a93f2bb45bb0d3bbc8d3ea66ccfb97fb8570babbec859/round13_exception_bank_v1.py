"""Round13 physically compact low43 rows plus native FP16 visual exceptions.

Every visual row keeps its original position. Native exception K is already
post-RoPE and is never transformed or metadata-calibrated again. The original
low bank is first encoded over the full image; compacting changes neither its
metadata statistics nor the retained rows' actual code bits.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
from pathlib import Path
import time

import torch
import r7_legacy_bridge_v1 as bridge

SCHEMA = 'trajkv.round13.fp16_exception.v1'
RUNTIME_FILES = ('round13_exception_bank_v1.py', 'round13_exception_attention_v1.py')


def require(value, message):
    if not value:
        raise ValueError('ROUND13_EXCEPTION: ' + message)


def unique_capacity(owners):
    unique = {}
    for owner in owners:
        for value in owner.named_tensors().values():
            if value.numel():
                unique.setdefault(bridge._storage(value), value.untyped_storage().nbytes())
    return sum(unique.values())


def r_stage_count(n):
    """Frozen R allocator, including its four-byte Round12 descriptor."""
    return min(224, (1670 * int(n) + 65408) // (18 * int(n)))


def r_budget(n):
    n = int(n)
    return (3816 + 18 * r_stage_count(n)) * n + 89737


def expected_owner_bytes(n, h, stages=43):
    """Actual compact layout: full original positions, bitmask/rank/high map.

    h=0/stages43 uses the unmodified R7 owner, with no unused new descriptor.
    V codes occupy ceil((N-h)/32) real uint32 words per channel. The bitmap
    and its prefix counts use one int32 each per original 32-row block.
    """
    n, h, stages = int(n), int(h), int(stages)
    require(n > 0 and n % 32 == 0 and 0 <= h <= n and 0 <= stages <= 43,
            'invalid complete-word input, exception count, or stage count')
    if h == 0 and stages == 43:
        return 4590 * n + 89733
    m = n - h
    padding = (-m) % 32
    routing = 8 * ((n + 31) // 32) + 4 * h + 4
    base = (3816 + 18 * stages) * n + 89733
    removable_per_row = 3808 + 18 * stages
    return base + h * (57344 - removable_per_row) + 1792 * padding + routing


def budget_plan(n, stages=43, *, requested_h=None, cap_bytes=None,
                technical_allow_over_budget=False):
    """Exhaustive integer search handles the sawtooth V padding price."""
    n, stages = int(n), int(stages)
    cap = r_budget(n) if cap_bytes is None else int(cap_bytes)
    legal = [h for h in range(n + 1) if expected_owner_bytes(n, h, stages) <= cap]
    require(legal, 'base format itself exceeds the original R budget')
    h = max(legal) if requested_h is None else int(requested_h)
    require(0 <= h <= n, 'requested_h out of range')
    actual = expected_owner_bytes(n, h, stages)
    require(actual <= cap or technical_allow_over_budget,
            f'requested h={h} costs {actual}, cap={cap}')
    return {'visual_tokens': n, 'ordinary_rows': n-h, 'exception_rows': h,
            'base_stage_count': stages, 'original_R_stage_count': r_stage_count(n),
            'original_R_complete_owner_bytes': cap, 'expected_complete_owner_bytes': actual,
            'remaining_budget_bytes': cap-actual, 'V_padding_rows': (-(n-h)) % 32,
            'routing_bytes': 0 if h == 0 and stages == 43 else 8*((n+31)//32)+4*h+4,
            'effective_bits_per_visual_element': actual/(3584*n),
            'technical_allow_over_budget': bool(technical_allow_over_budget)}


def _unpack_v1(payload, n):
    shift = torch.arange(32, dtype=torch.int64, device=payload.device)
    return ((payload.long()[..., None] >> shift) & 1).flatten(-2)[..., :n]


def _pack_v1(bits):
    m = int(bits.shape[-1])
    padded = torch.zeros((*bits.shape[:-1], ((m + 31)//32)*32),
                         device=bits.device, dtype=torch.int64)
    padded[..., :m] = bits
    shift = torch.arange(32, dtype=torch.int64, device=bits.device)
    return (padded.reshape(*padded.shape[:-1], -1, 32) << shift).sum(-1).int().contiguous()


def _stage_subset(original, stages):
    require(stages in (0, 41, 43), 'supported explicit donor profiles: 0,41,43')
    if stages == 43:
        return deepcopy(original)
    if stages == 0:
        return []
    # This is an explicit donor profile, never an implicit budget repair.
    k = [x for x in original if x[0] == 'k'][:21]
    selected = [list(x) for x in original if x[0] == 'v' or x in k]
    require(len(selected) == 41 and sum(x[0] == 'v' for x in selected) == 20,
            'm41 must explicitly preserve V20 and first-ranked K21')
    return selected


def compact_fields(state, original, selected, layer, ordinary_rows):
    """Physical copies, also callable by CPU technical checks without a model."""
    n = int(state.visual_pos.numel())
    ordinary_rows = ordinary_rows.to(device=state.visual_pos.device, dtype=torch.long)
    for name in ('direction_code', 'direction_gain'):
        setattr(state, name, getattr(state, name).index_select(2, ordinary_rows).contiguous())
    vbits = _unpack_v1(state.v_code, n).index_select(-1, ordinary_rows)
    state.v_code = _pack_v1(vbits)
    old_entries = [list(x) for x in original if int(x[1]) == int(layer)]
    new_entries = [list(x) for x in selected if int(x[1]) == int(layer)]
    if new_entries:
        slots = torch.tensor([old_entries.index(x) for x in new_entries],
                             device=ordinary_rows.device, dtype=torch.long)
        state.depth_residual_code = state.depth_residual_code.index_select(0, slots).index_select(1, ordinary_rows).contiguous()
        state.depth_residual_alpha = state.depth_residual_alpha.index_select(0, slots).index_select(1, ordinary_rows).contiguous()
    else:
        for name in ('depth_residual_code', 'depth_residual_alpha'):
            if hasattr(state, name):
                delattr(state, name)
    return new_entries


class Round13ExceptionOwner(bridge.R7LegacyOwner):
    def named_tensors(self, include_constants=True):
        result = super().named_tensors(include_constants)
        result['visual_exception.native_post_rope_k'] = self.state.round13_native_k
        result['visual_exception.native_v'] = self.state.round13_native_v
        return result

    owner_tensors = named_tensors


@torch.no_grad()
def build_round13_exception_owners(native_prefix_cache, visual_positions, variant, *,
        candidate, selected_indices=None, legacy_a2, legacy_experiments,
        legacy_scripts, kivi_source, expected_asset_sha256=None,
        expected_legacy_source_sha256=None):
    """Return actual owners and JSON proof; selector lives in a separate module.

    candidate.base_stages defaults43. 41 explicitly donates two K stages while
    keeping all V20; 0 explicitly donates all43. Formal h is the maximum legal
    integer under that image's original R byte cap. A smaller requested_h is
    allowed as an explicit dose/control. Over-budget is technical-only.
    """
    started = time.perf_counter()
    n = int(visual_positions.numel())
    stages = int(candidate.get('base_stages', 43))
    require(stages in (0, 41, 43), 'supported explicit donor profiles: 0,41,43')
    technical = bool(candidate.get('technical_only', False))
    requested_h = candidate.get('requested_h')
    if n <= int(candidate.get('small_n_fallback_max_tokens', 0)):
        require(stages == 43, 'small-N identity fallback requires original low43 donor profile')
        requested_h = 0
    plan = budget_plan(n, stages, requested_h=requested_h,
        technical_allow_over_budget=technical and bool(candidate.get('allow_over_budget', False)))
    h, m = plan['exception_rows'], plan['ordinary_rows']
    device = visual_positions.device
    if selected_indices is None:
        require(h == 0, 'question-free selector must supply exact exception row set')
        high = torch.empty(0, device=device, dtype=torch.long)
    else:
        high = torch.as_tensor(selected_indices, device=device, dtype=torch.long)
        require(high.ndim == 1 and high.numel() == h and torch.equal(high, high.unique(sorted=True))
                and (h == 0 or (int(high[0]) >= 0 and int(high[-1]) < n)),
                'selected_indices must be sorted unique original visual row indices of exact h')
    owners0, metadata = bridge.build_r7_owners(native_prefix_cache, visual_positions, variant,
        legacy_a2=legacy_a2, legacy_experiments=legacy_experiments,
        legacy_scripts=legacy_scripts, kivi_source=kivi_source,
        expected_asset_sha256=expected_asset_sha256,
        expected_legacy_source_sha256=expected_legacy_source_sha256)
    original = [list(x) for x in metadata['selected_stages']]
    require(len(original) == 43 and sum(x[0] == 'k' for x in original) == 23
            and sum(x[0] == 'v' for x in original) == 20, 'low43 lineage K23/V20')
    require(unique_capacity(owners0) == expected_owner_bytes(n, 0, 43), 'frozen low actual-byte identity')
    if h == 0 and stages == 43:
        metadata['round13_exception'] = dict(plan, schema=SCHEMA, h0_identity=True,
            original_reader_retained=True, selection=candidate.get('selection_name'))
        return owners0, metadata
    shared = owners0[0].shared
    keep_mask = torch.ones(n, device=device, dtype=torch.bool)
    keep_mask[high] = False
    ordinary = keep_mask.nonzero(as_tuple=False).flatten()
    require(ordinary.numel() == m, 'disjoint complete row partition')
    high_flags = (~keep_mask).long().reshape(-1, 32)
    shifts = torch.arange(32, device=device, dtype=torch.int64)
    words = (high_flags << shifts).sum(-1).int().contiguous()
    counts = high_flags.sum(-1)
    prefix = (counts.cumsum(0) - counts).int().contiguous()
    shared['round13_exception_mask'] = words
    shared['round13_exception_prefix'] = prefix
    shared['round13_exception_rows'] = high.int().contiguous()
    shared['round13_exception_format'] = torch.tensor([1, stages, 1, 0], device=device, dtype=torch.uint8)
    selected = _stage_subset(original, stages)
    lookup = shared['global_residual_lookup']
    lookup.fill_(-1)
    result, native_equality = [], []
    with bridge._legacy_paths(Path(legacy_a2), Path(legacy_experiments), Path(legacy_scripts)):
        reader = importlib.import_module('round13_exception_attention_v1').attention
        for layer, old in enumerate(owners0):
            state = old.state
            entries = compact_fields(state, original, selected, layer, ordinary)
            for slot, (axis, _, head, depth) in enumerate(entries):
                lookup[layer, int(axis == 'v'), int(head), int(depth)] = slot
            positions = visual_positions.index_select(0, high)
            native_k = native_prefix_cache.layers[layer].keys.index_select(2, positions).detach().contiguous()
            native_v = native_prefix_cache.layers[layer].values.index_select(2, positions).detach().contiguous()
            require(native_k.dtype == native_v.dtype == torch.float16, 'native FP16 fields')
            state.round13_native_k, state.round13_native_v = native_k, native_v
            state.round13_ordinary_rows = m
            state.round13_exception_mask = words
            state.round13_exception_prefix = prefix
            state.round13_exception_rows = shared['round13_exception_rows']
            state.depth_residual_lookup = lookup[layer]
            require(torch.equal(native_k.view(torch.int16), native_prefix_cache.layers[layer].keys.index_select(2, positions).view(torch.int16))
                    and torch.equal(native_v.view(torch.int16), native_prefix_cache.layers[layer].values.index_select(2, positions).view(torch.int16)),
                    'native exception bit identity')
            native_equality.append(True)
            result.append(Round13ExceptionOwner(layer, state, shared, reader, old.boundary))
    owners0.clear()
    actual = unique_capacity(result)
    require(actual == plan['expected_complete_owner_bytes'],
            f'actual complete owner {actual} differs from packed formula {plan}')
    forbidden = {bridge._storage(t) for layer in native_prefix_cache.layers for t in (layer.keys, layer.values)}
    extras = {'round13_native_k', 'round13_native_v', 'round13_exception_mask',
              'round13_exception_prefix', 'round13_exception_rows'}
    allowed = set(bridge._STATE_TENSORS) | extras
    for owner in result:
        require(all(not isinstance(t, torch.Tensor) or not t.numel() or key in allowed
                    for key, t in vars(owner.state).items()), 'unreported persistent state tensor')
        for name, tensor in owner.named_tensors().items():
            require(bridge._storage(tensor) not in forbidden and not tensor.requires_grad
                    and tensor.grad_fn is None, 'source alias or autograd retained: '+name)
    metadata.update(selected_stages=selected, selected_stage_count=len(selected),
        returned_owners_unique_tensor_capacity_bytes=actual, reader_unchanged=False,
        original_FP16_visual_storage_retained_by_owners=False)
    metadata['round13_exception'] = dict(plan, schema=SCHEMA, h0_identity=False,
        original_reader_retained=False, selection=candidate.get('selection_name'),
        selected_original_visual_rows=high.tolist(), ordinary_count=m,
        native_field_bit_identity_all_layers=all(native_equality),
        native_exception_pool_bytes=57344*h,
        routing_persistent_bytes=plan['routing_bytes'],
        ordinary_V_padding_capacity_bytes=1792*((-m)%32),
        native_pool_scope='selected_visual_exception_rows_only_not_text',
        ordinary_metadata_scope='unchanged_full_original_visual_token_population',
        ordinary_rows_physically_removed_from_all_low_pools=True,
        V_padding_in_actual_owner=True, native_post_RoPE_K=True, native_f0_applied=False,
        discarded_stages=[x for x in original if x not in selected],
        runtime_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in RUNTIME_FILES},
        encoding_seconds_including_original_lineage_and_compaction=time.perf_counter()-started,
        CUDA_equivalence_established_by_factory=False)
    return result, metadata
