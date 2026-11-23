"""CPU/GPU-common real FP16 sign-depth codec and complete-owner budgeting.

Stages are zero-indexed: stage 1 is legal only after stage 0 for the same
axis/layer/head. Nothing here pads a residual pool or rounds a budget upward.
"""
import math
import struct

import torch

from quantizers import pack1


def decode_sign_stage(base, code, alpha):
    """Decode one packed sign stage, with the reader's FP16 rounding point."""
    dimension = base.shape[-1]
    shifts = torch.arange(32, device=code.device, dtype=torch.int64)
    sign = (((code.long()[..., None] >> shifts) & 1).reshape(*code.shape[:-1], dimension).float() * 2 - 1)
    return (base.float() + sign * alpha.float()[..., None]).half()


def encode_sign_depth(source, base, depth):
    """Return true integer codes and alpha=MAD(error)/token/128 dims.

    Critically, the next error is against the previous *decoded FP16* value,
    rather than a hypothetical unrounded sum or the original first error.
    """
    if depth not in (0, 1, 2) or source.shape != base.shape or base.shape[-1] != 128:
        raise ValueError('D128 equal-shaped tensors and depth 0/1/2 required')
    if base.dtype != torch.float16:
        raise ValueError('The deployed base reconstruction must be FP16')
    decoded = base
    codes, alphas, distortions = [], [], []
    distortions.append(float((source.float() - decoded.float()).square().sum()))
    for _ in range(depth):
        error = source.float() - decoded.float()
        code, alpha = pack1(error >= 0), error.abs().mean(-1).half().contiguous()
        decoded = decode_sign_stage(decoded, code, alpha)
        codes.append(code)
        alphas.append(alpha)
        distortions.append(float((source.float() - decoded.float()).square().sum()))
    return codes, alphas, decoded, distortions


def validate_units(units, layers, heads):
    result = []
    for item in units:
        if len(item) != 3:
            raise ValueError('selected_units require [axis,layer,head] triples')
        axis, layer, head = str(item[0]).lower(), int(item[1]), int(item[2])
        if axis not in ('k', 'v') or not 0 <= layer < layers or not 0 <= head < heads:
            raise ValueError('Invalid depth priority unit')
        result.append((axis, layer, head))
    if len(set(result)) != len(result):
        raise ValueError('Duplicate depth priority units')
    return result


def validate_stages(stages, layers, heads):
    result, seen = [], set()
    for item in stages:
        if len(item) != 4:
            raise ValueError('selected_stages require [axis,layer,head,stage] quadruples')
        axis, layer, head = validate_units([item[:3]], layers, heads)[0]
        stage = int(item[3])
        entry = (axis, layer, head, stage)
        if stage not in (0, 1) or entry in seen:
            raise ValueError('Stage must be unique and zero-indexed 0/1')
        if stage == 1 and (axis, layer, head, 0) not in seen:
            raise ValueError('A second stage must follow its first stage in the priority order')
        seen.add(entry)
        result.append(entry)
    return result


def _pack_ids(entries, layers, heads, stage_ids):
    ids = []
    for item in entries:
        axis, layer, head = item[:3]
        value = (int(axis == 'v') * layers + layer) * heads + head
        ids.append(value * 2 + item[3] if stage_ids else value)
    if any(value > 65535 for value in ids):
        raise ValueError('Depth ranking IDs exceed uint16')
    return struct.pack('<' + 'H' * len(ids), *ids)


def plan_depth_stages(*, layers, heads, visual_tokens, visual_full_bytes,
                      base_bytes, factor_bytes, budget_bits=1.8,
                      selected_units=(), selected_stages=None,
                      depth_policy='incumbent', v_mixed_pair_fraction=.5,
                      stage_limit=None):
    """Choose stages only after lookup, actual ranking and policy constants.

    base_bytes excludes lookup/ranking/factors. For named policies the fixed K
    set is exactly the old MixedResidualBank affordable priority prefix's K
    subset, with its original full 2*L*H uint16 ranking charge. The *new* owner
    charges only actual ranking entries and its [L,2,H,2] int32 lookup. Each
    selected stage then costs 18*N bytes. No dummy stage or cold rank padding.
    """
    if visual_tokens <= 0 or layers <= 0 or heads <= 0:
        raise ValueError('Positive visual/model dimensions required')
    if not math.isfinite(budget_bits) or budget_bits <= 0:
        raise ValueError('Positive finite budget_bits required')
    explicit = selected_stages is not None
    if explicit:
        stages = validate_stages(selected_stages, layers, heads)
        ranking = _pack_ids(stages, layers, heads, True)
        policy, policy_constant = 'explicit', struct.pack('<Bf', 3, 0.)
        units = []
    else:
        units = validate_units(selected_units, layers, heads)
        if depth_policy not in ('incumbent', 'v_pair', 'v_mixed'):
            raise ValueError('depth_policy must be incumbent/v_pair/v_mixed')
        policy = depth_policy
        ranking = _pack_ids(units, layers, heads, False)
        if not math.isfinite(v_mixed_pair_fraction) or not 0 <= v_mixed_pair_fraction <= 1:
            raise ValueError('v_mixed_pair_fraction must be in [0,1]')
        # The actual five-byte descriptor always includes a float32 fraction:
        # zero for width-only, one for pairs, configured fraction for mixed.
        fraction = {'incumbent': 0., 'v_pair': 1., 'v_mixed': v_mixed_pair_fraction}[policy]
        policy_constant = struct.pack('<Bf', {'incumbent': 0, 'v_pair': 1, 'v_mixed': 2}[policy], fraction)
        v_mixed_pair_fraction = struct.unpack('<f', policy_constant[1:])[0]
    allowed = math.floor(budget_bits * visual_full_bytes / 16)
    unit_bytes = visual_tokens * 18
    lookup_bytes = layers * 2 * heads * 2 * 4
    fixed_bytes = base_bytes + factor_bytes + lookup_bytes + len(ranking) + len(policy_constant)
    if fixed_bytes > allowed:
        raise RuntimeError('Depth base plus complete lookup/ranking/constants exceeds budget')
    affordable = (allowed - fixed_bytes) // unit_bytes
    if stage_limit is not None:
        affordable = min(affordable, max(0, int(stage_limit)))
    incumbent_units, incumbent_cap = None, None
    if explicit:
        chosen = stages[:affordable]
    else:
        old_fixed = base_bytes + factor_bytes + layers * 2 * heads * 4 + 2 * (2 * layers * heads)
        incumbent_cap = min(len(units), max(0, (allowed - old_fixed) // unit_bytes))
        incumbent_units = units[:incumbent_cap]
        fixed_k = [unit for unit in incumbent_units if unit[0] == 'k']
        if len(fixed_k) > affordable:
            raise RuntimeError('Complete depth owner cannot preserve the incumbent K subset under this stage limit')
        candidates_v = [unit for unit in units if unit[0] == 'v']
        # Named comparisons all select the same number of stages. The width
        # control has only one stage per V candidate, which bounds all arms.
        v_budget = min(affordable - len(fixed_k), len(candidates_v))
        chosen = [(*unit, 0) for unit in fixed_k]
        if policy == 'incumbent':
            chosen.extend((*unit, 0) for unit in candidates_v[:v_budget])
        elif policy == 'v_pair':
            for unit in candidates_v[:v_budget // 2]:
                chosen.extend([(*unit, 0), (*unit, 1)])
            if v_budget % 2:
                chosen.append((*candidates_v[v_budget // 2], 0))
        else:
            # Fraction of V stage slots devoted to paired refinement; all
            # remaining slots widen coverage before any unavailable pair.
            pairs = min(len(candidates_v), int(v_budget * v_mixed_pair_fraction) // 2)
            width = min(len(candidates_v), v_budget - pairs)
            pairs = min(width, v_budget - width)
            chosen.extend((*unit, 0) for unit in candidates_v[:width])
            chosen.extend((*unit, 1) for unit in candidates_v[:pairs])
        validate_stages(chosen, layers, heads)
    predicted_bytes = fixed_bytes + len(chosen) * unit_bytes
    return dict(selected_stages=chosen, requested_stages=stages if explicit else None,
                requested_units=units if not explicit else None,
                incumbent_selected_units=incumbent_units, incumbent_stage_cap=incumbent_cap,
                depth_policy=policy, ranking_constant=ranking, policy_constant=policy_constant,
                lookup_bytes=lookup_bytes, stage_bytes=unit_bytes, allowed_bytes=allowed,
                fixed_bytes=fixed_bytes, affordable_stages=affordable,
                predicted_bytes=predicted_bytes,
                actual_budget_bits=16 * predicted_bytes / visual_full_bytes)
