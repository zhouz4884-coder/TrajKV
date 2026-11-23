"""Bounded W_O-geometry residual search, with the frozen Round11 reader.

SpatialWO retains the original R7 base. JointSpatialWO accepts the frozen
MSE-JointV proposal only under full H, performs two full-H base/sign sweeps,
then freezes that final base before spatial search. Both retain the original
R7 full-H floor. This is neither task loss nor unweighted-MSE nonincrease.
"""
from __future__ import annotations

from copy import copy
import hashlib
import importlib
import json
from pathlib import Path
import time

import torch
import round11_joint_value_v2 as numerical
import round11_value_bank_v2 as base
from attention_bridge import language_attention_modules

bridge = base.bridge
MODES = ("SpatialWO", "JointSpatialWO")
SCHEMA = "trajkv.round11.wo_candidate.v1"
TILE_TOKENS = 128
CANDIDATE_CHUNK = 32
SIGN_SWEEPS = 2
PINS = {
    "r7_legacy_bridge_v1.py": "b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b",
    "round11_joint_value_v2.py": "b819a540f9a07dba153513b45a248f7b56cbe3b46f0115b6e2c7671260f5ea82",
    "round11_value_bank_v2.py": "f65273b2f69c39e6ca9271a62f337b3e57b808d0f9e59325c6ef055456985e16",
    "attention_bridge.py": "eef35d19dfc04fab12f0b6f9ba618a8b34ed48f4488edd566ea4578fb3311876",
    "round11_attention_v1.py": "3290a1b372a8639f702e9ac3a9e183b06147c0b73ac402a24ef21bb35556119f",
}


def require(condition, message):
    if not condition:
        raise ValueError("ROUND11_WO: " + message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def full_h_loss(source, decoded, metric):
    """Declared FP64 quadratic on the actual Half decoder output."""
    error = source.double() - decoded.double()
    shape = error.shape[:-1]
    flat = error.reshape(-1, error.shape[-1])
    return ((flat @ metric) * flat).sum(-1).reshape(shape)


def decode_predictor(predictor, signs, alpha):
    return (predictor.float() + (signs.float() * 2 - 1) * alpha.float()[..., None]).half()


def metric_checks(metric, dimension):
    require(metric.dtype == torch.float64 and tuple(metric.shape) == (dimension, dimension), "FP64 full metric[D,D]")
    require(not metric.requires_grad and metric.grad_fn is None, "metric must not retain autograd")
    require(bool(torch.isfinite(metric).all()), "nonfinite metric")
    require(torch.equal(metric, metric.T), "canonical metric must be symmetric")
    diagonal = metric.diagonal()
    require(bool((diagonal >= 0).all()), "Gram diagonal must be nonnegative")
    total = metric.square().sum()
    offdiagonal = (total - diagonal.square().sum()).clamp_min(0.)
    fraction = torch.where(total > 0, torch.sqrt(offdiagonal / total.clamp_min(torch.finfo(torch.float64).tiny)), 0.)
    minimum, maximum = float(diagonal.min()), float(diagonal.max())
    return {"trace": float(diagonal.sum()), "diag_min": minimum, "diag_max": maximum,
            "diag_max_over_min": maximum / minimum if minimum > 0 else None,
            "offdiag_Frobenius_over_total_Frobenius": float(fraction),
            "symmetry_max_abs": float((metric - metric.T).abs().max()), "finite": True,
            "PSD_contract": "native factory derives Gram; standalone toy caller supplies PSD H",
            "eigendecomposition_or_ridge_or_eigenvalue_clipping_applied": False}


def conditional_alpha(source, predictor, signs, metric):
    """Exact real conditional minimizer when s'Hs>0; zero direction fallback.

    It proposes amplitudes, not a Half/global optimum. The existing legal
    point is always retained by the subsequent actual-Half acceptance.
    """
    s = signs.double() * 2 - 1
    e = source.double() - predictor.double()
    numerator = ((e @ metric) * s).sum(-1)
    denominator = ((s @ metric) * s).sum(-1)
    positive = denominator > 0
    safe = torch.where(positive, denominator, 1.)
    value = torch.where(positive, (numerator / safe).clamp_min(0.), 0.)
    return value, ~positive


def _amplitude_search(source, predictor, metric, signs, alpha, decoded, loss, proposals):
    """Strict actual-full-H acceptance; vectorized candidates and token tiles."""
    candidates = numerical.half_candidates(proposals, alpha)
    invalid = torch.zeros((), device=source.device, dtype=torch.long)
    result_alpha, result_decoded, result_loss = alpha.clone(), decoded.clone(), loss.clone()
    for token in range(0, source.shape[0], TILE_TOKENS):
        sl = slice(token, token + TILE_TOKENS)
        s = signs[sl, None]
        p = predictor[sl, None]
        for candidate in range(0, candidates.shape[1], CANDIDATE_CHUNK):
            amplitudes = candidates[sl, candidate:candidate + CANDIDATE_CHUNK]
            q = decode_predictor(p, s, amplitudes)
            legal = torch.isfinite(amplitudes) & ~numerical.alpha_mode(amplitudes) & torch.isfinite(q).all(-1)
            invalid += (~legal).sum()
            safe_q = torch.where(torch.isfinite(q), q, 0.)
            values = full_h_loss(source[sl, None], safe_q, metric)
            values = torch.where(legal & torch.isfinite(values), values, float("inf"))
            value, index = values.min(-1)
            row = torch.arange(value.shape[0], device=source.device)
            accept = value < result_loss[sl]
            result_alpha[sl] = torch.where(accept, amplitudes[row, index], result_alpha[sl])
            result_decoded[sl] = torch.where(accept[:, None], q[row, index], result_decoded[sl])
            result_loss[sl] = torch.where(accept, value, result_loss[sl])
    return result_alpha, result_decoded, result_loss, invalid


def actual_half_sign_sweep(source, predictor, metric, signs, alpha, decoded, loss):
    """One fixed cyclic sweep. No host scalar read/synchronization per channel.

    With delta=q'_c-q_c, L'-L=-2 delta (Hr)_c+delta^2 Hcc. The entire sweep
    is finally recomputed with full H and accepted strictly, or rolled back.
    """
    proposed_signs, proposed_decoded = signs.clone(), decoded.clone()
    hr = (source.double() - decoded.double()) @ metric
    flip_counts = torch.zeros(source.shape[0], dtype=torch.long, device=source.device)
    predictor_float, alpha_float = predictor.float(), alpha.float()
    for channel in range(source.shape[-1]):
        sign = proposed_signs[:, channel]
        flipped = (predictor_float[:, channel] - (sign.float() * 2 - 1) * alpha_float).half()
        finite = torch.isfinite(flipped)
        safe = torch.where(finite, flipped, proposed_decoded[:, channel])
        delta = safe.double() - proposed_decoded[:, channel].double()
        change = -2 * delta * hr[:, channel] + delta.square() * metric[channel, channel]
        accept = finite & torch.isfinite(change) & (change < 0)
        step = torch.where(accept, delta, 0.)
        proposed_signs[:, channel] = torch.where(accept, ~sign, sign)
        proposed_decoded[:, channel] = torch.where(accept, flipped, proposed_decoded[:, channel])
        hr -= step[:, None] * metric[channel, :][None]
        flip_counts += accept.long()
    recomputed = full_h_loss(source, proposed_decoded, metric)
    accept = torch.isfinite(recomputed) & (recomputed < loss)
    result_signs = torch.where(accept[:, None], proposed_signs, signs)
    result_decoded = torch.where(accept[:, None], proposed_decoded, decoded)
    result_loss = torch.where(accept, recomputed, loss)
    return result_signs, result_decoded, result_loss, torch.where(accept, flip_counts, 0), (~accept & (flip_counts > 0)).sum()


def actual_half_base_sign_sweep(source, l0, l1, metric, bits, signs, alpha, decoded, loss):
    """Four actual Half levels per channel; all tokens/levels vectorized."""
    new_bits, new_signs, new_q = bits.clone(), signs.clone(), decoded.clone()
    hr = (source.double() - decoded.double()) @ metric
    counts = torch.zeros(source.shape[0], dtype=torch.long, device=source.device)
    level_bits = torch.tensor([False, False, True, True], device=source.device)
    level_signs = torch.tensor([False, True, False, True], device=source.device)
    sign_values = level_signs.float() * 2 - 1
    row = torch.arange(source.shape[0], device=source.device)
    alpha_float, l0_float, l1_float = alpha.float(), l0.float(), l1.float()
    for channel in range(source.shape[-1]):
        endpoints = torch.where(level_bits, l1_float[channel], l0_float[channel])
        levels = (endpoints[None] + alpha_float[:, None] * sign_values[None]).half()
        finite = torch.isfinite(levels)
        safe = torch.where(finite, levels, new_q[:, channel, None])
        delta = safe.double() - new_q[:, channel, None].double()
        changes = -2 * delta * hr[:, channel, None] + delta.square() * metric[channel, channel]
        changes = torch.where(finite & torch.isfinite(changes), changes, torch.inf)
        change, choice = changes.min(-1)
        accept = torch.isfinite(change) & (change < 0)
        step = torch.where(accept, delta[row, choice], 0.)
        new_bits[:, channel] = torch.where(accept, level_bits[choice], new_bits[:, channel])
        new_signs[:, channel] = torch.where(accept, level_signs[choice], new_signs[:, channel])
        new_q[:, channel] = torch.where(accept, levels[row, choice], new_q[:, channel])
        hr -= step[:, None] * metric[channel, :][None]
        counts += accept.long()
    recomputed = full_h_loss(source, new_q, metric)
    accept = torch.isfinite(recomputed) & (recomputed < loss)
    return (torch.where(accept[:, None], new_bits, bits), torch.where(accept[:, None], new_signs, signs),
            torch.where(accept[:, None], new_q, decoded), torch.where(accept, recomputed, loss),
            torch.where(accept, counts, 0), (~accept & (counts > 0)).sum())


@torch.no_grad()
def joint_full_h_encode(source, l0, l1, bits, signs, alpha, metric):
    """Original point floor, guarded MSE proposal, then 2 local H sweeps."""
    numerical._validate_fields(source, bits, signs, alpha, l0, l1)
    require(l0.ndim == l1.ndim == 1, "joint H uses frozen group-all endpoints")
    original_q = decode_predictor(torch.where(bits, l1, l0), signs, alpha)
    require(bool(torch.isfinite(original_q).all()), "original joint parent is nonfinite")
    before = full_h_loss(source, original_q, metric)
    proposal_b, proposal_s, proposal_a, mse_stats = numerical.joint_encode(source, l0, l1, bits, signs, alpha,
        tile_tokens=TILE_TOKENS, candidate_chunk=CANDIDATE_CHUNK)
    proposal_q = decode_predictor(torch.where(proposal_b, l1, l0), proposal_s, proposal_a)
    proposal_loss = full_h_loss(source, proposal_q, metric)
    accept = torch.isfinite(proposal_loss) & (proposal_loss < before)
    current_b = torch.where(accept[:, None], proposal_b, bits)
    current_s = torch.where(accept[:, None], proposal_s, signs)
    current_a = torch.where(accept, proposal_a, alpha)
    current_q = torch.where(accept[:, None], proposal_q, original_q)
    current_loss = torch.where(accept, proposal_loss, before)
    proposal_accepts = accept.sum()
    updates = torch.zeros((), device=source.device, dtype=torch.long)
    rollbacks, invalid, zero_directions = updates.clone(), updates.clone(), updates.clone()
    for _ in range(SIGN_SWEEPS):
        current_b, current_s, current_q, current_loss, count, rollback = actual_half_base_sign_sweep(
            source, l0, l1, metric, current_b, current_s, current_a, current_q, current_loss)
        updates += count.sum()
        rollbacks += rollback
        predictor = torch.where(current_b, l1, l0)
        real, flat = conditional_alpha(source, predictor, current_s, metric)
        zero_directions += flat.sum()
        current_a, current_q, current_loss, bad = _amplitude_search(source, predictor, metric, current_s,
            current_a, current_q, current_loss, real[:, None])
        invalid += bad
    final_q = decode_predictor(torch.where(current_b, l1, l0), current_s, current_a)
    final_loss = full_h_loss(source, final_q, metric)
    accept = torch.isfinite(final_loss) & (final_loss < before)
    current_b = torch.where(accept[:, None], current_b, bits)
    current_s = torch.where(accept[:, None], current_s, signs)
    current_a = torch.where(accept, current_a, alpha)
    final_q = decode_predictor(torch.where(current_b, l1, l0), current_s, current_a)
    final_loss = full_h_loss(source, final_q, metric)
    require(bool((final_loss <= before).all()), "joint full-H original-point floor failed")
    return current_b, current_s, current_a, {
        "fullH_before_original_R7": float(before.sum()), "MSE_proposal_fullH": float(proposal_loss.sum()),
        "MSE_proposal_accepted_tokens": int(proposal_accepts), "fullH_after_joint": float(final_loss.sum()),
        "max_token_fullH_increase_vs_original_R7": float((final_loss - before).max()),
        "base_bits_changed": int((current_b != bits).sum()), "base_sign_coordinate_updates": int(updates),
        "joint_base_sign_sweeps": SIGN_SWEEPS, "sweep_roundoff_rollbacks": int(rollbacks),
        "invalid_candidates": int(invalid), "nonpositive_sign_quadratic_directions": int(zero_directions),
        "MSE_before": float((source.double() - original_q.double()).square().sum()),
        "MSE_after": float((source.double() - final_q.double()).square().sum()),
        "frozen_MSE_JointV_proposal": mse_stats, "MSE_proposal_is_not_unconditionally_written": True,
        "CPU_sync_per_channel": False, "ties": "retain_complete_original_R7_fields",
        "global_joint_optimality_claimed": False, "unweighted_MSE_or_task_nonincrease_claimed": False}


def optimize_predictor(source, predictor, metric, parent_signs, parent_alpha):
    """Current sign/alpha plus MSE seeds, then exactly two full-H sweeps."""
    signs = parent_signs.clone()
    zero = torch.zeros_like(parent_alpha)
    alpha = zero.clone()
    decoded = decode_predictor(predictor, signs, alpha)
    loss = full_h_loss(source, decoded, metric)
    invalid = torch.zeros((), device=source.device, dtype=torch.long)
    zero_directions = torch.zeros_like(invalid)
    # Actual parent amplitude/sign is an explicit proposal; safe zero is the
    # starting point when the same amplitude would overflow a new predictor.
    mse_error = source.double() - predictor.double()
    mad = mse_error.abs().mean(-1)
    diagonal = metric.diagonal()
    diagonal_sum = diagonal.sum()
    safe_sum = torch.where(diagonal_sum > 0, diagonal_sum, 1.)
    for seed_signs in (parent_signs, mse_error >= 0):
        signed_error = mse_error * (seed_signs.double() * 2 - 1)
        diagonal_alpha = torch.where(diagonal_sum > 0,
            (signed_error * diagonal).sum(-1).div(safe_sum).clamp_min(0.), 0.)
        proposals = torch.stack((diagonal_alpha, mad, zero.double(), parent_alpha.double()), -1)
        seed_alpha, seed_q, seed_loss, bad = _amplitude_search(source, predictor, metric, seed_signs,
            alpha, decoded, loss, proposals)
        accept = seed_loss < loss
        signs = torch.where(accept[:, None], seed_signs, signs)
        alpha, decoded, loss = seed_alpha, seed_q, seed_loss
        invalid += bad
    diagonal_seed_loss = loss.clone()
    for seed_signs in (parent_signs, mse_error >= 0):
        real, flat = conditional_alpha(source, predictor, seed_signs, metric)
        zero_directions += flat.sum()
        seed_alpha, seed_q, seed_loss, bad = _amplitude_search(source, predictor, metric, seed_signs,
            alpha, decoded, loss, real[:, None])
        accept = seed_loss < loss
        signs = torch.where(accept[:, None], seed_signs, signs)
        alpha, decoded, loss = seed_alpha, seed_q, seed_loss
        invalid += bad
    full_amplitude_seed_loss = loss.clone()
    flip_counts = torch.zeros(source.shape[0], dtype=torch.long, device=source.device)
    rollbacks = torch.zeros_like(invalid)
    for _ in range(SIGN_SWEEPS):
        signs, decoded, loss, flips, rollback = actual_half_sign_sweep(source, predictor, metric, signs, alpha, decoded, loss)
        flip_counts += flips
        rollbacks += rollback
        real, flat = conditional_alpha(source, predictor, signs, metric)
        zero_directions += flat.sum()
        alpha, decoded, loss, bad = _amplitude_search(source, predictor, metric, signs, alpha, decoded, loss, real[:, None])
        invalid += bad
    # End-to-end strict acceptance is evaluated again by the parent chooser.
    return signs, alpha, decoded, loss, {"invalid_candidates": invalid,
        "nonpositive_sign_quadratic_directions": zero_directions, "accepted_coordinate_flips": flip_counts.sum(),
        "sweep_roundoff_rollbacks": rollbacks, "diagonal_seed_fullH_sum": diagonal_seed_loss.sum(),
        "fullH_amplitude_seed_fullH_sum": full_amplitude_seed_loss.sum(),
        "after_sign_refit_fullH_sum": loss.sum()}


@torch.no_grad()
def encode_full_h_residual(source, l0, l1, bits, parent_signs, parent_alpha, metric):
    """Return sign bits/alpha modes; base bits and metadata are immutable."""
    numerical._validate_fields(source, bits, parent_signs, parent_alpha, l0, l1)
    geometry = metric_checks(metric, source.shape[-1])
    p0 = torch.where(bits, l1, l0)
    p1 = numerical.spatial_predictor(bits, l0, l1, shuffled=False)
    require(bool(torch.isfinite(p0).all() & torch.isfinite(p1).all()), "nonfinite predictor")
    original_q = decode_predictor(p0, parent_signs, parent_alpha)
    require(bool(torch.isfinite(original_q).all()), "current parent Half reconstruction is nonfinite")
    before = full_h_loss(source, original_q, metric)
    best_loss, best_signs, best_alpha = before.clone(), parent_signs.clone(), parent_alpha.clone()
    best_mode = torch.zeros(source.shape[0], device=source.device, dtype=torch.bool)
    diagnostics = []
    for mode, predictor in enumerate((p0, p1)):
        signs, alpha, q, _, detail = optimize_predictor(source, predictor, metric, parent_signs, parent_alpha)
        actual = full_h_loss(source, q, metric)
        accept = torch.isfinite(actual) & (actual < best_loss)
        best_signs = torch.where(accept[:, None], signs, best_signs)
        best_alpha = torch.where(accept, alpha, best_alpha)
        best_mode = torch.where(accept, bool(mode), best_mode)
        best_loss = torch.where(accept, actual, best_loss)
        diagnostics.append({key: float(value) if value.is_floating_point() else int(value) for key, value in detail.items()})
    encoded_alpha = numerical.with_alpha_mode(best_alpha, best_mode)
    final_q = numerical.decode_fields(bits, best_signs, encoded_alpha, l0, l1, spatial=True, shuffled=False)
    final_loss = full_h_loss(source, final_q, metric)
    # Recompute the deployed full-vector objective once more. Numerical ties
    # preserve every parent field, including its original sign bit pattern.
    accept = torch.isfinite(final_loss) & (final_loss < before)
    best_signs = torch.where(accept[:, None], best_signs, parent_signs)
    encoded_alpha = torch.where(accept, encoded_alpha, parent_alpha)
    final_q = numerical.decode_fields(bits, best_signs, encoded_alpha, l0, l1, spatial=True, shuffled=False)
    final_loss = full_h_loss(source, final_q, metric)
    require(bool((final_loss <= before).all()), "recomputed actual full-H objective increased")
    n, d = source.shape
    workspace = 128 * CANDIDATE_CHUNK * d * (2 + 8 + 8) + 8 * n * d * 8 + d * d * 8
    return best_signs, encoded_alpha, {"metric": "actual_Half_output_FP64_full_H_quadratic",
        "fullH_before": float(before.sum()), "fullH_after": float(final_loss.sum()),
        "max_token_fullH_increase": float((final_loss - before).max()),
        "MSE_before": float((source.double() - original_q.double()).square().sum()),
        "MSE_after": float((source.double() - final_q.double()).square().sum()),
        "strictly_improved_tokens": int(accept.sum()), "mode1_tokens": int(numerical.alpha_mode(encoded_alpha).sum()),
        "sign_sweeps": SIGN_SWEEPS, "channel_order": "cyclic0_to_Dminus1_twice",
        "CPU_sync_per_channel": False, "predictor_diagnostics": diagnostics, "metric_geometry": geometry,
        "encoding_workspace_major_tensor_bound_bytes_excluding_BLAS_allocator": workspace,
        "workspace_bound_scope": "conservative named major tensors, not measured allocator peak; W conversion/Gram/source layer separate",
        "ties": "retain_complete_parent_fields", "unweighted_MSE_nonincrease_claimed": False,
        "attention_or_task_nonincrease_claimed": False}


@torch.no_grad()
def derive_o_proj_metric(module, kv_head):
    """FP64 Gram of the existing GQA7 W_O input-column blocks, per V unit."""
    started = time.perf_counter()
    weight = module.o_proj.weight
    require(weight.dtype == torch.float16 and tuple(weight.shape) == (3584, 3584), "native M1 o_proj FP16[3584,3584]")
    require(not weight.requires_grad and int(module.num_key_value_groups) == 7 and int(module.head_dim) == 128, "native frozen M1 GQA7/D128 module")
    require(0 <= kv_head < 4, "M1 KV head")
    identity = (bridge._storage(weight), weight._version)
    metric = torch.zeros((128, 128), dtype=torch.float64, device=weight.device)
    for query_head in range(kv_head * 7, kv_head * 7 + 7):
        block = weight.detach()[:, query_head * 128:(query_head + 1) * 128].double()
        metric.add_(block.T @ block)
        del block
    metric = ((metric + metric.T) * 0.5).contiguous()
    require(identity == (bridge._storage(weight), weight._version), "native o_proj storage/version changed")
    info = {"source": "existing_native_W_O_input_columns_GQA7", "query_head_group": list(range(kv_head * 7, kv_head * 7 + 7)),
        "weight_shape": [3584, 3584], "H_shape": [128, 128], "H_dtype": "float64",
        "H_bytes": 128 * 128 * 8, "maximum_single_FP64_weight_block_bytes": 3584 * 128 * 8,
        "Gram_accumulation_FLOPs": 2 * 7 * 3584 * 128 * 128, "weight_mutated": False,
        "Gram_derived_rows": 7 * 3584, "H_geometry": metric_checks(metric, 128),
        "retained_derived_metric_bytes_after_encoding": 0,
        "host_Gram_seconds_including_synchronizing_geometry_not_controlled_GPU_timing": time.perf_counter() - started,
        "metric_note": "sum individual head projection energies; omits attention weights and cross-head terms, not true layer/task loss"}
    return metric, info


@torch.no_grad()
def transform_wo_state(state, sourceV, selected_stages, layer_idx, *, mode, metric_provider, encode=True):
    """Pure-return layer helper; provider(head)->(transient H, JSON geometry)."""
    require(mode in MODES and type(encode) is bool, "mode/encode")
    require(sourceV.ndim == 4 and sourceV.shape[0] == 1 and sourceV.dtype == torch.float16, "single-image native Half V")
    entries = base._layer_entries(state, selected_stages, layer_idx)
    slots = [(slot, head) for slot, (axis, _, head, _) in enumerate(entries) if axis == "v"]
    result = copy(state)
    result.round11_shuffled = False
    detail = {"layer": layer_idx, "V_slots": len(slots), "head_stats": [],
        "baseline": "original_R7_fullH_floor_guarded_joint_then_frozen_base" if mode == "JointSpatialWO" and encode else "original_R7_current_fields_on_original_base"}
    if not encode or not slots:
        detail.update(changed_base_words=0, changed_sign_words=0, changed_alpha_bits=0)
        return result, detail
    if mode == "JointSpatialWO":
        result.v_code = state.v_code.detach().clone()
    result.depth_residual_code = state.depth_residual_code.detach().clone()
    result.depth_residual_alpha = state.depth_residual_alpha.detach().clone()
    metrics = {}
    # Complete every covered head's base stage before constructing any P1 in
    # this layer. The graph is independent across heads/layers; this is the
    # same frozen B* as a whole-model base pass followed by a spatial pass.
    for slot, head in slots:
        l0, l1 = base.value_endpoints(state, head)
        bits = numerical.unpack_bits(state.v_code[0, head], sourceV.shape[-2]).T.contiguous()
        signs = numerical.unpack_bits(state.depth_residual_code[slot], sourceV.shape[-1])
        alpha = state.depth_residual_alpha[slot]
        metric, geometry = metric_provider(head)
        metric_checks(metric, sourceV.shape[-1])
        joint_stats = None
        if mode == "JointSpatialWO":
            bits, signs, alpha, joint_stats = joint_full_h_encode(sourceV[0, head], l0, l1, bits, signs, alpha, metric)
            result.v_code[0, head].copy_(numerical.pack_bits(bits.T.contiguous()))
            result.depth_residual_code[slot].copy_(numerical.pack_bits(signs))
            result.depth_residual_alpha[slot].copy_(alpha)
        metrics[head] = (metric, geometry, joint_stats)
        del metric, bits, signs, alpha
    frozen_base = result.v_code
    frozen_base_version = result.v_code._version
    for slot, head in slots:
        l0, l1 = base.value_endpoints(result, head)
        bits = numerical.unpack_bits(result.v_code[0, head], sourceV.shape[-2]).T.contiguous()
        signs = numerical.unpack_bits(result.depth_residual_code[slot], sourceV.shape[-1])
        alpha = result.depth_residual_alpha[slot]
        metric, geometry, joint_stats = metrics.pop(head)
        new_signs, new_alpha, stats = encode_full_h_residual(sourceV[0, head], l0, l1, bits, signs, alpha, metric)
        result.depth_residual_code[slot].copy_(numerical.pack_bits(new_signs))
        result.depth_residual_alpha[slot].copy_(new_alpha)
        item = {"head": head, "slot": slot, "metric_derivation": geometry, "WO": stats}
        if joint_stats is not None:
            item["joint_fullH"] = joint_stats
        item["fullH_before_original_R7"] = joint_stats["fullH_before_original_R7"] if joint_stats else stats["fullH_before"]
        item["fullH_after_joint"] = joint_stats["fullH_after_joint"] if joint_stats else stats["fullH_before"]
        item["fullH_after_spatial"] = stats["fullH_after"]
        detail["head_stats"].append(item)
        del metric, new_signs, new_alpha, bits, signs, alpha
    for name in ("direction_code", "direction_gain", "direction_mean", "depth_residual_lookup", "visual_pos", "v_scale", "v_min"):
        require(getattr(result, name) is getattr(state, name), "original field pointer changed: " + name)
    covered = {head for _, head in slots}
    for head in range(sourceV.shape[1]):
        if head not in covered:
            require(torch.equal(result.v_code[0, head], state.v_code[0, head]), "uncovered V base changed")
    for slot, (axis, _, _, _) in enumerate(entries):
        if axis == "k":
            require(torch.equal(result.depth_residual_code[slot], state.depth_residual_code[slot]) and
                    torch.equal(result.depth_residual_alpha[slot].view(torch.int16), state.depth_residual_alpha[slot].view(torch.int16)), "K residual changed")
    require(not metrics and result.v_code is frozen_base and result.v_code._version == frozen_base_version, "spatial stage changed its frozen final base")
    detail["frozen_base_unchanged_during_spatial"] = True
    detail["maximum_layer_transient_H_bytes"] = len(slots) * sourceV.shape[-1] ** 2 * 8
    detail.update(changed_base_words=int((result.v_code != state.v_code).sum()),
        changed_sign_words=int((result.depth_residual_code != state.depth_residual_code).sum()),
        changed_alpha_bits=int((result.depth_residual_alpha.view(torch.int16) != state.depth_residual_alpha.view(torch.int16)).sum()))
    json.dumps(detail, allow_nan=False)  # no derived tensor can hide in metadata
    return result, detail


@torch.no_grad()
def build_round11_wo_owners(native_prefix_cache, visual_positions, variant, *, model, mode,
        encode=True, legacy_a2, legacy_experiments, legacy_scripts, kivi_source,
        expected_asset_sha256=None, expected_legacy_source_sha256=None):
    require(mode in MODES and type(encode) is bool and model is not None, "explicit model/mode/encode")
    for name, digest in PINS.items():
        require(sha(Path(__file__).with_name(name)) == digest, "frozen source changed: " + name)
    require(expected_asset_sha256 is None or dict(expected_asset_sha256) == bridge._DEFAULT_ASSETS, "original low book/f0/policy only")
    owners, metadata = bridge.build_r7_owners(native_prefix_cache, visual_positions, variant,
        legacy_a2=legacy_a2, legacy_experiments=legacy_experiments, legacy_scripts=legacy_scripts,
        kivi_source=kivi_source, expected_asset_sha256=expected_asset_sha256,
        expected_legacy_source_sha256=expected_legacy_source_sha256)
    selected = metadata["selected_stages"]
    require(len(owners) == 28 and len(selected) == metadata["selected_stage_count"] == 43 and
        sum(item[0] == "k" for item in selected) == 23 and sum(item[0] == "v" for item in selected) == 20 and
        all(int(item[3]) == 0 for item in selected), "original K23/V20 stage0 coverage")
    require(variant.get("name") == "r7_joint_k1_gt_cond" and variant.get("v_group_size", "all") == "all", "original R7low/group-all")
    expected = 4590 * visual_positions.numel() + 89733
    before = base.unique_capacity(owners)
    require(before == expected, "original complete B0")
    shared = owners[0].shared
    shared_identity = dict(shared)
    initial_names = [set(owner.named_tensors()) for owner in owners]
    reader = importlib.import_module("round11_attention_v1").attention
    modules = None
    if encode:
        _, modules = language_attention_modules(model)
        require(len(modules) == 28, "native M1 layers")
    stats = []
    for layer, owner in enumerate(owners):
        started = time.perf_counter()
        source = native_prefix_cache.layers[layer].values.index_select(2, visual_positions).detach().contiguous()
        previous = owner.state
        provider = lambda head: derive_o_proj_metric(modules[layer], head)
        transformed, item = transform_wo_state(previous, source, selected, layer, mode=mode,
            metric_provider=provider, encode=encode)
        owner.state = transformed
        owner._reader = reader
        require(owner.shared is shared and set(owner.named_tensors()) == initial_names[layer], "owner tensor/constant names changed")
        require(all(not isinstance(value, torch.Tensor) or not value.numel() or name in bridge._STATE_TENSORS for name, value in vars(transformed).items()), "unreported persistent tensor")
        for name in ("depth_residual_code", "depth_residual_alpha"):
            require(hasattr(previous, name) == hasattr(transformed, name), "residual field presence changed")
        item["host_seconds_including_synchronizing_diagnostics"] = time.perf_counter() - started
        item["source_layer_workspace_bytes"] = source.numel() * source.element_size()
        stats.append(item)
        del source, previous, transformed, provider
    del modules
    require(set(shared) == set(shared_identity) and all(shared[name] is value for name, value in shared_identity.items()), "original shared constants changed")
    after = base.unique_capacity(owners)
    require(before == after == expected, "complete persistent B0 changed")
    require(all(bridge._sha(path) == value["sha256"] for path, value in metadata["verified_legacy_source_pins"].items()), "legacy source changed")
    require(all(bridge._sha(value["path"]) == value["sha256"] for value in metadata["fixed_assets"].values()), "fixed asset changed")
    forbidden = {bridge._storage(t) for layer in native_prefix_cache.layers for t in (layer.keys, layer.values)}
    for owner in owners:
        for name, tensor in owner.named_tensors().items():
            require(not tensor.requires_grad and tensor.grad_fn is None and bridge._storage(tensor) not in forbidden, "autograd/native source retained: " + name)
    metadata["round11_wo_candidate"] = {"schema": SCHEMA, "mode": mode, "encoding_applied": encode,
        "control_mode0_only": not encode, "fixed_K": True, "original_K23_V20_stage0_coverage": True,
        "old_f0_retained_and_charged": True, "shared_original_constants_unchanged": True,
        "no_new_tensor_fields": True, "complete_B0_formula": "4590*N+89733", "complete_B0_bytes": expected,
        "before_unique_capacity_bytes": before, "after_unique_capacity_bytes": after, "actual_extra_persistent_bytes": 0,
        "reader_is_new_spatial": True, "reader_mode": "spatial", "persistent_derived_metric_tensor_bytes": 0,
        "tile_tokens": TILE_TOKENS, "candidate_chunk": CANDIDATE_CHUNK, "sign_sweeps": SIGN_SWEEPS,
        "source_sha256": sha(__file__), "numerical_source_sha256": sha(__file__), "numerical_source_pins": PINS,
        "native_FP16_source_or_encoding_workspace_retained": False, "same_packed_payload_layout": True,
        "baseline": "original_R7_floor_then_fullH_guarded_joint_final_frozen_base" if mode == "JointSpatialWO" and encode else "original_R7_fields_on_original_frozen_base",
        "joint_base_sign_sweeps": SIGN_SWEEPS if mode == "JointSpatialWO" and encode else 0,
        "spatial_sign_sweeps": SIGN_SWEEPS if encode else 0,
        "full_H_nonincrease_vs_declared_parent_claimed": encode, "full_H_nonincrease_vs_original_R7_claimed": encode,
        "unweighted_MSE_nonincrease_claimed": False, "attention_or_task_nonincrease_claimed": False,
        "metric_is_attention_weighted_or_cross_head_output_loss": False, "quality_or_speed_gain_claimed": False,
        "new_parameters_fit": False, "questions_targets_calibration_consumed_by_metric": False,
        "layer_stats": stats, "CUDA_equivalence_established_by_factory": False}
    metadata["returned_owners_unique_tensor_capacity_bytes"] = after
    metadata["reader_unchanged"] = False
    return owners, metadata
