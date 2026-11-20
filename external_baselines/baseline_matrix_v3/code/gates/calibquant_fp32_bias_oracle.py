"""Independent FP64 Eq6 + declared-rounding oracle for FP32-bias port."""
from __future__ import annotations
import math
import torch
from methods.calibquant_adapter import CalibQuantState
from calibquant_dense_oracle import _codes, _source_text_qk, tensor_stats


def source_dense_attention(cache, layer_index, q, attention_mask, scaling=None):
    layer = cache.layers[layer_index]
    owner = layer.visual_owner
    groups = q.shape[1] // owner.k_packed.shape[1]
    repeat = lambda x: x.repeat_interleave(groups, dim=1)
    kcodes, vcodes = repeat(_codes(owner.k_packed, owner.bits)), repeat(_codes(owner.v_packed, owner.bits))
    ks, kz, vs, vz = map(repeat, (owner.k_scale, owner.k_zero, owner.v_scale, owner.v_zero))
    denominator = math.sqrt(q.shape[-1])
    scale = q.shape[-1] ** -.5 if scaling is None else scaling
    assert abs(scale - 1 / denominator) < 1e-12
    products = ks * kz * q
    assert bool(torch.isfinite(products).all()), "Scope exceeded: bias products themselves nonfinite; no precision extension authorized"
    # Independent declared-rounding reference: FP64 reductions, with original
    # FP16 multiplication/kernel output boundaries and final FP16 bias casting.
    raw = ((q * ks).double() @ kcodes.double().transpose(-1, -2)).to(q.dtype)
    bias64 = products.double().sum(-1, keepdim=True) / denominator
    bias = bias64.to(q.dtype)
    visual = raw / denominator - bias
    uncalibrated = visual.clone()
    raw_width = visual.amax(-1, keepdim=True) - visual.amin(-1, keepdim=True)
    if owner.bits <= 2:
        minimum, maximum = visual.amin(-1, keepdim=True), visual.amax(-1, keepdim=True)
        visual = (visual - minimum) / (maximum - minimum + 1e-8) * (maximum - 3 - minimum) + minimum
    # Independent ideal FP64 Eq6 checked in both algebraic forms. This is a
    # numerical diagnostic, not a second benchmark method or owner claim.
    ideal = ((q.double() * ks.double()) @ kcodes.double().transpose(-1, -2)
             - (q.double() * ks.double() * kz.double()).sum(-1, keepdim=True)) / denominator
    direct = q.double() @ (ks.double() * (kcodes.double() - kz.double())).transpose(-1, -2) / denominator
    torch.testing.assert_close(ideal, direct, atol=1e-8, rtol=1e-8)
    actual_visual = owner.read(q, calibrate=False, softmax_scale=scale)
    torch.testing.assert_close(actual_visual, uncalibrated, atol=.0078125, rtol=.005)
    published_visual = CalibQuantState.read(owner, q, softmax_scale=scale)
    finite_published = torch.isfinite(published_visual).all(-1, keepdim=True)
    selected = finite_published.expand_as(visual)
    finite_source_error = float((visual[selected].float() - published_visual[selected].float()).abs().max()) if bool(selected.any()) else None
    text = _source_text_qk(q, repeat(layer.keys))
    logits = torch.empty(q.shape[0], q.shape[1], q.shape[2], layer.logical_length, device=q.device, dtype=torch.float32)
    logits[..., layer.visual_positions] = visual.float()
    logits[..., layer.text_positions] = text.float()
    if attention_mask is None:
        mask = torch.arange(layer.logical_length, device=q.device)[None, :] <= layer.query_positions[:, None]
        logits.masked_fill_(~mask[None, None], -torch.inf)
    elif attention_mask.dtype == torch.bool:
        logits.masked_fill_(~attention_mask, -torch.inf)
    else:
        logits = logits + attention_mask
    probabilities = torch.softmax(logits, -1).to(q.dtype)
    w = probabilities.index_select(-1, layer.visual_positions)
    value_codes = (w.double() @ vcodes.double()).to(q.dtype)
    visual_output = value_codes * vs - vs * vz * w.sum(-1, keepdim=True)
    output = visual_output + probabilities.index_select(-1, layer.text_positions) @ repeat(layer.values)
    return output.transpose(1, 2).contiguous(), {
        "oracle": "independent_FP64_Eq6_two_forms_and_declared_FP16_boundaries_FP32_bias_sum_div",
        "q_stats": tensor_stats(q), "bias_products_stats": tensor_stats(products), "bias_after_division_stats": tensor_stats(bias),
        "ideal_Eq6_two_form_max_abs_error": float((ideal - direct).abs().max()),
        "ported_uncalibrated_vs_ideal_Eq6_max_abs_error": float((actual_visual.double() - ideal).abs().max()),
        "ported_uncalibrated_vs_ideal_Eq6_relative_l2": float(torch.linalg.vector_norm(actual_visual.double() - ideal) / torch.linalg.vector_norm(ideal).clamp_min(1e-12)),
        "published_FP16_nonfinite_head_queries": int((~finite_published).sum()),
        "published_FP16_finite_interval_calibrated_max_abs_difference": finite_source_error,
        "minimum_visual_logit_range": float(raw_width.min()) if bool(torch.isfinite(raw_width).all()) else None,
        "visual_queries_with_range_below_three": int((raw_width < 3).sum()),
        "visual_queries_with_zero_range": int((raw_width == 0).sum()),
        "expected_output_stats": tensor_stats(output)}
