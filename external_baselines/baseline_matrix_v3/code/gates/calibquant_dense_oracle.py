"""Independent dense-code CUDA oracle for actual CalibQuant layer inputs.

Diagnostic only: temporary expanded codes/FP64 sums are paid in gate peak/time,
never passed off as a compressed owner or benchmark runtime implementation.
"""
from __future__ import annotations
import ast
import math
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
_SOURCE_TEXT_EXPRESSION = None


def tensor_stats(tensor):
    finite = torch.isfinite(tensor)
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "nonfinite": int((~finite).sum()), "nan": int(torch.isnan(tensor).sum()),
            "inf": int(torch.isinf(tensor).sum()),
            "finite_max_abs": float(tensor[finite].abs().max()) if bool(finite.any()) else None}


def _source_text_qk(query_states, key_states_quant_t):
    global _SOURCE_TEXT_EXPRESSION
    if _SOURCE_TEXT_EXPRESSION is None:
        tree = ast.parse((ROOT / "third_party/calibquant/patch_internvl.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "forward_internlm2_calibquant")
        statement = next(node for node in ast.walk(function) if isinstance(node, ast.Assign) and
                         any(isinstance(t, ast.Name) and t.id == "attn_weights_quant_t" for t in node.targets))
        _SOURCE_TEXT_EXPRESSION = compile(ast.Expression(statement.value), "official_calibquant_text_qk", "eval")
    return eval(_SOURCE_TEXT_EXPRESSION, {"torch": torch, "math": math,
                "self": type("SourceHeadDim", (), {"head_dim": query_states.shape[-1]})(),
                "query_states": query_states, "key_states_quant_t": key_states_quant_t})


def _codes(packed, bits):
    # Deliberately independent from adapter.unpack_codes.
    return torch.stack([(packed >> shift) & ((1 << bits) - 1)
                        for shift in range(8 - bits, -1, -bits)], dim=-1).flatten(-2)


def source_dense_attention(cache, layer_index, q, attention_mask, scaling=None):
    layer = cache.layers[layer_index]
    owner = layer.visual_owner
    groups = q.shape[1] // owner.k_packed.shape[1]
    repeat = lambda x: x.repeat_interleave(groups, dim=1)
    kcodes, vcodes = repeat(_codes(owner.k_packed, owner.bits)), repeat(_codes(owner.v_packed, owner.bits))
    ks, kz, vs, vz = map(repeat, (owner.k_scale, owner.k_zero, owner.v_scale, owner.v_zero))
    scale = q.shape[-1] ** -.5 if scaling is None else scaling
    assert abs(scale - q.shape[-1] ** -.5) < 1e-12, "Actual model oracle declares native sqrt(D) scale"
    # Official packed kernels sum float32 products and emit FP16. FP64 provides
    # an independent reduction oracle before reproducing the FP16 boundary.
    query_scaled = q * ks
    raw_wide = query_scaled.double() @ kcodes.double().transpose(-1, -2)
    raw = raw_wide.to(q.dtype)
    bias_terms = ks * kz * q
    bias = bias_terms.sum(-1, keepdim=True)
    visual = raw / math.sqrt(q.shape[-1]) - bias / math.sqrt(q.shape[-1])
    uncalibrated = tensor_stats(visual)
    raw_width = visual.amax(-1, keepdim=True) - visual.amin(-1, keepdim=True)
    if owner.bits <= 2:
        minimum, maximum = visual.amin(-1, keepdim=True), visual.amax(-1, keepdim=True)
        visual = (visual - minimum) / (maximum - minimum + 1e-8) * (maximum - 3 - minimum) + minimum
    text = _source_text_qk(q, repeat(layer.keys))
    length = layer.logical_length
    # Preserve ORIGINAL sequence order, unlike the production reader's legal
    # permutation. Comparing these catches composition and mask-axis errors.
    logits = torch.empty(q.shape[0], q.shape[1], q.shape[2], length, device=q.device, dtype=torch.float32)
    logits[..., layer.visual_positions] = visual.float()
    logits[..., layer.text_positions] = text.float()
    if attention_mask is None:
        mask = torch.arange(length, device=q.device)[None, :] <= layer.query_positions[:, None]
        logits.masked_fill_(~mask[None, None], -torch.inf)
    elif attention_mask.dtype == torch.bool:
        logits.masked_fill_(~attention_mask, -torch.inf)
    else:
        logits = logits + attention_mask
    probabilities = torch.softmax(logits, -1).to(q.dtype)
    visual_prob = probabilities.index_select(-1, layer.visual_positions)
    text_prob = probabilities.index_select(-1, layer.text_positions)
    weighted_codes = (visual_prob.double() @ vcodes.double()).to(q.dtype)
    visual_output = weighted_codes * vs - vs * vz * visual_prob.sum(-1, keepdim=True)
    output = visual_output + text_prob @ repeat(layer.values)
    return output.transpose(1, 2).contiguous(), {
        "minimum_visual_logit_range": float(raw_width.min()) if bool(torch.isfinite(raw_width).all()) else None,
        "visual_queries_with_range_below_three": int((raw_width < 3).sum()),
        "visual_queries_with_zero_range": int((raw_width == 0).sum()),
        "visual_queries_with_nonfinite_range": int((~torch.isfinite(raw_width)).sum()),
        "q_stats": tensor_stats(q), "query_scaled_stats": tensor_stats(query_scaled),
        "raw_qk_fp64_stats": tensor_stats(raw_wide), "raw_qk_fp16_stats": tensor_stats(raw),
        "bias_terms_stats": tensor_stats(bias_terms), "bias_sum_stats": tensor_stats(bias),
        "uncalibrated_visual_stats": uncalibrated, "calibrated_visual_stats": tensor_stats(visual),
        "expected_output_stats": tensor_stats(output),
        "oracle": "independent_expanded_integer_codes_FP64_reduction_official_FP16_boundaries_original_key_order",
    }
