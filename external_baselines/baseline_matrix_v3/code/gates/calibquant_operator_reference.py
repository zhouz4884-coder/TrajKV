"""Execute author's attention AST at declared precision and mapped key order.

Only the bias sum/div output precision changes. Packed QK/WV calls are original
source kernels with declared bitwidth. The visual-first/text-tail permutation
matches the shared engine and its correspondingly permuted native mask.
"""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from methods.calibquant_adapter import _load_upstream
from calibquant_dense_oracle import _codes

ROOT = Path(__file__).resolve().parents[2]


def ulp_distance(a, b):
    assert a.dtype == b.dtype == torch.float16
    def ordered(x):
        bits = x.contiguous().view(torch.int16).to(torch.int32)
        magnitude = bits & 0x7fff
        return torch.where(bits < 0, 0x8000 - magnitude, 0x8000 + magnitude)
    return (ordered(a) - ordered(b)).abs()


def difference_summary(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    flat = int(delta.flatten().argmax())
    coordinates = []
    remaining = flat
    for width in reversed(delta.shape):
        coordinates.append(remaining % width)
        remaining //= width
    coordinates = list(reversed(coordinates))
    index = tuple(coordinates)
    ulps = ulp_distance(actual, expected)
    scalar = float(expected[index])
    magnitude = abs(scalar)
    if magnitude < 2 ** -14:
        spacing = 2 ** -24
    else:
        exponent = math.floor(math.log2(magnitude))
        spacing = 2 ** (exponent - 10)
        if scalar < 0 and magnitude == 2 ** exponent:
            spacing /= 2
    return {"max_abs": float(delta.max()), "relative_l2": float(torch.linalg.vector_norm(delta) /
                torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)),
            "abs_quantiles_0_50_90_99_999_100": torch.quantile(delta.flatten(), torch.tensor([0., .5, .9, .99, .999, 1.], device=delta.device)).cpu().tolist(),
            "max_ulp": int(ulps.max()), "elements": delta.numel(), "different_elements": int((delta != 0).sum()),
            "max_abs_location": coordinates, "actual_at_max_abs": float(actual[index]), "reference_at_max_abs": float(expected[index]),
            "output_ulp_at_max_abs": int(ulps[index]),
            "reference_spacing_up_at_max_abs": spacing,
            "spacing_basis": "IEEE binary16 bit spacing; no unsupported CUDA Half nextafter"}


def source_operator_attention(cache, index, q, native_mask):
    owner, layer = cache.layers[index].visual_owner, cache.layers[index]
    groups = q.shape[1] // owner.k_packed.shape[1]
    repeat = lambda x: x.repeat_interleave(groups, 1)
    source_path = ROOT / "third_party/calibquant/patch_internvl.py"
    function = next(n for n in ast.parse(source_path.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "forward_internlm2_calibquant")
    assigns = sorted([n for n in ast.walk(function) if isinstance(n, ast.Assign)], key=lambda n: n.lineno)
    def statement(name, number=0):
        return copy.deepcopy([n for n in assigns if any(isinstance(t, ast.Name) and t.id == name for t in n.targets)][number])
    bias_product = statement("bias_k", 0)
    bias_reduction = statement("bias_k", 1)
    sums = [n for n in ast.walk(bias_reduction.value) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "sum"]
    assert len(sums) == 1
    sums[0].keywords.append(ast.keyword(arg="dtype", value=ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()), attr="float32", ctx=ast.Load())))
    bias_reduction.value = ast.Call(func=ast.Attribute(value=bias_reduction.value, attr="to", ctx=ast.Load()),
                                   args=[ast.Attribute(value=ast.Name(id="query_states", ctx=ast.Load()), attr="dtype", ctx=ast.Load())], keywords=[])
    visual_assignment = [n for n in assigns if any(isinstance(t, ast.Name) and t.id == "attn_weights_quant_v" for t in n.targets)
                         and any(isinstance(c, ast.Name) and c.id == "fused_kernel" for c in ast.walk(n.value))][0]
    normalize = next(n for n in ast.walk(function) if isinstance(n, ast.If) and isinstance(n.test, ast.Attribute) and n.test.attr == "normalize")
    softmax = [n for n in assigns if any(isinstance(t, ast.Name) and t.id == "attn_weights" for t in n.targets)
               and any(isinstance(c, ast.Attribute) and c.attr == "softmax" for c in ast.walk(n.value))][0]
    value_first = [n for n in assigns if any(isinstance(t, ast.Name) and t.id == "attn_output_v" for t in n.targets)
                   and any(isinstance(c, ast.Name) and c.id == "fused_kernel_wv" for c in ast.walk(n.value))][0]
    value_rest = [n for n in assigns if n.lineno > value_first.lineno and n.lineno <= value_first.lineno + 4]
    def execute(nodes, namespace):
        exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[])), str(source_path), "exec"), namespace)
    upstream = _load_upstream()
    self_ref = SimpleNamespace(scale_key=repeat(owner.k_scale), zero_point_key=repeat(owner.k_zero),
        scale_value=repeat(owner.v_scale), zero_point_value=repeat(owner.v_zero), head_dim=q.shape[-1],
        unique_values_k=owner.k_lut, unique_values_v=owner.v_lut, sprompt_len=0,
        image_token_num=owner.visual_tokens, normalize=owner.bits <= 2)
    encoded_k, encoded_v = repeat(owner.k_packed).contiguous(), repeat(owner.v_packed).contiguous()
    text_k, text_v = repeat(layer.keys), repeat(layer.values)
    if native_mask is None:
        mask = (torch.arange(layer.logical_length, device=q.device)[None, :] <= layer.query_positions[:, None])[None, None]
    else:
        mask = native_mask
    mapped = torch.cat((mask.index_select(-1, layer.visual_positions), mask.index_select(-1, layer.text_positions)), -1)
    outputs, raw_kernels = [], []
    for t in range(q.shape[-2]):
        captured = {}
        def qk(lut, weights, encoded):
            value = upstream.fused_kernel(lut, weights.contiguous(), encoded, BITS=owner.bits, DATUM_PER_ELM=8 // owner.bits)
            captured["qk_raw"] = value
            return value
        def wv(lut, weights, encoded):
            value = upstream.fused_kernel_wv(lut, weights.contiguous(), encoded, BITS=owner.bits, DATUM_PER_ELM=8 // owner.bits)
            captured["wv_raw"] = value
            return value
        ns = {"torch": torch, "math": math, "nn": torch.nn, "self": self_ref,
              "query_states": q[:, :, t:t + 1], "encoded_features_k": encoded_k,
              "encoded_features_v": encoded_v, "fused_kernel": qk, "fused_kernel_wv": wv}
        execute([bias_product, bias_reduction, visual_assignment, normalize], ns)
        text_logits = (q[:, :, t:t + 1] @ text_k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        ns["attn_weights"] = torch.cat((ns["attn_weights_quant_v"].float(), text_logits.float()), -1)
        current_mask = mapped[..., t:t + 1, :]
        if current_mask.dtype == torch.bool:
            ns["attn_weights"] = ns["attn_weights"].masked_fill(~current_mask, -torch.inf)
        else:
            ns["attn_weights"] = ns["attn_weights"] + current_mask
        execute([softmax, value_first], ns)
        execute(value_rest, ns)
        weights = ns["attn_weights"]
        output = ns["attn_output_v"] + weights[..., owner.visual_tokens:] @ text_v
        outputs.append(output)
        captured["visual_weights"] = weights[..., :owner.visual_tokens]
        raw_kernels.append(captured)
    output = torch.cat(outputs, 2).transpose(1, 2).contiguous()
    codes = repeat(_codes(owner.v_packed, owner.bits)).double()
    weighted_fp64 = torch.cat([(r["visual_weights"].double() @ codes).to(q.dtype) for r in raw_kernels], 2)
    weighted_source = torch.cat([r["wv_raw"] for r in raw_kernels], 2)
    raw_qk_source = torch.cat([r["qk_raw"] for r in raw_kernels], 2)
    kcodes = repeat(_codes(owner.k_packed, owner.bits)).double()
    raw_qk_fp64 = ((q * self_ref.scale_key).double() @ kcodes.transpose(-1, -2)).to(q.dtype)
    return output, {"raw_qk_source_vs_fp64": difference_summary(raw_qk_source, raw_qk_fp64),
                    "raw_wv_source_vs_fp64_same_probabilities": difference_summary(weighted_source, weighted_fp64),
                    "weighted_source": weighted_source, "weighted_fp64": weighted_fp64,
                    "value_scale": self_ref.scale_value}
