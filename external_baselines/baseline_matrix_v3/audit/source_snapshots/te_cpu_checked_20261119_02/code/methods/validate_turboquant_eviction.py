"""P0 codec/selector acceptance; synthetic arithmetic is not model quality.

Runs original upstream code as reference, including AST-isolated NVIDIA
methods to avoid importing the unrelated Transformers integration framework.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from torch.nn import functional as F

from .turboquant_adapter import TurboQuantAdapter, _upstream, storage_bytes
from .eviction_adapter import PyramidKVVisualAdapter, pyramid_layer_budget, window_scores


def original_method(path, class_name, method_name):
    root = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in root.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    method.decorator_list = []
    env = {"torch": torch, "F": F, "math": math, "nn": torch.nn}
    exec(compile(ast.Module([method], type_ignores=[]), str(path), "exec"), env)
    return env[method_name]


def run(device):
    torch.manual_seed(91273)
    root = Path(__file__).resolve().parents[2]
    result = {"scope": "P0_SYNTHETIC_CODEC_SELECTOR_OWNER_ONLY", "device": str(device),
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "model_quality_measured": False, "tests": []}
    def record(name, **values):
        result["tests"].append({"name": name, "status": "PASS", **values})
    def check_close(a, b, atol=1e-5, rtol=1e-5):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        return float((a.float() - b.float()).abs().max())
    if device.type == "cuda":
        result.update(hardware=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)))
        assert torch.cuda.get_device_capability(device) == (8, 6), "This gate specifically targets RTX3090 SM86"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(1, 4, 192, 128, device=device, dtype=torch.float16)
    v = torch.randn_like(x)
    # Include exact-zero and constant groups; the upstream FP16 epsilon would underflow.
    x[..., 0, :] = 0
    v[..., 1, :] = 2
    query = torch.randn(1, 16, 3, 128, device=device, dtype=torch.float16)
    for bits in (2, 4):
        adapter = TurboQuantAdapter(value_bits=bits)
        owner = adapter.encode(x, v, layer_idx=2)
        kr, vr = owner.materialize()
        prod, qv, dv, _ = _upstream()
        ref = prod(dim=128, bits=3, device=device, dtype=torch.float32, seed=44)
        qref = ref.quantize(x.float())
        reference_k = ref.dequantize(qref)
        reference_v = dv(qv(v.float(), bits=bits, group_size=32), 32)
        kerr = check_close(kr, reference_k, atol=0, rtol=0)
        verr = check_close(vr, reference_v, atol=0, rtol=0)
        assert torch.isfinite(kr).all() and torch.isfinite(vr).all()
        assert owner.key_quantized.mse_indices.dtype == torch.uint8
        assert owner.key_quantized.qjl_signs.dtype == torch.uint8
        assert owner.value_quantized.data.dtype == torch.uint8
        # Independent Algorithm 2 identity: dequant dot equals asymmetric QJL estimator.
        score_ref = ref.attention_score(query[:, :4].float(), qref)
        score_dense = query[:, :4].float() @ kr.transpose(-2, -1)
        score_err = check_close(score_ref, score_dense, atol=2e-5, rtol=2e-5)
        exact_k = torch.randn(1, 4, 4, 128, device=device, dtype=torch.float16)
        exact_v = torch.randn_like(exact_k)
        exact_pos = torch.tensor([192, 193, 194, 195], device=device)
        qp = torch.tensor([193, 194, 195], device=device)
        out = owner.attend(query, exact_keys=exact_k, exact_values=exact_v,
                           exact_positions=exact_pos, query_positions=qp)
        all_k = torch.cat((kr, exact_k.float()), dim=-2).repeat_interleave(4, dim=1)
        all_v = torch.cat((vr, exact_v.float()), dim=-2).repeat_interleave(4, dim=1)
        visible = torch.arange(196, device=device)[None, :] <= qp[:, None]
        ref_out = F.scaled_dot_product_attention(query.float(), all_k, all_v,
                                                attn_mask=visible[None, None]).to(query.dtype)
        output_err = check_close(out, ref_out, atol=1e-3, rtol=1e-3)
        report = owner.owner_report()
        assert report["all_owner_bytes_including_constants"] == storage_bytes(owner.named_tensors())["total_bytes"]
        assert report["payload_bytes"] == x.numel() * (3 + bits) // 8
        original_k_storage = x.untyped_storage().data_ptr()
        original_v_storage = v.untyped_storage().data_ptr()
        assert all(t.untyped_storage().data_ptr() not in (original_k_storage, original_v_storage)
                   for t in owner.named_tensors().values())
        record(f"TurboQuant_K3V{bits}_source_exact_packed_real_reader", key_max_error=kerr,
               value_max_error=verr, qjl_identity_max_error=score_err, reader_sdpa_max_error=output_err,
               owner=report)
    snap_file = root / "third_party/kvpress/kvpress/presses/snapkv_press.py"
    pyramid_file = root / "third_party/kvpress/kvpress/presses/pyramidkv_press.py"
    source_score = original_method(snap_file, "SnapKVPress", "score")
    source_budget = original_method(pyramid_file, "PyramidKVPress", "get_layer_budget")
    window = 8
    keys = torch.randn(1, 4, 256, 128, device=device, dtype=torch.float16)
    vals = torch.randn_like(keys)
    q = torch.randn(1, 16, window, 128, device=device, dtype=torch.float16)
    repeated = keys.repeat_interleave(4, dim=1)
    logits = q @ repeated.transpose(-2, -1) / math.sqrt(128)
    causal = torch.arange(256, device=device)[None, :] <= torch.arange(248, 256, device=device)[:, None]
    attn = torch.softmax(logits.masked_fill(~causal[None, None], -torch.inf), dim=-1, dtype=torch.float32).to(q.dtype)
    dummy = SimpleNamespace(config=SimpleNamespace(num_attention_heads=16, num_key_value_heads=4, num_hidden_layers=28),
                            layer_idx=0)
    selector = SimpleNamespace(window_size=window, kernel_size=5)
    original = source_score(selector, dummy, torch.empty(1, 256, 2), keys, vals, attn, {})
    actual = window_scores(keys, q, window_size=window, kernel_size=5)
    score_err = check_close(actual, original, atol=0, rtol=0)
    budget_cases = 0
    for n in (16, 64, 256, 2048):
        for ratio in (0.0, 0.75, 0.875, 0.95):
            for layer in (0, 14, 27):
                config = SimpleNamespace(beta=20, window_size=window, compression_ratio=ratio)
                dummy.layer_idx = layer
                expected = source_budget(config, dummy, n)
                assert expected == pyramid_layer_budget(n, ratio, window, 20, layer, 28)
                budget_cases += 1
    record("PyramidKV_original_NVIDIA_score_and_budget", score_max_error=score_err, exact_budget_cases=budget_cases)
    vp = torch.arange(16, 240, device=device)
    adapter = PyramidKVVisualAdapter(compression_ratio=0.75, window_size=window)
    eo = adapter.encode(keys, vals, visual_positions=vp, window_queries=q, layer_idx=0,
                        prefix_is_question_free=True)
    # Independently reproduce original score topk after the declared visual-only restriction.
    text_count = keys.shape[-2] - len(vp)
    keep = max(0, pyramid_layer_budget(256, 0.75, window, 20, 0, 28) - text_count)
    expected_idx = vp[original.index_select(-1, vp).topk(keep, dim=-1).indices]
    torch.testing.assert_close(eo.positions, expected_idx, atol=0, rtol=0)
    torch.testing.assert_close(eo.keys, keys.gather(-2, expected_idx[..., None].expand(*expected_idx.shape, 128)), atol=0, rtol=0)
    assert eo.keys.untyped_storage().data_ptr() != keys.untyped_storage().data_ptr()
    # A genuine sparse reader using per-head original positions, plus all exact text.
    text_mask = torch.ones(256, device=device, dtype=torch.bool)
    text_mask[vp] = False
    tp = torch.arange(256, device=device)[text_mask]
    qr = torch.randn(1, 16, 2, 128, device=device, dtype=torch.float16)
    out = eo.attend(qr, exact_keys=keys[:, :, text_mask], exact_values=vals[:, :, text_mask],
                    exact_positions=tp, query_positions=torch.tensor([256, 257], device=device))
    sparse_k = torch.cat((eo.keys, keys[:, :, text_mask]), dim=-2).repeat_interleave(4, dim=1)
    sparse_v = torch.cat((eo.values, vals[:, :, text_mask]), dim=-2).repeat_interleave(4, dim=1)
    ref_out = F.scaled_dot_product_attention(qr.float(), sparse_k.float(), sparse_v.float()).to(qr.dtype)
    output_err = check_close(out, ref_out, atol=1e-3, rtol=1e-3)
    record("PyramidKV_visual_topk_compact_owner_sparse_reader", reader_max_error=output_err, owner=eo.owner_report())
    for condition, kw in (("unknown_rejects_question", {"question_in_observation": True, "prefix_is_question_free": True}),
                          ("unknown_rejects_unverified_prefix", {})):
        try:
            adapter.encode(keys, vals, visual_positions=vp, window_queries=q, **kw)
        except ValueError as exc:
            assert "INPUT_MISMATCH" in str(exc)
            record(condition)
        else:
            raise AssertionError(condition)
    if device.type == "cuda":
        torch.cuda.synchronize()
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["end_allocated_bytes"] = torch.cuda.memory_allocated(device)
    result["status"] = "PASS"
    result["source_hashes"] = {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__), Path(__file__).with_name("turboquant_adapter.py"),
                                         Path(__file__).with_name("eviction_adapter.py"), snap_file, pyramid_file)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(torch.device(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"status": result["status"], "output": str(args.output), "tests": len(result["tests"])}))
