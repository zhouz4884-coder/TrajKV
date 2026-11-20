"""Independent packed-storage and complete CalibQuant QK/calibration/WV gate.

CPU is a labeled control only. CUDA results additionally execute the selected
packed reader; this script does not claim model/suffix replay acceptance.
"""
from __future__ import annotations
import argparse
import ast
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import weakref

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from methods.calibquant_adapter import CalibQuantAdapter, repeat_kv, unpack_codes, UPSTREAM_COMMIT


def source_functions():
    path = ROOT / "third_party" / "calibquant" / "calibquant.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("quant", "encode"):
            node.decorator_list = []
            functions.append(node)
    namespace = {"torch": torch, "BITS_KV": 1}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["quant"], namespace["encode"]


def max_error(actual, expected):
    return float((actual.double() - expected.double()).abs().max())


def check_case(n, kv_heads, q_heads, bits, device, dtype, backend, seed):
    torch.manual_seed(seed)
    k = torch.randn(1, kv_heads, n, 128, device=device, dtype=dtype)
    v = torch.randn_like(k)
    q = torch.randn(1, q_heads, 3, 128, device=device, dtype=dtype)
    quant, encode = source_functions()
    kcodes, kscale, kzero = quant(k.transpose(1, 2), bits, True, .5 if bits == 1 else 1., True)
    vcodes, vscale, vzero = quant(v.transpose(1, 2), bits, True, .3 if bits == 1 else 1., True)
    kcodes = kcodes.transpose(1, 2).contiguous()
    vcodes = vcodes.transpose(1, 2).contiguous()
    state = CalibQuantAdapter(bits, backend).encode(k, v)
    assert torch.equal(state.k_packed, encode(kcodes.to(torch.int64), bits, 8 // bits))
    assert torch.equal(state.v_packed, encode(vcodes.to(torch.int64), bits, 8 // bits))
    assert torch.equal(unpack_codes(state.k_packed, bits), kcodes.to(torch.uint8))
    assert torch.equal(unpack_codes(state.v_packed, bits), vcodes.to(torch.uint8))
    assert torch.equal(state.k_scale, kscale.transpose(1, 2))
    assert torch.equal(state.k_zero, kzero.transpose(1, 2))
    assert torch.equal(state.v_scale, vscale.transpose(1, 2))
    assert torch.equal(state.v_zero, vzero.transpose(1, 2))
    groups = q_heads // kv_heads
    ks, kz = repeat_kv(state.k_scale, groups), repeat_kv(state.k_zero, groups)
    vs, vz = repeat_kv(state.v_scale, groups), repeat_kv(state.v_zero, groups)
    # Float64 oracle uses independently source-quantized integer codes. Round at
    # the official kernel's FP16 output boundary before post-scale/calibration.
    raw = torch.matmul((q * ks).double(), repeat_kv(kcodes, groups).double().transpose(-2, -1)).to(dtype)
    expected_logits = raw / math.sqrt(128) - (ks * kz * q).sum(-1, keepdim=True) / math.sqrt(128)
    actual_logits = state.read(q, calibrate=False)
    logits_error = max_error(actual_logits, expected_logits)
    torch.testing.assert_close(actual_logits, expected_logits, atol=.015625, rtol=.003)
    if bits <= 2:
        # Replay one token at a time exactly as released code, independent of
        # adapter's generalized multi-query implementation.
        calibrated_parts = []
        for t in range(q.shape[2]):
            score = expected_logits[:, :, t:t + 1]
            maximum = score.amax(dim=(2, 3), keepdim=True)
            minimum = score.amin(dim=(2, 3), keepdim=True)
            calibrated_parts.append((score - minimum) / (maximum - minimum + 1e-8)
                                    * (maximum - 3 - minimum) + minimum)
        expected_calibrated = torch.cat(calibrated_parts, dim=2)
    else:
        expected_calibrated = expected_logits
    actual_calibrated = state.read(q)
    calibrated_error = max_error(actual_calibrated, expected_calibrated)
    torch.testing.assert_close(actual_calibrated, expected_calibrated, atol=.03125, rtol=.005)
    # Include competing full-precision text logits. A visual-only softmax is
    # insufficient to test calibration's influence on modality probability mass.
    text_logits = torch.randn(1, q_heads, 3, 11, device=device, dtype=dtype)
    full_weights = torch.softmax(torch.cat([text_logits, actual_calibrated], -1), dim=-1, dtype=torch.float32).to(dtype)
    weights = full_weights[..., 11:]
    expected_wv = torch.matmul(weights.double(), repeat_kv(vcodes, groups).double()).to(dtype)
    expected_wv = expected_wv * vs - vs * vz * weights.sum(-1, keepdim=True)
    actual_wv = state.weighted_value(weights)
    wv_error = max_error(actual_wv, expected_wv)
    torch.testing.assert_close(actual_wv, expected_wv, atol=.0078125, rtol=.005)
    assert bool(torch.isfinite(actual_calibrated).all() & torch.isfinite(actual_wv).all())
    # Verify original FP16 cache can be released, and no owner shares its storage.
    original_storage = {(str(x.device), x.untyped_storage().data_ptr()) for x in (k, v)}
    owners = state.owner_tensors()
    assert not original_storage.intersection((str(x.device), x.untyped_storage().data_ptr()) for x in owners.values())
    original_refs = (weakref.ref(k), weakref.ref(v))
    del k, v
    gc.collect()
    assert all(reference() is None for reference in original_refs)
    assert len(owners) == 8
    expected_bytes = (2 * kv_heads * n * 128 * bits // 8 + 4 * kv_heads * 128 * torch.empty((), dtype=dtype).element_size()
                      + 2 * (1 << bits) * torch.empty((), dtype=dtype).element_size())
    assert state.byte_report()["stored_visual_KV_bytes"] == expected_bytes
    return {"n": n, "kv_heads": kv_heads, "q_heads": q_heads, "queries": 3,
            "bits": bits, "dtype": str(dtype), "qk_max_abs_error": logits_error,
            "calibrated_logits_max_abs_error": calibrated_error, "wv_max_abs_error": wv_error,
            "pack_matches_source": True, "quant_matches_source": True,
            "original_cache_released": True, "no_original_storage_alias": True,
            "owner": state.byte_report()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("torch_unfused", "upstream_triton"), default="torch_unfused")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[17, 257, 512])
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Frozen gate output already exists: {args.output}")
    started = time.time()
    result = {"gate": "calibquant_packed_math_owner", "upstream_commit": UPSTREAM_COMMIT,
              "device": args.device, "backend": args.backend, "cases": [],
              "not_covered": ["real model layer hooks", "all-layer suffix replay", "benchmark quality", "throughput advantage"]}
    failed = False
    try:
        if args.device.startswith("cuda"):
            torch.cuda.set_device(torch.device(args.device))
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats()
            result["hardware"] = torch.cuda.get_device_name()
            result["capability"] = list(torch.cuda.get_device_capability())
        for bits in args.bits:
            for n in args.tokens:
                # GQA geometries of M1/M2 and M3. No checkpoint is loaded here.
                for kvh, qh in [(4, 28), (8, 32)]:
                    result["cases"].append(check_case(n, kvh, qh, bits, args.device,
                                                       torch.float16, args.backend, 2718 + n + bits))
        result["status"] = "PASS_CUDA_MATH_OWNER_ONLY" if args.device.startswith("cuda") else "CPU_CONTROL_ONLY"
    except Exception as exc:
        failed = True
        result["status"] = "FAILED_RUN"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    result["wall_seconds"] = time.time() - started
    result["torch_version"] = torch.__version__
    result["cuda_version"] = torch.version.cuda
    result["code_hashes"] = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__), ROOT / "code/methods/calibquant_adapter.py"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: result[k] for k in ("status", "device", "backend", "wall_seconds")}), flush=True)
    print(f"cases={len(result['cases'])} output={args.output}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
