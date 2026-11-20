"""KIVI visual adapter acceptance with independent official-source codec oracle.

CPU builds mathematical fixtures using AST-isolated author PyTorch routines;
it NEVER substitutes a fake encoder into the production adapter. CUDA calls
the original minmax/pack Triton encoder and the actual packed online reader.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from methods.kivi_style import KIVIStyleAdapter, KIVIVisualState


def source_functions():
    path = ROOT / "third_party/kivi/quant/new_pack.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {"pack_tensor", "unpack_tensor", "quant_and_pack_kcache", "quant_and_pack_vcache",
              "unpack_and_dequant_kcache", "unpack_and_dequant_vcache"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in functions} == wanted
    env = {"torch": torch}
    exec(compile(ast.Module(functions, type_ignores=[]), str(path), "exec"), env)
    return env


def official_fixture(k, v, positions, bits, residual, env):
    """Direct author PyTorch source, transformed only to transposed K layout."""
    n, h = k.shape[-2], k.shape[1]
    nkq, nvq = n - n % 128, max(0, n - residual)
    if nkq:
        kc, ks, km = env["quant_and_pack_kcache"](k[:, :, :nkq].contiguous(), 128, bits)
        kc = kc.transpose(-2, -1).contiguous()
        ks, km = ks.squeeze(-2).transpose(-2, -1).contiguous(), km.squeeze(-2).transpose(-2, -1).contiguous()
    else:
        kc = torch.empty((1, h, 128, 0), device=k.device, dtype=torch.int32)
        ks, km = k.new_empty((1, h, 128, 0)), k.new_empty((1, h, 128, 0))
    if nvq:
        vc, vs, vm = env["quant_and_pack_vcache"](v[:, :, :nvq].contiguous(), 128, bits)
        vs, vm = vs.squeeze(-1).contiguous(), vm.squeeze(-1).contiguous()
    else:
        vc = torch.empty((1, h, 0, 128 // (32 // bits)), device=k.device, dtype=torch.int32)
        vs, vm = k.new_empty((1, h, 0, 1)), k.new_empty((1, h, 0, 1))
    return KIVIVisualState(kc, ks, km, vc, vs, vm, k[:, :, nkq:].clone(), v[:, :, nvq:].clone(),
                           positions.clone().contiguous(), bits, n, nkq, nvq, residual, "torch_dequant_reference")


def source_decode(state, env):
    if state.nkq:
        k = env["unpack_and_dequant_kcache"](state.k_code.transpose(-2, -1).contiguous(),
                  state.k_scale.transpose(-2, -1).unsqueeze(-2), state.k_min.transpose(-2, -1).unsqueeze(-2), 128, state.bits)
    else:
        k = state.k_tail[:, :, :0]
    if state.nvq:
        v = env["unpack_and_dequant_vcache"](state.v_code, state.v_scale.unsqueeze(-1), state.v_min.unsqueeze(-1), 128, state.bits)
    else:
        v = state.v_tail[:, :, :0]
    return torch.cat((k, state.k_tail), -2), torch.cat((v, state.v_tail), -2)


def expected_owner_bytes(state):
    h, d, n, kq, vq, bits = state.k_tail.shape[1], 128, state.nv, state.nkq, state.nvq, state.bits
    # Counts derive from scalar geometry, independently of tensor.numel/report.
    payload = h * d * kq * bits // 8 + h * vq * d * bits // 8
    metadata = 2 * 2 * (h * d * (kq // 128) + h * vq * (d // 128))
    tails = 2 * h * d * ((n - kq) + (n - vq))
    positions = 8 * n
    return {"payload": payload, "metadata": metadata, "tails": tails, "positions": positions,
            "total": payload + metadata + tails + positions}


def state_storage_bytes(state):
    seen, total = set(), 0
    for tensor in state.named_tensors().values():
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if identity not in seen:
            seen.add(identity)
            total += storage.nbytes()
    return total


def dense_reference(q, state, text_k, text_v, text_positions, query_positions, env):
    k, v = source_decode(state, env)
    groups = q.shape[1] // k.shape[1]
    keys = torch.cat((k, text_k), -2).repeat_interleave(groups, 1)
    values = torch.cat((v, text_v), -2).repeat_interleave(groups, 1)
    positions = torch.cat((state.positions, text_positions))
    visible = positions[None, :] <= query_positions[:, None]
    # Independent original source decode + Torch SDPA FP32 arithmetic oracle.
    return F.scaled_dot_product_attention(q.float(), keys.float(), values.float(),
                                         attn_mask=visible[None, None]).to(q.dtype)


def run(device):
    env = source_functions()
    torch.manual_seed(910311)
    result = {"status": "RUNNING", "scope": "KIVI_VISUAL_CODEC_TAIL_PACKED_READER_OWNER_SYNTHETIC_P0",
              "model_quality_measured": False, "device": str(device), "torch_version": torch.__version__,
              "actual_gpu_encoder_and_packed_reader": device.type == "cuda",
              "cpu_fixture_scope": "AST-executed official PyTorch codec only; production encoder not replaced",
              "tests": []}
    if device.type == "cuda":
        assert torch.cuda.get_device_capability(device) == (8, 6)
        result["hardware"] = torch.cuda.get_device_name(device)
        torch.cuda.reset_peak_memory_stats(device)
    for n in (17, 257, 512, 1008, 1792):
        for hq, hkv in ((28, 4), (32, 8)):
            k = torch.randn((1, hkv, n, 128), device=device, dtype=torch.float16) * .7
            v = torch.randn_like(k) * .7
            # Deliberately pass a noncontiguous position view with text holes.
            base_positions = torch.arange(3, 3 + 2 * n, device=device)
            positions = base_positions[::2]
            assert not positions.is_contiguous()
            prefix_length = int(positions[-1]) + 2
            prefix_text = torch.arange(prefix_length, device=device)
            prefix_text = prefix_text[~torch.isin(prefix_text, positions)]
            for bits in (2, 4):
                for residual in (0, 128):
                    fixture = official_fixture(k, v, positions, bits, residual, env)
                    if device.type == "cuda":
                        state = KIVIStyleAdapter(bits=bits, residual_length=residual).encode(k, v, positions)
                        for name in ("k_code", "k_scale", "k_min", "v_code", "v_scale", "v_min", "k_tail", "v_tail", "positions"):
                            torch.testing.assert_close(getattr(state, name), getattr(fixture, name), atol=0, rtol=0)
                    else:
                        state = fixture
                    independent = source_decode(state, env)
                    materialized = state.materialize()
                    for actual, expected in zip(materialized, independent):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    expected = expected_owner_bytes(state)
                    assert state_storage_bytes(state) == expected["total"]
                    owner_report = state.owner_report()
                    assert owner_report["stored_visual_KV_bytes"] == expected["total"]
                    assert owner_report["packed_payload_bytes"] == expected["payload"]
                    assert owner_report["coder_metadata_bytes"] == expected["metadata"]
                    assert owner_report["full_precision_visual_tail_bytes"] == expected["tails"]
                    assert owner_report["positions_bytes"] == expected["positions"]
                    native_ptrs = {k.untyped_storage().data_ptr(), v.untyped_storage().data_ptr()}
                    assert all(t.untyped_storage().data_ptr() not in native_ptrs for t in state.named_tensors().values() if t.numel())
                    assert state.k_tail.shape[-2] == n % 128 and state.v_tail.shape[-2] == min(residual, n)
                    for qn in (1, 9):
                        query_positions = torch.arange(prefix_length, prefix_length + qn, device=device)
                        text_positions = torch.cat((prefix_text, query_positions))
                        text_k = torch.randn((1, hkv, len(text_positions), 128), device=device, dtype=torch.float16) * .7
                        text_v = torch.randn_like(text_k) * .7
                        q = torch.randn((1, hq, qn, 128), device=device, dtype=torch.float16) * .7
                        oracle = dense_reference(q, state, text_k, text_v, text_positions, query_positions, env)
                        if device.type == "cuda":
                            # Enforce actual compressed-state consumption. Any fake
                            # materialize fallback fails before the output comparison.
                            def forbidden(*args, **kwargs):
                                raise AssertionError("Packed reader called full dense materialize")
                            state.materialize = forbidden
                            try:
                                actual = state.attend(q, exact_keys=text_k, exact_values=text_v,
                                                      exact_positions=text_positions, query_positions=query_positions,
                                                      backend="triton_packed")
                            finally:
                                del state.materialize
                        else:
                            actual = state.attend(q, exact_keys=text_k, exact_values=text_v,
                                                  exact_positions=text_positions, query_positions=query_positions,
                                                  backend="torch_dequant_reference")
                        error = (actual.float() - oracle.float()).abs()
                        torch.testing.assert_close(actual, oracle, atol=.002, rtol=.005)
                        assert torch.isfinite(actual).all()
                        result["tests"].append({"N": n, "Q": qn, "Hq": hq, "Hkv": hkv, "bits": bits, "R": residual,
                                                "status": "PASS", "source_code_scale_min_exact": True,
                                                "max_abs_error": float(error.max()), "mean_abs_error": float(error.mean()),
                                                "owner_bytes": expected, "K_tail_tokens": n - state.nkq,
                                                "V_tail_tokens": n - state.nvq,
                                                "packed_materialize_forbidden": device.type == "cuda"})
        print(json.dumps({"event": "N_completed", "N": n, "tests": len(result["tests"])}), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
        result["peak_allocated_bytes_including_dense_controls"] = torch.cuda.max_memory_allocated(device)
    result["source_hashes"] = {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (Path(__file__), ROOT / "code/methods/kivi_style.py",
                                           ROOT / "code/methods/kivi_reader.py", ROOT / "third_party/kivi/quant/new_pack.py")}
    result["status"] = "PASS"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(torch.device(args.device))
    except Exception:
        result = {"status": "FAILED_RUN", "device": args.device, "traceback": traceback.format_exc()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as out: json.dump(result, out, indent=2)
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out: json.dump(result, out, indent=2)
    print(json.dumps({"status": result["status"], "tests": len(result["tests"]), "output": str(args.output)}))
