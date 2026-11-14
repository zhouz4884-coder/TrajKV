"""CPU-only parity and packing checks for the KVarN mechanism port.

These synthetic checks validate wiring, shapes and upstream numerical parity.
They provide no model quality, GPU latency or memory-throughput evidence.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch
import kvarn_reference as ref


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261112)
    _, store, decode = ref._upstream()
    cases = []
    for axis in ("k", "v"):
        for bits in (2, 4):
            for rotation_dtype in ("float32", "float16"):
                x = torch.randn(2, 3, 259, 128, dtype=torch.float32)
                x = x * torch.linspace(0.4, 2.0, 259).view(1, 1, -1, 1)
                r = ref.encode_kvarn(x, axis, bits, 128, 16, rotation_dtype)
                y = ref.decode_kvarn(r)
                assert y.shape == x.shape and y.dtype == x.dtype
                assert torch.isfinite(y).all()
                assert torch.equal(y[..., 256:, :], x[..., 256:, :])
                p = {k: v[0] for k, v in r.packed.items()}
                if axis == "k":
                    tile = decode.kvarn_dequant_tile_k(**p, group=128, bits=bits).T
                else:
                    tile = decode.kvarn_dequant_tile_v(**p, head_dim=128, bits=bits)
                h = ref._hadamard(128, "cpu")
                expected = ((tile.half() @ h.half()).float() if rotation_dtype == "float16"
                            else tile @ h)
                parity = (expected - y.reshape(-1, 259, 128)[0, :128]).abs().max().item()
                # Batched and single-tile GEMMs may use different reduction
                # orders; the upstream dequantization precedes this GEMM.
                tolerance = 5e-3 if rotation_dtype == "float16" else 1e-5
                assert parity <= tolerance, (axis, bits, rotation_dtype, parity)
                q = torch.randint(0, 1 << bits, (3, 128, 128), dtype=torch.int32)
                packed = store._pack_lowbit(q, bits)
                unpacked = decode._unpack_lowbit(packed, 128, bits)
                assert torch.equal(q, unpacked.int())
                cases.append({"axis": axis, "bits": bits,
                              "rotation_dtype": rotation_dtype,
                              "shape": list(x.shape),
                              "upstream_dequant_max_abs": parity,
                              "tail_exact": True, "packing_roundtrip": True,
                              "finite": True})
    for dtype in (torch.float16, torch.bfloat16):
        for length in (0, 7, 128, 129):
            for value in (0.0, 2.5):
                x = torch.full((1, 2, length, 128), value, dtype=dtype)
                for axis in ("k", "v"):
                    y = ref.quant_dequant_kvarn(x, axis, bits=2)
                    assert y.shape == x.shape and y.dtype == x.dtype
                    assert torch.isfinite(y).all()
                    if length < 128:
                        assert torch.equal(x, y)
    # Confirm GQA-like KV-head shape does not depend on query-head count.
    x = torch.randn(1, 8, 256, 128, dtype=torch.float32).transpose(0, 1)
    y = ref.quant_dequant_kvarn(x, "k", bits=4)
    assert x.shape == y.shape and torch.isfinite(y).all()
    commit = subprocess.check_output(
        ["git", "-C", str(ref.UPSTREAM_ROOT), "rev-parse", "HEAD"], text=True).strip()
    assert commit == ref.UPSTREAM_COMMIT
    payload = {
        "status": "PASS_CPU_UPSTREAM_PARITY_AND_PACKING",
        "evidence_scope": "synthetic_cpu_correctness_only",
        "device": "cpu", "torch": torch.__version__,
        "source_commit": commit, "source_sha256": ref.source_fingerprints(),
        "cases": cases,
        "additional_checks": ["fp16_bf16_empty_and_short_tail", "constant_and_zero_tiles",
                              "noncontiguous_input", "kv_head_shape"],
        "not_measured": ["model_quality", "gpu_compatibility", "gpu_latency",
                         "peak_gpu_memory", "physical_dram_traffic"],
    }
    out = Path(__file__).with_name("kvarn_cpu_smoke.json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "cases": len(cases),
                      "result": str(out)}))


if __name__ == "__main__":
    main()
