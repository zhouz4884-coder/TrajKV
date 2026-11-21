"""CPU/CUDA operator acceptance for the separately named numeric port v2.

No model, dataset or reference answers. CUDA invocation is Root-scheduled.
The dense FP64 reference conditions on source hard-FP16 packed QK/WV returns;
it is an arithmetic/control gate, not a task-quality or deployment benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "scope": "SYNTHETIC_OPERATOR_CONTROL_NO_MODEL_OR_TASK_SCORE",
              "device": args.device, "references_read": False, "checks": [], "started_unix": time.time()}
    try:
        import torch
        from methods.calibquant_adapter import repeat_kv
        from methods.calibquant_fp32_affine_adapter import CalibQuantFP32AffineAdapter
        from methods.calibquant_fp32_text_adapter import CalibQuantFP32AffineTextAdapter
        device = torch.device(args.device)
        torch.set_grad_enabled(False); torch.manual_seed(532019)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        if device.type == "cuda":
            assert torch.cuda.get_device_capability() == (8, 6); torch.cuda.set_device(0)
            torch.cuda.reset_peak_memory_stats()
        backend = "upstream_triton" if device.type == "cuda" else "torch_unfused"
        report.update(torch_version=torch.__version__, reader_backend=backend,
                      manifest=CalibQuantFP32AffineTextAdapter(1, backend).manifest())

        def reference(owner, q, tk, tv, tp, qp, scale):
            group = q.shape[1] // owner.k_packed.shape[1]
            repeat = lambda x: repeat_kv(x, group)
            raw_qk = owner._raw_qk(q).double()
            ks, kz = repeat(owner.k_scale).double(), repeat(owner.k_zero).double()
            visual = raw_qk * scale - (ks * kz * q.double()).sum(-1, keepdim=True) * scale
            lo, hi = visual.amin(-1, keepdim=True), visual.amax(-1, keepdim=True)
            visual = ((visual - lo) / (hi - lo + 1e-8)) * (hi - 3 - lo) + lo
            text = q.double() @ repeat(tk).double().transpose(-1, -2)
            text = (text * scale).masked_fill(~(tp[None, :] <= qp[:, None])[None, None], -torch.inf)
            p = torch.softmax(torch.cat((visual, text), -1), -1).half()
            raw_v = owner._raw_weighted_codes(p[..., :owner.visual_tokens])
            vs, vz = repeat(owner.v_scale).double(), repeat(owner.v_zero).double()
            visual_out = (raw_v.double() * vs - (vs * vz) * p[..., :owner.visual_tokens].double().sum(-1, keepdim=True)).half()
            # Preserve actual source PV rounding boundaries in the reference.
            text_out = p[..., owner.visual_tokens:] @ repeat(tv)
            return visual_out + text_out

        for bits in (1, 2):
            for heads, kv_heads in ((28, 4), (32, 8)):
                n, d, queries = 13, 128, 3
                k = torch.randn(1, kv_heads, n, d, device=device, dtype=torch.float16) * .15
                v = torch.randn_like(k) * .1
                old = CalibQuantFP32AffineAdapter(bits, backend).encode(k, v)
                owner = CalibQuantFP32AffineTextAdapter(bits, backend).encode(k, v)
                assert set(vars(owner)) == set(vars(old)) and owner.byte_report() == old.byte_report()
                assert len(owner.owner_tensors()) == 8
                assert all(torch.equal(x, old.owner_tensors()[name]) for name, x in owner.owner_tensors().items())
                q = torch.randn(1, heads, queries, d, device=device, dtype=torch.float16) * .2
                tk = torch.randn(1, kv_heads, 5, d, device=device, dtype=torch.float16) * .2
                tv = torch.randn_like(tk) * .1
                tp = torch.tensor([0, 14, 15, 16, 17], device=device); qp = torch.tensor([15, 16, 17], device=device)
                scale = d ** -.5
                kwargs = dict(exact_keys=tk, exact_values=tv, exact_positions=tp, query_positions=qp, scale=scale)
                actual = owner.attend(q, **kwargs)
                ideal = reference(owner, q, tk, tv, tp, qp, scale)
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, ideal, atol=.001, rtol=.01)
                delta = (actual.float() - ideal.float()).abs()
                future = tv.clone(); future[:, :, 3:] += 100
                altered = owner.attend(q, **{**kwargs, "exact_values": future})
                assert torch.equal(actual[:, :, 0], altered[:, :, 0]), "Future exact-text V affected prior query"
                bad = tp.clone(); bad[1] = bad[0]
                try: owner.attend(q, **{**kwargs, "exact_positions": bad})
                except ValueError: pass
                else: raise AssertionError("Duplicate original positions accepted")
                try: owner.attend(q, **{**kwargs, "query_positions": qp + 20})
                except ValueError: pass
                else: raise AssertionError("Invalid prefix coverage accepted")
                # Finite operands whose unscaled Half text dot overflows.
                large_q = torch.full_like(q, 30); large_k = torch.full_like(tk, 30)
                half_raw = large_q @ repeat_kv(large_k, heads // kv_heads).transpose(-1, -2)
                assert torch.isinf(half_raw).all()
                guarded = owner.attend(large_q, **{**kwargs, "exact_keys": large_k})
                assert torch.isfinite(guarded).all()
                report["checks"].append({"status": "PASS", "bits": bits, "query_heads": heads, "kv_heads": kv_heads,
                    "visual_tokens": n, "head_dim": d, "owner_tensors": 8, "persistent_owner_equal_old_affine": True,
                    "conditioned_FP64_reference_max_abs": float(delta.max()), "future_V_causal_intervention_exact": True,
                    "duplicate_positions_rejected": True, "invalid_prefix_rejected": True,
                    "finite_operand_unscaled_Half_text_overflow_control": True, "new_joint_output_finite": True,
                    "physical_visual_owner_bytes": owner.byte_report()["stored_visual_KV_bytes"]})
        if device.type == "cuda":
            torch.cuda.synchronize(); report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), peak_scope="Includes FP32 temporaries and reference instruments; not canonical method timing/memory")
        report["status"] = "PASS_OPERATOR_CPU" if device.type == "cpu" else "PASS_OPERATOR_CUDA"
    except Exception as exc:
        report.update(status="FAILED_OPERATOR_GATE", error=str(exc), traceback=traceback.format_exc()); raise
    finally:
        report["completed_unix"] = time.time()
        report["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), ROOT / "code/methods/calibquant_fp32_text_adapter.py", ROOT / "code/methods/calibquant_fp32_affine_adapter.py", ROOT / "code/methods/calibquant_adapter.py"]}
        with (args.output / "complete.json").open("x", encoding="utf-8") as stream: json.dump(report, stream, indent=2)
        print(json.dumps({"status": report["status"], "checks": len(report["checks"])}), flush=True)


if __name__ == "__main__": main()
