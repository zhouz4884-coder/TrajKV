"""CPU-only format/finite-source/isolated-bias-overflow control for new port."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from methods.calibquant_adapter import CalibQuantAdapter, CalibQuantState
from methods.calibquant_fp32_bias_adapter import CalibQuantFP32BiasAdapter, CalibQuantFP32BiasState


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.manual_seed(191)
    results = []
    for bits in [1, 2]:
        for n in [32, 257, 512]:
            k, v, q = torch.randn(1, 4, n, 128).half(), torch.randn(1, 4, n, 128).half(), torch.randn(1, 28, 3, 128).half()
            old = CalibQuantAdapter(bits, "torch_unfused").encode(k, v)
            new = CalibQuantFP32BiasAdapter(bits, "torch_unfused").encode(k, v)
            assert isinstance(new, CalibQuantFP32BiasState)
            assert all(torch.equal(tensor, new.owner_tensors()[name]) for name, tensor in old.owner_tensors().items())
            assert old.byte_report() == new.byte_report()
            raw_old, raw_new = old.read(q, calibrate=False), new.read(q, calibrate=False)
            output = new.read(q)
            assert torch.isfinite(output).all()
            g = 7
            scale, zero = new.k_scale.repeat_interleave(g, 1), new.k_zero.repeat_interleave(g, 1)
            products = scale * zero * q
            expected_bias = (products.double().sum(-1, keepdim=True) / (128 ** .5)).half()
            old_bias = products.sum(-1, keepdim=True) * (128 ** -.5)
            expected_new = raw_old + old_bias - expected_bias
            # Different subtraction/addition rounding in this derived control
            # is bounded; real-model CUDA gate uses a direct independent oracle.
            torch.testing.assert_close(raw_new, expected_new, atol=.0078125, rtol=.005)
            results.append({"bits": bits, "n": n, "format_and_owner_equal": True,
                            "finite_source_uncalibrated_max_abs_difference": float((raw_new - raw_old).abs().max()),
                            "finite_source_calibrated_max_abs_difference": float((output - old.read(q)).abs().max())})
    # Construct a legal packed state with finite channel products but an FP16
    # bias sum >65504. Isolate Eq6 bias arithmetic with calibration disabled;
    # the original calibration map is deliberately not repaired or clamped.
    bits = 2
    k = (100 + torch.rand(1, 4, 32, 128)).half()
    v = torch.randn_like(k)
    q = torch.full((1, 28, 3, 128), 10., dtype=torch.float16)
    old = CalibQuantAdapter(bits, "torch_unfused").encode(k, v)
    new = CalibQuantFP32BiasAdapter(bits, "torch_unfused").encode(k, v)
    products = new.k_scale.repeat_interleave(7, 1) * new.k_zero.repeat_interleave(7, 1) * q
    assert torch.isfinite(products).all() and not torch.isfinite(products.sum(-1)).all()
    assert not torch.isfinite(old.read(q, calibrate=False)).all()
    assert torch.isfinite(new.read(q, calibrate=False)).all()
    result = {"status": "CPU_CONTROL_ONLY_FP32_BIAS_PORT", "cases": results,
              "isolated_finite_product_sum_overflow": {"published_FP16_nonfinite": True, "new_port_finite": True,
                  "calibration_disabled_only_for_isolation": True, "bias_product_max_abs": float(products.abs().max())},
              "new_method_manifest": CalibQuantFP32BiasAdapter().manifest(), "quality_measured": False,
              "source_hashes": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), ROOT / "code/methods/calibquant_fp32_bias_adapter.py"]}}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"status": result["status"], "healthy_cases": len(results), "isolated_overflow_control_passed": True}))


if __name__ == "__main__":
    main()
