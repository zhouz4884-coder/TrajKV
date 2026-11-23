"""Tiny Round11 v2 operator/encoder checks; no model or scientific answers.

Default execution requires actual CUDA. --cpu-only is an explicitly labelled
preflight and does not establish CUDA acceptance. Root controls GPU launch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import torch
import round11_joint_value_v1 as v1
import round11_joint_value_v2 as v2

FROZEN_V1_SHA = {
    "round11_joint_value_v1.py": "6d76a770163d17574f5f15fa888886edc5cd1e12c6e153d610a856a78c95378b",
    "round11_value_bank_v1.py": "59088f311564ccd269b6b7d6cf75398d90141c9eb6633e99958588962cfa6dfa",
}


def require(condition, message):
    if not condition:
        raise ValueError("ROUND11_NUMERICAL_CUDA_V2: " + message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_bits(left, right):
    return torch.equal(left.detach().contiguous().cpu().view(torch.uint8),
                       right.detach().contiguous().cpu().view(torch.uint8))


def neighbor_checks(device):
    raw = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = raw.view(torch.float16)
    ref_down = torch.nextafter(values, torch.full_like(values, -float("inf")))
    ref_up = torch.nextafter(values, torch.full_like(values, float("inf")))
    down, up = v2.half_neighbors(values.to(device))
    require(same_bits(down, ref_down) and same_bits(up, ref_up), "65536-pattern Half neighbor byte mismatch")
    # Preserve the entire original candidate construction on CPU, including
    # NaN quiet payloads, and the actual admitted nonnegative domain on CUDA.
    all_proposals = values.double()[None]
    old = torch.zeros(1, dtype=torch.float16)
    require(same_bits(v2.half_candidates(all_proposals, old), v1.half_candidates(all_proposals, old)), "full CPU candidate byte mismatch")
    admitted = torch.tensor([[0., 2. ** -24, 2. ** -14, 1., 65504., 65520., float("inf")]], dtype=torch.float64)
    expected = v1.half_candidates(admitted, old)
    actual = v2.half_candidates(admitted.to(device), old.to(device))
    require(same_bits(actual, expected), "admitted CUDA candidate byte mismatch")
    edge_raw = torch.tensor([0, 32768, 1, 1023, 1024, 31743, 31744, 64512, 31745, 32767], dtype=torch.int32)
    return dict(all65536_patterns_both_directions_bit_exact=True,
                signaling_NaN_quiet_payloads_preserved=True,
                all196609_CPU_candidate_elements_bit_exact=True,
                admitted_candidate_bytes_match_v1_CPU=True,
                edge_input_bits=edge_raw.tolist(),
                edge_down_bits=(ref_down[edge_raw.long()].view(torch.int16).int() & 65535).tolist(),
                edge_up_bits=(ref_up[edge_raw.long()].view(torch.int16).int() & 65535).tolist(),
                unsupported_Half_CUDA_nextafter_calls=0)


def make_cases():
    generator = torch.Generator().manual_seed(1102001)
    n, d = 32, 128
    cases = []
    l0 = (torch.randn(d, generator=generator) * .3 - .5).half()
    l1 = (l0.float() + torch.rand(d, generator=generator) + .2).half()
    x = torch.randn((n, d), generator=generator).half()
    bits = torch.rand((n, d), generator=generator) > .5
    error = x.float() - torch.where(bits, l1, l0).float()
    cases.append(("normal", x, l0, l1, bits, error >= 0, error.abs().mean(-1).half()))
    x = (((torch.arange(n * d).reshape(n, d) % 17) - 8).float() * (2. ** -24)).half()
    cases.append(("subnormal_alpha", x, torch.zeros(d).half(), torch.full((d,), 2. ** -14).half(),
                  torch.zeros((n, d), dtype=torch.bool), x >= 0, torch.zeros(n).half()))
    cases.append(("maxfinite_cancellation_overflow_rejection", torch.zeros((n, d)).half(),
                  torch.full((d,), 65504.).half(), torch.full((d,), -65504.).half(),
                  torch.zeros((n, d), dtype=torch.bool), torch.zeros((n, d), dtype=torch.bool), torch.full((n,), 65504.).half()))
    bits = torch.zeros((n, d), dtype=torch.bool)
    bits[4] = True; bits[4, 0::2] = True
    bits[6, 1::2] = True; bits[21, 1::2] = True
    x = bits.half(); x[5, 0::2] = .125; x[5, 1::2] = .5
    cases.append(("spatial_negative_zero_new_point", x, torch.zeros(d).half(), torch.ones(d).half(),
                  bits, torch.ones_like(bits), torch.zeros(n).half()))
    return cases


def encoder_checks(device):
    records = []
    for name, x, l0, l1, bits, signs, alpha in make_cases():
        cpu_fields = v1.joint_encode(x, l0, l1, bits, signs, alpha)
        target = [value.to(device) for value in (x, l0, l1, bits, signs, alpha)]
        fields = v2.joint_encode(*target)
        require(all(same_bits(left, right) for left, right in zip(fields[:3], cpu_fields[:3])), "JointV field mismatch: " + name)
        require(fields[3]["after_sse"] <= fields[3]["before_sse"], "JointV actual SSE increase")
        item = dict(name=name, joint_fields_bit_exact_to_v1_CPU=True,
                    joint_before_sse=fields[3]["before_sse"], joint_after_sse=fields[3]["after_sse"])
        for shuffled in (False, True):
            reference = v1.spatial_encode(x, l0, l1, bits, signs, alpha, shuffled=shuffled)
            actual = v2.spatial_encode(*target, shuffled=shuffled)
            require(all(same_bits(left, right) for left, right in zip(actual[:2], reference[:2])), "Spatial field mismatch: " + name)
            require(actual[2]["after_sse"] <= actual[2]["before_sse"], "Spatial actual SSE increase")
            key = "shuffled" if shuffled else "spatial"
            item[key] = dict(fields_bit_exact_to_v1_CPU=True, before_sse=actual[2]["before_sse"], after_sse=actual[2]["after_sse"], mode1_tokens=actual[2]["mode1_tokens"])
            if name == "spatial_negative_zero_new_point" and not shuffled:
                require(int(actual[1].cpu().view(torch.int16)[5]) == -32768, "negative-zero mode lost")
        # Match the actual composition order: Joint -> final frozen B -> space.
        reference = v1.spatial_encode(x, l0, l1, cpu_fields[0], cpu_fields[1], cpu_fields[2])
        actual = v2.spatial_encode(target[0], target[1], target[2], fields[0], fields[1], fields[2])
        require(all(same_bits(left, right) for left, right in zip(actual[:2], reference[:2])), "JointSpatial field mismatch: " + name)
        item["JointSpatial_fields_bit_exact_to_v1_CPU"] = True
        records.append(item)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "--output-root", dest="output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.perf_counter()
    root = Path(__file__).resolve().parent
    for name, digest in FROZEN_V1_SHA.items():
        require(sha(root / name) == digest, "frozen v1 source changed: " + name)
    device = torch.device("cpu" if args.cpu_only else args.device)
    require(device.type == ("cpu" if args.cpu_only else "cuda"), "actual CUDA is required without --cpu-only")
    if not args.cpu_only:
        require(torch.cuda.is_available(), "CUDA unavailable; do not substitute a CPU pass")
    output = args.output.resolve()
    allowed = str(output).replace("\\", "/")
    require((allowed.startswith("${EVIDENCE_ROOT}/results/round11/") or
             (allowed.startswith("/media/") and "/baseline_matrix_v3/results/round11_v1/" in allowed)), "output must use project data disk")
    report = dict(schema="trajkv.round11.numerical_cuda.v2", status="PASS_CPU_OPERATOR_EQUIVALENCE_ONLY" if args.cpu_only else "PASS_ACTUAL_CUDA_NUMERICAL_V2",
                  actual_cuda_executed=not args.cpu_only, operator_change="equivalent FP16 bit-neighbor replacement only; algorithm/candidate set/format/budget unchanged",
                  neighbor_checks=neighbor_checks(device), encoder_checks=encoder_checks(device), torch_version=torch.__version__,
                  no_model_calls=True, no_task_quality_evidence=True, frozen_v1_sha256=FROZEN_V1_SHA,
                  source_sha256={name: sha(root / name) for name in ("round11_joint_value_v2.py", "round11_value_bank_v2.py", "round11_numerical_cuda_v2.py")})
    if not args.cpu_only:
        torch.cuda.synchronize(device)
        report["GPU_name"] = torch.cuda.get_device_name(device)
        report["CUDA_version"] = torch.version.cuda
    require(all(sha(root / name) == digest for name, digest in FROZEN_V1_SHA.items()), "frozen v1 changed during check")
    report["elapsed_seconds"] = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    path = output / "NUMERICAL_CUDA_V2.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=report["status"], actual_cuda_executed=report["actual_cuda_executed"], output=str(path), sha256=sha(path))), flush=True)


if __name__ == "__main__":
    main()
