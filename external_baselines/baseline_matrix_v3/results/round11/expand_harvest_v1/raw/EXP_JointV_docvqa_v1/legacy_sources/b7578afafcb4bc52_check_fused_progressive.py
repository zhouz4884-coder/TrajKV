"""Validate the local 1-bit / independently refined P202 kernel extension.

Default: CPU packing/index checks and source-diff audit only; no GPU is used.
Explicit --run-gpu executes small synthetic fixtures through the official
generic encoder, the new fused reader, and dense same-representation attention.
No benchmark/heldout data, model weights, or answer labels are loaded.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT.parent / "VisContract_P14_2/scripts"
LOCAL_SOURCE = Path(__file__).with_name("fused_progressive.py")
COMBINATIONS = [(b, k, v) for b in (1, 2) for k, v in itertools.product((False, True), repeat=2)] + [(4, False, False)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference_helpers():
    """Run only two pure-Torch functions; do not import Triton on CPU."""
    tree = ast.parse(LOCAL_SOURCE.read_text(encoding="utf-8"))
    names = {"unpack_last_generic", "decode_last_generic"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {n.name for n in selected} != names:
        raise RuntimeError("Generic reference helper missing")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(LOCAL_SOURCE), "exec"), namespace)
    return namespace["unpack_last_generic"], namespace["decode_last_generic"]


class SplitRefinement(ast.NodeTransformer):
    """Describe the only intended kernel arithmetic change structurally."""
    def visit_If(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Name) and node.test.id == "REFINE":
            # Original K-r2 path contains nine statements, then V-sign starts.
            cut = next((i for i, stmt in enumerate(node.body)
                        if ast.unparse(stmt).startswith("ix = (owner * NVFULL")), None)
            if cut != 9 or node.orelse:
                raise RuntimeError("Legacy residual layout changed; manual audit required")
            return [ast.If(test=ast.Name(id="REFINE_K", ctx=ast.Load()), body=node.body[:cut], orelse=[]),
                    ast.If(test=ast.Name(id="REFINE_V", ctx=ast.Load()), body=node.body[cut:], orelse=[])]
        return node


def source_audit(legacy_source):
    old = legacy_source.read_text(encoding="utf-8")
    new = LOCAL_SOURCE.read_text(encoding="utf-8")
    old_tree, new_tree = ast.parse(old), ast.parse(new)
    old_kernel = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_read")
    new_kernel = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_read")
    for index, arg in enumerate(old_kernel.args.args):
        if arg.arg == "REFINE":
            arg.arg = "REFINE_K"
            old_kernel.args.args.insert(index + 1, ast.arg(arg="REFINE_V", annotation=arg.annotation))
            break
    expected = SplitRefinement().visit(old_kernel)
    kernel_match = ast.dump(expected, include_attributes=False) == ast.dump(new_kernel, include_attributes=False)
    if not kernel_match:
        raise RuntimeError("Kernel differs beyond independent K/V refinement constexpr split")
    diff = "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                     fromfile="legacy/p202_fused.py", tofile="local/fused_progressive.py"))
    return {"legacy_sha256": digest(legacy_source), "local_sha256": digest(LOCAL_SOURCE),
            "kernel_arithmetic_only_changed_by_refinement_split": kernel_match,
            "diff": diff,
            "index_reasoning": [
                "base words hold 32//BITS scalars; BITS=1 gives 32 per int32 word",
                "K base offset = owner*D*(NVFULL/pack) + channel*(NVFULL/pack) + token/pack",
                "V base offset = owner*NVFULL*(D/pack) + token*(D/pack) + channel/pack",
                "K residual planes retain owner-major [KV-head,D,NVFULL] flattening",
                "V residual sign retains owner-major [KV-head,NVFULL,D] flattening",
                "original visual/text positions, GQA owner mapping and causal mask are unchanged",
                "batch remains restricted to one; visual count/head_dim must satisfy g128 constraints"]}


def pack_words(codes, bits):
    pack = 32 // bits
    if codes.shape[-1] % pack:
        raise ValueError("Incomplete word")
    shifts = torch.arange(pack, dtype=torch.int64, device=codes.device) * bits
    grouped = codes.to(torch.int64).reshape(*codes.shape[:-1], -1, pack)
    return (grouped << shifts).sum(-1).to(torch.int32)


def pack_planes(codes, bits):
    flat = codes.flatten().byte()
    if flat.numel() % 8:
        raise ValueError("Incomplete residual bitplane byte")
    shifts = torch.arange(8, device=codes.device)
    return torch.stack([((((flat >> b) & 1).reshape(-1, 8).long() << shifts).sum(-1)).byte()
                        for b in range(bits)])


def unpack_planes(packed, bits, shape):
    shifts = torch.arange(8, device=packed.device)
    flat = sum((((packed[b, :, None].long() >> shifts) & 1).flatten() << b) for b in range(bits))
    return flat.reshape(shape).float()


def residual_r2(error):
    group = error.float().reshape(*error.shape[:-1], -1, 128)
    minimum = group.amin(-1).half()
    scale = ((group.amax(-1) - minimum.float()) / 3).half()
    codes = ((group - minimum.float()[..., None]) /
             scale.float().clamp_min(1e-30)[..., None]).round().clamp(0, 3).byte()
    return pack_planes(codes, 2), scale, minimum


def cpu_checks():
    unpack, decode = load_reference_helpers()
    generator = torch.Generator(device="cpu").manual_seed(20261112)
    checks = []
    for bits in (1, 2, 4):
        codes = torch.randint(0, 1 << bits, (1, 2, 128, 256), generator=generator, dtype=torch.int32)
        # Exercise bit31 and sign extension deterministically.
        codes[..., :32 // bits] = (1 << bits) - 1
        packed = pack_words(codes, bits)
        assert (packed[..., 0] == -1).all()
        restored = unpack(packed, bits)
        assert torch.equal(restored.int(), codes)
        scale = torch.full((1, 2, 128, 2), .125, dtype=torch.float16)
        minimum = torch.full_like(scale, -.5)
        expected = (codes.reshape(1, 2, 128, 2, 128).half() * scale[..., None] + minimum[..., None]).reshape(codes.shape)
        assert torch.equal(decode(packed, scale, minimum, 128, bits), expected)
        checks.append({"bits": bits, "signed_word_roundtrip": True,
                       "group_metadata_decode": True, "negative_word_exercised": True})
    for bits in (1, 2):
        codes = torch.randint(0, 1 << bits, (1, 2, 128, 256), generator=generator, dtype=torch.uint8)
        assert torch.equal(unpack_planes(pack_planes(codes, bits), bits, codes.shape).byte(), codes)
    return {"status": "PASS_CPU_PACKING_AND_METADATA", "base_width_checks": checks,
            "residual_bitplane_roundtrip": True, "gpu_used": False}


def dense_attention(query, visual_k, visual_v, state, selected):
    import torch.nn.functional as F
    if selected is None:
        positions = state.visual_pos[None, None].expand(*visual_k.shape[:2], -1)
    else:
        gather = selected[..., None].expand(*selected.shape, visual_k.shape[-1])
        visual_k, visual_v = visual_k.gather(2, gather), visual_v.gather(2, gather)
        positions = state.visual_pos[selected]
    keys = torch.cat((visual_k, state.text_k[:, :, :state.text_length]), 2)
    values = torch.cat((visual_v, state.text_v[:, :, :state.text_length]), 2)
    text_pos = state.text_pos[:state.text_length][None, None].expand(*visual_k.shape[:2], -1)
    positions = torch.cat((positions, text_pos), -1)
    groups = query.shape[1] // keys.shape[1]
    keys = keys.repeat_interleave(groups, 1)
    values = values.repeat_interleave(groups, 1)
    positions = positions.repeat_interleave(groups, 1)
    qpos = torch.arange(state.length - query.shape[-2], state.length, device=query.device)
    mask = positions[:, :, None] <= qpos[None, None, :, None]
    out = F.scaled_dot_product_attention(query, keys, values, mask, dropout_p=0., is_causal=False)
    scores = query.float() @ keys.float().transpose(-1, -2) * query.shape[-1] ** -.5
    norm = scores.masked_fill(~mask, -torch.inf).logsumexp(-1)
    return out, norm


def errors(actual, expected):
    delta = actual.float() - expected.float()
    return {"max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / expected.float().norm().clamp_min(1e-12))}


def gpu_checks(args):
    sys.path.insert(0, str(args.legacy_scripts))
    from p176_kivi_codec import load_official, decode_last
    from p202_fused import attention as legacy_attention
    from fused_progressive import attention, decode_last_generic
    spec = importlib.util.spec_from_file_location("_trajkv_canonical_bytes", ROOT / "code/bytes.py")
    byte_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(byte_module)
    codec = load_official(args.kivi_source)
    device = torch.device(args.device)
    torch.manual_seed(20261112)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for nv, nq in ((128, 1), (128, 17), (256, 33)):
        hkv, hq, d = 2, 8, 128
        total = 2 * nv + nq
        vp = torch.arange(0, 2 * nv, 2, device=device)
        tp = torch.cat((torch.arange(1, 2 * nv, 2, device=device), torch.arange(2 * nv, total, device=device)))
        vk = torch.randn(1, hkv, nv, d, device=device, dtype=torch.float16)
        vv = torch.randn_like(vk)
        query = torch.randn(1, hq, nq, d, device=device, dtype=torch.float16)
        tk = torch.randn(1, hkv, tp.numel(), d, device=device, dtype=torch.float16)
        tv = torch.randn_like(tk)
        for bits in (1, 2, 4):
            encoded = []
            for data in (vk.transpose(2, 3).contiguous(), vv):
                encoded.append(codec.triton_quantize_and_pack_along_last_dim(data, 128, bits))
            state = SimpleNamespace(visual_pos=vp, text_pos=tp, text_k=tk, text_v=tv,
                                    text_length=tp.numel(), length=total)
            for axis, record in zip(("k", "v"), encoded):
                for suffix, tensor in zip(("code", "scale", "min"), record):
                    setattr(state, axis + "_" + suffix, tensor)
            base_k = decode_last_generic(*encoded[0], group_size=128, bits=bits)
            base_v = decode_last_generic(*encoded[1], group_size=128, bits=bits)
            old_decode_parity = None
            if bits in (2, 4):
                old_decode_parity = all(torch.equal(x, decode_last(codec, *record, 128, bits))
                                        for x, record in zip((base_k, base_v), encoded))
                if not old_decode_parity:
                    raise RuntimeError("Generic decoder disagrees with historical 2/4-bit reference")
            refinement = None
            if bits in (1, 2):
                kr, ks, km = residual_r2(vk.transpose(2, 3).float() - base_k.float())
                verr = (vv.float() - base_v.float()).reshape(1, hkv, nv, -1, 128)
                va = verr.abs().mean(-1).half()
                vr = pack_planes((verr >= 0).byte(), 1)
                refinement = [kr, ks, km, vr, va, va]
            base_storage = [t for record in encoded for t in record]
            stored = base_storage + (refinement or [])
            for _, rk, rv in (c for c in COMBINATIONS if c[0] == bits):
                dense_k, dense_v = base_k.clone(), base_v.clone()
                if rk:
                    codes = unpack_planes(kr, 2, (1, hkv, d, nv))
                    residual = (codes.reshape(1, hkv, d, -1, 128) * ks.float()[..., None] + km.float()[..., None]).reshape(dense_k.shape)
                    dense_k = (dense_k.float() + residual).half()
                if rv:
                    signs = unpack_planes(vr, 1, (1, hkv, nv, d)).reshape(1, hkv, nv, -1, 128)
                    residual = ((2 * signs - 1) * va.float()[..., None]).reshape(dense_v.shape)
                    dense_v = (dense_v.float() + residual).half()
                dense_k = dense_k.transpose(2, 3).contiguous()
                for selected_name in ("all", "alternating_per_kv_head"):
                    selected = None
                    if selected_name != "all":
                        selected = torch.stack((torch.arange(0, nv, 2, device=device),
                                                torch.arange(1, nv, 2, device=device)))[None]
                    actual, norm = attention(query, state, selected=selected, refinement=refinement,
                                             bits=bits, observe=True, refine_k=rk, refine_v=rv)
                    expected, norm_ref = dense_attention(query, dense_k, dense_v, state, selected)
                    original, _ = dense_attention(query, vk, vv, state, selected)
                    torch.cuda.synchronize(device)
                    finite = bool(torch.isfinite(actual).all() and torch.isfinite(norm).all())
                    close = finite and torch.allclose(actual, expected, atol=args.atol, rtol=args.rtol)
                    legacy_error = None
                    if (bits == 2 and rk == rv) or bits == 4:
                        legacy = legacy_attention(query, state, selected=selected,
                                                  refinement=refinement if rk else None, bits=bits)
                        legacy_error = errors(actual, legacy)
                    results.append({"bits": bits, "refine_k": rk, "refine_v": rv,
                                    "visual_tokens": nv, "query_tokens": nq, "kv_heads": hkv,
                                    "query_heads": hq, "selection": selected_name,
                                    "same_representation_error": errors(actual, expected),
                                    "logsumexp_error": errors(norm, norm_ref),
                                    "vs_original_fp16_error": errors(actual, original),
                                    "legacy_kernel_error": legacy_error,
                                    "generic_vs_legacy_decode_exact": old_decode_parity,
                                    "base_storage_bytes": byte_module.unique_storage_bytes(base_storage),
                                    "all_refinement_storage_bytes": byte_module.unique_storage_bytes(refinement or []),
                                    "fixed_visual_storage_bytes": byte_module.unique_storage_bytes(stored),
                                    "finite": finite, "numerical_tolerance_pass": bool(close)})
    return {"status": "PASS_GPU_NUMERICAL_CHECK" if all(r["numerical_tolerance_pass"] for r in results) else "FAIL_GPU_NUMERICAL_CHECK",
            "gpu_used": True, "device": str(device), "gpu": torch.cuda.get_device_name(device),
            "atol": args.atol, "rtol": args.rtol, "results": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-scripts", type=Path, default=LEGACY)
    p.add_argument("--kivi-source", type=Path, default=LEGACY.parent / "third_party/KIVI/quant/new_pack.py")
    p.add_argument("--run-gpu", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--atol", type=float, default=2e-2)
    p.add_argument("--rtol", type=float, default=2e-2)
    p.add_argument("--output", type=Path, default=ROOT / "audit/onebit_extension.json")
    args = p.parse_args()
    if args.atol <= 0 or args.rtol <= 0:
        p.error("tolerances must be positive and declared before observing errors")
    old_source = args.legacy_scripts / "p202_fused.py"
    source_before = digest(old_source)
    report = {"label": "TrajKV local 1-bit progressive extension; not official KIVI",
              "evidence_scope": "synthetic_correctness_only_no_model_or_heldout_data",
              "source_audit": source_audit(old_source), "cpu": cpu_checks(),
              "torch": torch.__version__, "official_encoder_sha256": digest(args.kivi_source),
              "storage_policy": "base plus both refinements fixed across K/V read ablations",
              "planned_gpu_cases": 54,
              "not_established": ["model_quality_gain", "end_to_end_speedup", "physical_dram_traffic"]}
    if args.run_gpu:
        try:
            with torch.inference_mode():
                report["gpu"] = gpu_checks(args)
        except Exception as exc:
            report["gpu"] = {"status": "ERROR_GPU_CHECK", "gpu_used": True,
                             "exception_type": type(exc).__name__, "message": str(exc),
                             "quality_conclusion": "none; resolve correctness failure first"}
        report["status"] = report["gpu"]["status"]
    else:
        report["gpu"] = {"status": "NOT_RUN_WAITING_FOR_GPU_SCHEDULING", "gpu_used": False}
        report["status"] = "READY_FOR_GPU_CHECK"
    matrix = []
    for bits, rk, rv in COMBINATIONS:
        matched = [r for r in report["gpu"].get("results", [])
                   if (r["bits"], r["refine_k"], r["refine_v"]) == (bits, rk, rv)]
        matrix.append({"bits": bits, "refine_k": rk, "refine_v": rv,
                       "gpu_cases_measured": len(matched),
                       "max_abs_error": max((r["same_representation_error"]["max_abs"] for r in matched), default=None),
                       "max_relative_l2": max((r["same_representation_error"]["relative_l2"] for r in matched), default=None),
                       "all_pass": all(r["numerical_tolerance_pass"] for r in matched) if matched else None})
    report["error_matrix"] = matrix
    report["old_source_unchanged"] = digest(old_source) == source_before
    assert report["old_source_unchanged"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "old_source_unchanged": report["old_source_unchanged"],
                      "error_matrix": matrix}, indent=2))
    if args.run_gpu and report["status"] != "PASS_GPU_NUMERICAL_CHECK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
