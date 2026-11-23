"""Synthetic QA0 technical checks for Round11 packed attention.

CPU mode never imports Triton, a model, a dataset, or the legacy module. GPU
mode admits explicit frozen legacy source paths and uses synthetic M1 geometry.
Dense K/V below are temporary test references only; they are never owner fields.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import torch


R7_SHA = "b054e8629083b7685ad9e02289b500008b45fbeadbd2bc48455f91d0b6a728a0"
WHT_SHA = "9dc73998f66d5b7bafc96a5aa3db57974d1ca8e604e432e7539625f104d90ee3"
SIZES = (256, 768, 1792, 2304)
FIELDS = ("direction_code", "direction_gain", "direction_mean", "v_code",
          "v_scale", "v_min", "depth_residual_code", "depth_residual_alpha",
          "depth_residual_lookup", "visual_pos", "text_k", "text_v", "text_pos")


def require(value, message):
    if not value:
        raise AssertionError("ROUND11_KERNEL_TECHNICAL: " + message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def fields(state):
    return {name: tensor_sha(getattr(state, name)) for name in FIELDS}


def unpack(words, count):
    shift = torch.arange(32, device=words.device, dtype=torch.int64)
    return ((words.long()[..., None] >> shift) & 1).flatten(-2)[..., :count]


def pack(bits):
    shift = torch.arange(32, device=bits.device, dtype=torch.int64)
    return (bits.long().reshape(*bits.shape[:-1], -1, 32) << shift).sum(-1).int().contiguous()


def neighbor_scalar(i, shuffled):
    """Independent scalar definition of p^-1 N p on each 32-token word."""
    x = (13 * i + 7) % 32 if shuffled else i
    l, r, v = (x if x % 16 == 0 else x - 1), (x if x % 16 == 15 else x + 1), x ^ 16
    return tuple((5 * (a - 7)) % 32 if shuffled else a for a in (l, r, v))


def neighbor_indices(n, shuffled, device):
    result = torch.tensor([neighbor_scalar(i, shuffled) for i in range(32)], device=device)
    base = (torch.arange(n, device=device) // 32) * 32
    local = torch.arange(n, device=device) % 32
    return tuple(base + result[local, j] for j in range(3))


def decode_values_reference(state, *, original=False):
    """Materialize actual sequential Half decode, without reader/helper math."""
    n = state.visual_pos.numel()
    bits = unpack(state.v_code, n)
    scale, minimum = state.v_scale, state.v_min
    base = ((bits.half() * scale).half() + minimum).half().transpose(2, 3).contiguous()
    a0 = ((torch.zeros_like(scale) * scale).half() + minimum).half()
    a1 = ((torch.ones_like(scale) * scale).half() + minimum).half()
    l, r, v = neighbor_indices(n, bool(getattr(state, "round11_shuffled", False)), bits.device)
    numerator = 4 * bits + bits.index_select(-1, l) + bits.index_select(-1, r) + 2 * bits.index_select(-1, v)
    p1 = (((8 - numerator).float() * a0.float() + numerator.float() * a1.float()) / 8).half()
    p1 = p1.transpose(2, 3).contiguous()
    result = []
    for h in range(4):
        value = base[:, h]
        for stage in range(2):
            slot = int(state.depth_residual_lookup[1, h, stage])
            if slot < 0:
                continue
            sign = 2 * unpack(state.depth_residual_code[slot], 128).float() - 1
            alpha = state.depth_residual_alpha[slot]
            if stage == 0 and not original:
                raw = alpha.view(torch.int16).int() & 65535
                mode = (raw & 32768) != 0
                value = torch.where(mode[None, :, None], p1[:, h], value)
                alpha = (raw & 32767).short().view(torch.float16)
            value = (value.float() + sign[None] * alpha.float()[None, :, None]).half()
        result.append(value)
    return torch.stack(result, dim=1)


def decode_keys_reference(state, book):
    index = torch.stack((state.direction_code & 15, state.direction_code >> 4), -1).flatten(-2).long()
    base = (book.float()[index].flatten(-2) * state.direction_gain.float()).half()
    result = []
    for h in range(4):
        value = base[:, h]
        for stage in range(2):
            slot = int(state.depth_residual_lookup[0, h, stage])
            if slot >= 0:
                sign = 2 * unpack(state.depth_residual_code[slot], 128).float() - 1
                value = (value.float() + sign[None] * state.depth_residual_alpha[slot].float()[None, :, None]).half()
        result.append(value)
    return torch.stack(result, dim=1)


def _half(value):
    return struct.unpack("e", struct.pack("e", float(value)))[0]


def scalar_actual_half_check(state, materialized):
    """Scalar reference rounds each operation and bitcasts stored alpha via struct."""
    n = state.visual_pos.numel()
    cpu = SimpleNamespace(**{name: getattr(state, name).detach().cpu() for name in FIELDS})
    output = materialized.detach().cpu()
    checked = 0
    for h in range(4):
        for token in (0, 1, 7, 15, 16, 23, 31, 32, 255, n - 1):
            for c in (0, 1, 3, 31, 63, 95, 127):
                word = int(cpu.v_code[0, h, c, token // 32]) & 0xffffffff
                b = (word >> (token % 32)) & 1
                scale, minimum = float(cpu.v_scale[0, h, c, 0]), float(cpu.v_min[0, h, c, 0])
                a0, a1 = _half(_half(0 * scale) + minimum), _half(_half(1 * scale) + minimum)
                value = _half(_half(b * scale) + minimum)
                for stage in range(2):
                    slot = int(cpu.depth_residual_lookup[1, h, stage])
                    if slot < 0:
                        continue
                    raw = int(cpu.depth_residual_alpha[slot, token].view(torch.int16)) & 65535
                    if stage == 0:
                        if raw & 32768:
                            l, r, v = neighbor_scalar(token % 32, bool(getattr(state, "round11_shuffled", False)))
                            nn = 4 * b + ((word >> l) & 1) + ((word >> r) & 1) + 2 * ((word >> v) & 1)
                            value = _half(((8 - nn) * a0 + nn * a1) / 8)
                        raw &= 32767
                    alpha = struct.unpack("e", struct.pack("H", raw))[0]
                    sign = 2 * ((int(cpu.depth_residual_code[slot, token, c // 32]) >> (c % 32)) & 1) - 1
                    value = _half(value + sign * alpha)
                actual = float(output[0, h, token, c])
                require(struct.pack("e", value) == struct.pack("e", actual), "scalar actualHalf decode differs")
                checked += 1
    return checked


def synthetic(n, flag, *, shuffled=False, device="cpu", nq=19):
    generator = torch.Generator(device="cpu").manual_seed(110000 + n)
    rand = lambda *shape: torch.rand(*shape, generator=generator)
    code = torch.randint(0, 256, (1, 4, n, 16), generator=generator, dtype=torch.uint8)
    bits = torch.randint(0, 2, (1, 4, 128, n), generator=generator)
    # Includes folded endpoint pairs where Half(scale+minimum) differs from float addition.
    scale, minimum = (rand(1, 4, 128, 1) * 1.1 + 0.05).half(), (rand(1, 4, 128, 1) - 0.7).half()
    scale[0, :, 0, 0], minimum[0, :, 0, 0] = 0.10003662109375, 0.60009765625
    if flag == "tie":
        scale.zero_()
    lookup = torch.arange(16, dtype=torch.int32).reshape(2, 4, 2)
    lookup[0, 1, 1] = -1  # one K stage
    lookup[:, 2, :] = -1  # no stages on this owner
    lookup[1, 3, 1] = -1  # one V stage; head0 still exercises two
    rc = pack(torch.randint(0, 2, (16, n, 128), generator=generator))
    alpha = (rand(16, n) * 0.15).half()
    # Negative K and V-stage1 amplitudes prove the mode rule is stage0 V only.
    alpha[0::2].neg_()
    alpha[9::2].neg_()
    for h in range(4):
        slot = int(lookup[1, h, 0])
        if slot < 0:
            continue
        raw = (alpha[slot].abs().view(torch.int16).int() & 32767)
        raw[::13] = 0
        if flag == "flag1":
            raw |= 32768
        elif flag in ("mixed", "tie"):
            raw[1::2] |= 32768
            raw[0] |= 32768  # mandatory negative-zero case
        alpha[slot] = raw.short().view(torch.float16)
    nt = 37
    state = SimpleNamespace(direction_code=code, direction_gain=(rand(1, 4, n, 1) + 0.4).half(),
        direction_mean=((rand(1, 4, 1, 128) - 0.5) * 0.2).half(),
        v_code=pack(bits), v_scale=scale, v_min=minimum, depth_residual_code=rc,
        depth_residual_alpha=alpha, depth_residual_lookup=lookup,
        visual_pos=torch.arange(4, n + 4),
        text_k=((rand(1, 4, nt, 128) - 0.5) * 0.9).half(),
        text_v=((rand(1, 4, nt, 128) - 0.5) * 2.5).half(),
        text_pos=torch.cat((torch.arange(4), torch.arange(n + 4, n + nt))),
        text_length=nt, length=n + nt, round11_shuffled=shuffled)
    q = ((rand(1, 28, nq, 128) - 0.5) * 1.3).half()
    book = ((rand(16, 4) - 0.5) * 1.8).half()
    for name in FIELDS:
        setattr(state, name, getattr(state, name).to(device).contiguous())
    return state, q.to(device), book.to(device)


def source_checks(a2):
    reader = Path(__file__).with_name("round11_attention_v1.py")
    original = a2 / "round7_key_attention.py"
    require(sha(original) == R7_SHA and sha(a2 / "fast_wht.py") == WHT_SHA, "frozen R7/WHT source pin")
    old, new = original.read_text(), reader.read_text()
    # Exact copied K segment and common text + single online-softmax segment.
    k0, k1 = "   if KDIM==0:", "   if VB==16:"
    t0, t1 = "   score=tl.dot(qr,k)", "\n\n\ndef attention"
    require(old[old.index(k0):old.index(k1)] == new[new.index(k0):new.index(k1)], "K kernel math changed")
    require(old[old.index(t0):old.index(t1)] == new[new.index(t0):new.index(t1)], "text/causal/online-softmax math changed")
    require("from fast_wht import" not in new and "sys.path" not in new, "reader path mutation or ambient WHT import")
    require("alpha_half.to(tl.uint16,bitcast=True)" in new, "mode must be bitcast before promotion")
    require("alpha_bits & 32767" in new, "alpha magnitude bitmask absent")
    compile(new, str(reader), "exec")
    compile(Path(__file__).read_text(), __file__, "exec")
    return {"reader_sha256": sha(reader), "harness_sha256": sha(__file__),
            "shared_numerical_helper_sha256": sha(Path(__file__).with_name("round11_joint_value_v1.py")),
            "frozen_reader_sha256": sha(original), "frozen_wht_sha256": sha(a2 / "fast_wht.py"),
            "K_kernel_source_identical": True, "text_causal_online_softmax_source_identical": True,
            "reader_has_no_visual_materialization": True, "no_path_mutation": True}


def cpu_checks():
    helper = importlib.import_module("round11_joint_value_v1")
    require((helper.SHUFFLE_A, helper.SHUFFLE_B, helper.SHUFFLE_A_INV) == (13, 7, 5), "shuffle constants")
    maps = []
    for shuffled in (False, True):
        expected = torch.tensor([neighbor_scalar(i, shuffled) for i in range(32)]).T
        actual = torch.stack(helper.word_spatial_index_map(shuffled=shuffled))
        require(torch.equal(actual, expected), "shared shuffle helper differs from scalar conjugation")
        maps.append(expected)
    # Conjugation preserves self-loop counts, edge degree and 32-token containment.
    require((maps[0] == torch.arange(32)).sum().item() == (maps[1] == torch.arange(32)).sum().item(), "shuffle self fallback count")
    entries = []
    for n in SIZES:
        for flag in ("flag0", "flag1", "mixed", "tie"):
            for shuffled in (False, True):
                state, q, book = synthetic(n, flag, shuffled=shuffled)
                before = fields(state)
                decoded = decode_values_reference(state)
                checked = scalar_actual_half_check(state, decoded)
                bits = unpack(state.v_code, n).transpose(2, 3).bool()
                for h in range(4):
                    slot = int(state.depth_residual_lookup[1, h, 0])
                    if slot < 0:
                        continue
                    scale, minimum = state.v_scale[0, h, :, 0], state.v_min[0, h, :, 0]
                    l0 = ((torch.zeros_like(scale) * scale).half() + minimum).half()
                    l1 = ((torch.ones_like(scale) * scale).half() + minimum).half()
                    helper_value = helper.decode_fields(bits[0, h], unpack(state.depth_residual_code[slot], 128).bool(),
                        state.depth_residual_alpha[slot], l0, l1, spatial=True, shuffled=shuffled)
                    slot1 = int(state.depth_residual_lookup[1, h, 1])
                    if slot1 >= 0:
                        signs1 = 2 * unpack(state.depth_residual_code[slot1], 128).float() - 1
                        helper_value = (helper_value.float() + signs1 * state.depth_residual_alpha[slot1, :, None].float()).half()
                    require(torch.equal(helper_value.view(torch.int16), decoded[0, h].view(torch.int16)), "factory numerical helper actualHalf contract differs")
                if flag == "flag0":
                    require(torch.equal(decoded.view(torch.int16), decode_values_reference(state, original=True).view(torch.int16)), "flag0 old decode degeneration")
                require(before == fields(state), "CPU reference mutated source fields")
                require(torch.isfinite(decoded).all().item(), "nonfinite reference")
                entries.append({"N": n, "flag": flag, "shuffled": shuffled,
                                "scalar_half_checks": checked, "decoded_sha256": tensor_sha(decoded),
                                "field_hashes_unchanged": True})
    return {"cases": entries, "shared_map_matches_independent_scalar": True,
            "factory_numerical_helper_matches_independent_decode": True,
            "zero_one_two_residual_stages_tested": True,
            "negative_zero_tested": True, "flag0_exact_old_decode": True,
            "scalar_checks": sum(x["scalar_half_checks"] for x in entries)}


def dense_reference(q, state, book, wht, visual_bias=None, *, blockwise=False):
    key, value = decode_keys_reference(state, book), decode_values_reference(state)
    qr = wht(q, half=True).float()
    bias = (q.float() * state.direction_mean.repeat_interleave(7, 1).float()).sum(-1).div(128 ** .5)
    if visual_bias is not None:
        bias += visual_bias.repeat_interleave(7)[None, :, None]
    key, value = key.repeat_interleave(7, 1), value.repeat_interleave(7, 1)
    tk, tv = state.text_k.repeat_interleave(7, 1), state.text_v.repeat_interleave(7, 1)
    vs = (qr @ key.float().transpose(-1, -2)) * (128 ** -.5) + bias[..., None]
    ts = (q.float() @ tk.float().transpose(-1, -2)) * (128 ** -.5)
    nq = q.shape[-2]
    positions = torch.arange(state.length - nq, state.length, device=q.device)
    vs.masked_fill_(state.visual_pos[None, :] > positions[:, None], -float("inf"))
    ts.masked_fill_(state.text_pos[None, :] > positions[:, None], -float("inf"))
    if not blockwise:
        scores = torch.cat((vs, ts), dim=-1)
        weights = scores.softmax(-1)
        require(torch.isfinite(weights).all().item(), "dense full-history softmax nonfinite")
        return weights @ torch.cat((value, tv), dim=-2).float()
    # Mirrors only the reduction arithmetic after independently materialized K/V.
    # Visual/text blocks remain separate to match the unchanged reader partition.
    acc = torch.zeros_like(q, dtype=torch.float32)
    m = torch.full(q.shape[:-1], -float("inf"), device=q.device)
    denominator = torch.zeros_like(m)
    for scores, values in ((vs, value), (ts, tv)):
        for start in range(0, scores.shape[-1], 32):
            s, v = scores[..., start:start + 32], values[..., start:start + 32, :]
            nm = torch.maximum(m, s.amax(-1))
            safe = torch.where(nm == -float("inf"), 0., nm)
            a, weights = torch.exp(m - safe), torch.exp(s - safe[..., None])
            acc = acc * a[..., None] + weights.half().float() @ v.float()
            denominator = denominator * a + weights.sum(-1)
            m = nm
    return acc / denominator.clamp_min(1e-30)[..., None]


def gpu_checks(a2):
    require(torch.cuda.is_available(), "CUDA GPU required; CPU-only is a separate status")
    require("round7_key_attention" not in sys.modules, "GPU harness must admit the frozen module itself")
    previous, previous_bytecode = list(sys.path), sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(a2))
        old = importlib.import_module("round7_key_attention")
    finally:
        sys.path[:] = previous
        sys.dont_write_bytecode = previous_bytecode
    new = importlib.import_module("round11_attention_v1")
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = []
    for n in SIZES:
        for flag in ("flag0", "flag1", "mixed", "tie"):
            for shuffled in (False, True):
                for nq in (1, 19):
                    state, q, book = synthetic(n, flag, shuffled=shuffled, device="cuda", nq=nq)
                    before = fields(state)
                    visual_bias = torch.tensor([.05, -.04, .03, -.02], device="cuda", dtype=torch.float32)
                    actual = new.attention(q, state, book, n, visual_bias)
                    dense = dense_reference(q, state, book, old.wht, visual_bias)
                    online = dense_reference(q, state, book, old.wht, visual_bias, blockwise=True)
                    require(torch.isfinite(actual).all().item(), "GPU output nonfinite")
                    dense_error = float((actual.float() - dense).abs().max())
                    online_error = float((actual.float() - online).abs().max())
                    require(dense_error <= 0.002 and online_error <= 0.001, "GPU differs from actualHalf materialized reference")
                    bitwise_old = None
                    if flag == "flag0":
                        old_output = old.attention(q, state, book, n, visual_bias)
                        bitwise_old = torch.equal(actual.view(torch.int16), old_output.view(torch.int16))
                        require(bitwise_old, "mode0 reader is not bitwise original")
                    require(before == fields(state), "GPU reader mutated fields")
                    cases.append({"N": n, "flag": flag, "shuffled": shuffled, "nq": nq,
                                  "GQA": 7, "text_tokens": state.text_length,
                                  "max_abs_dense": dense_error, "max_abs_blockwise": online_error,
                                  "flag0_bitwise_old_reader": bitwise_old,
                                  "output_sha256": tensor_sha(actual), "field_hashes_unchanged": True})
    return {"cases": cases, "device": torch.cuda.get_device_name(), "torch": torch.__version__,
            "max_abs_dense": max(x["max_abs_dense"] for x in cases),
            "max_abs_blockwise": max(x["max_abs_blockwise"] for x in cases),
            "GQA7_text_visual_single_softmax_verified": True,
            "temporary_materialized_reference_only": True, "persistent_visual_FP16_bytes": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-a2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists; preserve frozen evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema": "trajkv.round11.kernel_technical.v1", "QA_events": 0,
              "new_free_generation_task_answers": 0, "model_loaded": False,
              "status": "PASS_CPU_SOURCE_ONLY_GPU_PENDING" if args.cpu_only else "PASS_GPU_SYNTHETIC_QA0"}
    result["source_checks"] = source_checks(args.legacy_a2.resolve())
    result["cpu_checks"] = cpu_checks()
    if not args.cpu_only:
        result["gpu_checks"] = gpu_checks(args.legacy_a2.resolve())
    with args.output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": result["status"], "QA_events": 0,
                      "cpu_cases": len(result["cpu_checks"]["cases"]), "output_sha256": sha(args.output)}), flush=True)


if __name__ == "__main__":
    main()
