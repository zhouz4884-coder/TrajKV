"""Round10 V-only LS applications and actual-level window feedback.

The immutable R7 builder and Round9 numerical helpers are reused. No K fields,
residual coverage or reader are changed. Source/A/B/carry are encoder workspace.
"""
from __future__ import annotations
from copy import copy
import hashlib
import importlib
import json
from pathlib import Path
import time

import torch
import round9_value_bank_v1 as old
import r7_legacy_bridge_v1 as bridge

MODES = ("LS-FZ", "LS-RE", "FB0", "FB-W")
WINDOWS = (1, 32, 64, 128, 256, "global")
THETA_SCHEMA = "trajkv.round10.value_theta.v1"
NUMERICAL_PINS = {
    "round9_value_bank_v1.py": "14d242a6643a738457c9b8593d0f0eb6e383c5e979b2e196fcca28e7d37a5f2e",
    "r7_legacy_bridge_v1.py": "b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b",
}


def require(value, message):
    if not value:
        raise ValueError("ROUND10_VALUE: " + message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_numerical_pins():
    for name, digest in NUMERICAL_PINS.items():
        require(sha(Path(__file__).with_name(name)) == digest, "frozen numerical dependency changed: " + name)


def tensor_sha(value):
    x = value.detach().contiguous().cpu()
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()


def normalize_window(window):
    require((type(window) is int and window in WINDOWS) or window == "global", "window must be1/32/64/128/256/global")
    return window


def _theta(value, *, device):
    require(set(value) >= {"value_head_factors", "value_center_offsets"} and "key_head_factors" not in value, "V224 only")
    result = {}
    for name in ("value_head_factors", "value_center_offsets"):
        t = torch.as_tensor(value[name], dtype=torch.float32, device=device).detach().clone().contiguous()
        require(tuple(t.shape) == (28, 4) and bool(torch.isfinite(t).all()), "finite full theta[28,4]")
        if name == "value_head_factors":
            require(bool((t > 0).all()), "positive gains")
        result[name] = t
    return result


def load_theta(path, expected_sha256, *, device):
    require(isinstance(path, (str, Path)) and isinstance(expected_sha256, str) and len(expected_sha256) == 64, "theta path/SHA pair")
    require(sha(path) == expected_sha256, "theta file SHA changed")
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(value.get("schema") == THETA_SCHEMA and value.get("parameter_count") == 224 and
            value.get("base_profile") == "r7_joint_k1_gt_cond" and
            value.get("base_metadata_factor_sha256") == old.BASE_FACTOR_SHA, "theta schema/base mismatch")
    return _theta(value, device=device), {"origin": "exact_file", "path": str(Path(path).resolve()),
        "sha256": expected_sha256, "schema": THETA_SCHEMA, "parameter_count": 224,
        "fit_provenance": value.get("fit_provenance")}


try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


if triton is not None:
    @triton.jit
    def _window_feedback(X, A, B, CODES, STATS, N: tl.constexpr, D: tl.constexpr,
                         WINDOW: tl.constexpr, NW: tl.constexpr):
        head = tl.program_id(0)
        win = tl.program_id(1)
        d = tl.arange(0, D)
        carry = tl.full((D,), 0, tl.float32)
        maximum = tl.full((D,), 0, tl.float32)
        square = tl.full((D,), 0, tl.float32)
        ratio_max = tl.full((D,), 0, tl.float32)
        ratio_square = tl.full((D,), 0, tl.float32)
        zero_gap = tl.full((D,), 0, tl.float32)
        bad = tl.full((D,), 0, tl.float32)
        overload = tl.full((D,), 0, tl.float32)
        for word in range(WINDOW // 32):
            packed = tl.full((D,), 0, tl.int32)
            for bit in range(32):
                token = win * WINDOW + word * 32 + bit
                offset = (head * N + token) * D + d
                x = tl.load(X + offset).to(tl.float32)
                a = tl.load(A + offset).to(tl.float32)
                b = tl.load(B + offset).to(tl.float32)
                u = x + carry
                choose = tl.abs(u - b) < tl.abs(u - a)
                decoded = tl.where(choose, b, a)
                carry = u - decoded
                gap = tl.abs(b - a)
                ratio = tl.where(gap > 0, tl.abs(carry) / tl.maximum(gap, 1.e-30), 0.)
                maximum = tl.maximum(maximum, tl.abs(carry))
                square += carry * carry
                ratio_max = tl.maximum(ratio_max, ratio)
                ratio_square += ratio * ratio
                zero_gap += (gap == 0).to(tl.float32)
                bad += ((tl.abs(carry) == float("inf")) | (carry != carry)).to(tl.float32)
                overload += ((u < tl.minimum(a, b)) | (u > tl.maximum(a, b))).to(tl.float32)
                packed |= choose.to(tl.int32) << bit
            tl.store(CODES + (head * D + d) * (N // 32) + win * (WINDOW // 32) + word, packed)
        base = ((head * NW + win) * D + d) * 8
        tl.store(STATS + base, carry)
        tl.store(STATS + base + 1, maximum)
        tl.store(STATS + base + 2, square)
        tl.store(STATS + base + 3, ratio_max)
        tl.store(STATS + base + 4, ratio_square)
        tl.store(STATS + base + 5, zero_gap)
        tl.store(STATS + base + 6, bad)
        tl.store(STATS + base + 7, overload)


def _torch_feedback(source, a, b, width):
    """Explicit independent-window CPU/reference recurrence, not a GPU fallback."""
    _, heads, n, d = source.shape
    nw = n // width
    decisions = torch.empty_like(source, dtype=torch.bool)
    stats = torch.zeros((heads, nw, d, 8), device=source.device, dtype=torch.float32)
    for win in range(nw):
        carry = torch.zeros((1, heads, d), device=source.device, dtype=torch.float32)
        for token in range(win * width, (win + 1) * width):
            lo, hi = a[:, :, token].float(), b[:, :, token].float()
            u = source[:, :, token].float() + carry
            decision = (u - hi).abs() < (u - lo).abs()
            carry = u - torch.where(decision, hi, lo)
            gap = (hi - lo).abs()
            ratio = torch.where(gap > 0, carry.abs() / gap.clamp_min(1.e-30), 0.)
            decisions[:, :, token] = decision
            st = stats[:, win]
            st[..., 1] = torch.maximum(st[..., 1], carry[0].abs())
            st[..., 2] += carry[0].square()
            st[..., 3] = torch.maximum(st[..., 3], ratio[0])
            st[..., 4] += ratio[0].square()
            st[..., 5] += (gap[0] == 0).float()
            st[..., 6] += (~torch.isfinite(carry[0])).float()
            st[..., 7] += ((u[0] < torch.minimum(lo[0], hi[0])) | (u[0] > torch.maximum(lo[0], hi[0]))).float()
        stats[:, win, :, 0] = carry[0]
    return old._pack(decisions.transpose(2, 3).contiguous()), stats


@torch.no_grad()
def feedback_assign_window(state, sourceV, *, window, tile_tokens=256, backend="auto"):
    """Return(newstate,JSON stats), fixed final Half levels and old residue.

Finite windows divide each256-token native tile. Global never resets at tiles.
W1 is the exact frozen FB0 operation. CPU is numerical instrumentation only.
"""
    check_numerical_pins()
    window = normalize_window(window)
    require(type(tile_tokens) is int and tile_tokens == 256, "M1 native tile256 only")
    require(sourceV.dtype == torch.float16 and sourceV.ndim == 4 and sourceV.shape[0] == 1 and
            sourceV.shape[-1] == 128 and sourceV.shape[1] == state.v_code.shape[1], "single native Half V[B,H,N,128]")
    n, d = sourceV.shape[-2:]
    require(n >= 256 and n % 256 == 0, "verified complete M1 tiles, no padding")
    width = n if window == "global" else window
    require(n % width == 0, "complete windows")
    a = old.decode_value_fields(state, n, base_bits_override=0)
    b = old.decode_value_fields(state, n, base_bits_override=1)
    require(a.device == b.device == sourceV.device and bool(torch.isfinite(sourceV).all() & torch.isfinite(a).all() & torch.isfinite(b).all()), "finite same-device actual levels")
    before = old.decode_value_fields(state, n)
    result = copy(state)
    nw = n // width
    if width == 1:
        result, _ = old.feedback_assign(state, sourceV, feedback=0)
        selected = old.decode_value_fields(result, n)
        c = sourceV.float() - selected.float()
        gap = (b.float() - a.float()).abs()
        ratio = torch.where(gap > 0, c.abs() / gap.clamp_min(1.e-30), 0.)
        stats = torch.stack((c, c.abs(), c.square(), ratio, ratio.square(), (gap == 0).float(),
                             (~torch.isfinite(c)).float(), ((sourceV < torch.minimum(a,b)) | (sourceV > torch.maximum(a,b))).float()), -1)[0].contiguous()
        # source is [1,H,N,D], and every token is a separate window.
        used = "frozen_FB0_W1_alias"
    elif sourceV.is_cuda and backend != "torch":
        require(triton is not None, "CUDA requires new Triton kernel; no implicit slow fallback")
        result.v_code = torch.empty_like(state.v_code)
        stats = torch.empty((sourceV.shape[1], nw, d, 8), device=sourceV.device, dtype=torch.float32)
        _window_feedback[(sourceV.shape[1], nw)](sourceV.contiguous(), a, b, result.v_code, stats,
            n, d, width, nw, num_warps=4, enable_fp_fusion=False)
        used = "Triton_independent_window_FP32_carry"
    else:
        result.v_code, stats = _torch_feedback(sourceV, a, b, width)
        used = "torch_explicit_window_reference"
    require(not bool(stats[..., 6].any()) and bool(torch.isfinite(stats).all()), "nonfinite carry/statistics; preserve failed run")
    decoded = old.decode_value_fields(result, n)
    bits = old._unpack(result.v_code, n).transpose(2, 3).bool()
    require(torch.equal(decoded, torch.where(bits, b, a)), "packed decode differs from chosen final levels")
    error = (decoded.double() - sourceV.double())[0].reshape(sourceV.shape[1], nw, width, d)
    terminal = stats[..., 0].double()
    closure = error.sum(2) + terminal
    old_error = before.double() - sourceV.double()
    new_error = decoded.double() - sourceV.double()
    # Statistics are encoder work and may synchronize. They are not serving data.
    meta = {"schema": "trajkv.round10.window_feedback.v1", "window": window, "effective_window_tokens": width,
        "window_count": nw, "tile_tokens": tile_tokens, "reset_policy": "image_only" if window == "global" else "each_window_and_native_tile",
        "window_over_tile": width / tile_tokens, "window_over_N": width / n, "backend": used,
        "carry_dtype": "float32", "tie": "bit0", "clip": False, "old_residue_fixed": True, "post_feedback_RE": False,
        "terminal_carry_rms": float(terminal.square().mean().sqrt()), "terminal_carry_maxabs": float(terminal.abs().max()),
        "path_carry_rms": float((stats[...,2].double().sum() / sourceV.numel()).sqrt()),
        "path_carry_maxabs": float(stats[...,1].max()), "max_carry_over_nonzero_gap": float(stats[...,3].max()),
        "nonzero_gap_ratio_rms": float((stats[...,4].double().sum() / max(1,sourceV.numel()-int(stats[...,5].sum()))).sqrt()),
        "zero_gap_elements": int(stats[...,5].sum()), "nonfinite_carry_elements": int(stats[...,6].sum()),
        "outside_level_range_steps": int(stats[...,7].sum()),
        "terminal_window_rms": terminal.square().mean((0,2)).sqrt().cpu().tolist(),
        "terminal_window_maxabs": terminal.abs().amax((0,2)).cpu().tolist(),
        "FP32_closure_maxabs": float(closure.abs().max()), "FP32_closure_rms": float(closure.square().mean().sqrt()),
        "actual_window_error_sum_rms": float(error.sum(2).square().mean().sqrt()),
        "native_element_MSE_before": float(old_error.square().mean()), "native_element_MSE_after": float(new_error.square().mean()),
        "native_element_MSE_increase_count": int((new_error.square() > old_error.square()).sum()),
        "decoded_mean_error_maxabs": float(new_error.mean(2).abs().max()),
        "changed_code_words": int((result.v_code != state.v_code).sum()),
        "changed_base_bits": int((old._unpack(result.v_code,n) != old._unpack(state.v_code,n)).sum()),
        "encoding_workspace_explicit_bytes": a.numel()*a.element_size()+b.numel()*b.element_size()+stats.numel()*stats.element_size()+decoded.numel()*decoded.element_size(),
        "workspace_note": "major explicit buffers only; allocator peak/time must be measured by caller; no owner tensor added",
        "quality_or_speed_gain_claimed": False}
    return result, meta


def _fields(state):
    return {name: tensor_sha(getattr(state,name)) for name in ("v_code","v_scale","v_min","depth_residual_code","depth_residual_alpha")
            if isinstance(getattr(state,name,None),torch.Tensor)}


@torch.no_grad()
def build_round10_value_owners(native_prefix_cache, visual_positions, variant, *, mode,
        theta_path=None, theta_sha256=None, theta_override=None, calibration_only=False,
        window=None, tile_tokens=256, legacy_a2, legacy_experiments, legacy_scripts, kivi_source,
        expected_asset_sha256=None, expected_legacy_source_sha256=None):
    check_numerical_pins()
    require(mode in MODES, "unknown Round10 mode")
    ls = mode in ("LS-FZ", "LS-RE")
    require((theta_override is None or calibration_only) and not(theta_override is not None and theta_path is not None), "in-memory theta only for explicit calibration")
    require(theta_override is None or theta_sha256 is None, "in-memory theta must not claim a file SHA")
    require((ls and window is None) or (not ls and theta_path is None and theta_sha256 is None and theta_override is None), "LS/FB families must not be mixed")
    require(not ls or theta_path is not None or theta_override is not None, "LS needs its exact absolute theta")
    if not ls:
        require(mode != "FB0" or window is None or window == 1, "FB0 cannot silently ignore a different window")
        window = 1 if mode == "FB0" else normalize_window(window)
    sources = [layer.values.index_select(2,visual_positions).detach().contiguous() for layer in native_prefix_cache.layers]
    owners, metadata = bridge.build_r7_owners(native_prefix_cache,visual_positions,variant,
        legacy_a2=legacy_a2,legacy_experiments=legacy_experiments,legacy_scripts=legacy_scripts,kivi_source=kivi_source,
        expected_asset_sha256=expected_asset_sha256,expected_legacy_source_sha256=expected_legacy_source_sha256)
    require(len(owners)==28 and metadata["selected_stage_count"]==43 and
            sum(x[0]=="k" for x in metadata["selected_stages"])==23 and sum(x[0]=="v" for x in metadata["selected_stages"])==20, "original K23/V20 only")
    before_bytes = old._unique(owners)
    if ls:
        if theta_override is not None:
            theta = _theta(theta_override,device=visual_positions.device)
            theta_info = {"origin":"calibration_only_in_memory","schema":THETA_SCHEMA,"parameter_count":224}
        else:
            theta,theta_info = load_theta(theta_path,theta_sha256,device=visual_positions.device)
        theta_info["tensor_sha256"]={name:tensor_sha(value) for name,value in theta.items()}
    else:
        theta = None;theta_info={"origin":"original_V0_direct_physical_fields_no_second_fold"}
    stats=[]
    with bridge._legacy_paths(Path(legacy_a2).resolve(),Path(legacy_experiments).resolve(),Path(legacy_scripts).resolve()):
        quant=importlib.import_module("quantizers");fold=importlib.import_module("affine_rate_bank").fold
        for layer,(owner,source) in enumerate(zip(owners,sources)):
            started=time.perf_counter();previous=owner.state;state=copy(previous)
            item={"layer":layer,"before_V_field_sha256":_fields(previous)}
            if ls:
                raw=quant.binary_centroids(source.transpose(2,3).contiguous(),group_size=source.shape[-2])
                require(torch.equal(raw[0],previous.v_code),"original unfolded binary assignment differs")
                _,state.v_scale,state.v_min=old.apply_theta_to_fields(raw,theta,layer_idx=layer,fold_fn=fold)
                if mode=="LS-RE":
                    state,detail=old.reencode_value_residuals(state,source,metadata["selected_stages"],layer);item.update(detail)
            else:
                state,detail=feedback_assign_window(previous,source,window=window,tile_tokens=tile_tokens);item.update(detail)
            for name in ("direction_code","direction_gain","direction_mean","depth_residual_lookup","visual_pos"):
                require(getattr(state,name) is getattr(previous,name),"K/lookup/positions replaced: "+name)
            for slot,(axis,_,_,_) in enumerate(x for x in metadata["selected_stages"] if x[1]==layer):
                if axis=="k":
                    require(torch.equal(state.depth_residual_code[slot],previous.depth_residual_code[slot]) and torch.equal(state.depth_residual_alpha[slot],previous.depth_residual_alpha[slot]),"K residual changed")
            if mode!="LS-RE":
                require(state.depth_residual_code is previous.depth_residual_code and state.depth_residual_alpha is previous.depth_residual_alpha,"frozen old residual replaced")
            if not ls:
                require(state.v_scale is previous.v_scale and state.v_min is previous.v_min,"FB changed physical old V0")
            require(bool(torch.isfinite(state.v_scale).all() & torch.isfinite(state.v_min).all()),"nonfinite folded V")
            owner.state=state
            item.update(after_V_field_sha256=_fields(state),K_fields_unchanged=True,
                        host_seconds_including_synchronizing_diagnostics=time.perf_counter()-started)
            stats.append(item)
        if ls:
            # Both original f0 and the real new V224 encoding asset are retained.
            owners[0].shared.update({"round10_theta."+name:value for name,value in theta.items()})
    after_bytes=old._unique(owners);delta=896 if ls else 0
    require(after_bytes==before_bytes+delta,"all persistent constants must be charged; expected +896 LS / +0 FB")
    require(all(bridge._sha(path)==item["sha256"] for path,item in metadata["verified_legacy_source_pins"].items()),"legacy changed during transform")
    require(all(bridge._sha(item["path"])==item["sha256"] for item in metadata["fixed_assets"].values()),"base assets changed")
    forbidden={bridge._storage(t) for layer in native_prefix_cache.layers for t in(layer.keys,layer.values)}
    for owner in owners:
        for name,tensor in owner.named_tensors().items():
            require(not tensor.requires_grad and tensor.grad_fn is None,"autograd retained")
            require(bridge._storage(tensor) not in forbidden,"native prefix/source retained: "+name)
    metadata["round10_value_candidate"]={"schema":"trajkv.round10.value_candidate.v1","mode":mode,
        "theta":theta_info,"window":window,"tile_tokens":tile_tokens,"calibration_only":bool(calibration_only),
        "fixed_K":True,"original_K23_V20_coverage":True,"reader_unchanged":True,"old_f0_retained_and_charged":True,
        "new_theta_asset_bytes":delta,"before_unique_capacity_bytes":before_bytes,"after_unique_capacity_bytes":after_bytes,
        "actual_extra_persistent_bytes":after_bytes-before_bytes,"same_packed_payload_layout":True,
        "source_sha256":sha(__file__),"numerical_source_pins":NUMERICAL_PINS,"layer_stats":stats,
        "native_FP16_source_or_feedback_workspace_retained":False,"quality_or_speed_gain_claimed":False}
    metadata["returned_owners_unique_tensor_capacity_bytes"]=after_bytes
    return owners,metadata
