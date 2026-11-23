"""Round9 B1 V-only transforms on the immutable original R7 whole-model bank.

The frozen builder owns source/asset admission and the original 43-stage plan.
This module changes no K data, reader, residual allocation or persistent format.
Dense source/reconstruction tensors below are encoding workspace, never owner state.
"""
from __future__ import annotations

from copy import copy
import hashlib
import importlib
import json
from pathlib import Path
import time

import torch

import r7_legacy_bridge_v1 as bridge

MODES = ("original", "M-V", "V-FZ", "V-RE", "FB0", "FB1")
THETA_SCHEMA = "trajkv.round9.value_theta.v1"
BASE_FACTOR_SHA = bridge._DEFAULT_ASSETS["metadata_factor_path"]


def require(value, message):
    if not value:
        raise ValueError("ROUND9_VALUE: " + message)


def load_value_theta(path, expected_sha256=None, *, device="cpu"):
    """Read an absolute 224-parameter V table, not an update to old factors."""
    require(path is not None and expected_sha256 is not None, "theta file requires exact SHA")
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest == expected_sha256, "theta SHA differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("schema") == THETA_SCHEMA, "theta schema")
    require(value.get("base_profile") == "r7_joint_k1_gt_cond", "theta base profile")
    require(value.get("base_metadata_factor_sha256") == BASE_FACTOR_SHA, "theta original factor source")
    require(value.get("parameter_count") == 224, "theta must contain exactly V224")
    require("key_head_factors" not in value, "K factors cannot be supplied")
    result = {}
    for name in ("value_head_factors", "value_center_offsets"):
        tensor = torch.tensor(value[name], dtype=torch.float32, device=device)
        require(tuple(tensor.shape) == (28, 4) and bool(torch.isfinite(tensor).all()), "finite theta[28,4]")
        if name.endswith("factors"):
            require(bool((tensor > 0).all()), "positive V factors")
        result[name] = tensor
    return result, {"path": str(path.resolve()), "sha256": digest, "origin": "supplied_absolute_theta",
                    "schema": THETA_SCHEMA, "parameter_count": 224,
                    "fit_provenance": value.get("fit_provenance")}


def apply_theta_to_fields(raw_record, theta, *, layer_idx=0, fold_fn=None):
    """Pure-return frozen fold. raw_record is the *unfolded* (code, scale, min)."""
    if fold_fn is None:
        fold_fn = importlib.import_module("affine_rate_bank").fold
    factor = theta["value_head_factors"][layer_idx][None, :, None, None]
    offset = theta["value_center_offsets"][layer_idx][None, :, None, None]
    return fold_fn(raw_record, 1, factor, offset)


def _unpack(code, length):
    shifts = torch.arange(32, device=code.device, dtype=torch.int64)
    return ((code.long()[..., None] >> shifts) & 1).reshape(*code.shape[:-1], -1)[..., :length]


def _pack(bits):
    require(bits.shape[-1] % 32 == 0, "packing requires a multiple of32")
    shifts = torch.arange(32, device=bits.device, dtype=torch.int64)
    return (bits.long().reshape(*bits.shape[:-1], -1, 32) << shifts).sum(-1).int().contiguous()


def decode_value_fields(state, N=None, *, base_bits_override=None, include_residual=True):
    """Actual sequential FP16 V decode [B,H,N,D], differentiable in scale/min.

    An override can be scalar0/1 or a full [B,H,D,N] bit tensor. Residual signs
    and alpha are fixed inputs; each stage is rounded to Half exactly as deployed.
    """
    N = int(N if N is not None else state.v_code.shape[-1] * 32)
    bits = _unpack(state.v_code, N) if base_bits_override is None else base_bits_override
    if not isinstance(bits, torch.Tensor):
        bits = torch.full((*state.v_code.shape[:-1], N), bits, device=state.v_code.device)
    require(tuple(bits.shape) == (*state.v_code.shape[:-1], N), "base bits shape")
    groups = state.v_scale.shape[-1]
    require(N % groups == 0, "V metadata groups")
    scale = state.v_scale.repeat_interleave(N // groups, dim=-1)
    minimum = state.v_min.repeat_interleave(N // groups, dim=-1)
    value = ((bits.to(torch.float16) * scale).to(torch.float16) + minimum).to(torch.float16)
    value = value.transpose(2, 3).contiguous()
    if include_residual and hasattr(state, "depth_residual_code"):
        # Preserve the per-head sequential FP16 additions, including depth2.
        heads = []
        for head in range(value.shape[1]):
            item = value[:, head]
            for stage in range(2):
                slot = int(state.depth_residual_lookup[1, head, stage])
                if slot >= 0:
                    sign = _unpack(state.depth_residual_code[slot], value.shape[-1]).float() * 2 - 1
                    item = (item.float() + sign[None] * state.depth_residual_alpha[slot].float()[None, :, None]).half()
            heads.append(item)
        value = torch.stack(heads, dim=1)
    return value


def reencode_value_residuals(state, sourceV, selected_stages, layer_idx):
    """Return (newstate, stats), preserving class and every original K slot.

    Hard sign/alpha selection is intentionally no-grad. The new state's scale/min
    objects remain attached, so a training proxy can differentiate their decode.
    """
    selected = [(axis, head, stage) for axis, layer, head, stage in selected_stages if int(layer) == layer_idx]
    for slot, (axis, head, stage) in enumerate(selected):
        require(axis in ("k", "v") and int(state.depth_residual_lookup[0 if axis == "k" else 1, head, stage]) == slot,
                "selected stage list and actual pooled lookup differ")
    v_slots = [(slot, head, stage) for slot, (axis, head, stage) in enumerate(selected) if axis == "v"]
    result = copy(state)
    if not v_slots:
        return result, {"V_slots": 0, "changed_code_elements": 0, "changed_alpha_elements": 0}
    encoder = importlib.import_module("round6_depth_codec").encode_sign_depth
    with torch.no_grad():
        result.depth_residual_code = state.depth_residual_code.detach().clone()
        result.depth_residual_alpha = state.depth_residual_alpha.detach().clone()
        base = decode_value_fields(state, sourceV.shape[-2], include_residual=False)
        for head in sorted({head for _, head, _ in v_slots}):
            entries = [(slot, stage) for slot, h, stage in v_slots if h == head]
            require(sorted(stage for _, stage in entries) == list(range(len(entries))), "V stage prefix")
            codes, alphas, _, _ = encoder(sourceV[0, head], base[0, head], len(entries))
            for slot, stage in entries:
                result.depth_residual_code[slot].copy_(codes[stage])
                result.depth_residual_alpha[slot].copy_(alphas[stage])
        changed_code = sum(int((result.depth_residual_code[s] != state.depth_residual_code[s]).sum()) for s, _, _ in v_slots)
        changed_alpha = sum(int((result.depth_residual_alpha[s] != state.depth_residual_alpha[s]).sum()) for s, _, _ in v_slots)
    return result, {"V_slots": len(v_slots), "changed_code_elements": changed_code,
                    "changed_alpha_elements": changed_alpha, "hard_RE_autograd": "stopped"}


try:
    import triton
    import triton.language as tl
except ImportError:  # CPU inspection and training helpers do not need Triton.
    triton = tl = None

if triton is not None:
    @triton.jit
    def _feedback_kernel(X, A, B, CODES, FINAL, MAXIMUM, BAD, N: tl.constexpr, D: tl.constexpr):
        head = tl.program_id(0)
        d = tl.arange(0, D)
        carry = tl.full((D,), 0, tl.float32)
        maximum = tl.full((D,), 0, tl.float32)
        bad = tl.full((D,), 0, tl.int32)
        for word in range(N // 32):
            packed = tl.full((D,), 0, tl.int32)
            for bit in range(32):
                token = word * 32 + bit
                offset = (head * N + token) * D + d
                u = tl.load(X + offset).to(tl.float32) + carry
                a = tl.load(A + offset).to(tl.float32)
                b = tl.load(B + offset).to(tl.float32)
                choose = tl.abs(u - b) < tl.abs(u - a)  # exact tie -> bit0
                decoded = tl.where(choose, b, a)
                carry = u - decoded
                bad = bad | ((tl.abs(carry) == float("inf")) | (carry != carry)).to(tl.int32)
                maximum = tl.maximum(maximum, tl.abs(carry))
                packed = packed | (choose.to(tl.int32) << bit)
            tl.store(CODES + (head * D + d) * (N // 32) + word, packed)
        tl.store(FINAL + head * D + d, carry)
        tl.store(MAXIMUM + head * D + d, maximum)
        tl.store(BAD + head * D + d, bad)


def feedback_assign(state, sourceV, *, feedback, backend="auto"):
    """Pure-return base-code assignment, fixed actual old-residual decode levels."""
    require(feedback in (0, 1), "only lambda0/lambda1")
    require(sourceV.dtype == torch.float16 and sourceV.shape[0] == 1, "FP16 single image")
    a = decode_value_fields(state, sourceV.shape[-2], base_bits_override=0)
    b = decode_value_fields(state, sourceV.shape[-2], base_bits_override=1)
    require(bool(torch.isfinite(a).all() & torch.isfinite(b).all() & torch.isfinite(sourceV).all()), "finite FB levels/source")
    result = copy(state)
    N, D = sourceV.shape[-2:]
    if feedback == 0:
        choose = (sourceV.float() - b.float()).abs() < (sourceV.float() - a.float()).abs()
        result.v_code = _pack(choose.transpose(2, 3).contiguous())
        final = sourceV[:, :, -1].float() - torch.where(choose, b, a)[:, :, -1].float()
        maximum = (sourceV.float() - torch.where(choose, b, a).float()).abs().amax(dim=2)[0]
        used = "vectorized_torch_FP32_nearest_tie0"
    elif sourceV.is_cuda and backend != "torch":
        require(triton is not None, "FB1 CUDA requires Triton; no implicit slow execution fallback")
        result.v_code = torch.empty_like(state.v_code)
        final = torch.empty((sourceV.shape[1], D), device=sourceV.device, dtype=torch.float32)
        maximum = torch.empty_like(final)
        bad = torch.empty_like(final, dtype=torch.int32)
        _feedback_kernel[(sourceV.shape[1],)](sourceV.contiguous(), a, b, result.v_code, final, maximum, bad, N, D, num_warps=4, enable_fp_fusion=False)
        require(not bool(bad.any()), "FB1 nonfinite carry")
        used = "serial_token_Triton_FP32_carry"
    else:
        choose = torch.empty_like(sourceV, dtype=torch.bool)
        carry = torch.zeros_like(sourceV[:, :, 0], dtype=torch.float32)
        maximum = torch.zeros_like(carry)
        for token in range(N):
            u = sourceV[:, :, token].float() + carry
            bit = (u - b[:, :, token].float()).abs() < (u - a[:, :, token].float()).abs()
            choose[:, :, token] = bit
            carry = u - torch.where(bit, b[:, :, token], a[:, :, token]).float()
            require(bool(torch.isfinite(carry).all()), "FB1 nonfinite carry")
            maximum = torch.maximum(maximum, carry.abs())
        result.v_code = _pack(choose.transpose(2, 3).contiguous())
        final = carry
        used = "serial_token_torch_reference_FP32_carry"
    require(bool(torch.isfinite(final).all()), "finite terminal carry")
    return result, {"lambda": feedback, "tie": "bit0", "clip": False, "backend": used,
                    "carry_dtype": "float32", "carry_terminal_maxabs": float(final.abs().max()),
                    "carry_path_maxabs": float(maximum.max()),
                    "changed_code_words": int((result.v_code != state.v_code).sum()),
                    "encoding_workspace_explicit_bytes": int(a.numel()*a.element_size()+b.numel()*b.element_size()+result.v_code.numel()*4+final.numel()*4+maximum.numel()*4),
                    "workspace_note": "explicit major tensors; allocator peak/time must be measured by executing caller; no deployment field added"}


def _unique(owners):
    result = {}
    for owner in owners:
        for tensor in owner.named_tensors().values():
            if tensor.numel(): result.setdefault(bridge._storage(tensor), tensor.untyped_storage().nbytes())
    return sum(result.values())


@torch.no_grad()
def build_round9_value_owners(native_prefix_cache, visual_positions, variant, *, mode,
                             theta_path=None, theta_sha256=None, legacy_a2,
                             legacy_experiments, legacy_scripts, kivi_source,
                             expected_asset_sha256=None, expected_legacy_source_sha256=None):
    require(mode in MODES, "unknown mode")
    require(theta_path is not None or theta_sha256 is None, "theta SHA without path")
    require(mode not in ("original", "M-V") or theta_path is None, "original/M-V never consume trained theta")
    # The immutable constructor executes the original codec, stage planner, pins
    # and hook/alias checks. This local source list is never attached to owners.
    sources = None if mode == "original" else [layer.values.index_select(2, visual_positions).detach().contiguous() for layer in native_prefix_cache.layers]
    owners, metadata = bridge.build_r7_owners(native_prefix_cache, visual_positions, variant,
        legacy_a2=legacy_a2, legacy_experiments=legacy_experiments, legacy_scripts=legacy_scripts,
        kivi_source=kivi_source, expected_asset_sha256=expected_asset_sha256,
        expected_legacy_source_sha256=expected_legacy_source_sha256)
    require(metadata["selected_stage_count"] == 43, "B1 only original low43 profile")
    before_bytes = _unique(owners)
    shared = owners[0].shared
    if theta_path is None:
        theta = {name: shared["encoder_factor."+name] for name in ("value_head_factors", "value_center_offsets")}
        theta_info = {"origin": "original_frozen_theta", "parameter_count": 224, "sha256": None}
    else:
        theta, theta_info = load_value_theta(theta_path, theta_sha256, device=visual_positions.device)
    stats = []
    if mode != "original":
        a2, experiments, scripts = Path(legacy_a2).resolve(), Path(legacy_experiments).resolve(), Path(legacy_scripts).resolve()
        with bridge._legacy_paths(a2, experiments, scripts):
            quant = importlib.import_module("quantizers")
            fold = importlib.import_module("affine_rate_bank").fold
            for layer, (owner, source) in enumerate(zip(owners, sources)):
                started = time.perf_counter()
                old = owner.state
                state = copy(old)
                item = {"layer": layer}
                if mode == "M-V":
                    reconstruction = decode_value_fields(old, source.shape[-2])
                    delta = source.float().mean(2, keepdim=True) - reconstruction.float().mean(2, keepdim=True)
                    state.v_min = (old.v_min.float() + delta.transpose(2, 3)).half().contiguous()
                    item.update(mean_delta_maxabs=float(delta.abs().max()),
                                actual_mean_error_before_maxabs=float(delta.abs().max()),
                                actual_mean_error_after_maxabs=float((source.float().mean(2)-decode_value_fields(state, source.shape[-2]).float().mean(2)).abs().max()))
                else:
                    raw = quant.binary_centroids(source.transpose(2, 3).contiguous(), group_size=source.shape[-2])
                    require(torch.equal(raw[0], old.v_code), "unfolded original assignment must match frozen bank")
                    _, state.v_scale, state.v_min = apply_theta_to_fields(raw, theta, layer_idx=layer, fold_fn=fold)
                    state.v_code = old.v_code
                    if mode == "V-RE":
                        state, detail = reencode_value_residuals(state, source, metadata["selected_stages"], layer)
                        item.update(detail)
                    elif mode in ("FB0", "FB1"):
                        state, detail = feedback_assign(state, source, feedback=int(mode[-1]))
                        item.update(detail)
                require(bool(torch.isfinite(state.v_scale).all() & torch.isfinite(state.v_min).all()), "nonfinite V metadata")
                # K arrays and selected K residual slots must remain exactly old.
                for name in ("direction_code", "direction_gain", "direction_mean", "depth_residual_lookup", "visual_pos"):
                    require(getattr(state, name) is getattr(old, name), "K/position/lookup object replaced: "+name)
                if hasattr(old, "depth_residual_code"):
                    for slot, (axis, index, head, stage) in enumerate(x for x in metadata["selected_stages"] if x[1] == layer):
                        if axis == "k":
                            require(torch.equal(state.depth_residual_code[slot], old.depth_residual_code[slot]) and torch.equal(state.depth_residual_alpha[slot], old.depth_residual_alpha[slot]), "K residual changed")
                owner.state = state
                item["host_elapsed_seconds_including_synchronizing_stats"] = time.perf_counter()-started
                stats.append(item)
            if mode != "M-V":
                # Replace existing V slots in the shared 336-FP32 factor table.
                for name in ("value_head_factors", "value_center_offsets"):
                    shared["encoder_factor."+name] = theta[name]
    require(_unique(owners) == before_bytes, "persistent unique capacity changed")
    require(all(bridge._sha(path) == item["sha256"] for path, item in metadata["verified_legacy_source_pins"].items()),
            "legacy source changed during candidate transform")
    require(all(bridge._sha(item["path"]) == item["sha256"] for item in metadata["fixed_assets"].values()),
            "fixed asset changed during candidate transform")
    forbidden = {bridge._storage(t) for layer in native_prefix_cache.layers for t in (layer.keys, layer.values)}
    for owner in owners:
        for name, tensor in owner.named_tensors().items():
            require(not tensor.requires_grad and tensor.grad_fn is None, "autograd graph retained")
            require(bridge._storage(tensor) not in forbidden, "native source retained")
    metadata["round9_value_candidate"] = {"schema": "trajkv.round9.value_candidate.v1", "mode": mode,
        "theta": theta_info, "base_original_variant_sha256": metadata["variant_sha256"],
        "fixed_K": True, "original_43_coverage": True, "reader_unchanged": True,
        "persistent_format_unchanged": True, "before_unique_capacity_bytes": before_bytes,
        "after_unique_capacity_bytes": _unique(owners), "layer_stats": stats,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "quality_or_speed_gain_claimed": False}
    metadata["returned_owners_unique_tensor_capacity_bytes"] = _unique(owners)
    return owners, metadata
