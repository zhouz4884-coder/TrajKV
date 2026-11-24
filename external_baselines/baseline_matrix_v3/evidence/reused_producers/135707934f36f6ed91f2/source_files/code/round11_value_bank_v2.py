"""Round11 factory v2: v1 contract with CUDA-compatible Half bit neighbors.

Only existing V20 fields are replaced. Encoding workspaces and native sources
are not attached to owners. The complete persistent contract stays B0 exactly.
"""
from __future__ import annotations

from copy import copy
import hashlib
import importlib
from pathlib import Path
import time

import torch
import r7_legacy_bridge_v1 as bridge
import round11_joint_value_v2 as numerical

MODES = ("JointV", "Spatial", "JointSpatial", "Shuffled")
SCHEMA = "trajkv.round11.value_candidate.v2"
NUMERICAL_PINS = {"r7_legacy_bridge_v1.py": "b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b"}


def require(condition, message):
    if not condition:
        raise ValueError("ROUND11_VALUE: " + message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def unique_capacity(owners):
    unique = {}
    for owner in owners:
        for tensor in owner.named_tensors().values():
            if tensor.numel():
                unique.setdefault(bridge._storage(tensor), tensor.untyped_storage().nbytes())
    return sum(unique.values())


def value_endpoints(state, head):
    require(tuple(state.v_scale.shape[-1:]) == tuple(state.v_min.shape[-1:]) == (1,), "original group-all only")
    scale, minimum = state.v_scale[0, head, :, 0], state.v_min[0, head, :, 0]
    zero = (torch.zeros_like(scale) * scale).half()
    one = (torch.ones_like(scale) * scale).half()
    return (zero + minimum).half(), (one + minimum).half()


def _layer_entries(state, selected_stages, layer_idx):
    entries = [tuple(x) for x in selected_stages if int(x[1]) == layer_idx]
    for slot, (axis, _, head, stage) in enumerate(entries):
        require(axis in ("k", "v") and int(stage) == 0, "only original stage0")
        require(int(state.depth_residual_lookup[int(axis == "v"), head, stage]) == slot, "pooled slot/selected list mismatch")
    return entries


@torch.no_grad()
def transform_value_state(state, sourceV, selected_stages, layer_idx, *, mode,
                          encode=True, tile_tokens=128, candidate_chunk=32):
    """Pure-return layer operation, also usable in isolated CPU validation."""
    require(mode in MODES, "unknown mode")
    require(sourceV.dtype == torch.float16 and sourceV.ndim == 4 and sourceV.shape[0] == 1, "native [1,H,N,D] Half")
    require(sourceV.shape[-2] % 32 == 0, "complete packed words")
    entries = _layer_entries(state, selected_stages, layer_idx)
    vslots = [(slot, head) for slot, (axis, _, head, _) in enumerate(entries) if axis == "v"]
    result = copy(state)
    detail = {"layer": layer_idx, "V_slots": len(vslots), "head_stats": [],
              "unchanged_residual_presence": True, "unchanged_uncovered_V": True}
    if mode != "JointV":
        result.round11_shuffled = mode == "Shuffled"
    if not encode or not vslots:
        detail.update(changed_base_words=0, changed_sign_words=0, changed_alpha_bits=0)
        return result, detail
    require(hasattr(state, "depth_residual_code") and hasattr(state, "depth_residual_alpha"), "covered V requires actual pool")
    # Each layer owns a compact pool in the frozen bank. Clone its exact shape;
    # all K slots are copied byte-for-byte, and the replaced pool is not retained.
    require(state.depth_residual_code.is_contiguous() and state.depth_residual_alpha.is_contiguous(), "compact old residual pools")
    result.depth_residual_code = state.depth_residual_code.detach().clone()
    result.depth_residual_alpha = state.depth_residual_alpha.detach().clone()
    if mode in ("JointV", "JointSpatial"):
        result.v_code = state.v_code.detach().clone()
    for slot, head in vslots:
        l0, l1 = value_endpoints(state, head)
        bits = numerical.unpack_bits(state.v_code[0, head], sourceV.shape[-2]).transpose(0, 1).contiguous()
        signs = numerical.unpack_bits(state.depth_residual_code[slot], sourceV.shape[-1])
        alpha = state.depth_residual_alpha[slot]
        require(not bool(numerical.alpha_mode(alpha).any()), "frozen original V alpha signbits")
        item = {"head": head, "slot": slot}
        if mode in ("JointV", "JointSpatial"):
            bits, signs, alpha, item["joint"] = numerical.joint_encode(sourceV[0, head], l0, l1, bits, signs, alpha,
                tile_tokens=tile_tokens, candidate_chunk=candidate_chunk)
            result.v_code[0, head].copy_(numerical.pack_bits(bits.transpose(0, 1).contiguous()))
        if mode != "JointV":
            # This immutable snapshot of the final base is used for every P1.
            signs, alpha, item["spatial"] = numerical.spatial_encode(sourceV[0, head], l0, l1, bits, signs, alpha,
                shuffled=mode == "Shuffled", tile_tokens=tile_tokens, candidate_chunk=candidate_chunk)
        result.depth_residual_code[slot].copy_(numerical.pack_bits(signs))
        result.depth_residual_alpha[slot].copy_(alpha)
        detail["head_stats"].append(item)
    for name in ("direction_code", "direction_gain", "direction_mean", "depth_residual_lookup", "visual_pos", "v_scale", "v_min"):
        require(getattr(result, name) is getattr(state, name), "original field pointer changed: " + name)
    for slot, (axis, _, _, _) in enumerate(entries):
        if axis == "k":
            require(torch.equal(result.depth_residual_code[slot], state.depth_residual_code[slot]) and
                    torch.equal(result.depth_residual_alpha[slot].view(torch.int16), state.depth_residual_alpha[slot].view(torch.int16)), "K slot changed")
    covered = {head for _, head in vslots}
    for head in range(sourceV.shape[1]):
        if head not in covered:
            require(torch.equal(result.v_code[0, head], state.v_code[0, head]), "uncovered V base changed")
    detail.update(changed_base_words=int((result.v_code != state.v_code).sum()),
        changed_sign_words=int((result.depth_residual_code != state.depth_residual_code).sum()),
        changed_alpha_bits=int((result.depth_residual_alpha.view(torch.int16) != state.depth_residual_alpha.view(torch.int16)).sum()))
    return result, detail


@torch.no_grad()
def build_round11_value_owners(native_prefix_cache, visual_positions, variant, *, mode,
        encode=True, tile_tokens=128, candidate_chunk=32, legacy_a2, legacy_experiments,
        legacy_scripts, kivi_source, expected_asset_sha256=None, expected_legacy_source_sha256=None):
    require(mode in MODES and type(encode) is bool, "mode/encode")
    require(tile_tokens == 128 and candidate_chunk == 32, "v1 mechanism fixed token/candidate tiling128/32")
    require(expected_asset_sha256 is None or dict(expected_asset_sha256) == bridge._DEFAULT_ASSETS,
            "Round11 retains the exact original low-profile book/f0/policy assets")
    for name, digest in NUMERICAL_PINS.items():
        require(sha(Path(__file__).with_name(name)) == digest, "frozen dependency changed: " + name)
    owners, metadata = bridge.build_r7_owners(native_prefix_cache, visual_positions, variant,
        legacy_a2=legacy_a2, legacy_experiments=legacy_experiments, legacy_scripts=legacy_scripts,
        kivi_source=kivi_source, expected_asset_sha256=expected_asset_sha256,
        expected_legacy_source_sha256=expected_legacy_source_sha256)
    selected = metadata["selected_stages"]
    require(len(owners) == 28 and len(selected) == metadata["selected_stage_count"] == 43 and
            sum(x[0] == "k" for x in selected) == 23 and sum(x[0] == "v" for x in selected) == 20 and
            all(int(x[3]) == 0 for x in selected), "only original K23/V20 stage0 coverage")
    require(variant.get("name") == "r7_joint_k1_gt_cond" and variant.get("v_group_size", "all") == "all", "original low/group-all profile")
    n = visual_positions.numel()
    before = unique_capacity(owners)
    expected = 4590 * n + 89733
    require(before == expected, "original complete B0 differs from4590*N+89733")
    shared = owners[0].shared
    shared_identity = {name: value for name, value in shared.items()}
    initial_names = [set(owner.named_tensors()) for owner in owners]
    spatial_reader = None
    if mode != "JointV":
        spatial_reader = importlib.import_module("round11_attention_v1").attention
    stats = []
    for layer, owner in enumerate(owners):
        started = time.perf_counter()
        source = native_prefix_cache.layers[layer].values.index_select(2, visual_positions).detach().contiguous()
        previous = owner.state
        transformed, detail = transform_value_state(previous, source, selected, layer, mode=mode,
            encode=encode, tile_tokens=tile_tokens, candidate_chunk=candidate_chunk)
        owner.state = transformed
        if spatial_reader is not None:
            owner._reader = spatial_reader
        require(owner.shared is shared and set(owner.named_tensors()) == initial_names[layer], "owner constants/field names changed")
        require(all(not isinstance(value, torch.Tensor) or not value.numel() or name in bridge._STATE_TENSORS
                    for name, value in vars(transformed).items()), "unreported persistent tensor field")
        for name in ("depth_residual_code", "depth_residual_alpha"):
            require(hasattr(previous, name) == hasattr(transformed, name), "absent residual fields created")
        detail["host_seconds_including_synchronizing_diagnostics"] = time.perf_counter() - started
        stats.append(detail)
        del previous, transformed, source
    require(set(shared) == set(shared_identity) and all(shared[name] is value for name, value in shared_identity.items()), "original shared constants changed")
    after = unique_capacity(owners)
    require(after == before == expected, "complete persistent B0 changed")
    require(all(bridge._sha(path) == item["sha256"] for path, item in metadata["verified_legacy_source_pins"].items()), "legacy source changed during transform")
    require(all(bridge._sha(item["path"]) == item["sha256"] for item in metadata["fixed_assets"].values()), "fixed asset changed during transform")
    forbidden = {bridge._storage(tensor) for layer in native_prefix_cache.layers for tensor in (layer.keys, layer.values)}
    for owner in owners:
        for name, tensor in owner.named_tensors().items():
            require(not tensor.requires_grad and tensor.grad_fn is None, "autograd retained: " + name)
            require(bridge._storage(tensor) not in forbidden, "native FP16 prefix retained: " + name)
    metadata["round11_value_candidate"] = {"schema": SCHEMA, "mode": mode, "encoding_applied": encode,
        "control_mode0_only": not encode, "fixed_K": True, "original_K23_V20_stage0_coverage": True,
        "complete_B0_formula": "4590*N+89733", "complete_B0_bytes": expected,
        "before_unique_capacity_bytes": before, "after_unique_capacity_bytes": after,
        "actual_extra_persistent_bytes": after - before, "no_new_tensor_fields": True,
        "old_f0_retained_and_charged": True, "shared_original_constants_unchanged": True,
        "reader_is_new_spatial": spatial_reader is not None, "reader_mode": "shuffled" if mode == "Shuffled" else ("spatial" if spatial_reader else "original"),
        "tile_tokens": tile_tokens, "candidate_chunk": candidate_chunk, "layer_stats": stats,
        "source_sha256": sha(__file__), "numerical_source_sha256": sha(numerical.__file__),
        "numerical_source_pins": NUMERICAL_PINS, "same_packed_payload_layout": True,
        "native_FP16_source_or_encoding_workspace_retained": False,
        "quality_or_speed_gain_claimed": False, "CUDA_equivalence_established_by_factory": False}
    metadata["returned_owners_unique_tensor_capacity_bytes"] = after
    metadata["reader_unchanged"] = spatial_reader is None
    return owners, metadata
