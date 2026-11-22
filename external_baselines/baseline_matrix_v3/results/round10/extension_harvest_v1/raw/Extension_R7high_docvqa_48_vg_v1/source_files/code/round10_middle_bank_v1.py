"""Round10 physical middle points on the immutable original R7 low bank.

V128-R43 changes V metadata grouping, folds the original V0 factors once and
re-encodes only its original V20 residuals. AddOldOrder copies the original 43
slots exactly and adds a compact prefix of the original full priority list.
The frozen packed attention function and all legacy source files are reused.
"""
from __future__ import annotations

from copy import copy
import hashlib
import importlib
from pathlib import Path
import struct
import time

import torch
import r7_legacy_bridge_v2 as dual_bridge
import round9_value_bank_v1 as old

bridge = dual_bridge._v1
MODES = ("V128-R43", "AddOldOrder")
NUMERICAL_PINS = {
    "r7_legacy_bridge_v1.py": "b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b",
    "r7_legacy_bridge_v2.py": "608b1660896c8e8fd05484be47acf73e99f260db71731239937d0279ed842dad",
    "round9_value_bank_v1.py": "14d242a6643a738457c9b8593d0f0eb6e383c5e979b2e196fcca28e7d37a5f2e",
}


def require(value, message):
    if not value:
        raise ValueError("ROUND10_MIDDLE: " + message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def check_numerical_pins():
    for name, digest in NUMERICAL_PINS.items():
        require(sha(Path(__file__).with_name(name)) == digest, "frozen dependency changed: " + name)


def budget_plan(n, *, extra_persistent_bytes=0):
    """Exact actual-owner budget; no payload padding or uncharged constants."""
    require(type(n) is int and n > 0 and n % 128 == 0, "N must be positive and divisible128")
    require(type(extra_persistent_bytes) is int and extra_persistent_bytes >= 0, "invalid extra asset charge")
    original = 3816 * n + 89733 + 18 * n * 43
    v128 = 5038 * n + 32389
    q = (v128 - original - extra_persistent_bytes) // (18 * n)
    require(q >= 0 and 43 + q <= 224, "middle-point prefix is not affordable/legal")
    more = original + 18 * n * q + extra_persistent_bytes
    require(0 <= v128 - more < 18 * n, "byte mismatch exceeds one real stage")
    return {"visual_tokens": n, "original_bytes": original, "V128_bytes": v128,
            "extra_persistent_asset_bytes": extra_persistent_bytes, "added_stages": q,
            "selected_stage_count": 43 + q, "AddOldOrder_bytes": more,
            "unspent_bytes_to_V128": v128 - more, "stage_bytes": 18 * n}


class _Group128Reader:
    """Only dispatch changes: the same frozen packed reader receives g128."""
    def __init__(self, reader):
        self.reader = reader

    def __call__(self, q, state, book, _ignored_whole_image_group):
        return self.reader(q, state, book, 128)


def _layer_stages(stages, layer):
    return [(axis, head, depth) for axis, index, head, depth in stages if int(index) == layer]


def _check_old_prefix(owners, metadata, ranking, validate):
    selected = [tuple(x) for x in metadata["selected_stages"]]
    ranking = validate(ranking, 28, 4)
    require(len(ranking) == 224 and all(x[3] == 0 for x in ranking), "original224 must consist of legal stage0 units")
    require(selected == ranking[:43] and len(selected) == 43 and
            sum(x[0] == "k" for x in selected) == 23 and sum(x[0] == "v" for x in selected) == 20,
            "only original K23/V20 priority prefix is admitted")
    require(len(owners) == 28 and all(x.shared is owners[0].shared for x in owners), "shared original constants required")
    for layer, owner in enumerate(owners):
        entries = _layer_stages(selected, layer)
        state = owner.state
        for slot, (axis, head, depth) in enumerate(entries):
            require(int(state.depth_residual_lookup[int(axis == "v"), head, depth]) == slot, "original compact slot order differs")
        require(hasattr(state, "depth_residual_code") == bool(entries) and
                hasattr(state, "depth_residual_alpha") == bool(entries), "original residual field presence differs")
        if entries:
            require(tuple(state.depth_residual_code.shape) == (len(entries), owner.positions.numel(), 4) and
                    tuple(state.depth_residual_alpha.shape) == (len(entries), owner.positions.numel()), "original compact pool shape differs")
    return selected, ranking


def _extend_layer(previous, native_key, native_value, old_selected, new_selected, layer,
                  lookup, book, *, encode_sign_depth, decode_key, wht):
    """Replace one compact pool; old slots copied and only genuinely new slots encoded."""
    state = copy(previous)
    old_entries, entries = _layer_stages(old_selected, layer), _layer_stages(new_selected, layer)
    n = native_value.shape[-2]
    require(entries[:len(old_entries)] == old_entries, "original per-layer slots must remain a prefix")
    if entries:
        state.depth_residual_code = torch.empty((len(entries), n, 4), device=native_value.device, dtype=torch.int32)
        state.depth_residual_alpha = torch.empty((len(entries), n), device=native_value.device, dtype=torch.float16)
        if old_entries:
            state.depth_residual_code[:len(old_entries)].copy_(previous.depth_residual_code)
            state.depth_residual_alpha[:len(old_entries)].copy_(previous.depth_residual_alpha)
        for slot, (axis, head, depth) in enumerate(entries):
            lookup[layer, int(axis == "v"), head, depth] = slot
        added = entries[len(old_entries):]
        # The admitted original224 ranking is all stage0. Thus a new head has
        # no previous residual, and its actual deployed base is the full base.
        if any(axis == "k" for axis, _, _ in added):
            mean = native_key.float().mean(-2, keepdim=True).half()
            target_key = wht(native_key.float() - mean.float())
            base_key = decode_key(previous.direction_code, previous.direction_gain, book)
        if any(axis == "v" for axis, _, _ in added):
            base_value = old.decode_value_fields(previous, n, include_residual=False)
        for slot in range(len(old_entries), len(entries)):
            axis, head, depth = entries[slot]
            require(depth == 0, "new middle stages must be stage0")
            target, base = (target_key[0, head], base_key[0, head]) if axis == "k" else (native_value[0, head], base_value[0, head])
            codes, alphas, _, _ = encode_sign_depth(target, base, 1)
            state.depth_residual_code[slot].copy_(codes[0])
            state.depth_residual_alpha[slot].copy_(alphas[0])
        if old_entries:
            require(torch.equal(state.depth_residual_code[:len(old_entries)], previous.depth_residual_code) and
                    torch.equal(state.depth_residual_alpha[:len(old_entries)], previous.depth_residual_alpha), "an original43 slot changed")
    else:
        require(not hasattr(state, "depth_residual_code") and not hasattr(state, "depth_residual_alpha"), "empty layer retained a residual pool")
    state.depth_residual_lookup = lookup[layer]
    return state, {"layer": layer, "original_slots": len(old_entries), "selected_slots": len(entries),
                   "added_slots": len(entries) - len(old_entries), "original_slots_bitwise_unchanged": True}


@torch.no_grad()
def build_round10_middle_owners(native_prefix_cache, visual_positions, variant, *, mode,
        legacy_a2, legacy_experiments, legacy_scripts, kivi_source,
        expected_asset_sha256=None, expected_legacy_source_sha256=None):
    check_numerical_pins()
    require(mode in MODES, "unknown middle mode")
    require(variant.get("name") == dual_bridge.LOW, "only original low profile")
    n = int(visual_positions.numel())
    plan = budget_plan(n)
    owners, metadata = dual_bridge.build_r7_owners(native_prefix_cache, visual_positions, variant,
        legacy_a2=legacy_a2, legacy_experiments=legacy_experiments, legacy_scripts=legacy_scripts,
        kivi_source=kivi_source, expected_asset_sha256=expected_asset_sha256,
        expected_legacy_source_sha256=expected_legacy_source_sha256)
    before = old._unique(owners)
    require(before == plan["original_bytes"], "original measured capacity differs from physical formula")
    shared = owners[0].shared
    stats = []
    with bridge._legacy_paths(Path(legacy_a2).resolve(), Path(legacy_experiments).resolve(), Path(legacy_scripts).resolve()):
        codec = importlib.import_module("round7_key_codec")
        selected, ranking = _check_old_prefix(owners, metadata, variant["selected_stages"], codec.validate_stages)
        if mode == "V128-R43":
            quant = importlib.import_module("quantizers")
            fold = importlib.import_module("affine_rate_bank").fold
            for layer, owner in enumerate(owners):
                started = time.perf_counter()
                previous = owner.state
                source = native_prefix_cache.layers[layer].values.index_select(2, visual_positions).detach().contiguous()
                state = copy(previous)
                raw = quant.binary_centroids(source.transpose(2, 3).contiguous(), group_size=128)
                state.v_code, state.v_scale, state.v_min = fold(raw, 1,
                    shared["encoder_factor.value_head_factors"][layer][None, :, None, None],
                    shared["encoder_factor.value_center_offsets"][layer][None, :, None, None])
                state, detail = old.reencode_value_residuals(state, source, selected, layer)
                for name in ("direction_code", "direction_gain", "direction_mean", "depth_residual_lookup", "visual_pos"):
                    require(getattr(state, name) is getattr(previous, name), "V128 replaced frozen K/lookup/positions: " + name)
                for slot, (axis, _, _) in enumerate(_layer_stages(selected, layer)):
                    if axis == "k":
                        require(torch.equal(state.depth_residual_code[slot], previous.depth_residual_code[slot]) and
                                torch.equal(state.depth_residual_alpha[slot], previous.depth_residual_alpha[slot]), "V128 changed a K residual")
                require(tuple(state.v_scale.shape) == (1, 4, 128, n // 128) and
                        torch.isfinite(state.v_scale).all() and torch.isfinite(state.v_min).all(), "invalid V128 folded metadata")
                owner.state = state
                owner._reader = _Group128Reader(owner._reader)
                stats.append({"layer": layer, **detail, "V_metadata_groups": n // 128,
                    "host_seconds_including_synchronizing_diagnostics": time.perf_counter() - started})
            new_selected = selected
            policy_detail = {"unchanged_original_explicit_descriptor": True, "bytes": 5}
        else:
            new_selected = ranking[:plan["selected_stage_count"]]
            lookup = torch.full_like(shared["global_residual_lookup"], -1)
            require(tuple(lookup.shape) == (28, 2, 4, 2) and lookup.dtype == torch.int32 and lookup.numel() * 4 == 1792, "new actual lookup geometry")
            wht = importlib.import_module("fast_wht").wht
            for layer, owner in enumerate(owners):
                started = time.perf_counter()
                native = native_prefix_cache.layers[layer]
                key = native.keys.index_select(2, visual_positions).detach().contiguous()
                value = native.values.index_select(2, visual_positions).detach().contiguous()
                previous = owner.state
                state, detail = _extend_layer(previous, key, value, selected, new_selected, layer, lookup, shared["book"],
                    encode_sign_depth=codec.encode_sign_depth, decode_key=codec.decode_key, wht=wht)
                for name in ("direction_code", "direction_gain", "direction_mean", "v_code", "v_scale", "v_min", "visual_pos"):
                    require(getattr(state, name) is getattr(previous, name), "AddOldOrder changed original base field: " + name)
                owner.state = state
                detail["host_seconds_including_synchronizing_diagnostics"] = time.perf_counter() - started
                stats.append(detail)
            shared["global_residual_lookup"] = lookup
            # Same actual five-byte charge, with explicit new semantics: tag4
            # means the original full ranking's prefix of the stored FP32 m.
            descriptor = struct.pack("<Bf", 4, float(len(new_selected)))
            shared["encoder_policy_bytes"] = torch.tensor(list(descriptor), device=visual_positions.device, dtype=torch.uint8)
            policy_detail = {"schema": "trajkv.round10.original_priority_prefix_descriptor.v1", "bytes": 5,
                "format": "little-endian uint8 tag4, float32 selected-prefix-count", "tag": 4,
                "selected_prefix_count": len(new_selected), "sha256": hashlib.sha256(descriptor).hexdigest()}
    after = old._unique(owners)
    expected = plan["V128_bytes" if mode == "V128-R43" else "AddOldOrder_bytes"]
    require(after == expected, "measured packed capacity differs from physical formula")
    require(all(bridge._sha(path) == item["sha256"] for path, item in metadata["verified_legacy_source_pins"].items()), "frozen legacy changed during transform")
    require(all(bridge._sha(item["path"]) == item["sha256"] for item in metadata["fixed_assets"].values()), "fixed asset changed")
    forbidden = {bridge._storage(t) for layer in native_prefix_cache.layers for t in (layer.keys, layer.values)}
    for owner in owners:
        for name, tensor in owner.named_tensors().items():
            require(not tensor.requires_grad and tensor.grad_fn is None and bridge._storage(tensor) not in forbidden, "native/autograd retained: " + name)
        require(all(not isinstance(t, torch.Tensor) or t.numel() == 0 or name in bridge._STATE_TENSORS
                    for name, t in vars(owner.state).items()), "unreported state tensor")
    metadata["original_selected_stages"] = [list(x) for x in selected]
    metadata["selected_stages"] = [list(x) for x in new_selected]
    metadata["selected_stage_count"] = len(new_selected)
    metadata["reader_unchanged"] = mode == "AddOldOrder"
    metadata["returned_owners_unique_tensor_capacity_bytes"] = after
    metadata["round10_middle_candidate"] = {
        "schema": "trajkv.round10.middle_candidate.v1", "mode": mode, "source_sha256": sha(__file__),
        "numerical_source_pins": NUMERICAL_PINS, "budget_plan": plan,
        "before_unique_capacity_bytes": before, "after_unique_capacity_bytes": after,
        "actual_extra_persistent_bytes": after - before, "new_persistent_constant_bytes": 0,
        "V_group_size": 128 if mode == "V128-R43" else n,
        "V_grouping": "128" if mode == "V128-R43" else "all",
        "original_K23_V20_coverage_unchanged": mode == "V128-R43",
        "original43_code_alpha_bitwise_unchanged": mode == "AddOldOrder",
        "original_K_base_fields_unchanged": True, "original_K23_code_alpha_unchanged": True,
        "original_V_base_fields_unchanged": mode == "AddOldOrder",
        "original_V20_reencoded": mode == "V128-R43", "original_V0_factors_folded_exactly_once": mode == "V128-R43",
        "reader_math_unchanged": True, "reader_dispatch_changed": mode == "V128-R43",
        "reader_dispatch": "frozen round7_key_attention.attention(q,state,book,128)" if mode == "V128-R43" else "original whole-image group",
        "actual_compact_per_layer_residual_pools": True, "mask_over_large_pool": False,
        "global_lookup_bytes": 1792, "original_full_ranking_bytes_reused": 448, "policy_descriptor": policy_detail,
        "original_43_coverage_retained": True, "added_stages": len(new_selected) - 43,
        "selected_stage_count": len(new_selected), "K_stages": sum(x[0] == "k" for x in new_selected),
        "V_stages": sum(x[0] == "v" for x in new_selected), "layer_stats": stats,
        "native_FP16_source_or_workspace_retained": False,
        "quality_or_speed_gain_claimed": False,
        "GPU_equivalence_established_by_factory": False,
    }
    return owners, metadata
