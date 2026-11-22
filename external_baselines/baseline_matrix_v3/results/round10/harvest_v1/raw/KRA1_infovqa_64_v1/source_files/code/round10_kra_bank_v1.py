"""KRA1: fixed-sign, fixed-base, rank-one K23 residual-alpha refinement.

Only the original23 K alpha rows can change. Original packed signs, every V
field, K code/gain/mean, compact allocation and reader remain frozen. The two
actual encoding-metric tensors are retained and charged as5980 persistent B.
"""
from __future__ import annotations
from copy import copy
import importlib
import json
from pathlib import Path
import time

import torch
import r7_legacy_bridge_v2 as dual_bridge
import round9_value_bank_v1 as old
import round10_kra_metric_v1 as metric

bridge = dual_bridge._v1
MODE = "KRA1"
NUMERICAL_PINS = {
    "r7_legacy_bridge_v1.py": "b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b",
    "r7_legacy_bridge_v2.py": "608b1660896c8e8fd05484be47acf73e99f260db71731239937d0279ed842dad",
    "round9_value_bank_v1.py": "14d242a6643a738457c9b8593d0f0eb6e383c5e979b2e196fcca28e7d37a5f2e",
    "round10_kra_metric_v1.py": "cb65703f1df7e0f8b0b88340ef3086834d6bdb36498d07d4a2acbb1572d82ee7",
}


def require(value, message):
    if not value:
        raise ValueError("ROUND10_KRA: " + message)


def check_numerical_pins():
    for name, digest in NUMERICAL_PINS.items():
        require(metric.sha(Path(__file__).with_name(name)) == digest, "frozen/new numerical dependency changed: " + name)


def solve_alpha(error, sign, u, gamma, previous_alpha):
    """Fixed-s real quadratic optimum, then the actual FP16 alpha write.

    Gamma0 bypasses arithmetic and returns the original alpha object; its
    deployed decode is the identity control, including old rounding details.
    """
    require(error.shape == sign.shape and error.shape[-1] == 128 and
            tuple(u.shape) == (128,) and previous_alpha.shape == error.shape[:-1], "D128 alpha geometry")
    require(bool(torch.isfinite(error).all() & torch.isfinite(sign).all() & torch.isfinite(u).all()) and
            bool(torch.isfinite(gamma)) and 0 <= float(gamma) <= 32, "finite nonnegative rank-one metric")
    if float(gamma) == 0:
        return previous_alpha, {"gamma": 0., "identity_bypass": True, "changed_alpha_elements": 0,
                                "max_abs_alpha_delta": 0., "clamped_real_optimum_tokens": 0}
    error, sign, direction, strength = error.float(), sign.float(), u.float(), gamma.float()
    su = (sign * direction).sum(-1)
    eu = (error * direction).sum(-1)
    numerator = (sign * error).sum(-1) + strength * su * eu
    denominator = sign.square().sum(-1) + strength * su.square()
    require(bool((denominator > 0).all()), "positive fixed-sign metric denominator")
    real = (numerator / denominator).clamp_min(0)
    alpha = real.half().contiguous()
    require(bool(torch.isfinite(alpha).all()) and bool((alpha >= 0).all()), "finite nonnegative actual Half alpha")
    return alpha, {"gamma": float(gamma), "identity_bypass": False,
        "changed_alpha_elements": int((alpha != previous_alpha).sum()),
        "max_abs_alpha_delta": float((alpha.float() - previous_alpha.float()).abs().max()),
        "clamped_real_optimum_tokens": int((numerator < 0).sum()),
        "alpha_min": float(alpha.min()), "alpha_max": float(alpha.max())}


def choose_deployed_alpha(target, base, sign, u, gamma, previous, candidate):
    """Choose between only old/candidate under their actual Half decodes.

    Strictly smaller FP64 H error accepts the candidate; ties retain original.
    This is a coordinate encoding check, never a task noninferiority claim.
    """
    def loss(alpha):
        decoded = (base.float() + sign.float() * alpha.float()[..., None]).half()
        error = target.double() - decoded.double()
        return error.square().sum(-1) + gamma.double() * (error * u.double()).sum(-1).square()
    old_loss, candidate_loss = loss(previous), loss(candidate)
    require(bool(torch.isfinite(old_loss).all() & torch.isfinite(candidate_loss).all()), "nonfinite actual-H comparison")
    accept = candidate_loss < old_loss
    result = torch.where(accept, candidate, previous).half().contiguous()
    selected_loss = torch.where(accept, candidate_loss, old_loss)
    return result, {"actual_H_candidate_accepted_tokens": int(accept.sum()),
        "actual_H_candidate_worse_tokens": int((candidate_loss > old_loss).sum()),
        "actual_H_candidate_tie_tokens": int((candidate_loss == old_loss).sum()),
        "actual_H_original_error_sum": float(old_loss.sum()),
        "actual_H_candidate_error_sum": float(candidate_loss.sum()),
        "actual_H_selected_error_sum": float(selected_loss.sum()),
        "actual_H_after_worse_tokens": int((selected_loss > old_loss).sum()),
        "actual_H_tie_policy": "retain original alpha", "actual_H_arithmetic": "FP64 error from actual sequential Half decode"}


@torch.no_grad()
def build_round10_kra_owners(native_prefix_cache, visual_positions, variant, *,
        metric_path=None, metric_sha256=None, metric_override=None, calibration_only=False,
        legacy_a2, legacy_experiments, legacy_scripts, kivi_source,
        expected_asset_sha256=None, expected_legacy_source_sha256=None):
    check_numerical_pins()
    require(variant.get("name") == dual_bridge.LOW, "only original R7low K1/V1/43 lineage")
    require((metric_override is None or calibration_only) and
            not (metric_override is not None and (metric_path is not None or metric_sha256 is not None)),
            "override only for explicit calibration without file identity claims")
    require(metric_override is not None or metric_path is not None, "an explicit metric is required")
    owners, metadata = dual_bridge.build_r7_owners(native_prefix_cache, visual_positions, variant,
        legacy_a2=legacy_a2, legacy_experiments=legacy_experiments, legacy_scripts=legacy_scripts,
        kivi_source=kivi_source, expected_asset_sha256=expected_asset_sha256,
        expected_legacy_source_sha256=expected_legacy_source_sha256)
    selected = [list(unit) for unit in metadata["selected_stages"]]
    require(len(owners) == 28 and selected == variant["selected_stages"][:43] and
            len(selected) == 43 and sum(unit[0] == "v" for unit in selected) == 20,
            "original low43 priority prefix required")
    units = metric.normalize_unit_order([unit for unit in selected if unit[0] == "k"])
    before = old._unique(owners)
    n = int(visual_positions.numel())
    require(before == 4590 * n + 89733, "original physical owner formula differs")
    shared = owners[0].shared
    require(all(owner.shared is shared for owner in owners), "one actual shared constant table required")
    if metric_override is None:
        tensors, info = metric.load_metric(metric_path, metric_sha256, device=visual_positions.device, expected_unit_order=units)
    else:
        tensors = metric.metric_tensors(metric_override, device=visual_positions.device)
        info = {"origin": "calibration_only_in_memory", "schema": metric.SCHEMA, "unit_order23": units,
                "actual_persistent_tensor_bytes": metric.METRIC_BYTES,
                "actual_tensor_sha256": {name: metric.tensor_sha(value) for name, value in tensors.items()}}
    stats = []
    with bridge._legacy_paths(Path(legacy_a2).resolve(), Path(legacy_experiments).resolve(), Path(legacy_scripts).resolve()):
        codec = importlib.import_module("round7_key_codec")
        wht = importlib.import_module("fast_wht").wht
        factors = json.loads(Path(variant["metadata_factor_path"]).read_text(encoding="utf-8-sig"))
        require(metric.sha(variant["metadata_factor_path"]) == metric.BASE_FACTOR_SHA and
                factors["key_gain_mode"] == "centered", "original 8f47 centered key-gain mode required")
        for layer, owner in enumerate(owners):
            previous, state = owner.state, copy(owner.state)
            local = [(index, head) for index, (_, unit_layer, head, _) in enumerate(units) if unit_layer == layer]
            old_entries = [(axis, head, depth) for axis, index, head, depth in selected if index == layer]
            if any(float(tensors["gamma"][index]) != 0 for index, _ in local):
                native = native_prefix_cache.layers[layer].keys.index_select(2, visual_positions).detach().contiguous()
                # This is exactly the original encoder's source target. Mean
                # is rounded before subtraction; use current frozen gain/book.
                source = native.float()
                mean = source.mean(-2, keepdim=True).half()
                target = wht(source - mean.float())
                base = codec.decode_key(previous.direction_code, previous.direction_gain, shared["book"])
                state.depth_residual_alpha = previous.depth_residual_alpha.detach().clone()
            for index, head in local:
                started = time.perf_counter()
                slot = int(previous.depth_residual_lookup[0, head, 0])
                require(slot >= 0 and old_entries[slot] == ("k", head, 0), "original K23 compact slot differs")
                gamma = tensors["gamma"][index]
                if float(gamma) == 0:
                    detail = {"gamma": 0., "identity_bypass": True, "changed_alpha_elements": 0,
                              "max_abs_alpha_delta": 0., "clamped_real_optimum_tokens": 0}
                else:
                    code = previous.depth_residual_code[slot]
                    sign = old._unpack(code, 128).float() * 2 - 1
                    alpha, detail = solve_alpha(target[0, head].float() - base[0, head].float(), sign,
                                                tensors["u"][index], gamma, previous.depth_residual_alpha[slot])
                    alpha, acceptance = choose_deployed_alpha(target[0, head], base[0, head], sign,
                        tensors["u"][index], gamma, previous.depth_residual_alpha[slot], alpha)
                    detail.update(acceptance)
                    detail["changed_alpha_elements"] = int((alpha != previous.depth_residual_alpha[slot]).sum())
                    detail["max_abs_alpha_delta"] = float((alpha.float() - previous.depth_residual_alpha[slot].float()).abs().max())
                    state.depth_residual_alpha[slot].copy_(alpha)
                stats.append({"unit_index": index, "unit": units[index], "layer": layer, "head": head, **detail,
                              "host_seconds_including_synchronizing_diagnostics": time.perf_counter() - started})
            for name in ("direction_code", "direction_gain", "direction_mean", "v_code", "v_scale", "v_min", "depth_residual_lookup", "visual_pos"):
                require(getattr(state, name) is getattr(previous, name), "frozen base/lookup/positions object replaced: " + name)
            require(hasattr(state, "depth_residual_code") == hasattr(previous, "depth_residual_code") and
                    hasattr(state, "depth_residual_alpha") == hasattr(previous, "depth_residual_alpha"), "residual field presence changed")
            if old_entries:
                require(state.depth_residual_code is previous.depth_residual_code, "a frozen residual sign code was replaced")
                for slot, (axis, _, _) in enumerate(old_entries):
                    if axis == "v":
                        require(torch.equal(state.depth_residual_alpha[slot], previous.depth_residual_alpha[slot]), "V20 alpha changed")
                require(bool(torch.isfinite(state.depth_residual_alpha).all()), "nonfinite residual alpha")
            owner.state = state
    require(len(stats) == 23 and {item["unit_index"] for item in stats} == set(range(23)), "exact K23 update coverage")
    shared.update({"kra_metric.u": tensors["u"], "kra_metric.gamma": tensors["gamma"]})
    after = old._unique(owners)
    require(after == before + metric.METRIC_BYTES, "all actual metric constants must be charged exactly5980B")
    require(all(bridge._sha(path) == item["sha256"] for path, item in metadata["verified_legacy_source_pins"].items()) and
            all(bridge._sha(item["path"]) == item["sha256"] for item in metadata["fixed_assets"].values()), "legacy source/fixed asset changed")
    forbidden = {bridge._storage(value) for layer in native_prefix_cache.layers for value in (layer.keys, layer.values)}
    for owner in owners:
        for name, value in owner.named_tensors().items():
            require(bridge._storage(value) not in forbidden and not value.requires_grad and value.grad_fn is None,
                    "native source/autograd graph retained: " + name)
        require(all(not isinstance(value, torch.Tensor) or value.numel() == 0 or name in bridge._STATE_TENSORS
                    for name, value in vars(owner.state).items()), "unreported persistent state tensor")
    metadata["returned_owners_unique_tensor_capacity_bytes"] = after
    metadata["round10_kra_candidate"] = {
        "schema": "trajkv.round10.kra_candidate.v1", "mode": MODE, "source_sha256": metric.sha(__file__),
        "numerical_source_pins": NUMERICAL_PINS, "metric": info, "calibration_only": bool(calibration_only),
        "unit_order23": units, "original_K23_V20_coverage": True, "selected_stage_count": 43,
        "original_K_base_fields_unchanged": True, "original_V_base_fields_unchanged": True,
        "all_residual_sign_codes_unchanged": True, "original_V20_alpha_unchanged": True,
        "original_lookup_ranking_policy_unchanged": True, "reader_unchanged": True,
        "zero_gamma_is_original_alpha_bitwise_identity": True,
        "all_gamma_zero": bool((tensors["gamma"] == 0).all()), "alpha_write_dtype": "float16",
        "native_K_target": "frozen wht(nativeK.float - nativeK.float.mean(-2).half.float)",
        "K_base": "frozen decode_key(current original code,gain,book), actual FP16",
        "alpha_rule": "max(0,s^T(I+gamma*u*u^T)(target-base))/(s^T(I+gamma*u*u^T)s), then FP16",
        "gamma0_rule": "direct reuse original alpha; do not recompute an identity via arithmetic",
        "deployed_alpha_acceptance": "compare only original/candidate actual Half decoded H error; strict improvement, ties retain original",
        "actual_H_after_worse_tokens": sum(item.get("actual_H_after_worse_tokens", 0) for item in stats),
        "metric_parameter_count": 2967, "new_persistent_constant_bytes": metric.METRIC_BYTES,
        "before_unique_capacity_bytes": before, "after_unique_capacity_bytes": after,
        "actual_extra_persistent_bytes": after - before, "per_unit_alpha_changes": stats,
        "changed_K_alpha_elements": sum(item["changed_alpha_elements"] for item in stats),
        "native_FP16_source_or_workspace_retained": False, "quality_or_speed_gain_claimed": False,
        "coordinate_quadratic_optimum_is_not_task_optimum": True}
    return owners, metadata
