"""One locked-source, QA0, actual-model TF controls for both middle points.

The frozen calibrator owns model/data admission and native teacher generation.
An explicit temporary technical callback executes only these two new packed
deployments. This is not its original lambda0/W1 four-control receipt, fitting,
parameter selection, a free-generation task score, or a speed benchmark.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import round10_calibrate_v1 as frozen
import round10_middle_runner_v1 as middle

BASE_CALIBRATOR_SHA = "abbc45f07e5899ff6018e0a22c5bba94c8764b03cd518e764de1949d9df87a8a"
MODES = ("V128-R43", "AddOldOrder")


def require(value, message):
    frozen.require(value, "MIDDLE_TECHNICAL: " + message)


def runtime_source_manifest():
    """CPU-only source lock helper for the root scheduler's new config."""
    require(frozen.sha(ROOT / "code/round10_calibrate_v1.py") == BASE_CALIBRATOR_SHA,
            "frozen setup/TF driver changed")
    roles = set(frozen.PINS) | set(middle.runtime_source_manifest()) | {
        "code/round10_calibrate_v1.py", "code/round10_middle_technical_v1.py"}
    return {role: frozen.sha(ROOT / role) for role in sorted(roles)}


def validate_config(config):
    frozen.validate_config(config)
    require(config.get("technical_only") is True, "technical_only=true required; no fit/select execution")
    index = config.get("technical_source_index", 0)
    require(type(index) is int and 0 <= index < 16, "one locked calSelect index0..15")
    expected = config["expected_source_hashes"]
    graph = runtime_source_manifest()
    require(set(graph).issubset(expected), "pin entire middle/setup runtime source graph via runtime_source_manifest()")
    require(all(expected[role] == digest for role, digest in graph.items()), "middle/setup runtime source lock differs")
    return config


def input_identity(config, row):
    fields = {role: config[role] for role in
              ("cal_fit_inputs", "cal_select_inputs", "input_manifest_path", "model_source_lock")}
    fields["variant_source"] = config["variant_source"]["path"]
    fields["source_image"] = row["image_path"]
    fields.update({"asset." + role: config["variant"][role] for role in config["expected_asset_sha256"]})
    result = {role: {"path": str(path), "sha256": frozen.sha(path)} for role, path in fields.items()}
    require(result["cal_fit_inputs"]["sha256"] == config["cal_fit_sha256"] and
            result["cal_select_inputs"]["sha256"] == config["cal_select_sha256"] and
            result["variant_source"]["sha256"] == config["variant_source"]["sha256"] and
            result["source_image"]["sha256"] == row["original_image_sha256"], "locked source/input identity differs")
    for role, digest in config["expected_asset_sha256"].items():
        require(result["asset." + role]["sha256"] == digest, "fixed asset identity differs")
    return result


def native_prefix_identity(native):
    import torch
    require(len(native.layers) == 28, "all28 native prefix layers")
    return {"logical_length": int(native.get_seq_length()), "layers": [
        {axis: {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
                "sha256": frozen.tensor_sha(tensor), "storage_bytes": tensor.untyped_storage().nbytes()}
         for axis, tensor in (("K", layer.keys), ("V", layer.values))}
        for layer in native.layers], "all_CUDA_FP16": all(
            tensor.is_cuda and tensor.dtype == torch.float16
            for layer in native.layers for tensor in (layer.keys, layer.values))}


def stage_signatures(owners, selected):
    result = []
    for axis, layer, head, depth in selected:
        state = owners[int(layer)].state
        slot = int(state.depth_residual_lookup[int(axis == "v"), int(head), int(depth)])
        require(slot >= 0, "original residual no longer has a physical slot")
        result.append({"stage": [axis, int(layer), int(head), int(depth)],
                       "code_sha256": frozen.tensor_sha(state.depth_residual_code[slot]),
                       "alpha_sha256": frozen.tensor_sha(state.depth_residual_alpha[slot])})
    return result


def build_cache(native, batch, variant, config, mode):
    from m1_protocol import clone_m1_prefix_cache
    from cache_bridge import HybridVisualCache
    from r7_frozen_pilot_v2 import share_original_visual_position_map
    from round10_middle_bank_v1 import build_round10_middle_owners
    from ownership import assert_no_full_prefix_alias
    clone = clone_m1_prefix_cache(native)
    source_storage_ids = {(str(t.device), t.untyped_storage().data_ptr())
                          for layer in clone.layers for t in (layer.keys, layer.values)}
    kwargs = {key: config[key] for key in ("legacy_a2", "legacy_experiments", "legacy_scripts", "kivi_source")}
    kwargs["expected_asset_sha256"] = config["expected_asset_sha256"]
    owners, metadata = build_round10_middle_owners(clone, batch.visual_positions, variant, mode=mode, **kwargs)
    require(metadata["variant_sha256"] == frozen.canonical(variant), "original encoding-lineage variant changed")
    signature = frozen.state_signatures(owners)
    cache = HybridVisualCache(clone, batch.visual_positions, owners, consume_native=True)
    share_original_visual_position_map(cache)
    assert_no_full_prefix_alias(cache, source_storage_ids)
    return cache, owners, metadata, signature


def dense_current_decode(owner, q, kwargs, packed):
    """Last actual query: dense scores from its currently deployed packed fields.

    This FP32 softmax/reduction is a diagnostic reference, not a claim of the
    frozen online kernel's exact reduction/FP16-weight rounding equivalence.
    Dense tensors are temporary, never assigned to any owner or cache.
    """
    import torch
    from fast_wht import wht
    from round7_key_codec import decode_key
    from round6_depth_codec import decode_sign_stage
    from round9_value_bank_v1 import decode_value_fields
    state = owner.state
    rotated_key = decode_key(state.direction_code, state.direction_gain, owner.shared["book"])
    heads = []
    for head in range(4):
        item = rotated_key[:, head]
        for depth in range(2):
            slot = int(state.depth_residual_lookup[0, head, depth])
            if slot >= 0:
                item = decode_sign_stage(item, state.depth_residual_code[slot], state.depth_residual_alpha[slot])
        heads.append(item)
    rotated_key = torch.stack(heads, dim=1).repeat_interleave(7, dim=1)
    value = decode_value_fields(state, int(owner.positions.numel())).repeat_interleave(7, dim=1)
    qlast = q[..., -1:, :].contiguous()
    rotated_q = wht(qlast, half=True)
    bias = (qlast.float() * state.direction_mean.repeat_interleave(7, dim=1).float()).sum(-1, keepdim=True) * (128 ** -.5)
    visual_scores = torch.matmul(rotated_q.float(), rotated_key.float().transpose(-2, -1)) * (128 ** -.5) + bias
    text_key = kwargs["exact_keys"].repeat_interleave(7, dim=1)
    text_value = kwargs["exact_values"].repeat_interleave(7, dim=1)
    text_scores = torch.matmul(qlast.float(), text_key.float().transpose(-2, -1)) * (128 ** -.5)
    positions = torch.cat((owner.positions, kwargs["exact_positions"]))
    scores = torch.cat((visual_scores, text_scores), dim=-1)
    scores = scores.masked_fill(positions[None, None, None] > kwargs["query_positions"][-1], -float("inf"))
    probability = scores.softmax(-1)
    values = torch.cat((value, text_value), dim=2)
    reference = torch.matmul(probability, values.float())
    half_weight_reference = torch.matmul(probability.half().float(), values.float())
    actual = packed[..., -1:, :].float()
    require(bool(torch.isfinite(reference).all() & torch.isfinite(actual).all()), "nonfinite actual-query dense comparison")
    error = actual - reference
    return {"layer_idx": owner.layer_idx, "actual_query": "first suffix call, final query position",
        "q_sha256": frozen.tensor_sha(qlast), "query_position": int(kwargs["query_positions"][-1]),
        "exact_positions_sha256": frozen.tensor_sha(kwargs["exact_positions"]),
        "visual_tokens": int(owner.positions.numel()), "exact_text_tokens": int(text_key.shape[-2]),
        "max_abs_diff_FP32_dense": float(error.abs().max()), "mean_abs_diff_FP32_dense": float(error.abs().mean()),
        "relative_L2_diff_FP32_dense": float(error.norm() / reference.norm().clamp_min(1e-20)),
        "max_abs_diff_global_FP16_weights": float((actual - half_weight_reference).abs().max()),
        "dense_decode_is_current_candidate": True, "dense_reference_workspace_retained": False,
        "reference_scope": "same packed Half K/V, Half WHT(query), native mean bias and current exact text; FP32 dense softmax/reduction",
        "exact_online_reduction_or_task_quality_acceptance": False}


def replay_with_dense_diagnostic(model, batch, cache, owners, history):
    diagnostic = []
    owner = owners[0]
    original = owner.attend
    def observed(q, **kwargs):
        result = original(q, **kwargs)
        if not diagnostic:
            diagnostic.append(dense_current_decode(owner, q, kwargs, result))
        return result
    owner.attend = observed
    try:
        loss = frozen.replay_branch(model, batch, cache, owners, history)
    finally:
        owner.attend = original
    require(len(diagnostic) == 1, "actual packed query diagnostic not observed")
    return loss, diagnostic[0]


def technical_one_source(model, processor, row, variant, config, factors, output):
    import torch
    from ownership import cache_storage
    source_before = runtime_source_manifest()
    input_before = input_identity(config, row)
    batch, native, context = frozen.prepare_case(model, processor, row)
    native_before = native_prefix_identity(native)
    require(native_before["all_CUDA_FP16"], "real CUDA FP16 native prefix required")
    history = frozen.teacher_history(model, batch, native)
    require(native_prefix_identity(native) == native_before, "native teacher mutated independent source prefix")
    reference_cache, reference_owners, reference_metadata, reference_signature = frozen.build_cache(native, batch, variant, config)
    original_stages = reference_metadata["selected_stages"]
    original_stage_signature = stage_signatures(reference_owners, original_stages)
    reference_owner = cache_storage(reference_cache, reference_cache.visual_positions)
    n = int(batch.visual_positions.numel())
    require(reference_owner["complete_visual_owner_bytes"] == 4590 * n + 89733, "actual base43 physical formula differs")
    reference_cache = reference_owners = None
    gc.collect()
    results = []
    for mode in MODES:
        cache, owners, metadata, signature = build_cache(native, batch, variant, config, mode)
        if mode == "V128-R43":
            require(signature["K_fields_sha256"] == reference_signature["K_fields_sha256"], "V128 changed complete original K signature")
            original43_preserved = None  # V20 is intentionally re-encoded.
        else:
            require(stage_signatures(owners, original_stages) == original_stage_signature, "AddOldOrder changed an original43 code/alpha")
            require(metadata["round10_middle_candidate"]["original43_code_alpha_bitwise_unchanged"] is True,
                    "factory original43 copy invariant absent")
            original43_preserved = True
        ownership_before = cache_storage(cache, cache.visual_positions)
        budget_before = middle.round10_budget_guard(mode, n, ownership_before["complete_visual_owner_bytes"], metadata, frozen.BASE_PROFILE)
        loss, dense = replay_with_dense_diagnostic(model, batch, cache, owners, history)
        require(frozen.state_signatures(owners) == signature, "packed reader fields changed during full TF")
        ownership_after = cache_storage(cache, cache.visual_positions)
        budget_after = middle.round10_budget_guard(mode, n, ownership_after["complete_visual_owner_bytes"], metadata, frozen.BASE_PROFILE)
        require(budget_before == budget_after, "visual budget changed during full TF")
        require(native_prefix_identity(native) == native_before, "middle replay mutated independent native source prefix")
        results.append({"mode": mode, "loss": loss, "metadata": metadata,
            "reader_field_sha256": signature["reader_field_sha256"], "K_fields_sha256": signature["K_fields_sha256"],
            "complete_K_signature_equals_base43": signature["K_fields_sha256"] == reference_signature["K_fields_sha256"],
            "complete_K_equality_required": mode == "V128-R43", "original43_code_alpha_preserved": original43_preserved,
            "original43_preservation_for_Add_independently_rechecked": mode == "AddOldOrder",
            "actual_owner_before_TF": ownership_before, "actual_owner_after_TF": ownership_after,
            "physical_budget_guard": budget_after, "packed_fields_unchanged_after_full_TF": True,
            "actual_query_dense_diagnostic": dense, "native_source_prefix_unchanged": True})
        cache = owners = None
        gc.collect()
        torch.cuda.synchronize()
    require(input_identity(config, row) == input_before, "locked inputs/assets changed during technical replay")
    require(runtime_source_manifest() == source_before, "middle/setup source graph changed during technical replay")
    result = {"schema": "trajkv.round10.middle_technical.v1", "status": "PASS_MIDDLE_ONE_SOURCE_ACTUAL_PACKED_FULL_TF",
        "driver_role": "one locked calSelect source; two middle-point deployment API and physical ownership controls",
        "frozen_setup_driver_sha256": BASE_CALIBRATOR_SHA, "original_calibration_four_controls_executed": False,
        "actual_GPU": True, "actual_GPU_name": torch.cuda.get_device_name(), "actual_GPU_capability": list(torch.cuda.get_device_capability()),
        "actual_model_parameter_count": sum(p.numel() for p in model.parameters()),
        "weight_dtype": "float16", "context": context, "controls": results,
        "native_prefix_identity_before": native_before, "native_prefix_identity_after": native_prefix_identity(native),
        "native_prefix_unchanged": True, "base43_reference_reader_signature": reference_signature,
        "base43_actual_owner": reference_owner, "base43_original_stage_signatures": original_stage_signature,
        "input_identity_before": input_before, "input_identity_after": input_identity(config, row),
        "middle_runtime_source_hashes_before": source_before, "middle_runtime_source_hashes_after": runtime_source_manifest(),
        "teacher_token_ids": history["token_ids"], "teacher_sources": 1, "teacher_tokens": len(history["token_ids"]),
        "teacher_EOS_reached": history["EOS_reached"], "teacher_cap": 32, "teacher_actual_native_history": True,
        "teacher_first_call": history["first_call"], "actual_TF_source_evaluations": 2,
        "QA_events": 0, "new_free_generation_task_answers": 0, "references_read": False,
        "new_parameters_selected": False, "theta_selected": None, "window_selected": None,
        "full_fit48_select16_completed": False, "quality_claim": False, "speed_or_peak_benchmark_claim": False}
    frozen.write_new(Path(output) / "MIDDLE_TECHNICAL_ROLE.json", {
        "schema": result["schema"], "driver_role": result["driver_role"], "actual_TF_source_evaluations": 2,
        "QA_events": 0, "technical_callback": "round10_middle_technical_v1.technical_one_source",
        "frozen_setup_and_model_admission": "round10_calibrate_v1.calibrate",
        "original_calibration_four_controls_executed": False})
    return result


def run(config, output):
    validate_config(config)
    original_callback = frozen.technical_one_source
    frozen.technical_one_source = technical_one_source
    try:
        return frozen.calibrate(config, output)
    finally:
        frozen.technical_one_source = original_callback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    config["calibration_config_sha256"] = frozen.sha(args.config)
    result = run(config, args.output)
    print(json.dumps({key: result[key] for key in ("status", "teacher_sources", "teacher_tokens", "actual_TF_source_evaluations", "QA_events")},
                     ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
