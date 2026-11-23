"""Derived-only WO192 encoder/resource analysis from all20 closed views.

No inference, fitting, GPU/SSH, source edits, private tensor loading, holdout
outcomes or task-score comparison occurs. H=e'He and 0.5*H are separately
labelled V20 local metrics. Timings are executed prototype descriptions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "ops"), str(ROOT / "code")]
import round11_wo_harvest_v1 as harvest

DOMAINS = ("textvqa", "infovqa", "docvqa", "chartqa")
WO_MODES = ("SpatialWO", "JointSpatialWO")
MODES = ("FP16", "R7low", "R7high", *WO_MODES)
COUNTS = dict(textvqa=32, infovqa=32, docvqa=16, chartqa=16)
FACTORY_SHA = "b0f62f1c8ddbad41336b161891cf0943876a0087e9dcdec63e9bc6e312aaf69e"
HARVEST_SHA = "d53152f6e8986c4ec2686f2d499689bee116389541434151391b845071544c79"
SOURCE_LOCK_SHA = "980da29bfdca806541d998115cb4f30148c8986c306b70db327f68ba54335570"
PLAN_PATH = Path("${EVIDENCE_ROOT}/results/round11/bootstrap_v1/WO_HARVEST_PLAN_v1.json")
MODEL_PARAMETER_COUNT = 7944373760


def require(value, message):
    if not value:
        raise ValueError("ROUND11_WO_STAGE_ANALYSIS: " + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def raw_lines(path):
    return [line for line in Path(path).read_bytes().splitlines(keepends=True) if line.strip()]


def stats(values):
    values = list(values)
    require(all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in values), "nonfinite/missing numeric diagnostic")
    return {"count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)} if values else None


def decrease(before, after):
    return 1 - after / before if before > 0 else None


def monotonic(before, after, label):
    require(math.isfinite(before) and math.isfinite(after) and after <= before + 1e-12 * max(1., abs(before)), label + " full-H increase")


def check_capture(capture):
    require(capture["actual_from_pretrained_calls"] == 1
        and capture["actual_model_parameter_count"] == MODEL_PARAMETER_COUNT
        and capture["only_weak_reference_retained_by_capture"] is True
        and capture["model_weights_borrowed_not_cloned_or_quantized"] is True,
        "actual single borrowed model capture differs")


def charged_visual(owner):
    tensors = owner["tensors"]
    visual = sum(value["charged_bytes"] for name, value in tensors.items() if name == "original_visual_positions" or ".visual." in name)
    text = sum(value["charged_bytes"] for name, value in tensors.items() if ".text_" in name)
    total = sum(value["charged_bytes"] for value in tensors.values())
    require(total == owner["total_bytes"] == visual + text
        and visual == owner["stored_visual_KV_bytes"] == owner["complete_visual_owner_bytes"], "global charged-storage visual/text accounting differs")
    return visual


def encoder_metadata(metadata, mode, n, capture):
    """Pure scalar extraction; usable with existing actual QA0 metadata too."""
    require(mode in WO_MODES and n > 0 and n % 256 == 0, "WO mode/native visual N")
    check_capture(capture)
    d = metadata["round11_wo_candidate"]
    expected = 4590 * n + 89733
    require(d["schema"] == "trajkv.round11.wo_candidate.v1" and d["mode"] == mode
        and d["source_sha256"] == FACTORY_SHA and d["encoding_applied"] is True and d["control_mode0_only"] is False,
        "actual frozen encoded WO source/flags differ")
    require(d["before_unique_capacity_bytes"] == d["after_unique_capacity_bytes"] == d["complete_B0_bytes"] == expected
        and d["actual_extra_persistent_bytes"] == d["persistent_derived_metric_tensor_bytes"] == 0
        and d["native_FP16_source_or_encoding_workspace_retained"] is False
        and d["tile_tokens"] == 128 and d["candidate_chunk"] == 32,
        "WO persistent/temporary distinction or B0 differs")
    require(all(d[key] is True for key in ("fixed_K", "original_K23_V20_stage0_coverage", "old_f0_retained_and_charged",
        "shared_original_constants_unchanged", "no_new_tensor_fields", "reader_is_new_spatial")), "WO owner invariants differ")
    require(d["joint_base_sign_sweeps"] == (2 if mode == "JointSpatialWO" else 0)
        and d["spatial_sign_sweeps"] == 2, "fixed per-stage sweep contract differs")
    selected = metadata["selected_stages"]
    coverage = {(int(layer), int(head)) for axis, layer, head, stage in selected if axis == "v" and stage == 0}
    require(metadata["selected_stage_count"] == len(selected) == 43
        and sum(item[0] == "k" for item in selected) == 23 and len(coverage) == 20,
        "original K23/V20 stage0 coverage differs")
    layers = d["layer_stats"]
    require(len(layers) == 28, "all-layer metadata required")
    units = []
    for layer in layers:
        if layer["head_stats"]:
            require(layer["frozen_base_unchanged_during_spatial"] is True, "spatial stage changed its final frozen layer base")
        for head in layer["head_stats"]:
            metric, spatial = head["metric_derivation"], head["WO"]
            geometry = metric["H_geometry"]
            key = (int(layer["layer"]), int(head["head"]))
            require(key in coverage and metric["H_dtype"] == "float64" and metric["H_shape"] == [128, 128]
                and metric["weight_shape"] == [3584, 3584] and metric["Gram_derived_rows"] == 25088
                and metric["query_head_group"] == list(range(key[1] * 7, key[1] * 7 + 7))
                and metric["retained_derived_metric_bytes_after_encoding"] == 0 and metric["weight_mutated"] is False,
                "native W_O geometry/coverage/transience differs")
            require(geometry["finite"] is True and geometry["symmetry_max_abs"] == 0
                and geometry["eigendecomposition_or_ridge_or_eigenvalue_clipping_applied"] is False,
                "scalar H geometry differs")
            require(spatial["sign_sweeps"] == 2 and spatial["CPU_sync_per_channel"] is False
                and spatial["max_token_fullH_increase"] <= 0, "spatial local-H search contract differs")
            joint = head.get("joint_fullH")
            h0, h1, h2 = (head[name] for name in ("fullH_before_original_R7", "fullH_after_joint", "fullH_after_spatial"))
            monotonic(h0, h1, "joint"); monotonic(h1, h2, "spatial")
            require(math.isclose(h1, spatial["fullH_before"], rel_tol=1e-12, abs_tol=1e-9)
                and math.isclose(h2, spatial["fullH_after"], rel_tol=1e-12, abs_tol=1e-9), "spatial H baseline is not actual final joint point")
            if mode == "JointSpatialWO":
                require(joint is not None and joint["joint_base_sign_sweeps"] == 2
                    and joint["CPU_sync_per_channel"] is False and joint["max_token_fullH_increase_vs_original_R7"] <= 0
                    and math.isclose(joint["fullH_before_original_R7"], h0, rel_tol=1e-12)
                    and math.isclose(joint["fullH_after_joint"], h1, rel_tol=1e-12), "guarded full-H joint baseline differs")
                l0, l1 = joint["MSE_before"], joint["MSE_after"]
                require(math.isclose(l1, spatial["MSE_before"], rel_tol=1e-12, abs_tol=1e-9), "unweighted L2 joint-final baseline differs")
            else:
                require(joint is None and h0 == h1, "SpatialWO must retain original R7 base, without joint stage")
                l0 = l1 = spatial["MSE_before"]
            predictors = spatial["predictor_diagnostics"]
            require(len(predictors) == 2, "P0/P1 search diagnostics required")
            for predictor in predictors:
                monotonic(predictor["diagonal_seed_fullH_sum"], predictor["fullH_amplitude_seed_fullH_sum"], "full-H amplitude proposal")
                monotonic(predictor["fullH_amplitude_seed_fullH_sum"], predictor["after_sign_refit_fullH_sum"], "fixed predictor sign refit")
            unit = {"layer": key[0], "head": key[1], "H_original_R7": h0, "H_guarded_joint_final": h1, "H_spatial_final": h2,
                "half_H_original_R7": .5 * h0, "half_H_guarded_joint_final": .5 * h1, "half_H_spatial_final": .5 * h2,
                "L2_SSE_original_R7": l0, "L2_SSE_guarded_joint_final": l1, "L2_SSE_spatial_final": spatial["MSE_after"],
                "deployed_mode1_tokens": spatial["mode1_tokens"], "covered_token_denominator": n,
                "joint_base_sign_sweeps": 2 if joint else 0, "spatial_sign_sweeps_per_predictor": 2,
                "joint_channel_visits_from_recorded_sweeps": n * 128 * 2 if joint else 0,
                "joint_four_level_values_evaluated_from_recorded_sweeps": n * 128 * 2 * 4 if joint else 0,
                "spatial_flip_channel_visits_across_P0_P1_from_recorded_sweeps": n * 128 * 2 * 2,
                "joint_base_bits_changed_vs_original_R7": joint["base_bits_changed"] if joint else 0,
                "joint_base_sign_coordinate_updates_search": joint["base_sign_coordinate_updates"] if joint else 0,
                "MSE_JointV_proposal_accepted_tokens_under_H": joint["MSE_proposal_accepted_tokens"] if joint else 0,
                "spatial_P0_sign_flip_updates_search": predictors[0]["accepted_coordinate_flips"],
                "spatial_P1_sign_flip_updates_search": predictors[1]["accepted_coordinate_flips"],
                "spatial_P0_diagonal_seed_H": predictors[0]["diagonal_seed_fullH_sum"],
                "spatial_P0_fullH_amplitude_seed_H": predictors[0]["fullH_amplitude_seed_fullH_sum"],
                "spatial_P0_after_sign_refit_H": predictors[0]["after_sign_refit_fullH_sum"],
                "spatial_P1_diagonal_seed_H": predictors[1]["diagonal_seed_fullH_sum"],
                "spatial_P1_fullH_amplitude_seed_H": predictors[1]["fullH_amplitude_seed_fullH_sum"],
                "spatial_P1_after_sign_refit_H": predictors[1]["after_sign_refit_fullH_sum"],
                "joint_max_recorded_token_H_increase": joint["max_token_fullH_increase_vs_original_R7"] if joint else 0.,
                "spatial_max_recorded_token_H_increase": spatial["max_token_fullH_increase"],
                "Gram_host_seconds_including_geometry_sync": metric["host_Gram_seconds_including_synchronizing_geometry_not_controlled_GPU_timing"],
                "Gram_accumulation_FLOPs_from_metric_geometry": metric["Gram_accumulation_FLOPs"],
                "H_trace": geometry["trace"], "H_diag_min": geometry["diag_min"], "H_diag_max": geometry["diag_max"],
                "H_diag_max_over_min": geometry["diag_max_over_min"],
                "H_offdiag_Frobenius_over_total_Frobenius": geometry["offdiag_Frobenius_over_total_Frobenius"],
                "single_H_bytes": metric["H_bytes"], "single_FP64_weight_block_bytes": metric["maximum_single_FP64_weight_block_bytes"],
                "major_WO_workspace_bound_excluding_BLAS_bytes": spatial["encoding_workspace_major_tensor_bound_bytes_excluding_BLAS_allocator"],
                "joint_search_seconds": None, "spatial_search_seconds": None,
                "stage_search_timing_status": "NOT_INSTRUMENTED"}
            require(0 <= unit["deployed_mode1_tokens"] <= n, "mode count/covered denominator differs")
            units.append(unit)
    require(len(units) == len({(unit["layer"], unit["head"]) for unit in units}) == 20
        and {(unit["layer"], unit["head"]) for unit in units} == coverage, "exact twenty distinct V units required")
    sum_keys = ("H_original_R7", "H_guarded_joint_final", "H_spatial_final", "half_H_original_R7", "half_H_guarded_joint_final", "half_H_spatial_final",
        "L2_SSE_original_R7", "L2_SSE_guarded_joint_final", "L2_SSE_spatial_final", "deployed_mode1_tokens", "covered_token_denominator",
        "joint_base_bits_changed_vs_original_R7", "joint_base_sign_coordinate_updates_search", "MSE_JointV_proposal_accepted_tokens_under_H",
        "spatial_P0_sign_flip_updates_search", "spatial_P1_sign_flip_updates_search", "Gram_host_seconds_including_geometry_sync",
        "joint_channel_visits_from_recorded_sweeps", "joint_four_level_values_evaluated_from_recorded_sweeps",
        "spatial_flip_channel_visits_across_P0_P1_from_recorded_sweeps", "Gram_accumulation_FLOPs_from_metric_geometry")
    request = {key: sum(unit[key] for unit in units) for key in sum_keys}
    request.update({"native_visual_tokens": n, "covered_V_units": 20, "full_B0_bytes": expected,
        "mode1_fraction_of_covered_tokens": request["deployed_mode1_tokens"] / (20 * n),
        "v_base_int32_words_changed_vs_original_R7": sum(layer["changed_base_words"] for layer in layers),
        "residual_sign_int32_words_changed_vs_original_R7": sum(layer["changed_sign_words"] for layer in layers),
        "alpha_Half_elements_changed_by_int16_identity_vs_original_R7": sum(layer["changed_alpha_bits"] for layer in layers),
        "WO_layer_host_seconds_including_synchronizing_diagnostics": sum(layer["host_seconds_including_synchronizing_diagnostics"] for layer in layers),
        "max_layer_transient_H_bytes": max(layer.get("maximum_layer_transient_H_bytes", 0) for layer in layers),
        "max_major_WO_workspace_bound_excluding_BLAS_bytes": max(unit["major_WO_workspace_bound_excluding_BLAS_bytes"] for unit in units),
        "max_single_H_bytes": max(unit["single_H_bytes"] for unit in units),
        "max_single_FP64_weight_block_bytes": max(unit["single_FP64_weight_block_bytes"] for unit in units),
        "captured_from_pretrained_calls_per_child": capture["actual_from_pretrained_calls"],
        "captured_model_parameter_count": capture["actual_model_parameter_count"],
        "persistent_W_or_H_derived_tensor_bytes": 0, "joint_base_sign_sweeps": 2 if mode == "JointSpatialWO" else 0,
        "spatial_sign_sweeps_per_predictor": 2, "joint_search_seconds": None, "spatial_search_seconds": None,
        "stage_search_timing_status": "NOT_INSTRUMENTED"})
    return request, units


def bind_view(entry, point, harvest_root):
    dataset, mode = entry["dataset"], entry["mode"]
    view = harvest_root / "views" / (dataset + "__" + mode)
    names = ("summary.json", "source_lineage.json", "selected_records.jsonl", "scored_records.jsonl")
    paths = {name: view / name for name in names}
    require(all(path.is_file() for path in paths.values()), "completed view missing: " + str(view))
    summary, lineage = read(paths["summary.json"]), read(paths["source_lineage.json"])
    require(summary == point and summary["status"] == "PASS_FULL_SOURCE_GROUP_SCORE", "mixed/failed summary and harvest roots")
    selected_lines, scored_lines = raw_lines(paths["selected_records.jsonl"]), raw_lines(paths["scored_records.jsonl"])
    selected, scored = [json.loads(line) for line in selected_lines], [json.loads(line) for line in scored_lines]
    require(len(selected) == len(scored) == point["expected_QA"] == COUNTS[dataset]
        and [row["id"] for row in selected] == [row["id"] for row in scored] == lineage["ids"]
        and all(row["status"] == "PASS" for row in selected) and all(row["score_status"] == "SCORED_PASS" for row in scored),
        "complete selected/scored identity differs")
    hashes = {name: sha(path) for name, path in paths.items()}
    require(hashes["source_lineage.json"] == summary["source_lineage_sha256"]
        and hashes["selected_records.jsonl"] == lineage["selected_records_sha256"]
        and hashes["scored_records.jsonl"] == summary["scored_records_sha256"], "selected/scored/lineage SHA binding differs")
    raw = Path(lineage["source_raw_dir"])
    if entry["kind"] == "fresh":
        manifest, done, config, proof = harvest.check_fresh_proof(raw, entry["item"])
        receipt_path = raw / harvest.PROOF_NAME
        require(lineage["fresh_inference_source"] is True and lineage["reused_reference_source"] is False
            and lineage["source_driver"] == "code/round11_wo_runner_v1.py", "fresh WO source relabelled")
    else:
        import round10_harvest_score_v1 as original
        require(raw == Path(entry["source_raw_dir"]), "reused original raw source changed")
        manifest, done, config, proof = original.check_local_proof(raw, entry["source_item"])
        receipt_path = raw / "source_retrieval_receipt_round10_v1.json"
        require(sha(receipt_path) == entry["source_receipt_sha256"] and lineage["fresh_inference_source"] is False
            and lineage["reused_reference_source"] is True and lineage["source_driver"] == "code/round10_runner_v1.py",
            "old reference producer/reuse provenance changed")
    require(proof["parent_cell_success"] is True and done["status"] == "PASS_RAW_COMPLETE"
        and done["coverage_complete"] is True and done["quality_scored"] is False and manifest["references_read"] is False,
        "source is not closed exit0 raw inference")
    require(sha(receipt_path) == lineage["source_receipt_sha256"] and proof["parent_producer"] == lineage["source_producer"]
        and manifest["source_hashes"] == lineage["source_runtime_sha256"], "source receipt/runtime provenance differs")
    for name, digest in lineage["source_raw_sha256"].items():
        require(sha(raw / name) == digest, "original producer raw SHA changed")
    require(done["manifest_sha256"] == sha(raw / "manifest.json") and done["records_jsonl_sha256"] == sha(raw / "records.jsonl"), "raw3 terminal binding differs")
    producer_lines = raw_lines(raw / "records.jsonl")
    require(len(lineage["source_lines"]) == len(selected_lines), "source line count differs")
    for chosen, parsed, source_line, score in zip(selected_lines, selected, lineage["source_lines"], scored):
        number = source_line["line_number"]
        require(1 <= number <= len(producer_lines) and chosen == producer_lines[number - 1]
            and hashlib.sha256(chosen).hexdigest() == source_line["exact_line_sha256"]
            and source_line["id"] == parsed["id"]
            and score["source_raw_jsonl_line_number"] == number
            and score["source_raw_jsonl_exact_line_sha256"] == source_line["exact_line_sha256"], "selected/scored exact producer line binding differs")
    return selected, manifest, {"view_file_sha256": hashes, "source_producer": lineage["source_producer"],
        "source_driver": lineage["source_driver"], "source_raw_sha256": lineage["source_raw_sha256"],
        "source_receipt_sha256": sha(receipt_path), "source_runtime_sha256": lineage["source_runtime_sha256"],
        "fresh_inference_source": lineage["fresh_inference_source"], "reused_reference_source": lineage["reused_reference_source"],
        "runtime_process": manifest.get("runtime_process")}


def timing_resource_point(dataset, mode, records, manifest, provenance):
    timing_names = ("prefill_ms", "encode_ms", "question_prefill_ms", "ttft_ms", "wall_ms")
    return {"dataset": dataset, "mode": mode, "count": len(records), "source_provenance": provenance,
        "timing_ms": {key: stats(row[key] for row in records) for key in timing_names},
        "actual_request_allocator_bytes": {key: stats(row[key] for row in records)
            for key in ("peak_allocated_bytes", "peak_reserved_bytes", "end_allocated_bytes")},
        "native_visual_tokens": stats(row["actual_visual_token_count"] for row in records),
        "complete_visual_owner_bytes": stats(row["complete_visual_owner_bytes"] for row in records),
        "timing_scope_from_producer": manifest["timing_scope"],
        "timing_is_controlled_speed_comparison": False,
        "peak_scope": "actual whole per-request allocator peak, including resident model/native prefix/encoder/generation; not H-workspace-only",
        "quality_scores_or_prediction_changes_analysed": False}


def aggregate_encoder(requests, units):
    sum_names = ("H_original_R7", "H_guarded_joint_final", "H_spatial_final", "half_H_original_R7", "half_H_guarded_joint_final", "half_H_spatial_final",
        "L2_SSE_original_R7", "L2_SSE_guarded_joint_final", "L2_SSE_spatial_final", "deployed_mode1_tokens", "covered_token_denominator",
        "joint_base_bits_changed_vs_original_R7", "joint_base_sign_coordinate_updates_search", "MSE_JointV_proposal_accepted_tokens_under_H",
        "spatial_P0_sign_flip_updates_search", "spatial_P1_sign_flip_updates_search", "v_base_int32_words_changed_vs_original_R7",
        "residual_sign_int32_words_changed_vs_original_R7", "alpha_Half_elements_changed_by_int16_identity_vs_original_R7",
        "joint_channel_visits_from_recorded_sweeps", "joint_four_level_values_evaluated_from_recorded_sweeps",
        "spatial_flip_channel_visits_across_P0_P1_from_recorded_sweeps", "Gram_accumulation_FLOPs_from_metric_geometry")
    total = {key: sum(row[key] for row in requests) for key in sum_names}
    total["mode1_fraction_of_covered_tokens"] = total["deployed_mode1_tokens"] / total["covered_token_denominator"]
    total["H_relative_decrease_joint_vs_original_R7"] = decrease(total["H_original_R7"], total["H_guarded_joint_final"])
    total["H_relative_decrease_spatial_vs_joint_final"] = decrease(total["H_guarded_joint_final"], total["H_spatial_final"])
    total["H_relative_decrease_final_vs_original_R7"] = decrease(total["H_original_R7"], total["H_spatial_final"])
    total["L2_SSE_relative_decrease_final_vs_original_R7_may_be_negative"] = decrease(total["L2_SSE_original_R7"], total["L2_SSE_spatial_final"])
    total["L2_SSE_increased_request_count"] = sum(row["L2_SSE_spatial_final"] > row["L2_SSE_original_R7"] for row in requests)
    total["joint_recorded_max_token_H_increase"] = max(unit["joint_max_recorded_token_H_increase"] for unit in units)
    total["spatial_recorded_max_token_H_increase"] = max(unit["spatial_max_recorded_token_H_increase"] for unit in units)
    total["H_geometry_ranges"] = {key: stats(unit[key] for unit in units if unit[key] is not None)
        for key in ("H_trace", "H_diag_min", "H_diag_max", "H_diag_max_over_min", "H_offdiag_Frobenius_over_total_Frobenius")}
    total["prototype_host_seconds"] = {key: stats(row[key] for row in requests)
        for key in ("WO_layer_host_seconds_including_synchronizing_diagnostics", "Gram_host_seconds_including_geometry_sync")}
    total["transient_estimate_bytes_not_allocator_peak"] = {key: stats(row[key] for row in requests)
        for key in ("max_layer_transient_H_bytes", "max_major_WO_workspace_bound_excluding_BLAS_bytes", "max_single_H_bytes", "max_single_FP64_weight_block_bytes")}
    total["fixed_predictor_search_diagnostics_only_not_deployed_mode_mix"] = {
        predictor: {name: sum(unit[predictor + "_" + name] for unit in units)
            for name in ("diagonal_seed_H", "fullH_amplitude_seed_H", "after_sign_refit_H")}
        for predictor in ("spatial_P0", "spatial_P1")}
    total.update({"covered_V_layer_heads": 20, "all_V_layer_heads": 112, "H_definition": "sum_v e_v^T H_unit e_v; no factor0.5",
        "half_H_definition": "exactly0.5 times the separately reported H quadratic; same baselines",
        "L2_definition": "unweighted sum of squared source-minus-actualHalf decoder values over covered V20 only",
        "H_and_L2_are_not_whole_model_or_task_loss": True,
        "original_baseline": "original frozen R7 V20 reconstruction",
        "joint_final_baseline": "full-H guarded joint-final (original point plus H-gated MSE proposal and2 four-level base/sign sweeps), not unconditional MSE-JointV-final",
        "spatial_baseline": "the respective final frozen base/current fields; H-joint-final for JointSpatialWO, original R7 for SpatialWO",
        "mode_fraction_denominator": "sum over requests of20*actual_visual_token_count; not all112 V units",
        "changed_alpha_count_semantics": "changed_alpha_bits in frozen factory counts unequal int16 Half elements including mode signbit; not bit-popcount or magnitude-only changes",
        "changed_word_count_semantics": "changed base/sign words are unequal packed int32 words, distinct from joint raw base-bit count",
        "H_geometry_repetition_scope": "model-derived per-unit H is fixed for the frozen model and repeated across requests; observation counts are not independent learned matrices",
        "search_update_count_semantics": "coordinate updates along evaluated search paths; P0/P1 diagnostics include the predictor not finally deployed",
        "operation_count_semantics": "channel visits and four-level values are derived from actual recorded2 sweeps, D128, N and2 predictors; not measured GPU kernel counts or throughput",
        "joint_search_seconds": None, "spatial_search_seconds": None, "stage_search_timing_status": "NOT_INSTRUMENTED",
        "no_inferred_stage_time_subtraction": True, "persistent_W_or_H_derived_tensor_bytes": 0,
        "model_capture_count_is_per_child_not_sum_over_requests": True,
        "captured_from_pretrained_calls_per_child": stats(row["captured_from_pretrained_calls_per_child"] for row in requests),
        "captured_model_parameter_count": MODEL_PARAMETER_COUNT})
    return total


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def write_rows(path, values):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, keys, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def analyse(harvest_root, summary_root, output):
    harvest_root, summary_root, output = map(Path, (harvest_root, summary_root, output))
    require(not output.exists() and output.resolve().is_relative_to(Path("${EVIDENCE_ROOT}/results/round11").resolve()), "unique new E Round11 output required")
    summary = read(summary_root / "summary.json")
    initial_summary_sha = sha(summary_root / "summary.json")
    require(summary["schema"] == "trajkv.round11_wo.harvest_summary.v1" and summary["status"] == "PASS_ALL_20_PROVEN_VIEWS"
        and summary["expected_cells_or_views"] == summary["closed_cells_or_views"] == summary["successful_complete_cells_or_views"] == 20
        and summary["expected_new_free_generation_QA"] == summary["actual_closed_new_free_generation_QA"] == 192
        and summary["expected_reused_reference_views"] == summary["actual_closed_reused_reference_views"] == 288,
        "final all20/192fresh/288reused summary required; no partial stage analysis")
    require(sha(ROOT / "ops/round11_wo_harvest_v1.py") == HARVEST_SHA, "frozen WO provenance guard changed")
    lock_path = PLAN_PATH.with_name("WO_SOURCE_LOCK_v1.json")
    require(sha(lock_path) == SOURCE_LOCK_SHA, "frozen WO source lock changed")
    plan, group = harvest.load_plan(PLAN_PATH)
    initial_plan_sha = sha(PLAN_PATH)
    require(summary["plan_sha256"] == initial_plan_sha, "summary/actual WO plan identity differs")
    frozen = {(point["dataset"], point["mode"]): point for point in summary["points"]}
    entries = {(entry["dataset"], entry["mode"]): entry for entry in plan["entries"]}
    require(len(frozen) == len(entries) == 20 and set(frozen) == set(entries) == {(dataset, mode) for dataset in DOMAINS for mode in MODES}, "exact20 point coverage required")
    points, request_rows, unit_rows = [], [], []
    all_provenance = {}
    for dataset in DOMAINS:
        baseline = None
        for mode in MODES:
            records, manifest, provenance = bind_view(entries[dataset, mode], frozen[dataset, mode], harvest_root)
            require([row["id"] for row in records] == group["datasets"][dataset]["ids"], "exact D96 source order changed")
            point = timing_resource_point(dataset, mode, records, manifest, provenance)
            all_provenance[dataset + "__" + mode] = provenance
            if mode == "R7low":
                baseline = {row["id"]: (row["actual_visual_token_count"], row["complete_visual_owner_bytes"]) for row in records}
            if mode in WO_MODES:
                requests, units = [], []
                for row in records:
                    n = row["actual_visual_token_count"]; expected = 4590 * n + 89733
                    require(baseline is not None and baseline[row["id"]] == (n, expected)
                        and row["complete_visual_owner_bytes"] == expected
                        and charged_visual(row["owner_before_suffix"]) == charged_visual(row["owner_after_generation"]) == expected,
                        "fresh WO or paired original R7low complete B0 differs")
                    d = row["r7_bridge"]["round11_wo_candidate"]
                    require(d["model_parameters_unchanged_during_encoding"] is True, "encoder mutated model parameter identities")
                    request, per_unit = encoder_metadata(row["r7_bridge"], mode, n, d["actual_model_capture"])
                    budget = harvest.runner.round11_budget_guard(mode, n, expected, row["r7_bridge"], manifest["config"]["variant_source"]["name"])
                    require(row["round10_budget_vs_original_R7_low"] == budget, "actual executed WO budget guard differs")
                    request.update(dataset=dataset, mode=mode, id=row["id"], source_cluster_id=row["source_cluster_id"],
                        encode_ms=row["encode_ms"], ttft_ms=row["ttft_ms"], wall_ms=row["wall_ms"],
                        peak_allocated_bytes=row["peak_allocated_bytes"], peak_reserved_bytes=row["peak_reserved_bytes"])
                    for unit in per_unit:
                        unit.update(dataset=dataset, mode=mode, id=row["id"], source_cluster_id=row["source_cluster_id"])
                    requests.append(request); units.extend(per_unit)
                point["encoder"] = aggregate_encoder(requests, units)
                request_rows.extend(requests); unit_rows.extend(units)
            points.append(point)
    require(len(points) == 20 and len(request_rows) == 192 and len(unit_rows) == 192 * 20, "complete derived metadata counts differ")
    fresh_pids = {point["source_provenance"]["runtime_process"]["PID"] for point in points if point["mode"] in WO_MODES}
    require(len(fresh_pids) == 8, "fresh WO cells must retain eight distinct executed child processes")
    require(sha(summary_root / "summary.json") == initial_summary_sha and sha(PLAN_PATH) == initial_plan_sha
        and sha(lock_path) == SOURCE_LOCK_SHA, "closed stage inputs changed during derived analysis")
    for key, provenance in all_provenance.items():
        for name, digest in provenance["view_file_sha256"].items():
            require(sha(harvest_root / "views" / key / name) == digest, "immutable view bytes changed during analysis")
    # All admission/provenance/statistical computation above is read-only.
    # Only a complete result may create this new immutable artifact directory.
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "per_request_encoder.jsonl", request_rows)
    write_rows(output / "per_request_unit_encoder.jsonl", unit_rows)
    view_csv = []
    for point in points:
        row = {"dataset": point["dataset"], "mode": point["mode"], "requests": point["count"],
            **{key + "_median": value["median"] for key, value in point["timing_ms"].items()},
            "actual_request_peak_allocated_bytes_max": point["actual_request_allocator_bytes"]["peak_allocated_bytes"]["max"],
            "actual_request_peak_reserved_bytes_max": point["actual_request_allocator_bytes"]["peak_reserved_bytes"]["max"],
            "complete_visual_owner_bytes_mean": point["complete_visual_owner_bytes"]["mean"]}
        if "encoder" in point:
            e = point["encoder"]
            row.update({key: e[key] for key in ("H_original_R7", "H_guarded_joint_final", "H_spatial_final", "half_H_original_R7", "half_H_guarded_joint_final", "half_H_spatial_final",
                "L2_SSE_original_R7", "L2_SSE_guarded_joint_final", "L2_SSE_spatial_final", "mode1_fraction_of_covered_tokens",
                "v_base_int32_words_changed_vs_original_R7", "residual_sign_int32_words_changed_vs_original_R7", "alpha_Half_elements_changed_by_int16_identity_vs_original_R7")})
            row.update(joint_search_seconds=None, spatial_search_seconds=None, stage_search_timing_status="NOT_INSTRUMENTED")
        view_csv.append(row)
    write_csv(output / "per_view_resources_and_local_metric.csv", view_csv)
    grouped = defaultdict(list)
    for row in unit_rows:
        grouped[row["dataset"], row["mode"], row["layer"], row["head"]].append(row)
    unit_csv = []
    for (dataset, mode, layer, head), rows in grouped.items():
        numeric_sums = ("H_original_R7", "H_guarded_joint_final", "H_spatial_final", "half_H_original_R7", "half_H_guarded_joint_final", "half_H_spatial_final",
            "L2_SSE_original_R7", "L2_SSE_guarded_joint_final", "L2_SSE_spatial_final", "deployed_mode1_tokens", "covered_token_denominator",
            "joint_base_bits_changed_vs_original_R7", "joint_base_sign_coordinate_updates_search", "spatial_P0_sign_flip_updates_search", "spatial_P1_sign_flip_updates_search",
            "joint_channel_visits_from_recorded_sweeps", "joint_four_level_values_evaluated_from_recorded_sweeps",
            "spatial_flip_channel_visits_across_P0_P1_from_recorded_sweeps", "Gram_accumulation_FLOPs_from_metric_geometry",
            "Gram_host_seconds_including_geometry_sync", "spatial_P0_diagonal_seed_H", "spatial_P0_fullH_amplitude_seed_H", "spatial_P0_after_sign_refit_H",
            "spatial_P1_diagonal_seed_H", "spatial_P1_fullH_amplitude_seed_H", "spatial_P1_after_sign_refit_H")
        row = {"dataset": dataset, "mode": mode, "layer": layer, "head": head, "requests": len(rows),
            **{key: sum(item[key] for item in rows) for key in numeric_sums}}
        for key in ("H_trace", "H_diag_min", "H_diag_max", "H_diag_max_over_min", "H_offdiag_Frobenius_over_total_Frobenius"):
            values = [item[key] for item in rows if item[key] is not None]
            row[key + "_min"] = min(values) if values else None; row[key + "_max"] = max(values) if values else None
        unit_csv.append(row)
    write_csv(output / "per_domain_unit_local_metric.csv", unit_csv)
    result = {"schema": "trajkv.round11.WO_derived_stage_analysis.v1", "status": "PASS_COMPLETE_WO192_METADATA_ANALYSIS",
        "created_utc": datetime.now(timezone.utc).isoformat(), "actual_new_free_QA": 192, "reused_reference_views": 288,
        "view_count": 20, "WO_encoder_views": 8, "request_unit_rows": 3840, "domains_not_pooled": True,
        "actual_fresh_child_processes": 8, "captured_loader_calls_per_fresh_child": 1,
        "captured_model_instances_over_eight_distinct_children": 8,
        "source_hashes": {"summary": initial_summary_sha, "plan": initial_plan_sha, "source_lock": SOURCE_LOCK_SHA,
            "analysis_script": sha(__file__), "provenance_guard": HARVEST_SHA},
        "points": points, "all20_view_provenance": all_provenance,
        "artifact_sha256": {path.name: sha(path) for path in output.iterdir() if path.is_file()},
        "no_inference_fitting_GPU_SSH_source_edits_or_R128_outcome_reads": True,
        "private_logits_or_weight_tensors_read_or_exported": False,
        "quality_scores_prediction_or_first_argmax_changes_analysed": False,
        "local_metric_scope": "covered V20 per-unit H quadratic and unweighted L2 only; not full-model, attention-output or task loss",
        "whole_request_allocator_peak_is_not_transient_workspace_estimate": True,
        "joint_and_spatial_search_wall_time": "NOT_INSTRUMENTED; null, no inferred subtraction, no profiling rerun",
        "timing_is_controlled_speed_or_HBM_comparison": False}
    write_json(output / "analysis.json", result)
    print(json.dumps({"status": result["status"], "points": 20, "WO_requests": 192, "output": str(output), "analysis_sha256": sha(output / "analysis.json")}))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--harvest-root", type=Path, required=True)
    p.add_argument("--summary-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    analyse(a.harvest_root, a.summary_root, a.output)


if __name__ == "__main__":
    main()
