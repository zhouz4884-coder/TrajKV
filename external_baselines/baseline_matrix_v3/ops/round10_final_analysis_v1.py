"""Final executed-scope statistics, only after complete fixed reserve224 closure.

No harvest/scoring producer is invoked. Development selection evidence and
reserve replication are separate tables; partial replication emits a failure
receipt with no success means. This script starts no GPU/network/Git action.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "ops"), str(ROOT / "code")]
import numpy as np
import round10_development_analysis_v1 as common

DOMAINS = common.DOMAINS
COUNTS = {"textvqa": 64, "infovqa": 64, "docvqa": 48, "chartqa": 48}
MODES = ("FP16", "R7low", "R7high", "V128-R43")
EXTENSION_DRIVER_SHA = "83b8c5003f891b92cd33ed7d3387852c50090709c7f5a0bfd0196f02d01007f1"


def readiness(summary, decision):
    """Pure gate; deliberately never returns partial quality values."""
    missing = []
    checks = ((summary.get("status") == "PASS_ALL_RESERVE224_REPLICATION_SOURCE_SCORE_PHYSICAL", "full reserve status"),
        (summary.get("selected_modes") == ["V128-R43"] == decision.get("selected_modes"), "fixed V128-only scope"),
        (summary.get("expected_cells") == summary.get("closed_cells") == 16, "all16 actual cells"),
        (summary.get("expected_answer_events") == summary.get("closed_answer_events") == 896, "all896 actual free answers"),
        (summary.get("actual_closed_unique_IDs") == summary.get("expected_distinct_replication_sources") == 224, "all224 source IDs"),
        (summary.get("D160_results_mixed_into_replication") is False, "separate reserve cohort"),
        (summary.get("new_reference_QA_required") is True, "fresh reserve references"),
        (summary.get("scientific_method_ids_and_calibration_parameters_unchanged") is True, "unchanged fixed scientific method"))
    missing.extend(label for good, label in checks if not good)
    points = summary.get("points", [])
    if len(points) != 16 or {(p.get("dataset"), p.get("mode")) for p in points} != {(d, m) for d in DOMAINS for m in MODES}:
        missing.append("exact16 matrix")
    if any(p.get("expected_QA") != COUNTS.get(p.get("dataset")) or p.get("primary_mean") is None or
           p.get("status") != "CLOSED_FULL_DOMAIN_CUSTOM_SOURCE_SCORE_PHYSICAL" for p in points):
        missing.append("whole-domain successful source/score/physical points")
    return missing


def rows_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def replication_cell(point, evidence, root):
    ds, mode = point["dataset"], point["mode"]
    n = COUNTS[ds]
    raw, scored = Path(point["raw"]).resolve(), Path(point["scored"]).resolve()
    common.require(raw.is_relative_to(root) and scored.is_relative_to(root) and raw.name == scored.name and
                   "extension" in raw.name.lower(), "explicit new reserve raw/scored paths")
    score = evidence.load(scored / "summary.json", point["summary_sha256"])
    common.require(score["status"] == "PASS_SCORED_COMPLETE" and score["expected_run_questions"] == n, "existing all-domain reserve score required")
    for name, expected in score["integrity"]["raw_files"].items():
        evidence.track(raw / name, expected)
    for name, expected in score["derived_file_hashes"].items():
        evidence.track(scored / name, expected)
    manifest, complete = evidence.load(raw / "manifest.json"), evidence.load(raw / "complete.json")
    cfg = manifest["config"]
    common.require(cfg["input_role"] == "reserved_unused_extension" and cfg["cohort_kind"] == "replication" and
        cfg["start"] == 0 and cfg["count"] == cfg["input_count"] == n and
        cfg["candidate"]["mode"] == ("original" if mode in ("R7low", "R7high") else mode) and
        complete["status"] == "PASS_RAW_COMPLETE" and complete["counts"] == {"PASS": n} and
        complete["recorded_QA"] == complete["expected_QA"] == n, "complete fresh reserve scope")
    common.require(manifest["source_hashes"] == cfg["expected_source_hashes"] and
        manifest["source_hashes"]["code/round10_extension_runner_v1.py"] == EXTENSION_DRIVER_SHA, "approved actual reserve driver/source graph")
    proof = evidence.load(raw / "source_retrieval_receipt_round10_v1.json", point["source_receipt_sha256"])
    common.require(proof["status"] == "PASS_ROUND10_SEQUENCE_CHILD_EXECUTED_SOURCE_SHA" and
        proof["remote_hashes_computed_independently"] is True and proof["parent_cell_success"] is True and
        proof["actual_driver"] == "code/round10_extension_runner_v1.py", "actual reserve parent-child/source closure")
    for item in proof["files"]:
        evidence.track(raw / item["file"], item["sha256"])
    lineage = cfg["extension_lineage"]
    parent = evidence.load(raw / "assets/extension_parent_config.json", lineage["parent_config_sha256"])
    allowed = {"inputs", "inputs_sha256", "input_count", "start", "count", "input_role", "cohort_kind", "scope",
               "expected_source_hashes", "extension_lineage", "input_manifest_sha256"}
    same = lambda record: {k: v for k, v in record.items() if k not in allowed}
    common.require(same(parent) == same(cfg) and parent["input_role"] == parent["cohort_kind"] == "development",
                   "reserve cannot change selected scientific method or parameters")
    refs_path = Path(score["integrity"]["refs_path"]).resolve()
    common.require(refs_path.is_relative_to(root) and refs_path.name == f"{ds}_reserve{n}_refs.jsonl", "exact frozen reserve reference cohort")
    evidence.track(refs_path, score["integrity"]["refs_sha256"])
    evidence.track(refs_path.parent / "source_lock.json", score["integrity"]["dataset_source_lock_sha256"])
    common.require(score["integrity"]["inputs_sha256"] == manifest["inputs_sha256"] == cfg["inputs_sha256"],
                   "reserve raw/scoring exact frozen input identity")
    evidence.track(raw / "source_inputs.jsonl", manifest["inputs_sha256"])
    refs = {r["id"]: r for r in map(json.loads, refs_path.read_text(encoding="utf-8").splitlines())}
    scores = {r["id"]: r for r in map(json.loads, (scored / "scored_records.jsonl").read_text(encoding="utf-8").splitlines())}
    common.require(len(refs) == len(scores) == n, "complete reserve references/scored cohort")
    for role, expected in score["integrity"]["metric_code_hashes"].items():
        evidence.track(ROOT / role, expected)
    evidence.track(ROOT / "code/score.py", score["integrity"]["scorer_sha256"])
    records = []
    with (raw / "records.jsonl").open("rb") as stream:
        for number, line in enumerate(stream, 1):
            r = json.loads(line)
            s = scores[r["id"]]
            common.require(r["status"] == "PASS" and s["score_status"] == "SCORED_PASS" and
                s["raw_jsonl_line_number"] == number and s["raw_jsonl_exact_line_sha256"] == hashlib.sha256(line).hexdigest(), "reserve exact raw/scored row binding")
            recomputed = common.metrics.score_question(ds, r["raw_prediction"], refs[r["id"]])
            common.require(abs(recomputed["score"] - s["score"]) < 1e-12 and recomputed["metric_id"] == s["metric_id"] and
                           recomputed["reference_texts"] == s["reference_texts"], "independent reserve frozen-metric recomputation")
            amount = common.physical_complete_bytes(r, manifest["model_spec"])
            nv = r["actual_visual_token_count"]
            if mode == "FP16":
                common.require(amount == 57352 * nv, "complete native owner including positions")
            elif mode == "R7low":
                common.require(amount == 4590 * nv + 89733, "complete original low owner")
            elif mode == "V128-R43":
                common.require(amount == 5038 * nv + 32389 and r["r7_bridge"]["selected_stage_count"] == 43,
                               "complete unchanged V128 representation")
            else:
                common.require(r["r7_bridge"]["returned_owners_unique_tensor_capacity_bytes"] == amount, "complete original high owner")
            common.require(abs(r["actual_effective_visual_bits"] - 16 * amount / r["full_kv_byte_denominator"]) < 1e-12 and
                r["original_full_prefix_released"] is True and r["original_FP16_visual_alias_retained"] is False and
                r["owner_after_generation"]["stored_visual_KV_bytes"] == amount, "reserve byte/bit/source release invariants")
            records.append({"id": r["id"], "dataset": ds, "mode": mode, "source_cluster_id": r["source_cluster_id"],
                "source_id": r["source_id"], "image_sha256": r["original_image_sha256"], "score": float(s["score"]),
                "raw_prediction": r["raw_prediction"], "normalized_prediction": s["normalized_prediction"],
                "input_identity_sha256": common.canonical(common.input_identity(r)), "N": nv,
                "complete_owner_bytes": amount, "actual_complete_bits": r["actual_effective_visual_bits"],
                "output_tokens": r["output_tokens"], **{key: r[key] for key in
                    ("prefill_ms", "encode_ms", "question_prefill_ms", "ttft_ms", "wall_ms")},
                "decode_interval_sum_ms": sum(r["inter_token_ms"]), "decode_intervals_ms": r["inter_token_ms"]})
    common.require(len(records) == n and [r["id"] for r in records] == manifest["expected_ids"] and
        len({r["source_cluster_id"] for r in records}) == len({r["image_sha256"] for r in records}) == n,
        "exact one-question/image reserve source cohort")
    quality = sum(r["score"] for r in records) / n
    common.require(abs(quality - point["primary_mean"]) < 1e-12 and abs(quality - score["primary_mean"]) < 1e-12, "reserve mean arithmetic")
    resource = {key: common.stats(r[key] for r in records) for key in ("complete_owner_bytes", "actual_complete_bits", "N")}
    timing = {key: common.stats(r[key] for r in records) for key in
        ("prefill_ms", "encode_ms", "question_prefill_ms", "ttft_ms", "decode_interval_sum_ms", "wall_ms", "output_tokens")}
    timing["pooled_natural_inter_token_ms"] = common.stats(v for r in records for v in r["decode_intervals_ms"])
    common.require(abs(resource["complete_owner_bytes"]["mean"] - point["resources"]["complete_bytes"]["mean"]) < 1e-8 and
        abs(resource["actual_complete_bits"]["mean"] - point["resources"]["actual_complete_bits"]["mean"]) < 1e-12,
        "reserve snapshot physical-resource arithmetic")
    for key in ("prefill_ms", "encode_ms", "question_prefill_ms", "ttft_ms", "wall_ms"):
        common.require(abs(timing[key]["mean"] - point["timing"][key]["mean"]) < 1e-8, "reserve snapshot timing arithmetic")
    return {"dataset": ds, "mode": mode, "QA": n, "quality": quality, "quality_percent": 100 * quality,
        "frozen_individual_CI95_percent": [100 * v for v in score["primary_ci95"]], "resources": resource, "timing": timing,
        "raw": str(raw), "scored": str(scored)}, records, common.canonical(common.native_context(manifest))


def calibration_events(root, evidence, technical_files):
    original = evidence.load(root / "cal64_summary_v1/summary.json")
    common.require(original["status"] == "PASS_LOCAL_SELECTOR_AND_COUNTER_RECOMPUTATION", "closed original calibration scalar review")
    for item in original["input_files"].values():
        evidence.track(item["path"], item["sha256"])
    original_done = evidence.load(root / "cal64_v1/complete.json")
    rows = [{"run": "original_cal64", "scope": "fit48+select16", "free_task_answers": 0,
        "new_fit_capture_sources": 48, "reused_fit_sources": 0, "fit_capture_outer_forwards": original_done["counts"]["fit_student_outer_forwards"],
        "teacher_generated_sources": original["teacher_generation_source_count_recomputed"],
        "teacher_generated_tokens": original["teacher_generation_token_total_producer_reported"],
        "logical_selection_TF_branches": original["logical_select_source_branches"],
        "actual_selection_TF_source_evaluations": original["actual_TF_sources"], "extra_original_identity_TF_source_evaluations": 0,
        "actual_student_TF_outer_forwards": original["actual_student_outer_forwards"],
        "exact_field_TF_reuses": original["exact_field_TF_reuses"], "selected_parameter": "lambda0.5; W64"}]
    kra = root / "kra_calibration_retrieved_v1"
    done, records = evidence.load(kra / "complete.json"), kra / "records.jsonl"
    common.require(done["status"] == "PASS_KRA_FIT48_REUSED_AND_ACTUAL_MODEL_SELECT16" and done["actual_GPU"] is True and
        done["new_fit_source_captures"] == 0 and done["fit_sources_reused"] == 48 and done["select_sources"] == 16 and
        done["new_free_generation_task_answers"] == 0, "independent KRA select16 closure")
    evidence.track(records, done["records_sha256"])
    events = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    teachers = [e for e in events if e["phase"] == "select_source_teacher"]
    branches = [e for e in events if e["phase"] == "select_branch"]
    common.require(len(teachers) == 16 and len(branches) == 64 and done["counts"]["teacher_generated_tokens"] == sum(len(e["teacher_token_ids"]) for e in teachers) and
        done["counts"]["TF_source_evaluations_actual"] == 64 and done["counts"]["original_identity_TF_source_evaluations_actual"] == 1,
        "KRA separate teacher/selection counters")
    identity = evidence.load(kra / "identity.json", done["identity_sha256"])
    actual_outer = sum(e["loss"]["positions"] for e in branches) + identity["original_R7_actual_TF"]["positions"]
    common.require(actual_outer == done["counts"]["TF_student_outer_forwards_actual"], "KRA full TF outer-forward counters")
    rows.append({"run": "KRA_calibration", "scope": "fit48 reused; select16 t0/.25/.5/1", "free_task_answers": 0,
        "new_fit_capture_sources": 0, "reused_fit_sources": 48, "fit_capture_outer_forwards": 0,
        "teacher_generated_sources": len(teachers), "teacher_generated_tokens": done["counts"]["teacher_generated_tokens"],
        "logical_selection_TF_branches": 64, "actual_selection_TF_source_evaluations": 64,
        "extra_original_identity_TF_source_evaluations": 1, "actual_student_TF_outer_forwards": actual_outer,
        "exact_field_TF_reuses": 0, "selected_parameter": str(done["selected_strength"])})
    for path in technical_files:
        receipt = evidence.load(path)
        common.require(receipt.get("actual_GPU") is True and receipt.get("new_free_generation_task_answers", 0) == 0,
                       "technical controls stay outside free-answer task counts")
        rows.append({"run": str(Path(path).parent.name), "scope": "QA0 technical control receipt", "free_task_answers": 0,
            "new_fit_capture_sources": 0, "reused_fit_sources": receipt.get("fit_sources_reused", 0), "fit_capture_outer_forwards": 0,
            "teacher_generated_sources": receipt.get("teacher_sources", receipt.get("counts", {}).get("teacher_generated_sources")),
            "teacher_generated_tokens": receipt.get("teacher_tokens", receipt.get("counts", {}).get("teacher_generated_tokens")),
            "logical_selection_TF_branches": None,
            "actual_selection_TF_source_evaluations": receipt.get("actual_TF_source_evaluations", receipt.get("counts", {}).get("TF_source_evaluations_actual")),
            "extra_original_identity_TF_source_evaluations": receipt.get("counts", {}).get("original_identity_TF_source_evaluations_actual", 0),
            "actual_student_TF_outer_forwards": receipt.get("counts", {}).get("TF_student_outer_forwards_actual"),
            "exact_field_TF_reuses": 0, "selected_parameter": "none (technical receipt)"})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("${EVIDENCE_ROOT}/results/round10"))
    parser.add_argument("--extension-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--technical-complete", type=Path, action="append", default=[])
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    common.require(output.is_relative_to(root) and not output.exists(), "new final analysis output on explicit data disk")
    output.mkdir(parents=True)
    evidence = common.Evidence()
    decision = evidence.load(root / "selection_freeze_v1/DECISION.json")
    if not args.extension_summary.is_file():
        common.write(output / "complete.json", {"status": "FAIL_NOT_FINAL_EXTENSION_INCOMPLETE", "missing": ["extension summary"],
            "success_quality_means": None, "new_GPU_jobs": 0, "partial_is_not_final": True})
        return 3
    extension = evidence.load(args.extension_summary)
    missing = readiness(extension, decision)
    if missing:
        common.write(output / "complete.json", {"status": "FAIL_NOT_FINAL_EXTENSION_INCOMPLETE", "missing": missing,
            "success_quality_means": None, "new_GPU_jobs": 0, "partial_is_not_final": True})
        return 3
    try:
        development = root / "development_analysis_v1"
        for receipt_path in (development / "complete.json", development / "kra_addendum_v1/complete.json"):
            receipt = evidence.load(receipt_path)
            common.require(receipt["status"].startswith("PASS_"), "closed independently reviewed development analysis")
            for name, expected in receipt["outputs"].items():
                evidence.track((receipt_path.parent / name) if receipt_path.parent == development else development / name, expected)
        dev_counts = evidence.load(development / "combined_counts.json")
        common.require(dev_counts["unique_actual_runs"] == 40 and dev_counts["unique_actual_free_answer_events"] == 1600 and
            dev_counts["declared_views"] == 64 and dev_counts["declared_view_answer_events"] == 2560 and
            dev_counts["total_reused_reference_view_events"] == 960, "exact frozen development scope")
        dev_quality = rows_csv(development / "combined_quality_complete_owner.csv")
        dev_timing = rows_csv(development / "combined_instrumented_timing.csv")
        dev_pairs = rows_csv(development / "combined_paired_bootstrap.csv")
        for name in ("primary_D160_closed_v1", "middle_watch_snapshot_v2", "kra_watch_snapshot_v3"):
            evidence.track(root / name / "summary.json")
        pairs_frozen = evidence.load(args.extension_summary.parent / "paired_comparisons.json")
        cells, rows, contexts = [], {}, {}
        for ds in DOMAINS:
            for mode in MODES:
                point = next(p for p in extension["points"] if p["dataset"] == ds and p["mode"] == mode)
                cell, records, context = replication_cell(point, evidence, root)
                cells.append(cell)
                rows[ds, mode], contexts[ds, mode] = records, context
        replication_pairs = []
        for ds in DOMAINS:
            for anchor in ("FP16", "R7low", "R7high"):
                pair, _ = common.paired(ds, "V128-R43", anchor, rows, contexts)
                old = pairs_frozen[f"{ds}__V128-R43_minus_{anchor}"]["primary"]
                common.require(abs(pair["difference_pp"] - old["difference_pp"]) < 1e-10 and
                    abs(pair["CI95_low_pp"] - old["ci95_pp"][0]) < 1e-10 and abs(pair["CI95_high_pp"] - old["ci95_pp"][1]) < 1e-10,
                    "independent reserve paired-bootstrap arithmetic differs")
                replication_pairs.append(pair)
        primary = evidence.load(root / "primary_D160_closed_v1/summary.json")
        disjoint = []
        for ds in DOMAINS:
            point = next(p for p in primary["points"] if p["dataset"] == ds and p["mode"] == "FP16")
            path = Path(point["scored"]) / "scored_records.jsonl"
            evidence.track(path)
            dev = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            reserve = rows[ds, "FP16"]
            common.require(not {r["id"] for r in dev} & {r["id"] for r in reserve} and
                not {r["original_image_sha256"] for r in dev} & {r["image_sha256"] for r in reserve}, "development and reserve source/image overlap")
            disjoint.append({"dataset": ds, "development_sources": common.COUNTS[ds], "replication_sources": COUNTS[ds],
                             "overlapping_IDs": 0, "overlapping_original_images": 0})
        technical = [root / "caltech_v1/complete.json", root / "kra_identity_retrieved_v1/complete.json", *args.technical_complete]
        technical = list(dict.fromkeys(path.resolve() for path in technical if path.is_file()))
        calibrations = calibration_events(root, evidence, technical)
        quality_rows, timing_rows = [], []
        for c in cells:
            amount, bits = c["resources"]["complete_owner_bytes"], c["resources"]["actual_complete_bits"]
            quality_rows.append({"cohort": "reserved_unused_extension_replication", "dataset": c["dataset"], "mode": c["mode"],
                "QA": c["QA"], "quality_percent": c["quality_percent"], "frozen_individual_CI95_low_percent": c["frozen_individual_CI95_percent"][0],
                "frozen_individual_CI95_high_percent": c["frozen_individual_CI95_percent"][1],
                "complete_owner_bytes_mean": amount["mean"], "complete_owner_bytes_min": amount["min"], "complete_owner_bytes_max": amount["max"],
                "complete_bits_mean": bits["mean"], "complete_bits_min": bits["min"], "complete_bits_max": bits["max"]})
            timing_rows.append({"dataset": c["dataset"], "mode": c["mode"], "QA": c["QA"], **{
                f"{key}_{stat}": c["timing"][key][stat] for key in
                ("prefill_ms", "encode_ms", "question_prefill_ms", "ttft_ms", "decode_interval_sum_ms", "wall_ms", "output_tokens")
                for stat in ("mean", "median", "p90")},
                "pooled_natural_inter_token_ms_mean": c["timing"]["pooled_natural_inter_token_ms"]["mean"],
                "pooled_natural_inter_token_ms_count": c["timing"]["pooled_natural_inter_token_ms"]["count"]})
        counts = {"development_actual_runs": 40, "development_actual_free_answers": 1600, "development_sources": 160,
            "replication_actual_runs": 16, "replication_actual_free_answers": 896, "replication_sources": 224,
            "replication_new_reference_answers": 672, "replication_new_fixed_candidate_answers": 224,
            "stage_actual_new_runs": 56, "stage_actual_new_free_answers": 2496,
            "declared_view_cells": 80, "declared_view_answer_events": 3456, "reused_reference_view_events": 960,
            "free_answers_exclude_all_teacher_calibration_TF_and_QA0_controls": True,
            "development_and_replication_quality_never_pooled": True}
        common.require(sum(c["QA"] for c in cells) == 896 and 1600 + sum(c["QA"] for c in cells) == 2496,
                       "exact final executed scope2496")
        evidence.unchanged()
        common.write(output / "analysis.json", {"schema": "trajkv.round10.final_executed_analysis.v1", "status": "PASS_FINAL_EXECUTED_SCOPE_ANALYSIS",
            "counts": counts, "development": {"quality_rows": dev_quality, "paired_rows": dev_pairs, "timing_rows": dev_timing,
                "selection_evidence_not_confirmation": True}, "replication": {"cells": cells, "paired_comparisons": replication_pairs,
                "fixed_candidate_before_reserve_outputs": decision, "development_disjointness": disjoint},
            "calibration_and_technical_events_separate": calibrations,
            "technical_receipt_list_is_not_a_census_of_all_debug_kernel_calls": True,
            "no_cross_dataset_or_cross_cohort_quality_average": True,
            "unadjusted_paired_CIs_are_not_equivalence_or_algorithm_winner_certificates": True,
            "controlled_speed_or_HBM_traffic_claimed": False, "canonical_SourceRun_admission_claimed": False,
            "whole_research_goal_complete_claimed": False, "new_GPU_jobs": 0, "new_parameters_selected": False})
        common.csv_write(output / "development_10methods_quality.csv", dev_quality)
        common.csv_write(output / "development_paired_bootstrap.csv", dev_pairs)
        common.csv_write(output / "development_instrumented_timing.csv", dev_timing)
        common.csv_write(output / "replication_quality_complete_owner.csv", quality_rows)
        common.csv_write(output / "replication_paired_bootstrap.csv", replication_pairs)
        common.csv_write(output / "replication_instrumented_timing.csv", timing_rows)
        common.csv_write(output / "calibration_events_separate.csv", calibrations)
        common.write(output / "counts.json", counts)
        common.write(output / "source_evidence.json", {"files": evidence.files, "unchanged_after_analysis": True,
            "source_sha256": common.digest(__file__), "frozen_harvest_and_metric_producers_modified": False})
        qmap = {(c["dataset"], c["mode"]): c for c in cells}
        dmap = {(c["dataset"], c["mode"]): c for c in dev_quality}
        low_contrasts = {p["dataset"]: p for p in replication_pairs if p["anchor"] == "R7low"}
        low_delta_text = "/".join(f"{low_contrasts[ds]['difference_pp']:+.4f}" for ds in DOMAINS)
        report = ["# Round10 已执行阶段研究报告草稿", "", "本报告仅覆盖已完成的56个实际run、2496次新增free回答。"
            "D160开发面板为40个run/1600次，reserve224固定候选复测为16个run/896次；80个声明view共3456事件，"
            "其中960是共用参照，不是额外推理。Teacher生成、TF、fit capture与QA0控制另列，不计入free回答。", "",
            f"固定V128-R43在reserve224相对R7low的Text/Info/Doc/Chart变化为{low_delta_text} pp。"
            "这是单独复测cohort的域内差异，下面分别给出配对区间、完整容量和计时；开发结果不与之混合。", "",
            "## 方法", "", "先在D160 development比较原生FP16、两档冻结R7、四个V候选、两个中档和KRA1。"
            "Root在读取reserve模型输出前冻结仅V128-R43进入复测。V128保留K23和原f0，将V改为g128并重编原20V残差；"
            "AddOldOrder把近同费用用于原ranking更多槽；KRA只改原K23 alpha且另计5980B。机制证据不充当task因果证明。", "",
            "## D160 开发选择证据", "", "| 方法 | TextVQA | InfoVQA | DocVQA | ChartQA |", "|---|---:|---:|---:|---:|"]
        for mode in common.MODES + ("KRA1",):
            report.append("| " + mode + " | " + " | ".join(f"{float(dmap[ds, mode]['quality_percent']):.4f}" for ds in DOMAINS) + " |")
        report += ["", "分数为各域百分制原指标，未跨域平均。LS-FZ四域净0；Text有2题改善、1题变差，gross各1.5625 pp。"
            "KRA160条预测与token序列逐项同R7low、每题多5980B；零差异区间仅说明这些观察，不是模型或分布等价。", "",
            "## reserve224 固定候选复测", "", "| 数据集（N） | FP16 | R7low | R7high | V128-R43 |",
            "|---|---:|---:|---:|---:|"]
        for ds in DOMAINS:
            report.append(f"| {ds}（{COUNTS[ds]}） | " + " | ".join(f"{qmap[ds, mode]['quality_percent']:.4f}" for mode in MODES) + " |")
        report += ["", "V128相对各参照的配对差异（pp；固定seed20261119，2000次source bootstrap，未校正95%区间）：", "",
                   "| 数据集 | 参照 | Δpp | 95%CI | 改善/变差题数 |", "|---|---|---:|---|---:|"]
        for p in replication_pairs:
            report.append(f"| {p['dataset']} | {p['anchor']} | {p['difference_pp']:+.4f} | [{p['CI95_low_pp']:.4f}, {p['CI95_high_pp']:.4f}] | {p['improved_questions']}/{p['degraded_questions']} |")
        report += ["", "## 完整容量与时延", "", "| 数据集 | 模式 | 平均完整owner B | 平均bit | encode p50 ms | wall p50 ms |",
                   "|---|---|---:|---:|---:|---:|"]
        for c in cells:
            report.append(f"| {c['dataset']} | {c['mode']} | {c['resources']['complete_owner_bytes']['mean']:.1f} | "
                f"{c['resources']['actual_complete_bits']['mean']:.6f} | {c['timing']['encode_ms']['median']:.2f} | {c['timing']['wall_ms']['median']:.2f} |")
        report += ["", "完整owner含codebook、factors、position map、lookup/ranking/policy及所有新增资产；逐条storage容量与bit分母重算。"
            "CSV另列native prefill、question prefill、TTFT和自然decode间隔之和。质量run包含同步诊断及不同自然输出长度，"
            "这些host时间不是受控加速、SLO或HBM流量结果。", "", "## 校准事件与局限", "",
            f"原cal64复用48fit+16select闭合证据：teacher64源/285token，实际selection fullTF208次，精确字段复用16次，"
            f"selection student outer forward936次；另有fit capture {original_fit_count(root)}次outer forward。"
            "KRA另复用48fit而未重新capture，16select teacher72token，64个候选fullTF加1次原R7 identity TF、共292次student outer forward。"
            "完整事件表保留额外技术控制，不与free回答相加。", "",
            "D160用于选择，reserve224是冻结候选的单模型、指定cohort复测；不替代正式500全split、跨模型验证或canonical SourceRun准入。"
            "各域、各cohort均分别报告；配对CI为描述性未校正区间，跨0不是等价，不跨0也不自动成为算法胜者证书。"
            "私有native/teacher logits/PCA输入没有在此重放，来源身份与公开标量核验的边界保留。", "", "## 下一步", "",
            "将本草稿与Root补充的数学提案合并审阅，并对已封存公开包执行独立冷验证与既定增量交付。"
            "本汇总不重新选参数、不改变方法、不启动后续GPU实验。"]
        with (output / "REPORT_DRAFT.zh.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(report) + "\n")
        common.write(output / "complete.json", {"status": "PASS_FINAL_EXECUTED_SCOPE_ANALYSIS", "actual_new_free_QA": 2496,
            "actual_runs": 56, "declared_views": 80, "declared_view_events": 3456, "reference_view_reuses": 960,
            "development_and_replication_quality_kept_separate": True, "replication_raw_QA_independently_recomputed": 896,
            "source_sha256": common.digest(__file__), "outputs": {path.name: common.digest(path) for path in output.iterdir() if path.is_file()},
            "new_GPU_jobs": 0, "new_parameters_selected": False, "whole_goal_complete_claimed": False})
        print(json.dumps({"status": "PASS_FINAL_EXECUTED_SCOPE_ANALYSIS", "actual_new_free_QA": 2496,
                          "actual_runs": 56, "output": str(output)}, ensure_ascii=False))
        return 0
    except Exception as error:
        common.write(output / "complete.json", {"status": "FAIL_FINAL_INTEGRITY_CHECK", "error_type": type(error).__name__,
            "error": str(error), "success_quality_means": None, "partial_is_not_final": True, "new_GPU_jobs": 0})
        raise


def original_fit_count(root):
    return json.loads((root / "cal64_v1/complete.json").read_text(encoding="utf-8"))["counts"]["fit_student_outer_forwards"]


if __name__ == "__main__":
    raise SystemExit(main())
