"""Append independently checked KRA D160 results to the closed development review.

Original primary+middle analysis files remain byte-for-byte unchanged. No
reserved224 output is opened; the selection-freeze decision is read as scope.
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
import round10_development_analysis_v1 as base

KRA_FACTORY = "ede14d949ff905a15b25d83a928e8a79352065bc9850cc9eaf5759acff75503d"


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_kra(point, evidence, root, low_rows, low_context):
    ds, n = point["dataset"], base.COUNTS[point["dataset"]]
    raw, scored = Path(point["raw"]).resolve(), Path(point["scored"]).resolve()
    base.require(raw.parent == (root / "harvest_v1/raw").resolve() and scored.parent == (root / "harvest_v1/scored").resolve() and
                 raw.name == scored.name == f"KRA1_{ds}_{n}_v1", "exact KRA D160 run path")
    score = evidence.load(scored / "summary.json", point["summary_sha256"])
    base.require(score["status"] == "PASS_SCORED_COMPLETE" and score["expected_run_questions"] == n, "closed KRA full-domain score")
    for name, expected in score["integrity"]["raw_files"].items():
        evidence.track(raw / name, expected)
    for name, expected in score["derived_file_hashes"].items():
        evidence.track(scored / name, expected)
    manifest, complete = evidence.load(raw / "manifest.json"), evidence.load(raw / "complete.json")
    cfg = manifest["config"]
    base.require(cfg["input_role"] == cfg["cohort_kind"] == "development" and cfg["candidate"]["mode"] == "KRA1" and
                 complete["status"] == "PASS_RAW_COMPLETE" and complete["counts"] == {"PASS": n}, "KRA full development scope")
    base.require(base.canonical(base.native_context(manifest)) == low_context, "KRA/low actual native runtime context")
    proof = evidence.load(raw / "source_retrieval_receipt_round10_v1.json", point["source_receipt_sha256"])
    base.require(proof["status"] == "PASS_ROUND10_SEQUENCE_CHILD_EXECUTED_SOURCE_SHA" and
                 proof["remote_hashes_computed_independently"] is True and proof["parent_cell_success"] is True, "actual KRA child completion/source")
    for item in proof["files"]:
        evidence.track(raw / item["file"], item["sha256"])
    refs_path = Path(score["integrity"]["refs_path"]).resolve()
    base.require(refs_path.parent == (root / "input_preparation_v1/inputs").resolve() and
                 refs_path.name == f"{ds}_dev{n}_refs.jsonl", "exact dev-only reference path")
    evidence.track(refs_path, score["integrity"]["refs_sha256"])
    refs = {r["id"]: r for r in map(json.loads, refs_path.read_text(encoding="utf-8").splitlines())}
    scores = {r["id"]: r for r in map(json.loads, (scored / "scored_records.jsonl").read_text(encoding="utf-8").splitlines())}
    low = {r["id"]: r for r in low_rows}
    low_raw_dir = Path(next(r for r in json.loads((root / "primary_D160_closed_v1/summary.json").read_text())["points"]
                            if r["dataset"] == ds and r["mode"] == "R7low")["raw"])
    evidence.track(low_raw_dir / "records.jsonl")
    low_predictions = {r["id"]: (r["raw_prediction"], r["output_token_ids"]) for r in
                       map(json.loads, (low_raw_dir / "records.jsonl").read_text(encoding="utf-8").splitlines())}
    rows = []
    same_prediction = same_tokens = 0
    with (raw / "records.jsonl").open("rb") as stream:
        for number, line in enumerate(stream, 1):
            r = json.loads(line)
            s = scores[r["id"]]
            base.require(r["status"] == "PASS" and s["score_status"] == "SCORED_PASS" and s["raw_jsonl_line_number"] == number and
                         s["raw_jsonl_exact_line_sha256"] == hashlib.sha256(line).hexdigest(), "KRA raw/score exact row")
            fresh = base.metrics.score_question(ds, r["raw_prediction"], refs[r["id"]])
            base.require(abs(fresh["score"] - s["score"]) < 1e-12 and fresh["metric_id"] == s["metric_id"], "KRA frozen metric recomputation")
            amount = base.physical_complete_bytes(r, manifest["model_spec"])
            nv = r["actual_visual_token_count"]
            d = r["r7_bridge"]["round10_kra_candidate"]
            base.require(amount == 4590 * nv + 89733 + 5980 and amount - low[r["id"]]["complete_owner_bytes"] == 5980 and
                         d["source_sha256"] == KRA_FACTORY and d["actual_extra_persistent_bytes"] == d["new_persistent_constant_bytes"] == 5980 and
                         d["actual_H_after_worse_tokens"] == 0 and d["native_FP16_source_or_workspace_retained"] is False,
                         "KRA exact5980B fixed physical/source contract")
            base.require(abs(r["actual_effective_visual_bits"] - 16 * amount / r["full_kv_byte_denominator"]) < 1e-12 and
                         r["owner_after_generation"]["stored_visual_KV_bytes"] == amount and
                         r["original_full_prefix_released"] is True and r["original_FP16_visual_alias_retained"] is False,
                         "KRA byte/bit/read release invariant")
            same_prediction += r["raw_prediction"] == low_predictions[r["id"]][0]
            same_tokens += r["output_token_ids"] == low_predictions[r["id"]][1]
            rows.append({"id": r["id"], "dataset": ds, "mode": "KRA1", "source_cluster_id": r["source_cluster_id"],
                "source_id": r["source_id"], "image_sha256": r["original_image_sha256"], "score": float(s["score"]),
                "raw_prediction": r["raw_prediction"], "normalized_prediction": s["normalized_prediction"],
                "input_identity_sha256": base.canonical(base.input_identity(r)), "N": nv, "complete_owner_bytes": amount,
                "actual_complete_bits": r["actual_effective_visual_bits"], "output_tokens": r["output_tokens"],
                **{key: r[key] for key in ("prefill_ms", "encode_ms", "question_prefill_ms", "ttft_ms", "wall_ms")},
                "decode_interval_sum_ms": sum(r["inter_token_ms"]), "decode_intervals_ms": r["inter_token_ms"]})
    base.require(len(rows) == len(scores) == len(refs) == n and same_prediction == same_tokens == n and
                 [r["id"] for r in rows] == manifest["expected_ids"], "observed KRA160 prediction/token identity with R7low")
    mean = sum(r["score"] for r in rows) / n
    base.require(abs(mean - point["primary_mean"]) < 1e-12 and abs(mean - score["primary_mean"]) < 1e-12, "KRA score mean")
    resources = {key: base.stats(r[key] for r in rows) for key in ("complete_owner_bytes", "actual_complete_bits", "N")}
    timing = {key: base.stats(r[key] for r in rows) for key in
              ("prefill_ms", "encode_ms", "question_prefill_ms", "ttft_ms", "decode_interval_sum_ms", "wall_ms", "output_tokens")}
    timing["pooled_natural_inter_token_ms"] = base.stats(v for r in rows for v in r["decode_intervals_ms"])
    return {"dataset": ds, "mode": "KRA1", "QA": n, "quality": mean, "quality_percent": 100 * mean,
            "frozen_individual_CI95_percent": [100 * value for value in score["primary_ci95"]],
            "resources": resources, "timing": timing, "same_raw_prediction_count": same_prediction,
            "same_output_token_ids_count": same_tokens, "actual_extra_persistent_bytes": 5980}, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("${EVIDENCE_ROOT}/results/round10"))
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    root, analysis = args.root.resolve(), args.analysis.resolve()
    output = analysis / "kra_addendum_v1"
    base.require(analysis.parent == root and not output.exists(), "new additive KRA analysis under original review")
    output.mkdir()
    evidence = base.Evidence()
    prior_complete = evidence.load(analysis / "complete.json")
    base.require(prior_complete["status"] == "PASS_D160_DEVELOPMENT_ANALYSIS", "closed primary+middle review")
    for name, expected in prior_complete["outputs"].items():
        evidence.track(analysis / name, expected)
    snapshot = evidence.load(root / "kra_watch_snapshot_v3/summary.json")
    primary = evidence.load(root / "primary_D160_closed_v1/summary.json")
    decision = evidence.load(root / "selection_freeze_v1/DECISION.json")
    base.require(snapshot["status"] == "PASS_ALL16_D160_KRA_CUSTOM_SOURCE_SCORE_PHYSICAL" and
                 snapshot["expected_new_KRA1_answer_events"] == 160 and snapshot["reused_reference_answer_events"] == 480,
                 "closed KRA development scope")
    base.require(decision["selected_modes"] == ["V128-R43"] and decision["reserved224_model_outputs_read_at_freeze"] is False and
                 decision["D160_source"]["sha256"] == evidence.track(root / "kra_watch_snapshot_v3/summary.json"),
                 "Root selection freeze uses only closed development evidence")
    points = {(p["dataset"], p["mode"]): p for p in primary["points"]}
    cells, rows, contexts, comparisons = [], {}, {}, []
    for ds in base.DOMAINS:
        for mode in ("FP16", "R7low", "R7high"):
            _, rows[ds, mode], contexts[ds, mode] = base.load_cell(points[ds, mode], evidence, root)
        point = next(p for p in snapshot["points"] if p["dataset"] == ds and p["mode"] == "KRA1")
        cell, records = load_kra(point, evidence, root, rows[ds, "R7low"], contexts[ds, "R7low"])
        cells.append(cell)
        rows[ds, "KRA1"], contexts[ds, "KRA1"] = records, contexts[ds, "R7low"]
        for anchor in ("FP16", "R7low", "R7high"):
            pair, _ = base.paired(ds, "KRA1", anchor, rows, contexts)
            comparisons.append(pair)
            if anchor == "R7low":
                base.require(abs(pair["difference_pp"]) < 1e-12 and pair["CI95_low_pp"] == pair["CI95_high_pp"] == 0 and
                             pair["improved_questions"] == pair["degraded_questions"] == 0,
                             "KRA zero observed score differences, not a distributional equivalence proof")
    quality_rows, timing_rows = read_rows(analysis / "quality_complete_owner.csv"), read_rows(analysis / "instrumented_timing.csv")
    paired_rows = read_rows(analysis / "paired_bootstrap.csv")
    for cell in cells:
        b, bits = cell["resources"]["complete_owner_bytes"], cell["resources"]["actual_complete_bits"]
        quality_rows.append({"dataset": cell["dataset"], "mode": "KRA1", "QA": cell["QA"], "quality_percent": cell["quality_percent"],
            "frozen_individual_CI95_low_percent": cell["frozen_individual_CI95_percent"][0],
            "frozen_individual_CI95_high_percent": cell["frozen_individual_CI95_percent"][1],
            "complete_owner_bytes_mean": b["mean"], "complete_owner_bytes_min": b["min"], "complete_owner_bytes_max": b["max"],
            "complete_bits_mean": bits["mean"], "complete_bits_min": bits["min"], "complete_bits_max": bits["max"], "stage": "kra_new"})
        timing_rows.append({"dataset": cell["dataset"], "mode": "KRA1", "QA": cell["QA"], **{
            f"{key}_{stat}": cell["timing"][key][stat] for key in
            ("prefill_ms", "encode_ms", "question_prefill_ms", "ttft_ms", "decode_interval_sum_ms", "wall_ms", "output_tokens")
            for stat in ("mean", "median", "p90")},
            "pooled_natural_inter_token_ms_mean": cell["timing"]["pooled_natural_inter_token_ms"]["mean"],
            "pooled_natural_inter_token_ms_count": cell["timing"]["pooled_natural_inter_token_ms"]["count"]})
    evidence.unchanged()
    base.write(output / "analysis.json", {"schema": "trajkv.round10.development_kra_addendum.v1", "cells": cells,
        "paired_comparisons": comparisons, "new_free_answer_events": 160, "reused_reference_view_events": 480,
        "same_raw_prediction_count": sum(c["same_raw_prediction_count"] for c in cells),
        "same_output_token_ids_count": sum(c["same_output_token_ids_count"] for c in cells),
        "observed_D160_identity_is_not_distributional_or_model_equivalence": True,
        "reserved224_predictions_or_scores_opened": False, "root_selection_freeze": decision})
    base.write(output / "source_evidence.json", {"files": evidence.files, "unchanged_after_analysis": True,
        "source_sha256": base.digest(__file__), "reserved224_predictions_or_scores_opened": False})
    base.csv_write(analysis / "combined_quality_complete_owner.csv", quality_rows)
    base.csv_write(analysis / "combined_instrumented_timing.csv", timing_rows)
    base.csv_write(analysis / "combined_paired_bootstrap.csv", [*paired_rows, *comparisons])
    base.write(analysis / "combined_counts.json", {"primary_actual_free_answer_events": 1120,
        "middle_new_free_answer_events": 320, "KRA_new_free_answer_events": 160,
        "unique_actual_runs": 40, "unique_actual_free_answer_events": 1600, "distinct_question_image_sources": 160,
        "middle_reference_view_reuse": 480, "KRA_reference_view_reuse": 480,
        "total_reused_reference_view_events": 960, "declared_views": 64, "declared_view_answer_events": 2560,
        "reference_reuses_are_not_new_inference": True, "reserved224_predictions_or_scores_opened": False})
    findings = (
        "本分析只用D160 development：primary1120＋middle新增320＋KRA新增160＝1600次实际free回答，40个run；"
        "两个追加视图各复用480次primary参照。合计2560个view事件不等于2560次独立推理。\n\n"
        "V128-R43相对R7low（Text/Info/Doc/Chart）为+2.6563/−0.9549/+2.0833/0 pp；配对95%CI分别为"
        "[0,6.8750]/[−4.6875,1.8403]/[0,6.2500]/[0,0]。相对近字节AddOldOrder为"
        "+4.8438/−1.0938/+6.2500/+12.5000 pp。Text的未校正区间[0.6250,10.1563]仍是选择用development证据。\n\n"
        "LS-FZ四域净分均与R7low相同；Text实际2题改善、1题变差，gross gain/loss各1.5625 pp，净0，"
        "不能写成逐题恒等或等价。\n\n"
        "新增KRA1独立复核：160/160的raw_prediction和output_token_ids都与R7low相同，四域score变化均为0，"
        "每题完整owner额外5980B。零差异bootstrap CI=[0,0]仅反映这160个已观察样本，不证明模型或分布等价。"
        "这是有实际alpha坐标动作但没有观察到任务收益的有界负结果。\n\n"
        "Root已在selection_freeze_v1/DECISION.json固定仅V128-R43进入224复测，不扩展KRA；本分析没有读取任何"
        "reserved224预测/评分。D160用于候选选择，不作为独立确认；各域指标不混合平均。\n\n"
        "完整owner字节/bit逐题核查，V128与AddOldOrder差额小于一个18N stage而非精确同字节。"
        "combined表保留native prefill、encode、question prefill、TTFT、自然长度decode间隔之和、wall及输出长度；"
        "均为含同步诊断的仪器化host时间，不是受控加速/HBM结论。\n")
    with (analysis / "combined_findings.zh.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(findings)
    base.write(output / "complete.json", {"status": "PASS_KRA_D160_ADDENDUM_AND_COMBINED_DEVELOPMENT_SUMMARY",
        "newly_checked_KRA_QA": 160, "same_raw_prediction_and_output_token_ids": 160,
        "combined_unique_actual_free_events": 1600, "new_paired_contrasts": 12,
        "original_analysis_outputs_unmodified": True, "reserved224_predictions_or_scores_opened": False,
        "source_sha256": base.digest(__file__), "outputs": {str(path.relative_to(analysis)): base.digest(path) for path in
            (output / "analysis.json", output / "source_evidence.json", analysis / "combined_quality_complete_owner.csv",
             analysis / "combined_instrumented_timing.csv", analysis / "combined_paired_bootstrap.csv",
             analysis / "combined_counts.json", analysis / "combined_findings.zh.md")}})
    print(json.dumps({"status": "PASS_KRA_D160_ADDENDUM_AND_COMBINED_DEVELOPMENT_SUMMARY", "KRA_QA": 160,
        "combined_unique_actual_free_events": 1600, "reserved224_predictions_or_scores_opened": False,
        "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
