"""Read-only detail addendum to the already closed final analysis.

Preserves every original script/report/receipt byte. Emits per-domain byte
ratios/p50 timings and traceable task flips from final scored rows only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
import round10_development_analysis_v1 as common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory, output = args.analysis.resolve(), args.output.resolve()
    common.require(output.is_relative_to(directory) and not output.exists(), "new additive detail directory")
    evidence = common.Evidence()
    complete = evidence.load(directory / "complete.json")
    common.require(complete["status"] == "PASS_FINAL_EXECUTED_SCOPE_ANALYSIS" and
                   complete["actual_new_free_QA"] == 2496 and complete["actual_runs"] == 56, "final exact scope is closed")
    for name, expected in complete["outputs"].items():
        evidence.track(directory / name, expected)
    original_source = evidence.load(directory / "source_evidence.json")["files"]
    analysis = evidence.load(directory / "analysis.json")
    cells = {(c["dataset"], c["mode"]): c for c in analysis["replication"]["cells"]}
    score_rows = {}
    for key, cell in cells.items():
        path = (Path(cell["scored"]) / "scored_records.jsonl").resolve()
        evidence.track(path, original_source[str(path)]["sha256"])
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        common.require(len(records) == cell["QA"] and len({r["id"] for r in records}) == cell["QA"], "complete final scored cohort")
        score_rows[key] = {r["id"]: r for r in records}
    ratios, display, timings, flips = [], [], [], []
    for ds in common.DOMAINS:
        candidate = cells[ds, "V128-R43"]
        for mode in ("FP16", "R7low", "R7high", "V128-R43"):
            cell = cells[ds, mode]
            display.append({"dataset": ds, "mode": mode, "QA": cell["QA"], "quality_percent": cell["quality_percent"],
                "complete_owner_bytes_mean": cell["resources"]["complete_owner_bytes"]["mean"],
                "actual_complete_bits_mean": cell["resources"]["actual_complete_bits"]["mean"]})
            timings.append({"dataset": ds, "mode": mode, "QA": cell["QA"], **{
                key + "_p50": cell["timing"][key]["median"] for key in
                ("prefill_ms", "encode_ms", "question_prefill_ms", "ttft_ms", "decode_interval_sum_ms", "wall_ms", "output_tokens")}})
        for anchor in ("FP16", "R7low", "R7high"):
            amount = candidate["resources"]["complete_owner_bytes"]["mean"]
            anchor_amount = cells[ds, anchor]["resources"]["complete_owner_bytes"]["mean"]
            ratios.append({"dataset": ds, "candidate": "V128-R43", "anchor": anchor,
                "paired_total_owner_byte_ratio": amount / anchor_amount,
                "owner_byte_delta_percent": 100 * (amount / anchor_amount - 1),
                "complete_owner_byte_saving_percent": 100 * (1 - amount / anchor_amount),
                "ratio_scope": "same whole-domain paired question count; ratio of actual summed complete-owner bytes"})
            for identity, left in score_rows[ds, anchor].items():
                right = score_rows[ds, "V128-R43"][identity]
                common.require(all(left[key] == right[key] for key in
                    ("model_commit", "dataset_revision", "split", "question_id", "original_image_sha256", "full_prompt_hash",
                     "question_suffix_sha256", "actual_visual_token_count", "source_cluster_id")), "exact paired task identity")
                delta = right["score"] - left["score"]
                prediction_changed = left["raw_prediction"] != right["raw_prediction"]
                tokens_changed = left["output_token_ids"] != right["output_token_ids"]
                if delta == 0 and not prediction_changed and not tokens_changed:
                    continue
                flips.append({"dataset": ds, "candidate": "V128-R43", "anchor": anchor, "QA": candidate["QA"], "id": identity,
                    "source_cluster_id": left["source_cluster_id"], "original_image_sha256": left["original_image_sha256"],
                    "anchor_score": left["score"], "candidate_score": right["score"], "per_question_score_difference": delta,
                    "contribution_to_domain_difference_pp": 100 * delta / candidate["QA"],
                    "score_change": "improved" if delta > 0 else "degraded" if delta < 0 else "equal",
                    "raw_prediction_changed": prediction_changed, "output_token_ids_changed": tokens_changed,
                    "anchor_prediction": left["raw_prediction"], "candidate_prediction": right["raw_prediction"],
                    "continuous_frozen_metric_change_not_manual_semantic_error_classification": True})
    evidence.unchanged()
    output.mkdir()
    common.csv_write(output / "four_methods_quality_bits.csv", display)
    common.csv_write(output / "actual_complete_owner_byte_ratios.csv", ratios)
    common.csv_write(output / "four_methods_p50_timing.csv", timings)
    common.csv_write(output / "replication_taskflips_vs_refs.csv", flips)
    common.csv_write(output / "replication_taskflips_vs_R7low.csv", [r for r in flips if r["anchor"] == "R7low"])
    common.csv_write(output / "replication_score_changes_only.csv", [r for r in flips if r["score_change"] != "equal"])
    low_pairs = [p for p in analysis["replication"]["paired_comparisons"] if p["anchor"] == "R7low"]
    low_changes = [r for r in flips if r["anchor"] == "R7low" and r["score_change"] != "equal"]
    brief = {"schema": "trajkv.round10.final_detail_addendum.v1", "status": "PASS_FINAL_DETAIL_SOURCE_BOUND",
        "counts": analysis["counts"], "replication_four_method_quality_and_bits": display, "byte_ratios": ratios,
        "p50_timings": timings, "V128_minus_R7low": low_pairs, "score_changes_vs_R7low": low_changes,
        "V128_D224_score_improvements_vs_low": sum(p["improved_questions"] for p in low_pairs),
        "V128_D224_score_degradations_vs_low": sum(p["degraded_questions"] for p in low_pairs),
        "quality_across_domains_or_cohorts_not_averaged": True,
        "paired_helper_legacy_flag_note": "replication.paired_comparisons inherits a development-named descriptive/unadjusted flag; its actual cohort is the separately stored reserved_unused_extension replication, not D160",
        "instrumented_host_timings_are_not_controlled_speed_or_HBM_traffic": True,
        "statistical_scope": "fixed candidate replication; unadjusted source bootstrap; observed zero differences are not model/distribution equivalence",
        "new_GPU_jobs": 0, "new_parameters_selected": False}
    common.write(output / "ROOT_BRIEF.json", brief)
    common.write(output / "source_evidence.json", {"files": evidence.files, "unchanged_after_detail_read": True,
        "original_final_analysis_outputs_unmodified": True, "original_final_script_sha256": complete["source_sha256"],
        "detail_script_sha256": common.digest(__file__)})
    common.write(output / "complete.json", {"status": "PASS_FINAL_DETAIL_SOURCE_BOUND", "source_sha256": common.digest(__file__),
        "original_final_script_sha256": complete["source_sha256"], "original_outputs_unmodified": True,
        "outputs": {p.name: common.digest(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({"status": "PASS_FINAL_DETAIL_SOURCE_BOUND", "score_improved_vs_low": len([r for r in low_changes if r["score_change"] == "improved"]),
                      "score_degraded_vs_low": len([r for r in low_changes if r["score_change"] == "degraded"]), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
