"""Read-only independent arithmetic review of primary+middle D160 only.

Reads only the two named closed snapshots and their explicitly enumerated
development runs. Never opens reserved224 predictions/scores or KRA results.
Recomputes frozen primary metrics, physical ownership, paired source-bootstrap
intervals and natural-length instrumented host timings; starts no GPU work.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "ops")]
import numpy as np
import metrics
from verify_public_round10_v1 import physical_complete_bytes
from round10_harvest_score_v1 import context as native_context, identity as input_identity

DOMAINS = ("textvqa", "infovqa", "docvqa", "chartqa")
COUNTS = {"textvqa": 64, "infovqa": 64, "docvqa": 16, "chartqa": 16}
PRIMARY = ("FP16", "R7low", "R7high", "LS-FZ", "LS-RE", "FB0", "FB-W")
MIDDLE = ("V128-R43", "AddOldOrder")
MODES = (*PRIMARY, *MIDDLE)
SEED = 20261119
DRAWS = 2000


def require(value, message):
    if not value:
        raise ValueError("ROUND10_DEVELOPMENT_ANALYSIS: " + message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def csv_write(path, rows):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    values = list(values)
    return {"count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "min": float(min(values)), "max": float(max(values))} if values else None


class Evidence:
    def __init__(self):
        self.files = {}

    def track(self, path, expected=None):
        path = Path(path).resolve()
        if str(path) not in self.files:
            stat = path.stat()
            self.files[str(path)] = {"sha256": digest(path), "bytes": stat.st_size,
                                     "stat": [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]}
        actual = self.files[str(path)]["sha256"]
        require(expected is None or actual == expected, "existing evidence SHA differs: " + str(path))
        return actual

    def load(self, path, expected=None):
        self.track(path, expected)
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))

    def unchanged(self):
        for name, item in self.files.items():
            path = Path(name)
            st = path.stat()
            require([st.st_size, st.st_mtime_ns, st.st_ctime_ns] == item["stat"] and digest(path) == item["sha256"],
                    "input evidence changed during analysis: " + name)


def load_cell(point, evidence, base):
    ds, mode = point["dataset"], point["mode"]
    n = COUNTS[ds]
    raw, scored = Path(point["raw"]).resolve(), Path(point["scored"]).resolve()
    require(raw.parent == (base / "harvest_v1/raw").resolve() and scored.parent == (base / "harvest_v1/scored").resolve() and
            raw.name == scored.name and "reserve" not in raw.name.lower() and "extension" not in raw.name.lower(),
            "only explicitly enumerated D160 development run directories")
    summary = evidence.load(scored / "summary.json", point["summary_sha256"])
    require(summary["status"] == "PASS_SCORED_COMPLETE" and summary["expected_run_questions"] == n,
            "whole-domain frozen score completion required")
    for name, expected in summary["integrity"]["raw_files"].items():
        evidence.track(raw / name, expected)
    for name, expected in summary["derived_file_hashes"].items():
        evidence.track(scored / name, expected)
    manifest = evidence.load(raw / "manifest.json")
    complete = evidence.load(raw / "complete.json")
    require(manifest["config"]["input_role"] == "development" and manifest["config"]["cohort_kind"] == "development" and
            complete["status"] == "PASS_RAW_COMPLETE" and complete["counts"] == {"PASS": n} and
            complete["recorded_QA"] == complete["expected_QA"] == n, "full D160 development scope only")
    proof = evidence.load(raw / "source_retrieval_receipt_round10_v1.json", point["source_receipt_sha256"])
    require(proof["status"] == "PASS_ROUND10_SEQUENCE_CHILD_EXECUTED_SOURCE_SHA" and
            proof["remote_hashes_computed_independently"] is True and proof["parent_cell_success"] is True,
            "frozen actual-child source completion proof")
    for entry in proof["files"]:
        evidence.track(raw / entry["file"], entry["sha256"])
    for role, expected in summary["integrity"]["metric_code_hashes"].items():
        evidence.track(ROOT / role, expected)
    evidence.track(ROOT / "code/score.py", summary["integrity"]["scorer_sha256"])
    refs_path = Path(summary["integrity"]["refs_path"]).resolve()
    require(refs_path.name == f"{ds}_dev{n}_refs.jsonl" and refs_path.parent == (base / "input_preparation_v1/inputs").resolve(),
            "only exact D160 reference files may be read")
    evidence.track(refs_path, summary["integrity"]["refs_sha256"])
    references = {r["id"]: r for r in map(json.loads, refs_path.read_text(encoding="utf-8").splitlines())}
    scored_rows = {r["id"]: r for r in map(json.loads, (scored / "scored_records.jsonl").read_text(encoding="utf-8").splitlines())}
    require(len(references) == len(scored_rows) == n, "complete one-domain references/scored IDs")
    rows = []
    with (raw / "records.jsonl").open("rb") as stream:
        for number, line in enumerate(stream, 1):
            r = json.loads(line)
            s = scored_rows[r["id"]]
            require(r["status"] == "PASS" and s["score_status"] == "SCORED_PASS" and s["raw_jsonl_line_number"] == number and
                    s["raw_jsonl_exact_line_sha256"] == hashlib.sha256(line).hexdigest(), "exact raw/scored line binding")
            fresh = metrics.score_question(ds, r["raw_prediction"], references[r["id"]])
            require(abs(fresh["score"] - s["score"]) < 1e-12 and fresh["metric_id"] == s["metric_id"] and
                    fresh["reference_texts"] == s["reference_texts"], "independent frozen metric recomputation differs")
            amount = physical_complete_bytes(r, manifest["model_spec"])
            nv = r["actual_visual_token_count"]
            require(abs(r["actual_effective_visual_bits"] - 16 * amount / r["full_kv_byte_denominator"]) < 1e-12 and
                    r["owner_after_generation"]["stored_visual_KV_bytes"] == amount and
                    r["original_full_prefix_released"] is True and r["original_FP16_visual_alias_retained"] is False,
                    "physical byte/bit/native release evidence differs")
            if mode == "FP16":
                require(amount == 57352 * nv, "complete FP16 owner including original position map")
            elif mode in ("R7low", "LS-FZ", "LS-RE", "FB0", "FB-W"):
                require(amount == 4590 * nv + 89733 + (896 if mode in ("LS-FZ", "LS-RE") else 0), "complete low/LS persistent formula")
            elif mode in MIDDLE:
                q = (448 * nv - 57344) // (18 * nv)
                expected = 5038 * nv + 32389 if mode == "V128-R43" else 4590 * nv + 89733 + 18 * nv * q
                require(amount == expected, "actual middle complete-owner formula")
            else:
                require(mode == "R7high" and r["r7_bridge"]["returned_owners_unique_tensor_capacity_bytes"] == amount,
                        "original high complete owner agrees with actual bridge table")
            rows.append({"id": r["id"], "dataset": ds, "mode": mode, "source_cluster_id": r["source_cluster_id"],
                "source_id": r["source_id"], "image_sha256": r["original_image_sha256"], "score": float(s["score"]),
                "raw_prediction": r["raw_prediction"], "normalized_prediction": s["normalized_prediction"],
                "input_identity_sha256": canonical(input_identity(r)), "N": nv, "complete_owner_bytes": amount,
                "actual_complete_bits": r["actual_effective_visual_bits"], "output_tokens": r["output_tokens"],
                **{key: r[key] for key in ("prefill_ms", "encode_ms", "question_prefill_ms", "ttft_ms", "wall_ms")},
                "decode_interval_sum_ms": sum(r["inter_token_ms"]), "decode_intervals_ms": r["inter_token_ms"]})
    require(len(rows) == n and [r["id"] for r in rows] == manifest["expected_ids"] and
            len({r["source_cluster_id"] for r in rows}) == len({r["image_sha256"] for r in rows}) == n,
            "one question per unique source/image and exact expected source ordering")
    mean = float(np.mean([r["score"] for r in rows]))
    require(abs(mean - point["primary_mean"]) < 1e-12 and abs(mean - summary["primary_mean"]) < 1e-12, "snapshot/frozen score mean differs")
    resource = {name: stats(r[name] for r in rows) for name in ("complete_owner_bytes", "actual_complete_bits", "N")}
    timing = {name: stats(r[name] for r in rows) for name in
              ("prefill_ms", "encode_ms", "question_prefill_ms", "ttft_ms", "decode_interval_sum_ms", "wall_ms", "output_tokens")}
    timing["pooled_natural_inter_token_ms"] = stats(v for r in rows for v in r["decode_intervals_ms"])
    for existing, key in (("complete_bytes", "complete_owner_bytes"), ("actual_complete_bits", "actual_complete_bits")):
        require(abs(resource[key]["mean"] - point["resources"][existing]["mean"]) < 1e-9, "snapshot resource arithmetic differs")
    for key in ("prefill_ms", "encode_ms", "question_prefill_ms", "ttft_ms", "wall_ms", "output_tokens"):
        require(abs(timing[key]["mean"] - point["timing"][key]["mean"]) < 1e-8, "snapshot instrumented timing arithmetic differs")
    return {"dataset": ds, "mode": mode, "QA": n, "quality": mean, "quality_percent": 100 * mean,
            "frozen_individual_CI95_percent": [100 * value for value in summary["primary_ci95"]],
            "resources": resource, "timing": timing, "raw": str(raw), "scored": str(scored),
            "raw_records_sha256": evidence.track(raw / "records.jsonl"), "scored_summary_sha256": evidence.track(scored / "summary.json")}, rows, canonical(native_context(manifest))


def paired(ds, candidate, anchor, rows, contexts):
    left, right = rows[ds, anchor], rows[ds, candidate]
    require(contexts[ds, anchor] == contexts[ds, candidate] and len(left) == len(right), "same actual native/runtime context")
    right = {r["id"]: r for r in right}
    differences, flips, byte_gaps = [], [], []
    for l in left:
        r = right[l["id"]]
        require(l["input_identity_sha256"] == r["input_identity_sha256"] and
                l["source_cluster_id"] == r["source_cluster_id"], "same original question/image/positions for every pair")
        delta = r["score"] - l["score"]
        differences.append(delta)
        byte_gaps.append(r["complete_owner_bytes"] - l["complete_owner_bytes"])
        if delta != 0 or l["raw_prediction"] != r["raw_prediction"]:
            flips.append({"dataset": ds, "candidate": candidate, "anchor": anchor, "id": l["id"],
                "source_cluster_id": l["source_cluster_id"], "anchor_score": l["score"], "candidate_score": r["score"],
                "difference_pp": 100 * delta, "anchor_prediction": l["raw_prediction"], "candidate_prediction": r["raw_prediction"]})
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    samples = np.concatenate([values[rng.integers(0, len(values), size=(min(100, DRAWS - start), len(values)))].mean(1)
                              for start in range(0, DRAWS, 100)])
    ci = np.percentile(samples, [2.5, 97.5]) * 100
    result = {"dataset": ds, "candidate": candidate, "anchor": anchor, "QA": len(values),
        "difference_pp": float(values.mean() * 100), "CI95_low_pp": float(ci[0]), "CI95_high_pp": float(ci[1]),
        "improved_questions": int((values > 0).sum()), "degraded_questions": int((values < 0).sum()),
        "equal_score_questions": int((values == 0).sum()), "gross_gain_pp": float(np.maximum(values, 0).mean() * 100),
        "gross_loss_pp": float(np.maximum(-values, 0).mean() * 100),
        "raw_prediction_changed_questions": sum(l["raw_prediction"] != right[l["id"]]["raw_prediction"] for l in left),
        "normalized_prediction_changed_questions": sum(l["normalized_prediction"] != right[l["id"]]["normalized_prediction"] for l in left),
        "mean_owner_byte_difference": float(np.mean(byte_gaps)), "min_owner_byte_difference": min(byte_gaps),
        "max_owner_byte_difference": max(byte_gaps), "bootstrap_seed": SEED, "bootstrap_draws": DRAWS,
        "source_cluster_count": len(values), "one_question_one_image": True,
        "descriptive_development_unadjusted_not_confirmation_or_equivalence": True}
    return result, flips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("${EVIDENCE_ROOT}/results/round10"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.root.resolve()
    require(not args.output.exists() and args.output.resolve().is_relative_to(base), "new analysis directory on explicit data disk")
    args.output.mkdir(parents=True)
    evidence = Evidence()
    p = evidence.load(base / "primary_D160_closed_v1/summary.json")
    m = evidence.load(base / "middle_watch_snapshot_v2/summary.json")
    existing_pairs = {**evidence.load(base / "primary_D160_closed_v1/paired_comparisons.json"),
                      **evidence.load(base / "middle_watch_snapshot_v2/paired_comparisons.json")}
    require(p["status"] == "PASS_ALL28_D160_CUSTOM_SOURCE_SCORE_PHYSICAL" and p["closed_cells"] == 28 and
            m["status"] == "PASS_ALL20_D160_MIDDLE_CUSTOM_SOURCE_SCORE_PHYSICAL" and m["closed_cells"] == 20,
            "only final all-domain primary/middle snapshots")
    points = {(point["dataset"], point["mode"]): point for point in p["points"]}
    for point in m["points"]:
        key = point["dataset"], point["mode"]
        if key in points:
            require(all(point[field] == points[key][field] for field in ("raw", "scored", "summary_sha256", "source_receipt_sha256")),
                    "middle reference view must reuse exact same primary inference/scoring artifacts")
        else:
            points[key] = point
    require(set(points) == {(ds, mode) for ds in DOMAINS for mode in MODES}, "exact 36 unique development runs")
    cells, rows, contexts = [], {}, {}
    for ds in DOMAINS:
        for mode in MODES:
            cell, records, context = load_cell(points[ds, mode], evidence, base)
            cells.append(cell)
            rows[ds, mode], contexts[ds, mode] = records, context
    comparisons, flips = [], []
    contrasts = [(mode, "FP16") for mode in MODES if mode != "FP16"] + [
        (mode, anchor) for mode in (*PRIMARY[3:], *MIDDLE) for anchor in ("R7low", "R7high")] + [
        ("R7high", "R7low"), ("LS-RE", "LS-FZ"), ("FB-W", "FB0"), ("V128-R43", "AddOldOrder")]
    checked_existing = 0
    for ds in DOMAINS:
        for candidate, anchor in contrasts:
            pair, changes = paired(ds, candidate, anchor, rows, contexts)
            key, inverse = f"{ds}__{candidate}_minus_{anchor}", f"{ds}__{anchor}_minus_{candidate}"
            if key in existing_pairs or inverse in existing_pairs:
                old = existing_pairs.get(key, existing_pairs.get(inverse))["primary"]
                direction = 1 if key in existing_pairs else -1
                expected_ci = old["ci95_pp"] if direction == 1 else [-old["ci95_pp"][1], -old["ci95_pp"][0]]
                require(abs(pair["difference_pp"] - direction * old["difference_pp"]) < 1e-10 and
                        max(abs(pair[field] - value) for field, value in zip(("CI95_low_pp", "CI95_high_pp"), expected_ci)) < 1e-10,
                        "independent paired bootstrap differs from frozen snapshot")
                checked_existing += 1
            comparisons.append(pair)
            if (candidate, anchor) in (("LS-FZ", "R7low"), ("V128-R43", "R7low"), ("V128-R43", "AddOldOrder")):
                flips.extend(changes)
    quality_rows, timing_rows = [], []
    for cell in cells:
        b, bits = cell["resources"]["complete_owner_bytes"], cell["resources"]["actual_complete_bits"]
        quality_rows.append({"dataset": cell["dataset"], "mode": cell["mode"], "QA": cell["QA"],
            "quality_percent": cell["quality_percent"], "frozen_individual_CI95_low_percent": cell["frozen_individual_CI95_percent"][0],
            "frozen_individual_CI95_high_percent": cell["frozen_individual_CI95_percent"][1],
            "complete_owner_bytes_mean": b["mean"], "complete_owner_bytes_min": b["min"], "complete_owner_bytes_max": b["max"],
            "complete_bits_mean": bits["mean"], "complete_bits_min": bits["min"], "complete_bits_max": bits["max"],
            "stage": "primary" if cell["mode"] in PRIMARY else "middle_new"})
        timing_rows.append({"dataset": cell["dataset"], "mode": cell["mode"], "QA": cell["QA"], **{
            f"{key}_{stat}": cell["timing"][key][stat] for key in
            ("prefill_ms", "encode_ms", "question_prefill_ms", "ttft_ms", "decode_interval_sum_ms", "wall_ms", "output_tokens")
            for stat in ("mean", "median", "p90")},
            "pooled_natural_inter_token_ms_mean": cell["timing"]["pooled_natural_inter_token_ms"]["mean"],
            "pooled_natural_inter_token_ms_count": cell["timing"]["pooled_natural_inter_token_ms"]["count"]})
    byte_pair_checks = []
    for ds in DOMAINS:
        additive = {r["id"]: r for r in rows[ds, "AddOldOrder"]}
        for r in rows[ds, "V128-R43"]:
            difference = r["complete_owner_bytes"] - additive[r["id"]]["complete_owner_bytes"]
            require(0 <= difference < 18 * r["N"], "near-byte middle pair differs by less than one actual stage")
            byte_pair_checks.append({"dataset": ds, "id": r["id"], "N": r["N"], "V128_bytes": r["complete_owner_bytes"],
                "AddOldOrder_bytes": additive[r["id"]]["complete_owner_bytes"], "gap_bytes": difference,
                "stage_bytes": 18 * r["N"]})
    count_report = {"primary_unique_runs": 28, "primary_actual_free_answer_events": 1120,
        "middle_declared_view_cells": 20, "middle_view_answer_events": 800, "middle_new_unique_runs": 8,
        "middle_actual_new_free_answer_events": 320, "middle_reused_reference_view_events": 480,
        "combined_declared_view_cells": 48, "combined_view_answer_events": 1920,
        "combined_unique_actual_runs": 36, "combined_unique_actual_free_answer_events": sum(c["QA"] for c in cells),
        "distinct_original_question_image_sources": sum(COUNTS.values()),
        "references_are_same_primary_artifacts_not_independent_replications": True,
        "per_domain": {ds: {"distinct_sources": n, "primary_free_events": 7 * n,
            "middle_new_free_events": 2 * n, "middle_reused_reference_view_events": 3 * n} for ds, n in COUNTS.items()}}
    require(count_report["combined_unique_actual_free_answer_events"] == 1440, "unique actual free-answer count")
    evidence.unchanged()
    write(args.output / "analysis.json", {"schema": "trajkv.round10.development_independent_analysis.v1",
        "status": "PASS_PRIMARY_AND_MIDDLE_D160_ARITHMETIC_PHYSICAL_SOURCE_REVIEW", "counts": count_report,
        "cells": cells, "paired_comparisons": comparisons, "near_byte_middle_pairs": byte_pair_checks,
        "bootstrap": {"seed": SEED, "draws": DRAWS, "method": "paired source-cluster percentile bootstrap; one question/image per source"},
        "existing_paired_CIs_independently_matched": checked_existing,
        "D160_is_selection_development_not_independent_confirmation": True,
        "unadjusted_CIs_not_equivalence_or_algorithm_winner_certificate": True,
        "cross_dataset_quality_average_computed": False,
        "timing_scope": "instrumented natural-output host timing; encode contains synchronizing diagnostics; not controlled speed/HBM traffic",
        "decode_interval_sum_definition": "sum recorded inter_token_ms after first token; do not replace with wall-minus-TTFT",
        "reserved224_predictions_or_scores_opened": False, "KRA_predictions_or_scores_opened": False,
        "new_GPU_jobs": 0, "new_free_answers": 0, "source_run_admission_claimed": False})
    csv_write(args.output / "quality_complete_owner.csv", quality_rows)
    csv_write(args.output / "paired_bootstrap.csv", comparisons)
    csv_write(args.output / "instrumented_timing.csv", timing_rows)
    csv_write(args.output / "key_question_flips.csv", flips)
    write(args.output / "counts.json", count_report)
    write(args.output / "source_evidence.json", {"schema": "trajkv.round10.development_analysis_sources.v1", "files": evidence.files,
        "input_sources_unchanged_after_analysis": True, "analysis_script_sha256": digest(__file__),
        "python": platform.python_version(), "numpy": np.__version__, "no_reserved224_or_KRA_predictions_or_scores": True})
    key_pairs = {(r["dataset"], r["candidate"], r["anchor"]): r for r in comparisons}
    quality_map = {(c["dataset"], c["mode"]): c["quality_percent"] for c in cells}
    lines = ["本次仅复核D160 development：36个实际run、1440次原始free回答；middle视图另复用primary的480次参照，合并视图为1920次，不能当作1920次独立推理。未读取reserved224或KRA预测/评分。", "",
        "V128-R43相对R7low的各域变化（pp，配对source bootstrap 95%区间）："]
    for ds in DOMAINS:
        r = key_pairs[ds, "V128-R43", "R7low"]
        lines.append(f"- {ds}: {quality_map[ds, 'R7low']:.4f}% → {quality_map[ds, 'V128-R43']:.4f}%，Δ={r['difference_pp']:+.4f} pp，CI [{r['CI95_low_pp']:.4f}, {r['CI95_high_pp']:.4f}]；score改善/变差 {r['improved_questions']}/{r['degraded_questions']}题。")
    lines += ["", "与近同字节AddOldOrder比较，V128-R43的变化："]
    for ds in DOMAINS:
        r = key_pairs[ds, "V128-R43", "AddOldOrder"]
        lines.append(f"- {ds}: Δ={r['difference_pp']:+.4f} pp，CI [{r['CI95_low_pp']:.4f}, {r['CI95_high_pp']:.4f}]。")
    text = key_pairs["textvqa", "LS-FZ", "R7low"]
    lines += ["", f"LS-FZ四域净分均与R7low相同。TextVQA不是逐题恒等：{text['improved_questions']}题score改善、{text['degraded_questions']}题变差，gross gain/loss各{text['gross_gain_pp']:.4f} pp，净0；配对CI [{text['CI95_low_pp']:.4f}, {text['CI95_high_pp']:.4f}]。其余三域逐题score相同，预测字符串差异另见flips表。",
        "", "V128在Text/Doc/Chart没有观察到相对R7low的score下降题，但Info略降；相对近字节AddOldOrder的Text区间不跨0也只是同一development面板上的未校正描述，不能升级为独立确认、等价或算法胜者结论。不同数据集指标不作混合平均。",
        "", "每题实测V128与AddOldOrder的容量差严格小于一个18N残差stage，非精确同字节。完整owner含原position map、codebook、factors、lookup/ranking/policy；LS额外896B保留计费。详见quality_complete_owner.csv和analysis.json逐题容量。",
        "", "计时为真实quality-run的仪器化host统计，保留native prefill、encode、question prefill、TTFT、自然长度decode间隔之和及wall。encode含同步诊断，输出长度与初始化状态不同；这些数据不支持受控加速或HBM流量结论。",
        "", f"配对bootstrap固定seed={SEED}、{DRAWS}次source抽样；{checked_existing}个既有配对区间独立重算吻合。1440条原始预测均已用冻结metric与dev参考答案重算，raw行SHA/评分派生文件/物理storage表及来源SHA均核对，输入证据执行前后不变。"]
    with (args.output / "findings.zh.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    write(args.output / "complete.json", {"status": "PASS_D160_DEVELOPMENT_ANALYSIS", "unique_actual_free_events": 1440,
        "cells": 36, "paired_contrasts": len(comparisons), "existing_paired_CIs_matched": checked_existing,
        "reserved224_predictions_or_scores_opened": False, "GPU_jobs_started": 0,
        "source_sha256": digest(__file__), "outputs": {name: digest(args.output / name) for name in
            ("analysis.json", "counts.json", "quality_complete_owner.csv", "paired_bootstrap.csv", "instrumented_timing.csv",
             "key_question_flips.csv", "source_evidence.json", "findings.zh.md")}})
    print(json.dumps({"status": "PASS_D160_DEVELOPMENT_ANALYSIS", "cells": 36, "actual_free_events": 1440,
                      "paired_contrasts": len(comparisons), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
