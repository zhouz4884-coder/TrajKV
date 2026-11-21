"""R7 closed8 fixed500 competition table; no network, GPU or score selection.

All eight custom-driver raw/source/score closures must pass before a table is
written. Canonical baseline evidence is refreshed from its declared actual
paths; incomplete methods stay null. Official-full FP16 never enters this
same500 matrix. Original canonical large-raw proof is reused explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
E = Path("${EVIDENCE_ROOT}")
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "ops"), str(ROOT / "code")]
SNAPSHOT = E / "results/reports/M1_all_methods_fixed500_snapshot_20261121_v1"
FINAL = E / "results/r7_matrix500_v2/competitive_final_v1"
R7_ANALYSIS_SHA = "3d282b201d474a61473c16c397fb90a3bde287b8fec6f35c6ecb21c42e5fcda9"
EXTRACT_SHA = "dd01e20220043e2b548ea242ae78ee5b3597a447c416c4329fd156e2f90d967f"
DOMAINS = ("textvqa", "infovqa", "docvqa", "chartqa")


def require(ok, message):
    if not ok:
        raise ValueError("R7_COMPETITION: " + message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def module(name, path, expected):
    require(sha(path) == expected, "frozen helper changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def csv_file(path, rows):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def snapshot_inputs(directory):
    receipt, index = load(directory / "receipt.json"), load(directory / "source_index.json")
    require(receipt["status"] == "PASS_M1_ALL_CURRENT_METHODS_FIXED500_SNAPSHOT" and not receipt["errors"], "baseline snapshot has identity errors")
    require(sha(directory / "source_index.json") == receipt["source_index_sha256"], "snapshot/source-index SHA mismatch")
    require(index["schema"] == "baseline_matrix.M1_all_current_methods_fixed500_snapshot.v1" and
            index["M1_declared_baseline_cells"] == 52 and index["R7_planned_cells"] == 8 and index["method_rows"] == 17,
            "original all-methods snapshot scope differs")
    for name, key in [("table.csv", "table_csv_sha256"), ("table.md", "table_md_sha256")]:
        require(sha(directory / name) == receipt[key], "snapshot table changed")
    return index, receipt


def source_label(tag, method):
    """Names describe the actual code path, never a published-paper score."""
    family = method["family"]
    args = method.get("constructor_args", {})
    if tag == "F":
        return "FP16：同一fixed500", "native_FP16_same500"
    if family == "kivi_source_example":
        require(args["group_size"] == 32 and args["residual_length"] == 32, "KIVI source-default identity changed")
        return "KIVI source example：2bit / g32 / R32", "pinned_source_example_g32R32"
    if family == "kivi_style":
        return f'KIVI-style：{args.get("bits")}bit / g{args.get("group_size")} / R{args.get("residual_length")}', "technical_style_anchor_not_source_default"
    if family == "calibquant":
        require(method["adapter_class"] == "CalibQuantFP32AffineTextAdapter", "retired Calib cannot enter numeric-v2 table")
        return f'CalibQuant numeric-v2：{args["bits"]}bit', "common_FP16_FP32_affine_exactText32_port_not_author_model_calibration"
    if family == "onlyterp_mse":
        require(method["adapter_class"] == "OnlyTerpMSEAdapter", "OnlyTerp source path differs")
        return f'OnlyTerp-MSE：Hmix{args["bits_regular"]}/{args["bits_outlier"]}，QJL=0', "third_party_Algorithm1_MSE_QJL0_not_full_TurboQuant"
    if family == "turboquant":
        require(method["adapter_class"] == "TurboQuantBufferedAdapter" and args == {"key_bits": 3, "value_bits": 2, "group_size": 32, "recent_window": 128}, "full TQ source-default identity differs")
        return "TurboQuant 0xSero source：K3/V2 g32 R128", "third_party_full_MSE_QJL_source_default_not_Google_official_or_exact_paper_bitpoint"
    if family == "pyramidkv":
        return f'PyramidKV：W{args["window_size"]} / drop{args["compression_ratio"]}', "query_unknown_visual_prefix_source_port"
    raise ValueError("R7_COMPETITION: unknown actual family " + family)


def original_source_metadata(descriptor, common, analysis):
    """Recheck small proof/dependencies; preserve original large-raw SHA proof.

    This does not instantiate a new SourceRun or claim to have rescanned its
    records. SourceRun/line/score closure was already independently established
    by the immutable score's derived provenance and original source receipts.
    """
    from prediction_cache import _dependencies, normalize_behavior
    path = Path(descriptor["raw_dir"])
    for name, digest in descriptor["source_proof_hashes"].items():
        require(sha(path / name) == digest, "original source metadata changed")
    require(sha(path / "retrieval_receipt.json") == descriptor["retrieval_receipt_sha256"] and
            sha(path / "source_retrieval_receipt_v2.json") == descriptor["source_retrieval_receipt_sha256"], "original retrieval receipt changed")
    retrieval = load(path / "retrieval_receipt.json")
    require(retrieval["status"] == "PASS_REMOTE_RAW_SHA256_EXACT" and retrieval["remote_hashes_computed_independently"] is True, "canonical original raw receipt absent")
    entries = {x["file"]: x for x in retrieval["files"]}
    require(set(entries) == {"manifest.json", "records.jsonl", "complete.json"}, "original raw3 receipt incomplete")
    for name, digest in descriptor["raw_hashes"].items():
        require(entries[name]["sha256"] == digest and (path / name).stat().st_size == entries[name]["bytes"], "original raw receipt/size differs")
        if name != "records.jsonl":
            require(sha(path / name) == digest, "original small raw changed")
    m, complete, inventory, launch = [load(path / name) for name in ("manifest.json", "complete.json", "source_inventory.json", "launch_receipt.json")]
    require(m["schema"] == "baseline_matrix.raw_inference.v1" and m["references_read"] is False and complete["coverage_complete"] is True,
            "canonical source closure differs")
    require(complete["manifest_sha256"] == descriptor["raw_hashes"]["manifest.json"] and complete["records_jsonl_sha256"] == descriptor["raw_hashes"]["records.jsonl"], "canonical original hash chain differs")
    require(launch["source_snapshot"] == m["runtime_source_root"] and launch["source_inventory_sha256"] == sha(path / "source_inventory.json") and
            launch["runtime_python"] == m["runtime_executable"] == launch["argv_executed"][0], "canonical source launch/runtime differs")
    deps = _dependencies(m, inventory, path / "source_evaluate.py")
    contract = {"schema": "baseline_matrix.effective_prediction_contract.v1", "behavior": normalize_behavior(m["config"], m["model_spec"]),
                "model_spec": m["model_spec"], **analysis.native_context(m), "relevant_source_hashes": deps}
    require(canonical(contract) == descriptor["effective_contract_sha256"], "canonical original effective contract/source changed")
    require(analysis.native_context(m) == {k: common["context"][k] for k in analysis.native_context(m)} and m["model_spec"] == common["context"]["model_spec"], "source native model/processor/generation/runtime mismatch")
    for name in ("code/m1_protocol.py", "code/cache_bridge.py", "code/attention_bridge.py", "code/ownership.py"):
        require(deps[name] == analysis.PINS[name], "canonical common protocol source changed")


def refresh_baselines(index, extraction, common, analysis):
    import final_union_scope_v1 as scope
    live = [r for r in scope.quality_matrix("p2")["rows"] if r["model"] == "M1"]
    old = {(r["method_key"], r["dataset"]): r for r in index["cells"] if not r["method_key"].startswith("R7_") and r["status"] != "NOT_PLANNED_IN_DECLARED_MATRIX"}
    require(len(live) == len(old) == 52 and len({r["method"] for r in live}) == 15, "declared52 baseline scope changed")
    cells, fingerprints, seen_sources = {}, {}, set()
    for row in live:
        tag, ds = row["method"], row["dataset"]
        prior = old[tag, ds]
        require(row["config"] == prior["config"] and row["config_sha256"] == prior["config_sha256"] == sha(ROOT / row["config"]), "baseline target identity changed")
        cfg = load(ROOT / row["config"])
        label, identity = source_label(tag, cfg["method"])
        cell = extraction.extract_cell("M1", ds, tag, cfg, row["view"], row["scored"], 500,
                                       index["inputs"][ds]["sha256"], row["config_sha256"])
        cell.update(label=label, source_identity=identity, method_descriptor=cfg["method"], config=row["config"], config_sha256=row["config_sha256"])
        if prior["status"] == "COMPLETE_FIXED500":
            require(cell["status"] == "COMPLETE_FIXED500" and cell["score_summary_sha256"] == prior["score_summary_sha256"] and cell["raw_file_sha256"] == prior["raw_file_sha256"], "previously complete baseline drifted")
        if cell["status"] == "COMPLETE_FIXED500":
            summary = analysis.score_metadata(cell["scored"], cell["score_summary_sha256"])
            m = load(Path(cell["view"]) / "manifest.json")
            domain = common["common_contracts_by_dataset"][ds]
            require(analysis.native_context(m) == {k: domain["context"][k] for k in analysis.native_context(m)}, "baseline view native context differs")
            require(m["expected_ids"] == index["inputs"][ds]["expected_ids"] and summary["metric_ids"] == [index["quality_main_rule"][ds]], "baseline ordered500/metric differs")
            sources = m.get("source_runs", [])
            require(bool(sources), "canonical completed view lacks original SourceRun descriptors")
            for descriptor in sources:
                key = (descriptor["raw_dir"], descriptor["effective_contract_sha256"])
                if key not in seen_sources:
                    original_source_metadata(descriptor, domain, analysis)
                    seen_sources.add(key)
            fingerprint = cell.pop("per_ID_input_fingerprint")
            fingerprints[tag, ds] = fingerprint
            cell["input_image_prompt_N_fingerprint_sha256"] = canonical(fingerprint)
            cell["source_evidence_boundary"] = "original_SourceRun_and_score_proofs_reused; current_small_proofs_numeric_dependencies_and_scored_cache_rechecked; no_new_large_raw_scan"
        cells[tag, ds] = cell
    for ds in DOMAINS:
        require(cells["F", ds]["status"] == "COMPLETE_FIXED500", "same500 FP16 anchor incomplete")
        for key, value in fingerprints.items():
            if key[1] == ds:
                require(value == fingerprints["F", ds], "baseline per-ID image/prompt/N differs from fixed500 F")
    return cells


def make_table(args, analysis, rows, ready):
    index, snapshot_receipt = snapshot_inputs(args.snapshot)
    extraction = module("r7_allmethods_snapshot_extract", args.snapshot / "reproduce_snapshot_v2.py", EXTRACT_SHA)
    _, common = analysis.load_references(analysis.REFERENCES, analysis.TOKENIZER_PROOF)
    r7 = [analysis.r7_point(row, item["cell"], common) for row, item in zip(rows, ready["cells"])]
    cells = refresh_baselines(index, extraction, common, analysis)
    order = list(extraction.ORDER)
    for point in r7:
        tag, ds = "R7_" + point["profile"], point["dataset"]
        cells[tag, ds] = {"model": "M1", "dataset": ds, "method_key": tag, "status": "COMPLETE_FIXED500", "expected_QA": 500,
                         "primary_mean": point["quality"], "CI95": point["CI95"], "primary_metric_id": point["metric_id"],
                         "label": "R7 K1 gt_cond43" if tag == "R7_K1" else "R7 K2 G4_256 cond",
                         "source_identity": "custom_frozen_R7_driver_original_profile_Not_canonical_SourceRun",
                         "resources": {"mean_complete_bits": point["mean_complete_bits"], "min_complete_bits": point["min_complete_bits"],
                             "max_complete_bits": point["max_complete_bits"], "mean_complete_visual_bytes": point["complete_mean_bytes"],
                             "strict_all_QA_1.3": point["strict_all_request_budget"]["1.3"]["eligible"],
                             "strict_all_QA_1.8": point["strict_all_request_budget"]["1.8"]["eligible"]},
                         "scored": point["scored"], "view": point["raw"], "score_summary_sha256": point["score_summary_sha256"],
                         "raw_file_sha256": point["raw_file_sha256"], "source_receipt_sha256": point["source_receipt_sha256"],
                         "input_fingerprint": point["input_fingerprint"], "legacy_auxiliary": point["legacy_author_strip_aux"], "groups": point["groups"]}
    for tag in order:
        for ds in DOMAINS:
            cells.setdefault((tag, ds), {"model": "M1", "dataset": ds, "method_key": tag, "status": "NOT_PLANNED_IN_DECLARED_MATRIX",
                                          "expected_QA": 500, "primary_mean": None, "CI95": None, "resources": None})
    require(len(cells) == 68 and sum(cells[t, d]["status"] == "COMPLETE_FIXED500" for t in ("R7_K1", "R7_K2") for d in DOMAINS) == 8,
            "must retain17methods/all68 cells and real8R7")
    long = []
    for tag in order:
        for ds in DOMAINS:
            cell, fp16 = cells[tag, ds], cells["F", ds]["primary_mean"]
            value, resource = cell["primary_mean"], cell.get("resources") or {}
            long.append({"dataset": ds, "method_key": tag, "method": cell.get("label", extraction.LABELS[tag]), "source_identity": cell.get("source_identity"),
                         "status": cell["status"], "QA": 500, "metric_id": index["quality_main_rule"][ds],
                         "quality_percent": None if value is None else 100 * value, "CI95_percent": None if cell.get("CI95") is None else [100 * z for z in cell["CI95"]],
                         "candidate_minus_same500_FP16_pp": None if value is None else 100 * (value - fp16),
                         "mean_complete_MiB": None if not resource else resource["mean_complete_visual_bytes"] / 2**20,
                         "mean_complete_bits": resource.get("mean_complete_bits"), "max_request_complete_bits": resource.get("max_complete_bits"),
                         "strict_all500_1p3": resource.get("strict_all_QA_1.3"), "strict_all500_1p8": resource.get("strict_all_QA_1.8"),
                         "score_summary_sha256": cell.get("score_summary_sha256")})
    wide = []
    for tag in order:
        row = {"method_key": tag, "method": next(x["method"] for x in long if x["method_key"] == tag)}
        for ds in DOMAINS:
            point = next(x for x in long if x["method_key"] == tag and x["dataset"] == ds)
            for key in ("quality_percent", "candidate_minus_same500_FP16_pp", "mean_complete_bits", "max_request_complete_bits", "status"):
                row[ds + "_" + key] = point[key]
        wide.append(row)
    report = {"schema": "trajkv.r7.full8_competitive_fixed500.v1", "status": "PASS_CLOSED8_ALL_CURRENT_METHODS_MATCHED_FIXED500",
              "R7_cells": 8, "R7_QA": 4000, "declared_baseline_cells": 52, "method_rows": 17, "matrix_cells_including_not_planned": 68,
              "status_counts": dict(Counter(c["status"] for c in cells.values())), "cells": [cells[t, d] for t in order for d in DOMAINS],
              "source_snapshot_index_sha256": sha(args.snapshot / "source_index.json"), "source_snapshot_receipt_sha256": sha(args.snapshot / "receipt.json"),
              "source_snapshot_captured_utc": index["captured_utc"], "current_baseline_paths_refreshed_without_overwriting_snapshot": True,
              "R7_custom_driver_not_SourceRun": True, "official_full_FP16_excluded_from_competition": True, "incomplete_scores_are_null": True,
              "primary_metrics_not_cross_domain_averaged": True, "source_provenance_names_not_published_paper_SOTA_replication_claims": True,
              "no_winner_or_equivalence_from_point_estimates_or_unadjusted_CI": True, "whole_external126_24_or_Round9_goal_completed": False,
              "new_GPU_QA": 0, "new_rescoring": False, "helper_source_sha256": sha(__file__), "frozen_R7_analysis_sha256": R7_ANALYSIS_SHA,
              "legacy_auxiliary_not_primary": True, "canonical_baseline_source_proofs_reused_not_large_raw_rescanned": True}
    out = args.output
    require(not out.exists(), "use a new immutable complete-table output")
    out.mkdir(parents=True)
    write(out / "source_index.json", report)
    csv_file(out / "competitive_long.csv", long)
    csv_file(out / "competitive_wide.csv", wide)
    md = ["# R7：两冻结点四域fixed500竞争力", "", "8格/4000条R7回答已完整闭合并通过独立custom-driver来源、同输入/模型/原生协议及冻结评分核验。下表全部采用各域原固定500题；官方全量FP16不进入此表。", "",
          "Text=十票soft accuracy；Info/Doc=冻结vendor raw-length ANLS；Chart=relaxed accuracy。质量单位为%，Δ为相对同500 FP16的pp；不同域不汇总成一个分数。", "",
          "| 方法 | Text | Info | Doc | Chart |", "|---|---:|---:|---:|---:|"]
    for row in wide:
        values = []
        for ds in DOMAINS:
            value = row[ds + "_quality_percent"]
            values.append("未计划" if row[ds + "_status"] == "NOT_PLANNED_IN_DECLARED_MATRIX" else "待完成/无主分" if value is None else f"{value:.3f} ({row[ds + '_candidate_minus_same500_FP16_pp']:+.3f})")
        md.append("| " + row["method"] + " | " + " | ".join(values) + " |")
    md += ["", "完整资源逐域列于 competitive_long.csv：mean MiB、mean complete bit、max请求bit，以及全500请求严格1.3/1.8资格。不能用均bit掩盖超预算请求，不能把名义payload当完整预算。不同资源的分数是前沿比较，不是等字节因果收益。", "",
           "KIVI-style技术锚与源示例g32/R32分列；Calib为common-FP16/FP32-affine+text32数值port，未冒称作者针对本模型的新校准；OnlyTerp-MSE/QJL0与0xSero完整MSE+QJL/R128源路径分列，后者也不冒称Google官方实现或论文精确bit点。", "",
           "部分外部方法尚未完成，保留null；本报告不声称所有126/24外部目标或Round9新方法研究已完成。FP16为经验同题参考，并非数学上界。未按本表分数事后挑选方法；CI不直接解释为统计赢家或等价。", "",
           "R7的8组大raw各只进行一次严格来源/输入/物理owner核验；基线复用此前SourceRun/逐行/物理原证明，重核当前小来源文件、数字依赖和评分缓存，未重复重建其大raw。没有新GPU或重新评分。", ""]
    (out / "REPORT.zh.md").write_text("\n".join(md), encoding="utf-8")
    write(out / "COMPLETION.json", {"status": report["status"], "R7_cells": 8, "R7_QA": 4000,
          "files": [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(out.iterdir()) if p.is_file()], "new_GPU_QA": 0})
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    p.add_argument("--harvest", type=Path, default=E / "results/r7_matrix500_v2/harvest")
    p.add_argument("--output", type=Path, default=FINAL)
    p.add_argument("--allow-pending", action="store_true", help="only write readiness; never partial tables")
    args = p.parse_args()
    require(args.output.resolve().is_relative_to(E.resolve()), "output must stay on E project data disk")
    analysis = module("r7_competition_strict8", ROOT / "ops/analyze_r7_fixed500_v2.py", R7_ANALYSIS_SHA)
    for name, expected in analysis.PINS.items():
        require(sha(ROOT / name) == expected, "frozen prerequisite differs: " + name)
    rows = analysis.formal_rows(analysis.MANIFEST)
    ready = analysis.readiness(rows, args.harvest)
    if args.allow_pending or not ready["ready"]:
        status = {k: ready[k] for k in ("status", "expected_cells", "expected_QA", "closed_metadata_cells")}
        status.update(competitive_table_written=False, final_quality_claimed=False,
                      cells=[{k: x[k] for k in ("profile", "dataset", "missing")} for x in ready["cells"]])
        if args.allow_pending:
            require(args.output.resolve() != FINAL.resolve() and not args.output.exists(), "pending proof must use a separate immutable directory")
            write(args.output / "READINESS.json", status)
        print(json.dumps(status))
        return
    report = make_table(args, analysis, rows, ready)
    print(json.dumps({k: report[k] for k in ("status", "R7_cells", "R7_QA", "status_counts")}))


if __name__ == "__main__":
    main()
