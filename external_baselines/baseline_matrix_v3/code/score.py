"""Immutable CPU-only offline scoring with raw SHA/coverage/source checks.

Detailed scored artifacts default to the E mirror. Only small summaries and
pointers are copied to the workspace. Raw files and references are never edited.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from metrics import score_question, METRIC_CONTRACT_VERSION

RAW_FILES = ("manifest.json", "records.jsonl", "complete.json")
FINISHED_STATUSES = {"PASS", "UNSUPPORTED_MODEL", "HARDWARE_UNSUPPORTED", "OOM_AT_DECLARED_INPUT",
                     "METHOD_PORT_UNVERIFIED", "INPUT_MISMATCH", "OWNER_AUDIT_FAILED", "FAILED_RUN", "STALE_PROTOCOL"}


class IntegrityError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows, hashes = [], []
    with Path(path).open("rb") as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), f"Blank JSONL row {number}: {path}")
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise IntegrityError(f"Invalid JSONL row {number}: {path}") from exc
            hashes.append(hashlib.sha256(line).hexdigest())
    return rows, hashes


def unique_lookup(rows, label):
    require(all(isinstance(row, dict) and isinstance(row.get("id"), str) for row in rows), f"Missing string IDs in {label}")
    ids = [r["id"] for r in rows]
    require(len(ids) == len(set(ids)), f"Duplicate IDs in {label}")
    return {r["id"]: r for r in rows}


def resolve_raw(path):
    path = Path(path)
    pointer = path / "artifact_pointer.json"
    if pointer.is_file():
        path = Path(read_json(pointer)["local_raw_dir"])
    require(path.is_dir(), f"Raw output directory missing: {path}")
    return path.resolve()


def validate_raw(path, refs_path, *, allow_fixture=False):
    raw_dir = resolve_raw(path)
    before = {name: sha(raw_dir / name) for name in RAW_FILES}
    manifest, complete = read_json(raw_dir / "manifest.json"), read_json(raw_dir / "complete.json")
    derived_proof = None
    is_derived = manifest.get("schema") == "baseline_matrix.derived_prediction_view.v1"
    if is_derived:
        require(not allow_fixture, "A derived source view cannot bypass its provenance with fixture scope")
        from prediction_cache import validate_view_source_provenance
        derived_proof = validate_view_source_provenance(raw_dir)
        require(derived_proof["status"] == "PASS_DERIVED_SOURCE_PROVENANCE" and derived_proof["inference_run"] is False,
                "Derived view source evidence was not verified")
        require(derived_proof["raw_hashes"] == before, "Derived source validator disagrees with local raw SHA")
    else:
        receipt = read_json(raw_dir / "retrieval_receipt.json")
        if allow_fixture:
            require(receipt["status"] == "CPU_LOCAL_FIXTURE_ONLY", "Fixture scope must be explicit")
        else:
            require(receipt["status"] == "PASS_REMOTE_RAW_SHA256_EXACT" and receipt["remote_hashes_computed_independently"] is True,
                    "Independent remote SHA retrieval receipt required, including complete.json")
        anchored = {r["file"]: r for r in receipt["files"]}
        require(set(anchored) == set(RAW_FILES), "Retrieval receipt must anchor exactly all three raw files")
        for name in RAW_FILES:
            require(anchored[name]["sha256"] == before[name] and anchored[name]["bytes"] == (raw_dir / name).stat().st_size,
                    f"Remote-anchored raw bytes mismatch: {name}")
    require(complete["manifest_sha256"] == before["manifest.json"], "Complete/manifest SHA mismatch")
    require(complete["records_jsonl_sha256"] == before["records.jsonl"], "Complete/records SHA mismatch")
    require(complete["coverage_complete"] is True, "Runner did not close expected coverage")
    require(complete["quality_scored"] is False and manifest["references_read"] is False, "Expected unscored, reference-free raw inference")
    rows, line_hashes = read_jsonl(raw_dir / "records.jsonl")
    records = unique_lookup(rows, "raw records")
    expected = manifest["expected_ids"]
    require(isinstance(expected, list) and len(expected) == len(set(expected)), "Invalid expected ID sequence")
    require(len(rows) == len(expected) == manifest["expected_QA"] == complete["expected_QA"] == complete["recorded_QA"], "Expected/recorded QA counts differ")
    require([r["id"] for r in rows] == expected, "Raw records must cover the exact expected IDs once, in manifest order")
    require(all(r.get("status") in FINISHED_STATUSES for r in rows), "Unfinished/unknown raw status in completed run")
    counts = dict(Counter(r["status"] for r in rows))
    require(counts == complete["counts"], "Raw status counts differ from complete.json")
    expected_status = "PASS_RAW_COMPLETE" if counts.get("PASS", 0) == len(rows) else "COMPLETE_WITH_FAILURES"
    require(complete["status"] == expected_status, "Complete status disagrees with record failures")
    refs_path = Path(refs_path).resolve()
    require(refs_path.name.endswith("_refs.jsonl"), "Explicit frozen *_refs.jsonl required")
    input_path = refs_path.with_name(refs_path.name.replace("_refs.jsonl", "_inputs.jsonl"))
    source_lock_path = refs_path.parent / "source_lock.json"
    source_lock = read_json(source_lock_path)
    for p in (refs_path, input_path):
        entry = source_lock["files"][p.name]
        require(sha(p) == entry["sha256"] and p.stat().st_size == entry["bytes"], f"Frozen dataset file differs: {p.name}")
    require(manifest["inputs_sha256"] == sha(input_path), "Run input cohort SHA differs from the paired refs/input cohort")
    config_input_name = PurePosixPath(manifest["config"]["inputs"].replace("\\", "/")).name
    require(config_input_name == input_path.name, "Manifest refers to a different input phase/cohort")
    inputs, _ = read_jsonl(input_path)
    references, _ = read_jsonl(refs_path)
    input_lookup, ref_lookup = unique_lookup(inputs, "frozen inputs"), unique_lookup(references, "frozen references")
    require(set(input_lookup) == set(ref_lookup), "Frozen input/reference ID cohorts differ")
    config = manifest["config"]
    if config.get("ids") is not None:
        selected = list(config["ids"])
    else:
        start = int(config.get("start", 0))
        selected = [r["id"] for r in inputs[start:start + int(config.get("count", len(inputs) - start))]]
    require(expected == selected, "Expected IDs disagree with config selection over locked inputs")
    require(set(expected).issubset(ref_lookup), "Missing references for expected run IDs")
    datasets = {input_lookup[i]["dataset"] for i in expected}
    require(len(datasets) == 1, "One dataset per raw run is required")
    dataset = datasets.pop()
    for row in rows:
        frozen, ref = input_lookup[row["id"]], ref_lookup[row["id"]]
        for field in ("dataset", "dataset_revision", "split", "source_cluster_id", "source_id", "question_id", "original_image_sha256", "question_sha256"):
            require(row.get(field) == frozen.get(field), f"Raw/frozen input mismatch: {row['id']} {field}")
        require(ref["source_cluster_id"] == frozen["source_cluster_id"], "Reference source-cluster identity mismatch")
        if dataset == "chartqa":
            require(ref.get("group") in ("human", "aug", "augmented"), "ChartQA must retain explicit human/aug group")
            if row.get("group") is not None:
                require(row["group"] == ref["group"], "ChartQA raw/reference group mismatch")
        require(row.get("official_score") is None, "Raw result unexpectedly contains an existing score")
        if row["status"] == "PASS":
            require(isinstance(row.get("raw_prediction"), str), "PASS requires original prediction string")
            token_ids = row.get("output_token_ids")
            require(isinstance(token_ids, list) and token_ids and all(type(i) is int for i in token_ids), "PASS requires genuine output token IDs")
            if row.get("raw_question") is not None:
                require(row["raw_question"] == frozen["question"], "Raw question changed")
    return {"raw_dir": raw_dir, "manifest": manifest, "complete": complete, "rows": rows, "line_hashes": line_hashes,
            "expected": expected, "refs": ref_lookup, "refs_path": refs_path, "input_path": input_path,
            "source_lock_path": source_lock_path, "source_lock": source_lock, "dataset": dataset,
            "raw_hashes": before, "reference_cohort_questions": len(references), "fixture_only": allow_fixture,
            "derived_proof": derived_proof}


def cluster_bootstrap(rows, key, repetitions, seed):
    if not rows:
        return {"mean": None, "ci95": None, "questions": 0, "source_clusters": 0}
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_cluster_id"]].append(float(row[key]))
    ordered = sorted(groups)
    sums = np.asarray([sum(groups[k]) for k in ordered], dtype=np.float64)
    sizes = np.asarray([len(groups[k]) for k in ordered], dtype=np.float64)
    mean = float(sums.sum() / sizes.sum())
    result = {"mean": mean, "questions": len(rows), "source_clusters": len(ordered),
              "estimand": "question-weighted mean", "resampling_unit": "source_cluster_id"}
    if len(ordered) < 2:
        result.update(ci95=None, ci_status="INSUFFICIENT_INDEPENDENT_CLUSTERS")
        return result
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(ordered), size=(repetitions, len(ordered)))
    # A sampled cluster contributes ALL its questions. This ratio is not an
    # unweighted mean of cluster means when source question counts differ.
    sampled = sums[draws].sum(1) / sizes[draws].sum(1)
    result.update(ci95=np.quantile(sampled, [.025, .975]).tolist(), ci_status="PERCENTILE_CLUSTER_BOOTSTRAP")
    return result


def summarize_group(rows, repetitions, seed):
    valid = [r for r in rows if r["score_status"] == "SCORED_PASS"]
    conditional = cluster_bootstrap(valid, "score", repetitions, seed)
    failures = len(rows) - len(valid)
    primary = dict(conditional) if not failures else {"mean": None, "ci95": None, "questions": len(rows),
        "source_clusters": len({r["source_cluster_id"] for r in rows}), "ci_status": "UNAVAILABLE_WITH_FAILURES"}
    return {"expected_questions": len(rows), "scored_pass_questions": len(valid), "failure_questions": failures,
            "coverage_fraction": len(valid) / len(rows) if rows else None,
            "primary": primary, "conditional_success_only": {**conditional, "is_full_cohort_quality": failures == 0},
            "failed_ids": [r["id"] for r in rows if r["score_status"] != "SCORED_PASS"]}


def score_run(raw_path, refs_path, output_dir, *, repetitions=2000, seed=20261119, summary_dir=None, allow_fixture=False):
    require(repetitions >= 1000, "Use at least 1000 cluster bootstrap draws")
    validated = validate_raw(raw_path, refs_path, allow_fixture=allow_fixture)
    gate_path = ROOT / "audit/metrics_gate_v2.json"
    metric_gate = read_json(gate_path)
    for relative, expected_sha in metric_gate["code_hashes"].items():
        require(sha(ROOT / relative) == expected_sha, f"Frozen metric source changed: {relative}")
    output_dir = Path(output_dir)
    require(not output_dir.exists(), "Scored destination already exists; use a new immutable version")
    output_dir.mkdir(parents=True)
    scored = []
    for number, (raw, raw_line_sha) in enumerate(zip(validated["rows"], validated["line_hashes"]), 1):
        ref = validated["refs"][raw["id"]]
        row = {"id": raw["id"], "dataset": validated["dataset"], "model_id": raw["model_id"], "method": raw["method"],
               "source_cluster_id": ref["source_cluster_id"], "group": ref.get("group"), "raw_status": raw["status"],
               "raw_prediction": raw.get("raw_prediction"), "output_token_ids": raw.get("output_token_ids"),
               "raw_jsonl_line_number": number, "raw_jsonl_exact_line_sha256": raw_line_sha,
               "score": None, "exact": None, "score_status": "NOT_SCORED_RAW_FAILURE"}
        for field in ("original_image_sha256", "model_commit", "dataset_revision", "split", "question_id", "full_prompt_hash",
                      "question_suffix_sha256", "actual_visual_token_count", "stored_visual_KV_bytes", "actual_effective_visual_bits",
                      "wall_ms", "encode_ms", "prefill_ms", "ttft_ms", "peak_allocated_bytes"):
            if field in raw:
                row[field] = raw[field]
        if raw["status"] == "PASS":
            try:
                metric = score_question(validated["dataset"], raw["raw_prediction"], ref)
                row.update(metric, id=raw["id"], source_cluster_id=ref["source_cluster_id"], raw_status=raw["status"], score_status="SCORED_PASS")
            except Exception as exc:
                row.update(score_status="SCORE_FAILED", score_error_type=type(exc).__name__, score_error=str(exc))
        else:
            row.update(raw_error_type=raw.get("error_type"), raw_error=raw.get("error"))
        scored.append(row)
    group_stats = {"overall": summarize_group(scored, repetitions, seed)}
    if validated["dataset"] == "chartqa":
        for group in ("human", "aug"):
            subset = [r for r in scored if ("aug" if r.get("group") == "augmented" else r.get("group")) == group]
            if subset:
                group_stats[group] = summarize_group(subset, repetitions, seed)
    diagnostic_means = {}
    diagnostic_names = sorted({name for row in scored for name in row.get("diagnostic_scores", {})})
    for name in diagnostic_names:
        diagnostic_rows = [{**row, "diagnostic_value": row["diagnostic_scores"][name]} for row in scored if row["score_status"] == "SCORED_PASS"]
        value = cluster_bootstrap(diagnostic_rows, "diagnostic_value", repetitions, seed)
        diagnostic_means[name] = {"primary_full_cohort_mean": value["mean"] if len(diagnostic_rows) == len(scored) else None,
                                  "primary_full_cohort_ci95": value["ci95"] if len(diagnostic_rows) == len(scored) else None,
                                  "conditional_success_only": value}
    primary = group_stats["overall"]["primary"]
    summary = {"schema": "baseline_matrix.offline_score.v1", "status": "PASS_SCORED_COMPLETE" if primary["mean"] is not None else "COMPLETE_SCORED_WITH_FAILURES",
        "dataset": validated["dataset"], "model_id": validated["manifest"]["model_spec"]["model_id"],
        "method": validated["manifest"]["config"]["method"]["id"], "metric_contract_version": METRIC_CONTRACT_VERSION,
        "metric_ids": sorted({row["metric_id"] for row in scored if row["score_status"] == "SCORED_PASS"}),
        "metric_provenance": sorted({row["metric_provenance"] for row in scored if row["score_status"] == "SCORED_PASS"}),
        "groups": group_stats, "primary_mean": primary["mean"], "primary_ci95": primary["ci95"],
        "primary_mean_percent": 100 * primary["mean"] if primary["mean"] is not None else None,
        "primary_ci95_percent": [100 * x for x in primary["ci95"]] if primary["ci95"] is not None else None,
        "diagnostic_metrics_not_selected_by_score": diagnostic_means,
        "raw_status_counts": dict(Counter(r["raw_status"] for r in scored)), "score_status_counts": dict(Counter(r["score_status"] for r in scored)),
        "expected_run_questions": len(scored), "reference_cohort_questions": validated["reference_cohort_questions"],
        "entire_reference_cohort_selected": len(scored) == validated["reference_cohort_questions"],
        "expected_ids_sha256": canonical_sha(validated["expected"]),
        "coverage_rule": "Any raw/scoring failure makes primary full-run mean/CI null; successful conditional mean is separately labeled, no fake zero or success-subset primary",
        "bootstrap": {"repetitions": repetitions, "seed": seed, "type": "percentile source-cluster resampling with replacement, question-weighted ratio", "scale": "0..1"},
        "integrity": {"status": "PASS_ALL_SHA_AND_EXPECTED_ID_CHECKS", "raw_files": validated["raw_hashes"],
            "retrieval_receipt_sha256": sha(validated["raw_dir"] / "retrieval_receipt.json") if validated["derived_proof"] is None else None,
            "derived_source_provenance": validated["derived_proof"], "refs_sha256": sha(validated["refs_path"]),
            "inputs_sha256": sha(validated["input_path"]), "dataset_source_lock_sha256": sha(validated["source_lock_path"]),
            "metric_source_lock_sha256": sha(ROOT / "audit/upstream/metrics_source_lock_v1.json"),
            "metric_code_hashes": metric_gate["code_hashes"], "scorer_sha256": sha(Path(__file__)),
            "raw_dir": str(validated["raw_dir"]), "refs_path": str(validated["refs_path"]),
            "fixture_only": allow_fixture}, "created_unix": time.time(), "scoring_uses_GPU": False,
        "prediction_provenance": ({"type": "derived_prediction_view", "inference_run": False,
                                   "origin_counts": validated["derived_proof"]["origin_counts"],
                                   "source_runs": validated["derived_proof"]["source_runs"]}
                                  if validated["derived_proof"] is not None else
                                  {"type": "original_raw_inference" if not allow_fixture else "CPU_fixture",
                                   "inference_run": not allow_fixture, "origin_counts": {"fresh": len(scored), "reused": 0}})}
    with (output_dir / "scored_records.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in scored:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    csv_fields = ["id", "dataset", "model_id", "method", "source_cluster_id", "group", "raw_status", "score_status", "metric_id", "score", "exact", "raw_prediction"]
    with (output_dir / "scores.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(scored)
    summary["derived_file_hashes"] = {name: sha(output_dir / name) for name in ["scored_records.jsonl", "scores.csv"]}
    require(all(sha(validated["raw_dir"] / name) == value for name, value in validated["raw_hashes"].items()), "Raw changed during scoring")
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with (output_dir / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
        fields = ["dataset", "model_id", "method", "group", "expected_questions", "scored_pass_questions", "failure_questions", "primary_mean", "ci95_low", "ci95_high", "conditional_mean"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for group, data in group_stats.items():
            ci = data["primary"]["ci95"] or [None, None]
            writer.writerow({"dataset": summary["dataset"], "model_id": summary["model_id"], "method": summary["method"], "group": group,
                **{name: data[name] for name in ["expected_questions", "scored_pass_questions", "failure_questions"]},
                "primary_mean": data["primary"]["mean"], "ci95_low": ci[0], "ci95_high": ci[1], "conditional_mean": data["conditional_success_only"]["mean"]})
    if summary_dir is not None:
        summary_dir = Path(summary_dir)
        require(not summary_dir.exists(), "Workspace summary destination exists; use new immutable version")
        summary_dir.mkdir(parents=True)
        for name in ["summary.json", "summary.csv"]:
            (summary_dir / name).write_bytes((output_dir / name).read_bytes())
        (summary_dir / "artifact_pointer.json").write_text(json.dumps({"scored_artifact_dir": str(output_dir.resolve()),
            "raw_dir": str(validated["raw_dir"]), "summary_sha256": sha(output_dir / "summary.json")}, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_output_dirs", nargs="+", type=Path)
    parser.add_argument("--refs", type=Path)
    parser.add_argument("--refs-root", type=Path, default=ROOT / "configs/datasets/v3")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-root", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()
    for directory in args.raw_output_dirs:
        raw = resolve_raw(directory)
        manifest = read_json(raw / "manifest.json")
        input_name = PurePosixPath(manifest["config"]["inputs"].replace("\\", "/")).name
        refs = args.refs or args.refs_root / input_name.replace("_inputs.jsonl", "_refs.jsonl")
        summary = score_run(raw, refs, args.output_root / raw.name, repetitions=args.bootstrap, seed=args.seed,
                            summary_dir=args.summary_root / raw.name if args.summary_root else None)
        print(json.dumps({key: summary[key] for key in ["status", "dataset", "model_id", "method", "expected_run_questions", "primary_mean", "primary_ci95"]}))


if __name__ == "__main__":
    main()
