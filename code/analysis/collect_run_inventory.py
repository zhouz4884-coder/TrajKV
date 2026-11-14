"""Inventory raw run structure and provenance, without reading score summaries.

Counts describe output records and image clusters. They do not establish
quality, independence between methods/budgets, or a publication conclusion.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

SHA = re.compile(r"[0-9a-fA-F]{64}\Z")
SOURCE_FIELDS = {"sources", "source_sha256", "source_hashes", "input_hashes"}
DIGEST_CACHE = {}


def digest(path):
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in DIGEST_CACHE:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1048576), b""):
                h.update(chunk)
        DIGEST_CACHE[key] = h.hexdigest()
    return DIGEST_CACHE[key]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def excluded(path, results):
    parts = path.relative_to(results).parts
    return any(part.startswith("analysis") or part in ("source_snapshot", "recovered_inputs", "__pycache__") for part in parts)


def local_path(name, project):
    text = str(name).replace("\\", "/")
    direct = Path(name)
    if direct.is_file():
        return direct
    marker = "/TrajKV/"
    if marker in text:
        candidate = project / text.split(marker, 1)[1]
        if candidate.is_file():
            return candidate
    jobs_marker = "/TrajKV/jobs/"
    if jobs_marker in text:
        candidate = project / "audit/job_sources" / text.split(jobs_marker, 1)[1]
        if candidate.is_file():
            return candidate
    if not direct.is_absolute():
        candidate = project / text
        if candidate.is_file():
            return candidate
    return None


def declared_sources(value, prefix=""):
    found = []
    if not isinstance(value, dict):
        return found
    for key, child in value.items():
        location = prefix + "." + key if prefix else key
        if key in SOURCE_FIELDS and isinstance(child, dict):
            for name, expected in child.items():
                if isinstance(expected, str) and SHA.fullmatch(expected):
                    found.append((location, name, expected.lower()))
        elif isinstance(child, dict):
            found.extend(declared_sources(child, location))
    return found


def retained_source_index(project):
    index = defaultdict(list)
    candidates = set()
    for relative in ("audit/job_sources", "audit/recovered-by-content-hash", "audit/source_recovery_candidates"):
        root = project / relative
        if root.is_dir():
            candidates.update(p for p in root.rglob("*") if p.is_file() and p.suffix in (".py", ".json"))
    # Historical source copies may live in any leaf run, including numbered
    # timing snapshots. JSON is needed for exact model metadata identities;
    # only bytes are hashed, never scored or interpreted as observations.
    for snapshot in (project / "results").rglob("source_snapshot"):
        if snapshot.is_dir():
            candidates.update(p for p in snapshot.rglob("*") if p.is_file() and p.suffix in (".py", ".json"))
    for path in sorted(candidates):
        index[digest(path)].append(path)
    return index


def source_checks(manifests, folder, project, retained):
    checks = []
    for manifest_name, value in manifests.items():
        for field, name, expected in declared_sources(value):
            basename = str(name).replace("\\", "/").rsplit("/", 1)[-1]
            candidates = []
            snapshot = folder / "source_snapshot" / basename
            if snapshot.is_file():
                candidates.append(("local_source_snapshot", snapshot))
            current = local_path(name, project)
            if current and all(current.resolve() != p.resolve() for _, p in candidates):
                candidates.append(("current_local_copy", current))
            for recovered in retained.get(expected, []):
                if all(recovered.resolve() != p.resolve() for _, p in candidates):
                    candidates.append(("retained_exact_content_hash_copy", recovered))
                    break
            observed = [{"kind": kind, "path": str(p), "sha256": digest(p)} for kind, p in candidates]
            matches = [x for x in observed if x["sha256"] == expected]
            checks.append({"manifest": manifest_name, "field": field, "declared_path": name,
                           "expected_sha256": expected, "status": "VERIFIED_AVAILABLE_COPY" if matches else "AVAILABLE_COPY_MISMATCH" if observed else "RECORDED_HASH_ONLY_NO_LOCAL_COPY",
                           "matching_copy": matches[0] if matches else None, "available_copies": observed})
    counts = Counter(x["status"] for x in checks)
    return {"declared_files": len(checks), "status_counts": dict(counts),
            "all_declared_files_verified": bool(checks) and counts["VERIFIED_AVAILABLE_COPY"] == len(checks),
            "checks": checks,
            "scope": "A matching retained source snapshot can validate a historical hash even when current code differs; unavailable historical copies remain explicit"}


def make_sample_lookup(project):
    lookup = {}
    for path in sorted((project / "configs/data").glob("*_runner.json")):
        try:
            rows = load(path)
        except (ValueError, OSError):
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("id"):
                lookup.setdefault(str(row["id"]), row)
    return lookup


def row_category(row, filename):
    if filename == "canonical_full.jsonl":
        return "canonical_generation_for_timing"
    if "runner_result" in row or row.get("teacher_forced_output_is_not_application_quality"):
        return "teacher_forced_timing"
    if "canonical_generation_side" in row or "native_branch" in row or "parity_scope" in row:
        return "native_vs_fused_diagnostic"
    if "canonical_turns" in row or "intervention_id" in row or "numerical_null_distribution" in row:
        return "teacher_forced_diagnostic"
    if row.get("protocol") == "fixed_tokens" or row.get("quality_scoring_allowed") is False:
        return "teacher_forced_generation_diagnostic"
    if row.get("protocol") == "free_running" and isinstance(row.get("turns"), list):
        return "free_running_answers"
    return "structural_diagnostic_or_unspecified"


def image_identity(row, samples):
    sid = str(row.get("session_id", row.get("id", "")))
    sample = samples.get(sid, {})
    nested = row.get("runner_result", {}) if isinstance(row.get("runner_result"), dict) else {}
    iid = row.get("source_image_id", row.get("image_id", nested.get("image_id", sample.get("source_image_id", sample.get("image_id")))))
    hashes = row.get("hashes", sample.get("hashes", []))
    if isinstance(hashes, str):
        hashes = [hashes]
    hashes = sorted(set(str(h) for h in hashes if isinstance(h, str) and SHA.fullmatch(h)))
    dataset = row.get("dataset", nested.get("dataset", sample.get("dataset")))
    if hashes:
        key = "sha256:" + "+".join(hashes)
    elif iid is not None:
        key = "source-id:" + str(dataset or "unknown") + ":" + str(iid)
    else:
        return None
    return {"key": key, "source_image_id": str(iid) if iid is not None else None,
            "image_hashes": hashes, "dataset": dataset, "session_id": sid or None,
            "identity_scope": "exact_image_hash" if hashes else "source_id_fallback"}


def scan_jsonl(path, samples, primary):
    before = path.stat()
    h = hashlib.sha256()
    categories, protocols, models = Counter(), Counter(), set()
    sessions, images, unique_turns, paired_keys = set(), {}, set(), Counter()
    rows = answer_turns = unscored_descriptions = forced_turns = diagnostic_canonical_turns = turn_items = 0
    errors = []
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            h.update(line)
            if not line.strip():
                continue
            try:
                row = json.loads(line.decode("utf-8-sig"))
                if not isinstance(row, dict):
                    raise ValueError("JSONL row is not an object")
            except (ValueError, UnicodeDecodeError) as error:
                errors.append({"line": number, "error_type": type(error).__name__})
                continue
            rows += 1
            category = row_category(row, path.name)
            categories[category] += 1
            sid = str(row.get("session_id", row.get("id", "")))
            if sid:
                sessions.add(sid)
            identity = image_identity(row, samples)
            if identity:
                images[identity["key"]] = identity
            nested = row.get("runner_result", {}) if isinstance(row.get("runner_result"), dict) else {}
            for value in (row.get("model"), nested.get("model")):
                if value:
                    models.add(str(value))
            protocol = row.get("protocol", nested.get("protocol"))
            if protocol:
                protocols[str(protocol)] += 1
            turns = row.get("turns", nested.get("turns", []))
            turns = turns if isinstance(turns, list) else []
            for index, turn in enumerate(turns, 1):
                if not isinstance(turn, dict):
                    continue
                unique_turns.add((sid, str(turn.get("turn", index))))
            if row.get("turn") is not None:
                unique_turns.add((sid, str(row["turn"])))
            if category == "free_running_answers":
                answer_turns += sum(isinstance(t, dict) and "answer" in t for t in turns)
                if row.get("first_turn_scored") is False and turns:
                    unscored_descriptions += 1
            elif category in ("teacher_forced_timing", "teacher_forced_generation_diagnostic"):
                forced_turns += len(turns)
            canonical = row.get("canonical_turns", [])
            turn_items += len(turns)
            if isinstance(canonical, list):
                diagnostic_canonical_turns += len(canonical)
                turn_items += len(canonical)
                for index, turn in enumerate(canonical, 1):
                    unique_turns.add((sid, str(turn.get("turn", index) if isinstance(turn, dict) else index)))
            if row.get("method") is not None:
                paired_keys[(sid, str(row["method"]), str(row.get("policy", "")))] += 1
    after = path.stat()
    stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return {"path": str(path), "primary": primary, "bytes_at_scan_start": before.st_size,
            "bytes_at_scan_end": after.st_size, "sha256_of_bytes_read": h.hexdigest(),
            "stable_during_read": stable, "valid_json_records": rows, "invalid_lines": errors,
            "categories": dict(categories), "protocols": dict(protocols), "models": sorted(models),
            "sessions": sorted(sessions), "independent_images": sorted(images.values(), key=lambda x: x["key"]),
            "turn_items": turn_items,
            "unique_session_turn_identifiers": len(unique_turns), "free_running_answer_turns": answer_turns,
            "unscored_description_turns": unscored_descriptions, "teacher_forced_turn_items": forced_turns,
            "diagnostic_canonical_turn_items": diagnostic_canonical_turns,
            "duplicate_session_method_policy_records": sum(n - 1 for n in paired_keys.values()),
            "duplicates_scope": "Repeated timing trials and dose/budget records are retained output records, not new independent images"}


def scan_run(folder, project, results, samples, retained):
    manifests, manifest_meta = {}, {}
    for name in ("manifest.json", "suite_manifest.json", "sweep_manifest.json", "job_source_sha256.json"):
        path = folder / name
        if path.is_file():
            try:
                value = load(path)
                manifests[name] = value
                manifest_meta[name] = {"sha256": digest(path), "valid_json": isinstance(value, dict)}
            except (ValueError, OSError):
                manifest_meta[name] = {"sha256": digest(path), "valid_json": False}
    complete_path = folder / "complete.json"
    complete = {"present": complete_path.is_file(), "valid_json_object": False}
    if complete_path.is_file():
        complete["sha256"] = digest(complete_path)
        try:
            value = load(complete_path)
            complete["valid_json_object"] = isinstance(value, dict)
            if isinstance(value, dict):
                complete["declared_status"] = value.get("status")
                complete["declared_counts"] = {k: value[k] for k in ("records", "records_added", "sessions", "paid_calls") if k in value and isinstance(value[k], (int, float, str))}
        except ValueError:
            pass
    jsonls = sorted(folder.glob("*.jsonl"))
    manifest = manifests.get("manifest.json", {})
    arguments = manifest.get("arguments", {}) if isinstance(manifest, dict) else {}
    is_composite = "suite_manifest.json" in manifests or "sweep_manifest.json" in manifests
    expected_children = []
    if "suite_manifest.json" in manifests:
        expected_children = [r["name"] for r in manifests["suite_manifest.json"].get("plan", {}).get("runs", [])]
    if "sweep_manifest.json" in manifests:
        expected_children = [str(r["output"]).replace("\\", "/").rsplit("/", 1)[-1] for r in manifests["sweep_manifest.json"].get("plan", {}).get("children", [])]
    kind = "composite" if is_composite else "hardware_profile" if (folder / "raw_metrics.csv").exists() else "timing" if "timing" in folder.name else "diagnostic" if (folder / "surgery_records.jsonl").exists() or "canonical_generation" in manifest or "native_sdpa_function_module" in manifest else "generation_or_other"
    primary_names = {"sessions.jsonl"} if (folder / "sessions.jsonl").is_file() else {"records.jsonl"}
    raw = [scan_jsonl(path, samples, path.name in primary_names) for path in jsonls]
    images, sessions, models, categories, protocols = {}, set(), set(), Counter(), Counter()
    free_turns = forced_turns = canonical_turns = primary_records = 0
    for file in raw:
        if not file["primary"]:
            continue
        primary_records += file["valid_json_records"]
        images.update({x["key"]: x for x in file["independent_images"]})
        sessions.update(file["sessions"])
        models.update(file["models"])
        categories.update(file["categories"])
        protocols.update(file["protocols"])
        free_turns += file["free_running_answer_turns"]
        forced_turns += file["teacher_forced_turn_items"]
        canonical_turns += file["diagnostic_canonical_turn_items"]
    if arguments.get("model"):
        models.add(str(arguments["model"]))
    csvs = []
    if kind == "hardware_profile":
        for path in sorted(folder.glob("*.csv")):
            with path.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
                csv_rows = sum(bool(row) for row in csv.reader(stream))
            csvs.append({"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size,
                         "nonempty_csv_rows_including_headers": csv_rows,
                         "category": "hardware_counter_raw_not_application_answers"})
        sid = manifest.get("target_session_id")
        if sid:
            identity = image_identity({"session_id": sid}, samples)
            if identity:
                images[identity["key"]] = identity
    stable = all(file["stable_during_read"] for file in raw)
    invalid = sum(len(file["invalid_lines"]) for file in raw)
    valid_complete = complete["valid_json_object"] and stable and not invalid
    return {"run": folder.relative_to(results).as_posix(), "kind": kind,
            "completion": complete, "structurally_complete_snapshot": bool(valid_complete),
            "manifests": manifest_meta, "source_provenance": source_checks(manifests, folder, project, retained),
            "raw_files": raw, "raw_counter_files": csvs, "primary_record_count": primary_records,
            "all_raw_jsonl_records_including_derived": sum(file["valid_json_records"] for file in raw),
            "primary_turn_items": sum(file["turn_items"] for file in raw if file["primary"]),
            "free_running_answer_turns": free_turns, "teacher_forced_turn_items": forced_turns,
            "diagnostic_canonical_turn_items": canonical_turns, "primary_record_categories": dict(categories),
            "protocols": dict(protocols), "models": sorted(models), "session_ids": sorted(sessions),
            "independent_images": sorted(images.values(), key=lambda x: x["key"]),
            "independent_image_count": len(images), "budget_bytes": arguments.get("read_budget_bytes"),
            "composite_expected_child_names": expected_children,
            "counting_scope": "Primary raw files only for run counts; derived scatter/control/warmup files retained separately"}


def collect(project, results):
    project, results = Path(project).resolve(), Path(results).resolve()
    samples = make_sample_lookup(project)
    retained = retained_source_index(project)
    folders = {p.parent for p in results.rglob("*.jsonl") if not excluded(p, results)}
    for name in ("suite_manifest.json", "sweep_manifest.json", "raw_metrics.csv"):
        folders.update(p.parent for p in results.rglob(name) if not excluded(p, results))
    runs = [scan_run(folder, project, results, samples, retained) for folder in sorted(folders)]
    by_run = {r["run"]: r for r in runs}
    # Composite entries are containers, never additional observations.
    for run in sorted(runs, key=lambda x: x["run"].count("/"), reverse=True):
        if run["kind"] != "composite":
            continue
        prefix = run["run"] + "/"
        descendants = [r for r in runs if r["run"].startswith(prefix) and r["kind"] != "composite"]
        expected = [prefix + name for name in run["composite_expected_child_names"]]
        missing = [name for name in expected if name not in by_run]
        images = {x["key"]: x for child in descendants for x in child["independent_images"]}
        run.update(child_runs=[r["run"] for r in descendants],
                   child_primary_record_count=sum(r["primary_record_count"] for r in descendants),
                   independent_images=sorted(images.values(), key=lambda x: x["key"]),
                   independent_image_count=len(images),
                   models=sorted({m for child in descendants for m in child["models"]}),
                   missing_declared_children=missing,
                   child_complete_status={r["run"]: r["structurally_complete_snapshot"] for r in descendants},
                   contributes_additional_observations=False)
        run["structurally_complete_snapshot"] = run["structurally_complete_snapshot"] and bool(descendants) and not missing and all(r["structurally_complete_snapshot"] for r in descendants)
    cohort_groups = defaultdict(list)
    global_images, free_images, categories = {}, {}, Counter()
    for run in runs:
        identities = sorted(x["key"] for x in run["independent_images"])
        run["image_cohort_sha256"] = hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()
        if run["kind"] != "composite":
            global_images.update({x["key"]: x for x in run["independent_images"]})
            if run["primary_record_categories"].get("free_running_answers"):
                free_images.update({x["key"]: x for x in run["independent_images"]})
            categories.update(run["primary_record_categories"])
            if identities:
                cohort_groups[run["image_cohort_sha256"]].append(run["run"])
    repeated = [{"image_cohort_sha256": signature, "independent_images": by_run[names[0]]["independent_image_count"], "runs": names,
                 "contains_budget_sweep_children": any("budget_" in name for name in names)} for signature, names in sorted(cohort_groups.items()) if len(names) > 1]
    return {"schema": "trajkv.run_inventory.v1", "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "project_root": str(project), "results_root": str(results), "collector_sha256": digest(Path(__file__)),
            "scope": "Raw structure and provenance only; no answer scoring, no quality/timing statistics, no analysis summaries read",
            "excluded": "Every analysis* directory, recovered_inputs, source_snapshot JSONL, and all summary.json files",
            "runs": runs, "totals": {"run_containers": len(runs), "leaf_runs": sum(r["kind"] != "composite" for r in runs),
            "composite_containers": sum(r["kind"] == "composite" for r in runs), "primary_records_by_category": dict(categories),
            "global_independent_image_union": len(global_images), "free_running_image_union": len(free_images),
            "global_image_identities": sorted(global_images.values(), key=lambda x: x["key"])},
            "repeated_image_cohorts": repeated,
            "independence_warning": "Record, method, policy, dose, replay, timing repeat and budget counts must not be summed as independent samples; image union and paired cohorts are reported separately"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--results-root", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    results = args.results_root or args.project_root / "results"
    output = args.output or args.project_root / "audit/run_inventory.json"
    report = collect(args.project_root, results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "run_containers": report["totals"]["run_containers"],
                      "leaf_runs": report["totals"]["leaf_runs"], "independent_image_union": report["totals"]["global_independent_image_union"]}))


if __name__ == "__main__":
    main()
