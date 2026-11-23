"""Read-only admission, then optional UNSEALED Round11 four-cohort contract.

Default execution creates nothing. --prepare requires complete existing actual
scores, producer proofs, all pairs, a verified zero-inference combined view and
Root's report/table. No missing score is generated. Root alone may later seal,
deep-validate, export or publish a different reviewed stage copy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
E = Path("${EVIDENCE_ROOT}/results/round11")
BOOT = E / "bootstrap_v1"
REFS = ("FP16", "R7low", "R7high")
SELECTED = ("JointV", "JointSpatialWO")
SUPPORT_SHA = "aa4cef41a9e2ed1ab7da5078d6f0c1e7706c23bdb9c37a089e705c5d8bf27e43"
FAILED_SHA = "a773d93c3457f1ce070cb63fbccac55054776652cf3945df9ac5acc59dd438cf"
PUB_LOCK_SHA = "db21e6d3fb879dc75a0f8ec4e557591e114fe111a0c2fafd65572d79c9bc0617"
SELECTION_SHA = "45409e4adedae77660825c1bf1f9a39cf03d8a5dd14eae34374705b3b823f915"
DESIGN_SHA = "ebe818bbc67b1d189b4e27b15e2fdd4af59c0a7e5633c5b1125add1ccfac0c19"
PHASE_LABEL = "ROUND11_FINAL_DEV576_ORIGINAL128_ADDITIONAL384"
SPECS = {
    "primary": {
        "adapter": "round11_harvest_v4", "adapter_sha": "97891891fc054df918f5d43a00bb9c6f818fb586f6e06efcd1f47977243f8afd",
        "plan": "HARVEST_PLAN_v2.json", "plan_sha": "caecf7efc9783af23d83712190fa15ced3c56bbd42053224136e5a5fb6ae7fab",
        "schema": "trajkv.round11.harvest_summary.v1", "status": "PASS_ALL_28_PROVEN_VIEWS",
        "summary_sha": "f95a451ed652f2136efe73de99ce2fe04e53a16cb2f144f186d2fd7ab8ae6b67",
        "counts": dict(textvqa=32, infovqa=32, docvqa=16, chartqa=16), "views": 28, "fresh": 384, "reuse": 288, "pairs": 64,
        "modes": (*REFS, "JointV", "Spatial", "JointSpatial", "Shuffled"), "kind": "development",
        "cohort": "SourceGroupD96 primary development"},
    "wo": {
        "adapter": "round11_wo_harvest_v1", "adapter_sha": "d53152f6e8986c4ec2686f2d499689bee116389541434151391b845071544c79",
        "plan": "WO_HARVEST_PLAN_v1.json", "plan_sha": "6e45d4d2bf379a85008414ba0c89c659a773ed16d3b69d5defe1941b0ea39e90",
        "schema": "trajkv.round11_wo.harvest_summary.v1", "status": "PASS_ALL_20_PROVEN_VIEWS",
        "summary_sha": "92c8d97edfa97d0aa34c25b1009c0e92cfdce4d0c84ae42484ab9b383c1c2be4",
        "counts": dict(textvqa=32, infovqa=32, docvqa=16, chartqa=16), "views": 20, "fresh": 192, "reuse": 288, "pairs": 28,
        "modes": (*REFS, "SpatialWO", "JointSpatialWO"), "kind": "bounded_development",
        "cohort": "SourceGroupD96 bounded full-H WO development"},
    "confirm128": {
        "adapter": "round11_confirmation_harvest_v2", "adapter_sha": "6435fee9fffc38b873c73b82d9dcc82010c73c1acb123c2d7f6c47f46ea4ba9e",
        "plan": "CONFIRM_HARVEST_PLAN_v3.json", "plan_sha": "ca236c70452946206ee51a6ecd323e7d768ee5f61dae84ea8e035e4ece3e1df5",
        "schema": "trajkv.round11.confirmation_harvest_summary.v2", "status": "PASS_FRESH_R128_ALL_CELLS",
        "summary_sha": "1f0ed29c8bbf34f73c2e69c869241cfff5042d82ed53140d0dd6d1fa509c5329",
        "counts": dict(textvqa=48, infovqa=48, docvqa=16, chartqa=16), "views": 20, "fresh": 640, "reuse": 0, "pairs": 24,
        "modes": (*REFS, *SELECTED), "kind": "fresh_confirmation",
        "cohort": "Original128 fresh confirmation; methods selected on full D96 before original outcomes"},
    "expand384": {
        "adapter": "round11_expand_harvest_v1", "adapter_sha": "70a134f4343e58a170ba52d0e4603da2247221c09e1370038cf18dc296f5b2ba",
        "plan": "EXPAND384_HARVEST_PLAN_v1.json", "plan_sha": "9e04a9305fe5572252cbc7c986e7f9ccc856f2fcdebf2c440d0070d77e20272a",
        "schema": "trajkv.round11.expand384_harvest_summary.v1", "status": "PASS_FRESH_EXPAND384_ALL_CELLS", "summary_sha": None,
        "counts": dict(textvqa=144, infovqa=144, docvqa=48, chartqa=48), "views": 20, "fresh": 1920, "reuse": 0, "pairs": 24,
        "modes": (*REFS, *SELECTED), "kind": "fresh_fixed_extension",
        "cohort": "Additional384 fixed after partial original128 observation; primary new replication; original/additional/combined separate"},
}
EXPLICIT_STAGE_FILES = (
    "code/round11_confirmation_data_v1.py", "code/round11_expand384_data_v1.py",
    "ops/round11_prepare_primary_v2.py", "ops/round11_prepare_wo_v1.py", "ops/round11_prepare_confirmation_v2.py", "ops/round11_prepare_expand384_v1.py",
    "ops/round11_combine_holdouts_v1.py",
    "ops/round11_wo_stage_analysis_v1.py", "ops/round11_assemble_final_contract_v1.py",
    "configs/round11_v1/confirm128_data_plan_v1.json", "configs/round11_v1/expand384_data_plan_v1.json",
    "docs/PLAN_2026-11-23_ROUND11_SPATIAL_JOINT_v1.md", "docs/MATH_2026-11-23_ROUND11_CONDITIONAL_CODE_v1.md",
    "docs/PLAN_2026-11-23_ROUND11_WO_ADDENDUM_v1.md", "docs/PLAN_2026-11-23_ROUND11_EXPAND512_v1.md",
    "docs/THEORY_2026-11-23_WO_QUERY_AGNOSTIC_BOUND_v1.md",
)


class Pending(Exception):
    pass


def require(condition, message):
    if not condition:
        raise ValueError("ROUND11_FINAL_ASSEMBLY: " + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def ensure_files(paths):
    missing = [str(path) for path in paths if path is None or not Path(path).is_file()]
    if missing:
        raise Pending("missing existing evidence: " + "; ".join(missing))


def validate_header(label, summary):
    """Reject partial denominators before reading cell scores or creating output."""
    spec = SPECS[label]
    require(summary.get("schema") == spec["schema"], label + " summary schema differs")
    if summary.get("status") != spec["status"]:
        raise Pending(label + " actual final summary is incomplete")
    require(summary.get("canonical_SourceRun_admission_claimed") is False, label + " cannot establish canonical admission")
    if label in ("primary", "wo"):
        if any(summary.get(key, 0) < spec["views"] for key in ("closed_cells_or_views", "successful_complete_cells_or_views")) or summary.get("actual_closed_new_free_generation_QA", 0) < spec["fresh"] or summary.get("actual_closed_reused_reference_views", 0) < spec["reuse"]:
            raise Pending(label + " actual closed counts are incomplete")
        require(all(summary.get(key) == spec["views"] for key in ("expected_cells_or_views", "closed_cells_or_views", "successful_complete_cells_or_views")), label + " full view count differs")
        require(summary.get("expected_new_free_generation_QA") == summary.get("actual_closed_new_free_generation_QA") == spec["fresh"]
                and summary.get("expected_reused_reference_views") == summary.get("actual_closed_reused_reference_views") == spec["reuse"], label + " actual fresh/reuse count differs")
        point_status, fresh_key = "PASS_FULL_SOURCE_GROUP_SCORE", "fresh_free_generation_QA"
    else:
        if summary.get("actual_closed_fresh_QA", 0) < spec["fresh"]:
            raise Pending(label + " actual fresh denominator is incomplete")
        require(summary.get("expected_new_QA") == summary.get("actual_closed_fresh_QA") == spec["fresh"] and summary.get("reused_reference_views") == 0
                and summary.get("expected_fresh_reference_QA") == 3 * sum(spec["counts"].values()), label + " full actual fresh denominator differs")
        point_status, fresh_key = ("PASS_FRESH_R128_SOURCE_SCORE" if label == "confirm128" else "PASS_FRESH_EXPAND384_SOURCE_SCORE"), "fresh_QA"
        require(summary.get("selection_manifest_sha256") == SELECTION_SHA, label + " old selection changed")
        if label == "expand384":
            require(summary.get("expansion_design_sha256") == DESIGN_SHA, "expansion design differs")
    points = summary.get("points", [])
    expected = {(ds, mode) for ds in spec["counts"] for mode in spec["modes"]}
    keys = {(row.get("dataset"), row.get("mode")) for row in points}
    if len(points) < spec["views"] and keys <= expected and len(points) == len(keys):
        raise Pending(label + " existing score matrix is incomplete")
    require(len(points) == len(keys) == spec["views"] and keys == expected, label + " missing/duplicated/wrong-mode points")
    for row in points:
        n = spec["counts"][row["dataset"]]
        fresh = label not in ("primary", "wo") or row["mode"] not in REFS
        if row.get("status") != point_status or row.get("primary_mean") is None:
            raise Pending(label + " has a noncomplete scored point")
        require(row.get("expected_QA") == n and row.get(fresh_key) == (n if fresh else 0)
                and row.get("reused_reference_views") == (0 if fresh else n), label + " point denominator differs")
    require(summary.get("plan_sha256") == spec["plan_sha"], label + " actual plan SHA differs")


def contrasts(label, pairs):
    spec = SPECS[label]
    by_domain = {}
    for key in pairs:
        require("__" in key and "_minus_" in key, label + " malformed pair key")
        ds, pair = key.split("__", 1)
        require(ds in spec["counts"], label + " unknown paired domain")
        candidate, anchor = pair.split("_minus_", 1)
        require(candidate in spec["modes"] and anchor in spec["modes"] and candidate != anchor, label + " wrong pair modes")
        by_domain.setdefault(ds, set()).add((candidate, anchor))
    if len(pairs) < spec["pairs"]:
        raise Pending(label + " existing pair matrix is incomplete")
    require(len(pairs) == spec["pairs"] and set(by_domain) == set(spec["counts"])
            and all(value == by_domain["textvqa"] for value in by_domain.values()), label + " incomplete or inconsistent actual pair matrix")
    actual = by_domain["textvqa"]
    candidates = spec["modes"][3:]
    expected = {(mode, ref) for mode in candidates for ref in REFS}
    if label == "primary":
        expected |= {("JointSpatial", "JointV"), ("JointSpatial", "Spatial"), ("Shuffled", "JointSpatial"), ("Shuffled", "Spatial")}
    elif label == "wo":
        expected.add(("JointSpatialWO", "SpatialWO"))
    require(actual == expected, label + " frozen contrast set changed")
    return [[candidate, anchor] for candidate, anchor in sorted(actual)]


@contextmanager
def existing_only_writers(adapter):
    """Replay frozen CPU calculations while replacing both writers by verifiers.

    A missing immutable/JSON target is never repaired, including in readiness.
    Scoped replacement restores the exact original callables on every exit.
    """
    original = adapter.immutable, adapter.write

    def existing_bytes(path, data):
        ensure_files([path])
        require(Path(path).read_bytes() == data, "existing frozen bytes differ: " + str(path))

    def existing_json(path, value):
        ensure_files([path])
        require(canonical(load(path)) == canonical(value), "existing frozen JSON differs: " + str(path))

    adapter.immutable, adapter.write = existing_bytes, existing_json
    try:
        yield
    finally:
        adapter.immutable, adapter.write = original


def failed_setup_support(public):
    root = E / "confirm_preflight_failed_v2"
    path = root / "EXACT_RETRIEVAL_AND_QA0_PROOF.json"
    require(sha(path) == FAILED_SHA, "failed setup proof epoch differs")
    proof = load(path)
    require(proof["status"] == "PASS_PROVEN_FAILED_SETUP_BEFORE_ANY_MODEL_CHILD"
            and len(proof["files"]) == 10 and proof["independent_remote_hashes"] is True, "failed setup proof incomplete")
    files = [{"path": str(path), "sha256": FAILED_SHA, "public_path": "supporting_evidence/confirmation_setup_failed_v2/EXACT_RETRIEVAL_AND_QA0_PROOF.json"}]
    for item in proof["files"]:
        local = root / item["file"]
        require(local.resolve().is_relative_to(root.resolve()) and sha(local) == item["sha256"] and local.stat().st_size == item["bytes"], "failed setup exact source differs")
        public.file_allowed(local, "supporting_scalar_or_source")
        files.append({"path": str(local), "sha256": item["sha256"], "public_path": "supporting_evidence/confirmation_setup_failed_v2/" + item["file"], "source_aliases": [item["remote_file"]]})
    values = (("status", "PASS_PROVEN_FAILED_SETUP_BEFORE_ANY_MODEL_CHILD"), ("actual_model_forward_calls", 0), ("actual_GPU_forward_calls", 0),
              ("actual_new_free_QA", 0), ("independent_remote_hashes", True), ("old_setup_parents_only", True),
              ("readback_reverified_twice", True), ("successful_confirmation_QA_accounted_separately", True))
    return {"evidence_id": "confirmation_setup_failed_v2", "disposition": "CLOSED_WITH_FAILURES", "task_quality_QA": 0,
            "teacher_TF_or_technical_is_free_answer_QA": False, "primary_mean": None,
            "scope": "Two original setup-v2 parents failed before START_CELL/model child; later successful v3 outputs are separate",
            "files": files, "scalar_assertions": [{"path": str(path), "keys": [key], "value": value} for key, value in values]}


def bind_development(public, group, adapter, plan, source_group, summary, pairs, add, fresh_sources, reused_sources):
    """Same frozen selected/score/pair computations; all writes are intercepted."""
    label, spec = group["group_id"], SPECS[group["group_id"]]
    frozen = {(row["dataset"], row["mode"]): row for row in summary["points"]}
    bundles = {}
    add(plan["source_group_manifest_path"], "source_group_manifest", plan["source_group_manifest_sha256"])
    add(source_group["reference_input_manifest_path"], "input_manifest", plan["input_manifest_sha256"])
    for domain in plan["domains"].values():
        for key, role in (("inputs", "source_inputs"), ("refs", "reference_answers"), ("source_lock", "source_lock")):
            add(domain[key], role, domain[key + "_sha256"])
    with existing_only_writers(adapter):
        for entry in plan["entries"]:
            ds, mode = entry["dataset"], entry["mode"]
            selected = adapter.selected_view(entry, plan, source_group, group["harvest_root"])
            require(selected is not None, label + " source producer not closed")
            actual, bundle = adapter.score_view(entry, selected, plan)
            require(canonical(actual) == canonical(frozen[ds, mode]) and bundle is not None, label + " existing frozen score differs")
            bundles[ds, mode] = bundle
            raw, view, lineage = selected["raw"], selected["view"], selected["lineage"]
            proofname = adapter.PROOF_NAME if entry["kind"] == "fresh" else "source_retrieval_receipt_round10_v1.json"
            proof = load(raw / proofname)
            for item in proof["files"]:
                add(raw / item["file"], "actual_producer_source_or_raw", item["sha256"], aliases=[item["remote_file"]])
            for rel, digest in selected["manifest"]["source_hashes"].items():
                add(raw / "source_files" / rel, "executed_runtime_source", digest, rel)
            for name in (proofname, "retrieval_receipt.json"):
                add(raw / name, "independent_actual_source_proof")
            for name in ("selected_records.jsonl", "source_lineage.json", "scored_records.jsonl", "summary.json"):
                add(view / name, "frozen_source_subset_score")
            for rel, digest in actual["scoring_source_pins"].items():
                add(ROOT / rel, "frozen_scoring_dependency", digest)
            identity = tuple(sha(raw / name) for name in ("manifest.json", "records.jsonl", "complete.json"))
            if entry["kind"] == "fresh":
                require(identity not in fresh_sources, "fresh producer counted twice")
                fresh_sources.add(identity)
            else:
                key = (*identity, tuple(lineage["ids"]))
                require(key not in reused_sources or reused_sources[key] == spec["counts"][ds], "reused subset denominator differs")
                reused_sources[key] = spec["counts"][ds]
    for ds in spec["counts"]:
        for candidate, anchor in group["paired_contrasts"]:
            key = ds + "__" + candidate + "_minus_" + anchor
            require(canonical(pairs[key]) == canonical(adapter.paired(bundles[ds, anchor], bundles[ds, candidate])), label + " paired source/runtime/CI replay differs")


def read_table(path):
    if Path(path).suffix.lower() == ".csv":
        import csv
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    data = load(path)
    return data if isinstance(data, list) else data["points"]


def check_table(rows, summaries, combined):
    expected = {(label, row["dataset"], row["mode"]): row for label, summary in summaries.items() for row in summary["points"]}
    keys = [(row["group_id"], row["dataset"], row["mode"]) for row in rows]
    require(len(keys) == len(set(keys)), "Root quality table contains duplicate cohort/dataset/mode rows")
    actual = {key: row for key, row in zip(keys, rows) if key[0] != "combined512"}
    if set(actual) < set(expected):
        raise Pending("Root table actual view matrix is incomplete")
    require(set(actual) == set(expected) and len(expected) == 88, "Root table must contain all88 actual result views without partial/adaptive exclusions")
    for key, row in actual.items():
        point = expected[key]
        require(math.isclose(float(row["primary_mean"]), point["primary_mean"], rel_tol=1e-12, abs_tol=1e-12), "Root table primary differs: " + str(key))
        require(math.isclose(float(row["mean_complete_visual_owner_bytes"]), point["resources"]["complete_bytes"]["mean"], rel_tol=1e-12, abs_tol=1e-6), "Root table complete owner bytes differ: " + str(key))
        n = row.get("expected_QA", row.get("selected_QA_or_views"))
        require(n is not None and int(n) == point["expected_QA"], "Root table count differs: " + str(key))
    derived = {key: row for key, row in zip(keys, rows) if key[0] == "combined512"}
    if derived:
        points = {("combined512", p["dataset"], p["mode"]): p for p in combined["points"]}
        require(set(derived) == set(points), "combined table rows must remain a complete separate derived-only group")
        for key, row in derived.items():
            require(math.isclose(float(row["primary_mean"]), points[key]["primary_mean"], rel_tol=1e-12, abs_tol=1e-12)
                    and int(row["new_inference_events"]) == 0, "derived table cannot add inference or alter quality")
    return {"actual_view_rows": 88, "optional_combined_derived_rows": len(derived), "combined_rows_add_inference": 0}


def combined_body(path):
    body = load(path)
    return body.get("original128_additional384_combined512_separate_report", body)


def check_combined_header(body):
    if body.get("status") in ("WAIT_BOTH_SOURCE_COHORTS", "NOT_IN_THIS_CUTOFF", "UNAVAILABLE_WITH_FAILED_OR_UNRUN_SOURCE_CELLS"):
        raise Pending("combined512 actual source cohorts are incomplete")
    require(body.get("status") == "PASS_DERIVED_ORIGINAL128_PLUS_ADDITIONAL384_COMBINED512" and body.get("combined_quality_computed") is True
            and body.get("original_sources") == 128 and body.get("additional_sources") == 384 and body.get("combined_sources") == 512
            and body.get("original_fresh_QA") == 640 and body.get("additional_fresh_QA") == 1920 and body.get("new_inference_events") == 0
            and body.get("additional384_primary_new_replication") is True and body.get("combined512_preregistered_before_original128_claimed") is False
            and body.get("interim_original128_observation_disclosed") is True and body.get("no_p_stop_or_method_reselection") is True,
            "combined report must be the separate zero-inference original128+additional384 derived view with interim observation disclosed")
    if len(body.get("points", [])) < 20 or len(body.get("pairs", {})) < 24:
        raise Pending("combined report is incomplete")
    require(len(body["points"]) == 20 and len(body["pairs"]) == 24, "combined report has excess points or pairs")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-final", type=Path, default=E / "summary_final_primary_v4")
    parser.add_argument("--wo-final", type=Path, default=E / "wo_summary_final_v1")
    parser.add_argument("--original128-final", type=Path, default=E / "confirmation_summary_final_v3")
    parser.add_argument("--add384-final", type=Path, default=E / "expand_summary_final_v1")
    parser.add_argument("--primary-harvest", type=Path, default=E / "harvest_v4")
    parser.add_argument("--wo-harvest", type=Path, default=E / "wo_harvest_v1")
    parser.add_argument("--original128-harvest", type=Path, default=E / "confirmation_harvest_v3")
    parser.add_argument("--add384-harvest", type=Path, default=E / "expand_harvest_v1")
    parser.add_argument("--combined512-derived", type=Path, help="actual prepackaging output of round11_combine_holdouts_v1.py; a final COLD result is not required")
    parser.add_argument("--final-report", type=Path)
    parser.add_argument("--quality-table", type=Path)
    parser.add_argument("--extra-files-manifest", type=Path, help="optional explicit list of path/sha256/role/public_path descriptors; no directory scans")
    parser.add_argument("--output", type=Path, default=E / "final_contract_candidate_v1")
    parser.add_argument("--prepare", action="store_true")
    return parser.parse_args(argv)


def assemble(args):
    finals = dict(primary=args.primary_final, wo=args.wo_final, confirm128=args.original128_final, expand384=args.add384_final)
    harvests = dict(primary=args.primary_harvest, wo=args.wo_harvest, confirm128=args.original128_harvest, expand384=args.add384_harvest)
    require(args.output.resolve().is_relative_to(E.resolve()) and args.output.resolve() != E.resolve(), "new candidate output must be a child of E Round11")
    require(not args.output.exists(), "candidate output already exists; use a new immutable name")
    require(all(path.resolve().is_relative_to(E.resolve()) for path in (*finals.values(), *harvests.values())), "evidence roots must remain on E Round11")
    ensure_files([*[path / name for path in finals.values() for name in ("summary.json", "paired_comparisons.json")],
                  args.combined512_derived, args.final_report, args.quality_table])
    summaries = {label: load(path / "summary.json") for label, path in finals.items()}
    for label, summary in summaries.items():
        validate_header(label, summary)
    # Every required existing cell/score/proof is checked before importing a
    # replay helper. No replay can repair a missing artifact.
    for label, spec in SPECS.items():
        planpath = BOOT / spec["plan"]
        ensure_files([planpath, ROOT / "ops" / (spec["adapter"] + ".py")])
        require(sha(planpath) == spec["plan_sha"], label + " frozen plan epoch changed")
        if spec["summary_sha"]:
            require(sha(finals[label] / "summary.json") == spec["summary_sha"], label + " actual final summary epoch differs")
        plan = load(planpath)
        for entry in plan["entries"]:
            if label in ("primary", "wo"):
                view = harvests[label] / "views" / (entry["dataset"] + "__" + entry["mode"])
                ensure_files([view / name for name in ("summary.json", "source_lineage.json", "selected_records.jsonl", "scored_records.jsonl")])
            else:
                score = harvests[label] / "scored" / entry["item"]["cell"]["name"]
                ensure_files([score / name for name in ("summary.json", "scored_records.jsonl")])
            raw = harvests[label] / "raw" / entry["item"]["cell"]["name"] if entry["kind"] == "fresh" else Path(entry["source_raw_dir"])
            proof = ("source_retrieval_receipt_round11_confirmation_v2.json" if label == "confirm128" else "source_retrieval_receipt_round11_expand384_v1.json" if label == "expand384" else "source_retrieval_receipt_round11_wo_v1.json" if label == "wo" else "source_retrieval_receipt_round11_v1.json") if entry["kind"] == "fresh" else "source_retrieval_receipt_round10_v1.json"
            ensure_files([raw / name for name in ("manifest.json", "records.jsonl", "complete.json", proof)])
    lockpath = E / "public_v5_prepare_v1/FROZEN_FOR_ROOT_LOCAL_QA_v1.json"
    ensure_files([lockpath])
    require(sha(lockpath) == PUB_LOCK_SHA, "Root PUB5 source freeze differs")
    lock = load(lockpath)
    for name, digest in lock["source_sha256"].items():
        require(sha(ROOT / "ops" / name) == digest, "frozen PUB5 source changed: " + name)
    sys.path[:0] = [str(ROOT / "ops"), str(ROOT / "code")]
    public = importlib.import_module("round11_public_contract_v5")
    confirmation = importlib.import_module("round11_public_confirmation_v5")
    expansion = importlib.import_module("round11_public_expansion_v5")
    candidate = public.candidate()
    groups, contexts = [], {}
    for label, spec in SPECS.items():
        path = ROOT / "ops" / (spec["adapter"] + ".py")
        require(sha(path) == spec["adapter_sha"], label + " actual adapter epoch differs")
        adapter = public.adapter_module({"path": str(path), "sha256": spec["adapter_sha"]})
        planpath = BOOT / spec["plan"]
        plan, auxiliary = adapter.load_plan(planpath)
        pairs = load(finals[label] / "paired_comparisons.json")
        pairset = contrasts(label, pairs)
        require(len(plan["entries"]) == spec["views"] and {(e["dataset"], e["mode"]) for e in plan["entries"]} == {(p["dataset"], p["mode"]) for p in summaries[label]["points"]}, label + " plan/summary matrix differs")
        group = {"group_id": label, "kind": spec["kind"], "execution_disposition": "EXECUTED", "cohort_label": spec["cohort"],
                 "plan": {"path": str(planpath), "sha256": spec["plan_sha"]}, "adapter": {"path": str(path), "sha256": spec["adapter_sha"]},
                 "counts": spec["counts"], "modes": list(spec["modes"]), "harvest_root": str(harvests[label]), "summary_root": str(finals[label]),
                 "cells": [{"dataset": e["dataset"], "mode": e["mode"], "expected_QA": spec["counts"][e["dataset"]], "kind": e["kind"], "disposition": "CLOSED_COMPLETE"} for e in plan["entries"]],
                 "paired_contrasts": pairset}
        if label in ("confirm128", "expand384"):
            group["paired_contrasts"] = confirmation.contrasts(list(SELECTED))
            require({tuple(pair) for pair in group["paired_contrasts"]} == {tuple(pair) for pair in pairset}, "confirmation contrasts differ")
            group["selection_basis"] = {role: {"group_id": role, "path": str(finals[role] / "summary.json"), "sha256": sha(finals[role] / "summary.json")} for role in ("primary", "wo")}
            data_root = E / ("confirmation_inputs_v1" if label == "confirm128" else "expand384_inputs_v1") / "remote_materialization_v1"
            ensure_files([data_root / "LOCAL_FETCH_RECORDS_v2.json", data_root / "INDEPENDENT_REMOTE_SHA_v1.json"])
            records = load(data_root / "LOCAL_FETCH_RECORDS_v2.json")
            require(len(records) == (41 if label == "confirm128" else 50), label + " actual data readback closure differs")
            group["confirmation_data" if label == "confirm128" else "expansion_data"] = {
                "fetch_inventory": {"path": str(data_root / "LOCAL_FETCH_RECORDS_v2.json"), "sha256": sha(data_root / "LOCAL_FETCH_RECORDS_v2.json")},
                "remote_sha_receipt": {"path": str(data_root / "INDEPENDENT_REMOTE_SHA_v1.json"), "sha256": sha(data_root / "INDEPENDENT_REMOTE_SHA_v1.json")}}
        groups.append(group)
        contexts[label] = adapter, plan, auxiliary, pairs
    # Memory inventory exercises the exact PUB5 safety/collision/alias rules.
    inventory, fresh_sources, reused_sources = {}, set(), {}
    add = lambda *a, **kw: public.add_file(inventory, *a, **kw)
    results = []
    for group in groups:
        label = group["group_id"]
        adapter, plan, auxiliary, pairs = contexts[label]
        add(group["adapter"]["path"], "executed_harvest_source", group["adapter"]["sha256"])
        add(group["plan"]["path"], "actual_execution_plan", group["plan"]["sha256"])
        for name in ("summary.json", "paired_comparisons.json"):
            add(finals[label] / name, "frozen_group_summary")
        if label in ("primary", "wo"):
            bind_development(public, group, adapter, plan, auxiliary, summaries[label], pairs, add, fresh_sources, reused_sources)
        else:
            helper = confirmation if label == "confirm128" else expansion
            with existing_only_writers(adapter):
                result = helper.collect_group(group, adapter, plan, auxiliary, add, True, fresh_sources, ROOT, groups)
            require(result["actual_fresh_answer_events"] == result["closed_quality_answer_events"] == SPECS[label]["fresh"], label + " actual source tally differs")
            results.append(result)
    require(len(fresh_sources) == 64 and len(reused_sources) == 12 and sum(reused_sources.values()) == 288, "unique fresh producers/old reference subsets differ")
    combined = combined_body(args.combined512_derived)
    check_combined_header(combined)
    physical = importlib.import_module("verify_public_round10_v3")
    scorer = importlib.import_module("score")

    class LocalSource:
        path = staticmethod(Path)
        doc = staticmethod(load)

    actual_combined = expansion.combined_512_views(LocalSource(), {"groups": results}, scorer, physical)
    require(canonical(actual_combined) == canonical(combined), "combined512 does not exactly match the complete original/additional source-derived union")
    table_check = check_table(read_table(args.quality_table), summaries, combined)
    supportpath = BOOT / "SUPPORTING_EVIDENCE_CANDIDATE_v1.json"
    require(sha(supportpath) == SUPPORT_SHA, "original eight support groups changed")
    support = deepcopy(load(supportpath)["supporting_evidence"])
    require(len(support) == 8 and sum(len(row["files"]) for row in support) == 195, "original support inventory differs")
    support.append(failed_setup_support(public))
    for evidence in support:
        require(evidence.get("task_quality_QA") == 0 and evidence.get("teacher_TF_or_technical_is_free_answer_QA") is False and evidence.get("primary_mean") is None, "supporting QA0 cannot become task quality")
        for item in evidence["files"]:
            add(item["path"], "supporting_scalar_or_source", item["sha256"], item.get("public_path"), item.get("source_aliases", []))
        for assertion in evidence.get("scalar_assertions", []):
            require(any(Path(item["path"]).resolve() == Path(assertion["path"]).resolve() for item in evidence["files"]), "support scalar must bind an explicit file")
            value = load(assertion["path"])
            for key in assertion["keys"]:
                value = value[key]
            require(value == assertion["value"], "support scalar assertion differs")
    files = [
        {"path": str(args.final_report), "sha256": sha(args.final_report), "role": "scope_report", "public_path": "docs/ROUND11_FINAL_REPORT" + args.final_report.suffix.lower()},
        {"path": str(args.quality_table), "sha256": sha(args.quality_table), "role": "quality_bytes_table", "public_path": "results/round11/final/QUALITY_BYTES" + args.quality_table.suffix.lower()},
        {"path": str(args.combined512_derived), "sha256": sha(args.combined512_derived), "role": "separate_combined512_derived_report", "public_path": "results/round11/final/COMBINED512_SOURCE_DERIVED.json"},
        {"path": str(supportpath), "sha256": SUPPORT_SHA, "role": "supporting_candidate_inventory", "public_path": "evidence/ROUND11_SUPPORTING_EVIDENCE_CANDIDATE_v1.json"},
        {"path": str(lockpath), "sha256": PUB_LOCK_SHA, "role": "frozen_public5_source_lock", "public_path": "evidence/ROUND11_PUBLIC5_SOURCE_LOCK.json"},
    ]
    for rel in EXPLICIT_STAGE_FILES:
        path = ROOT / rel
        ensure_files([path])
        files.append({"path": str(path), "sha256": sha(path), "role": "explicit_stage_source_or_plan", "public_path": rel})
    for name in ("EXPANSION_DESIGN_v1.json",):
        path = BOOT / name
        files.append({"path": str(path), "sha256": DESIGN_SHA, "role": "fixed_expansion_design", "public_path": "evidence/" + name})
    for rel in ("expand384_inputs_v1/remote_materialization_v1/INDEPENDENT_LOCAL_AUDIT_v1.json", "bootstrap_v1/WO_QUERY_AGNOSTIC_BOUND_MATH_REVIEW_v1.json"):
        path = E / rel
        if path.is_file():
            files.append({"path": str(path), "sha256": sha(path), "role": "supporting_independent_review", "public_path": "results/round11/" + rel})
    if args.extra_files_manifest is not None:
        ensure_files([args.extra_files_manifest])
        extra = load(args.extra_files_manifest)
        require(isinstance(extra, list) and all(set(item) <= {"path", "sha256", "role", "public_path", "source_aliases"} and {"path", "sha256", "role", "public_path"} <= set(item) for item in extra), "extra files require explicit safe descriptors; directory scans are forbidden")
        files.extend(extra)
    for item in files:
        add(item["path"], item["role"], item["sha256"], item.get("public_path"), item.get("source_aliases", []))
    require({"scope_report", "quality_bytes_table"} <= {item["role"] for item in files}, "report/table required")
    accounting = {"view_cells": 88, "actual_fresh_answer_events": 3136, "actual_view_answer_events": 3712,
                  "reused_reference_view_events": 576, "unique_reused_reference_view_events": 288, "repeated_reused_reference_view_events": 288,
                  "unique_reused_reference_producer_views": 12, "fresh_producer_cells": 64, "paired_comparisons": 140,
                  "combined512_new_inference_events": 0}
    require(sum(spec["fresh"] for spec in SPECS.values()) == accounting["actual_fresh_answer_events"] and sum(spec["pairs"] for spec in SPECS.values()) == 140, "final scope arithmetic differs")
    candidate.update(phase_label=PHASE_LABEL, stage_state="PENDING_ROOT_SEAL", root_seal=None, groups=groups, supporting_evidence=support, files=files,
                     final_four_cohort_unsealed_candidate=True, final_research_stage_completion_claimed=False, publication_authorized_by_this_candidate=False,
                     expected_final_accounting=accounting, combined512_derived_only={"path": str(args.combined512_derived), "sha256": sha(args.combined512_derived),
                     "new_inference_events": 0, "original_additional_combined_reported_separately": True, "combined512_preregistered_before_original128_claimed": False})
    ready = public.readiness(candidate)
    require(ready["missing"] == ["Root actual final cutoff seal", "Root seal path/SHA"] and ready["publication_eligible"] is False, "candidate must remain unsealed")
    receipt = {"schema": "trajkv.round11.final_contract_assembly.v1", "status": "READY_UNSEALED_FOUR_ACTUAL_COHORTS", "accounting": accounting,
               "source_summary_sha256": {label: sha(path / "summary.json") for label, path in finals.items()},
               "source_pair_sha256": {label: sha(path / "paired_comparisons.json") for label, path in finals.items()},
               "source_plans": {label: {"path": str(BOOT / spec["plan"]), "sha256": spec["plan_sha"]} for label, spec in SPECS.items()},
               "original_selection_sha256": SELECTION_SHA, "expansion_design_sha256": DESIGN_SHA,
               "combined512_source_sha256": sha(args.combined512_derived), "combined512_exact_source_recomputation": True,
               "quality_table_check": table_check, "supporting_groups": 9, "supporting_file_descriptors": sum(len(item["files"]) for item in support),
               "memory_inventory_files": len(inventory), "source_files_verified_before_candidate_write": True,
               "model_GPU_network_Git_seal_export_calls": 0, "new_or_repaired_scores": 0, "publication_eligible": False,
               "Root_readiness": ready, "assembler_source_sha256": sha(__file__), "created_utc": datetime.now(timezone.utc).isoformat()}
    return candidate, receipt, inventory


def prepare_new(output, candidate, receipt, inventory):
    """Final commit: only called after every complete-source gate has passed."""
    require(not output.exists(), "candidate output exists; zero writes required")
    # Detect concurrent source mutations between admission and the first write.
    for item in inventory.values():
        stat = item["path"].stat()
        require((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == item["stat"], "source changed after admission; re-run readiness")
    output.mkdir(parents=True, exist_ok=False)
    body = (json.dumps(candidate, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    with (output / "CONTRACT_CANDIDATE.json").open("xb") as stream:
        stream.write(body)
    receipt = {**receipt, "status": "PASS_UNSEALED_FOUR_COHORT_CONTRACT_PREPARED", "contract_sha256": hashlib.sha256(body).hexdigest(), "Root_seal_created": False,
               "stage_or_archive_exported": False, "Git_or_GitHub_written": False}
    with (output / "PREPARATION_RECEIPT.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main(argv=None):
    args = parse_args(argv)
    output_existed = args.output.exists()
    try:
        candidate, receipt, inventory = assemble(args)
        if args.prepare:
            receipt = prepare_new(args.output, candidate, receipt, inventory)
        print(json.dumps({**receipt, "candidate_files_written": 2 if args.prepare else 0}, ensure_ascii=False, allow_nan=False))
        return 0
    except Pending as exc:
        print(json.dumps({"schema": "trajkv.round11.final_contract_readiness.v1", "status": "PENDING", "reason": str(exc), "candidate_files_written": 0,
                          "publication_eligible": False, "model_GPU_network_Git_seal_export_calls": 0}, ensure_ascii=False))
        return 3
    except (ValueError, KeyError, TypeError, OSError) as exc:
        written = 0 if output_existed else sum((args.output / name).is_file() for name in ("CONTRACT_CANDIDATE.json", "PREPARATION_RECEIPT.json"))
        print(json.dumps({"schema": "trajkv.round11.final_contract_readiness.v1", "status": "FAIL_CLOSED", "reason": str(exc), "candidate_files_written": written,
                          "publication_eligible": False, "model_GPU_network_Git_seal_export_calls": 0}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
