"""Prepare bounded WO descriptors only after completed D96 and Root decision.

This file is not an executor. It performs no SSH, model load, GPU call, queue
launch, Git operation or holdout-reference read. Root invokes it only after
the primary28/384-new/288-reused summary and decision are frozen. No prepare()
call occurs on import; both CLI admission arguments are mandatory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "ops"), str(ROOT / "code")]
import round11_jobs_v1 as jobs
import round11_wo_runner_v1 as runner
import round11_wo_technical_v1 as technical
import round11_wo_harvest_v1 as harvest

MODES = ("SpatialWO", "JointSpatialWO")
DATASET_ORDER = ("textvqa", "infovqa", "docvqa", "chartqa")
COUNTS = dict(textvqa=32, infovqa=32, docvqa=16, chartqa=16)
PRIMARY_MODES = ("FP16", "R7low", "R7high", "JointV", "Spatial", "JointSpatial", "Shuffled")
SOURCE_GROUP_SHA = "663076f21fc7df7d00253d06604e9f46da06ffe9d62532cd66d82794c05fa9b4"
INPUT_MANIFEST_SHA = "b2ba5f54516abf96344e02356c7cee079b4ff48e2a270ea8fc61ea1c78baaecf"
FACTORY_SHA = "b0f62f1c8ddbad41336b161891cf0943876a0087e9dcdec63e9bc6e312aaf69e"
SEQUENCE_SHA = "2d0dce9da80156fb0224a04d48a9a3e3869ede5e9885885a6a25073ab511fe00"
REFERENCE_PLAN = jobs.E / "driver_cpu_v6/reference_only_plan.json"
BOOTSTRAP = jobs.E / "bootstrap_v1"
SOURCE_LOCK = BOOTSTRAP / "WO_SOURCE_LOCK_v1.json"
SOURCE_LOCK_SHA = "980da29bfdca806541d998115cb4f30148c8986c306b70db327f68ba54335570"


def require(value, message):
    if not value:
        raise ValueError("ROUND11_WO_PREPARATION: " + message)


def json_bytes(value):
    # Exactly the existing Root helper serialization; hashes can be computed
    # before any output is materialized and independently checked at freezing.
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def object_sha(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def admission(primary_summary_path, decision_path):
    primary_summary_path, decision_path = Path(primary_summary_path), Path(decision_path)
    primary, decision = jobs.read(primary_summary_path), jobs.read(decision_path)
    require(primary.get("schema") == "trajkv.round11.harvest_summary.v1"
        and primary.get("status") == "PASS_ALL_28_PROVEN_VIEWS"
        and primary.get("expected_cells_or_views") == primary.get("closed_cells_or_views") == primary.get("successful_complete_cells_or_views") == 28,
        "complete successful actual primary28 summary required; partial/failed primary cannot generate WO configs")
    require(primary.get("expected_new_free_generation_QA") == primary.get("actual_closed_new_free_generation_QA") == 384
        and primary.get("expected_reused_reference_views") == primary.get("actual_closed_reused_reference_views") == 288
        and primary.get("source_group_manifest_sha256") == SOURCE_GROUP_SHA,
        "primary must close384 actual new QA and288 reused views on exact D96")
    primary_plan = BOOTSTRAP / "HARVEST_PLAN_v2.json"
    require(primary.get("plan_sha256") == jobs.sha(primary_plan), "primary summary does not bind the approved actual primary plan")
    points = primary.get("points", [])
    require(len(points) == 28 and {(point["dataset"], point["mode"]) for point in points}
        == {(dataset, mode) for dataset in COUNTS for mode in PRIMARY_MODES}, "primary28 mode/domain coverage differs")
    for point in points:
        count = COUNTS[point["dataset"]]
        is_reference = point["mode"] in PRIMARY_MODES[:3]
        require(point.get("status") == "PASS_FULL_SOURCE_GROUP_SCORE" and point.get("expected_QA") == count
            and point.get("raw_failure_questions") == 0
            and point.get("fresh_free_generation_QA") == (0 if is_reference else count)
            and point.get("reused_reference_views") == (count if is_reference else 0),
            "every primary view must be complete and successful with exact fresh/reused accounting")
    for role, digest in primary.get("pins", {}).items():
        path = PurePosixPath(role)
        require(not path.is_absolute() and ".." not in path.parts
            and re.fullmatch("[0-9a-f]{64}", digest) is not None and jobs.sha(ROOT / role) == digest,
            "actual primary source pin changed: " + role)
    require(bool(primary.get("pins")), "actual primary producer source pins required")
    require(decision.get("schema") == "trajkv.round11.wo_development_decision.v1"
        and decision.get("modes") == list(MODES)
        and decision.get("primary_summary_sha256") == jobs.sha(primary_summary_path)
        and decision.get("R128_outcomes_read") is False,
        "explicit Root decision for exactly SpatialWO/JointSpatialWO and unread R128 outcomes required")
    require(not (set(decision) & runner.frozen.FORBIDDEN), "decision cannot supply reference answers or inference scores")
    return {"primary_summary_path": str(primary_summary_path), "primary_summary_sha256": jobs.sha(primary_summary_path),
        "primary_status": primary["status"], "decision_path": str(decision_path), "decision_sha256": jobs.sha(decision_path),
        "decision_schema": decision["schema"], "modes": list(MODES), "R128_outcomes_read": False,
        "primary_new_QA": 384, "primary_reused_views": 288}


def source_graphs():
    require(tuple(runner.MODES) == MODES and runner.COUNTS == COUNTS, "actual WO source modes/counts differ")
    require(jobs.sha(ROOT / "code/round11_wo_value_v1.py") == FACTORY_SHA, "CPU-verified frozen WO factory changed")
    require(jobs.sha(ROOT / "code/round10_sequence_v1.py") == SEQUENCE_SHA, "frozen parent source changed")
    graph, technical_graph = runner.runtime_source_manifest(), technical.runtime_source_manifest()
    require(len(graph) == 20 and len(technical_graph) == 24, "complete actual WO20/technical24 source graphs required")
    empty = hashlib.sha256(b"").hexdigest()
    for current in (graph, technical_graph):
        require(all(re.fullmatch("[0-9a-f]{64}", digest) and digest != empty
            and (ROOT / role).stat().st_size > 0 and jobs.sha(ROOT / role) == digest
            for role, digest in current.items()), "actual source graph has a missing, empty or changed source")
    require(jobs.sha(SOURCE_LOCK) == SOURCE_LOCK_SHA, "Root frozen-before-GPU WO source lock changed")
    lock = jobs.read(SOURCE_LOCK)
    require(lock.get("schema") == "trajkv.round11.wo_source_lock.v1" and lock.get("state") == "FROZEN_BEFORE_GPU"
        and lock.get("primary_runtime") == graph and lock.get("technical_runtime") == technical_graph,
        "actual WO20/technical24 graphs differ from Root reviewed/frozen source lock")
    require(len(lock["sources"]) == 5 and all(jobs.sha(ROOT / role) == digest for role, digest in lock["sources"].items()),
        "one of five independently reviewed/frozen WO sources changed")
    for key in ("cpu_acceptance", "protocol_review"):
        require(jobs.sha(lock[key]["path"]) == lock[key]["sha256"], "Root frozen acceptance/review evidence changed")
    harvest.check_pins()
    return graph, technical_graph, {"path": str(SOURCE_LOCK), "sha256": SOURCE_LOCK_SHA,
        "math_addendum_sha256": lock["addendum_sha256"], "cpu_acceptance_sha256": lock["cpu_acceptance"]["sha256"],
        "protocol_review_sha256": lock["protocol_review"]["sha256"]}


@contextmanager
def structural_driver_bindings():
    names = ("SCHEMA", "MODES", "RUNTIME_FILES", "candidate_assets", "validate_config", "build_candidate_owners",
             "round10_budget_guard", "runtime_source_manifest", "write_new")
    saved = {name: getattr(runner.frozen, name) for name in names}
    try:
        runner.install_driver_overrides()
        yield
    finally:
        for name, value in saved.items():
            setattr(runner.frozen, name, value)


def unchanged_template(original, config):
    allowed = {"schema", "count", "method", "scope", "candidate", "expected_source_hashes",
               "input_manifest_sha256", "source_group_manifest_path", "source_group_manifest_sha256"}
    changed = {key for key in set(original) | set(config) if original.get(key) != config.get(key)}
    require(changed <= allowed, "unexpected model/prompt/generation/data/codec template change: " + repr(changed - allowed))
    require(config["variant"] == original["variant"] and config["variant_source"] == original["variant_source"]
        and config["expected_asset_sha256"] == original["expected_asset_sha256"], "original R7low variant/assets changed")
    require(config["candidate"] in ({"mode": mode} for mode in MODES), "no fitted/mixed/technical candidate parameters")
    require(not (set(config) & runner.frozen.FORBIDDEN), "references entered inference config")
    shadow = deepcopy(config)
    shadow["method"]["family"] = "trajkv_round10_research"
    with structural_driver_bindings():
        runner._validate_base(shadow)
    return sorted(changed)


def planned_job(name, script, args, gpu):
    value = {"name": name, "env": jobs.env(gpu), "argv": [jobs.MAIN_PYTHON, "-B", jobs.REMOTE_ROOT + "/" + script, *args]}
    return {"name": name, "path": str(ROOT / "configs/jobs" / (name + ".json")),
            "sha256": object_sha(value), "job": value}


def prepare(primary_summary_path, decision_path):
    # Admission and all prospective configurations are checked before the first
    # new candidate/queue/job descriptor is written. This function never starts
    # any job, including the separately prepared technical gate.
    accepted = admission(primary_summary_path, decision_path)
    graph, technical_graph, source_lock = source_graphs()
    group_path = ROOT / "configs/round11_v1/SOURCE_GROUP_MANIFEST.json"
    require(jobs.sha(group_path) == SOURCE_GROUP_SHA, "SourceGroupD96 changed")
    group = jobs.read(group_path)
    require(group["reference_input_manifest_sha256"] == INPUT_MANIFEST_SHA
        and sum(value["count"] for value in group["datasets"].values()) == 96, "old D96 source lineage differs")
    references = jobs.read(REFERENCE_PLAN)
    require(references["schema"] == "trajkv.round11.harvest_plan.v1" and len(references["entries"]) == 12
        and all(entry["kind"] == "reused" for entry in references["entries"])
        and references["source_group_manifest_sha256"] == SOURCE_GROUP_SHA
        and sum(COUNTS[entry["dataset"]] for entry in references["entries"]) == 288,
        "exact frozen twelve old reference views288 required")
    artifacts, queues, new_entries, config_receipts = [], [], [], []
    source_group_remote = jobs.REMOTE_ROOT + "/configs/round11_v1/SOURCE_GROUP_MANIFEST.json"
    for gpu, mode in enumerate(MODES):
        cells = []
        for dataset in DATASET_ORDER:
            detail = group["datasets"][dataset]
            count, original_count = COUNTS[dataset], detail["input_count"]
            template_path = ROOT / f"configs/round10_v1/R7low_{dataset}_{original_count}_v1.json"
            original = jobs.read(template_path)
            require(detail["count"] == count and detail["start"] == original["start"] == 0
                and original["inputs"] == detail["input_path"] and original["inputs_sha256"] == detail["input_sha256"]
                and original["input_count"] == original_count, "original D96 template/source slice differs")
            name = f"WO_{mode}_{dataset}_v1"
            config = deepcopy(original)
            config.update(schema=runner.SCHEMA, count=count,
                method={"id": "Round11_" + mode + "_D96_v1", "family": "trajkv_round11_wo_research",
                        "encode_scope": "visual", "information_condition": "query_unknown"},
                candidate={"mode": mode}, expected_source_hashes=graph,
                scope="Round11 bounded W_O followup on observed SourceGroupD96; requires complete primary and Root decision; not R128 confirmation",
                input_manifest_sha256=INPUT_MANIFEST_SHA, source_group_manifest_path=source_group_remote,
                source_group_manifest_sha256=SOURCE_GROUP_SHA)
            changed = unchanged_template(original, config)
            relative = f"configs/round11_v1/{name}.json"
            digest = object_sha(config)
            artifacts.append((ROOT / relative, config, digest))
            cell = {"name": name, "dataset": dataset, "mode": mode, "script": "code/round11_wo_runner_v1.py",
                "script_sha256": graph["code/round11_wo_runner_v1.py"], "config": jobs.REMOTE_ROOT + "/" + relative,
                "config_sha256": digest, "output": jobs.REMOTE_ROOT + "/results/round11_v1/wo_dev/" + name,
                "expected_QA": count}
            cells.append(cell)
            config_receipts.append({"name": name, "local_path": str(ROOT / relative), "sha256": digest,
                "template_path": str(template_path), "template_sha256": jobs.sha(template_path),
                "explicit_changed_fields": changed, "all_other_model_prompt_generation_input_and_codec_fields_unchanged": True})
        require([cell["dataset"] for cell in cells] == list(DATASET_ORDER)
            and len(cells) == 4 and sum(cell["expected_QA"] for cell in cells) == 96, "one WO GPU must be4 cells96QA in Text/Info/Doc/Chart order")
        relative = f"configs/round10_v1/R11_wo_gpu{gpu}_v1.json"
        sequence = {"schema": "trajkv.round10.sequence.v1", "scope": "Round11 bounded W_O D96 followup; Root-approved after complete primary",
                    "expected_QA": 96, "entries": cells}
        digest = object_sha(sequence); artifacts.append((ROOT / relative, sequence, digest))
        parent_output = jobs.REMOTE_ROOT + f"/results/round11_v1/queues/wo_gpu{gpu}_v1"
        args = ["--config", jobs.REMOTE_ROOT + "/" + relative, "--output", parent_output]
        job = planned_job(f"bm_R11_wo_gpu{gpu}_v1", "code/round10_sequence_v1.py", args, gpu)
        queues.append({"GPU": gpu, "mode": mode, "expected_new_QA": 96, "sequence_path": str(ROOT / relative),
            "sequence_remote": jobs.REMOTE_ROOT + "/" + relative, "sequence_sha256": digest,
            "parent_output": parent_output, "job": job, "cells": cells})
        for cell in cells:
            new_entries.append({"kind": "fresh", "dataset": cell["dataset"], "mode": mode,
                "item": {"root": jobs.REMOTE_ROOT, "job": job["job"], "job_sha256": job["sha256"], "cell": cell,
                    "sequence_sha256": digest, "sequence_script_sha256": SEQUENCE_SHA,
                    "parent_script": "code/round10_sequence_v1.py", "sequence_schema": "trajkv.round10.sequence.v1",
                    "input_manifest_sha256": INPUT_MANIFEST_SHA, "source_group_manifest_sha256": SOURCE_GROUP_SHA}})
    require(len(config_receipts) == 8 and len(new_entries) == 8
        and sum(entry["item"]["cell"]["expected_QA"] for entry in new_entries) == 192, "exact8 WO cells192new QA required")
    harvest_plan = deepcopy(references)
    harvest_plan.update(branch="round11_bounded_WO_v1", entries=[*references["entries"], *new_entries],
        primary_new_free_generation_QA=192, reused_reference_views=288, total_scored_views=480,
        reference_only_plan_sha256=jobs.sha(REFERENCE_PLAN), inference_references_read=False,
        harvest_source_path="ops/round11_wo_harvest_v1.py", harvest_source_sha256=jobs.sha(ROOT / "ops/round11_wo_harvest_v1.py"),
        source_runtime_hashes=graph, development_admission=accepted,
        wo_source_lock=source_lock,
        old_288_reference_views_repeat_primary_group=True, R128_outcomes_read=False)
    technical_template_path = ROOT / "configs/round11_v1/technical_v2.json"
    old_technical = jobs.read(technical_template_path)
    technical_config = deepcopy(old_technical)
    technical_config.update(schema=technical.SCHEMA, expected_source_hashes=technical_graph,
        scope="Round11 bounded W_O actual original1+noop2+encoded2 teacher-forced controls, native teacher cap4; QA0, not task quality")
    require({key for key in set(old_technical) | set(technical_config) if old_technical.get(key) != technical_config.get(key)}
        <= {"schema", "expected_source_hashes", "scope"}, "WO technical config changed original model/data/math settings")
    technical_relative = "configs/round11_v1/wo_technical_v1.json"
    technical_digest = object_sha(technical_config)
    artifacts.append((ROOT / technical_relative, technical_config, technical_digest))
    technical_args = ["--config", jobs.REMOTE_ROOT + "/" + technical_relative,
        "--output", jobs.REMOTE_ROOT + "/results/round11_v1/wo_model_technical_v1"]
    technical_job = planned_job("bm_R11_wo_technical_gpu0_v1", "code/round11_wo_technical_v1.py", technical_args, 0)
    source_pins = {"code/round11_wo_runner_v1.py": graph["code/round11_wo_runner_v1.py"],
        "code/round11_wo_value_v1.py": FACTORY_SHA,
        "code/round11_wo_technical_v1.py": technical_graph["code/round11_wo_technical_v1.py"],
        "ops/round11_wo_harvest_v1.py": jobs.sha(ROOT / "ops/round11_wo_harvest_v1.py"),
        "ops/round11_prepare_wo_v1.py": jobs.sha(__file__)}
    # All scope, lineage, source and admission checks above have passed. The
    # following writes only freeze local descriptors; Root stages/launches.
    for path, value, digest in artifacts:
        require(jobs.freeze(path, value) == digest, "Root serialization/hash differs from preflight")
    for queue in queues:
        job = queue["job"]
        args = ["--config", queue["sequence_remote"], "--output", queue["parent_output"]]
        require(jobs.job(job["name"], "code/round10_sequence_v1.py", args, queue["GPU"]) == job, "Root job helper differs from planned queue")
    require(jobs.job(technical_job["name"], "code/round11_wo_technical_v1.py", technical_args, 0) == technical_job, "Root technical job helper differs")
    harvest_path = BOOTSTRAP / "WO_HARVEST_PLAN_v1.json"
    harvest_sha = jobs.freeze(harvest_path, harvest_plan)
    validated, _ = harvest.load_plan(harvest_path)
    require(len(validated["entries"]) == 20 and len([entry for entry in validated["entries"] if entry["kind"] == "fresh"]) == 8,
        "WO harvest must bind12 reused plus8 actual-source fresh views")
    technical_plan = {"schema": "trajkv.round11.wo_technical_job_plan.v1", "job": technical_job,
        "config_path": str(ROOT / technical_relative), "config_sha256": technical_digest,
        "technical_runtime_source_hashes": technical_graph, "original_technical_template_sha256": jobs.sha(technical_template_path),
        "development_admission": accepted, "wo_source_lock": source_lock, "expected_TF_source_evaluations": 5, "new_free_generation_task_answers": 0,
        "new_GPU_jobs_started": 0, "actual_model_execution": False}
    technical_plan_path = BOOTSTRAP / "WO_TECHNICAL_JOB_PLAN_v1.json"
    technical_plan_sha = jobs.freeze(technical_plan_path, technical_plan)
    require(runner.runtime_source_manifest() == graph and technical.runtime_source_manifest() == technical_graph
        and all(jobs.sha(ROOT / role) == digest for role, digest in source_pins.items()), "actual source changed during local preparation")
    preparation = {"schema": "trajkv.round11.wo_preparation.v1", "status": "PASS_LOCAL_WO_DESCRIPTOR_PREPARATION_NO_LAUNCH",
        "source_script_sha256": jobs.sha(__file__), "development_admission": accepted,
        "wo_source_lock": source_lock,
        "actual_execution_source_pins": source_pins, "runtime_source_hashes": graph,
        "technical_runtime_source_hashes": technical_graph, "source_group_manifest_sha256": SOURCE_GROUP_SHA,
        "input_manifest_sha256": INPUT_MANIFEST_SHA, "declared_new_cells": 8, "expected_new_free_generation_QA": 192,
        "reused_reference_views": 288, "total_scored_views": 480, "dataset_order": list(DATASET_ORDER),
        "GPU_allocation": {"0": "SpatialWO", "1": "JointSpatialWO"}, "queues": queues, "config_receipts": config_receipts,
        "harvest_plan_path": str(harvest_path), "harvest_plan_sha256": harvest_sha,
        "technical_job_plan_path": str(technical_plan_path), "technical_job_plan_sha256": technical_plan_sha,
        "new_GPU_jobs_started": 0, "SSH_used": False, "Git_used": False, "R128_reference_files_read": False,
        "candidate_quality_status": "NOT_RUN", "actual_remote_runtime_validation_still_required": True}
    path = BOOTSTRAP / "WO_JOB_PLAN_v1.json"
    digest = jobs.freeze(path, preparation)
    return {"status": preparation["status"], "cells": 8, "new_QA": 192, "reused_views": 288,
        "job_plan": str(path), "job_plan_sha256": digest, "harvest_plan": str(harvest_path), "harvest_plan_sha256": harvest_sha,
        "technical_job_plan": str(technical_plan_path), "technical_job_plan_sha256": technical_plan_sha}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-summary", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.primary_summary, args.decision), ensure_ascii=False))


if __name__ == "__main__":
    main()
