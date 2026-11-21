"""Final matched500 R7 analysis. No network, GPU, scoring or method selection.

The final destination is created only after all eight real source/score
closures and exact input/context checks pass. R7 is admitted by its dedicated
executed-driver proof; it is never relabelled as a canonical SourceRun.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
E = Path("${EVIDENCE_ROOT}")
sys.path.insert(0, str(ROOT/"code")); sys.path.insert(0, str(ROOT/"ops")); sys.dont_write_bytecode = True
from prediction_cache import input_key, evaluate_semantics

MANIFEST = E/"results/research_execution/R7_M1_two_points_four_domains_fixed500_v2/CONFIG_JOB_MANIFEST.json"
MANIFEST_SHA = "5d4d4c0f7a02604882ff3be393cda934f0e4f806bdffeed3fbab76a69f7a87fe"
HARVEST = E/"results/r7_matrix500_v2/harvest"
REFERENCES = E/"results/r7_alignment/M1_four_domains_full500_baseline_reference_v1"
FINAL = E/"results/r7_matrix500_v2/analysis_final_v1"
REF_PINS = {
    "summary.json": "c93512f9dd716874f1a3df6c56636e584a259857030488cc5ecdb18c70cdb725",
    "SOURCE_REFERENCE_INDEX.json": "b6b528b7c40b0c57911e766d8d6f55007dc817c2a2f155f9afb0b432448eed3f",
    "INPUT_AND_COMMON_CONTRACTS500.json": "44ad498776c0ec0a90bc686ec621a220ddb154a19bd22599db76a26525004cf3",
}
PINS = {
    "code/r7_frozen_pilot_v2.py": "3ff896405be5ac91067bdbe8d736c0552e46f835dea90a93c9835ceeb8fd3a91",
    "code/r7_legacy_bridge_v2.py": "608b1660896c8e8fd05484be47acf73e99f260db71731239937d0279ed842dad",
    "code/r7_legacy_bridge_v1.py": "b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b",
    "code/m1_protocol.py": "4ba12190fe3c6513c4a45822c4294c3e40cf82ea13ac791c6cb677aed3871427",
    "code/cache_bridge.py": "36b860522276895dd361120d315d745a785fd020cbddbff9e75051e26a907025",
    "code/attention_bridge.py": "eef35d19dfc04fab12f0b6f9ba618a8b34ed48f4488edd566ea4578fb3311876",
    "code/ownership.py": "8443adca9e991663e42da6eff0609b54eb7ef0b13fee8ac09f4e12d1894a8456",
    "code/score.py": "72b5182eeb97e930403a970feebb4f1fadac23d04544c9646c6c9fd6b34bfba6",
    "code/metrics.py": "98b260c766e095f2b629d603de5cfd6519a91dccaaee471f02486955bb590d47",
    "code/gates/metrics_gate.py": "e91887b55561759c2349860cf9a5acb2ed9284be99d6b3ae6469e8f2c4731606",
    "code/vendor_textvqa_metric.py": "b1a8e90abbf35e32e1e6c146c36c4a47ad72b71e73011f97cc3729aa1402b4c5",
    "code/evaluate.py": "dbb556d87827c146e596b1e01de6c58a7a8b08f788c4c13d18129f4b1f13f75c",
    "code/model_registry.py": "969b39c1df27b63166f4b783187084022cf8548799a6c08178b8665b54d13c58",
    "code/dataset_registry.py": "4aaf39f4d0725080e66a926754ba50c33247f115253c36e55433805d79bf0983",
    "code/compare_runs.py": "ce73705f17cfcbfe66358f3525eb3121e2e141056bc9a043e7423546878d15a1",
    "ops/prepare_remaining_phase_delivery_v1.py": "621b9a3bf29811e3b11292397c1248b19046154a8924dee87cd2fa1ae10c0650",
}
DECLARED = set(list(PINS)[:7]) | {"code/model_registry.py", "code/dataset_registry.py", "code/evaluate.py"}
DATASETS = ("textvqa", "infovqa", "docvqa", "chartqa")
ANCHORS = ("F", "K32", "C2")  # Prespecified; no best-score selection.
AUTHOR_AUX = "anls_docvqa_author_strip_ge0p5_v1"
TOKENIZER_SHA = "b336af8282760f5e596726eb9834ce20dd6625d55aa7df73030ef4c59028567f"
TOKENIZER_PROOF = ROOT.parent/"TrajKV/audit/recovered-by-content-hash"/TOKENIZER_SHA/"tokenizer_config.json"


def require(value, message):
    if not value: raise ValueError(message)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")


def table(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)


def formal_rows(path):
    require(sha(path) == MANIFEST_SHA, "exact frozen8 manifest differs")
    rows = [r for r in load(path)["rows"] if r["tier"] == "fixed500"]
    require(len(rows) == 8 and {(r["profile"], r["dataset"]) for r in rows} ==
            {(p, d) for p in ("K1", "K2") for d in DATASETS}, "need both frozen profiles/all four500 domains")
    for r in rows:
        require(r["count"] == 500 and sha(ROOT/r["config"]) == r["config_sha256"] and
                sha(ROOT/r["job"]) == r["job_sha256"], "frozen config/job differs")
        r.update(name=Path(r["job"]).stem)  # Explicit actual job key.
    return rows


def readiness(rows, harvest):
    state_path = Path(harvest)/"state.json"
    state = load(state_path) if state_path.exists() else {"cells": {}}
    result = []; complete = True
    for row in rows:
        cell = state.get("cells", {}).get(row["name"])
        missing = []
        receipt = Path(harvest)/"cell_receipts"/(row["name"]+".json")
        if not cell or not receipt.is_file(): missing.append("actual_immutable_cell_receipt")
        else:
            require(load(receipt) == cell and cell["status"] == "CLOSED_COMPLETE500_SOURCE_AND_OFFLINE_SCORE" and
                    cell["QA"] == 500 and cell["SourceRun_admission_established"] is False, "foreign/unclosed cell receipt")
            require((cell["profile"], cell["dataset"]) == (row["profile"], row["dataset"]) and
                    cell["config_sha256"] == row["config_sha256"] and cell["job_sha256"] == row["job_sha256"], "cell identity differs")
            for name in ("manifest.json", "records.jsonl", "complete.json", "source_retrieval_receipt_r7_v2.json"):
                if not (Path(cell["raw"])/name).is_file(): missing.append(name)
            if not (Path(cell["scored"])/"summary.json").is_file(): missing.append("score_summary")
            if cell["raw_status"] != "PASS_RAW_COMPLETE" or cell["score_status"] != "PASS_SCORED_COMPLETE":
                missing.append("complete_primary_unavailable_failure_preserved")
        complete &= not missing
        result.append({"profile": row["profile"], "dataset": row["dataset"], "name": row["name"], "missing": missing, "cell": cell})
    return {"status": "ALL8_METADATA_CLOSED_DEEP_VALIDATION_PENDING" if complete else "PENDING_REAL8_FULL500_SOURCE_SCORE",
            "ready": complete, "expected_cells": 8, "expected_QA": 4000,
            "closed_metadata_cells": sum(not x["missing"] for x in result), "cells": result,
            "smoke64_or400_may_substitute": False, "new_GPU_QA": 0}


def native_context(manifest):
    """Read actual fields; an absent field is an error, never borrowed from F."""
    runtime = {}
    for key, aliases in {
        "torch_version": ("torch_version",), "transformers_version": ("transformers_version",),
        "cuda_version": ("cuda_version", "CUDA_version"), "python_version": ("python_version",),
        "hardware": ("hardware",), "runtime_executable": ("runtime_executable",),
    }.items():
        values = [manifest[k] for k in aliases if k in manifest]
        require(values and all(v == values[0] for v in values), "unrecorded/conflicting actual runtime field: " + key)
        runtime[key] = values[0]
    files = sorted(({k: r[k] for k in ("file", "bytes", "sha256")} for r in manifest["model_identity"]["files"]), key=lambda r: r["file"])
    return {"model_files": files, "runtime": runtime,
            **{k: manifest[k] for k in ("processor_id", "processor_version", "processor_file_hashes", "generation_native", "generation_overrides")}}


def input_identity(raw):
    return {"input_key": input_key(raw), "image_sha256": raw["original_image_sha256"],
            "input_ids_sha256": raw["input_ids_sha256"], "full_prompt_hash": raw["full_prompt_hash"],
            "suffix_sha256": raw["question_suffix_sha256"], "positions_sha256": raw["native_position_ids_sha256"],
            "positions_shape": raw["native_position_ids_shape"], "visual_positions_sha256": canonical(raw["visual_positions"]),
            "N": raw["actual_visual_token_count"], "input_tokens": raw["input_tokens"],
            "boundary": raw["compression_boundary"], "suffix_tokens": raw["question_suffix_tokens"]}


def score_metadata(folder, expected_sha=None):
    p = Path(folder); summary = load(p/"summary.json")
    require(summary["status"] == "PASS_SCORED_COMPLETE" and summary["expected_run_questions"] == 500, "only all500 primary PASS")
    require(expected_sha is None or sha(p/"summary.json") == expected_sha, "score summary changed")
    require(summary["integrity"]["scorer_sha256"] == PINS["code/score.py"], "scorer source differs")
    for key in ("code/metrics.py", "code/vendor_textvqa_metric.py", "code/gates/metrics_gate.py"):
        require(summary["integrity"]["metric_code_hashes"][key] == PINS[key], "metric source differs")
    for name, digest in summary["derived_file_hashes"].items(): require(sha(p/name) == digest, "scored cache differs")
    return summary


def check_source(row, cell):
    raw = Path(cell["raw"]); m, c = load(raw/"manifest.json"), load(raw/"complete.json")
    require(m["config"] == load(ROOT/row["config"]) and m["config_sha256"] == row["config_sha256"], "actual R7 config differs")
    require(m["expected_QA"] == c["expected_QA"] == c["recorded_QA"] == 500 and c["coverage_complete"] and
            c["counts"] == {"PASS": 500} and c["status"] == "PASS_RAW_COMPLETE", "R7 complete source is not all500")
    require(sha(raw/"manifest.json") == c["manifest_sha256"] == cell["raw_file_sha256"]["manifest.json"] and
            sha(raw/"complete.json") == cell["raw_file_sha256"]["complete.json"], "small raw closure differs")
    proof_path = raw/"source_retrieval_receipt_r7_v2.json"; proof = load(proof_path)
    require(sha(proof_path) == cell["source_receipt_sha256"] and proof["status"] == "PASS_R7_EXECUTED_DRIVER_AND_DEPENDENCY_SOURCE_SHA" and
            proof["remote_hashes_computed_independently"] is True and proof["SourceRun_admission_established"] is False,
            "dedicated actual R7 source proof required")
    entries = {x["file"]: x for x in proof["files"]}
    required = {"source_inventory.json", "launch_receipt.json", "source_config.json", "source_job.json", "source_driver.py"} | {"source_files/"+x for x in DECLARED}
    require(required <= set(entries) and len(entries) == len(proof["files"]), "incomplete/duplicate executed source proof")
    for name, entry in entries.items():
        require((name in required or (name.startswith("legacy_sources/") and name.endswith(".py"))) and
                entry["bytes"] <= 4*1024**2 and not Path(name).is_absolute() and ".." not in Path(name).parts and
                sha(raw/name) == entry["sha256"] and (raw/name).stat().st_size == entry["bytes"], "executed proof file changed")
    require(sha(raw/"source_config.json") == row["config_sha256"] and
            sha(raw/"source_job.json") == row["job_sha256"], "executed config/job proof differs")
    job = load(ROOT/row["job"]); launch = load(raw/"launch_receipt.json")
    remote = job["argv"][job["argv"].index("--output")+1].split("/results/", 1)[0]
    snapshot = remote+"/jobs/"+row["name"]+".source"
    expected = [snapshot+"/code/r7_frozen_pilot_v2.py" if a == remote+"/code/r7_frozen_pilot_v2.py" else a for a in job["argv"]]
    require(type(launch["pid"]) is int and launch["pid"] > 0 and launch["name"] == row["name"] and
            launch["job_sha256"] == row["job_sha256"] and launch["argv_executed"] == expected and
            launch["source_snapshot"] == m["runtime_source_root"] == snapshot and launch["runtime_python"] == m["runtime_executable"] == job["argv"][0], "actual producer/runtime binding differs")
    require(proof["producer_name"] == row["name"] and proof["source_snapshot"] == snapshot and
            str(launch["CUDA_VISIBLE_DEVICES"]) == str(job["env"]["CUDA_VISIBLE_DEVICES"]) == str(row["GPU"]), "producer proof/GPU differs")
    require(sha(raw/"source_inventory.json") == launch["source_inventory_sha256"], "inventory launch binding differs")
    iv = load(raw/"source_inventory.json")
    require(set(m["source_hashes"]) == DECLARED, "unexpected or missing declared inference dependency")
    for name in DECLARED:
        require(sha(raw/"source_files"/name) == iv[name] == m["source_hashes"][name], "actual dependency source differs")
        if name in PINS: require(iv[name] == PINS[name], "frozen actual dependency changed")
    require(sha(raw/"source_driver.py") == PINS["code/r7_frozen_pilot_v2.py"], "wrong R7 driver alias")
    return m, c


def load_references(directory, tokenizer_proof):
    directory = Path(directory)
    for name, digest in REF_PINS.items(): require(sha(directory/name) == digest, "frozen23-reference index changed")
    index = load(directory/"summary.json"); common = load(directory/"INPUT_AND_COMMON_CONTRACTS500.json")
    require(index["complete_baseline_points"] == 23 and index["QA_per_dataset"] == 500, "wrong frozen reference scope")
    require(sha(tokenizer_proof) == TOKENIZER_SHA and load(tokenizer_proof)["clean_up_tokenization_spaces"] is False,
            "baseline implicit cleanupFalse needs its actual locked tokenizer config")
    references = {(b["dataset"], b["point"]): b for b in index["baselines"] if b["point"] in ANCHORS}
    require(set(references) == {(d, a) for d in DATASETS for a in ANCHORS}, "all12 prespecified anchors required")
    common["shared_evaluate_support_semantics"] = {}
    for item in common["rows"]: require(sha(item["input_path"]) == item["input_sha256"], "locked full500 inputs changed")
    for (ds, _), b in references.items():
        q = score_metadata(b["scored"], b["score_summary_sha256"])
        require(q["metric_ids"] == [b["quality"]["primary_metric_id"]] and
                q["integrity"]["raw_files"] == b["view_raw_sha256"], "reference source/score identity differs")
        vm = load(Path(b["view"])/"manifest.json")
        require(sha(Path(b["view"])/"manifest.json") == b["view_raw_sha256"]["manifest.json"], "reference manifest changed")
        context = common["common_contracts_by_dataset"][ds]["context"]
        require(native_context(vm) == {k: context[k] for k in native_context(vm)}, "actual baseline runtime/processor/generation differs")
        require(vm["effective_contract"]["model_spec"] == context["model_spec"], "baseline verified model revision differs")
        resource = b["resources"]
        if resource.get("physical_proof"):
            require(sha(resource["physical_proof"]) == resource["physical_proof_sha256"], "frozen baseline physical proof changed")
        dependencies = vm["effective_contract"]["relevant_source_hashes"]
        for key in ("code/m1_protocol.py", "code/cache_bridge.py", "code/attention_bridge.py", "code/ownership.py"):
            require(dependencies[key] == PINS[key], "baseline common native support source differs")
        semantic = dependencies["code/evaluate.py"]
        require(common["shared_evaluate_support_semantics"].get(ds, semantic) == semantic, "baseline shared evaluator semantics differ")
        common["shared_evaluate_support_semantics"][ds] = semantic
        b["_score_summary"] = q
    return references, common


def r7_point(row, cell, common):
    m, c = check_source(row, cell); ds = row["dataset"]
    summary = score_metadata(cell["scored"], cell["score_summary_sha256"])
    require(summary["metric_ids"] == [row["primary_metric_id"]] and summary["method"] == row["method_id"], "wrong R7 score identity")
    context = common["common_contracts_by_dataset"][ds]["context"]
    require(native_context(m) == {k: context[k] for k in native_context(m)} and m["model_spec"] == context["model_spec"], "actual R7 native context differs")
    require(m["decode_overrides"] == {"clean_up_tokenization_spaces": False, "skip_special_tokens": True} and
            m["processor_file_hashes"]["tokenizer_config.json"] == TOKENIZER_SHA, "R7 decode contract differs")
    source_eval = Path(cell["raw"])/"source_files/code/evaluate.py"
    # Only the shared support module uses legacy AST-equivalence. This never
    # admits the custom R7 runner as a canonical baseline SourceRun.
    baseline_behavior = common["common_contracts_by_dataset"][ds]["behavior"]
    cfg = m["config"]
    require(cfg["preprocessing"] == baseline_behavior["preprocessing"] and cfg["question_suffix"] == baseline_behavior["question_suffix"], "R7 preprocessing/prompt contract differs")
    input_spec = next(x for x in common["rows"] if x["dataset"] == ds)
    require(m["inputs_sha256"] == cfg["inputs_sha256"] == input_spec["input_sha256"] and
            m["expected_ids"] == input_spec["selected_ids"], "R7 is not the original ordered full500")
    support_semantic = evaluate_semantics(source_eval, sha(source_eval))
    require(support_semantic == common["shared_evaluate_support_semantics"][ds], "shared evaluation support numerical semantics differ")
    retrieval_path = Path(cell["raw"])/"retrieval_receipt.json"; retrieval = load(retrieval_path)
    require(sha(retrieval_path) == summary["integrity"]["retrieval_receipt_sha256"] and
            retrieval["status"] == "PASS_REMOTE_RAW_SHA256_EXACT" and retrieval["remote_hashes_computed_independently"] is True,
            "independent raw retrieval proof changed")
    anchors = {x["file"]: x for x in retrieval["files"]}
    require(set(anchors) == {"manifest.json", "records.jsonl", "complete.json"}, "exact raw3 receipt required")
    for key, item in anchors.items():
        require(item["sha256"] == cell["raw_file_sha256"][key] and item["bytes"] == (Path(cell["raw"])/key).stat().st_size,
                "raw receipt/hash/size differs")
    scored = {}
    with (Path(cell["scored"])/"scored_records.jsonl").open("rb") as f:
        for line in f:
            item = json.loads(line); require(item["id"] not in scored, "duplicate R7 score ID"); scored[item["id"]] = item
    identities = {}; bits = []; total = 0; over = {"1.3": 0, "1.8": 0}; raw_digest = hashlib.sha256()
    import prepare_remaining_phase_delivery_v1 as frozen_physical
    with (Path(cell["raw"])/"records.jsonl").open("rb") as f:
        for number, line in enumerate(f, 1):
            raw_digest.update(line); raw = json.loads(line); score = scored[raw["id"]]
            require(raw["id"] == m["expected_ids"][number-1] and raw["status"] == "PASS" and score["score_status"] == "SCORED_PASS" and
                    raw["model_id"] == "M1" and raw["method"] == score["method"] == row["method_id"] and
                    score["raw_jsonl_line_number"] == number and score["raw_jsonl_exact_line_sha256"] == hashlib.sha256(line).hexdigest(), "R7 source/score exact line differs")
            first = raw["first_generation_call"]
            require(first["native_first_logits_injected"] is False and first["cache_tokens_before"] == raw["compression_boundary"] and
                    first["input_tokens"] == raw["question_suffix_tokens"] and first["input_ids_sha256"] == raw["question_suffix_sha256"],
                    "first answer was not computed from this branch's full suffix")
            require(score["stored_visual_KV_bytes"] == raw["stored_visual_KV_bytes"] and
                    score["actual_effective_visual_bits"] == raw["actual_effective_visual_bits"], "R7 scored owner differs")
            actual = frozen_physical.physical_complete_bytes(raw, m["model_spec"])
            denominator = 28*2*4*raw["actual_visual_token_count"]*128*2
            require(raw["full_kv_byte_denominator"] == denominator and abs(16*actual/denominator-raw["actual_effective_visual_bits"]) < 1e-12, "R7 complete bits/geometry differs")
            identities[raw["id"]] = input_identity(raw); bits.append(16*actual/denominator); total += actual
            over["1.3"] += 160*actual > 13*denominator; over["1.8"] += 80*actual > 9*denominator
    require(len(bits) == len(identities) == len(scored) == 500 and
            raw_digest.hexdigest() == c["records_jsonl_sha256"] == cell["raw_file_sha256"]["records.jsonl"] == summary["integrity"]["raw_files"]["records.jsonl"], "R7 full500 raw closure differs")
    require(canonical(identities) == common["parent_full500_alignment"][ds]["per_ID_identity_fingerprint"], "per-ID image/prompt/token/visual/position/boundary differs from frozen matched anchors")
    aux = summary.get("diagnostic_metrics_not_selected_by_score", {}).get(AUTHOR_AUX)
    if ds in ("docvqa", "infovqa"):
        require(aux is not None, "legacy author-strip auxiliary must be kept separately")
        aux = {"metric_id": AUTHOR_AUX, "mean": aux["primary_full_cohort_mean"],
               "CI95": aux["primary_full_cohort_ci95"], "role": "legacy_auxiliary_not_primary"}
    return {"profile": row["profile"], "dataset": ds, "QA": 500, "method": row["method_id"],
            "metric_id": row["primary_metric_id"], "quality": summary["primary_mean"], "CI95": summary["primary_ci95"],
            "legacy_author_strip_aux": aux, "groups": summary["groups"], "complete_mean_bytes": total/500,
            "mean_complete_bits": sum(bits)/500, "max_complete_bits": max(bits), "min_complete_bits": min(bits),
            "strict_all_request_budget": {b: {"eligible": n == 0, "over_budget_QA": n} for b, n in over.items()},
            "raw": cell["raw"], "scored": cell["scored"], "raw_file_sha256": summary["integrity"]["raw_files"],
            "score_summary_sha256": sha(Path(cell["scored"])/"summary.json"), "source_receipt_sha256": cell["source_receipt_sha256"],
            "input_fingerprint": canonical(identities), "shared_evaluate_support_semantics": support_semantic,
            "SourceRun_admission_claimed": False,
            "additional_R7_runtime_library_sources_not_imputed_to_baselines": m.get("runtime_library_source_sha256", {})}


def analyze(rows, ready, references, tokenizer_proof, output):
    require(not Path(output).exists(), "use an immutable new final destination")
    baselines, common = load_references(references, tokenizer_proof)
    points = [r7_point(row, item["cell"], common) for row, item in zip(rows, ready["cells"])]
    # Every R7 raw has been streamed exactly once before any final content exists.
    from compare_runs import compare
    comparisons = []; pair_values = []
    for point in points:
        for anchor in ANCHORS:
            baseline = baselines[(point["dataset"], anchor)]
            value = compare(Path(baseline["scored"]), Path(point["scored"]))
            require(value["status"] == "PASS_PAIRED_COMPLETE" and value["expected_QA"] == value["paired_scored_QA"] == 500, "paired full500 not complete")
            file = Path(output)/"pairs"/(point["profile"]+"_"+point["dataset"]+"_minus_"+anchor+".json")
            pair_values.append((file, value))
            comparisons.append({"profile": point["profile"], "dataset": point["dataset"], "anchor": anchor,
                                "primary": value["primary"], "file": str(file),
                                "different_resource_costs_not_iso_byte_causality": True})
    require(len(comparisons) == 24, "exact24 prespecified comparisons")
    anchor_points = []
    for (dataset, label), b in baselines.items():
        resource = b["resources"]
        anchor_points.append({"profile": label, "dataset": dataset, "QA": 500, "method": b["method"]["id"],
            "metric_id": b["quality"]["primary_metric_id"], "quality": b["quality"]["primary_mean"], "CI95": b["quality"]["primary_CI95"],
            "legacy_author_strip_aux": b["quality"]["auxiliary_metrics"].get(AUTHOR_AUX),
            "groups": b["_score_summary"]["groups"], "complete_mean_bytes": resource["mean_complete_visual_bytes"],
            "mean_complete_bits": resource["mean_actual_complete_bits"], "max_complete_bits": resource["max_actual_complete_bits"],
            "strict_all_request_budget": {key: {"eligible": resource["over_budget_QA"][key] == 0,
                                               "over_budget_QA": resource["over_budget_QA"][key]} for key in ("1.3", "1.8")},
            "baseline_frozen_physical_evidence_reused_not_reconstructed": True, "score_summary_sha256": b["score_summary_sha256"]})
    groups = []
    for p in points + anchor_points:
        if p["dataset"] == "chartqa":
            for name in ("human", "aug"):
                group = p["groups"][name]; require(group["expected_questions"] == 250, "fixed ChartQA groups changed")
                groups.append({"profile": p["profile"], "group": name, "QA": 250, "primary": group["primary"]})
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    for (file, value), item in zip(pair_values, comparisons): write(file, value); item["sha256"] = sha(file)
    report = {"status": "PASS_REAL8_FULL500_MATCHED_QUALITY_COMPLETE_BYTES_AND24PAIRS", "R7_cells": 8, "R7_QA": 4000,
              "points": points, "baseline_anchor_points": anchor_points, "paired_comparisons": comparisons, "ChartQA_human_aug": groups,
              "anchors_prespecified": list(ANCHORS), "reference_index_points": 23,
              "quality_and_CI_unit": "fraction_0_to_1; paired *_pp fields are percentage points",
              "anchor_label_C2": "CalibQuant common-FP16 numeric-v2 2bit reference (Calib2), not R7 K2",
              "author_strip_is_legacy_aux_not_primary": True, "new_GPU_QA_by_analysis": 0,
              "unadjusted_CI_not_winner_equivalence_or_FP16_upper_bound": True,
              "different_metrics_not_averaged": True, "fixed500_exploratory_not_independent_confirmation": True,
              "whole_external126_24_completion_claimed": False, "SourceRun_admission_claimed_for_R7": False,
              "Round9_goal_completion_assessed": False,
              "scope": "Fixed two-profile full500 reference step only; not calibration/mechanism research completion.",
              "source": {"manifest_sha256": MANIFEST_SHA, "reference_pins": REF_PINS, "helper_pins": PINS,
                         "tokenizer_cleanupFalse_proof": str(tokenizer_proof), "tokenizer_sha256": TOKENIZER_SHA,
                         "analysis_source_sha256": sha(__file__)}}
    write(out/"summary.json", report)
    table(out/"points.csv", [{k: p[k] for k in ("profile", "dataset", "QA", "method", "metric_id", "quality", "CI95", "complete_mean_bytes", "mean_complete_bits", "max_complete_bits", "strict_all_request_budget", "legacy_author_strip_aux")} for p in points + anchor_points])
    table(out/"paired_comparisons.csv", [{"profile": p["profile"], "dataset": p["dataset"], "anchor": p["anchor"], **p["primary"]} for p in comparisons])
    table(out/"ChartQA_human_aug.csv", groups)
    write(out/"ANALYSIS_RECEIPT.json", {"status": report["status"], "actual_R7_cells": 8, "actual_R7_QA": 4000,
          "paired_comparisons": 24, "new_GPU_QA": 0, "R7_large_raw_passes_per_cell": 1,
          "baseline_large_raw_reconstruction_or_scan_repeated": False,
          "outputs": [{"file": str(p.relative_to(out)), "sha256": sha(p)} for p in sorted(out.rglob("*")) if p.is_file()]})
    return report


def probe_smoke(raw, output):
    """Schema only on one already completed4QA source; no400/500 extrapolation."""
    raw = Path(raw); m, c = load(raw/"manifest.json"), load(raw/"complete.json")
    require(m["expected_QA"] == c["expected_QA"] == c["recorded_QA"] == 4 and c["coverage_complete"], "only actual closed smoke4")
    with (raw/"records.jsonl").open("rb") as f: line = f.readline(); row = json.loads(line)
    context = native_context(m); identity = input_identity(row)
    write(Path(output)/"SCHEMA_ONLY.json", {"status": "PASS_REAL_SMOKE4_FIRST_ROW_SCHEMA_EXTRACTION_ONLY", "raw": str(raw),
          "recorded_QA": 4, "first_ID": row["id"], "first_line_sha256": hashlib.sha256(line).hexdigest(),
          "actual_native_context": context, "identity_fields": sorted(identity),
          "quality_statistics_computed": False, "formal500_admitted": False, "new_GPU_QA": 0})


def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--harvest", type=Path, default=HARVEST); p.add_argument("--references", type=Path, default=REFERENCES)
    p.add_argument("--tokenizer-config-proof", type=Path, default=TOKENIZER_PROOF)
    p.add_argument("--output", type=Path, default=FINAL); p.add_argument("--check-readiness", action="store_true")
    p.add_argument("--probe-smoke", type=Path); a = p.parse_args()
    for name, digest in PINS.items(): require(sha(ROOT/name) == digest, "frozen helper changed: " + name)
    require(a.output.resolve().is_relative_to(E.resolve()), "use E data disk")
    if a.probe_smoke:
        require(a.output.resolve() != FINAL.resolve() and not a.output.exists(), "schema probe needs a separate new output")
        probe_smoke(a.probe_smoke, a.output); print("SCHEMA_ONLY_NO500_STATISTICS"); return
    rows = formal_rows(a.manifest); ready = readiness(rows, a.harvest)
    if a.check_readiness:
        require(a.output.resolve() != FINAL.resolve() and not a.output.exists(), "preparation must not occupy final destination")
        write(a.output/"READINESS.json", ready); print(ready["status"]); return
    if not ready["ready"]:
        print(json.dumps({k: ready[k] for k in ("status", "expected_cells", "expected_QA", "closed_metadata_cells")})); return
    try:
        result = analyze(rows, ready, a.references, a.tokenizer_config_proof, a.output)
    except Exception as exc:
        print(json.dumps({"status": "PENDING_EXACT_SOURCE_INPUT_OR_SCORE_BINDING", "error": str(exc), "no_partial_quality_claim": True})); raise
    print(result["status"])


if __name__ == "__main__": main()
