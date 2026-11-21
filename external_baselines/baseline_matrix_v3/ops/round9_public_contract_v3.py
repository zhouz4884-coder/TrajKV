"""R9 custom inference/fit closure, distinct from immutable R7 publication V1.

No SourceRun admission, transport, GPU or publishing. Components remain actual
fresh runs; a 32+32 logical point never acquires a fictitious fresh64 raw run.
Private calibration frames, W_O tensors and original images are not selected.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("${EVIDENCE_ROOT}")
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "ops")]
sys.dont_write_bytecode = True
from public_phase_contract_v2 import require, load, sha, resolve, inside, validate_scored
from round9_inputs_v1 import load_round9_inputs
from verify_public_custom_driver_v1 import R7_ROLE_PINS, r7_role_path, r7_verify_executed_roles, r7_verify_record_role_pins
from analyze_r7_fixed500_v2 import PINS as CORE_PINS

SCHEMA = "trajkv.round9_research.raw.v1"
MANIFEST = ROOT / "configs/round9_inputs_v1/INPUT_MANIFEST.json"
MANIFEST_SHA = "583860001cbb4e7fdd1dc974263c3bf630d8daef917d5c1bb089f5a65a57a650"
CALIB_INPUT_SHA = "2de0d5308892ee86bee48c3205c6eb49eee196ffee32ef6f8bcab19a30b6c216"
CALIB_COMPLETE_SHA = "c838b3a79cb970233a9579f8972ae4276157fad1001fd9b4e9118945bc0dc521"
CALIB_META = EVIDENCE / "results/round9/calibration32_v1_public_metadata_v1"
POLICY = EVIDENCE / "results/publication/R9_custom_driver_support_v3/R9_SOURCE_POLICY.json"
POLICY_REL = "results/publication/R9_custom_driver_support_v3/R9_SOURCE_POLICY.json"
RAW_ROOT = EVIDENCE / "results/round9_v1/raw"
SCORE_ROOT = EVIDENCE / "results/round9_v1/scored"
BASE_BOOK = "af29f26ddcbdf4ac6f098c3df497d2d741f0e7909c13ecc090194446846c9589"
DRIVERS = {"code/round9_runner_v1.py": "435b619f066d3a82188af46f7ad7f2d96125bc3680ace8546e0683cd087444e0",
           "code/round9_runner_v2.py": "cebbb10de62fa5ee8911c0afe5f6414f31c4180b27edbd4e500a83f6d8484bea",
           "code/round9_runner_v3.py": "bb6f766167c13f47b59f5c32aeb35e0556d1b9137596d9ea69ef009bebe38a8c",
           "code/round9_runner_v4.py": "cf4885b77b03fd94aadd5bbabc5a1e6d62e3259ba6037b7a708fabab71758961"}
MODES = {"FP16", "original", "M-V", "V-FZ", "V-RE", "FB0", "FB1", "K-MSE", "K-Func", "COMBO", "TIER", "K-Mu", "K-Func_FZJ", "K-Book_FZJ", "BRPE_Lookup8", "BRPE_Drop4", "BRPE_Exchange"}
BRPE_MODES = {"BRPE_Lookup8": "Lookup8", "BRPE_Drop4": "Drop4", "BRPE_Exchange": "Exchange"}
BRPE_HELPER_SHA = "15e075936d235cdd0a3935961b8083c2f10c9df59c545f1a6bd2c7aecd3dd9fe"
BRPE_CAL_SOURCE_SHA = "8b986799062ef8d3fbc67624344d1740cfea4726b64aa3fc4d995ffaebc54438"
BRPE_PROFILE_SHA = "7ec305a76cebf5ca9e358842e11e68fe6a77e6af9f2d83c39af723ff17eb5c22"
BRPE_PROOF = EVIDENCE / "results/round9/exchange_calibration_v2_public_proof_v1"
BRPE_DONORS = [[14, 0, 0], [21, 1, 0], [16, 1, 0], [21, 2, 0]]
FIT_CODES = {"round9_train_v1.py": "4809904d5a88643090535b925b7de9f090f2409c53d219671c07614dda466779",
             "round9_calibration_v1.py": "757cc51740ad2f92d8ebd9ad9adadf2719131ae634b16b24052363f8e45ba84f",
             "round9_keyfit_v1.py": "96adf8b9a94e51da00ab99eaeebb83989b005f15bb3762b18d6c01d4ef5c6ec9"}
CURRENT_RUNS = ["FP16_infovqa_64_v2", "FP16_textvqa_64_v2", "original_infovqa_64_v2", "original_textvqa_64_v2",
                "K_MSE_infovqa_screen32_v1", "K_MSE_textvqa_screen32_v1", "K_Func_infovqa_screen32_v1", "K_Func_textvqa_screen32_v1",
                "K_MSE_textvqa_hold32_v1", "K_MSE_infovqa_hold32_v1", "KMu_textvqa_diagnostic32_v1", "KMu_infovqa_diagnostic32_v1",
                "MSE_FZJ_textvqa_diagnostic32_v1", "MSE_FZJ_infovqa_diagnostic32_v1"]
BRPE_RUNS = [f"BRPE_{mode}_{ds}_screen32_v1" for mode in ("Drop4", "Exchange") for ds in ("textvqa", "infovqa")]
STAGE_RUNS = CURRENT_RUNS + BRPE_RUNS


def expected_sources(driver):
    names = ("r7_legacy_bridge_v1.py", "r7_legacy_bridge_v2.py", "m1_protocol.py", "model_registry.py", "dataset_registry.py", "evaluate.py", "cache_bridge.py", "attention_bridge.py", "ownership.py")
    result = {"code/" + name: CORE_PINS["code/" + name] for name in names}
    result.update({driver: DRIVERS[driver], "code/round9_value_bank_v1.py": "14d242a6643a738457c9b8593d0f0eb6e383c5e979b2e196fcca28e7d37a5f2e",
        "code/round9_inputs_v1.py": "46ccb0d3fffe93994b1339474887ecf86b730eb3edd9f7cee9fa4d8b96e68ac9",
        "code/round9_extended_bank_v1.py": "57ec77e7c2d7f3b8abbd3f9fcaaa996c60228a2821d4085dcaf3604a98ba6001"})
    if driver != "code/round9_runner_v1.py":
        result.update({"code/round9_extended_bank_v2.py": "5745e4cda9c1c5d1e2256a49908c15a350321834e1d0d36097a7fac99734acff",
                       "code/round9_key_controls_v1.py": "e73e8594e80f7b3b14dfc6bc3c271858dc42798de9f67843de14b7376f474878"})
    if driver in ("code/round9_runner_v3.py", "code/round9_runner_v4.py"):
        result["code/round9_key_controls_v2.py"] = "44500c370f4605e377750fc003b1dfeff052f129f122a7fabb1ba3c70d4dfdf4"
    if driver == "code/round9_runner_v4.py": result["code/round9_precision_exchange_v1.py"] = BRPE_HELPER_SHA
    return result


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False); stream.write("\n")


def check_files(directory, receipt, *, include_raw=True):
    entries = {x["file"]: x for x in receipt["files"]}
    require(len(entries) == len(receipt["files"]), "Duplicate original evidence file")
    selected = []
    for name, item in entries.items():
        pure = PurePosixPath(name)
        require(not pure.is_absolute() and ".." not in pure.parts and "\\" not in name, "Unsafe source proof path")
        require(pure.suffix in (".json", ".jsonl", ".py", ".md", ".csv"), "No private tensor/image/weight evidence")
        path = Path(directory).joinpath(*pure.parts)
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Original proof SHA/bytes differs: " + name)
        if include_raw or name not in ("manifest.json", "records.jsonl", "complete.json"): selected.append(path)
    return entries, selected


def actual_context(m):
    require(m["model_identity"]["status"] == "PASS_EXACT_HF_REQUIRED_TREE" and
            m["model_identity"]["revision"] == m["model_spec"]["revision"] == m["config"]["model_revision"] and m["model_spec"]["model_id"] == "M1", "Actual M1 model identity differs")
    runtime = {}
    for name, aliases in {"torch_version": ("torch_version",), "transformers_version": ("transformers_version",),
        "cuda_version": ("cuda_version", "CUDA_version"), "python_version": ("python_version",),
        "hardware": ("hardware",), "runtime_executable": ("runtime_executable",)}.items():
        values = [m[x] for x in aliases if x in m]
        require(values and all(x == values[0] for x in values), "Missing/conflicting actual runtime: " + name)
        runtime[name] = values[0]
    return {"runtime": runtime, "model_spec": m["model_spec"],
        "model_files": sorted(({k: r[k] for k in ("file", "bytes", "sha256")} for r in m["model_identity"]["files"]), key=lambda x: x["file"]),
        **{k: m[k] for k in ("processor_id", "processor_version", "processor_file_hashes", "generation_native", "generation_overrides", "decode_overrides")}}


def verify_brpe_profile(profile):
    require(profile["schema"] == "trajkv.round9.precision_exchange_profile.v1" and profile["base_profile"] == "r7_joint_k1_gt_cond" and
            profile["upgraded_layer"] == 14 and profile["dropped_k_units"] == BRPE_DONORS, "Unreviewed conditional exchange profile")
    require(profile["low_book_sha256"] == BASE_BOOK and profile["high_book_sha256"] == "9d3ad3683cf6888c30f3c8ba9c6dd0d4dbcf40106a34567d578479245e7309ae" and
            profile["base_metadata_factor_sha256"] == "8f47c56af5fe5e4da99bae05b83084173c5a55601c46c91130894075907e9cea", "Static books/original factors differ")
    require(profile["calibration_phase"] == "suffix_last" and profile["calibration_sources"] == 32 and profile["events"] == 896 and profile["references_read"] is False and
            profile["K112_and_V224_factors_unchanged"] is True and profile["retained_K_stages"] == 19 and profile["V_stages_unchanged"] == 20, "Conditional profile scope differs")
    source = profile["source_provenance"]
    require(source["calibrator"] == BRPE_CAL_SOURCE_SHA and source["exchange_helper"] == BRPE_HELPER_SHA and source["collection_complete"] == CALIB_COMPLETE_SHA,
            "Failed v1/new native-query epoch cannot become actual BRPE selector")
    norm = profile["normalization"]
    require(norm["implementation"] == "explicit FP64 amax/shift/exp/sum/log" and norm["row_sum_atol"] == 1e-12 and norm["original_KL_chain_and_additivity_thresholds_unchanged"] is True,
            "Selector normalization/thresholds differ")
    require(abs(profile["route_loss_after"] - profile["route_loss_predicted_from_conditional_marginals"]) <= 1e-8 * (1 + abs(profile["route_loss_after"])), "Conditional four-drop replay not verified")


def verify_brpe_row(row, spec, mode):
    from prepare_remaining_phase_delivery_v1 import physical_complete_bytes
    require(mode in BRPE_MODES, "Unknown BRPE mode")
    actual = physical_complete_bytes(row, spec); n = row["actual_visual_token_count"]
    b0 = 3816 * n + 89733 + 18 * n * 43
    delta = -1344 if mode == "BRPE_Lookup8" else -72 * n - 1304 if mode == "BRPE_Drop4" else 744 - 8 * n
    bridge = row["r7_bridge"]; proof = bridge["round9_precision_exchange"]
    require(n >= 256 and n % 32 == 0 and actual == b0 + delta <= b0 and proof["before_unique_bytes"] == b0 and
            proof["actual_delta_bytes"] == proof["expected_delta_bytes"] == delta and proof["after_unique_bytes"] == actual, "BRPE complete physical ledger differs")
    require(proof["mode"] == BRPE_MODES[mode] and proof["source_sha256"] == BRPE_HELPER_SHA and proof["lookup_dtype"] == "torch.int8" and
            proof["lookup_bytes"] == 448 and proof["old_lookup_and_original_pools_released"] is True and proof["V20_base_parameters_and_residual_values_unchanged"] is True, "BRPE signed lookup/release/V20 proof differs")
    require(bridge["selected_stage_count"] == (43 if mode == "BRPE_Lookup8" else 39) and sum(x[0] == "v" for x in bridge["selected_stages"]) == 20, "K donors/V stages differ")
    if mode != "BRPE_Lookup8": require(proof["dropped_k_units"] == BRPE_DONORS and proof["format_bytes"] == 28 and proof["donor_descriptor_bytes"] == 12, "Real descriptors not paid")
    require(proof["extra_high_book_bytes"] == (2048 if mode == "BRPE_Exchange" else 0) and proof["old_low_book_bytes"] == 128, "Both static book constants not charged")
    tensors = row["owner_before_suffix"]["tensors"]
    for layer in range(28):
        item = tensors[f"layers.{layer}.visual.encoder_constants.lookup_view"]
        require(item["dtype"] == "torch.int8" and item["shape"] == [2, 4, 2] and item["storage_bytes"] == 448, "All28 views must use actual448B signed storage")
    budget = row["BRPE_budget_vs_original_R7_low"]
    require(budget["old_R7_low_B0_bytes"] == b0 and budget["actual_complete_bytes"] == actual and budget["exact_expected_delta_bytes"] == delta and
            budget["at_or_below_R7_low_B0"] is True and budget["strict1p3_budget_is_separate"] is True, "Byte envelope cannot masquerade as strict1.3")
    return actual


def validate_brpe_calibration():
    require(BRPE_PROOF.is_dir(), "WAIT_ACTUAL_BRPE_CALV2_PUBLIC_PROOF")
    receipt = BRPE_PROOF / "FETCH_RECEIPT.json"; require(receipt.is_file(), "WAIT_INDEPENDENT_BRPE_CAL_SOURCE_RECEIPT")
    proof = load(receipt); entries, files = check_files(BRPE_PROOF, proof)
    require(proof["status"] == "PASS_INDEPENDENT_REAL_EXECUTED_SOURCE_AND_METADATA" and proof["actual_executed_driver_sha256"] == BRPE_CAL_SOURCE_SHA and
            proof["independent_probe_then_SFTP_then_second_remote_rehash"] is True and proof["private_PT_WO_weights_or_images_transferred"] is False,
            "Independent calibration readback/availability claim differs")
    require("profile.json" in entries and entries["profile.json"]["sha256"] == BRPE_PROFILE_SHA, "Actual selected profile not independently anchored")
    profile = load(BRPE_PROOF / "profile.json"); verify_brpe_profile(profile)
    complete, summary, launch, job, inventory, cfg = [load(BRPE_PROOF / x) for x in ("complete.json", "summary.json", "launch_receipt.json", "actual_job.json", "source_inventory.json", "actual_config.json")]
    require(complete["status"] == summary["status"] == "PASS_CAL_ONLY_CONDITIONAL_EXCHANGE_PROFILE" and complete["actual_GPU"] is summary["actual_GPU"] is True and
            complete["calibration_sources"] == 32 and complete["actual_events"] == 896 and complete["new_model_forwards"] == 0 and complete["profile_sha256"] == BRPE_PROFILE_SHA,
            "No failed/unrun calibrator may supply a deployment profile")
    require(sha(BRPE_PROOF / "summary.json") == complete["summary_sha256"] == proof["summary_sha256"] and sha(BRPE_PROOF / "complete.json") == proof["complete_sha256"], "Actual selector summary closure differs")
    require(sha(BRPE_PROOF / "source_inventory.json") == launch["source_inventory_sha256"] and sha(BRPE_PROOF / "actual_job.json") == launch["job_sha256"] and
            launch["name"] == job["name"] == proof["producer"] and type(launch["pid"]) is int and launch["pid"] > 0 and canonical(cfg) == summary["source_provenance"]["configuration_canonical_sha256"], "Actual selector producer binding differs")
    driver = "code/round9_exchange_calibrate_v2.py"
    require(inventory[driver] == BRPE_CAL_SOURCE_SHA, "Actual selected source is not fixed F64 calibrator v2")
    matches = [p for p in files if p.name == "round9_exchange_calibrate_v2.py"]
    require(matches and sha(matches[0]) == BRPE_CAL_SOURCE_SHA and inventory["code/round9_precision_exchange_v1.py"] == BRPE_HELPER_SHA, "Selector/helper bytes absent")
    validate_calibration_metadata()
    return {"profile": profile, "files": files + [receipt] + [BRPE_PROOF / x for x in ("independent_remote_inspection.json", "independent_remote_rehash_after_fetch.json", "derived_source_before.json")], "proof": proof}


def validate_inference(raw):
    raw = resolve(raw); m, c = load(raw / "manifest.json"), load(raw / "complete.json")
    require(m.get("runner_schema") == SCHEMA and m["schema"] == "baseline_matrix.raw_inference.v1", "Unknown custom producer; no SourceRun fallback")
    p = load(raw / "source_retrieval_receipt_round9_v2.json")
    require(p["status"] == "PASS_INDEPENDENT_CUSTOM_DRIVER_SOURCE_INSPECTION" and p["SourceRun_admission_established"] is False, "R9 source inspection receipt required")
    entries, files = check_files(raw, p)
    driver = p["actual_driver"]
    require(driver in DRIVERS and sha(raw / "source_driver.py") == DRIVERS[driver] == m["source_hashes"][driver], "Unreviewed actual driver source")
    require(m["source_hashes"] == expected_sources(driver), "Actual R9 numeric dependency epoch not approved")
    job, launch, cfg, inventory = [load(raw / name) for name in ("source_job.json", "launch_receipt.json", "source_config.json", "source_inventory.json")]
    require(p["producer_name"] == job["name"], "Independent source receipt producer must equal actual job")
    require(cfg == m["config"] and sha(raw / "source_config.json") == m["config_sha256"] and
            sha(raw / "source_job.json") == launch["job_sha256"] and sha(raw / "source_inventory.json") == launch["source_inventory_sha256"], "Actual config/job/inventory source chain differs")
    root = job["argv"][job["argv"].index("--output") + 1].split("/results/", 1)[0]
    snapshot = root + "/jobs/" + job["name"] + ".source"
    executed = [snapshot + x[len(root):] if x.startswith(root + "/code/") and x.endswith(".py") else x for x in job["argv"]]
    require(type(launch["pid"]) is int and launch["pid"] > 0 and launch["name"] == job["name"] and
            launch["argv_executed"] == executed and launch["source_snapshot"] == m["runtime_source_root"] == snapshot and
            launch["runtime_python"] == m["runtime_executable"] == job["argv"][0], "Actual R9 producer/runtime/argv differs")
    require(driver == PurePosixPath(executed[2]).relative_to(snapshot).as_posix(), "Source receipt driver is not executed entry")
    for name, digest in m["source_hashes"].items():
        require(inventory[name] == digest and entries["source_files/" + name]["sha256"] == digest, "Declared dependency absent from actual snapshot")
    require(m["references_read"] is False and m["prediction_cache_used"] is False and c["quality_scored"] is False and c["coverage_complete"] is True,
            "Inference must be fresh/reference-free")
    require(sha(raw / "manifest.json") == c["manifest_sha256"] and sha(raw / "records.jsonl") == c["records_jsonl_sha256"], "R9 raw closure differs")
    raw_receipt = load(raw / "retrieval_receipt.json")
    require(raw_receipt["status"] == "PASS_REMOTE_RAW_SHA256_EXACT" and raw_receipt["remote_hashes_computed_independently"] is True, "Independent raw3 SHA missing")
    anchored = {x["file"]: x for x in raw_receipt["files"]}
    require(set(anchored) == {"manifest.json", "records.jsonl", "complete.json"}, "Raw3 receipt set differs")
    for name, value in anchored.items(): require(value["sha256"] == entries[name]["sha256"] and value["bytes"] == entries[name]["bytes"], "Independent raw/source receipts disagree")
    require(cfg["schema"] == SCHEMA and cfg["candidate"]["mode"] in MODES and cfg["method"]["family"] == "trajkv_round9_research", "Candidate schema/mode differs")
    mode = cfg["candidate"]["mode"]
    if driver == "code/round9_runner_v3.py":
        require(mode in ("K-Mu", "K-Book_FZJ") and cfg["candidate"].get("diagnostic_only") is True and cfg["candidate"].get("method_frontier_eligible") is False,
                "Actual v3 controls are diagnostics, not frontier candidates")
    if driver == "code/round9_runner_v4.py":
        require(mode in BRPE_MODES and cfg["preprocessing"] == {"max_patches": 8} and cfg["variant_source"]["name"] == "r7_joint_k1_gt_cond", "v4 is only the declared whole-layer BRPE stage")
        require(cfg["candidate"]["profile_sha256"] == entries["assets/candidate_profile.json"]["sha256"] == BRPE_PROFILE_SHA and
                not any(cfg["candidate"].get(x) for x in ("book_path", "book_sha256", "theta_path", "theta_sha256")), "BRPE uses conditional profile/static high book, not fitted global64 theta")
        profile = load(raw / "assets/candidate_profile.json"); verify_brpe_profile(profile)
        require(entries["assets/candidate_high_book.json"]["sha256"] == profile["high_book_sha256"], "Actual static high book not anchored")
    require(cfg["cohort_kind"] in ("development", "formal500") and type(cfg["count"]) is int and 1 <= cfg["count"] <= 500 and cfg["input_count"] in (64, 500), "Explicit real source slice required")
    require(type(cfg["start"]) is int and cfg["start"] >= 0 and cfg["start"] + cfg["count"] <= cfg["input_count"], "Source slice outside parent cohort")
    if cfg["cohort_kind"] == "formal500": require(cfg["start"] == 0 and cfg["count"] == cfg["input_count"] == 500, "No partial formal500")
    require(c["expected_QA"] == c["recorded_QA"] == m["expected_QA"] == cfg["count"] and c["counts"] == {"PASS": cfg["count"]}, "No failed/partial run may become complete main phase")
    if cfg["cohort_kind"] == "development":
        path = ROOT / "configs/round9_inputs_v1" / PurePosixPath(cfg["inputs"]).name
        inputs = load_round9_inputs(path, MANIFEST, cfg["inputs_sha256"])
        require(sha(MANIFEST) == MANIFEST_SHA, "Frozen calibration/development manifest changed")
    else:
        from dataset_registry import load_inputs
        path = ROOT / "configs/datasets/v3" / PurePosixPath(cfg["inputs"]).name
        require(sha(path) == cfg["inputs_sha256"], "Formal500 original input changed")
        inputs = load_inputs(path)
    require(len(inputs) == cfg["input_count"] and [x["id"] for x in inputs[cfg["start"]:cfg["start"] + cfg["count"]]] == m["expected_ids"], "Actual original ordered slice differs")
    rows = []; mode = cfg["candidate"]["mode"]
    with (raw / "records.jsonl").open("rb") as stream:
        for number, line in enumerate(stream):
            r = json.loads(line); require(r["id"] == m["expected_ids"][number] and r["status"] == "PASS", "Raw order/status differs")
            if mode != "FP16":
                r7_verify_record_role_pins(cfg, r)
            if mode in BRPE_MODES: verify_brpe_row(r, m["model_spec"], mode)
            rows.append(r)
    require(len(rows) == cfg["count"], "Actual row count differs")
    if mode != "FP16": r7_verify_executed_roles(cfg, c["executed_legacy_source_sha256"])
    # Historical label defects require exact independently corrected V2 proof;
    # no producer guard is relaxed and no prediction/scoring bytes are rewritten.
    old_path = raw / "source_retrieval_receipt_round9_v1.json"; label_issue = False
    historical = []
    if old_path.exists():
        old = load(old_path); label_issue = old["producer_name"] != p["producer_name"]
        require({k: v for k, v in old.items() if k != "producer_name"} == {k: v for k, v in p.items() if k != "producer_name"}, "Corrected receipt changed more than producer label")
        historical.append(old_path)
        if label_issue:
            correction_path = raw / "PRODUCER_LABEL_CORRECTION_v1.json"; correction = load(correction_path)
            require(correction["status"] == "PASS_PRODUCER_LABEL_CORRECTED_BY_INDEPENDENT_READBACK" and
                    correction["old_receipt_sha256"] == sha(old_path) and correction["new_receipt_sha256"] == sha(raw / "source_retrieval_receipt_round9_v2.json") and
                    correction["actual_producer"] == job["name"] and correction["model_predictions_and_scores_changed"] is False and
                    old["producer_name"] == list(c["executed_legacy_source_sha256"])[-1], "Invalid historical producer-only correction")
            historical.append(correction_path)
    return {"raw": raw, "manifest": m, "complete": c, "config": cfg, "source_files": files + historical + [raw / "source_retrieval_receipt_round9_v2.json", raw / "retrieval_receipt.json"],
            "context": actual_context(m), "rows": rows, "actual_producer": job["name"], "source_receipt_producer_label_defect_preserved": label_issue,
            "SourceRun_admission": False}


def validate_union(components):
    require(components, "Empty logical point")
    first = components[0]; cfg = first["config"]
    key = {k: cfg[k] for k in ("candidate", "method", "inputs_sha256", "preprocessing", "cohort_kind", "input_count", "max_new_tokens", "question_suffix")}
    ids = []; cursor = min(x["config"]["start"] for x in components)
    for item in sorted(components, key=lambda x: x["config"]["start"]):
        right = item["config"]
        require({k: right[k] for k in key} == key and item["manifest"]["source_hashes"] == first["manifest"]["source_hashes"] and item["context"] == first["context"], "Union different algorithm/source/protocol")
        require(right["start"] == cursor, "Union overlap/gap or noncontiguous source slices")
        cursor += right["count"]; ids += item["manifest"]["expected_ids"]
    require(len(ids) == len(set(ids)) and len(ids) in (32, 64, 500), "Duplicate/wrong logical union count")
    return {"kind": "custom_original_fresh_run" if len(components) == 1 else "custom_prediction_union_not_a_new_inference_run",
            "source_runs": len(components), "QA": len(ids), "start": min(x["config"]["start"] for x in components), "ids": ids,
            "SourceRun_admission": False, "single_fresh64_claim": len(components) == 1 and len(ids) == 64}


def calibration_metadata_missing():
    required = ("manifest.json", "complete.json", "launch_receipt.json", "source_job.json", "source_inventory.json", "source_config.json", "source_files/code/round9_collect_v1.py", "INDEPENDENT_SMALL_METADATA_RETRIEVAL.json")
    return [str(CALIB_META / name) for name in required if not (CALIB_META / name).is_file()]


def validate_calibration_metadata():
    require(not calibration_metadata_missing(), "Missing actual calibration32 small proof")
    p = load(CALIB_META / "INDEPENDENT_SMALL_METADATA_RETRIEVAL.json")
    require(p["status"] == "PASS_ACTUAL_CAL32_EXECUTED_SOURCE_AND_SMALL_METADATA" and p["actual_compressed_query_sources"] == 32 and
            p["native_query_captured"] is False and p["new_live_collector_not_substituted"] is True and
            p["private_PT_WO_images_downloaded"] is False and p["private_tensor_hashes_are_producer_recorded_not_rehashed_by_this_small_fetch"] is True,
            "Wrong calibration epoch/query source or tensor availability claim")
    entries, files = check_files(CALIB_META, p)
    actual_driver = "2d709806122c8127a766747ef5d29821fb22f389238f8448f0dfd25c363a61df"
    require(p["actual_driver_sha256"] == actual_driver == entries["source_files/code/round9_collect_v1.py"]["sha256"], "Unrun native-q collector cannot replace actual compressed-q collector")
    m, c, job, launch, cfg, inventory = [load(CALIB_META / name) for name in ("manifest.json", "complete.json", "source_job.json", "launch_receipt.json", "source_config.json", "source_inventory.json")]
    require(sha(CALIB_META / "complete.json") == CALIB_COMPLETE_SHA and c["status"] == "PASS_CALIBRATION_32_REAL_MODEL_SOURCES" and
            c["expected_sources"] == m["expected_sources"] == len(c["sources"]) == 32 and c["all_28_layers_captured"] is True and
            c["actual_GPU"] is True and c["references_read"] is m["references_read"] is False, "Actual calibration collection differs")
    require(cfg == m["config"] and sha(CALIB_META / "source_config.json") == entries["source_config.json"]["sha256"] and
            cfg["inputs_sha256"] == CALIB_INPUT_SHA == entries["inputs/calibration32_inputs.jsonl"]["sha256"], "Calibration config/input identity differs")
    require(job["name"] == launch["name"] == p["producer_name"] and sha(CALIB_META / "source_job.json") == launch["job_sha256"] and
            sha(CALIB_META / "source_inventory.json") == launch["source_inventory_sha256"] and inventory["code/round9_collect_v1.py"] == actual_driver,
            "Calibration actual job/launch/source identity differs")
    expected = [launch["source_snapshot"] + "/code/round9_collect_v1.py" if x.endswith("/code/round9_collect_v1.py") else x for x in job["argv"]]
    require(launch["argv_executed"] == expected and type(launch["pid"]) is int and launch["pid"] > 0, "Calibration actual executed argv differs")
    ids = [x["id"] for x in load_round9_inputs(MANIFEST.parent / "calibration32_inputs.jsonl", MANIFEST)]
    require([x["id"] for x in c["sources"]] == ids, "Collected calibration sources differ from frozen32 order")
    return {"manifest": m, "complete": c, "files": files + [CALIB_META / "INDEPENDENT_SMALL_METADATA_RETRIEVAL.json"],
            "native_query_captured": False, "private_tensor_hashes_independently_rehashed_here": False}


def validate_fit(book_sha):
    folders = [EVIDENCE / "results/round9" / name for name in ("fit_K_MSE100_v1", "fit_K_Func100_v1")]
    matched = [p for p in folders if (p / "fit/key_book.json").is_file() and sha(p / "fit/key_book.json") == book_sha]
    require(len(matched) == 1, "No exact actual fitted-book source")
    folder = matched[0]; proof = load(folder / "EXECUTED_FIT_SOURCE_RETRIEVAL.json")
    require(proof["status"] == "PASS_REAL_GPU_FIT_ARTIFACT_AND_EXECUTED_SOURCE" and proof["actual_GPU"] is True and proof["artifact_sha256"] == book_sha, "Real GPU fit proof missing")
    _, files = check_files(folder, proof)
    m, c, book = load(folder / "manifest.json"), load(folder / "complete.json"), load(folder / "fit/key_book.json")
    require(c["status"] == "PASS_ROUND9_CALIBRATION_FIT" and c["actual_GPU"] is True and c["artifact_sha256"] == book_sha and
            m["references_read"] is False and m["calibration_sources"] == c["calibration_sources"] == 32, "Fit completion differs")
    require(book["original_book_sha256"] == BASE_BOOK and book["dimension"] == 4 and book["centroids"] == 16 and len(book["centers"]) == 16 and all(len(x) == 4 for x in book["centers"]), "Same-capacity original G4_16 basis required")
    require(m["collection_complete_sha256"] == CALIB_COMPLETE_SHA and m["query_source"] == "compressed_query_same_student_text" and c["student_state_refreshes"] == 0 and c["frozen_student_snapshot_only"] is True,
            "S0 only has compressed student q; do not relabel native queries/refresh")
    objective = book["round9_fit"]["objective"]
    require(m["config"]["kind"] == c["kind"] == {"mse": "K-MSE", "functional": "K-Func"}[objective], "Actual fit kind does not match book objective")
    provenance = book["round9_fit"]["train_provenance"]
    require(provenance["train_labels_or_development_answers_read"] is False and provenance["fixture_only"] is False and
            provenance["collection_provenance"]["complete_sha256"] == CALIB_COMPLETE_SHA, "Fit provenance differs")
    require(sha(ROOT / "configs/round9_inputs_v1/calibration32_inputs.jsonl") == CALIB_INPUT_SHA and sha(MANIFEST) == MANIFEST_SHA, "Calibration cohort changed")
    ids = [x["id"] for x in load_round9_inputs(ROOT / "configs/round9_inputs_v1/calibration32_inputs.jsonl", MANIFEST)]
    require(sorted(provenance["calibration_ids"]) == sorted(ids), "Training source IDs differ from frozen32")
    collection = validate_calibration_metadata()
    require(m["private_WO_sha256"] == collection["complete"]["wo_sha256"] and
            sorted(provenance["private_train_shards"]) == sorted([[x["file"], x["sha256"]] for x in collection["complete"]["sources"]]), "Producer-recorded calibration tensor hash linkage differs")
    inventory = load(folder / "source_inventory.json"); launch = load(folder / "launch_receipt.json"); job = load(folder / "source_job.json")
    require(sha(folder / "source_inventory.json") == launch["source_inventory_sha256"] and sha(folder / "source_job.json") == launch["job_sha256"] and
            launch["name"] == job["name"] and type(launch["pid"]) is int and launch["pid"] > 0, "Fit actual launch/source binding differs")
    for name, digest in FIT_CODES.items(): require(inventory["code/" + name] == digest == sha(folder / "source_files" / name), "Wrong executed trainer module")
    config_remote = job["argv"][job["argv"].index("--config") + 1]
    require("/configs/round9_v1/" in config_remote, "Unexpected actual fit config path")
    config_local = ROOT / ("configs/round9_v1/" + config_remote.split("/configs/round9_v1/", 1)[1])
    require(sha(config_local) == m["config_sha256"] and load(config_local) == m["config"], "Actual fitted config content unavailable by original SHA")
    callback_files = []
    # Same bytes from independently retrieved inference snapshots can witness
    # a training callback only when the actual FIT snapshot inventory pins it.
    for entry in provenance["callback_sources"].values():
        basename = PurePosixPath(entry["file"]).name; digest = entry["sha256"]
        if "/code/" in entry["file"] and ".source/code/" in entry["file"]:
            require(inventory["code/" + basename] == digest, "Fit callback not in actual snapshot inventory")
        choices = list(folder.glob("source_files/" + basename))
        for run in CURRENT_RUNS:
            path = RAW_ROOT / run / "source_files/code" / basename
            if path.is_file() and sha(path) == digest: choices.append(path)
            legacy = RAW_ROOT / run / "legacy_sources"
            choices += [x for x in legacy.glob("**/" + basename) if sha(x) == digest]
        require(choices and sha(choices[0]) == digest, "Missing exact callback source bytes: " + basename)
        callback_files.append(choices[0])
    files += [folder / "EXECUTED_FIT_SOURCE_RETRIEVAL.json", config_local] + collection["files"] + callback_files
    return {"folder": folder, "files": files, "book": book, "fit_manifest": m, "fit_complete": c,
            "query_source": "compressed_query_same_student_text", "private_WO_and_sample_tensors_archived": False,
            "exact_fit_rerun_requires_external_model_and_private_calibration_tensors": True}


def validate_candidate_fit(candidate):
    fit = validate_fit(candidate["book_sha256"])
    if candidate["mode"] in ("K-MSE", "K-Func"):
        expected = {"K-MSE": "mse", "K-Func": "functional"}[candidate["mode"]]
        require(fit["book"]["round9_fit"]["objective"] == expected and fit["fit_manifest"]["config"]["kind"] == candidate["mode"], "Candidate label does not match actual book objective/fit kind")
    return fit


def contract_from_arguments(args):
    require(args.contract is not None, "Explicit Round9 publication contract required")
    path = resolve(args.contract); value = load(path)
    require(value["schema"] == "baseline_matrix.public_phase_contract.v2" and value.get("family") == "round9_custom_producer_v3", "Explicit R9 producer contract required")
    return value, path


def validate_phase(contract):
    from score import validate_raw
    require(contract.get("family") == "round9_custom_producer_v3" and contract.get("completion_scope") == "declared_matrix_only", "No canonical/R7 fallback or whole-goal claim")
    matrix_path = resolve(contract["matrix"]); matrix = load(matrix_path); declared = matrix["rows"]
    require(declared and len({x["config"] for x in declared}) == len(declared), "Empty/duplicate actual source components")
    require(len(declared) == 18 and {x["cell"] for x in declared} == set(STAGE_RUNS) and sum(x["count"] for x in declared) == 704,
            "Whole v3 scope is original14/576 plus four32 BRPE runs, not a truncated success subset")
    artifacts = []; missing = []; observed = []; points = defaultdict(list)
    for row in declared:
        raw, scored = resolve(row["view"]), resolve(row["scored"])
        if not (raw / "complete.json").is_file() or not (scored / "summary.json").is_file() or not (raw / "source_retrieval_receipt_round9_v2.json").is_file():
            missing.append({"cell": row["cell"], "missing": "actual_raw_source_and_score"}); continue
        item = validate_inference(raw); cfg = item["config"]
        require(cfg["count"] in (32, 64, 500), "Small diagnostics may only be completed support, never main quality cells")
        require(cfg == load(resolve(row["config"])) and sha(resolve(row["config"])) == row["config_sha256"] and row["count"] == cfg["count"], "Declared source config/slice changed")
        root = ROOT / ("configs/round9_inputs_v1" if cfg["cohort_kind"] == "development" else "configs/datasets/v3")
        refs = root / PurePosixPath(cfg["inputs"]).name.replace("_inputs.jsonl", "_refs.jsonl")
        validated = validate_raw(raw, refs); summary, failures = validate_scored(scored, validated)
        require(summary["model_id"] == "M1" and summary["method"] == cfg["method"]["id"] and summary["expected_run_questions"] == cfg["count"], "Source/scored model/count differs")
        if failures or summary["status"] != "PASS_SCORED_COMPLETE":
            missing.append({"cell": row["cell"], "missing": "all_expected_QA_pass"}); continue
        if cfg["candidate"].get("book_sha256"): validate_candidate_fit(cfg["candidate"])
        if cfg["candidate"]["mode"] in BRPE_MODES: validate_brpe_calibration()
        require(cfg["candidate"]["mode"] not in ("V-FZ", "V-RE", "FB0", "FB1", "COMBO") or cfg["candidate"].get("theta_sha256") is None,
                "Trained V/COMBO artifact closure must be explicitly registered before publication")
        diagnostic = cfg["candidate"].get("diagnostic_only", False)
        artifact = {"cell": row["cell"], "config": resolve(row["config"]), "raw": raw, "scored": scored,
                    "kind": "custom_R9_diagnostic_not_frontier" if diagnostic else "custom_R9_original_fresh_inference_run", "expected_QA": cfg["count"], "raw_status": "PASS_RAW_COMPLETE", "origin_counts": {"fresh": cfg["count"]}}
        artifacts.append(artifact); observed.append({"cell": row["cell"], "questions": cfg["count"], "kind": artifact["kind"], "raw_status": "PASS_RAW_COMPLETE", "score_status": summary["status"]})
        points[row["logical_point"]].append(item)
    unions = {key: validate_union(items) for key, items in points.items()}
    contexts = {}
    for items in points.values():
        for item in items:
            cfg = item["config"]; key = (cfg["inputs_sha256"], canonical(cfg["preprocessing"]))
            if key in contexts: require(contexts[key] == item["context"], "Methods differ in actual model/processor/generation/runtime")
            contexts[key] = item["context"]
    phase = {"status": "PASS_DECLARED_MATRIX_COMPLETE" if not missing else "INCOMPLETE_DECLARED_MATRIX_PREVIEW", "phase_label": contract["phase_label"],
             "family": "round9_custom_producer_v3", "coverage_scope": contract["coverage_scope"], "completion_scope": "declared_matrix_only",
             "whole_external_library_completion_assessed": False, "declared_matrix_source_sha256": sha(matrix_path),
             "expected_cells": len(declared), "verified_scored_cells": len(observed), "closed_artifact_cells": len(artifacts),
             "expected_questions": sum(x["count"] for x in declared), "verified_scored_questions": sum(x["questions"] for x in observed), "missing": missing, "observed": observed,
             "logical_points": unions, "component_count_is_not_logical_point_count": True, "SourceRun_admission": False,
             "development_Info_is_true_train_not_official_test": True}
    diagnostics = sum(x["expected_QA"] for x in artifacts if x["kind"] == "custom_R9_diagnostic_not_frontier")
    phase.update(expected_source_runs=18, expected_all_answer_events=704, expected_diagnostic_answer_events=128,
                 verified_diagnostic_answer_events=diagnostics, verified_non_diagnostic_answer_events=phase["verified_scored_questions"] - diagnostics,
                 diagnostics_are_not_method_frontier_points=True)
    return phase, artifacts, matrix_path


def source_closure(artifacts):
    nodes = []; fits = {}
    for artifact in artifacts:
        item = validate_inference(artifact["raw"]); cfg = item["config"]; files = item["source_files"]
        if cfg["candidate"]["mode"] in BRPE_MODES:
            files += validate_brpe_calibration()["files"] + validate_calibration_metadata()["files"]
        if cfg["candidate"].get("book_sha256"):
            fit = validate_candidate_fit(cfg["candidate"]); files += fit["files"]
            fits[cfg["candidate"]["book_sha256"]] = {"folder": fit["folder"].relative_to(EVIDENCE).as_posix(),
                "callback_sources": {name: entry["sha256"] for name, entry in fit["book"]["round9_fit"]["train_provenance"]["callback_sources"].items()}}
        files += [MANIFEST, MANIFEST.parent / "source_lock.json", MANIFEST.parent / "calibration32_inputs.jsonl"]
        for key in ("disjoint_proof", "info_source_locators", "pinned_info_sources"):
            value = load(MANIFEST)[key]; path = resolve(value["path"])
            require(sha(path) == value["sha256"], "Frozen data source/disjoint proof changed")
            files.append(path)
        files += [p for p in MANIFEST.parent.glob("*_inputs.jsonl")] + [p for p in MANIFEST.parent.glob("*_refs.jsonl")]
        # All source identities are original snapshots; current source code is
        # never substituted for the executed 435b/cebb/2d7098/4809 epochs.
        require(all(p.suffix not in (".pt", ".png", ".jpg", ".safetensors") for p in files), "Private tensors/weights/images forbidden")
        nodes.append({"path": artifact["raw"], "kind": "custom_R9_original_fresh_inference_run", "files": list(dict.fromkeys(files))})
    policy = {"schema": "trajkv.public_round9_source_policy.v2", "SourceRun_admission": False, "runner_schema": SCHEMA,
              "drivers": DRIVERS, "source_epochs": {name: expected_sources(name) for name in DRIVERS}, "fits": fits,
              "input_manifest_file": MANIFEST.relative_to(ROOT).as_posix(), "input_manifest_sha256": MANIFEST_SHA,
              "calibration_metadata_root": CALIB_META.relative_to(EVIDENCE).as_posix(), "calibration_complete_sha256": CALIB_COMPLETE_SHA,
              "calibration_query_source": "compressed_query_same_student_text", "native_query_captured": False,
              "BRPE_profile_sha256": BRPE_PROFILE_SHA, "BRPE_calibration_root": BRPE_PROOF.relative_to(EVIDENCE).as_posix(),
              "BRPE_calibrator_sha256": BRPE_CAL_SOURCE_SHA, "BRPE_helper_sha256": BRPE_HELPER_SHA,
              "private_tensor_bytes_archived": False, "training_tensor_hashes_producer_recorded_not_refetched": True}
    if POLICY.exists(): require(canonical(load(POLICY)) == canonical(policy), "Use a new policy version/cutoff for a changed R9 phase")
    else: write(POLICY, policy)
    for node in nodes: node["files"].append(POLICY)
    return nodes


def prepare_bundle(output, runs):
    output = resolve(output); require(not output.exists(), "Use unique R9 publication bundle output")
    require(len(runs) == 18 and set(runs) == set(STAGE_RUNS), "Explicit all18 scope required; no dropping missing BRPE outputs")
    items = []
    for name in runs:
        raw = RAW_ROOT / name
        if (raw / "complete.json").is_file() and (raw / "source_retrieval_receipt_round9_v2.json").is_file() and (SCORE_ROOT / name / "summary.json").is_file():
            item = validate_inference(raw); cfg_path = raw / "source_config.json"; dataset = item["rows"][0]["dataset"]
        else:
            require(name in BRPE_RUNS, "Original14 closed sources cannot be missing")
            job = load(ROOT / "configs/jobs" / ("bm_R9_" + name + ".json")); remote_cfg = job["argv"][job["argv"].index("--config") + 1]
            cfg_path = ROOT / ("configs/" + remote_cfg.split("/configs/", 1)[1]); cfg = load(cfg_path)
            require(cfg["count"] == 32 and cfg["start"] == 0 and cfg["candidate"]["profile_sha256"] == BRPE_PROFILE_SHA, "Pending config is not the exact declared profile slice")
            dataset = "textvqa" if "_textvqa_" in name else "infovqa"; item = {"raw": raw, "config": cfg, "pending": True}
        item.update(config_path=cfg_path, dataset=dataset); items.append(item)
    declared = []
    for item in items:
        cfg = item["config"]; candidate = cfg["candidate"]
        point = canonical({"candidate": candidate, "dataset": item["dataset"], "inputs_sha256": cfg["inputs_sha256"], "cap": cfg["preprocessing"], "cohort": cfg["cohort_kind"]})[:20]
        declared.append({"cell": item["raw"].name, "logical_point": point, "model": "M1", "dataset": item["dataset"], "count": cfg["count"],
                         "config": str(item["config_path"]), "config_sha256": sha(item["config_path"]), "view": str(item["raw"]), "scored": str(SCORE_ROOT / item["raw"].name)})
    # Metadata is not a completion declaration. Validation below must actually
    # close every listed component and all parameter/source dependencies.
    output.mkdir(parents=True); matrix = output / "DECLARED_ACTUAL_SOURCE_COMPONENTS.json"
    write(matrix, {"schema": "baseline_matrix.fixed_matrix.v1", "rows": declared, "source_runs": len(declared), "logical_points": len({x["logical_point"] for x in declared}), "components_not_fictitious_fresh_unions": True})
    contract = {"schema": "baseline_matrix.public_phase_contract.v2", "family": "round9_custom_producer_v3", "phase_label": "ROUND9_SELECTED_ACTUAL_COMPONENTS",
                "matrix": str(matrix), "view_root": str(RAW_ROOT), "score_root": str(SCORE_ROOT), "completion_scope": "declared_matrix_only", "completion_policy": "all_expected_QA_pass",
                "coverage_scope": "Only explicitly declared real R9 source components and separately derived logical unions; true train/validation development, not whole Round9 completion or official test."}
    write(output / "PHASE_CONTRACT.json", contract)
    return contract


def prepare_diagnostic_support(output, runs):
    """Known-runner K-Mu/FZJ small controls, explicitly zero main-quality QA."""
    files = {}; directories = []
    for name in runs:
        item = validate_inference(RAW_ROOT / name); cfg = item["config"]
        require(cfg["candidate"]["mode"] in ("K-Mu", "K-Func_FZJ") and cfg["cohort_kind"] == "development", "Only explicit reviewed K controls here")
        selected = item["source_files"]
        if cfg["candidate"].get("book_sha256"): selected += validate_candidate_fit(cfg["candidate"])["files"]
        for path in selected: files[str(path)] = {"path": str(path), "sha256": sha(path), "role": "actual_R9_K_control_source"}
        directories.append({"path": str(item["raw"]), "QA_events": cfg["count"], "mode": cfg["candidate"]["mode"]})
    value = {"schema": "baseline_matrix.public_support_files.v1", "label": "R9_K_Mu_and_generic_FZJ_actual_small_controls", "completed_evidence_only": True,
             "primary_quality_QA_added": 0, "quality_QA_added": 0, "scope": "Explicit real same-model small controls; not32/64/500 quality claims.",
             "counts": {"diagnostic_producers": directories, "primary_quality_QA_added": 0}, "files": list(files.values())}
    write(output, value); return value


def prepare_brpe_gpu_support(output):
    directory = EVIDENCE / "results/round9/exchange_GPU2Nv_v1_public_proof_v1"
    review = load(directory / "INDEPENDENT_REVIEW.json")
    require(review["status"] == "PASS_ACTUAL_GPU2NV_FIELDS_READERS_AND_COMPLETE_BYTES_INDEPENDENTLY_REVIEWED" and review["actual_GPU"] is True and
            review["all28_reader_comparisons"] == 168 and review["model_quality_QA_added"] == 0 and review["source_driver_sha256"] == BRPE_HELPER_SHA and
            review["profile_sha256"] == BRPE_PROFILE_SHA, "GPU controls need their actual source/profile, not CPU fake owners")
    proof = load(directory / "FETCH_RECEIPT.json"); _, selected = check_files(directory, proof)
    selected += [directory / name for name in ("FETCH_RECEIPT.json", "INDEPENDENT_REMOTE_SHA_REHASH.json", "INDEPENDENT_REVIEW.json", "verify.py")]
    value = {"schema": "baseline_matrix.public_support_files.v1", "label": "BRPE_actual_2Nv_3modes_28layer_reader_and_storage_controls",
             "completed_evidence_only": True, "primary_quality_QA_added": 0, "quality_QA_added": 0,
             "scope": "Actual 2Nv x3modes x28 reader/field bitwise controls and real storage; selected L14 original sole K residual is dropped, so upgraded retained-KRE GPU path is empty, not a nonempty branch test.",
             "counts": {"actual_GPU_geometries": 2, "mode_cases": 6, "reader_field_comparisons": 168, "primary_quality_QA_added": 0,
                        "selected_layer_retained_K_residuals": 0, "nonempty_upgraded_layer_KRE_GPU_path_covered": False},
             "files": [{"path": str(path), "sha256": sha(path), "role": "completed_BRPE_numeric_control"} for path in dict.fromkeys(selected)],
             "collection_performed_new_inference": False}
    write(output, value); return value


def inspect_current(output):
    rows = []; errors = []; pending = []
    for name in STAGE_RUNS:
        if not (RAW_ROOT / name / "complete.json").is_file() or not (SCORE_ROOT / name / "summary.json").is_file() or not (RAW_ROOT / name / "source_retrieval_receipt_round9_v2.json").is_file():
            pending.append(name); continue
        try:
            item = validate_inference(RAW_ROOT / name); cfg = item["config"]
            rows.append({"run": name, "QA": cfg["count"], "start": cfg["start"], "mode": cfg["candidate"]["mode"], "cohort_kind": cfg["cohort_kind"],
                         "source_driver": item["manifest"]["source_hashes"], "producer_label_defect": item["source_receipt_producer_label_defect_preserved"]})
        except Exception as exc: errors.append({"run": name, "error": str(exc)})
    result = {"status": "WAIT_MISSING_DECLARED_BRPE_SOURCE_COMPONENTS" if pending else "ALL18_METADATA_PRESENT_DEEP_PHASE_VALIDATION_STILL_REQUIRED",
              "actual_inference_sources": rows, "errors": errors, "missing_calibration_metadata": calibration_metadata_missing(),
              "pending_source_runs": pending, "declared_source_runs": 18, "declared_all_answer_events": 704,
              "stage_or_public_archive_created": False, "GPU": False, "network": False, "SourceRun_admission": False}
    write(output, result); return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--inspect-current", type=Path); p.add_argument("--prepare-bundle", type=Path); p.add_argument("--prepare-scope", type=Path); p.add_argument("--runs", nargs="+"); a = p.parse_args()
    if a.inspect_current:
        value = inspect_current(a.inspect_current)
        print(json.dumps({"status": value["status"], "actual_inference_runs": len(value["actual_inference_sources"]), "errors": value["errors"], "missing_small_files": len(value["missing_calibration_metadata"])}))
    else:
        require(a.prepare_bundle is not None or a.prepare_scope is not None, "Explicit inspect or preparation output required")
        require(a.prepare_scope is not None or a.runs, "Explicit all18 source list required")
        print(json.dumps(prepare_bundle(a.prepare_scope or a.prepare_bundle, STAGE_RUNS if a.prepare_scope else a.runs)))


if __name__ == "__main__": main()
