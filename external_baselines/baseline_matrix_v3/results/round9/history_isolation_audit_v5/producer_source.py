"""Read-only historical-source audit and score-blind future-confirm reservation.

Frozen160, old results and official500 files are never changed. No raw model
answers, GT correctness or compression losses are read for source selection.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "TrajKV"
E = Path("${EVIDENCE_ROOT}")
WORK = E / "results/round9/history_isolation_and_confirm_reserve_v1"
SEED = "trajkv-round9-post-exchange-unobserved-confirm32-v1:20261121"
sys.dont_write_bytecode = True


def require(ok, text):
    if not ok: raise ValueError("R9_RESERVE: " + text)


def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""): h.update(chunk)
    return h.hexdigest()
def rows(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]
def priority(kind, value): return hashlib.sha256((SEED + "\0" + kind + "\0" + str(value)).encode()).hexdigest()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as out: json.dump(value, out, ensure_ascii=False, sort_keys=True, indent=2); out.write("\n")
def write_rows(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as out:
        for x in values: out.write(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def old_project_path(value):
    value = str(value).replace("\\", "/")
    if "/TrajKV/" in value: return OLD / value.split("/TrajKV/", 1)[1]
    path = Path(value)
    return path if path.is_absolute() else OLD / path


LEGACY_ALIASES = {}


def identity(row, canonical_by_sha):
    ds = {"docvqa_grouped": "docvqa", "infographicvqa": "infovqa"}.get(row.get("dataset"), row.get("dataset"))
    require(ds in ("docvqa", "textvqa", "infovqa", "chartqa"), "Unsupported identity dataset")
    digest = row.get("original_image_sha256")
    if not digest:
        values = list(row.get("hashes", []))
        values += [str(row.get(k, ""))[7:] for k in ("image_id", "source_image_id") if str(row.get(k, "")).startswith("sha256:")]
        values = [x for x in values if re.fullmatch("[0-9a-f]{64}", str(x))]
        require(values and len(set(values)) == 1, "Missing/conflicting historical original-byte SHA")
        digest = values[0]
    canonical = canonical_by_sha.get((ds, digest))
    alias_method = None
    if canonical is None:
        index = row.get("source_row_index")
        paths = row.get("images", [])
        if index is None and paths:
            match = re.search(r"(?:textvqa|docvqa)_(\d+)\.[a-zA-Z]+$", str(paths[0]).replace("\\", "/"))
            if match: index = int(match[1])
        if (ds, index) in LEGACY_ALIASES:
            canonical = LEGACY_ALIASES[ds, index]; alias_method = "frozen_legacy_row_identity_map_not_byte_SHA_equality"
    if canonical:
        source, cluster = canonical["source_id"], canonical["source_cluster_id"]
    elif ds == "docvqa":
        document = row.get("source_document_id")
        if document is None:
            # Retain exact content exposure; do not silently drop early lists
            # or pretend a content hash is a recovered document identifier.
            source = "UNRESOLVED_SOURCE:" + digest; cluster = "UNRESOLVED_DOC_PARENT:" + digest
        else:
            source = str(document) + "/" + str(row.get("source_page_id", "unknown")); cluster = "docvqa:ucsf:" + str(document)
    else:
        source = row.get("source_original_image_id", row.get("source_id"))
        if source is None or str(source).startswith("sha256:"): source = row.get("source_page_id")
        source = "UNRESOLVED_SOURCE:" + digest if source is None else str(source)
        cluster = ("UNRESOLVED_IMAGE_PARENT:" if source.startswith("UNRESOLVED_") else ds + ":image:") + source
    return {"dataset": ds, "sha256": digest, "source_id": str(source), "cluster": cluster, "id": str(row.get("id", row.get("session_id", "unknown"))),
            "source_identity_recovered": not str(source).startswith("UNRESOLVED_"), "cluster_identity_recovered": not cluster.startswith("UNRESOLVED_")}


def pool_role(path):
    name = path.as_posix().lower()
    if "standard5000" in name: return "reserved_full_validation_pool_not_alone_execution_proof"
    if any(w in name for w in ("confirm", "confirmation", "future")): return "historical_confirm_reservation"
    if any(w in name for w in ("calib", "train", "select", "dev", "pilot", "context", "mechanism")): return "historical_calibration_tuning_reservation"
    return "historical_reserved_pool_unclassified_not_proven_execution"


def sets(values):
    return {"sha256": {x["sha256"] for x in values}, "source_id": {(x["dataset"], x["source_id"]) for x in values}, "cluster": {x["cluster"] for x in values}}


def audit(output):
    require(not output.exists(), "Use a new immutable audit output")
    canonical = []
    formal = []
    for ds in ("textvqa", "infovqa", "docvqa", "chartqa"):
        canonical += rows(ROOT / f"configs/datasets/v3/{ds}_full_inputs.jsonl")
        formal += rows(ROOT / f"configs/datasets/v3/{ds}_p2_inputs.jsonl")
    by_sha = {(r["dataset"], r["original_image_sha256"]): r for r in canonical}
    # Early legacy images were re-encoded JPEGs. Resolve source identity using
    # original frozen row/order proof, while keeping their different byte SHA.
    doc_proof = read(OLD / "audit/a2/next_docvqa_source_split_v1.json")
    doc_legacy = OLD.parent / "benchmark_samples/docvqa_samples.jsonl"
    require(sha(doc_legacy) == doc_proof["legacy1000_sha256"] and doc_proof["legacy_order_matches"] == 1000, "Frozen Doc legacy row mapping changed")
    doc_rows = {r["original_split_row"]: r for r in canonical if r["dataset"] == "docvqa"}
    text_legacy = OLD.parent / "benchmark_samples/textvqa_samples.jsonl"
    text_proof = read(OLD / "audit/a2/textvqa_balanced128_membership.json")
    require(sha(text_legacy) == text_proof["source_sha256"], "Frozen Text legacy identity mapping changed")
    text_rows = {r["source_id"]: r for r in canonical if r["dataset"] == "textvqa"}
    decoder = json.JSONDecoder()
    for index, line in enumerate(text_legacy.read_text(encoding="utf-8-sig").splitlines()):
        match = re.search(r'"image_id"\s*:\s*', line); require(match is not None, "Legacy Text original image ID missing")
        source_id = str(decoder.raw_decode(line, match.end())[0]); require(source_id in text_rows, "Legacy Text image ID not in canonical source")
        LEGACY_ALIASES["textvqa", index] = text_rows[source_id]
    for index in range(1000): LEGACY_ALIASES["docvqa", index] = doc_rows[index]
    current = {name: rows(ROOT / "configs/round9_inputs_v1" / (name + "_inputs.jsonl")) for name in ("calibration32", "textvqa_dev64", "infovqa_dev64")}
    current_id = {key: [identity(x, by_sha) for x in value] for key, value in current.items()}
    registry = {}; inventory = []; role_rows = defaultdict(list); errors = []
    # Decode only list identities, not reference/annotation files or raw answers.
    for path in sorted((OLD / "configs/a2").rglob("*.json")):
        if any(word in path.name.lower() for word in ("references", "refs")): continue
        try: value = read(path)
        except (ValueError, OSError): continue
        if not (isinstance(value, list) and value and isinstance(value[0], dict) and value[0].get("dataset") in ("docvqa_grouped", "textvqa", "infovqa", "infographicvqa")): continue
        try: normalized = [identity(r, by_sha) for r in value]
        except ValueError as exc:
            errors.append({"path": str(path), "error": str(exc)}); continue
        role = pool_role(path.relative_to(OLD)); registry[path.resolve()] = (value, normalized)
        role_rows[role] += normalized
        inventory.append({"path": str(path), "sha256": sha(path), "kind": role, "QA_rows": len(value), "unique_images": len({x["sha256"] for x in normalized})})
    executed = []; only_fp16 = []
    for path in sorted((OLD / "results").rglob("manifest.json")):
        try: m = read(path)
        except (ValueError, OSError): continue
        cfg = m.get("config", {}) if isinstance(m, dict) else {}
        if not isinstance(cfg, dict): continue
        arguments = cfg.get("arguments", {}); sample = arguments.get("samples", cfg.get("samples"))
        if not isinstance(sample, str): continue
        sample_path = old_project_path(sample).resolve()
        if sample_path not in registry: continue
        original, normalized = registry[sample_path]; lookup = {r.get("id", r.get("session_id")): n for r, n in zip(original, normalized)}
        selected = m.get("selected_ids")
        if not isinstance(selected, list) or not selected: continue
        require(set(selected) <= set(lookup), "Actual manifest source IDs absent from locked source list")
        chosen = [lookup[x] for x in selected]
        variants = cfg.get("variants", m.get("executed_candidate_variants", []))
        methods = [x.get("method") for x in variants if isinstance(x, dict)]
        capture = isinstance(cfg.get("codec"), dict)
        if not methods:
            methods = ["compressed_codec_capture"] if capture else ["unspecified_reserved_not_proven_FP16"]
        fp16_only = all(x in ("full", "fp16", "FP16") for x in methods)
        complete_path = path.parent / "complete.json"; complete = read(complete_path) if complete_path.exists() else None
        if fp16_only: kind = "actual_FP16_only_not_compression_tuning"
        elif capture: kind = "actual_compressed_calibration_capture_not_task_quality"
        elif methods == ["unspecified_reserved_not_proven_FP16"]: kind = "actual_unspecified_source_reservation_not_method_proof"
        else: kind = "actual_completed_compression_evaluation" if complete is not None else "actual_compression_attempt_reservation_not_complete_proof"
        role_rows[kind] += chosen
        item = {"manifest": str(path), "manifest_sha256": sha(path), "source_list": str(sample_path), "selected_QA": len(selected),
                "selected_unique_images": len({x["sha256"] for x in chosen}), "method_kinds": sorted(set(methods)), "classification": kind,
                "complete_file": str(complete_path) if complete is not None else None, "complete_sha256": sha(complete_path) if complete is not None else None,
                "complete_status": complete.get("status") if isinstance(complete, dict) else None}
        (only_fp16 if fp16_only else executed).append(item)
    intersections = []
    for name, cohort in current_id.items():
        for role, historical in role_rows.items():
            old = sets(historical)
            for ds in sorted({x["dataset"] for x in cohort}):
                values = [x for x in cohort if x["dataset"] == ds]
                intersections.append({"current": name, "dataset": ds, "historical_role": role, "current_images": len(values),
                    **{kind + "_intersection": len(sets(values)[kind] & old[kind]) for kind in old},
                    "historical_unresolved_source_images": len({x["sha256"] for x in historical if x["dataset"] == ds and not x.get("source_identity_recovered", True)}),
                    "source_cluster_counts_are_lower_bounds_if_unresolved": any(not x.get("cluster_identity_recovered", True) for x in historical if x["dataset"] == ds),
                    "intersecting_current_ids_by_raw_SHA": [x["id"] for x in values if x["sha256"] in old["sha256"]]})
    # Current official-full native scores are exposure to FP16 outputs, not an
    # assertion that compression was tuned on all those images.
    fp16_benchmark = {ds: [identity(x, by_sha) for x in canonical if x["dataset"] == ds] for ds in ("textvqa", "infovqa", "docvqa", "chartqa")}
    fp16_overlap = [{"current": key, "FP16_benchmark_image_overlap": len(sets(value)["sha256"] & sets([x for group in fp16_benchmark.values() for x in group])["sha256"]),
                     "counts_as_compression_tuning": False} for key, value in current_id.items()]
    deferred_path = OLD / "audit/a2/standard5000_deferred_20261114.json"
    deferred = read(deferred_path)
    require(deferred["status"] == "DEFERRED" and len(deferred["workers"]) == 2 and
            all(x["status"] == "DEFERRED_BEFORE_GPU" and x["model_children"] == 0 and x["free_answers_generated"] == 0 for x in deferred["workers"]),
            "Cannot exclude the full5000 catalog from exposure without actual deferred-before-GPU evidence")
    not_observed_roles = {"actual_FP16_only_not_compression_tuning", "reserved_full_validation_pool_not_alone_execution_proof"}
    exclusion = [x for key, value in role_rows.items() if key not in not_observed_roles for x in value]
    exclusion += [identity(x, by_sha) for x in formal] + [x for value in current_id.values() for x in value]
    excluded = sets(exclusion)
    candidates = {}
    for ds in ("textvqa", "infovqa"):
        eligible = [r for r in canonical if r["dataset"] == ds and r["original_image_sha256"] not in excluded["sha256"] and
                    r["source_cluster_id"] not in excluded["cluster"] and (ds, r["source_id"]) not in excluded["source_id"]]
        candidates[ds] = {"eligible_validation_questions": len(eligible), "eligible_validation_sources": len({x["source_cluster_id"] for x in eligible}),
                          "exclusion_is_conservative_includes_registered_but_unproven_executed_reservations": True}
    text_meta = read(ROOT / "audit/dataset_sources/lmms-lab-encoder__textvqa_info_v1.json")
    train = [x for x in text_meta["siblings"] if x["rfilename"].startswith("data/train-") and x["rfilename"].endswith(".parquet")]
    chosen_shard = min(train, key=lambda x: priority("TextTRAIN_shard", x["rfilename"]))
    result = {"schema": "trajkv.round9.historical_isolation_audit.v1", "selection_seed": SEED, "current_frozen160_modified": False,
        "historical_sample_list_inventory": inventory, "actual_compressed_manifests": executed, "actual_old_FP16_only_manifests": only_fp16,
        "legacy_reencoded_source_identity_maps": {"Doc_row_order_proof": str(OLD / "audit/a2/next_docvqa_source_split_v1.json"),
            "Doc_legacy_SHA": sha(doc_legacy), "Text_legacy_SHA": sha(text_legacy), "no_reference_or_model_score_used": True,
            "same_source_different_encoded_SHA_not_claimed_identical_bytes": True},
        "full5000_plan_was_not_executed": {"path": str(deferred_path), "sha256": sha(deferred_path), "workers": deferred["workers"],
            "catalog_not_counted_as_compression_tuning": True},
        "intersections": intersections, "current_native_full_FP16_benchmark_overlap_not_tuning": fp16_overlap, "validation_confirmation_availability": candidates,
        "history_metadata_errors": errors, "notes": ["Text64 deliberately reuses historical128 source pool; never all-history independent",
            "Current MSE second32 is observed development recheck (Text0, Info-4.08 reported); cannot be a blind confirm for a later exchange candidate",
            "Info current train72 is new relative to old Info validation; original validation500 covers all val images",
            "Reservation != proven completed inference; compressed evaluation != automatically a causal account of hyperparameter tuning",
            "OnlyFP16 benchmark outputs are disclosed separately and do not make an image compression-tuned"],
        "TextTRAIN_download_options": {"repo": text_meta["id"], "revision": text_meta["sha"], "all_shards": len(train), "all_bytes": sum(x["size"] for x in train),
            "predeclared_single_hash_shard": chosen_shard, "single_shard_is_a_restricted_pool_not_full_train_representativeness": True},
        "confirmation_selection_rule": "sha256(seed NUL dataset/source NUL sourceID); independent sha256(seed NUL QA NUL sourceID/questionID); no score/correctness selection",
        "new_GPU_calls": 0, "remote_writes_or_Git": False, "new_confirmation_inputs_created": False}
    output.mkdir(parents=True); write(output / "HISTORICAL_INTERSECTIONS.json", result)
    write_rows(output / "EXCLUSION_IDENTITIES.jsonl", sorted({(x["dataset"], x["sha256"], x["cluster"]): x for x in exclusion}.values(), key=lambda x: (x["dataset"], x["cluster"], x["sha256"])))
    write_rows(output / "CURRENT160_IDENTITIES.jsonl", [dict(x, cohort=key) for key, values in current_id.items() for x in values])
    write_rows(output / "HISTORICAL_POOL_IDENTITIES.jsonl", [dict(x, role=key) for key, values in role_rows.items() for x in {(z["dataset"], z["sha256"], z["cluster"]): z for z in values}.values()])
    write(output / "RECEIPT.json", {"status": "PASS_READONLY_HISTORY_IDENTITIES_NOT_NEW_CONFIRMATION", "audit_sha256": sha(output / "HISTORICAL_INTERSECTIONS.json"),
        "script_sha256": sha(__file__), "sample_lists": len(inventory), "actual_compressed_manifests": len(executed), "metadata_errors": len(errors), "GPU": False, "raw_predictions_read": False})
    return result


def reserve(audit_dir, output):
    """Reserve32/source and QA double-hash, without reading any model score."""
    import pyarrow.parquet as pq
    from PIL import Image
    sys.path.insert(0, str(ROOT / "ops"))
    from remote_control import REMOTE_ROOT  # constants only; no connect call
    report = read(audit_dir / "HISTORICAL_INTERSECTIONS.json")
    require(not report["history_metadata_errors"], "Resolve historical identity errors before reservation")
    require(not output.exists(), "Use a new immutable confirmation output")
    excluded_rows = rows(audit_dir / "EXCLUSION_IDENTITIES.jsonl"); excluded = sets(excluded_rows)
    registry = ROOT / "configs/datasets/v3"
    text_all = rows(registry / "textvqa_full_inputs.jsonl")
    by_source = defaultdict(list)
    for item in text_all:
        if item["original_image_sha256"] in excluded["sha256"] or item["source_cluster_id"] in excluded["cluster"] or ("textvqa", item["source_id"]) in excluded["source_id"]: continue
        by_source[item["source_cluster_id"]].append(item)
    source_order = sorted(by_source, key=lambda source: (priority("textvqa_source", source), source))
    require(len(source_order) >= 32, "Insufficient historical-disjoint Text validation sources")
    text = [min(by_source[source], key=lambda x: (priority("textvqa_QA", source + "\0" + x["question_id"]), x["question_id"])) for source in source_order[:32]]
    # Existing Info TRAIN full24 LFS-verified sources, no whole data download.
    preparation = E / "results/round9/input_preparation_v1"
    source_proof = read(preparation / "PINNED_INFO_TRAIN_SOURCES.json")
    for item in source_proof["files"]:
        path = Path(item["local_path"])
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Pinned Info train source changed")
    ids = rows(preparation / "infovqa_train_all_candidate_ids.jsonl")
    info_groups = defaultdict(list)
    for item in ids:
        if item["source_cluster_id"] in excluded["cluster"] or ("infovqa", item["source_id"]) in excluded["source_id"]: continue
        info_groups[item["source_cluster_id"]].append(item)
    info_order = sorted(info_groups, key=lambda source: (priority("infovqa_source", source), source))
    output.mkdir(parents=True); images = output / "images"; images.mkdir()
    info = []; info_refs = []; locators = []; image_stage = []; rejected_sha = []
    forbidden_sha = excluded["sha256"] | {x["original_image_sha256"] for x in text}
    for source in info_order:
        loc = min(info_groups[source], key=lambda x: (priority("infovqa_QA", source + "\0" + x["question_id"]), x["question_id"]))
        parquet = pq.ParquetFile(loc["source_file"]); group = start = 0
        while start + parquet.metadata.row_group(group).num_rows <= loc["row_in_file"]:
            start += parquet.metadata.row_group(group).num_rows; group += 1
        table = parquet.read_row_group(group, columns=["image", "questionId", "question", "answers"])
        original = table.slice(loc["row_in_file"] - start, 1).to_pylist()[0]; del table
        require(str(original["questionId"]) == loc["question_id"], "Pinned source QA locator differs")
        encoded = original["image"]["bytes"]; digest = hashlib.sha256(encoded).hexdigest()
        if digest in forbidden_sha:
            rejected_sha.append({"source": source, "sha256": digest}); continue
        with Image.open(BytesIO(encoded)) as image: width, height = image.size; fmt = image.format.lower()
        ext = {"jpeg": "jpg", "png": "png", "webp": "webp", "tiff": "tiff"}[fmt]
        image = images / (digest + "." + ext)
        with image.open("xb") as stream: stream.write(encoded)
        remote = REMOTE_ROOT + "/data/round9_confirm_reserve_v1/images/" + image.name
        item = {"dataset": "infovqa", "dataset_revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e", "split": "train",
                "source_id": loc["source_id"], "source_cluster_id": source, "question_id": loc["question_id"], "id": "infovqa:train:" + loc["question_id"],
                "image_path": remote, "original_image_sha256": digest, "width": width, "height": height, "image_format": fmt, "image_bytes": len(encoded),
                "original_split_row": loc["original_split_row"], "question": original["question"], "question_sha256": hashlib.sha256(original["question"].encode()).hexdigest()}
        info.append(item); info_refs.append({"id": item["id"], "source_cluster_id": source, "answers": original["answers"]}); locators.append(loc)
        image_stage.append({"local": str(image), "remote": remote, "bytes": len(encoded), "sha256": digest}); forbidden_sha.add(digest)
        if len(info) == 32: break
    require(len(info) == 32, "Insufficient raw-SHA-disjoint Info train sources")
    text_ref_lookup = {x["id"]: x for x in rows(registry / "textvqa_full_refs.jsonl")}
    text_refs = [text_ref_lookup[x["id"]] for x in text]
    for dataset, values in [("textvqa", text), ("infovqa", info)]:
        require(len(values) == len({x["source_cluster_id"] for x in values}) == len({x["original_image_sha256"] for x in values}) == 32, "Confirmation sources must be unique")
        for item in values:
            require(item["original_image_sha256"] not in excluded["sha256"] and item["source_cluster_id"] not in excluded["cluster"] and
                    (dataset, item["source_id"]) not in excluded["source_id"], "Historical/current/formal overlap after reservation")
    files = {}; stage = list(image_stage)
    for dataset, inputs, refs in [("textvqa", text, text_refs), ("infovqa", info, info_refs)]:
        for label, values in [("inputs", inputs), ("refs", refs)]:
            path = output / f"{dataset}_confirm32_{label}.jsonl"; write_rows(path, values)
            files[path.name] = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "count": len(values), "inference_inputs": label == "inputs"}
            stage.append({"local": str(path), "remote": REMOTE_ROOT + "/configs/round9_confirm_reserve_v1/" + path.name, "sha256": sha(path), "bytes": path.stat().st_size})
    lock = output / "source_lock.json"
    write(lock, {"schema": "trajkv.round9.new_confirmation_source_lock.v1", "files": {key: {k: value[k] for k in ("sha256", "bytes")} for key, value in files.items()},
                 "actual_split": {"textvqa": "validation", "infovqa": "train"}, "not_official_full_benchmark": True})
    stage.append({"local": str(lock), "remote": REMOTE_ROOT + "/configs/round9_confirm_reserve_v1/source_lock.json", "sha256": sha(lock), "bytes": lock.stat().st_size})
    write_rows(output / "INFO32_LOCATORS.jsonl", locators)
    manifest = {"schema": "trajkv.round9.unobserved_confirmation_reserve.v1", "status": "FROZEN_UNRUN_CONFIRMATION_RESERVE_NOT_INFERENCE_OR_GENERALIZATION_RESULT",
        "seed": SEED, "source_selection": "SHA256(seed NUL dataset_source NUL source_cluster_id)", "QA_selection": "SHA256(seed NUL dataset_QA NUL source_cluster_id NUL question_id)",
        "selection_reads_model_outputs_or_correctness": False, "refs_decoded_only_after_ID_selection": True,
        "source_pool": {"textvqa": {"split": "validation", "eligible_sources": len(source_order), "no_new_TextTRAIN_download": True},
                        "infovqa": {"split": "train", "eligible_source_IDs_before_rawSHA_filter": len(info_order), "all24_pinned_sources_reused": True}},
        "new_confirmation_sources": {"textvqa": 32, "infovqa": 32}, "input_and_ref_files": files,
        "exclusion_audit": {"path": str(audit_dir / "HISTORICAL_INTERSECTIONS.json"), "sha256": sha(audit_dir / "HISTORICAL_INTERSECTIONS.json")},
        "exclusion_identity_file": {"path": str(audit_dir / "EXCLUSION_IDENTITIES.jsonl"), "sha256": sha(audit_dir / "EXCLUSION_IDENTITIES.jsonl")},
        "source_cluster_rawSHA_intersection_with_current160_formal2000_historical_compression_or_confirm": 0,
        "Info_rawSHA_rejected_before32": rejected_sha, "new_original_Info_images_extracted": 32,
        "existing_Text_images_referenced_not_downloaded": 32, "remote_stage_files": stage,
        "only_FP16_benchmark_exposure_is_separate_not_candidate_tuning": True,
        "no_claim_model_pretraining_unseen_or_universal_generalization": True,
        "current160_and_second32_MSE_are_observed_development_not_confirmation": True,
        "GPU_calls": 0, "remote_writes": 0, "Git_calls": 0, "script_sha256": sha(__file__)}
    write(output / "CONFIRMATION_RESERVE_MANIFEST.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(); p.add_argument("--audit", type=Path, default=WORK); p.add_argument("--reserve", type=Path); args = p.parse_args()
    if args.reserve:
        value = reserve(args.audit, args.reserve)
        print(json.dumps({"status": value["status"], "counts": value["new_confirmation_sources"], "source_pool": value["source_pool"]}))
    else:
        value = audit(args.audit)
        print(json.dumps({"sample_lists": len(value["historical_sample_list_inventory"]), "actual_compressed_manifests": len(value["actual_compressed_manifests"]),
            "validation_availability": value["validation_confirmation_availability"], "TextTRAIN": value["TextTRAIN_download_options"]}, ensure_ascii=False))


if __name__ == "__main__": main()
