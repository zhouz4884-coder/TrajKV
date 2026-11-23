"""CPU-only score-blind additional384 source materializer; original R128 frozen.

All source catalogs, parquet bytes and known-history exclusions are inherited
and pinned. Source selection is frozen before reference text is read. Nothing
in this module imports a model, reads predictions, launches a scientific job,
downloads a source, or changes the frozen Round10 builder.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

sys.dont_write_bytecode = True
import round10_data_builder_v1 as old

SCHEMA = "trajkv.round11.expand384_inputs.v1"
PLAN_SCHEMA = "trajkv.round11.expand384_data_plan.v1"
SEED = "Round11_expand384_20261123_v1"
COUNTS = dict(textvqa=144, infovqa=144, docvqa=48, chartqa=48)
DS = tuple(COUNTS)
ROLE = "confirmation_expansion_holdout"
BASE_R128_SHA = "acdfb720120d55793a4f6e7dc2b43e5681c0727aff28e8c906ecbc6696578380"
ORIGINAL_SELECTION_SHA = "45409e4adedae77660825c1bf1f9a39cf03d8a5dd14eae34374705b3b823f915"
EXPANSION_DESIGN_SHA = "ebe818bbc67b1d189b4e27b15e2fdd4af59c0a7e5633c5b1125add1ccfac0c19"
FILE_SPECS = {dataset + "_expand" + str(count): (count, {dataset: count}) for dataset, count in COUNTS.items()}
FILE_SPECS["expand384"] = (384, COUNTS)
PINS = {
    "round10_data_builder_v1.py": "914dfa258e6fbcfc00c709df6e78cfba6cba5c385db7c51f1e15ebb5d0f59936",
    "round10_inputs_v1.py": "61cc6dca53d95f0f24ab0a3b160eeb7be5fbac179e5ded510a2c2852e8e3dd77",
    "round11_confirmation_data_v1.py": "44297a3c58a493c26a17084a72c4800303b0b70143543217fe0ede3db02548cf",
}
FIELD_KEYS = old.FIELDS
METADATA_KEYS = FIELD_KEYS - {"question"}
sha, read, rows, inside = old.sha, old.read, old.rows, old.inside
freeze, freeze_json, freeze_rows = old.freeze, old.freeze_json, old.freeze_rows
priority = old.priority


def require(condition, message):
    if not condition:
        raise ValueError("ROUND11_EXPAND384_DATA: " + message)


class Identities(old.Identities):
    def add(self, row):
        super().add(row)
        # Original bytes and any explicitly recorded legacy re-encoding hashes
        # are conservative blockers. Complete source/parent identities dominate
        # even when historical image bytes were re-encoded.
        for key in ("original_image_sha256", "sha256", "image_sha256"):
            value = row.get(key)
            if isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value):
                self.shas.add(value)


def same_project_device(path, project):
    p = Path(path).resolve()
    require(inside(p, project), "path escapes resolved project: " + str(path))
    ancestor = p
    while not ancestor.exists():
        ancestor = ancestor.parent
    require(ancestor.stat().st_dev == project.stat().st_dev, "path is not on project data device: " + str(path))
    return p


def pinned(item, project):
    p = same_project_device(item["path"], project)
    require(p.is_file() and sha(p) == item["sha256"], "missing/changed pinned file: " + str(p))
    if "bytes" in item:
        require(p.stat().st_size == item["bytes"], "pinned file byte count differs")
    return p


def validate_rows(values, *, expected_count=None):
    require(values and (expected_count is None or len(values) == expected_count), "wrong input count")
    for row in values:
        require(FIELD_KEYS - {"group"} <= set(row) <= FIELD_KEYS, "unknown/reference or missing model input fields")
        ds = row["dataset"]
        require(ds in old.ALLOWED and row["dataset_revision"] == old.ALLOWED[ds][0] and row["split"] in old.ALLOWED[ds][1], "revision/true split differs")
        require(row["split"] == {"textvqa": "validation", "infovqa": "train", "docvqa": "train", "chartqa": "test"}[ds], "additional384 true pool split differs")
        require(isinstance(row["question_id"], str) and row["id"] == f'{ds}:{row["split"]}:{row["question_id"]}', "question ID/split differs")
        require(isinstance(row["question"], str) and row["question"].strip() and hashlib.sha256(row["question"].encode()).hexdigest() == row["question_sha256"], "question hash differs")
        require(isinstance(row["source_id"], str) and row["source_id"], "source ID missing")
        expected_cluster = "docvqa:ucsf:" + row["source_id"].rsplit("/", 1)[0] if ds == "docvqa" else ds + ":image:" + row["source_id"]
        require(row["source_cluster_id"] == expected_cluster, "source/parent cluster differs")
        require(re.fullmatch("[0-9a-f]{64}", row["original_image_sha256"]) is not None, "image SHA invalid")
        require(all(type(row[key]) is int and row[key] > 0 for key in ("width", "height", "image_bytes")), "image dimensions/bytes invalid")
        require(type(row["original_split_row"]) is int and row["original_split_row"] >= 0, "source row invalid")
        path = PurePosixPath(row["image_path"])
        allowed = "/baseline_matrix_v3/data/" in str(path) or "/baseline_matrix_v3/results/round11_v1/data_expand384_v1/images/" in str(path)
        require(path.is_absolute() and ".." not in path.parts and "\\" not in str(path) and allowed and path.stem == row["original_image_sha256"], "noncanonical image path")
    for key in ("id", "source_cluster_id", "original_image_sha256"):
        require(len({row[key] for row in values}) == len(values), "duplicate question/source/content: " + key)
    return values


def load_expand_inputs(path, manifest_path, expected_sha256=None, expected_role=ROLE):
    """No-answer loader for later model use; never opens a reference file."""
    path = Path(path)
    require(path.name.endswith("_inputs.jsonl"), "only no-answer input files")
    manifest = read(manifest_path)
    require(expected_role == ROLE, "old R128/development role cannot admit expansion inputs")
    require(manifest.get("schema") == SCHEMA and manifest.get("answers_in_encoder_inputs") is False
        and manifest.get("counts") == COUNTS and manifest.get("total_count") == 384
        and manifest.get("base_R128_manifest_sha256") == BASE_R128_SHA
        and manifest.get("original_selection_sha256") == ORIGINAL_SELECTION_SHA
        and manifest.get("expansion_design_sha256") == EXPANSION_DESIGN_SHA, "unknown/unbound expansion manifest")
    key = path.name[:-len("_inputs.jsonl")]
    require(key in FILE_SPECS and key in manifest["input_files"], "cohort not bound by expansion manifest")
    count, domains = FILE_SPECS[key]
    entry = manifest["input_files"][key]
    digest = sha(path)
    require(entry["sha256"] == digest and entry["bytes"] == path.stat().st_size and entry["answers_included"] is False
        and entry["role"] == ROLE and entry["count"] == count and entry["domains"] == domains
        and Path(entry["path"]).name == path.name, "full expansion input binding differs")
    require(expected_sha256 is None or expected_sha256 == digest, "explicit input SHA differs")
    values = validate_rows(rows(path), expected_count=count)
    require(dict(Counter(row["dataset"] for row in values)) == entry["domains"], "cohort domain counts differ")
    return values


def metadata_catalog(path, refs, dataset):
    # Canonical catalogs are no-answer inputs. Raw question strings already in
    # that catalog are discarded immediately and never used by selection.
    result = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            require(set(row) <= FIELD_KEYS and row["dataset"] == dataset, "catalog contains reference/unknown fields")
            metadata = {key: value for key, value in row.items() if key in METADATA_KEYS}
            result.append(dict(kind="canonical", input_metadata=metadata, inputs_file=str(path), refs_file=str(refs),
                               **{key: metadata[key] for key in ("id", "dataset", "dataset_revision", "split", "source_id", "source_cluster_id", "question_id")}))
    return result


class Materializer(old.Materializer):
    def __init__(self, plan, source_records, project):
        super().__init__(plan, source_records)
        self.project = project
        self.input_cache = {}

    def inspect_image(self, candidate):
        if candidate["kind"] == "canonical":
            row = dict(candidate["input_metadata"])
            path = same_project_device(row["image_path"], self.project)
            require(inside(path, self.project / "data") and path.is_file() and path.stat().st_size == row["image_bytes"] and sha(path) == row["original_image_sha256"], "canonical original image differs")
            return row, None
        from PIL import Image
        raw = self.raw(candidate, ["image"])
        image = raw["image"]
        data = image["bytes"] if isinstance(image, dict) else image
        require(isinstance(data, bytes), "source lacks original image bytes")
        digest = hashlib.sha256(data).hexdigest()
        with Image.open(BytesIO(data)) as im:
            width, height, fmt = *im.size, im.format.lower()
        suffix = "jpg" if fmt == "jpeg" else fmt
        path = Path(self.plan["images_root"]) / (digest + "." + suffix)
        row = {key: candidate[key] for key in ("id", "dataset", "dataset_revision", "split", "source_id", "source_cluster_id", "question_id", "original_split_row")}
        row.update(original_image_sha256=digest, image_path=str(path), width=width, height=height,
                   image_bytes=len(data), image_format=fmt, group=candidate.get("group"))
        return row, data

    def commit_image(self, row, data):
        if data is not None:
            path = same_project_device(row["image_path"], self.project)
            freeze(path, data)
            self.images[str(path)] = dict(path=str(path), bytes=len(data), sha256=row["original_image_sha256"])

    def finish(self, candidate, selected):
        if candidate["kind"] == "canonical":
            name = candidate["inputs_file"]
            if name not in self.input_cache:
                self.input_cache[name] = {row["id"]: row for row in rows(name)}
            row = dict(self.input_cache[name][candidate["id"]])
            name = candidate["refs_file"]
            if name not in self.ref_cache:
                self.ref_cache[name] = {row["id"]: row for row in rows(name)}
            raw_ref = self.ref_cache[name][row["id"]]
            ref = {key: raw_ref[key] for key in ("id", "source_cluster_id", "answers")}
            if raw_ref.get("group") is not None:
                ref["group"] = raw_ref["group"]
        else:
            raw = self.raw(candidate, ["questionId", "question", "answers"])
            require(str(raw["questionId"]) == candidate["question_id"], "question locator differs")
            question = raw["question"]
            row = dict(selected, question=question, question_sha256=hashlib.sha256(question.encode()).hexdigest())
            ref = dict(id=row["id"], source_cluster_id=row["source_cluster_id"], answers=raw["answers"])
        validate_rows([row], expected_count=1)
        require(ref["id"] == row["id"] and ref["source_cluster_id"] == row["source_cluster_id"] and isinstance(ref["answers"], list) and ref["answers"], "offline reference identity/answers invalid")
        return row, ref


def ordered_sources(candidates, dataset, seed=SEED):
    groups = defaultdict(list)
    for candidate in candidates:
        groups[candidate["source_cluster_id"]].append(candidate)
    for cluster in sorted(groups, key=lambda value: (priority(seed, "expand384:source:" + dataset, value), value)):
        chosen = min(groups[cluster], key=lambda row: (priority(seed, "question:" + dataset, cluster + "\0" + row["question_id"]), row["question_id"]))
        yield chosen


def build(plan):
    require(plan["schema"] == PLAN_SCHEMA and plan["seed"] == SEED and plan["confirmation_counts"] == COUNTS and plan["total_count"] == 384, "fixed v1 cohort/seed contract")
    require(plan["selection_reads_model_predictions_or_correctness"] is False and plan["selection_reads_question_text"] is False and plan["reference_text_decoded_only_after_source_selection"] is True and plan["allow_pinned_public_download"] is False, "score-blind no-download selection contract")
    require(plan["base_R128_manifest"]["sha256"] == BASE_R128_SHA
        and plan["original_selection"]["sha256"] == ORIGINAL_SELECTION_SHA
        and plan["expansion_design_sha256"] == EXPANSION_DESIGN_SHA
        and plan["selected_methods_fixed"] == ["FP16", "R7low", "R7high", "JointV", "JointSpatialWO"]
        and plan["repeated_source_measurements_added"] == 0, "fixed original selection/design/R128 source contract")
    for name, digest in PINS.items():
        require(sha(Path(__file__).with_name(name)) == digest, "frozen dependency changed: " + name)
    project = Path(plan["project_root"]).resolve()
    require(os.name != "nt" and str(project).startswith("/media/") and project.name == "baseline_matrix_v3", "materialize only on declared remote project data disk")
    require(plan["runtime_project_root"] == str(project), "runtime project relocation must be explicit new version")
    expected_output = project / "results/round11_v1/data_expand384_v1"
    require(Path(plan["output_root"]) == expected_output and Path(plan["images_root"]) == expected_output / "images" and Path(plan["cache_root"]) == expected_output / "cache" and Path(plan["inputs_root"]) == project / "configs/round11_expand384_inputs_v1", "new output/cache/image roots differ from v1 contract")
    for key in ("output_root", "inputs_root", "images_root", "cache_root", "sources_root"):
        same_project_device(plan[key], project)
    output, inputs = Path(plan["output_root"]), Path(plan["inputs_root"])
    require(not (output / "COMPLETE.json").exists(), "completed v1 data exists; never rebuild/overwrite")
    cache = Path(plan["cache_root"])
    for variable in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH"):
        location = cache / variable.lower()
        location.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(location)
    freeze_json(output / "FROZEN_PLAN.json", {key: value for key, value in plan.items() if not key.startswith("_")})
    freeze_json(output / "BUILDER_IDENTITY.json", dict(schema="trajkv.round11.expand384_builder_identity.v1", builder_sha256=sha(__file__), dependency_pins=PINS, plan_sha256=plan["_plan_sha256"], no_model_GPU_calls=True))
    prior = read(pinned(plan["inherited_round10_manifest"], project))
    for key, count in (("calibration64", 64), ("D160", 160), ("reserve224", 224)):
        entry = prior["input_files"][key]
        require(entry["count"] == count and any(item["path"] == entry["path"] and item["sha256"] == entry["sha256"] for item in plan["used_inputs"]), "all Round10 cohorts must be excluded: " + key)
    base_manifest = read(pinned(plan["base_R128_manifest"], project))
    require(base_manifest["schema"] == "trajkv.round11.confirm128_inputs.v1" and base_manifest["total_count"] == 128,
        "exact original R128 manifest required")
    base_inputs = base_manifest["input_files"]["R128"]
    require(base_inputs["count"] == 128 and any(item["path"] == base_inputs["path"] and item["sha256"] == base_inputs["sha256"]
        for item in plan["used_inputs"]), "all original R128 inputs must be explicitly excluded")
    base_identities = Identities()
    base_rows = rows(pinned(base_inputs, project))
    require(len(base_rows) == 128, "original R128 input count differs")
    for row in base_rows:
        base_identities.add(row)
    require(len(base_identities.ids) == len(base_identities.clusters) == len(base_identities.shas) == 128, "original R128 source identities not unique")
    # Fingerprint only: no decision/outcome/reference text is decoded before
    # the new source list has been frozen. The fixed methods are plan constants.
    original_selection_path = pinned(plan["original_selection"], project)
    blocked, selected_identities = Identities(), Identities()
    coverage = []
    for item in plan["history_exclusions"] + plan["used_inputs"]:
        path = pinned(item, project)
        values = rows(path)
        for row in values:
            blocked.add(row)
        if "count" in item:
            require(len(values) == item["count"], "exclusion count differs")
        coverage.append(dict(path=str(path), sha256=item["sha256"], count=len(values), role=item.get("role", "known_historical_source_lists_v5")))
    catalogs = defaultdict(list)
    for pool in plan["canonical_pools"]:
        inputs_path, refs_path = pinned(pool["inputs"], project), pinned(pool["refs"], project)
        catalogs[pool["dataset"]].extend(metadata_catalog(inputs_path, refs_path, pool["dataset"]))
    source_records, source_proof, offsets = {}, [], defaultdict(int)
    for spec in plan["parquet_sources"]:
        path = Path(plan["sources_root"]) / spec["dataset"] / Path(spec["upstream_path"]).name
        # Missing shards are an explicit blocker, never a download/replacement.
        pinned(dict(path=str(path), bytes=spec["bytes"], sha256=spec["sha256"]), project)
        require(path.name not in source_records, "ambiguous source basename")
        source_records[path.name] = path
        effective = dict(spec, original_offset=offsets[(spec["dataset"], spec["split"])])
        candidates = old.parquet_catalog(path, effective)
        offsets[(spec["dataset"], spec["split"])] += len(candidates)
        if not spec.get("calibration_only"):
            catalogs[spec["dataset"]].extend(candidates)
        source_proof.append(dict(effective, actual_path=str(path), actual_SHA_verified=True, candidate_QA=len(candidates), used_as_confirmation_candidate_pool=not spec.get("calibration_only", False)))
    materializer = Materializer(plan, source_records, project)
    selected, rejections, pool_counts = {}, [], {}
    for dataset, count in COUNTS.items():
        values = []
        ordered = list(ordered_sources(catalogs[dataset], dataset, plan["seed"]))
        pool_counts[dataset] = dict(QA_rows=len(catalogs[dataset]), source_clusters=len(ordered), requested=count)
        for candidate in ordered:
            why = blocked.reasons(candidate) + selected_identities.reasons(candidate)
            if why:
                continue
            row, data = materializer.inspect_image(candidate)
            why = blocked.reasons(row) + selected_identities.reasons(row)
            if why:
                rejections.append(dict(id=candidate["id"], source_cluster_id=candidate["source_cluster_id"], original_image_sha256=row["original_image_sha256"], reasons=sorted(set(why))))
                continue
            materializer.commit_image(row, data)
            selected_identities.add(row)
            values.append((candidate, row))
            if len(values) == count:
                break
        selected[dataset] = values
        pool_counts[dataset]["selected"] = len(values)
        if len(values) != count:
            freeze_json(output / "INSUFFICIENT_SOURCE_POOL.json", dict(schema="trajkv.round11.expand384_insufficient.v1", dataset=dataset, requested=count, selected=len(values), seed=plan["seed"], pool_counts=pool_counts, original_plan_preserved=True, references_not_opened=True, no_model_GPU_calls=True))
            require(False, "insufficient disjoint source pool: " + dataset)
    # This file is durably frozen before the first question/reference finishing
    # call. Re-running a partial version must reproduce identical source bytes.
    provenance = [dict(dataset=dataset, id=row["id"], source_id=row["source_id"], source_cluster_id=row["source_cluster_id"], original_image_sha256=row["original_image_sha256"], locator={key: value for key, value in candidate.items() if key not in ("input_metadata", "inputs_file", "refs_file")}) for dataset in DS for candidate, row in selected[dataset]]
    locators = freeze_rows(output / "SELECTED_SOURCE_LOCATORS.jsonl", provenance)
    base_overlap = {name: len(getattr(selected_identities, name) & getattr(base_identities, name)) for name in ("ids", "sources", "clusters", "shas")}
    require(all(value == 0 for value in base_overlap.values()), "new source list overlaps original R128")
    freeze_json(output / "SELECTION_FROZEN_BEFORE_REFERENCES.json", dict(schema="trajkv.round11.expand384_selection_frozen.v1", source_locators=locators, selected_count=384, seed=plan["seed"], reference_text_not_opened=True, selection_reads_model_predictions_or_correctness=False, base_R128_intersections=base_overlap, original_selection_sha256=ORIGINAL_SELECTION_SHA, expansion_design_sha256=EXPANSION_DESIGN_SHA))
    completed = {dataset: [materializer.finish(candidate, row) for candidate, row in selected[dataset]] for dataset in DS}
    all_inputs = [row for dataset in DS for row, _ in completed[dataset]]
    all_refs = [ref for dataset in DS for _, ref in completed[dataset]]
    validate_rows(all_inputs, expected_count=384)
    require(len(selected_identities.clusters) == len(selected_identities.shas) == 384 and all(not blocked.reasons(row) for row in all_inputs), "independent final source/content intersection failed")
    input_files, reference_files = {}, {}
    cohorts = {dataset + "_expand" + str(COUNTS[dataset]): completed[dataset] for dataset in DS}
    cohorts["expand384"] = list(zip(all_inputs, all_refs))
    for key, values in cohorts.items():
        model_rows, refs = [row for row, _ in values], [ref for _, ref in values]
        entry = freeze_rows(inputs / (key + "_inputs.jsonl"), model_rows)
        input_files[key] = dict(entry, count=len(model_rows), role=ROLE, domains=dict(Counter(row["dataset"] for row in model_rows)), answers_included=False)
        entry = freeze_rows(inputs / (key + "_refs.jsonl"), refs)
        reference_files[key] = dict(entry, count=len(refs), role="CPU_SCORING_ONLY", answers_not_encoder_inputs=True)
    proof = freeze_json(output / "SOURCE_DISJOINT_PROOF.json", dict(schema="trajkv.round11.expand384_disjoint.v1", selected_count=384, counts=COUNTS, known_history_and_used_and_formal500_overlap=0, Round10_all_cal64_D160_reserve224_overlap=0, base_R128_intersections=base_overlap, repeated_source_measurements_added=0, identity_fields=["question_id", "dataset/source_id", "source_cluster_id including Doc parent", "original_image_sha256 and explicit historical image hashes"], exclusion_coverage=coverage, pool_counts=pool_counts, SHA_rejections=rejections, historical_coverage_limits=plan["historical_coverage_limits"], seed=plan["seed"], no_pretraining_unseen_or_absolute_history_independence_claim=True, selection_reads_model_outcomes=False, reference_text_decoded_after_complete_source_selection=True))
    raw_proof = freeze_json(output / "RAW_SOURCE_PROOF.json", dict(sources=source_proof, canonical_catalogs=plan["canonical_pools"], all_pinned_bytes_verified=True, downloads_performed=0))
    lock = freeze_json(inputs / "expand384_source_lock_v1.json", dict(schema="trajkv.round11.expand384_source_lock.v1", files={Path(value["path"]).name: {key: value[key] for key in ("sha256", "bytes")} for value in list(input_files.values()) + list(reference_files.values())}, true_splits={dataset: sorted({row["split"] for row, _ in completed[dataset]}) for dataset in DS}))
    manifest = dict(schema=SCHEMA, status="READY_FROZEN_CPU_EXPANSION_DATA_NO_MODEL_EXECUTION", input_files=input_files, reference_files=reference_files, answers_in_encoder_inputs=False, counts=COUNTS, total_count=384, seed=plan["seed"], builder_source_sha256=sha(__file__), frozen_dependency_pins=PINS, plan_sha256=plan["_plan_sha256"], inherited_round10_manifest=plan["inherited_round10_manifest"], base_R128_manifest=plan["base_R128_manifest"], base_R128_manifest_sha256=BASE_R128_SHA, original_selection=plan["original_selection"], original_selection_sha256=ORIGINAL_SELECTION_SHA, expansion_design_sha256=EXPANSION_DESIGN_SHA, original_selection_fingerprint_verified_without_reading_outcomes=True, original_R128_sources=128, combined_sources_if_appended=512, repeated_source_measurements_added=0, additional_size_fixed_after_partially_observed_R128=True, report_original_additional_combined_separately=True, source_locators=locators, disjoint_proof=proof, raw_source_proof=raw_proof, source_lock=lock, actual_model_or_GPU_calls=0, source_split_note=plan["source_split_note"], no_pretraining_unseen_claim=True)
    manifest_file = freeze_json(inputs / "EXPAND384_INPUT_MANIFEST_v1.json", manifest)
    for entry in input_files.values():
        load_expand_inputs(entry["path"], manifest_file["path"], entry["sha256"])
    freeze_json(output / "IMAGE_MATERIALIZATION.json", dict(extracted=list(materializer.images.values()), original_bytes_preserved=True, canonical_images_verified_on_same_project_data_device=True))
    result = dict(status="PASS_ROUND11_EXPAND384_FROZEN_CPU_DATA", manifest=manifest_file, counts=COUNTS, total=384, unique_sources=384, seed=plan["seed"], input_references_separate=True, actual_model_or_GPU_calls=0, scientific_jobs_launched=False)
    freeze_json(output / "COMPLETE.json", result)
    print(json.dumps(result), flush=True)
    return result


def self_check():
    fixture = [dict(source_cluster_id=cluster, question_id=question) for cluster in ("s0", "s1", "s2") for question in ("q0", "q1")]
    chosen = list(ordered_sources(fixture, "textvqa"))
    require(chosen == list(ordered_sources(list(reversed(fixture)), "textvqa")) and len({row["source_cluster_id"] for row in chosen}) == 3, "source/question hash ordering depends on input order")
    identities = Identities()
    identities.add(dict(dataset="docvqa", id="docvqa:train:1", source_id="parent/page1", source_cluster_id="docvqa:ucsf:parent", original_image_sha256="a" * 64, sha256="b" * 64))
    require(identities.reasons(dict(dataset="docvqa", id="new", source_id="parent/page2", source_cluster_id="docvqa:ucsf:parent", original_image_sha256="c" * 64)) == ["source_or_document_cluster"], "Doc parent isolation")
    require("original_image_bytes_SHA" in identities.reasons(dict(dataset="textvqa", id="new", source_id="new", source_cluster_id="new", original_image_sha256="b" * 64)), "historical hash alias isolation")
    print(json.dumps(dict(passed=True, synthetic_only=True, source_order_input_permutation_invariant=True, one_question_per_source=True, Doc_parent_and_image_SHA_blockers=True, no_model_or_GPU=True)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        require(args.plan is None, "self-check does not materialize a plan")
        self_check()
        return
    require(args.plan is not None, "--plan is required for CPU materialization")
    plan = read(args.plan)
    plan["_plan_sha256"] = sha(args.plan)
    build(plan)


if __name__ == "__main__":
    main()
