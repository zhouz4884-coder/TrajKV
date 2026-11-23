"""CPU-only, versioned Round10 data preparation on the project data volume.

Selection uses IDs/source clusters and original-byte SHA, never predictions or
reference correctness. The plan pins every input index. No model is imported.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

sys.dont_write_bytecode = True
from round10_inputs_v1 import ALLOWED, FILE_SPECS, FIELDS, SCHEMA, validate_rows, load_round10_inputs

DS = ("textvqa", "infovqa", "docvqa", "chartqa")
DEV = dict(textvqa=64, infovqa=64, docvqa=16, chartqa=16)
RESERVE = dict(textvqa=64, infovqa=64, docvqa=48, chartqa=48)


def require(value, message):
    if not value:
        raise ValueError("ROUND10_DATA: " + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def freeze(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == body, "immutable output differs: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(body)
    return dict(path=str(path), bytes=len(body), sha256=hashlib.sha256(body).hexdigest())


def freeze_json(path, value):
    return freeze(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def freeze_rows(path, values):
    return freeze(path, ("".join(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for v in values)).encode("utf-8"))


def pinned(item):
    p = Path(item["path"])
    require(p.is_file() and sha(p) == item["sha256"], "pinned input differs: " + str(p))
    if "bytes" in item:
        require(p.stat().st_size == item["bytes"], "pinned input size differs")
    return p


class Identities:
    def __init__(self):
        self.ids, self.sources, self.clusters, self.shas = set(), set(), set(), set()

    def add(self, r):
        ds = r["dataset"]
        if r.get("id"):
            self.ids.add(r["id"])
        source = r.get("source_id")
        cluster = r.get("source_cluster_id", r.get("cluster"))
        digest = r.get("original_image_sha256", r.get("sha256"))
        if source and not str(source).startswith("UNRESOLVED_"):
            self.sources.add((ds, str(source)))
        if cluster and not str(cluster).startswith("UNRESOLVED_"):
            self.clusters.add(str(cluster))
        if digest:
            self.shas.add(digest)

    def reasons(self, r):
        found = []
        if r.get("id") in self.ids:
            found.append("question_id")
        if (r["dataset"], str(r.get("source_id"))) in self.sources:
            found.append("source_id")
        if r.get("source_cluster_id", r.get("cluster")) in self.clusters:
            found.append("source_or_document_cluster")
        if r.get("original_image_sha256", r.get("sha256")) in self.shas:
            found.append("original_image_bytes_SHA")
        return found


def priority(seed, kind, value):
    return hashlib.sha256((seed + "\0" + kind + "\0" + str(value)).encode("utf-8")).hexdigest()


def download_source(item, destination):
    """Explicit pinned public source only; no credentials or signed URLs logged."""
    destination = Path(destination)
    if destination.exists():
        require(destination.stat().st_size == item["bytes"] and sha(destination) == item["sha256"], "cached parquet differs")
        return destination
    require(item["bytes"] <= 1024**3, "explicit approval/source plan needed for unexpectedly large shard")
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".partial")
    offset = part.stat().st_size if part.exists() else 0
    require(offset <= item["bytes"], "partial parquet larger than source")
    if offset < item["bytes"]:
        url = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{upstream_path}".format(**item)
        headers = {"User-Agent": "TrajKV-Round10-pinned-public-data/1"}
        if offset:
            headers["Range"] = "bytes=%d-" % offset
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
            resume = offset and response.getcode() == 206
            if resume:
                require(response.headers.get("Content-Range", "").startswith("bytes %d-" % offset), "wrong HTTP resume range")
            with part.open("ab" if resume else "wb") as out:
                for data in iter(lambda: response.read(1024**2), b""):
                    out.write(data)
    require(part.stat().st_size == item["bytes"] and sha(part) == item["sha256"], "downloaded public shard SHA differs")
    require(not destination.exists(), "concurrent source materializer")
    part.rename(destination)
    return destination


class Materializer:
    def __init__(self, plan, source_records):
        self.plan, self.sources = plan, source_records
        self.cache_key, self.cache_table = None, None
        self.images = {}
        self.ref_cache = {}

    def raw(self, c, columns):
        import pyarrow.parquet as pq
        p = self.sources[c["source_basename"]]
        key = (str(p), c["row_group"], tuple(columns))
        if self.cache_key != key:
            self.cache_table = pq.ParquetFile(p).read_row_group(c["row_group"], columns=columns)
            self.cache_key = key
        return self.cache_table.slice(c["row_in_group"], 1).to_pylist()[0]

    def image(self, c):
        """Read only image/source columns; no reference answers during selection."""
        if c["kind"] == "canonical":
            r = c["input"]
            p = Path(r["image_path"])
            require(inside(p, Path(self.plan["project_root"]) / "data"), "canonical image escapes project data")
            require(p.is_file() and p.stat().st_size == r["image_bytes"] and sha(p) == r["original_image_sha256"], "canonical original image differs")
            return dict(r)
        from PIL import Image
        raw = self.raw(c, ["image", "questionId"] if c["dataset"] in ("infovqa", "docvqa") else ["image", "imgname"])
        image = raw["image"]
        data = image["bytes"] if isinstance(image, dict) else image
        require(isinstance(data, bytes), "pinned source lacks original image bytes")
        digest = hashlib.sha256(data).hexdigest()
        with Image.open(BytesIO(data)) as im:
            width, height, fmt = *im.size, im.format.lower()
        suffix = "jpg" if fmt == "jpeg" else fmt
        name = digest + "." + suffix
        p = Path(self.plan["images_root"]) / name
        freeze(p, data)
        runtime = self.plan["runtime_project_root"].rstrip("/") + "/data/round10_inputs_v1/images/" + name
        self.images[str(p)] = dict(path=str(p), runtime_path=runtime, bytes=len(data), sha256=digest)
        return dict(id=c["id"], dataset=c["dataset"], dataset_revision=c["dataset_revision"], split=c["split"],
                    source_id=c["source_id"], source_cluster_id=c["source_cluster_id"], question_id=c["question_id"],
                    original_image_sha256=digest, image_path=runtime, width=width, height=height,
                    image_bytes=len(data), image_format=fmt, original_split_row=c["original_split_row"], group=c.get("group"))

    def finish(self, c, selected, need_ref):
        if c["kind"] == "canonical":
            r = dict(selected)
            ref = None
            if need_ref:
                name = c["refs_file"]
                if name not in self.ref_cache:
                    self.ref_cache[name] = {r["id"]: r for r in rows(name)}
                ref = self.ref_cache[name][r["id"]]
        else:
            is_document = c["dataset"] in ("infovqa", "docvqa")
            cols = ["questionId", "question"] if is_document else ["query", "type"]
            if need_ref:
                cols += ["answers"] if is_document else ["label"]
            raw = self.raw(c, cols)
            question = raw["question"] if is_document else raw["query"]
            if is_document:
                require(str(raw["questionId"]) == c["question_id"], "question locator differs")
            r = dict(selected, question=question, question_sha256=hashlib.sha256(question.encode("utf-8")).hexdigest())
            ref = None
            if need_ref:
                answers = raw["answers"] if is_document else [str(raw["label"])]
                ref = dict(id=r["id"], source_cluster_id=r["source_cluster_id"], answers=answers)
                if r.get("group") is not None:
                    ref["group"] = r["group"]
        validate_rows([r], expected_count=1)
        require(not need_ref or ref["id"] == r["id"] and ref["source_cluster_id"] == r["source_cluster_id"], "reference identity differs")
        return r, ref


def parquet_catalog(path, spec):
    import pyarrow.parquet as pq
    p = pq.ParquetFile(path)
    ds = spec["dataset"]
    columns = (["questionId", "image.path"] if ds == "infovqa" else
               ["questionId", "ucsf_document_id", "ucsf_document_page_no"] if ds == "docvqa" else ["imgname", "type"])
    result = []
    start = 0
    for rg in range(p.num_row_groups):
        table = p.read_row_group(rg, columns=columns).to_pylist()
        for index, r in enumerate(table):
            sid = (r["image"]["path"] if ds == "infovqa" else str(r["ucsf_document_id"]) + "/" + str(r["ucsf_document_page_no"]) if ds == "docvqa" else str(r["imgname"]))
            qid = str(r["questionId"]) if ds in ("infovqa", "docvqa") else "row_%06d" % (start + index)
            result.append(dict(kind="parquet", dataset=ds, dataset_revision=spec["revision"], split=spec["split"],
                               source_id=sid, source_cluster_id=("docvqa:ucsf:" + str(r["ucsf_document_id"]) if ds == "docvqa" else ds + ":image:" + sid), question_id=qid,
                               id=ds + ":" + spec["split"] + ":" + qid, original_split_row=spec.get("original_offset", 0) + start + index,
                               source_basename=Path(path).name, row_group=rg, row_in_group=index,
                               group=r.get("type")))
        start += len(table)
    return result


def build(plan):
    require(plan["schema"] == "trajkv.round10.data_plan.v1" and plan["selection_reads_model_predictions_or_correctness"] is False,
            "explicit score-blind data plan required")
    project = Path(plan["project_root"]).resolve()
    for key in ("output_root", "inputs_root", "images_root", "sources_root"):
        require(inside(plan[key], project), "new data output escapes named project: " + key)
    if os.name == "nt":
        require(inside(project, Path("${EVIDENCE_ROOT}")), "all local new data must use E volume")
    else:
        require(str(project).startswith("/media/") and "baseline_matrix_v3" == project.name, "remote output must use the named data-volume project")
    output, inputs = Path(plan["output_root"]), Path(plan["inputs_root"])
    require(not (output / "COMPLETE.json").exists(), "completed data version exists; do not rebuild")
    seed = plan["seed"]
    coverage = []
    blocked, formal = Identities(), Identities()
    for item in plan["history_exclusions"]:
        p = pinned(item)
        for r in rows(p):
            blocked.add(r)
        coverage.append(dict(path=str(p), sha256=sha(p), role="historical_compression_or_tuning_source_exclusion"))
    for item in plan["used_inputs"]:
        p = pinned(item)
        for r in rows(p):
            blocked.add(r)
            if item["role"] == "formal500":
                formal.add(r)
        coverage.append(dict(path=str(p), sha256=sha(p), role=item["role"]))
    catalogs, calibration = defaultdict(list), defaultdict(list)
    for item in plan["canonical_pools"]:
        p, refs = pinned(item["inputs"]), pinned(item["refs"])
        for r in rows(p):
            c = dict(kind="canonical", input=r, refs_file=str(refs), **{k: r[k] for k in ("id", "dataset", "dataset_revision", "split", "source_id", "source_cluster_id", "question_id")})
            catalogs[r["dataset"]].append(c)
            if r["dataset"] != "chartqa":
                calibration[r["dataset"]].append(c)
        coverage.append(dict(path=str(p), sha256=sha(p), role="candidate_metadata_catalog_not_compression_exposure"))
    source_records, source_proof, split_offsets = {}, [], defaultdict(int)
    for spec in plan["parquet_sources"]:
        p = Path(plan["sources_root"]) / spec["dataset"] / Path(spec["upstream_path"]).name
        if not p.exists():
            require(plan.get("allow_pinned_public_download", False), "missing required pinned source: " + str(p))
            download_source(spec, p)
        require(p.stat().st_size == spec["bytes"] and sha(p) == spec["sha256"], "actual pinned parquet bytes differ")
        source_records[p.name] = p
        split_key = (spec["dataset"], spec["split"])
        effective = dict(spec, original_offset=spec.get("original_offset", split_offsets[split_key]))
        candidates = parquet_catalog(p, effective)
        split_offsets[split_key] += len(candidates)
        target = calibration if spec.get("calibration_only") else catalogs
        target[spec["dataset"]].extend(candidates)
        if spec["dataset"] in ("infovqa", "docvqa") and target is not calibration:
            calibration[spec["dataset"]].extend(candidates)
        source_proof.append(dict(effective, actual_path=str(p), actual_sha_verified=True, candidate_QA=len(candidates)))
    # Prefer explicitly declared old calibration, but never used formal500.
    preferred = defaultdict(list)
    for item in plan["reuse_calibration_inputs"]:
        p = pinned(item)
        for r in rows(p):
            if formal.reasons(r):
                continue
            preferred[r["dataset"]].append(dict(kind="canonical", input=r, refs_file="",
                old_calibration_reused=(item.get("reuse_kind") != "preselected_round10" or r["id"] in plan.get("historically_reused_calibration_ids", [])),
                **{k: r[k] for k in ("id", "dataset", "dataset_revision", "split", "source_id", "source_cluster_id", "question_id")}))
        coverage.append(dict(path=str(p), sha256=sha(p), role="explicit_calibration_reuse_allowed_not_unseen"))
    materializer = Materializer(plan, source_records)
    occupied = Identities()
    selected = {"calibration": defaultdict(list), "development": defaultdict(list), "reserve": defaultdict(list)}
    rejections = []

    def choose(candidates, dataset, count, role, *, allow_history=False):
        groups = defaultdict(list)
        for c in candidates:
            groups[c["source_cluster_id"]].append(c)
        ordered = sorted(groups, key=lambda k: (priority(seed, role + ":source:" + dataset, k), k))
        result = []
        for cluster in ordered:
            c = min(groups[cluster], key=lambda r: (priority(seed, "question:" + dataset, cluster + "\0" + r["question_id"]), r["question_id"]))
            why = occupied.reasons(c) + formal.reasons(c)
            if not allow_history:
                why += blocked.reasons(c)
            if why:
                continue
            r = materializer.image(c)
            why = occupied.reasons(r) + formal.reasons(r)
            if not allow_history:
                why += blocked.reasons(r)
            if why:
                rejections.append(dict(role=role, id=c["id"], source=cluster, sha256=r["original_image_sha256"], reasons=sorted(set(why))))
                continue
            occupied.add(r)
            result.append((c, r, c.get("old_calibration_reused", allow_history)))
            if len(result) == count:
                break
        return result

    for ds in DS:
        old = choose(preferred[ds], ds, 16, "calibration_reuse", allow_history=True)
        fresh = choose(calibration[ds], ds, 16 - len(old), "calibration_fresh") if len(old) < 16 else []
        selected["calibration"][ds] = old + fresh
        require(len(old + fresh) == 16, "insufficient calibration source pool: " + ds)
    for role, target in (("development", DEV), ("reserve", RESERVE)):
        for ds in DS:
            result = choose(catalogs[ds], ds, target[ds], role)
            require(len(result) == target[ds], "insufficient fresh source pool: " + ds)
            selected[role][ds] = result
    # IDs and content selection is now complete. Decode question/reference text
    # only after source selection; calibration never opens reference columns.
    manifests, references, provenance = {}, {}, []
    selected_rows = {}
    for role, domains in selected.items():
        selected_rows[role] = {}
        for ds, values in domains.items():
            complete = [materializer.finish(c, r, role != "calibration") for c, r, _ in values]
            selected_rows[role][ds] = [r for r, _ in complete]
            for (c, r, reused), (finished, ref) in zip(values, complete):
                provenance.append(dict(role=role, dataset=ds, id=r["id"], source_id=r["source_id"], source_cluster_id=r["source_cluster_id"],
                                       image_sha256=r["original_image_sha256"], old_calibration_reused=reused, locator={k: v for k, v in c.items() if k not in ("input", "refs_file")}))
            if role != "calibration":
                references[(role, ds)] = [ref for _, ref in complete]
    files_to_write = {}
    cal_all, fit, select = [], [], []
    for ds in DS:
        values = sorted(selected_rows["calibration"][ds], key=lambda r: (priority(seed, "FitSelect:" + ds, r["source_cluster_id"]), r["id"]))
        cal_all += values
        fit += values[:12]
        select += values[12:]
    files_to_write.update(calibration64=cal_all, cal_fit48=fit, cal_select16=select)
    for role, target, suffix in (("development", DEV, "dev"), ("reserve", RESERVE, "reserve")):
        merged = []
        for ds in DS:
            key = ds + "_" + suffix + str(target[ds])
            files_to_write[key] = selected_rows[role][ds]
            merged += selected_rows[role][ds]
        files_to_write["D160" if role == "development" else "reserve224"] = merged
    input_files, ref_files = {}, {}
    for key, values in files_to_write.items():
        count, role = FILE_SPECS[key]
        validate_rows(values, expected_count=count)
        q = inputs / (key + "_inputs.jsonl")
        proof = freeze_rows(q, values)
        input_files[key] = dict(proof, path=plan["runtime_project_root"].rstrip("/") + "/configs/round10_inputs_v1/" + q.name,
                                count=count, role=role, domains=dict(Counter(r["dataset"] for r in values)), answers_included=False)
        if key not in ("calibration64", "cal_fit48", "cal_select16"):
            chosen_refs = {r["id"]: r for r in sum(list(references.values()), [])}
            refs = [chosen_refs[r["id"]] for r in values]
            target = inputs / (key + "_refs.jsonl")
            proof = freeze_rows(target, refs)
            ref_files[key] = dict(proof, path=plan["runtime_project_root"].rstrip("/") + "/configs/round10_inputs_v1/" + target.name,
                                  count=count, role="CPU_SCORING_ONLY", answers_not_encoder_inputs=True)
    proof = dict(schema="trajkv.round10.source_disjoint.v1", cal_fit48_vs_select16=0, calibration_vs_D160=0,
                 calibration_vs_reserve224=0, D160_vs_reserve224=0, selected_unique_sources=448,
                 identity_fields=["question_id", "dataset/source_id", "source_cluster_id including Doc parent", "original_image_sha256"],
                 source_selection="SHA256(seed NUL role/domain/source); QA=min SHA256(seed NUL question/domain/source/question_id)",
                 seed=seed, selection_reads_model_outcomes=False, reference_text_decoded_after_source_selection=True,
                 historical_coverage=coverage, coverage_limits=plan["historical_coverage_limits"],
                 prior_native_full_FP16_benchmark_is_not_compression_tuning=True, no_pretraining_unseen_or_absolute_historical_independence_claim=True,
                 calibration_reuse_counts=dict(Counter(r["dataset"] for r in provenance if r["role"] == "calibration" and r["old_calibration_reused"])))
    # Independent set intersections use completed rows, not assumed counts.
    split_sets = {name: Identities() for name in ("fit", "select", "dev", "reserve")}
    for name, values in (("fit", fit), ("select", select), ("dev", files_to_write["D160"]), ("reserve", files_to_write["reserve224"])):
        for r in values:
            require(not formal.reasons(r), "formal content/source overlap survived")
            split_sets[name].add(r)
            if name in ("dev", "reserve"):
                require(not blocked.reasons(r), "historical compressed selection overlap survived")
    for left, right in (("fit", "select"), ("fit", "dev"), ("select", "dev"), ("fit", "reserve"), ("select", "reserve"), ("dev", "reserve")):
        for attr in ("ids", "sources", "clusters", "shas"):
            require(not getattr(split_sets[left], attr) & getattr(split_sets[right], attr), "source/content split leakage")
    require(len(occupied.clusters) == len(occupied.shas) == 448, "unexpected duplicate selected original content")
    disjoint = freeze_json(output / "SOURCE_DISJOINT_PROOF.json", proof)
    locators = freeze_rows(output / "SELECTED_SOURCE_LOCATORS.jsonl", provenance)
    raw_sources = freeze_json(output / "RAW_SOURCE_PROOF.json", dict(sources=source_proof, all_parquet_LFS_SHA_verified=True))
    lock = freeze_json(inputs / "source_lock.json", dict(schema="trajkv.round10.source_lock.v1", files={Path(v["path"]).name: {k: v[k] for k in ("sha256", "bytes")} for v in list(input_files.values()) + list(ref_files.values())},
                      true_splits={k: sorted({r["split"] for r in v}) for k, v in files_to_write.items()}, calibration_references_exported=False))
    manifest = dict(schema=SCHEMA, status="READY_FROZEN_CPU_DATA_NO_MODEL_EXECUTION", input_files=input_files, reference_files=ref_files,
                    answers_in_encoder_inputs=False, calibration_mix={d: 16 for d in DS}, fit_mix={d: 12 for d in DS}, select_mix={d: 4 for d in DS},
                    development_sources=DEV, reserved_unused_extension_sources=RESERVE, seed=seed,
                    disjoint_proof=disjoint, source_locators=locators, raw_source_proof=raw_sources, source_lock=lock,
                    loader_source_sha256=sha(Path(__file__).with_name("round10_inputs_v1.py")), builder_source_sha256=sha(__file__),
                    plan_sha256=plan["_plan_sha256"], actual_model_or_GPU_calls=0, calibration_answer_text_decoded=False,
                    reference_bytes_hashed_for_source_lock_only_before_selection=True,
                    source_split_note="True source split is explicit per row/entry: Info train; Chart calibration val and development test; Text validation; Doc canonical validation or explicitly pinned train pool. No split relabelling or pretraining-unseen claim.")
    manifest_file = freeze_json(inputs / "INPUT_MANIFEST.json", manifest)
    for entry in input_files.values():
        load_round10_inputs(inputs / Path(entry["path"]).name, inputs / "INPUT_MANIFEST.json", entry["sha256"], entry["role"])
    freeze_json(output / "IMAGE_MATERIALIZATION.json", dict(extracted=list(materializer.images.values()), existing_canonical_images_verified=True))
    result = dict(status="PASS_ROUND10_CAL64_FIT48_SELECT16_D160_RESERVE224", manifest=manifest_file,
                  input_counts={k: v["count"] for k, v in input_files.items()}, unique_sources=448, answers_in_encoder_inputs=False,
                  calibration_reference_files_exported=False, no_model_GPU_calls=True, scientific_jobs_launched=False)
    freeze_json(output / "COMPLETE.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    a = p.parse_args()
    plan = read(a.plan)
    plan["_plan_sha256"] = sha(a.plan)
    build(plan)


if __name__ == "__main__":
    main()
