"""Build immutable QA/ref registry; stage raw image bytes only to project data disk.

Old Parquets/configs/images are read-only. A legacy image is hardlinked only
after full raw SHA identity. No reference or OCR field enters an input row.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from dataset_registry import DATASETS, SELECTION_SEED, load_inputs, load_jsonl, select_subset
from remote_control import REMOTE_ROOT, connect, mkdirs

OLD = ROOT.parent / "TrajKV"
LOCAL_PARQUETS = {"textvqa": OLD / "data/textvqa_validation_pinned/data",
                  "docvqa": OLD / "data/docvqa_validation_pinned",
                  "infovqa": OLD / "data/infovqa_validation_round8_pinned",
                  "chartqa": Path("${EVIDENCE_ROOT}_mirror/datasets/chartqa_af8b6f5c")}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(value, out, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def bytes_from_image(value):
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, list) and all(isinstance(x, int) and 0 <= x <= 255 for x in value):
        return bytes(value)
    raise ValueError("SOURCE_SCHEMA_UNSUPPORTED: image lacks original encoded bytes")


def source_entries(dataset):
    spec = DATASETS[dataset]
    name = spec["repo"].replace("/", "__") + "_info_v1.json"
    metadata_path = ROOT / "audit/dataset_sources" / name
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["sha"] == spec["revision"]
    rows = []
    for upstream in sorted(metadata["siblings"], key=lambda e: e["rfilename"]):
        if upstream["rfilename"].startswith(spec["prefix"]) and upstream["rfilename"].endswith(".parquet"):
            local = LOCAL_PARQUETS[dataset] / Path(upstream["rfilename"]).name
            expected = upstream["lfs"]["sha256"]
            assert local.stat().st_size == upstream["size"] and digest(local) == expected, "SOURCE_HASH_MISMATCH: " + str(local)
            rows.append({"local_path": str(local), "upstream_path": upstream["rfilename"], "bytes": local.stat().st_size,
                         "sha256": expected, "metadata_sha256": digest(metadata_path)})
    assert rows
    return rows


def build(version):
    import pyarrow.parquet as pq
    from PIL import Image
    directory = ROOT / "configs/datasets" / version
    directory.mkdir(parents=True, exist_ok=False)
    lock = {"version": version, "selection_seed": SELECTION_SEED,
            "selection_rule": "sha256(seed NUL dataset NUL source_cluster_id); first original split question; chart human/aug balanced distinct sources preferred",
            "input_has_reference_or_ocr": False, "sources": {}, "files": {}}
    for dataset, spec in DATASETS.items():
        sources = source_entries(dataset)
        all_rows, refs, locators = [], [], []
        for source in sources:
            parquet = pq.ParquetFile(source["local_path"])
            wanted = {"image", "question", "question_id", "image_id", "questionId", "ucsf_document_id",
                      "ucsf_document_page_no", "answers", "imgname", "query", "label", "type"}
            columns = [name for name in parquet.schema_arrow.names if name in wanted]
            file_row = 0
            for batch in parquet.iter_batches(batch_size=8, columns=columns):
                for raw in batch.to_pylist():
                    encoded = bytes_from_image(raw["image"])
                    sha = hashlib.sha256(encoded).hexdigest()
                    with Image.open(BytesIO(encoded)) as image:
                        width, height = image.size
                        fmt = image.format.lower()
                    extension = {"jpeg": "jpg", "mpo": "mpo", "png": "png", "webp": "webp", "tiff": "tiff", "bmp": "bmp", "gif": "gif"}[fmt]
                    raw_path = raw["image"].get("path") if isinstance(raw["image"], dict) else None
                    if dataset == "textvqa":
                        qid, source_id = str(raw["question_id"]), str(raw["image_id"])
                        cluster = "textvqa:image:" + source_id
                        answers = raw["answers"]
                        assert len(answers) == 10, "TextVQA must preserve original ten votes"
                    elif dataset == "docvqa":
                        qid = str(raw["questionId"])
                        source_id = str(raw["ucsf_document_id"]) + "/" + str(raw["ucsf_document_page_no"])
                        cluster = "docvqa:ucsf:" + str(raw["ucsf_document_id"])
                        answers = raw["answers"]
                    elif dataset == "infovqa":
                        qid, source_id = str(raw["questionId"]), str(raw_path)
                        assert raw_path
                        cluster, answers = "infovqa:image:" + source_id, raw["answers"]
                    else:
                        qid = "row_" + str(len(all_rows)).zfill(6)
                        source_id = str(raw.get("imgname") or "sha256:" + sha)
                        cluster, answers = "chartqa:image:" + source_id, [raw["label"]]
                    assert answers and all(isinstance(answer, str) for answer in answers), "Reference strings must remain source-exact"
                    question = raw["query"] if dataset == "chartqa" else raw["question"]
                    assert isinstance(question, str)
                    identity = dataset + ":" + spec["split"] + ":" + qid
                    row = {"id": identity, "dataset": dataset, "dataset_revision": spec["revision"], "split": spec["split"],
                           "source_id": source_id, "source_cluster_id": cluster, "question_id": qid,
                           "image_path": REMOTE_ROOT + "/data/images/" + sha + "." + extension,
                           "original_image_sha256": sha, "image_format": fmt, "image_bytes": len(encoded),
                           "width": width, "height": height, "question": question, "original_split_row": len(all_rows),
                           "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest()}
                    ref = {"id": identity, "answers": answers, "source_cluster_id": cluster}
                    if dataset == "chartqa":
                        assert raw["type"] in ("human", "aug")
                        row["group"] = ref["group"] = raw["type"]
                    all_rows.append(row)
                    refs.append(ref)
                    locators.append({"id": identity, "image_sha256": sha, "local_parquet": source["local_path"],
                                     "source_sha256": source["sha256"], "row_in_file": file_row})
                    file_row += 1
        assert len(all_rows) == spec["expected_questions"]
        assert len({r["id"] for r in all_rows}) == len(all_rows), "Do not drop duplicate official rows; investigate duplicated question IDs"
        # Freeze every full-split row and reference before deriving any subset.
        write_jsonl(directory / f"{dataset}_full_inputs.jsonl", all_rows)
        write_jsonl(directory / f"{dataset}_full_refs.jsonl", refs)
        write_jsonl(directory / f"{dataset}_locators.jsonl", locators)
        by_id = {r["id"]: r for r in refs}
        subsets = {}
        for stage in ("p1", "p2"):
            selected = select_subset(all_rows, stage)
            write_jsonl(directory / f"{dataset}_{stage}_inputs.jsonl", selected)
            write_jsonl(directory / f"{dataset}_{stage}_refs.jsonl", [by_id[r["id"]] for r in selected])
            subsets[stage] = {"questions": len(selected), "sources": len({r["source_cluster_id"] for r in selected}),
                              "groups": dict(Counter(r.get("group", "all") for r in selected))}
        lock["sources"][dataset] = {**spec, "parquets": sources, "actual_questions": len(all_rows),
                                      "actual_sources": len({r["source_cluster_id"] for r in all_rows}),
                                      "unique_raw_images": len({r["original_image_sha256"] for r in all_rows}),
                                      "groups": dict(Counter(r.get("group", "all") for r in all_rows)), "subsets": subsets}
        print(json.dumps({"event": "metadata_frozen", "dataset": dataset, "questions": len(all_rows),
                          "sources": lock["sources"][dataset]["actual_sources"], "subsets": subsets}), flush=True)
    for path in sorted(directory.glob("*.jsonl")):
        if path.name.endswith("inputs.jsonl"):
            load_inputs(path)
        lock["files"][path.name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    write_json(directory / "source_lock.json", lock)
    print(json.dumps({"status": "PASS", "registry": str(directory), "lock_sha256": digest(directory / "source_lock.json")}))


def legacy_map():
    paths = [OLD / "configs/a2/textvqa_standard5000.json", OLD / "configs/a2/round8_v1/infovqa_validation64.json"]
    result = {}
    for path in paths:
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            for sha, image in zip(row.get("hashes", []), row.get("remote_images", [])):
                result.setdefault(sha, image)
    return result


REMOTE_PREPARE = r'''
import hashlib,json,os,sys
from pathlib import Path
j=json.load(sys.stdin);base=Path(j['root']).resolve();images=base/'data/images';images.mkdir(parents=True,exist_ok=True)
old=base.parent/'TrajKV/data';result=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
for row in j['images']:
 p=Path(row['image_path']);assert p.parent.resolve()==images.resolve();expected=row['original_image_sha256']
 if p.exists():
  assert p.stat().st_size==row['image_bytes'] and sha(p)==expected
  result.append(dict(sha256=expected,status='VERIFIED_EXISTING',path=str(p)));continue
 candidate=row.get('candidate');s=Path(candidate) if candidate else None
 if s is not None and s.is_file() and str(s.resolve()).startswith(str(old.resolve())+'/') and sha(s)==expected:
  assert s.stat().st_size==row['image_bytes'];os.link(s,p);assert sha(p)==expected
  result.append(dict(sha256=expected,status='VERIFIED_LEGACY_HARDLINK',path=str(p),source=str(s),inode=p.stat().st_ino));continue
 result.append(dict(sha256=expected,status='MISSING',path=str(p)))
print(json.dumps(result))
'''

REMOTE_FINALIZE = r'''
import hashlib,json,os,sys
from pathlib import Path
j=json.load(sys.stdin);base=Path(j['root']).resolve();images=(base/'data/images').resolve();result=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
for row in j['images']:
 p=Path(row['image_path']);assert p.parent.resolve()==images;expected=row['original_image_sha256']
 if row.get('temporary'):
  t=Path(row['temporary']);assert t.parent.resolve()==images and t.name.startswith(p.name+'.upload_')
  assert t.stat().st_size==row['image_bytes'] and sha(t)==expected
  if not p.exists():os.link(t,p)
  assert sha(p)==expected;t.unlink()
 assert p.stat().st_size==row['image_bytes'] and sha(p)==expected
 result.append(dict(sha256=expected,path=str(p),bytes=p.stat().st_size,status='PASS'))
print(json.dumps(result))
'''


def remote_call(client, code, value):
    inp, out, err = client.exec_command(shlex.join(["/usr/bin/python3", "-B", "-c", code]), timeout=120)
    inp.write(json.dumps(value)); inp.flush(); inp.channel.shutdown_write()
    raw, error = out.read().decode(), err.read().decode()
    if out.channel.recv_exit_status():
        raise RuntimeError("Remote image identity check failed: " + error[-1600:])
    return json.loads(raw)


def stage(version, dataset, stage_name, run_id):
    import pyarrow.parquet as pq
    directory = ROOT / "configs/datasets" / version
    lock = json.loads((directory / "source_lock.json").read_text(encoding="utf-8"))
    path = directory / f"{dataset}_{stage_name}_inputs.jsonl"
    assert digest(path) == lock["files"][path.name]["sha256"]
    rows = load_inputs(path)
    unique = {r["original_image_sha256"]: r for r in rows}
    candidates = legacy_map()
    images = [{**r, "candidate": candidates.get(sha)} for sha, r in unique.items()]
    ledger = ROOT / "audit/dataset_staging" / f"{run_id}.json"
    assert not ledger.exists(), "Use a new run ID; never overwrite StageIdentity"
    client = connect()
    uploaded = []
    try:
        prepared = remote_call(client, REMOTE_PREPARE, {"root": REMOTE_ROOT, "images": images})
        missing = {r["sha256"] for r in prepared if r["status"] == "MISSING"}
        locators = {}
        for row in load_jsonl(directory / f"{dataset}_locators.jsonl"):
            if row["image_sha256"] in missing:
                locators.setdefault(row["image_sha256"], row)
        files = defaultdict(dict)
        for sha, location in locators.items():
            files[location["local_parquet"]][location["row_in_file"]] = sha
        sftp = client.open_sftp()
        try:
            for filename, needed in files.items():
                expected = next(l["source_sha256"] for l in locators.values() if l["local_parquet"] == filename)
                assert digest(filename) == expected
                offset = 0
                for batch in pq.ParquetFile(filename).iter_batches(batch_size=8, columns=["image"]):
                    for raw in batch.to_pylist():
                        sha = needed.get(offset)
                        offset += 1
                        if sha is None:
                            continue
                        encoded = bytes_from_image(raw["image"])
                        assert hashlib.sha256(encoded).hexdigest() == sha
                        row = unique[sha]
                        temporary = row["image_path"] + ".upload_" + run_id
                        with sftp.open(temporary, "wx") as output:
                            output.set_pipelined(True)
                            output.write(encoded)
                        uploaded.append({**row, "temporary": temporary})
                        if len(uploaded) % 32 == 0:
                            print(json.dumps({"event": "raw_images_streamed", "dataset": dataset, "uploaded": len(uploaded), "missing": len(missing)}), flush=True)
        finally:
            sftp.close()
        assert len(uploaded) == len(missing)
        finalized = remote_call(client, REMOTE_FINALIZE, {"root": REMOTE_ROOT, "images": uploaded})
        # Publish metadata only after selected image identity is verified. Other
        # full/sampling metadata may reference images staged by a later phase.
        sftp = client.open_sftp()
        metadata = []
        try:
            for local in sorted(directory.glob("*.json*")):
                if "_locators" in local.name:
                    continue
                is_ref = local.name.endswith("_refs.jsonl")
                target = REMOTE_ROOT + "/data/" + ("refs/" if is_ref else "registry/") + version + "/" + local.name
                mkdirs(sftp, str(Path(target).parent).replace("\\", "/"))
                raw = local.read_bytes()
                try:
                    with sftp.open(target, "rb") as f: assert f.read() == raw
                except FileNotFoundError:
                    with sftp.open(target, "wx") as f: f.write(raw)
                with sftp.open(target, "rb") as f: assert hashlib.sha256(f.read()).hexdigest() == hashlib.sha256(raw).hexdigest()
                metadata.append({"path": target, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        finally:
            sftp.close()
        result = {"status": "PASS", "dataset": dataset, "stage": stage_name, "version": version,
                  "questions": len(rows), "sources": len({r["source_cluster_id"] for r in rows}), "unique_raw_images": len(unique),
                  "prepared": prepared, "uploaded_and_verified": finalized, "metadata": metadata,
                  "input_manifest": REMOTE_ROOT + f"/data/registry/{version}/{path.name}",
                  "reference_manifest": REMOTE_ROOT + f"/data/refs/{version}/{dataset}_{stage_name}_refs.jsonl",
                  "source_lock_sha256": digest(directory / "source_lock.json"), "code_sha256": digest(Path(__file__)),
                  "registry_code_sha256": digest(ROOT / "code/dataset_registry.py")}
        write_json(ledger, result)
        print(json.dumps({k: result[k] for k in ("status", "dataset", "stage", "questions", "sources", "unique_raw_images", "input_manifest", "reference_manifest")}), flush=True)
    except Exception as exc:
        failure = {"status": "FAILED_RUN", "dataset": dataset, "stage": stage_name, "error_type": type(exc).__name__,
                   "error": str(exc), "uploaded_temporaries": [{"sha256": r["original_image_sha256"], "temporary": r["temporary"]} for r in uploaded]}
        write_json(ledger, failure)
        raise
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build"); b.add_argument("--version", required=True)
    s = sub.add_parser("stage"); s.add_argument("--version", required=True)
    s.add_argument("--dataset", choices=DATASETS, required=True); s.add_argument("--stage", choices=("p1", "p2", "full"), required=True)
    s.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.action == "build": build(args.version)
    else: stage(args.version, args.dataset, args.stage, args.run_id)
