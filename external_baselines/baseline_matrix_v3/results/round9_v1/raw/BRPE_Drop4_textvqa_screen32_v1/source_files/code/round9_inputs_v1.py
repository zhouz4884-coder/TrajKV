"""Strict research input loader; actual Info TRAIN is never relabelled as val.

This independent module does not change the frozen official-split registry.
It imports only the standard library and works inside a code-only snapshot.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

FIELDS = {"id", "dataset", "dataset_revision", "split", "source_id", "source_cluster_id", "question_id",
          "image_path", "original_image_sha256", "width", "height", "question", "group", "image_format",
          "image_bytes", "original_split_row", "question_sha256"}
ALLOWED = {"infovqa": [("539088ef8a8ada01ac8e2e6d4e372586748a265e", "train")],
           "docvqa": [("539088ef8a8ada01ac8e2e6d4e372586748a265e", "validation")],
           "textvqa": [("9c0699cd19768ac5ab97568f6b3cbac4c0062884", "validation")]}


def _require(ok, message):
    if not ok:
        raise ValueError("ROUND9_INPUTS: " + message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_round9_inputs(path, manifest_path=None, expected_sha256=None):
    """Load only frozen no-answer rows, with a manifest and/or exact file SHA.

    Manifest paths are optional for isolated code snapshots. Without one an
    explicit expected_sha256 is mandatory. Counts/IDs/source hashes remain in
    the runner's input hash and records, not borrowed from old predictions.
    """
    path = Path(path)
    actual = _sha(path)
    _require(manifest_path is not None or expected_sha256 is not None, "frozen manifest or expected SHA required")
    if expected_sha256 is not None:
        _require(re.fullmatch("[0-9a-f]{64}", expected_sha256) is not None and actual == expected_sha256, "input SHA mismatch")
    frozen_count = None
    if manifest_path is not None:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
        _require(manifest.get("schema") == "trajkv.round9.inputs.v1", "wrong input manifest")
        entries = [r for r in manifest["input_files"].values() if Path(r["path"]).name == path.name]
        _require(len(entries) == 1 and entries[0]["sha256"] == actual, "file absent from frozen manifest")
        frozen_count = entries[0]["count"]
        _require(manifest["allowed_dataset_revision_splits"] == {k: [list(z) for z in v] for k, v in ALLOWED.items()}, "manifest source whitelist changed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    _require(bool(rows) and (frozen_count is None or len(rows) == frozen_count), "empty or wrong input count")
    for row in rows:
        _require(not (set(row) - FIELDS) and FIELDS - {"group"} <= set(row), "unexpected/reference or missing input fields")
        ds = row["dataset"]
        _require(ds in ALLOWED and (row["dataset_revision"], row["split"]) in ALLOWED[ds], "unapproved true split/revision")
        _require(isinstance(row["question_id"], str) and row["id"] == f'{ds}:{row["split"]}:{row["question_id"]}', "question identity mismatch")
        _require(isinstance(row["question"], str) and row["question"].strip(), "empty/non-string raw question")
        _require(hashlib.sha256(row["question"].encode()).hexdigest() == row["question_sha256"], "raw question SHA mismatch")
        _require(isinstance(row["original_image_sha256"], str) and re.fullmatch("[0-9a-f]{64}", row["original_image_sha256"]) is not None, "invalid original-image SHA")
        _require(all(type(row[k]) is int and row[k] > 0 for k in ("width", "height", "image_bytes")), "invalid original image geometry/size")
        _require(type(row["original_split_row"]) is int and row["original_split_row"] >= 0, "invalid source row")
        _require(isinstance(row["source_id"], str) and row["source_id"], "missing original source ID")
        expected_cluster = ("docvqa:ucsf:" + row["source_id"].rsplit("/", 1)[0] if ds == "docvqa"
                            else ds + ":image:" + row["source_id"])
        _require(row["source_cluster_id"] == expected_cluster, "source cluster/document identity mismatch")
        image = PurePosixPath(row["image_path"])
        _require(image.is_absolute() and ".." not in image.parts and "\\" not in row["image_path"], "invalid absolute data-disk image path")
        _require("/baseline_matrix_v3/data/" in row["image_path"] and image.stem == row["original_image_sha256"], "image path not canonical content-addressed project data")
    for field in ("id", "source_cluster_id", "original_image_sha256"):
        _require(len({row[field] for row in rows}) == len(rows), "duplicate question/source/image: " + field)
    return rows
