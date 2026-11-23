"""Frozen Round10 source inputs, independent of official-split registries.

Standard-library only. References are separate CPU assets and are never opened
by this loader. Actual train/val/test labels remain explicit.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

SCHEMA = "trajkv.round10.inputs.v1"
FIELDS = {"id", "dataset", "dataset_revision", "split", "source_id", "source_cluster_id", "question_id",
          "image_path", "original_image_sha256", "width", "height", "question", "group", "image_format",
          "image_bytes", "original_split_row", "question_sha256"}
ALLOWED = {
    "textvqa": ("9c0699cd19768ac5ab97568f6b3cbac4c0062884", {"validation"}),
    "docvqa": ("539088ef8a8ada01ac8e2e6d4e372586748a265e", {"train", "validation"}),
    "infovqa": ("539088ef8a8ada01ac8e2e6d4e372586748a265e", {"train"}),
    "chartqa": ("af8b6f5c08c95085271561c2a3f9d15f2b5a9031", {"train", "val", "test"}),
}
FILE_SPECS = {
    "calibration64": (64, "calibration_all"), "cal_fit48": (48, "cal_fit"),
    "cal_select16": (16, "cal_select"), "D160": (160, "development_audit"),
    "textvqa_dev64": (64, "development"), "infovqa_dev64": (64, "development"),
    "docvqa_dev16": (16, "development"), "chartqa_dev16": (16, "development"),
    "reserve224": (224, "reserved_unused_extension"),
    "textvqa_reserve64": (64, "reserved_unused_extension"),
    "infovqa_reserve64": (64, "reserved_unused_extension"),
    "docvqa_reserve48": (48, "reserved_unused_extension"),
    "chartqa_reserve48": (48, "reserved_unused_extension"),
}


def require(value, message):
    if not value:
        raise ValueError("ROUND10_INPUTS: " + message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_rows(rows, *, expected_count=None):
    require(rows and (expected_count is None or len(rows) == expected_count), "wrong input count")
    for row in rows:
        require(FIELDS - {"group"} <= set(row) <= FIELDS, "unknown/reference or missing input fields")
        ds = row["dataset"]
        require(ds in ALLOWED and row["dataset_revision"] == ALLOWED[ds][0] and row["split"] in ALLOWED[ds][1],
                "source revision/true split not allowed")
        require(isinstance(row["question_id"], str) and row["id"] == f'{ds}:{row["split"]}:{row["question_id"]}', "ID/source split differs")
        require(isinstance(row["question"], str) and row["question"].strip() and
                hashlib.sha256(row["question"].encode("utf-8")).hexdigest() == row["question_sha256"], "raw question SHA differs")
        require(isinstance(row["source_id"], str) and row["source_id"], "missing original source ID")
        cluster = ("docvqa:ucsf:" + row["source_id"].rsplit("/", 1)[0] if ds == "docvqa"
                   else ds + ":image:" + row["source_id"])
        require(row["source_cluster_id"] == cluster, "source/document cluster differs")
        require(re.fullmatch("[0-9a-f]{64}", row["original_image_sha256"]) is not None, "invalid original-byte image SHA")
        require(all(type(row[k]) is int and row[k] > 0 for k in ("width", "height", "image_bytes")), "invalid image geometry")
        require(type(row["original_split_row"]) is int and row["original_split_row"] >= 0, "invalid original source row")
        image = PurePosixPath(row["image_path"])
        require(image.is_absolute() and ".." not in image.parts and "\\" not in row["image_path"] and
                "/baseline_matrix_v3/data/" in row["image_path"] and image.stem == row["original_image_sha256"],
                "image path is not canonical content-addressed project data")
    for key in ("id", "source_cluster_id", "original_image_sha256"):
        require(len({row[key] for row in rows}) == len(rows), "duplicate source/content/question: " + key)
    return rows


def load_round10_inputs(path, manifest_path=None, expected_sha256=None, expected_role=None):
    path = Path(path)
    require(path.name.endswith("_inputs.jsonl"), "only no-answer input files can be loaded")
    key = path.name[:-len("_inputs.jsonl")]
    require(key in FILE_SPECS, "undeclared Round10 input cohort")
    count, role = FILE_SPECS[key]
    require(manifest_path is not None or expected_sha256 is not None, "frozen manifest or explicit SHA required")
    digest = sha(path)
    if expected_sha256 is not None:
        require(re.fullmatch("[0-9a-f]{64}", expected_sha256) is not None and digest == expected_sha256, "input SHA differs")
    if manifest_path is not None:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
        require(manifest.get("schema") == SCHEMA and manifest.get("answers_in_encoder_inputs") is False, "unknown input manifest")
        entry = manifest["input_files"][key]
        require(Path(str(entry["path"]).replace("\\", "/")).name == path.name and entry["sha256"] == digest and
                entry["bytes"] == path.stat().st_size and entry["count"] == count and entry["role"] == role and entry["answers_included"] is False,
                "manifest input binding differs")
    if expected_role is not None:
        require(expected_role == role, "wrong calibration/development/reserve role")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return validate_rows(rows, expected_count=count)
