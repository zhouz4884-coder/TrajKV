"""Frozen official-split identities and score-blind source-cluster subsets.

Inference inputs and references are separate files. This module never loads a
model, reads model scores, changes raw question text, or normalizes answers.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

SELECTION_SEED = 20261119
DATASETS = {
    "textvqa": {"repo": "lmms-lab-encoder/textvqa", "revision": "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
                "split": "validation", "expected_questions": 5000, "prefix": "data/validation-", "cluster": "image_id"},
    "docvqa": {"repo": "lmms-lab-encoder/DocVQA", "revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
               "split": "validation", "expected_questions": 5349, "prefix": "DocVQA/validation-", "cluster": "ucsf_document_id"},
    "infovqa": {"repo": "lmms-lab-encoder/DocVQA", "revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
                "split": "validation", "expected_questions": 2801, "prefix": "InfographicVQA/validation-", "cluster": "image.path"},
    "chartqa": {"repo": "ahmed-masry/ChartQA", "revision": "af8b6f5c08c95085271561c2a3f9d15f2b5a9031",
                "split": "test", "expected_questions": 2500, "prefix": "data/test-", "cluster": "imgname_or_raw_sha"},
}
INPUT_FIELDS = {"id", "dataset", "dataset_revision", "split", "source_id", "source_cluster_id", "question_id",
                "image_path", "original_image_sha256", "width", "height", "question", "group", "image_format",
                "image_bytes", "original_split_row", "question_sha256"}


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_inputs(path):
    rows = load_jsonl(path)
    for row in rows:
        unknown = set(row) - INPUT_FIELDS
        if unknown:
            raise ValueError("INPUT_CONTRACT_VIOLATION: unexpected/ref/OCR fields: " + repr(sorted(unknown)))
        for field in INPUT_FIELDS - {"group"}:
            if field not in row:
                raise ValueError("INPUT_CONTRACT_VIOLATION: missing " + field)
        spec = DATASETS[row["dataset"]]
        if row["dataset_revision"] != spec["revision"] or row["split"] != spec["split"]:
            raise ValueError("INPUT_CONTRACT_VIOLATION: source revision/split changed")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("INPUT_CONTRACT_VIOLATION: duplicate official question identity")
    return rows


def load_references(path):
    rows = load_jsonl(path)
    return {row["id"]: row for row in rows}


def cluster_priority(dataset, cluster_id):
    return hashlib.sha256(f"{SELECTION_SEED}\0{dataset}\0{cluster_id}".encode("utf-8")).hexdigest()


def select_subset(rows, stage):
    if stage in ("full", "p3"):
        return list(rows)
    dataset = rows[0]["dataset"]
    if stage not in ("p1", "p2"):
        raise ValueError("stage must be p1/p2/full")
    # First question is the first ORIGINAL split row in the source cluster.
    first = {}
    for row in sorted(rows, key=lambda r: r["original_split_row"]):
        key = (row.get("group"), row["source_cluster_id"]) if dataset == "chartqa" else row["source_cluster_id"]
        first.setdefault(key, row)
    if dataset != "chartqa":
        limit = 128 if stage == "p1" else 500
        return sorted(first.values(), key=lambda r: (cluster_priority(dataset, r["source_cluster_id"]), r["source_cluster_id"]))[:limit]
    limit = 64 if stage == "p1" else 250
    selected, used = [], set()
    for group in ("human", "aug"):
        candidates = sorted((r for r in first.values() if r.get("group") == group),
                            key=lambda r: (cluster_priority(dataset, r["source_cluster_id"]), r["source_cluster_id"]))
        unique = [r for r in candidates if r["source_cluster_id"] not in used]
        overlap = [r for r in candidates if r["source_cluster_id"] in used]
        chosen = (unique + overlap)[:limit]
        selected.extend(chosen)
        used.update(r["source_cluster_id"] for r in chosen)
    return selected
