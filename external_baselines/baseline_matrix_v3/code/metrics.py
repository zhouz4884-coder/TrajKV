"""Frozen offline scoring contracts; references never enter model inference.

Doc/Info primary ANLS is explicitly the commit-locked Qwen-VL vendor version,
not a claim about today's inaccessible RRC server. Its raw-uppercase-length
denominator is intentional. A normalized-length diagnostic is always emitted.
ChartQA follows the unmodified Pix2Struct relaxed_correctness semantics.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Any

from vendor_textvqa_metric import EvalAIAnswerProcessor, TextVQAAccuracyEvaluator

METRIC_CONTRACT_VERSION = "matrix_v3_metrics_v1"
TEXTVQA_METRIC_ID = "textvqa_mmf_ten_vote_leave_one_out_v1"
CHARTQA_METRIC_ID = "chartqa_pix2struct_relaxed_5pct_v1"
_TEXTVQA = TextVQAAccuracyEvaluator()
_ANSWER_PROCESSOR = EvalAIAnswerProcessor()


def levenshtein(a: str, b: str) -> int:
    if len(a) > len(b):
        a, b = b, a
    row = list(range(len(a) + 1))
    for j, cb in enumerate(b, 1):
        new = [j]
        for i, ca in enumerate(a, 1):
            new.append(min(row[i] + 1, new[-1] + 1, row[i - 1] + (ca != cb)))
        row = new
    return row[-1]


def normalize_anls(text: str) -> str:
    # Exactly the normalization in the locked vendor script; no punctuation or
    # articles are removed. Uppercase lengths below refer to ORIGINAL strings.
    return " ".join(text.strip().lower().split())


def anls_vendor_rawlen(prediction: str, references: list[str]) -> float:
    distances = []
    for answer in references:
        distance = levenshtein(normalize_anls(answer), normalize_anls(prediction))
        length = max(len(answer.upper()), len(prediction.upper()))
        distances.append(0.0 if length == 0 else distance / length)
    similarity = 1 - min(distances)
    return similarity if similarity >= 0.5 else 0.0


def anls_normalized_length(prediction: str, references: list[str]) -> float:
    pred = normalize_anls(prediction)
    similarities = []
    for answer in references:
        target = normalize_anls(answer)
        length = max(len(target), len(pred))
        similarities.append(1.0 if length == 0 else 1 - levenshtein(target, pred) / length)
    score = max(similarities)
    return score if score >= 0.5 else 0.0


def anls_author_strip(prediction: str, references: list[str]) -> float:
    # Rubèn Tito's released MP-DocVQA framework, normal answer_type='string'.
    pred = prediction.lower().strip()
    if not pred or pred == "none":
        return 0.0
    scores = [1 - levenshtein(answer.lower().strip(), pred) /
              max(len(answer.lower().strip()), len(pred)) for answer in references]
    score = max(scores)
    return score if score >= 0.5 else 0.0


def _to_float(text: str) -> float | None:
    try:
        if text.endswith("%"):
            return float(text.rstrip("%")) / 100.0
        return float(text)
    except ValueError:
        return None


def chartqa_relaxed_correctness(target: str, prediction: str,
                                max_relative_change: float = 0.05) -> bool:
    # Keep target_float truthiness, not `is not None`: the canonical function
    # falls back to case-insensitive exact strings when the numeric target is 0.
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        return abs(prediction_float - target_float) / abs(target_float) <= max_relative_change
    return prediction.lower() == target.lower()


def _dataset_and_group(dataset: str, reference: Any) -> tuple[str, str | None]:
    name = dataset.lower().replace("-", "_")
    group = None
    if name.startswith("chartqa_"):
        tail = name.removeprefix("chartqa_").removeprefix("test_")
        if tail in ("human", "aug", "augmented"):
            group = "human" if tail == "human" else "aug"
            name = "chartqa"
    name = {"textvqa_val": "textvqa", "docvqa_val": "docvqa", "infovqa_val": "infovqa",
            "infographicvqa": "infovqa", "infographicsvqa": "infovqa",
            "infographicsvqa_val": "infovqa"}.get(name, name)
    if name not in ("textvqa", "docvqa", "infovqa", "chartqa"):
        raise ValueError(f"Unknown dataset scoring contract: {dataset}")
    if isinstance(reference, dict):
        declared = reference.get("group", reference.get("chartqa_group", reference.get("subset")))
        if declared is not None and name == "chartqa":
            mapping = {"human": "human", "aug": "aug", "augmented": "aug"}
            if declared not in mapping:
                raise ValueError("ChartQA group must explicitly be human, aug or augmented")
            candidate = mapping[declared]
            if group is not None and group != candidate:
                raise ValueError("Conflicting ChartQA groups")
            group = candidate
    return name, group


def _reference_texts(reference: Any, dataset: str) -> list[str]:
    value = reference
    if isinstance(value, dict):
        for key in ("answers", "references", "annotation", "label", "answer"):
            if key in value:
                value = value[key]
                break
        else:
            raise ValueError("Reference object must provide answers/references/annotation/label/answer")
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError("Reference answers must be a nonempty list or string")
    answers = []
    for answer in value:
        if isinstance(answer, dict) and "answer" in answer:
            answer = answer["answer"]
        if not isinstance(answer, str):
            raise TypeError("Reference answer text must be a string; registry must preserve and explicitly convert source labels")
        answers.append(answer)
    if dataset == "textvqa" and len(answers) != 10:
        raise ValueError("Official TextVQA scoring requires exactly ten annotator answers")
    return answers


def score_question(dataset: str, prediction: str, reference: Any) -> dict[str, Any]:
    """Return one immutable-question score and normalization/metric provenance.

    `reference` may be a string, list of strings, or an object containing an
    answers/reference/label field plus explicit ChartQA group. No answer prefix,
    punctuation, decimal, sentence or EOS heuristic is applied to predictions.
    """
    if not isinstance(prediction, str):
        raise TypeError("Prediction must be the original decoded string")
    name, group = _dataset_and_group(dataset, reference)
    answers = _reference_texts(reference, name)
    diagnostics = {}
    if name == "textvqa":
        normalized = _ANSWER_PROCESSOR(prediction)
        normalized_answers = [_ANSWER_PROCESSOR(a) for a in answers]
        score = _TEXTVQA._compute_answer_scores(answers).get(normalized, 0.0)
        metric_id = TEXTVQA_METRIC_ID
    elif name in ("docvqa", "infovqa"):
        normalized = normalize_anls(prediction)
        normalized_answers = [normalize_anls(a) for a in answers]
        score = anls_vendor_rawlen(prediction, answers)
        metric_id = f"{name}_anls_qwenvl_vendor_rawlen_ge0p5_v1"
        diagnostics = {
            "anls_normalized_length_ge0p5_v1": anls_normalized_length(prediction, answers),
            "anls_docvqa_author_strip_ge0p5_v1": anls_author_strip(prediction, answers),
        }
    else:
        # Lowercase is only a display normalization. Numeric matching retains
        # original strings, including percent suffix and zero fallback behavior.
        normalized = prediction.lower()
        normalized_answers = [a.lower() for a in answers]
        score = float(any(chartqa_relaxed_correctness(a, prediction) for a in answers))
        metric_id = CHARTQA_METRIC_ID
    return {"dataset": name, "score": float(score), "official_score": float(score),
            "metric_id": metric_id, "metric_contract_version": METRIC_CONTRACT_VERSION,
            "raw_prediction": prediction, "reference_texts": answers,
            "normalized": normalized, "normalized_prediction": normalized,
            "normalized_references": normalized_answers,
            "exact": float(normalized in normalized_answers),
            "normalized_exact": float(normalized in normalized_answers),
            "raw_exact": float(prediction in answers), "group": group,
            "diagnostic_scores": diagnostics,
            "metric_provenance": ("Meta MMF official TextVQA scorer" if name == "textvqa" else
                                  "explicit vendor ANLS protocol; current server identity unverified" if name in ("docvqa", "infovqa") else
                                  "Google Pix2Struct primary relaxed-correctness implementation")}


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Question-weighted means, with ChartQA human/aug kept separately."""
    groups = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["metric_id"])
        groups[key].append(row)
    results = []
    for (dataset, metric_id), values in sorted(groups.items()):
        summaries = {}
        for group_name in ["overall", "human", "aug", "ungrouped"]:
            selected = values if group_name == "overall" else [v for v in values if (v.get("group") or "ungrouped") == group_name]
            if selected:
                summaries[group_name] = {"questions": len(selected), "score": sum(v["score"] for v in selected) / len(selected),
                                         "normalized_exact": sum(v["normalized_exact"] for v in selected) / len(selected)}
        results.append({"dataset": dataset, "metric_id": metric_id, "groups": summaries})
    return {"metric_contract_version": METRIC_CONTRACT_VERSION, "datasets": results}
