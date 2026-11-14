"""Offline DocVQA/GQA scoring of run_cycle records; no inference or judging API.

DocVQA primary ANLS ports the organizer's pinned MP-DocVQA evaluator: lowercase,
outer strip, max over references, similarity >= .5, empty prediction -> zero.
A strict-threshold auxiliary column exposes the paper/code boundary difference.
GQA primary accuracy uses exact answer identity, without CLEVR/VQA aliases.
ConvBench outputs only explicitly labelled lexical diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import analyze_cycle as shared

DOCVQA_COMMIT = "7939f521fe5a82a442f54831b1c0e96daab331d9"
DOCVQA_SOURCE = ("https://raw.githubusercontent.com/rubenpt91/MP-DocVQA-Framework/"
                 + DOCVQA_COMMIT + "/metrics.py")
DOCVQA_SOURCE_SHA256 = "481fe17e8d443a6dcf852b8348fcf06b7973ac1e99336f230a749008e992bf70"
PROTOCOLS = {
    "docvqa": "docvqa_organizer_code_anls_inclusive_0.5_v1",
    "gqa": "gqa_exact_answer_identity_grouped_subset_v1",
    "convbench": "convbench_lexical_diagnostic_not_official_quality_v1",
}
METRICS = {"docvqa": "ANLS_organizer_code", "gqa": "exact_accuracy",
           "convbench": "lexical_token_f1_diagnostic"}


def levenshtein(left: str, right: str) -> int:
    """Unit-cost character edit distance with O(min(length)) space."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def answer_list(targets: Any) -> list[str]:
    result = targets if isinstance(targets, (list, tuple)) else [targets]
    if not result or any(not isinstance(x, str) for x in result):
        raise ValueError("References must contain at least one string answer")
    return list(result)


def score_docvqa(prediction: str, targets: Any) -> dict[str, Any]:
    """String-answer subset of the pinned organizer evaluator, not RRC server.

    Punctuation, articles, numeric words and internal whitespace are retained.
    The upstream 'none' string special case is preserved for answerable items.
    This source's .5-inclusive boundary differs from NL < .5 in the paper.
    """
    if not isinstance(prediction, str):
        raise ValueError("Prediction must be a string")
    pred = prediction.lower().strip()
    gold = [x.lower().strip() for x in answer_list(targets)]
    exact = float(pred != "none" and pred in gold)
    similarities = ([1.0 - levenshtein(pred, g) / max(len(pred), len(g)) for g in gold]
                    if pred and pred != "none" else [])
    similarity = max(similarities, default=0.0)
    score = similarity if similarity >= .5 else 0.0
    strict = similarity if similarity > .5 else 0.0
    return {"score": score, "exact": exact, "anls": score,
            "anls_strict_threshold_aux": strict,
            "anls_threshold_boundary_hit": bool(similarities and similarity == .5),
            "normalized_prediction": pred, "normalized_targets": gold}


def score_gqa(prediction: str, targets: Any) -> dict[str, Any]:
    if not isinstance(prediction, str):
        raise ValueError("Prediction must be a string")
    gold = answer_list(targets)
    exact = float(prediction in gold)
    return {"score": exact, "exact": exact,
            "case_whitespace_exact_aux": float(prediction.lower().strip() in
                                                 [g.lower().strip() for g in gold]),
            "normalized_prediction": prediction, "normalized_targets": gold}


def score_convbench_lexical(prediction: str, targets: Any) -> dict[str, Any]:
    """Multiset word overlap only; it does not judge factual/semantic quality."""
    def tokens(text):
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    predicted = tokens(prediction)
    values = []
    for target in answer_list(targets):
        gold = tokens(target)
        common = sum((Counter(predicted) & Counter(gold)).values())
        values.append(2.0 * common / (len(predicted) + len(gold)) if predicted and gold else 0.0)
    return {"score": max(values), "exact": None, "lexical_diagnostic": True,
            "official_quality_metric": False, "normalized_prediction": " ".join(predicted)}


SCORERS = {"docvqa": score_docvqa, "gqa": score_gqa, "convbench": score_convbench_lexical}


def load_references(paths: list[Path]) -> dict:
    references = {}
    for path in paths:
        for row in shared.load_jsonl(path):
            sid = str(row.get("id", row.get("session_id", "")))
            if not sid:
                raise ValueError("Reference has no session ID")
            for item in row["references"]:
                if item.get("answer_type", "string") != "string":
                    raise ValueError("This SP-DocVQA adapter only supports answerable string references")
                targets = answer_list(item.get("answers", item.get("answer")))
                key = (sid, int(item["turn"]))
                if key in references and references[key] != targets:
                    raise ValueError(f"Conflicting references: {key}")
                references[key] = targets
    return references


def aggregate(rows, *, repeats, seed):
    estimate = shared.cluster_bootstrap(((r["image_id"], r["score"]) for r in rows),
                                         repeats=repeats, seed=seed)
    exact = [r["exact"] for r in rows if r.get("exact") is not None]
    auxiliary = [r["anls_strict_threshold_aux"] for r in rows if "anls_strict_threshold_aux" in r]
    return {**estimate, "exact_accuracy": sum(exact) / len(exact) if exact else None,
            "anls_strict_threshold_aux": sum(auxiliary) / len(auxiliary) if auxiliary else None,
            "threshold_boundary_questions": sum(r.get("anls_threshold_boundary_hit", False) for r in rows)}


def analyze(paths, references, samples, *, dataset, baseline="full:base", repeats=2000, seed=1212):
    scored, excluded, warnings = [], [], []
    seen = set()
    descriptions = {}
    for path in paths:
        settings = shared.infer_settings(path)
        for record in shared.load_jsonl(path):
            sid = str(record["session_id"])
            if record.get("quality_scoring_allowed") is False or record.get("protocol") != "free_running":
                excluded.append({"session_id": sid, "method": record["method"],
                                 "reason": "not_free_running_application_quality", "source": str(path)})
                continue
            sample = samples.get(sid, {})
            image = record.get("image_id", sample.get("source_image_id", sample.get("image_id")))
            if not image:
                raise ValueError(f"{sid}: source image cluster missing")
            observed_dataset = record.get("dataset", sample.get("dataset"))
            if observed_dataset and dataset not in str(observed_dataset).lower():
                raise ValueError(f"{sid}: dataset {observed_dataset!r} does not match {dataset!r}")
            native = str(record.get("native_session_id", sample.get("native_session_id", sid)))
            offset = int(record.get("reference_turn_offset", sample.get("reference_turn_offset", 0)))
            description = {"model": record.get("model"), "dataset": dataset,
                           "split": record.get("split", sample.get("split")),
                           "protocol": "free_running", "backend": record.get("backend"),
                           "data_protocol": record.get("data_protocol", sample.get("protocol")),
                           "settings": settings}
            group = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()[:12]
            descriptions[group] = description
            variant = record["method"] + ":" + record.get("policy", "base")
            identity = (group, sid, variant)
            if identity in seen:
                raise ValueError(f"Duplicate run: {identity}")
            seen.add(identity)
            for turn in record.get("turns", []):
                number = int(turn["turn"])
                ref_turn = number - offset
                if ref_turn <= 0:
                    continue
                key = (native, ref_turn)
                if key not in references:
                    raise ValueError(f"Missing reference: {key}")
                metrics = SCORERS[dataset](turn.get("answer", ""), references[key])
                scored.append({"group_id": group, "session_id": sid, "native_session_id": native,
                               "image_id": str(image), "turn": number, "reference_turn": ref_turn,
                               "method": record["method"], "policy": record.get("policy", "base"),
                               "variant": variant, "metric": METRICS[dataset],
                               "score_protocol": PROTOCOLS[dataset], "prediction": turn.get("answer", ""),
                               "reference_answers": references[key], **metrics})
    keys = [(r["group_id"], r["session_id"], r["reference_turn"], r["variant"]) for r in scored]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate scored turn")
    method_summary, turn_summary = [], []
    groups = defaultdict(list)
    for row in scored:
        groups[(row["group_id"], row["variant"])].append(row)
    for (group, variant), rows in sorted(groups.items()):
        for scope, chosen in (("all_scored", rows),
                              ("later_turns", [r for r in rows if r["reference_turn"] >= 2])):
            method_summary.append({"group_id": group, "variant": variant, "scope": scope,
                                   "metric": METRICS[dataset], **aggregate(chosen, repeats=repeats, seed=seed)})
        for turn in sorted({r["reference_turn"] for r in rows}):
            chosen = [r for r in rows if r["reference_turn"] == turn]
            turn_summary.append({"group_id": group, "variant": variant, "reference_turn": turn,
                                 "metric": METRICS[dataset], **aggregate(chosen, repeats=repeats, seed=seed)})
    paired, paired_summary = [], []
    for group in sorted(descriptions):
        variants = {r["variant"] for r in scored if r["group_id"] == group}
        if baseline not in variants:
            warnings.append(f"Group {group}: no {baseline}, no paired comparison")
            continue
        for variant in sorted(variants - {baseline}):
            base = {(r["session_id"], r["reference_turn"]): r for r in scored
                    if r["group_id"] == group and r["variant"] == baseline}
            method = {(r["session_id"], r["reference_turn"]): r for r in scored
                      if r["group_id"] == group and r["variant"] == variant}
            rows = []
            for key in sorted(base.keys() & method.keys()):
                a, b = base[key], method[key]
                if a["image_id"] != b["image_id"]:
                    raise ValueError(f"Paired source image mismatch: {key}")
                rows.append({"group_id": group, "session_id": key[0], "reference_turn": key[1],
                             "image_id": a["image_id"], "baseline": baseline, "variant": variant,
                             "baseline_prediction": a["prediction"], "method_prediction": b["prediction"],
                             "baseline_score": a["score"], "method_score": b["score"],
                             "delta": b["score"] - a["score"],
                             "exact_repaired": (a["exact"] == 0 and b["exact"] == 1)
                                 if a["exact"] is not None and b["exact"] is not None else None,
                             "exact_new_errors": (a["exact"] == 1 and b["exact"] == 0)
                                 if a["exact"] is not None and b["exact"] is not None else None,
                             "positive_score_repaired": a["score"] == 0 and b["score"] > 0,
                             "positive_score_new_errors": a["score"] > 0 and b["score"] == 0})
            paired.extend(rows)
            for scope, chosen in (("all_scored", rows),
                                  ("later_turns", [r for r in rows if r["reference_turn"] >= 2])):
                estimate = shared.cluster_bootstrap(((r["image_id"], r["delta"]) for r in chosen),
                                                     repeats=repeats, seed=seed)
                paired_summary.append({"group_id": group, "baseline": baseline, "variant": variant,
                                       "scope": scope, "metric": METRICS[dataset], **estimate,
                                       "improved_questions": sum(r["delta"] > 0 for r in chosen),
                                       "worsened_questions": sum(r["delta"] < 0 for r in chosen),
                                       "exact_repaired": sum(bool(r["exact_repaired"]) for r in chosen)
                                           if dataset != "convbench" else None,
                                       "exact_new_errors": sum(bool(r["exact_new_errors"]) for r in chosen)
                                           if dataset != "convbench" else None,
                                       "positive_score_repaired": sum(r["positive_score_repaired"] for r in chosen),
                                       "positive_score_new_errors": sum(r["positive_score_new_errors"] for r in chosen),
                                       "unmatched_baseline_questions": sum(scope == "all_scored" or k[1] >= 2
                                            for k in base.keys() - method.keys()),
                                       "unmatched_method_questions": sum(scope == "all_scored" or k[1] >= 2
                                            for k in method.keys() - base.keys())})
    return {"groups": descriptions, "scored_turns": scored, "excluded_records": excluded,
            "method_summary": method_summary, "turn_summary": turn_summary,
            "paired_diffs": paired, "paired_summary": paired_summary, "warnings": warnings}


def main(argv=None):
    shared._INPUT_BYTES.clear()
    shared._READ_WARNINGS.clear()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=tuple(PROTOCOLS))
    p.add_argument("--records", nargs="+", type=Path, required=True)
    p.add_argument("--references", nargs="+", type=Path, required=True)
    p.add_argument("--samples", nargs="*", type=Path, default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", default="full:base")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1212)
    args = p.parse_args(argv)
    references = load_references(args.references)
    result = analyze(args.records, references, shared.load_sample_metadata(args.samples),
                     dataset=args.dataset, baseline=args.baseline, repeats=args.bootstrap, seed=args.seed)
    completeness = [shared.input_completeness(path) for path in args.records]
    args.output.mkdir(parents=True, exist_ok=True)
    for key in ("scored_turns", "excluded_records", "method_summary", "turn_summary", "paired_diffs", "paired_summary"):
        shared.write_csv(args.output / (key + ".csv"), result[key])
    summary = {k: v for k, v in result.items() if k not in ("scored_turns", "paired_diffs")}
    partial = any(c["status"] != "COMPLETE" for c in completeness)
    summary.update({"status": ("PARTIAL" if partial else "OK") if result["scored_turns"] else "EMPTY_SCORED_INPUT",
                    "partial": partial,
                    "input_completeness": completeness,
                    "dataset": args.dataset, "primary_metric": METRICS[args.dataset],
                    "score_protocol": PROTOCOLS[args.dataset], "n_scored_turns": len(result["scored_turns"]),
                    "official_quality_metric": args.dataset != "convbench",
                    "official_leaderboard_submission": False,
                    "bootstrap": {"cluster": "source image", "repeats": args.bootstrap,
                                  "seed": args.seed, "interval": "paired percentile 95%"},
                    "inference_scope": "exploratory paired intervals; no multiplicity-adjusted significance claim",
                    "metric_sources": {"docvqa": DOCVQA_SOURCE,
                                       "docvqa_source_sha256": DOCVQA_SOURCE_SHA256,
                                       "gqa": "https://cs.stanford.edu/people/dorarad/gqa/evaluate.html"},
                    "metric_boundary_note": "ANLS organizer code includes similarity .5; strict-threshold auxiliary excludes it. Empty prediction scores zero.",
                    "reference_answers_read_only_by_offline_analysis": True,
                    "script_sha256": digest(Path(__file__)),
                    "input_sha256": {str(f.resolve()): shared.sha256(f)
                                     for f in args.records + args.references + args.samples}})
    summary["warnings"].extend(shared._READ_WARNINGS)
    shared.write_json(args.output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "dataset": args.dataset,
                      "scored_turns": summary["n_scored_turns"], "output": str(args.output)}))
    return 0


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
