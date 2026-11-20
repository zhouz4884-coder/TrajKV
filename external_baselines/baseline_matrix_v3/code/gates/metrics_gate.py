"""CPU contract checks against locked primary scoring functions and fixtures."""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from metrics import score_question, aggregate_scores, anls_normalized_length


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(path, names, namespace):
    selected = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node.decorator_list = []
            if isinstance(node, ast.ClassDef) and node.name == "EvalAIAnswerProcessor":
                node.bases = []
            selected.append(node)
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "audit/metrics_gate_v1.json")
    args = parser.parse_args()
    assert not args.output.exists(), "Frozen gate exists; use a new output filename"
    lock_path = ROOT / "audit/upstream/metrics_source_lock_v1.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for source in lock["sources"]:
        assert sha(ROOT / source["local_path"]) == source["sha256"]
    assert sha(ROOT / "code/vendor_textvqa_metric.py") == lock["textvqa_port_sha256"]
    source_root = ROOT / "third_party/metric_sources"
    text_namespace = extract(source_root / "mmf_processors.py", ["EvalAIAnswerProcessor"], {"re": re})
    text_namespace = extract(source_root / "mmf_m4c_evaluators.py", ["TextVQAAccuracyEvaluator"], text_namespace)
    official_text = text_namespace["TextVQAAccuracyEvaluator"]()
    vendor_namespace = {"__name__": "locked_vendor_reference"}
    exec(compile((source_root / "qwenvl_infographicsvqa_eval.py").read_text(encoding="utf-8"), "locked_qwenvl_vendor", "exec"), vendor_namespace)
    chart_namespace = extract(source_root / "pix2struct_metrics.py", ["relaxed_correctness"], {"Optional": Optional})
    # The independent edit-distance function comes from the locked vendor, not
    # the implementation under test. Remove only the optional C-extension import.
    author_namespace = extract(source_root / "docvqa_author_metrics.py", ["Evaluator"],
                               {"editdistance": SimpleNamespace(eval=vendor_namespace["levenshtein_distance"])})
    author = author_namespace["Evaluator"]()
    pix_anls = extract(source_root / "pix2struct_metrics.py", ["anls_metric"],
                       {"editdistance": SimpleNamespace(eval=vendor_namespace["levenshtein_distance"])})["anls_metric"]
    cases = []
    for count in range(11):
        references = ["red"] * count + ["blue"] * (10 - count)
        result = score_question("textvqa", "red", references)
        reference_score = official_text.eval_pred_list([{"pred_answer": "red", "gt_answers": references}])
        expected = [0., .3, .6, .9, 1., 1., 1., 1., 1., 1., 1.][count]
        assert abs(result["score"] - reference_score) < 1e-14
        assert abs(result["score"] - expected) < 1e-14
        cases.append({"kind": "textvqa_votes", "votes": count, "score": result["score"], "reference_score": reference_score})
    normalization_pairs = [("The ONE!", "1"), ("dont", "don't"), ("two", "2"),
                           ("acme, inc.", "acme inc"), ("A Cat\n", "cat"), ("3.14", "314")]
    for prediction, target in normalization_pairs:
        references = [target] * 10
        result = score_question("textvqa", prediction, references)
        reference_score = official_text.eval_pred_list([{"pred_answer": prediction, "gt_answers": references}])
        assert result["score"] == reference_score
        assert result["normalized"] == official_text.answer_processor(prediction)
        cases.append({"kind": "textvqa_normalization", "prediction": prediction, "references": references,
                      "score": result["score"], "reference_score": reference_score})
    rng = random.Random(1927)
    vocabulary = ["red", "blue", "one", "1", "two", "2", "the cat", "cat!", "3.14", "314", "don't", "dont", "zero", "0"]
    for index in range(200):
        references = [rng.choice(vocabulary) for _ in range(10)]
        prediction = rng.choice(vocabulary)
        result = score_question("textvqa", prediction, references)
        reference_score = official_text.eval_pred_list([{"pred_answer": prediction, "gt_answers": references}])
        assert result["score"] == reference_score
        cases.append({"kind": "textvqa_source_crosscheck", "index": index, "prediction": prediction,
                      "references": references, "score": result["score"], "reference_score": reference_score})

    doc_cases = [
        ("1970", ["1985"]), ("  ax  ", ["ab"]), ("A  B", ["a b"]),
        ("no.", ["no"]), ("the cat", ["cat"]), ("abc", ["zzz", "abq"]),
        ("", [""]), ("none", ["none"]), ("strabe", ["straße"]),
        ("ab\ncd", ["ab cd"]), ("a", ["z"]), ("ACME", ["acme"])]
    fixture_dir = args.output.parent / (args.output.stem + "_fixtures")
    fixture_dir.mkdir(parents=True, exist_ok=False)
    for dataset in ["docvqa", "infovqa"]:
        gt = {"dataset_name": dataset, "data": [{"questionId": i, "question": f"synthetic boundary {i}", "answers": references}
                                                 for i, (_, references) in enumerate(doc_cases)]}
        submission = [{"questionId": i, "answer": prediction} for i, (prediction, _) in enumerate(doc_cases)]
        gt_path, sub_path = fixture_dir / f"{dataset}_gt.json", fixture_dir / f"{dataset}_pred.json"
        gt_path.write_text(json.dumps(gt), encoding="utf-8")
        sub_path.write_text(json.dumps(submission), encoding="utf-8")
        official = vendor_namespace["evaluate_method"](str(gt_path), str(sub_path), SimpleNamespace(answer_types=False, anls_threshold=.5))
        author_scores = author.get_metrics([refs for _, refs in doc_cases], [p for p, _ in doc_cases])
        for index, (prediction, references) in enumerate(doc_cases):
            result = score_question(dataset, prediction, references)
            reference_score = official["per_sample_result"][str(index)]["score"]
            assert result["score"] == reference_score
            assert result["diagnostic_scores"]["anls_docvqa_author_strip_ge0p5_v1"] == author_scores["anls"][index]
            normalized_pred = " ".join(prediction.lower().split())
            independent_similarities = []
            for ref in references:
                norm_ref = " ".join(ref.lower().split())
                length = max(len(norm_ref), len(normalized_pred))
                independent_similarities.append(1. if length == 0 else 1 - vendor_namespace["levenshtein_distance"](norm_ref, normalized_pred) / length)
            expected_norm = max(independent_similarities)
            expected_norm = expected_norm if expected_norm >= .5 else 0.
            assert anls_normalized_length(prediction, references) == expected_norm
            cases.append({"kind": dataset + "_boundary", "prediction": prediction, "references": references,
                          "score": result["score"], "reference_score": reference_score,
                          "diagnostic_scores": result["diagnostic_scores"]})
    # Explicitly expose the strict-inequality disagreement rather than silently
    # adopting the variant with a higher score.
    assert pix_anls("1985", "1970") == 0
    assert score_question("docvqa", "1970", ["1985"])["score"] == .5
    assert score_question("docvqa", "  ax  ", ["ab"])["score"] > .5
    assert score_question("docvqa", "  ax  ", ["ab"])["diagnostic_scores"]["anls_normalized_length_ge0p5_v1"] == .5

    chart_cases = [
        ("100", "105", True), ("100", "105.0001", False), ("100", "95", True),
        ("-100", "-95", True), ("50%", "0.5", True), ("0.5", "50%", True),
        ("0", "0.0", False), ("0.0", "0", False), ("0", "0", True),
        ("0%", "0", False), ("yes", "YES", True), ("yes", "yes.", False),
        ("No", "No\n", False), ("one", "1", False), ("1,000", "1000", False),
        ("$100", "100", False), ("50%", "50", False), ("5%", "0.05", True),
        ("50% ", ".5", False), ("0", " 0", False), ("1", " 1 ", True),
        ("0", "-0", False), ("NaN", "NaN", False), ("inf", "inf", False)]
    for target, prediction, expected in chart_cases:
        result = score_question("chartqa", prediction, [target])
        reference_score = chart_namespace["relaxed_correctness"](target, prediction)
        assert bool(result["score"]) == reference_score == expected, (target, prediction, result, reference_score)
        cases.append({"kind": "chartqa_boundary", "prediction": prediction, "references": [target],
                      "score": result["score"], "reference_score": float(reference_score)})
    for index in range(200):
        value = rng.choice([0., -100., -1., 1., 10., 100., 10000.])
        target = str(value) if index % 3 else str(value) + "%"
        prediction = str(value * rng.choice([.9, .9499, .95, 1., 1.05, 1.0501, 1.1]))
        if index % 3 == 0:
            prediction += "%"
        result = score_question("chartqa", prediction, [target])
        reference_score = chart_namespace["relaxed_correctness"](target, prediction)
        assert bool(result["score"]) == reference_score
        cases.append({"kind": "chartqa_source_crosscheck", "index": index, "prediction": prediction,
                      "references": [target], "score": result["score"], "reference_score": float(reference_score)})
    original_reference = {"answers": ["YES"], "group": "human"}
    snapshot = deepcopy(original_reference)
    group_rows = [score_question("chartqa", "YES", original_reference),
                  score_question("chartqa_human", "no", ["yes"]),
                  score_question("chartqa", "100", {"answers": ["100"], "group": "augmented"}),
                  score_question("chartqa_test_augmented", "1000", ["100"])]
    assert original_reference == snapshot
    assert group_rows[0]["raw_prediction"] == "YES"
    summary = aggregate_scores(group_rows)
    group_summary = summary["datasets"][0]["groups"]
    assert group_summary["overall"]["score"] == .5
    assert group_summary["human"]["score"] == .5 and group_summary["human"]["questions"] == 2
    assert group_summary["aug"]["score"] == .5 and group_summary["aug"]["questions"] == 2
    rejected = []
    for dataset, prediction, reference in [("textvqa", "red", ["red"] * 9),
                                           ("docvqa", "answer", []),
                                           ("chartqa", "1", {"answers": ["1"], "group": "unknown"})]:
        try:
            score_question(dataset, prediction, reference)
        except (ValueError, TypeError) as exc:
            rejected.append(type(exc).__name__)
        else:
            raise AssertionError("Invalid metric input silently accepted")
    result = {"status": "PASS_CPU_OFFICIAL_SOURCE_CONTRACTS", "cases": len(cases), "case_results": cases,
              "source_functions_compared": ["MMF TextVQAAccuracyEvaluator + EvalAIAnswerProcessor",
                                            "Qwen-VL vendor evaluate_method on real JSON fixture files",
                                            "Pix2Struct relaxed_correctness AST extracted unchanged",
                                            "DocVQA author Evaluator diagnostic", "Pix2Struct strict threshold boundary"],
              "chartqa_group_aggregation": summary, "invalid_inputs_rejected": len(rejected),
              "main_anls_identity": "Predeclared explicit Qwen vendor semantics; current official RRC server code unavailable",
              "reference_data_in_model": False, "quality_measurements": False,
              "source_lock_sha256": sha(lock_path),
              "code_hashes": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in
                              [Path(__file__), ROOT / "code/metrics.py", ROOT / "code/vendor_textvqa_metric.py"]}}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"status": result["status"], "cases": result["cases"], "invalid_inputs_rejected": len(rejected),
                      "output": str(args.output), "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
