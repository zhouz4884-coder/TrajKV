"""Scoring edge cases and pairing safeguards; no model or GPU dependencies."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_external as external


class MetricBoundaries(unittest.TestCase):
    def test_organizer_boundary_and_empty_prediction(self):
        half = external.score_docvqa("ab", ["ac"])
        self.assertEqual(half["anls"], .5)
        self.assertEqual(half["anls_strict_threshold_aux"], 0)
        self.assertTrue(half["anls_threshold_boundary_hit"])
        self.assertEqual(external.score_docvqa("", [""])["anls"], 0)
        self.assertEqual(external.score_docvqa(" ", ["answer"])["anls"], 0)
        self.assertEqual(external.score_docvqa("abcde", ["abxyz"])["anls"], 0)

    def test_docvqa_only_case_and_outer_whitespace_are_normalized(self):
        self.assertEqual(external.score_docvqa("  AbC\n", ["abc"])["anls"], 1)
        self.assertEqual(external.score_docvqa("a  b", ["a b"])["anls"], .75)
        self.assertEqual(external.score_docvqa("yes.", ["yes"])["anls"], .75)
        self.assertEqual(external.score_docvqa("Café", ["cafe"])["anls"], .75)
        self.assertEqual(external.score_docvqa("red", ["blue", "red"])["anls"], 1)
        self.assertEqual(external.score_docvqa("none", ["none"])["anls"], 0)
        with self.assertRaises(ValueError):
            external.score_docvqa("red", [])

    def test_gqa_is_not_clevr_or_vqa_normalization(self):
        self.assertEqual(external.score_gqa("red", ["red"])["score"], 1)
        for pred, gold in (("Red", "red"), ("red.", "red"), ("Answer: red", "red"),
                           ("two", "2"), ("ball", "sphere"), ("not red", "red")):
            self.assertEqual(external.score_gqa(pred, [gold])["score"], 0)
        self.assertEqual(external.score_gqa(" Red ", ["red"])["case_whitespace_exact_aux"], 1)

    def test_convbench_overlap_is_only_diagnostic(self):
        result = external.score_convbench_lexical("red red", ["red"])
        self.assertAlmostEqual(result["score"], 2 / 3)
        self.assertFalse(result["official_quality_metric"])
        self.assertIsNone(result["exact"])


class PairedSafeguards(unittest.TestCase):
    def run_fixture(self, records, references):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
            return external.analyze([path], references, {}, dataset="gqa",
                                    baseline="full:base", repeats=200, seed=1)

    def record(self, sid, method, answer, *, protocol="free_running", image="same-image"):
        return {"session_id": sid, "method": method, "policy": "base", "image_id": image,
                "dataset": "gqa", "protocol": protocol, "model": "fixture",
                "backend": "fixture", "split": "dev",
                "turns": [{"turn": 1, "answer": answer}]}

    def test_repeated_dialogs_cluster_by_original_image(self):
        records = [self.record("s1", "full", "bad"), self.record("s1", "base2", "yes"),
                   self.record("s2", "full", "yes"), self.record("s2", "base2", "bad")]
        result = self.run_fixture(records, {("s1", 1): ["yes"], ("s2", 1): ["yes"]})
        pair = next(r for r in result["paired_summary"] if r["scope"] == "all_scored")
        self.assertEqual(pair["n_images"], 1)
        self.assertEqual(pair["n_questions"], 2)
        self.assertEqual(pair["mean"], 0)
        self.assertIsNone(pair["ci_lower"])
        self.assertEqual((pair["exact_repaired"], pair["exact_new_errors"]), (1, 1))

    def test_forced_outputs_are_excluded_from_quality(self):
        result = self.run_fixture([self.record("s1", "full", "yes"),
                                   self.record("s1", "base2", "yes", protocol="fixed_tokens")],
                                  {("s1", 1): ["yes"]})
        self.assertEqual(len(result["scored_turns"]), 1)
        self.assertEqual(len(result["excluded_records"]), 1)
        self.assertFalse(result["paired_summary"])

    def test_missing_reference_and_duplicate_run_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Missing reference"):
            self.run_fixture([self.record("s1", "full", "yes")], {})
        row = self.record("s1", "full", "yes")
        with self.assertRaisesRegex(ValueError, "Duplicate run"):
            self.run_fixture([row, row], {("s1", 1): ["yes"]})

    def test_pairing_never_crosses_original_image_identity(self):
        with self.assertRaisesRegex(ValueError, "source image mismatch"):
            self.run_fixture([self.record("s1", "full", "yes", image="a"),
                              self.record("s1", "base2", "no", image="b")],
                             {("s1", 1): ["yes"]})


if __name__ == "__main__":
    unittest.main()
