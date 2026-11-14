"""Mathematical/API checks. Fixtures are test cases, never figure source data."""
import json
from pathlib import Path
import tempfile
import unittest

from analyze_cycle import (analyze_records, cluster_bootstrap, load_references,
                           main, score_answer, summarize_resources)


class AnalysisChecks(unittest.TestCase):
    def test_scoring_preserves_strict_short_answer_contract(self):
        self.assertEqual(score_answer("Answer: Two!", "2")["exact"], 1)
        self.assertEqual(score_answer("There are two red objects", "2")["exact"], 0)
        score = score_answer("block", "cube")
        self.assertEqual(score["exact"], 0)
        self.assertEqual(score["clevr_alias"], 1)

    def test_cluster_bootstrap_keeps_all_turns_of_one_image_together(self):
        result = cluster_bootstrap([("a", 1), ("a", 1), ("b", 0)], repeats=500)
        self.assertAlmostEqual(result["mean"], 2/3)
        self.assertEqual(result["n_images"], 2)
        self.assertEqual(result["n_questions"], 3)
        single = cluster_bootstrap([("a", 1), ("a", 0)], repeats=500)
        self.assertIsNone(single["ci_lower"])
        self.assertEqual(single["status"], "insufficient_clusters")

    def test_replay_read_cost_includes_text_and_base(self):
        result = summarize_resources({"turns": [{"token_ids": [1, 2], "margin": [.1, .9], "replay_count": 1}],
            "read_events": [dict(pass_kind="committed", base_visual_bytes=100, text_bytes=50, residual_bytes=0),
                            dict(pass_kind="replay", base_visual_bytes=100, text_bytes=50, residual_bytes=20)]})
        self.assertEqual(result["logical_total_read_bytes"], 320)
        self.assertEqual(result["replay_logical_bytes"], 170)
        self.assertEqual(result["replays_per_margin_step"], .5)
        self.assertFalse(result["logical_read_is_measured_hbm_traffic"])

    def test_probe_extra_counts_second_base_and_text_read_without_double_counting_local_two_read_charge(self):
        result=summarize_resources(dict(extra_logical_read_bytes=175,read_budget_bytes=175,read_ledger_exact=True,
            intra_step_probes=[dict(charged_bytes=305,decision=dict(upgrade=True))],
            read_events=[dict(pass_kind="committed",base_visual_bytes=100,text_bytes=50,residual_bytes=20),
                         dict(pass_kind="probe",base_visual_bytes=100,text_bytes=50,residual_bytes=5)]))
        self.assertEqual(result["computed_extra_logical_read_bytes"],175)
        self.assertEqual(result["probe_extra_logical_bytes"],155)
        self.assertEqual(result["controller_two_read_probe_charged_bytes"],305)
        self.assertEqual(result["logical_total_read_bytes"],325)
        self.assertTrue(result["extra_read_ledger_equal"])
        self.assertTrue(result["read_budget_ok_computed"])

    def test_compact_axis_layout_cannot_be_pooled_under_same_variant(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            refs=directory/"refs.jsonl";records=directory/"records.jsonl"
            refs.write_text("".join(json.dumps(dict(id=s,references=[dict(turn=1,answer="red")]))+"\n" for s in ("s1","s2")))
            rows=[dict(session_id=s,image_id=s,method="asym1_k",policy="always",protocol="free_running",split="development",
                       owned_initial=dict(visual_fp16_denominator=320,visual_base=30,visual_residual=res),
                       turns=[dict(turn=1,answer="red",token_ids=[1])]) for s,res in (("s1",50),("s2",20))]
            records.write_text("".join(json.dumps(r)+"\n" for r in rows))
            with self.assertRaisesRegex(ValueError,"Storage/kernel revision"):
                analyze_records([records],load_references([refs]),{},repeats=100,seed=2,baseline="full:base")

    def test_bridge_alignment_and_paired_flips(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            refs = directory / "refs.jsonl"
            records = directory / "records.jsonl"
            refs.write_text(json.dumps({"id": "s", "references": [{"turn": 1, "answer": "red"}]})+"\n")
            rows = []
            for method, answer in (("full", "red"), ("base2", "blue")):
                rows.append(dict(session_id="s_bridge", native_session_id="s", reference_turn_offset=1,
                    image_id="image_a", method=method, policy="base", split="development", protocol="free_running",
                    turns=[dict(turn=1, answer="description", token_ids=[1]), dict(turn=2, answer=answer, token_ids=[2])]))
            records.write_text("".join(json.dumps(r)+"\n" for r in rows))
            result = analyze_records([records], load_references([refs]), {}, repeats=200, seed=2, baseline="full:base")
            self.assertEqual(sum(r["scored"] for r in result["scored_turns"]), 2)
            self.assertEqual(result["paired_summary"][0]["delta_pp"], -100)
            self.assertEqual(result["paired_summary"][0]["new_errors"], 1)

    def test_empty_input_is_explicit_and_emits_no_figure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records, refs = directory / "records.jsonl", directory / "refs.jsonl"
            records.write_text("")
            refs.write_text("")
            output, figures = directory / "out", directory / "figures"
            self.assertEqual(main(["--records", str(records), "--references", str(refs),
                                  "--output", str(output), "--figures", str(figures)]), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "EMPTY_INPUT")
            self.assertEqual(summary["figures"], [])
            self.assertFalse(figures.exists())


if __name__ == "__main__":
    unittest.main()
