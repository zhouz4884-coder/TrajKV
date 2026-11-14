"""Table contracts using explicit mathematical fixtures, not experiment data."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from policies.fit_gain_table import fit_gain_tables


def fixture_action(group, current, future):
    return dict(action=f"layer{group}:early", layer_group=group, phase="early", write_turn=2,
        current_fidelity=dict(content_fidelity_gain_mean=current, content_token_count=1,
                              kl_direction="D_KL(p_candidate || p_same_token_allref)"),
        current_read_cost=dict(totals=dict(base_visual_bytes=100, residual_bytes=20, text_bytes=50)),
        future=[dict(target_turn=3, fidelity=dict(content_fidelity_gain_mean=future, content_token_count=1))])


class GainTableChecks(unittest.TestCase):
    def test_matched_library_can_produce_different_current_future_rankings(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            rows = [dict(schema="trajkv.mechanism_fidelity.v2", session_id=f"s{i}",
                         image_id=f"im{i}", split="development",
                         actions=[fixture_action(0, .3, .05), fixture_action(1, .1, .4)]) for i in range(2)]
            path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
            result = fit_gain_tables([path])
            self.assertEqual(result["training_session_ids"], ["s0", "s1"])
            self.assertEqual(result["tables"]["current"]["phase_layer_group_ranking"]["early"], [0, 1])
            self.assertEqual(result["tables"]["future"]["phase_layer_group_ranking"]["early"], [1, 0])
            self.assertTrue(result["same_training_observations_for_current_and_future"])
            self.assertEqual(result["tables"]["future"]["entries"][0]["unit_residual_read_bytes"], 20)
            self.assertIn(str(path.resolve()), result["source_sha256"])

    def test_v1_likelihood_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            path.write_text(json.dumps(dict(schema="trajkv.mechanism.v1", session_id="s", actions=[]))+"\n")
            with self.assertRaisesRegex(ValueError, "v1 likelihood"):
                fit_gain_tables([path])


if __name__ == "__main__":
    unittest.main()
