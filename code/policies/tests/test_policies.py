"""CPU-only checks for resource accounting, causal feature use and commits."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from policies import (Availability, Budget, Cost, CostLedger, Feature,
                      FeatureAvailabilityError, MarginTransaction, ReadEvent,
                      TokenSpan, make_policy)


def event(index=0, *, margin=0.1, carrier=False, residual_bytes=20):
    return ReadEvent(event_id=f"s0:t1:p{index}", session_id="s0", turn=1,
                     token_start=index, token_end=index + 1, step_index=index,
                     base_read_bytes=100, residual_read_bytes=residual_bytes,
                     features={"margin": Feature(margin), "fact_carrier": Feature(carrier),
                               "oracle_future_gain": Feature(index + 1, Availability.FUTURE)})


class PolicyContractTests(unittest.TestCase):
    def test_online_policies_reject_privileged_features_they_use(self):
        e = event()
        e = replace(e, features={"margin": Feature(0.01, Availability.REFERENCE)})
        with self.assertRaises(FeatureAvailabilityError):
            make_policy("current_margin", threshold=0.2).allocate([e], Budget())
        e = replace(e, unit_key="x")
        with self.assertRaises(FeatureAvailabilityError):
            make_policy("future_gain_table", table={"x": Feature(3, Availability.FUTURE)}).allocate([e], Budget())

    def test_margin_cost_includes_replay_not_just_residual(self):
        # 20 residual bytes alone would fit twice. Actual extra traffic is
        # 100 reread base + 20 residual, so only one refinement fits.
        p = make_policy("current_margin", threshold=0.2)
        result = p.allocate([event(0), event(1)], Budget(max_extra_read_bytes=120))
        self.assertEqual(result.selected_ids, ("s0:t1:p0",))
        self.assertEqual(result.projected_cost.base_read_bytes, 200)
        self.assertEqual(result.projected_cost.replay_read_bytes, 100)
        self.assertEqual(result.projected_cost.residual_read_bytes, 20)
        self.assertEqual(result.projected_cost.total_read_bytes, 320)

    def test_trial_commit_and_actual_overrun_are_visible(self):
        ledger = CostLedger()
        e = event()
        tx = MarginTransaction(e, make_policy("current_margin", threshold=0.2),
                               Budget(max_extra_read_bytes=120), ledger)
        decision = tx.trial(actual_trial_cost=Cost(base_read_bytes=100))
        self.assertTrue(decision.refined)
        self.assertEqual(ledger.total.total_read_bytes, 0)
        actual = tx.commit(refined=True,
                           actual_replay_cost=Cost(residual_read_bytes=20, replay_read_bytes=105),
                           cost_source="counted")
        self.assertFalse(actual.budget_ok)
        self.assertEqual(ledger.total.extra_read_bytes, 125)
        self.assertEqual(actual.cost_source, "counted")
        with self.assertRaises(RuntimeError):
            tx.commit(refined=True)

    def test_untriggered_and_aborted_trials_still_cost_reads(self):
        ledger = CostLedger()
        tx = MarginTransaction(event(margin=0.9), make_policy("current_margin", threshold=0.2),
                               Budget(), ledger)
        self.assertFalse(tx.trial().refined)
        with self.assertRaises(RuntimeError):
            tx.commit(refined=True)
        tx.abort()
        self.assertEqual(ledger.total.base_read_bytes, 100)
        self.assertEqual(ledger.total.extra_read_bytes, 0)

    def test_fact_carriers_include_single_tokens_and_spans(self):
        p = make_policy("fact_carrier_margin", threshold=0.2)
        short = replace(event(0), carrier_spans=(TokenSpan(0, 1, label="count"),))
        split_entity = replace(event(2), carrier_spans=(TokenSpan(1, 4, label="entity"),))
        noncarrier = event(4)
        result = p.allocate([short, split_entity, noncarrier], Budget())
        self.assertEqual(result.selected_ids, (short.event_id, split_entity.event_id))

    def test_oracle_is_explicit_and_uses_future_gain(self):
        result = make_policy("offline_oracle").allocate([event(i) for i in range(3)],
                                                        Budget(max_extra_read_bytes=20))
        self.assertEqual(result.selected_ids, ("s0:t1:p2",))
        self.assertTrue(result.oracle)
        self.assertTrue(all(d.oracle for d in result.decisions))
        self.assertEqual(result.projected_cost.base_read_bytes, 300)
        self.assertEqual(result.projected_cost.extra_read_bytes, 20)

    def test_schedules_are_causal_and_seed_reproducible(self):
        events = [event(i) for i in range(12)]
        p = make_policy("uniform", rate=0.25)
        chunks = p.allocate(events[:5], Budget()).selected_ids + p.allocate(events[5:], Budget()).selected_ids
        whole = make_policy("uniform", rate=0.25).allocate(events, Budget()).selected_ids
        self.assertEqual(chunks, whole)
        self.assertEqual(whole, ("s0:t1:p3", "s0:t1:p7", "s0:t1:p11"))
        a = make_policy("random", rate=0.5, seed=39).allocate(events, Budget()).selected_ids
        b = make_policy("random", rate=0.5, seed=39).allocate(events, Budget()).selected_ids
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
