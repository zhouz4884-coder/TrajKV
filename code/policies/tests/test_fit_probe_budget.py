"""CPU mathematical fixtures only; no calibration result or experiment is fabricated."""
from copy import deepcopy
from pathlib import Path
import json
import math
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from policies.fit_probe_budget import (fit_group, fit_probe_budgets, load_calibration,
                                      literal_policy, threshold_candidates)


def session(name="s0", floor=10, events=((.1, 40), (.2, 40))):
    return dict(session_id=name, image_id=name, fixed_probe_extra_bytes=floor,
                events=[dict(relative_l2=delta, planned_upgrade_bytes=cost) for delta, cost in events])


def calibration_fixture():
    return dict(session_id="s0", image_id="image0", split="development", protocol="free_running",
        method="asym", policy="probe_l7_t1000000", model="fixture_model", read_budget_bytes=None,
        visual_tokens=256, kernel_variant="fixture", kernel_source_sha256="fixture_hash",
        owned_initial=dict(visual_fp16_denominator=320, visual_base=100, visual_residual=20),
        read_events=[dict(pass_kind="committed", layer=7, calls=1, base_visual_bytes=100, text_bytes=50, residual_bytes=0),
                     dict(pass_kind="probe", layer=7, calls=1, base_visual_bytes=100, text_bytes=50, residual_bytes=20)],
        intra_step_probes=[dict(event_id=0, turn=1, phase="decode", input_count=1, probe_layer=7,
            probe_observed=True, charged_bytes=320,
            decision=dict(relative_l2=.1, planned_upgrade_bytes=40, threshold=1_000_000,
                threshold_met=False, upgrade=False, use_enhanced_probe=False, finite_signal=True,
                feature_availability="online: same current query", shape=[1, 28, 1, 128], input_dtype="float16"))],
        intra_step_probe_skipped=[], extra_logical_read_bytes=170, read_ledger_exact=True,
        turns=[dict(answer="ANSWER_FIELD_MUST_NOT_BE_USED", token_ids=[1])])


class ProbeBudgetBoundaryChecks(unittest.TestCase):
    def test_greater_equal_boundary_uses_next_float_above_delta(self):
        result=fit_group([session()],probe_layer=7,budgets_bytes=(60,),safety_fraction=.9)
        chosen=result["budgets"][0]
        self.assertEqual(chosen["threshold"],math.nextafter(.1,math.inf))
        self.assertEqual(chosen["predicted_mean_extra_bytes"],50)
        self.assertEqual(chosen["per_session_predictions"][0]["triggered_events"],1)
        self.assertEqual(float(chosen["literal_policy_name"].split("_")[2][1:]),chosen["threshold"])

    def test_zero_delta_and_ties_are_not_split_arbitrarily(self):
        zero=fit_group([session(floor=0,events=((0.,100),))],probe_layer=0,
                       budgets_bytes=(10,),safety_fraction=1)
        self.assertEqual(zero["budgets"][0]["threshold"],math.nextafter(0.,math.inf))
        ties=fit_group([session(floor=0,events=((.1,30),(.1,40)))],probe_layer=0,
                       budgets_bytes=(50,),safety_fraction=1)
        self.assertEqual(ties["budgets"][0]["threshold"],math.nextafter(.1,math.inf))
        self.assertEqual(ties["budgets"][0]["predicted_mean_extra_bytes"],0)

    def test_probe_floor_is_irreducible_and_exact_safety_boundary_is_feasible(self):
        blocked=fit_group([session(floor=91,events=((.1,10),))],probe_layer=14,
                          budgets_bytes=(100,),safety_fraction=.9)["budgets"][0]
        self.assertEqual(blocked["status"],"INFEASIBLE_PROBE_FLOOR")
        self.assertIsNone(blocked["literal_policy_name"])
        boundary=fit_group([session(floor=90,events=((.1,10),))],probe_layer=14,
                           budgets_bytes=(100,),safety_fraction=.9)["budgets"][0]
        self.assertEqual(boundary["status"],"FEASIBLE_MEAN_PREDICTION")
        self.assertEqual(boundary["predicted_mean_extra_bytes"],90)

    def test_equal_session_mean_does_not_guarantee_individual_cap(self):
        result=fit_group([session("a",0,((.2,100),)),session("b",0,((.2,0),))],
                         probe_layer=7,budgets_bytes=(60,),safety_fraction=1)["budgets"][0]
        self.assertEqual(result["threshold"],0)
        self.assertEqual(result["predicted_mean_extra_bytes"],50)
        self.assertEqual(result["predicted_sessions_over_hard_cap"],1)
        self.assertTrue(result["deployment_hard_cap_required"])

    def test_actual_probe_extra_not_local_two_read_charge_and_answer_is_not_projected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"records.jsonl"
            path.write_text(json.dumps(calibration_fixture())+"\n")
            result=fit_probe_budgets([path],budgets_bytes=(200,),safety_fraction=.9)
            chosen=result["groups"][0]["budgets"][0]
            self.assertEqual(result["groups"][0]["mean_fixed_probe_extra_bytes"],170)
            self.assertEqual(chosen["status"],"FEASIBLE_MEAN_PREDICTION")
            self.assertNotIn("ANSWER_FIELD_MUST_NOT_BE_USED",json.dumps(result))
            self.assertIn(str(path.resolve()),result["source_sha256"])

    def test_capped_upgraded_skipped_or_privileged_traces_are_rejected(self):
        changes=[dict(read_budget_bytes=1000),dict(split="heldout"),
                 dict(intra_step_probe_skipped=[dict(reason="budget")])]
        modified=deepcopy(calibration_fixture())
        modified["intra_step_probes"][0]["decision"]["upgrade"]=True
        privileged=deepcopy(calibration_fixture())
        privileged["intra_step_probes"][0]["decision"]["feature_availability"]="reference"
        unobserved=deepcopy(calibration_fixture())
        unobserved["intra_step_probes"][0].update(probe_observed=False,decision=None)
        cases=[dict(calibration_fixture(),**change) for change in changes]+[modified,privileged,unobserved]
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"records.jsonl"
            for record in cases:
                path.write_text(json.dumps(record)+"\n")
                with self.subTest(record=record.get("split")),self.assertRaises(ValueError):
                    load_calibration([path])

    def test_empty_input_has_no_invented_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"records.jsonl"
            path.write_text("")
            result=fit_probe_budgets([path])
            self.assertEqual(result["status"],"EMPTY_CALIBRATION")
            self.assertEqual(result["groups"],[])

    def test_finite_threshold_domain_and_round_trip(self):
        self.assertEqual(threshold_candidates([0.,.1,.1]),[0.,math.nextafter(0.,math.inf),math.nextafter(.1,math.inf)])
        for value in (0.,math.nextafter(0.,math.inf),math.nextafter(.1,math.inf),1e100):
            self.assertEqual(float(literal_policy(7,value).split("_")[2][1:]),value)
        largest=float.fromhex("0x1.fffffffffffffp+1023")
        result=fit_group([session(floor=0,events=((largest,100),))],probe_layer=7,
                         budgets_bytes=(10,),safety_fraction=1)["budgets"][0]
        self.assertEqual(result["status"],"INFEASIBLE_FINITE_THRESHOLD")


if __name__=="__main__":
    unittest.main()
