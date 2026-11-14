"""Mathematical tests for clustered label-swap inference, without model data."""
import unittest

from analyze_frozen_eval64 import holm_adjust,paired_signflip_counts


class FrozenInferenceChecks(unittest.TestCase):
    def test_holm_adjustment_is_step_down_monotone(self):
        values=holm_adjust([.01,.04,.03])
        self.assertAlmostEqual(values[0],.03)
        self.assertAlmostEqual(values[1],.06)
        self.assertAlmostEqual(values[2],.06)

    def test_cluster_signflip_matches_two_image_exact_case(self):
        result=paired_signflip_counts([1,1],draws=20000,seed=101)
        self.assertAlmostEqual(result["p_mc_two_sided"],.5,delta=.02)
        self.assertEqual(result["informative_images"],2)

    def test_no_image_level_net_change_is_degenerate_not_640_trials(self):
        result=paired_signflip_counts([0]*64,draws=100,seed=2)
        self.assertEqual(result["p_mc_two_sided"],1)
        self.assertEqual(result["informative_images"],0)


if __name__=="__main__":
    unittest.main()
