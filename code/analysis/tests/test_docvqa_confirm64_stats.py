"""Statistical boundary checks, independent of any research outcomes."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_docvqa_confirm64 import holm, signflip_pvalues, PRIMARY, COMPARATORS


class ConfirmationStats(unittest.TestCase):
    def test_holm_monotonic_stepdown_and_original_order(self):
        result = holm([.01, .04, .03])
        np.testing.assert_allclose(result, [.03, .06, .06])
        self.assertEqual(holm([1., 1.]), [1., 1.])

    def test_question_weighted_effect_uses_image_sums(self):
        # One +1 question and three -1 questions: question mean=-.5;
        # averaging the two image means would incorrectly give zero.
        result = signflip_pvalues([1., -3.], total_questions=4, repeats=1000, seed=9)
        self.assertEqual(result["effect"], [-.5])

    def test_zero_effect_and_one_cluster_do_not_fake_significance(self):
        self.assertEqual(signflip_pvalues(np.zeros((8, 1)), 24, 1000, 3)["p_two_sided"], [1.])
        self.assertEqual(signflip_pvalues([1.], 1, 1000, 3)["p_two_sided"], [1.])

    def test_same_seed_reproduces_nonzero_monte_carlo_result(self):
        a = signflip_pvalues(np.ones((8, 1)), 8, 4096, 5)
        b = signflip_pvalues(np.ones((8, 1)), 8, 4096, 5)
        self.assertEqual(a, b)
        self.assertGreater(a["p_two_sided"][0], 0)
        self.assertLess(a["p_two_sided"][0], .02)

    def test_fixed_families_and_invalid_pvalues(self):
        self.assertEqual(len(PRIMARY), 2)
        self.assertEqual(len(COMPARATORS) - len(PRIMARY), 9)
        with self.assertRaises(ValueError):
            holm([float("nan")])


if __name__ == "__main__":
    unittest.main()
