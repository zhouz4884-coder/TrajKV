"""CPU-only numerical direction checks; no model or GPU execution."""
import unittest

import torch

from mechanism_fidelity import logit_comparison, logit_fidelity_gain


class FidelityDirectionChecks(unittest.TestCase):
    def test_reference_action_is_positive_and_identical_action_is_zero(self):
        reference = [torch.tensor([[3., 0., -1.]])]
        base = [torch.tensor([[0., 2., -1.]])]
        toward = logit_fidelity_gain(base, reference, reference, [1], set())
        unchanged = logit_fidelity_gain(base, base, reference, [1], set())
        self.assertGreater(toward["content_fidelity_gain_mean"], 0)
        self.assertEqual(unchanged["content_fidelity_gain_mean"], 0)
        self.assertAlmostEqual(toward["mean_action_kl_to_reference"], 0)

    def test_reinforcing_base_answer_can_have_negative_fidelity(self):
        reference = [torch.tensor([[3., 0., -1.]])]
        base = [torch.tensor([[0., 2., -1.]])]
        reinforce_base = [torch.tensor([[-1., 4., -1.]])]
        diagnostic = logit_comparison(reinforce_base, base, [1], set())
        fidelity = logit_fidelity_gain(base, reinforce_base, reference, [1], set())
        self.assertGreater(diagnostic["content_logprob_gain_mean"], 0)
        self.assertLess(fidelity["content_fidelity_gain_mean"], 0)

    def test_empty_content_does_not_invent_a_fidelity_estimate(self):
        values = [torch.tensor([[1., 0.]])]
        result = logit_fidelity_gain(values, values, values, [0], {0})
        self.assertIsNone(result["content_fidelity_gain_mean"])
        self.assertEqual(result["fidelity_gain_mean"], 0)


if __name__ == "__main__":
    unittest.main()
