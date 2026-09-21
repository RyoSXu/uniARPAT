import unittest

import torch

from model.losses import sumnorm_klw_loss
from tools.eval.c2_1b_loss_attribution import sumnorm_loss_components


class TestC21BLossAttribution(unittest.TestCase):
    def test_components_reconstruct_production_sumnorm_loss(self):
        torch.manual_seed(42)
        logits = torch.randn(3, 7, requires_grad=True)
        target = torch.softmax(torch.randn(3, 7), dim=-1)
        coverage = torch.ones_like(target, dtype=torch.bool)
        parts = sumnorm_loss_components(logits, target, coverage, False, huber_delta=0.02)
        reconstructed = parts["kl"] + 1.7 * parts["w1"] + 0.3 * parts["huber"]
        production = sumnorm_klw_loss(logits, target, coverage, False, 1.7, 0.3, 0.02)
        self.assertTrue(torch.allclose(reconstructed, production, atol=1e-7, rtol=1e-7))

    def test_components_have_finite_gradient(self):
        torch.manual_seed(7)
        logits = torch.randn(2, 5, requires_grad=True)
        target = torch.softmax(torch.randn(2, 5), dim=-1)
        coverage = torch.ones_like(target, dtype=torch.bool)
        parts = sumnorm_loss_components(logits, target, coverage, False, huber_delta=0.02)
        (parts["kl"] + parts["w1"] + parts["huber"]).mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
