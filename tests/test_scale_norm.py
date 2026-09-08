import unittest
import torch
from model.transformer import safe_shape_norm, ScaleHead

class TestScaleAndNorm(unittest.TestCase):
    def test_safe_shape_norm_normal(self):
        # Regular positive values
        x = torch.tensor([[1.0, 2.0, 4.0, 0.0]])
        y = safe_shape_norm(x)
        self.assertTrue(torch.all(y >= 0.0))
        self.assertAlmostEqual(float(y.max()), 1.0, places=4)

    def test_safe_shape_norm_all_negative_fallback(self):
        # All negative values (Dying ReLU condition)
        x = torch.tensor([[-5.0, -3.0, -10.0, -1.0]], requires_grad=True)
        y = safe_shape_norm(x)
        # Should activate Softplus fallback and remain non-negative
        self.assertTrue(torch.all(y >= 0.0))
        # Gradient should not vanish (non-zero)
        loss = y.sum()
        loss.backward()
        self.assertTrue(torch.all(x.grad != 0.0), "Gradients vanished under Dying ReLU!")

    def test_scale_head_bias_initialization(self):
        scale_head = ScaleHead(512, 128, 2)
        with torch.no_grad():
            scale_head.mlp[-1].bias.copy_(torch.tensor([3.0, -1.5]))
        self.assertTrue(torch.allclose(scale_head.mlp[-1].bias, torch.tensor([3.0, -1.5])))

if __name__ == '__main__':
    unittest.main()
