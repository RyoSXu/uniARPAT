import unittest
import torch
from model.heads import PostDecoderGatedCrossAttention

class TestGatedCrossAttention(unittest.TestCase):
    def test_parameter_count(self):
        d_model = 512
        nhead = 8
        gate_module = PostDecoderGatedCrossAttention(d_model, nhead)
        param_count = sum(p.numel() for p in gate_module.parameters())
        self.assertEqual(param_count, 2103298, "Gated cross-attention param count mismatch!")

    def test_zero_init_identity_property(self):
        d_model = 512
        nhead = 8
        gate_module = PostDecoderGatedCrossAttention(d_model, nhead)

        # Gates must be zero initialized
        self.assertEqual(float(gate_module.alpha_e.item()), 0.0)
        self.assertEqual(float(gate_module.alpha_p.item()), 0.0)

        # Forward pass with zero gate should return LayerNorm(hs) with zero cross-attention leakage
        B, L_e, L_p = 2, 128, 64
        h_e = torch.randn(B, L_e, d_model)
        h_p = torch.randn(B, L_p, d_model)

        h_e_out, h_p_out = gate_module(h_e, h_p)
        expected_e = gate_module.norm_e(h_e)
        expected_p = gate_module.norm_p(h_p)
        self.assertTrue(torch.allclose(h_e_out, expected_e, atol=1e-6), "Gated output should be LayerNorm(input) at init!")
        self.assertTrue(torch.allclose(h_p_out, expected_p, atol=1e-6), "Gated output should be LayerNorm(input) at init!")

if __name__ == '__main__':
    unittest.main()
