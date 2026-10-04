import unittest
import torch
from model.heads import DeepConv1dHead, MultiScaleResidualHead, ScaleHead

class TestModelHeads(unittest.TestCase):
    def test_parameter_counts(self):
        ph_head = DeepConv1dHead(512, 256)
        e_head = MultiScaleResidualHead(512, 256)
        scale_head = ScaleHead(512, 128, 2)

        ph_params = sum(p.numel() for p in ph_head.parameters())
        e_params = sum(p.numel() for p in e_head.parameters())
        scale_params = sum(p.numel() for p in scale_head.parameters())

        self.assertEqual(ph_params, 787969, "phDOS DeepConv1dHead parameter mismatch!")
        self.assertEqual(e_params, 788225, "eDOS MultiScaleResidualHead parameter mismatch!")
        self.assertEqual(abs(ph_params - e_params), 256, "Head parameter discrepancy must be exactly 256!")
        self.assertEqual(scale_params, 74050, "ScaleHead parameter mismatch!")

    def test_forward_shapes(self):
        B = 4
        d_model = 512
        ph_head = DeepConv1dHead(d_model, 256)
        e_head = MultiScaleResidualHead(d_model, 256)

        x_ph = torch.randn(B, d_model, 64)
        x_e = torch.randn(B, d_model, 128)

        out_ph = ph_head(x_ph)
        out_e = e_head(x_e)

        self.assertEqual(out_ph.shape, (B, 1, 64))
        self.assertEqual(out_e.shape, (B, 1, 128))

if __name__ == '__main__':
    unittest.main()
