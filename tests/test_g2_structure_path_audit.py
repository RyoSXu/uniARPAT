"""冻结结构通路诊断的数值、排列和读取边界。"""

import unittest
from unittest.mock import Mock

import numpy as np
import torch

from model.transformer import Transformer
from tools.eval.g2_structure_path_audit import (
    FeatureProbe, contrast_metrics, load_split, matched_atom_rms,
    numerical_control, relative_rms, zero_g2_residuals,
)


class TestStructurePathMetrics(unittest.TestCase):
    def test_element_matching_is_permutation_invariant(self):
        left = np.array([[1., 2.], [4., 7.], [8., 3.]])
        elements = np.array([14, 8, 14])
        order = [2, 0, 1]
        self.assertEqual(matched_atom_rms(left, left[order], elements, elements[order]), 0.)
        with self.assertRaisesRegex(ValueError, "element counts"):
            matched_atom_rms(left, left, elements, [14, 8, 8])

    def test_matching_cannot_exchange_different_elements(self):
        left = np.array([[1., 0.], [0., 1.]])
        self.assertGreater(matched_atom_rms(left, left[::-1], [14, 8], [14, 8]), 1.)

    def test_relative_rms_is_scale_normalized(self):
        left, right = np.array([1., 2.]), np.array([2., 4.])
        self.assertAlmostEqual(relative_rms(left, right), relative_rms(17 * left, 17 * right))

    def test_more_spectral_contrast_can_have_wrong_direction(self):
        a, b = np.array([.8, .2]), np.array([.2, .8])
        right = contrast_metrics(a, b, a, b)
        wrong = contrast_metrics(b, a, a, b)
        self.assertEqual(right["contrast_error_tv"], 0.)
        self.assertAlmostEqual(wrong["predicted_tv"], right["predicted_tv"])
        self.assertGreater(wrong["contrast_error_tv"], 1.)
        self.assertAlmostEqual(wrong["contrast_cosine"], -1.)

    def test_only_train_valid_can_be_loaded(self):
        builder = Mock()
        with self.assertRaisesRegex(ValueError, "train/valid"):
            load_split(builder, "test")
        builder.get_dataset.assert_not_called()
        load_split(builder, "valid")
        builder.get_dataset.assert_called_once_with(split="valid", dos_minmax=True, dos_sumnorm=True)


class TestStructurePathProbe(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.transformer = Transformer(
            d_model=16, nhead=4, num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=32, edos_num=128, phdos_num=64, dropout=0.,
            use_g2=True, scale_mode="eta",
        ).eval()
        with torch.no_grad():
            self.transformer.encoder.g2_msgs[0].alpha.fill_(.05)
        inp = torch.tensor([[126, 127, 14, 8, 0], [126, 127, 14, 14, 0]])
        pos = torch.zeros(2, 5, 3)
        pos[:, 0] = torch.tensor([5., 5., .2])
        pos[:, 1] = 90.
        pos[:, 3] = torch.tensor([.25, .25, .25])
        self.processed = [inp, pos, inp.eq(0)] + [None] * 15

    def test_hooks_are_observations_and_numerical_controls_pass(self):
        probe = FeatureProbe(self.transformer)
        try:
            with torch.inference_mode():
                outputs, features = probe.forward(self.processed)
                direct = self.transformer(*[self.processed[i] for i in (0, 2, 1)])
                for key in outputs:
                    torch.testing.assert_close(outputs[key], direct[key], rtol=0, atol=0)
                self.assertEqual(features["encoder"].shape, (2, 3, 16))
                self.assertEqual(features["decoder"].shape, (2, 128, 16))
                self.assertTrue((features["residual_0"] > 0).all())
                self.assertLessEqual(max(numerical_control(probe, self.processed, outputs).values()), 1e-5)
        finally:
            probe.close()
        self.assertFalse(self.transformer.encoder._forward_hooks)
        self.assertFalse(self.transformer.decoder._forward_hooks)

    def test_alpha_restored_even_when_probe_fails(self):
        before = self.transformer.encoder.g2_msgs[0].alpha.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "probe failed"):
            with zero_g2_residuals(self.transformer):
                self.assertEqual(self.transformer.encoder.g2_msgs[0].alpha.item(), 0.)
                raise RuntimeError("probe failed")
        torch.testing.assert_close(before, self.transformer.encoder.g2_msgs[0].alpha, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
