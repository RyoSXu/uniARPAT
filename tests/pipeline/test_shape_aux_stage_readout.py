"""Decision rules and hook contract for the shape-auxiliary stage readout.

Synthetic only. The locked epoch-25 run is not loaded here.
"""
import unittest

import numpy as np
import torch

from tools.eval.shape_aux_stage_readout import (
    SEPARATION_FLOOR,
    ShapeAuxStageExtractor,
    decide,
    relative_pair_l2,
    separation_class,
)
from utils.crystal import CrystalGeometry


class _Atom(torch.nn.Module):
    def forward(self, h, *args, **kwargs):
        return h


class _Decoder(torch.nn.Module):
    def forward(self, tgt, memory, **kwargs):
        return tgt, None


class _Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.local_layers = torch.nn.ModuleList([_Atom()])
        self.global_layers = torch.nn.ModuleList([_Atom()])

    @property
    def execution_order(self):
        return (("local", 0), ("global", 0))


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder()
        self.decoupled_decoder = False

    def forward(self, src, mask, pos):
        h = src
        for kind, index in self.encoder.execution_order:
            module = self.encoder.local_layers[index] if kind == "local" else self.encoder.global_layers[index]
            h = module(h)
        batch, width = src.shape[0], src.shape[-1]
        self.decoder(torch.zeros(batch, 4, width), h)
        self.decoder(torch.zeros(batch, 2, width), h)
        return h


class TestShapeAuxStageReadout(unittest.TestCase):
    def test_separation_floor_is_exclusive_at_the_boundary(self):
        self.assertEqual(separation_class({"ci95": [0.0, SEPARATION_FLOOR * 0.5]}), "indistinguishable")
        self.assertEqual(separation_class({"ci95": [SEPARATION_FLOOR * 2, SEPARATION_FLOOR * 3]}), "separated")
        self.assertEqual(separation_class({"ci95": [SEPARATION_FLOOR * 0.5, SEPARATION_FLOOR * 2]}), "uncertain")
        self.assertEqual(separation_class({"ci95": [SEPARATION_FLOOR, SEPARATION_FLOOR]}), "uncertain")
        self.assertEqual(separation_class({"ci95": None}), "uncertain")

    def test_relative_distance_is_symmetric_and_zero_for_copies(self):
        features = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 0.0]], dtype=np.float32)
        left = np.array([0, 0, 1])
        right = np.array([2, 1, 0])
        distance = relative_pair_l2(features, left, right)
        self.assertEqual(distance[0], 0.0)
        self.assertAlmostEqual(distance[1], relative_pair_l2(features, right[1:2], left[1:2])[0])
        self.assertGreater(distance[1], 0.0)

    def test_decision_order_matches_the_locked_rules(self):
        separated = "separated"
        same = "indistinguishable"
        cases = [
            ((separated, separated, separated, True, True, True, True), "premise_absent"),
            ((same, separated, separated, False, False, False, False), "local_message"),
            ((separated, same, separated, True, False, False, False), "global_mixing"),
            ((separated, separated, separated, True, True, True, False), "readout_head"),
            ((separated, separated, separated, True, True, False, False), "decoder_or_readout"),
            ((separated, separated, separated, False, False, False, False), "stop_no_bottleneck"),
            ((separated, "uncertain", separated, False, False, False, False), "unresolved"),
        ]
        for args, action in cases:
            with self.subTest(action=action):
                self.assertEqual(decide(*args)["action"], action)

    def test_extractor_keeps_both_decoder_calls_and_rejects_a_third(self):
        model = _Transformer()
        extractor = ShapeAuxStageExtractor(model)
        cell = torch.eye(3).view(1, 3, 3)
        pos = CrystalGeometry(cell, torch.zeros(1, 3, 3))
        mask = torch.zeros(1, 3, dtype=torch.bool)
        src = torch.ones(1, 3, 5)
        model(src, mask, pos)
        features = extractor.stacked()
        extractor.close()
        self.assertEqual(tuple(features["local_last"].shape), (1, 10))
        self.assertEqual(tuple(features["global_last"].shape), (1, 10))
        self.assertEqual(tuple(features["decoder_edos"].shape), (1, 4, 5))
        self.assertEqual(tuple(features["decoder_phdos"].shape), (1, 2, 5))
        self.assertEqual(extractor.hooks, [])

        again = ShapeAuxStageExtractor(model)
        model.decoder(torch.zeros(1, 3, 5), src)
        model.decoder(torch.zeros(1, 3, 5), src)
        with self.assertRaises(ValueError):
            model.decoder(torch.zeros(1, 3, 5), src)
        again.close()


if __name__ == "__main__":
    unittest.main()
