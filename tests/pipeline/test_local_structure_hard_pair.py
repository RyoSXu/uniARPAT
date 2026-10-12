"""CPU tests for the hard-pair attribution helpers (2026-10-12 diagnosis).

Covers only pure comparison/grouping helpers: no model update, no CUDA, and no
dependence on run artifacts. The evidence replay itself lives in the run's
independent_verification.py.txt.
"""
import unittest

import numpy as np
import torch

from tools.eval.local_structure_hard_pair import (
    PARAM_GROUPS, atom_set_diff, compare_edges, fingerprint, param_group_map,
    spectrum_diff, token_diff)


def _edge_arrays(z_dst, z_src, distances):
    distances = np.asarray(distances, dtype=np.float64)
    return {
        "z_dst": np.asarray(z_dst, dtype=int),
        "z_src": np.asarray(z_src, dtype=int),
        "distances": distances,
        "shifts": np.zeros((len(distances), 3), dtype=int),
        "vectors": np.zeros((len(distances), 3), dtype=np.float64),
        "n_edges": int(len(distances)),
    }


class _FakeTransformer:
    def __init__(self, names):
        self._names = list(names)

    def named_parameters(self):
        for name in self._names:
            yield name, torch.zeros(1, requires_grad=True)


class HardPairHelperTest(unittest.TestCase):
    def test_fingerprint_and_compare_detect_equal_multiset(self):
        left = _edge_arrays([13, 13], [13, 13], [2.5, 4.75])
        right = _edge_arrays([13, 13], [13, 13], [2.5, 4.75])
        self.assertEqual(fingerprint(left), fingerprint(right))
        result = compare_edges(left, right)
        self.assertTrue(result["multiset_equal"])
        self.assertEqual(result["max_distance_delta"], 0.0)
        self.assertEqual(result["mean_abs_distance_delta"], 0.0)

    def test_compare_detects_distance_and_count_differences(self):
        left = _edge_arrays([13, 13], [13, 13], [2.5, 4.75])
        shifted = _edge_arrays([13, 13], [13, 13], [2.5, 4.85])
        result = compare_edges(left, shifted)
        self.assertFalse(result["multiset_equal"])
        self.assertAlmostEqual(result["mean_abs_distance_delta"], 0.05, places=6)
        longer = _edge_arrays([13, 13, 13], [13, 13, 13], [2.5, 4.75, 5.5])
        self.assertFalse(compare_edges(left, longer)["multiset_equal"])

    def test_atom_set_diff_is_permutation_invariant(self):
        rng = np.random.default_rng(0)
        left = rng.normal(size=(5, 8))
        right = rng.normal(size=(5, 8))
        base = atom_set_diff(left, right)
        shuffled = atom_set_diff(left, right[rng.permutation(5)])
        self.assertAlmostEqual(base["matched_rms"], shuffled["matched_rms"], places=6)
        self.assertAlmostEqual(base["chamfer_mean"], shuffled["chamfer_mean"], places=6)

    def test_atom_and_token_distances_use_the_same_feature_units(self):
        left = np.array([[1., 2., 3., 4.]])
        right = np.array([[2., 1., 4., 3.]])
        atom, token = atom_set_diff(left, right), token_diff(left, right)
        self.assertAlmostEqual(atom["matched_rms"], 1.0)
        self.assertAlmostEqual(atom["relative_matched_rms"], token["relative_rms"])
        self.assertEqual(atom["distance_units"], "per-feature-rms-v2")

    def test_atom_distance_does_not_grow_when_features_are_repeated(self):
        rng = np.random.default_rng(4)
        left, right = rng.normal(size=(3, 8)), rng.normal(size=(3, 8))
        base = atom_set_diff(left, right)
        repeated = atom_set_diff(np.tile(left, (1, 64)), np.tile(right, (1, 64)))
        for key in ("matched_rms", "matched_max", "state_scale_rms", "relative_matched_rms",
                    "chamfer_mean", "pooled_mean_rms_diff"):
            self.assertAlmostEqual(base[key], repeated[key], places=10, msg=key)

    def test_atom_distance_rejects_empty_or_mismatched_states(self):
        for left, right in ((np.zeros((0, 4)), np.zeros((1, 4))),
                            (np.zeros((1, 3)), np.zeros((1, 4)))):
            with self.assertRaises(ValueError):
                atom_set_diff(left, right)

    def test_spectrum_diff_zero_prediction_is_zero_skill(self):
        target_left = np.zeros(4)
        target_left[0] = 1.0
        target_right = np.zeros(4)
        target_right[1] = 1.0
        result = spectrum_diff(np.full(4, 0.25), np.full(4, 0.25),
                               target_left, target_right)
        self.assertAlmostEqual(result["difference_error"], result["target_diff_tv"],
                               places=9)
        self.assertAlmostEqual(result["skill"], 0.0, places=9)

    def test_token_diff_reports_zero_for_identical_tokens(self):
        tokens = np.arange(12.0).reshape(3, 4)
        result = token_diff(tokens, tokens.copy())
        self.assertEqual(result["rms_diff"], 0.0)
        self.assertEqual(result["relative_rms"], 0.0)

    def test_param_group_map_assigns_every_parameter(self):
        names = ["tok_emb.weight", "atom_proj.weight", "atom_norm.bias",
                 "encoder.local_layers.0.message.blocks.0.edge_mlp.0.weight",
                 "encoder.global_layers.1.attn.in_proj_bias",
                 "decoder.layers.0.linear1.weight", "edos_query_embed",
                 "edos_trunk.fc1.bias", "phdos_tgt", "edos_out_head.layers.0.weight",
                 "eta_head.mlp.0.bias"]
        groups = param_group_map(_FakeTransformer(names))
        self.assertEqual(tuple(groups), PARAM_GROUPS)
        self.assertEqual(sum(len(v) for v in groups.values()), len(names))
        self.assertEqual(len(groups["atom_entry"]), 3)
        self.assertEqual(len(groups["decoder"]), 4)
        with self.assertRaises(ValueError):
            param_group_map(_FakeTransformer(["mystery.weight"]))


if __name__ == "__main__":
    unittest.main()
