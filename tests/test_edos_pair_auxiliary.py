"""CPU contracts for the eDOS same-composition pair auxiliary path."""

import inspect
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from model.losses import edos_pair_contrast_loss
from run_ablation_experiments import (
    maybe_get_pair_aux_loader,
    validate_pair_aux_config,
)
from utils.ablation_checkpoint import (
    build_ablation_checkpoint,
    restore_ablation_checkpoint,
)
from utils.experiment_config import ExperimentConfig
from utils.pair_aux_batches import (
    batch_pair_plan,
    build_pair_universe,
    composition_keys,
    pair_plan_hash,
    pair_universe_hash,
    PairPlanBatchSampler,
    schedule_auxiliary_batches,
    select_epoch_pairs,
)


def _row(*atoms):
    return np.asarray((126, 127, *atoms, *([0] * (8 - len(atoms)))), dtype=np.int64)


class TestPairPlan(unittest.TestCase):
    def setUp(self):
        self.elements = np.stack([
            _row(6, 8),
            _row(8, 6),
            _row(6, 6, 8, 8),
            _row(6, 6, 8, 8),
            _row(6, 8, 8),
        ])
        self.ids = ["a", "b", "c", "d", "e"]

    def test_absolute_and_reduced_composition_keys(self):
        absolute, reduced = composition_keys(_row(8, 6, 8, 6))
        self.assertEqual(absolute, ((6, 2), (8, 2)))
        self.assertEqual(reduced, ((6, 1), (8, 1)))

    def test_universe_uses_all_combinations_without_crossing_absolute_counts(self):
        universe = build_pair_universe(self.elements, self.ids)
        endpoints = {(pair.mpid_a, pair.mpid_b) for pair in universe}
        self.assertEqual(endpoints, {("a", "b"), ("c", "d")})
        self.assertTrue(all(pair.absolute_key in (
            ((6, 1), (8, 1)), ((6, 2), (8, 2))
        ) for pair in universe))

    def test_each_reduced_group_has_one_deterministic_rotating_pair(self):
        universe = build_pair_universe(self.elements, self.ids)
        epoch0 = select_epoch_pairs(universe, seed=42, epoch=0)
        epoch1 = select_epoch_pairs(universe, seed=42, epoch=1)
        self.assertEqual(len(epoch0), 1)
        self.assertNotEqual(epoch0, epoch1)
        self.assertEqual(epoch0, select_epoch_pairs(universe, seed=42, epoch=0))
        self.assertEqual(len({pair.reduced_key for pair in epoch0}), len(epoch0))
        self.assertNotEqual(
            pair_plan_hash(epoch0, seed=42, epoch=0),
            pair_plan_hash(epoch1, seed=42, epoch=1),
        )

    def test_batches_and_even_schedule(self):
        pairs = select_epoch_pairs(build_pair_universe(self.elements, self.ids), 42, 0)
        repeated = pairs * 1198
        batches = batch_pair_plan(repeated, pairs_per_batch=16)
        self.assertEqual(len(batches), 75)
        self.assertEqual(len(batches[-1]), 14)
        schedule = schedule_auxiliary_batches(585, 75)
        self.assertEqual(len(schedule), 75)
        self.assertEqual([index for _, index in schedule], list(range(75)))
        steps = [step for step, _ in schedule]
        self.assertEqual(len(set(steps)), 75)
        self.assertGreaterEqual(min(steps), 0)
        self.assertLess(max(steps), 585)
        gaps = np.diff(steps)
        self.assertLessEqual(int(gaps.max() - gaps.min()), 1)

    def test_batch_sampler_keeps_pair_halves_aligned(self):
        universe = build_pair_universe(self.elements, self.ids)
        sampler = PairPlanBatchSampler(universe, seed=42, pairs_per_batch=2)
        selected = sampler.epoch_pairs()
        batch = next(iter(sampler))
        count = len(batch) // 2
        self.assertEqual(batch[:count], [pair.index_a for pair in selected[:count]])
        self.assertEqual(batch[count:], [pair.index_b for pair in selected[:count]])
        self.assertEqual(sampler.plan_hash(), pair_plan_hash(selected, 42, 0))
        self.assertEqual(sampler.frozen_plan_hash(), pair_universe_hash(universe, 42))

    def test_pair_builder_has_no_label_argument(self):
        parameters = set(inspect.signature(build_pair_universe).parameters)
        self.assertEqual(parameters, {"elements", "sample_ids"})

    def test_full_q1_train_counts_when_local_cache_exists(self):
        root = Path(__file__).resolve().parents[1] / "data/train4ARPAT/train"
        elements_path = root / "elements_train.npy"
        index_path = root / "train_index.npy"
        if not elements_path.exists() or not index_path.exists():
            self.skipTest("local Q1 cache is unavailable")
        elements = np.load(elements_path, mmap_mode="r")
        ids = np.load(index_path, mmap_mode="r")
        universe = build_pair_universe(elements, ids)
        self.assertEqual(len(universe), 2591)
        self.assertEqual(len({pair.absolute_key for pair in universe}), 1331)
        self.assertEqual(len({pair.reduced_key for pair in universe}), 1198)
        material_indices = {index for pair in universe for index in (pair.index_a, pair.index_b)}
        self.assertEqual(len(material_indices), 3115)
        self.assertEqual(len(select_epoch_pairs(universe, 42, 0)), 1198)


class TestPairLoss(unittest.TestCase):
    def test_exact_match_is_zero_and_has_finite_gradient(self):
        target_a = torch.tensor([[0.4, 0.35, 0.25]])
        target_b = torch.tensor([[0.2, 0.3, 0.5]])
        logits_a = target_a.log().requires_grad_()
        logits_b = target_b.log().requires_grad_()
        loss = edos_pair_contrast_loss(logits_a, logits_b, target_a, target_b)
        loss.backward()
        self.assertAlmostEqual(loss.item(), 0.0, places=7)
        self.assertTrue(torch.isfinite(logits_a.grad).all())
        self.assertTrue(torch.isfinite(logits_b.grad).all())

    def test_swapping_both_endpoints_is_invariant(self):
        logits_a = torch.tensor([[2.0, -1.0, 0.2]])
        logits_b = torch.tensor([[-0.5, 1.5, 0.0]])
        target_a = torch.tensor([[0.7, 0.2, 0.1]])
        target_b = torch.tensor([[0.1, 0.6, 0.3]])
        forward = edos_pair_contrast_loss(logits_a, logits_b, target_a, target_b)
        swapped = edos_pair_contrast_loss(logits_b, logits_a, target_b, target_a)
        torch.testing.assert_close(forward, swapped)

    def test_wrong_difference_direction_costs_more(self):
        target_a = torch.tensor([[0.8, 0.2]])
        target_b = torch.tensor([[0.2, 0.8]])
        correct = edos_pair_contrast_loss(
            target_a.log(), target_b.log(), target_a, target_b
        )
        reversed_direction = edos_pair_contrast_loss(
            target_b.log(), target_a.log(), target_a, target_b
        )
        self.assertLess(correct.item(), reversed_direction.item())

    def test_rejects_invalid_shapes_and_nonfinite_targets(self):
        valid = torch.zeros(1, 3)
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            edos_pair_contrast_loss(valid, torch.zeros(2, 3), valid, valid)
        with self.assertRaisesRegex(ValueError, "finite"):
            edos_pair_contrast_loss(valid, valid, torch.full((1, 3), math.nan), valid)


class TestPairIntegrationBoundaries(unittest.TestCase):
    def test_default_configuration_does_not_build_pair_loader(self):
        config = ExperimentConfig()
        self.assertEqual(config.pair_aux_arm, "none")
        self.assertEqual(config.pair_ratio, 0.0)
        validate_pair_aux_config(config)
        self.assertEqual(maybe_get_pair_aux_loader(object(), config), (None, None))

    def test_active_recipe_requires_test_isolation_and_rejects_conflicts(self):
        valid = ExperimentConfig(
            epochs=10,
            tag="_pcaux",
            pair_aux_arm="candidate",
            pair_ratio=0.10,
            skip_test_eval=True,
            init_ckpt="./output/ablation_m1_e9ctl/checkpoint_best.pth",
            reset_rng_after_init=True,
        )
        validate_pair_aux_config(valid)
        invalid_test = ExperimentConfig(
            epochs=10, tag="_pcaux", pair_aux_arm="candidate", pair_ratio=0.10,
            init_ckpt="./output/ablation_m1_e9ctl/checkpoint_best.pth",
            reset_rng_after_init=True,
        )
        with self.assertRaisesRegex(ValueError, "skip_test_eval"):
            validate_pair_aux_config(invalid_test)
        invalid_conflict = ExperimentConfig(**{**valid.__dict__, "use_g2": True})
        with self.assertRaisesRegex(ValueError, "conflicts"):
            validate_pair_aux_config(invalid_conflict)

    def test_pair_checkpoint_round_trip_and_mismatch_rejection(self):
        torch.manual_seed(7)
        source = torch.nn.Linear(3, 2)
        source_optimizer = torch.optim.AdamW(source.parameters())
        plan_hash = "a" * 64
        payload = build_ablation_checkpoint(
            1, "M1", 42, False, source, source_optimizer, 0.5,
            pair_aux_arm="candidate", pair_ratio=0.10,
            pair_lambda=0.25, pair_plan_hash=plan_hash,
        )
        target = torch.nn.Linear(3, 2)
        target_optimizer = torch.optim.AdamW(target.parameters())
        metadata = restore_ablation_checkpoint(
            payload, target, target_optimizer, False,
            pair_aux_arm="candidate", pair_ratio=0.10,
            pair_lambda=0.25,
            pair_plan_hash=plan_hash,
        )
        self.assertEqual(metadata["pair_lambda"], 0.25)
        with self.assertRaisesRegex(ValueError, "different pair plan"):
            restore_ablation_checkpoint(
                payload, target, target_optimizer, False,
                pair_aux_arm="candidate", pair_ratio=0.10,
                pair_lambda=0.25,
                pair_plan_hash="b" * 64,
            )
        with self.assertRaisesRegex(ValueError, "disabled"):
            restore_ablation_checkpoint(
                payload, target, target_optimizer, False,
                pair_aux_arm="none", pair_ratio=0.0,
            )
        with self.assertRaisesRegex(ValueError, "different pair_lambda"):
            restore_ablation_checkpoint(
                payload, target, target_optimizer, False,
                pair_aux_arm="candidate", pair_ratio=0.10,
                pair_lambda=0.20, pair_plan_hash=plan_hash,
            )


if __name__ == "__main__":
    unittest.main()
