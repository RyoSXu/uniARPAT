"""Controls differ in geometry access while keeping the agreed backbone/identity."""
from dataclasses import asdict, replace
import unittest
from unittest.mock import patch

import torch

from model.local_structure_controls import ControlLocalMessage, PointwiseControlBlock, RadialControlBlock
from model.periodic_messages import prepare_shared_message_geometry
from model.transformer import Transformer
from tests.model.test_local_global_encoder import fixture, parameters
from tools.eval.local_global_compare import training_command
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import build_arg_parser, configure_local_global_execution
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.crystal import CrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import (
    capture_training_rng, comparison_contract, initialize_comparison_model, validate_local_global_config,
)
from utils.local_structure_experiment import (
    CONTROL_CONTRACT_SCHEMA, FROZEN, is_shared_control_parameter, structure_control_configs,
)
from utils.periodic_geometry import build_periodic_records
from utils.spectral_inference import reconstruct_h1_spectra


def control_parameters(route):
    return {**parameters(route), "global_depth": 6, "q1_coord": True}


def configuration(route="C0"):
    return ExperimentConfig(**{**FROZEN, "local_route": route, "tag": "_unit_structure"})


def predict(model, z, geometry):
    return model(z, z.eq(0), geometry, torch.linspace(-6, 6, 8, dtype=geometry.dtype),
                 torch.linspace(0, 1200, 4, dtype=geometry.dtype))


class TestLocalStructureControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def create(self, route, dtype=torch.float32):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            params = control_parameters(route)
            model = Transformer(**params)
            info = initialize_comparison_model(model, params, 42)
        finally:
            torch.set_default_dtype(previous)
        return model.eval(), info

    def test_c0_is_exactly_independent_of_geometry_and_skips_record_building(self):
        z, geometry = fixture()
        model, _ = self.create("C0")
        configure_local_global_execution(model)
        self.assertIsNone(model._local_geometry_cache)
        self.assertEqual(model._local_geometry_cache_disabled_reason, "no_geometry")
        with patch("utils.periodic_geometry.build_periodic_records", side_effect=AssertionError("records read")), \
                patch("model.local_global_encoder.prepare_shared_message_geometry",
                      side_effect=AssertionError("geometry read")), torch.no_grad():
            expected = predict(model, z, geometry)
            moved = CrystalGeometry(geometry.cell * 1.9, torch.rand_like(geometry.frac_coords))
            actual = predict(model, z, moved)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "geometry-consuming"):
            model.enable_local_geometry_cache()
        with self.assertRaisesRegex(ValueError, "no geometry"):
            model.encoder(torch.rand(2, 4, 16), object(), z.eq(0))

    def test_c1_has_no_angular_features_and_responds_to_radial_environment(self):
        h = torch.randn(1, 1, 16)
        mask = torch.zeros(1, 1, dtype=torch.bool)
        cell = torch.eye(3)[None] * 3.1
        coordinates = torch.zeros(1, 1, 3)
        local = ControlLocalMessage("C1", d_model=16)
        records = build_periodic_records(CrystalGeometry(cell, coordinates), mask, 6.)
        with patch("model.periodic_messages.o3.spherical_harmonics",
                   side_effect=AssertionError("angular feature read")):
            prepared = prepare_shared_message_geometry(records, "C1")
            self.assertEqual(prepared.unit.numel(), 0)
            self.assertIsNone(prepared.harmonics)
            self.assertIsNone(prepared.pairs)
            expected = local(h, records, mask, prepared_geometry=prepared)
            poisoned = replace(prepared, unit=torch.full((len(records.distances), 3), float("nan")),
                               harmonics=torch.full((len(records.distances), 25), float("nan")))
            actual = local(h, records, mask, prepared_geometry=poisoned)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            other_records = build_periodic_records(CrystalGeometry(cell * 1.2, coordinates), mask, 6.)
            changed = local(h, other_records, mask)
            self.assertGreater(float((changed - expected).abs().max()), 1e-5)
            self.assertTrue(((records.dst == 0) & (records.src == 0)).all())
            self.assertGreater(len(records.distances), 1)  # distinct self-images retained
            empty = build_periodic_records(CrystalGeometry(cell * 10, coordinates), mask, 6.)
            self.assertTrue(torch.isfinite(local(h, empty, mask)).all())

    def test_effective_c0_c1_block_parameter_counts_are_matched(self):
        count = lambda module: sum(value.numel() for value in module.parameters())
        self.assertEqual(count(PointwiseControlBlock()), 51985)
        self.assertEqual(count(RadialControlBlock()), 51968)
        self.assertEqual(4 * (count(PointwiseControlBlock()) - count(RadialControlBlock())), 68)

    def test_common_values_are_copied_verbatim_from_untrained_b_plus_with_rng_preserved(self):
        reference, ref_info = self.create("B+")
        shared = {key: value for key, value in reference.state_dict().items()
                  if is_shared_control_parameter(key)}
        hashes = []
        for route in ("C0", "C1"):
            params = control_parameters(route)
            model = Transformer(**params)
            rng = torch.get_rng_state().clone()
            info = initialize_comparison_model(model, params, 42)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(info["reference_state_sha256"], ref_info["state_sha256"])
            self.assertEqual(info["common_state_sha256"], ref_info["common_state_sha256"])
            hashes.append(info["shared_control_state_sha256"])
            state = model.state_dict()
            for key, value in shared.items():
                torch.testing.assert_close(state[key], value, atol=0, rtol=0)
                self.assertNotEqual(state[key].data_ptr(), value.data_ptr())
            repeat, other_info = self.create(route)
            self.assertEqual(info["state_sha256"], other_info["state_sha256"])
            del repeat
        self.assertEqual(hashes[0], hashes[1])

    def test_both_spectra_are_invariant_and_atom_permutation_padding_are_safe(self):
        for dtype, tolerance in ((torch.float32, 2e-5), (torch.float64, 2e-8)):
            z, geometry = fixture(dtype)
            for route in ("C0", "C1"):
                model, _ = self.create(route, dtype)
                with torch.no_grad():
                    expected = predict(model, z, geometry)
                    expected.update(reconstruct_h1_spectra(expected, [3, 2], [16., 8.]))
                    for name, matrix, translation in e3_operations():
                        q, t = torch.as_tensor(matrix, dtype=dtype), torch.as_tensor(translation, dtype=dtype)
                        cell = geometry.cell @ q
                        frac = geometry.frac_coords + torch.bmm(t.expand(2, 1, 3), torch.linalg.inv(cell))
                        actual = predict(model, z, CrystalGeometry(cell, frac))
                        actual.update(reconstruct_h1_spectra(actual, [3, 2], [16., 8.]))
                        for key in expected:
                            with self.subTest(route=route, dtype=dtype, operation=name, key=key):
                                torch.testing.assert_close(actual[key], expected[key], atol=tolerance, rtol=tolerance)
                    permutation = torch.tensor([2, 0, 3, 1])
                    coords = geometry.frac_coords[:, permutation].clone()
                    zz = z[:, permutation]
                    coords[zz.eq(0)] = float("nan")
                    actual = predict(model, zz, CrystalGeometry(geometry.cell, coords))
                    actual.update(reconstruct_h1_spectra(actual, [3, 2], [16., 8.]))
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], atol=tolerance, rtol=tolerance)

    def test_radial_cache_matches_uncached_outputs_and_parameter_gradients(self):
        z, geometry = fixture(torch.float64)
        model, _ = self.create("C1", torch.float64)
        outputs, gradients = [], []
        for cached in (False, True):
            model.enable_local_geometry_cache(enabled=cached, max_bytes=2**20)
            model.zero_grad(set_to_none=True)
            output = predict(model, z, geometry)
            sum(value.square().mean() for value in output.values()).backward()
            outputs.append(output)
            gradients.append({key: None if value.grad is None else value.grad.clone()
                              for key, value in model.named_parameters()})
        for key in outputs[0]:
            torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=1e-11, rtol=1e-9)
        for key, gradient in gradients[0].items():
            other = gradients[1][key]
            if gradient is None:
                self.assertIsNone(other, key)
            else:
                torch.testing.assert_close(gradient, other, atol=1e-11, rtol=1e-9)

    def test_contracts_are_isolated_and_runner_roundtrip_preserves_the_frozen_recipe(self):
        configs = structure_control_configs(configuration("B+"), "unit")
        self.assertEqual([cfg.local_route for cfg in configs], ["C0", "C1"])
        contracts = []
        for cfg in configs:
            validate_local_global_config(cfg)
            args = build_arg_parser().parse_args(training_command(cfg)[2:])
            self.assertEqual(asdict(cfg), asdict(ExperimentConfig.from_args(args)))
            contract = comparison_contract(cfg, control_parameters(cfg.local_route))
            self.assertEqual(contract["schema"], CONTROL_CONTRACT_SCHEMA)
            contracts.append(contract)
        source = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(source.parameters())
        payload = build_ablation_checkpoint(1, "M1", 42, False, source, optimizer, .5,
                                           model_contract=contracts[0], training_rng=capture_training_rng())
        old_contract = comparison_contract(configuration("B+"), control_parameters("B+"))
        for other in (contracts[1], old_contract, None):
            with self.assertRaisesRegex(ValueError, "model contract"):
                restore_ablation_checkpoint(payload, source, optimizer, False, model_contract=other)
        for change in ({"epochs": 100}, {"seed": 43}, {"batch_size": 64}, {"global_depth": 2},
                       {"local_layout": "alternating"}, {"dropout": .1}, {"augment": True}):
            with self.assertRaises(ValueError):
                validate_local_global_config(replace(configs[0], **change))


if __name__ == "__main__":
    unittest.main()
