"""Check physical mappings and reject wrong spectral scaling or topology."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from model.transformer import Transformer
from tests.data.test_spectral_balance import protocol
from tests.model.test_local_global_encoder import parameters
from tools.eval.crystal_model_invariance import (
    CASE_NAMES, mapped_geometry_report, output_comparison, physical_outputs,
    record_arrays, select_train_rows, transform_case,
)
from utils.crystal import CrystalGeometry
from utils.periodic_geometry import build_periodic_records
from utils.zproj_alignment import state_hash


class CrystalModelInvarianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, dtype):
        return torch.tensor([14, 14, 8]), CrystalGeometry(
            torch.tensor([[9., 0., 0.], [1., 10., 0.], [.4, -.6, 11.]], dtype=dtype),
            torch.tensor([[.1, .1, .1], [.23, .14, .2], [.31, .3, .18]], dtype=dtype))

    def records(self, geometry):
        batched = CrystalGeometry(geometry.cell[None], geometry.frac_coords[None])
        return record_arrays(build_periodic_records(batched, torch.zeros(1, len(geometry.frac_coords), dtype=torch.bool), 6.))

    def test_all_native_transform_maps_preserve_each_receiver_neighborhood(self):
        for dtype in (torch.float32, torch.float64):
            numbers, geometry = self.fixture(dtype)
            original = [tensor.clone() for tensor in geometry]
            reference = self.records(geometry)
            for name in CASE_NAMES:
                with self.subTest(dtype=dtype, case=name):
                    case = transform_case(numbers, geometry, name)
                    actual = self.records(case.geometry)
                    report = mapped_geometry_report(reference, actual, case)
                    self.assertEqual(report['status'], 'pass', report)
                    multiplier = 2 if name.startswith('supercell') else 1
                    self.assertEqual(len(case.numbers), multiplier*len(numbers))
                    self.assertEqual(len(actual['d']), multiplier*len(reference['d']))
                    if name == 'basis_swap':
                        self.assertLess(float(torch.linalg.det(case.geometry.cell)), 0)
            for before, after in zip(original, geometry):
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_boundary_topology_is_reported_and_stable_record_corruption_fails(self):
        numbers = torch.tensor([13])
        geometry = CrystalGeometry(torch.eye(3)*6, torch.zeros(1, 3))
        case = transform_case(numbers, geometry, 'repeat')
        empty = {'keys': np.empty((0, 6), dtype=int), 'r': np.empty((0, 3)), 'd': np.empty(0)}
        actual = {'keys': np.array([[0, 0, 0, 1, 0, 0]]), 'r': np.array([[5.99999, 0, 0]]), 'd': np.array([5.99999])}
        self.assertEqual(mapped_geometry_report(empty, actual, case)['status'], 'boundary_limited')
        actual['r'][0, 0] = actual['d'][0] = 5.9
        self.assertEqual(mapped_geometry_report(empty, actual, case)['status'], 'fail')

    def test_normalized_output_rule_rejects_shape_mass_and_supercell_scaling_errors(self):
        policy = protocol((8, 4))
        raw = {'edos': torch.zeros(1, 8), 'phdos': torch.zeros(1, 4), 'mass_logits': torch.ones(1, 2)}
        reference = physical_outputs(raw, 3, policy)
        expanded = physical_outputs(raw, 6, policy)
        self.assertTrue(output_comparison(reference, expanded, 3, 2, policy, torch.float32)['passed'])
        self.assertFalse(output_comparison(reference, reference, 3, 2, policy, torch.float32)['passed'])
        for key in ('shape_edos', 'mass_phdos', 'per_atom_edos'):
            changed = deepcopy(reference)
            changed[key] *= 1.1
            self.assertFalse(output_comparison(reference, changed, 3, 1, policy, torch.float32)['passed'])

    def test_complete_small_lr2_model_outputs_and_state_across_all_transforms(self):
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(42)
            model = Transformer(**{**parameters('LR2'), 'global_depth': 6, 'scale_mode': 'per_atom'}).to(dtype).eval()
            model.requires_grad_(False)
            numbers, geometry = self.fixture(dtype)
            before = state_hash(model.state_dict())
            policy = protocol((8, 4))
            def predict(numbers, geometry):
                with torch.no_grad():
                    raw = model(numbers[None], torch.zeros(1, len(numbers), dtype=torch.bool),
                                CrystalGeometry(geometry.cell[None], geometry.frac_coords[None]))
                return physical_outputs(raw, len(numbers), policy)
            expected = predict(numbers, geometry)
            for name in CASE_NAMES:
                case = transform_case(numbers, geometry, name)
                result = output_comparison(expected, predict(case.numbers, case.geometry), len(numbers),
                                           len(case.numbers)/len(numbers), policy, dtype)
                self.assertTrue(result['passed'], (dtype, name, result))
            self.assertEqual(before, state_hash(model.state_dict()))

    def test_train_selection_is_deterministic_label_free_and_respects_exclusions(self):
        ids = np.array([f'material_{i}' for i in range(30)])
        store = SimpleNamespace(material_ids=ids, n_atoms=torch.arange(1, 31),
                                cell=torch.eye(3)[None].repeat(30, 1, 1),
                                manifest={'training_exclusions': [ids[0], ids[-1]]})
        first = select_train_rows(store)
        self.assertEqual(first, select_train_rows(store))
        self.assertEqual(len(set(first)), 16)
        self.assertTrue({1, 28} <= set(first))
        self.assertFalse({0, 29} & set(first))


if __name__ == '__main__':
    unittest.main()
