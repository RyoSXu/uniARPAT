"""Cache identity, resource bounds and unchanged learned computation."""
from dataclasses import fields
import unittest
from unittest.mock import patch

import torch

from model.periodic_geometry_cache import PeriodicGeometryCache
from model.periodic_messages import PeriodicLocalMessage, prepare_shared_message_geometry
from model.transformer import Transformer
from tests.model.test_local_global_encoder import fixture, parameters
from utils.crystal import CrystalGeometry
from utils.local_global_experiment import initialize_comparison_model
from utils.periodic_geometry import build_periodic_records


class TestPeriodicGeometryCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def compare_geometry(self, cache, geometry, mask, radius=6.):
        records, prepared = cache.get(geometry, mask, radius=radius)
        expected = build_periodic_records(geometry, mask, radius)
        reference = prepare_shared_message_geometry(expected, cache.route)
        for name in ('batch', 'dst', 'src', 'shifts', 'padding_mask', 'vectors', 'distances'):
            torch.testing.assert_close(getattr(records, name), getattr(expected, name))
        for field in fields(reference):
            name = field.name
            if name in ('record_context', 'prepared_route'):
                continue
            a, b = getattr(prepared, name), getattr(reference, name)
            if isinstance(a, tuple):
                for x, y in zip(a, b):
                    torch.testing.assert_close(x, y, atol=0, rtol=0)
            elif a is not None:
                torch.testing.assert_close(a, b)
            else:
                self.assertIsNone(b)
        return records, prepared

    def test_rebatch_atom_reorder_and_nonterminal_padding_preserve_records(self):
        for route in ('A', 'B+'):
            for dtype in (torch.float32, torch.float64):
                atoms, geometry = fixture(dtype)
                cache = PeriodicGeometryCache(route, 'cpu', max_bytes=2**20)
                self.compare_geometry(cache, geometry, atoms.eq(0))
                self.assertEqual(cache.misses, 2)
                self.compare_geometry(cache, geometry, atoms.eq(0))
                self.assertEqual(cache.hits, 2)
                rows, slots = torch.tensor([1, 0]), torch.tensor([2, 0, 3, 1])
                transformed = CrystalGeometry(geometry.cell[rows], geometry.frac_coords[rows][:, slots])
                self.compare_geometry(cache, transformed, atoms[rows][:, slots].eq(0))
                poisoned = geometry.frac_coords.clone()
                poisoned[atoms.eq(0)] = float('nan')
                hits = cache.hits
                self.compare_geometry(cache, CrystalGeometry(geometry.cell, poisoned), atoms.eq(0))
                self.assertEqual(cache.hits, hits+2)

    def test_content_dtype_radius_and_coordinate_gradient_invalidation(self):
        atoms, geometry = fixture()
        mask = atoms.eq(0)
        cache = PeriodicGeometryCache('A', 'cpu', max_bytes=2**20)
        self.compare_geometry(cache, geometry, mask)
        changed = CrystalGeometry(geometry.cell.clone(), geometry.frac_coords.clone())
        changed.frac_coords[0, 0, 0] += .02
        self.compare_geometry(cache, changed, mask)
        self.assertEqual(cache.misses, 3)
        self.compare_geometry(cache, geometry, mask)
        self.assertEqual(cache.hits, 3)
        changed.cell[0, 0, 0] += .1
        self.compare_geometry(cache, changed, mask)
        self.assertEqual(cache.misses, 4)
        self.compare_geometry(cache, geometry.to(dtype=torch.float64), mask)
        self.assertEqual(len(cache._entries), 2)
        self.assertTrue(all(e.records.vectors.dtype == torch.float64 for e in cache._entries.values()))
        self.compare_geometry(cache, geometry.to(dtype=torch.float64), mask, radius=5.5)
        self.assertTrue(all(e.records.radius == 5.5 for e in cache._entries.values()))
        grad_geometry = CrystalGeometry(geometry.cell.clone().requires_grad_(),
                                       geometry.frac_coords.clone().requires_grad_())
        entries = len(cache._entries)
        self.assertIsNone(cache.get(grad_geometry, mask, radius=6.))
        self.assertEqual(len(cache._entries), entries)

    def test_lru_budget_oversize_entries_empty_rows_and_resource_fallback(self):
        atoms, geometry = fixture()
        mask = atoms.eq(0)
        cache = PeriodicGeometryCache('A', 'cpu', max_bytes=2**20)
        cache.get(geometry, mask, radius=6.)
        limit = max(e.nbytes for e in cache._entries.values())
        cache.clear()
        cache.max_bytes = limit
        for row in (0, 1, 1, 0):
            one = CrystalGeometry(geometry.cell[row:row+1], geometry.frac_coords[row:row+1])
            self.compare_geometry(cache, one, mask[row:row+1])
            self.assertLessEqual(cache.nbytes, limit)
        self.assertEqual(cache.hits, 1)
        self.assertGreaterEqual(cache.evictions, 2)
        tiny = PeriodicGeometryCache('A', 'cpu', max_bytes=1)
        self.compare_geometry(tiny, geometry, mask)
        self.assertEqual(tiny.nbytes, 0)
        self.assertEqual(len(tiny._entries), 0)
        empty = mask.clone()
        empty[0] = True
        self.assertIsNone(cache.get(geometry, empty, radius=6.))
        zero_batch = CrystalGeometry(geometry.cell[:0], geometry.frac_coords[:0])
        self.assertIsNone(cache.get(zero_batch, mask[:0], radius=6.))
        rng = torch.get_rng_state()
        with patch.object(cache, '_make_entry', side_effect=torch.cuda.OutOfMemoryError('probe')), \
                patch('torch.cuda.empty_cache'):
            cache.clear()
            self.assertIsNone(cache.get(geometry, mask, radius=6.))
        self.assertEqual(cache.nbytes, 0)
        self.assertEqual(cache.oom_fallbacks, 1)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)

    def test_workspace_pressure_releases_cache_without_changing_batch_records(self):
        atoms, geometry = fixture()
        cache = PeriodicGeometryCache('A', 'cpu', max_bytes=2**20)
        records, prepared = cache.get(geometry, atoms.eq(0), radius=6.)
        initial = cache.nbytes
        with patch('torch.cuda.mem_get_info', side_effect=lambda _: (initial-cache.nbytes, initial)), \
                patch('torch.cuda.memory_reserved', return_value=0), \
                patch('torch.cuda.memory_allocated', return_value=0):
            cache._reserve_workspace(torch.device('cuda:0'), initial)
        self.assertEqual(cache.nbytes, 0)
        self.assertTrue(prepared.pairs[0].numel())
        expected = build_periodic_records(geometry, atoms.eq(0), 6.)
        torch.testing.assert_close(records.vectors, expected.vectors)

    def test_prepared_geometry_rejects_unrelated_or_modified_records(self):
        atoms, geometry = fixture()
        records = build_periodic_records(geometry, atoms.eq(0), 6.)
        prepared = prepare_shared_message_geometry(records, 'A')
        message = PeriodicLocalMessage('A', d_model=16)
        h = torch.randn(2, 4, 16)
        other = build_periodic_records(geometry, atoms.eq(0), 6.)
        with self.assertRaisesRegex(ValueError, 'prepared geometry'):
            message(h, other, atoms.eq(0), prepared_geometry=prepared)
        records.vectors.add_(.001)
        with self.assertRaisesRegex(ValueError, 'prepared geometry'):
            message(h, records, atoms.eq(0), prepared_geometry=prepared)

    def test_full_models_preserve_outputs_gradients_rng_and_learned_updates(self):
        for route in ('A', 'B+'):
            for layout in ('staged', 'alternating'):
                for dtype in (torch.float32, torch.float64):
                    with self.subTest(route=route, layout=layout, dtype=dtype):
                        config = parameters(route, layout)
                        config['dropout'] = .05
                        model = Transformer(**config).to(dtype=dtype).train()
                        initialize_comparison_model(model, config, 42)
                        atoms, geometry = fixture(dtype)
                        expected_state = {k: v.clone() for k, v in model.state_dict().items()}
                        rng = torch.get_rng_state()

                        def run(shared, cached):
                            torch.set_rng_state(rng)
                            model.zero_grad(set_to_none=True)
                            model.encoder.shared_geometry = shared
                            model.enable_local_geometry_cache(enabled=cached, max_bytes=2**20)
                            output = model(atoms, atoms.eq(0), geometry)
                            sum(v.square().mean() for v in output.values()).backward()
                            gradients = {k: v.grad.clone() for k, v in model.named_parameters()
                                         if v.grad is not None}
                            return output, gradients, torch.get_rng_state()

                        reference = run(False, False)
                        actual = run(True, True)
                        tolerance = 2e-5 if dtype == torch.float32 else 2e-10
                        for key in reference[0]:
                            torch.testing.assert_close(actual[0][key], reference[0][key], atol=tolerance, rtol=tolerance)
                        self.assertEqual(reference[1].keys(), actual[1].keys())
                        for key in reference[1]:
                            torch.testing.assert_close(actual[1][key], reference[1][key], atol=tolerance, rtol=tolerance)
                        torch.testing.assert_close(reference[2], actual[2], atol=0, rtol=0)
                        for key, value in model.state_dict().items():
                            torch.testing.assert_close(value, expected_state[key], atol=0, rtol=0)
                        cache = model._local_geometry_cache
                        with torch.no_grad():
                            model.atom_proj.bias.add_(.05)
                            torch.set_rng_state(rng)
                            changed = model(atoms, atoms.eq(0), geometry)
                            self.assertGreater(cache.hits, 0)
                            self.assertTrue(any(not torch.equal(changed[k], actual[0][k]) for k in changed))
                            model.enable_local_geometry_cache(enabled=False)
                            torch.set_rng_state(rng)
                            uncached = model(atoms, atoms.eq(0), geometry)
                        for key in changed:
                            torch.testing.assert_close(changed[key], uncached[key], atol=tolerance, rtol=tolerance)

    def test_input_gradients_remain_live_and_model_moves_clear_entries(self):
        for route in ('A', 'B+'):
            model = Transformer(**parameters(route)).double()
            atoms, geometry = fixture(torch.float64)
            geometry = CrystalGeometry(geometry.cell.requires_grad_(), geometry.frac_coords.requires_grad_())
            gradients = []
            for enabled in (False, True):
                model.enable_local_geometry_cache(enabled=enabled, max_bytes=2**20)
                output = model(atoms, atoms.eq(0), geometry)
                gradients.append(torch.autograd.grad(sum(v.square().mean() for v in output.values()), geometry))
            for a, b in zip(*gradients):
                torch.testing.assert_close(a, b, atol=2e-11, rtol=1e-8)
            self.assertEqual(model._local_geometry_cache.bypasses, 1)
            normal = CrystalGeometry(geometry.cell.detach(), geometry.frac_coords.detach())
            model(atoms, atoms.eq(0), normal)
            self.assertGreater(model._local_geometry_cache.nbytes, 0)
            model.float()
            self.assertEqual(model._local_geometry_cache.nbytes, 0)
            model.enable_local_geometry_cache(enabled=False)
            reference = model(atoms, atoms.eq(0), normal)
            model.enable_local_geometry_cache(max_bytes=2**20)
            actual = model(atoms, atoms.eq(0), normal)
            for name in actual:
                torch.testing.assert_close(actual[name], reference[name], atol=0, rtol=0)
            self.assertEqual(model._local_geometry_cache.bypasses, 1)
            self.assertEqual(model._local_geometry_cache.nbytes, 0)

    def test_inference_population_can_be_reused_for_training(self):
        model = Transformer(**parameters()).eval()
        model.enable_local_geometry_cache(max_bytes=2**20)
        atoms, geometry = fixture()
        with torch.inference_mode():
            model(atoms, atoms.eq(0), geometry)
        self.assertTrue(all(not e.geometry.angular_basis.is_inference()
                            for e in model._local_geometry_cache._entries.values()))
        model.train()
        values = model(atoms, atoms.eq(0), geometry)
        sum(v.square().mean() for v in values.values()).backward()
        self.assertGreater(model._local_geometry_cache.hits, 0)
        self.assertTrue(all(p.grad is not None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
