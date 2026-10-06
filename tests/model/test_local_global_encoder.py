"""Full scalar Encoder/DOS contracts; no optimizer or training data required."""
import unittest
from unittest.mock import patch

import torch

from model.transformer import Transformer
from tools.eval.periodic_geometry_acceptance import e3_operations
from utils.crystal import CrystalGeometry
from utils.local_global_experiment import initialize_comparison_model
from utils.spectral_inference import reconstruct_h1_spectra


def parameters(route='A', layout='staged', depth=2):
    return dict(d_model=16, nhead=4, edos_num=8, phdos_num=4,
                num_decoder_layers=1, dim_feedforward=32, dropout=0.,
                atom_feat_mode='z_only_proj', scale_mode='eta', local_route=route,
                local_layout=layout, local_depth=depth)


def fixture(dtype=torch.float32):
    z = torch.tensor([[14, 14, 8, 0], [6, 6, 0, 0]])
    cell = torch.tensor([[12., 0., 0.], [1., 13., 0.], [.4, -.6, 14.]], dtype=dtype)
    frac = torch.tensor([[[.1, .1, .1], [.23, .14, .2], [.31, .3, .18], [0., 0., 0.]],
                         [[.05, .05, .05], [.16, .11, .22], [0., 0., 0.], [0., 0., 0.]]], dtype=dtype)
    return z, CrystalGeometry(cell[None].repeat(2, 1, 1), frac)


class TestLocalGlobalEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def create(self, route='A', layout='staged', depth=2, dtype=torch.float32, q1_coord=False, dropout=0.):
        before = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            params = parameters(route, layout, depth)
            params['q1_coord'] = q1_coord
            params['dropout'] = dropout
            model = Transformer(**params)
            info = initialize_comparison_model(model, params, 42)
        finally:
            torch.set_default_dtype(before)
        return model.eval(), info

    def test_full_A_keeps_activations_with_equal_training_outputs_gradients_and_rng(self):
        from model.periodic_messages import checkpoint

        for dtype in (torch.float32, torch.float64):
            z, geometry = fixture(dtype)
            for layout in ('staged', 'alternating'):
                with self.subTest(dtype=dtype, layout=layout):
                    model, _ = self.create(layout=layout, dtype=dtype, dropout=.05)
                    model.train()
                    blocks = [block for layer in model.encoder.local_layers for block in layer.message.blocks]
                    self.assertTrue(all(not block.recompute_angles for block in blocks))
                    self.assertTrue(all(block.pair_chunk == 4096 for block in blocks))
                    state = {key: value.clone() for key, value in model.state_dict().items()}
                    rng = torch.get_rng_state()
                    results = []
                    for recompute in (True, False):
                        for block in blocks:
                            block.recompute_angles = recompute
                        torch.set_rng_state(rng)
                        model.zero_grad(set_to_none=True)
                        with patch('model.periodic_messages.checkpoint', wraps=checkpoint) as calls:
                            out = model(z, z.eq(0), geometry)
                            sum(value.square().mean() for value in out.values()).backward()
                            if recompute:
                                self.assertGreater(calls.call_count, 0)
                            else:
                                calls.assert_not_called()
                        grads = {name: p.grad.clone() if p.grad is not None else None
                                 for name, p in model.named_parameters()}
                        self.assertTrue(all(torch.isfinite(g).all() for g in grads.values() if g is not None))
                        results.append((out, grads, torch.get_rng_state()))
                    atol, rtol = (2e-6, 2e-5) if dtype == torch.float32 else (2e-12, 2e-10)
                    for key in results[0][0]:
                        torch.testing.assert_close(results[1][0][key], results[0][0][key], atol=atol, rtol=rtol)
                    for key, expected in results[0][1].items():
                        actual = results[1][1][key]
                        if expected is None:
                            self.assertIsNone(actual, key)
                        else:
                            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                    torch.testing.assert_close(results[1][2], results[0][2], atol=0, rtol=0)
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value, state[key], atol=0, rtol=0)

    def test_all_routes_and_layouts_have_full_output_e3_invariance(self):
        for dtype in (torch.float32, torch.float64):
            tolerance = 2e-5 if dtype == torch.float32 else 2e-8
            z, geometry = fixture(dtype)
            for route in ('A', 'B+'):
                for layout in ('staged', 'alternating'):
                    model, _ = self.create(route, layout, dtype=dtype)
                    with torch.no_grad():
                        expected = model(z, z.eq(0), geometry)
                        expected.update(reconstruct_h1_spectra(expected, [3, 2], [16., 8.]))
                        for name, matrix, translation in e3_operations():
                            q, t = torch.as_tensor(matrix, dtype=dtype), torch.as_tensor(translation, dtype=dtype)
                            cell = geometry.cell @ q
                            frac = geometry.frac_coords + torch.bmm(
                                t.expand(2, 1, 3), torch.linalg.inv(cell))
                            actual = model(z, z.eq(0), CrystalGeometry(cell, frac))
                            actual.update(reconstruct_h1_spectra(actual, [3, 2], [16., 8.]))
                            for key in expected:
                                with self.subTest(dtype=dtype, route=route, layout=layout, operation=name, key=key):
                                    torch.testing.assert_close(actual[key], expected[key], atol=tolerance, rtol=tolerance)

    def test_atom_reordering_padding_poison_and_geometry_bypass(self):
        z, geometry = fixture()
        for route in ('A', 'B+'):
            model, _ = self.create(route, 'alternating')
            with patch('model.transformer.compute_relative_features', side_effect=AssertionError('old geometry read')), \
                    patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as decoder:
                with torch.no_grad():
                    expected = model(z, z.eq(0), geometry)
                    permutation = torch.tensor([2, 0, 3, 1])
                    reordered = model(z[:, permutation], z[:, permutation].eq(0),
                                      CrystalGeometry(geometry.cell, geometry.frac_coords[:, permutation]))
                    poison = geometry.frac_coords.clone()
                    poison[z.eq(0)] = float('nan')
                    poisoned = model(z, z.eq(0), CrystalGeometry(geometry.cell, poison))
                self.assertTrue(all(call.kwargs['pos'] is None for call in decoder.call_args_list))
            self.assertFalse(any('rp_proj' in key or 'rp_encoder' in key for key in model.state_dict()))
            for key in expected:
                torch.testing.assert_close(reordered[key], expected[key], atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(poisoned[key], expected[key], atol=0, rtol=0)

    def test_finite_full_backward_reaches_every_component_and_independent_layers(self):
        z, geometry = fixture()
        for route in ('A', 'B+'):
            model, _ = self.create(route, 'alternating')
            out = model(z, z.eq(0), geometry)
            sum(value.square().mean() for value in out.values()).backward()
            for prefix in ('tok_emb', 'atom_norm', 'atom_proj', 'encoder.local_layers.0', 'encoder.local_layers.1',
                           'encoder.global_layers.0', 'encoder.global_layers.1', 'decoder',
                           'edos_out_head', 'phdos_out_head', 'eta_head'):
                grads = [p.grad for name, p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
                self.assertTrue(grads, prefix)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads), prefix)
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0, prefix)
            storage = [p.data_ptr() for p in model.parameters() if p.numel()]
            self.assertEqual(len(storage), len(set(storage)))

    def test_q1_parameter_gradients_are_present_and_respect_zero_initialization(self):
        z, geometry = fixture()
        grids = (torch.linspace(-6., 6., 8), torch.linspace(-280., 980., 4))
        for route in ('A', 'B+'):
            for layout in ('staged', 'alternating'):
                with self.subTest(route=route, layout=layout):
                    model, _ = self.create(route, layout, q1_coord=True)
                    out = model(z, z.eq(0), geometry, *grids)
                    sum(value.square().mean() for value in out.values()).backward()
                    for prefix in ('atom_norm', 'edos_trunk', 'phdos_trunk'):
                        params = [p for name, p in model.named_parameters() if name.startswith(prefix)]
                        self.assertTrue(params, prefix)
                        self.assertTrue(all(p.requires_grad and p.grad is not None for p in params), prefix)
                        self.assertTrue(all(torch.isfinite(p.grad).all() for p in params), prefix)
                        self.assertGreater(sum(float(p.grad.abs().sum()) for p in params), 0, prefix)
                    # The initial zero output projection blocks gradients to
                    # fc1 on this first backward; this is an intentional gate.
                    for trunk in (model.edos_trunk, model.phdos_trunk):
                        self.assertGreater(float(trunk.proj.weight.grad.abs().sum()), 0)
                        for p in trunk.fc1.parameters():
                            torch.testing.assert_close(p.grad, torch.zeros_like(p), atol=0, rtol=0)

    def test_empty_structures_and_padding_encoder_states_are_finite_zero(self):
        for route in ('A', 'B+'):
            model, _ = self.create(route)
            for length in (0, 3):
                z = torch.zeros(2, length, dtype=torch.long)
                geometry = CrystalGeometry(torch.eye(3)[None].repeat(2, 1, 1) * 15,
                                           torch.full((2, length, 3), float('nan')))
                captured = []
                handle = model.encoder.register_forward_hook(lambda m, args, out: captured.append(out))
                with torch.no_grad():
                    out = model(z, z.eq(0), geometry)
                handle.remove()
                self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))
                self.assertTrue((captured[0] == 0).all())
                spectra = reconstruct_h1_spectra(out, [0, 0], [0, 0])
                self.assertTrue((spectra['blind_edos'] == 0).all())
                self.assertTrue((spectra['blind_phdos'] == 0).all())

    def test_mixed_empty_rows_match_separate_inference_and_keep_backward_finite(self):
        z, geometry = fixture()
        z[1] = 0
        frac = geometry.frac_coords.clone()
        frac[1] = float('nan')
        geometry = CrystalGeometry(geometry.cell, frac)
        for route in ('A', 'B+'):
            for layout in ('staged', 'alternating'):
                with self.subTest(route=route, layout=layout):
                    model, _ = self.create(route, layout)
                    captured = []
                    hook = model.encoder.register_forward_hook(lambda m, args, out: captured.append(out.detach()))
                    with torch.no_grad():
                        mixed = model(z, z.eq(0), geometry)
                    hook.remove()
                    self.assertTrue((captured[0][1] == 0).all())
                    with torch.no_grad():
                        nonempty = model(z[:1], z[:1].eq(0),
                            CrystalGeometry(geometry.cell[:1], geometry.frac_coords[:1]))
                        # Removing all atom slots exercises the zero-width
                        # decoder boundary as well as the padded empty row.
                        empty = model(z[1:, :0], z[1:, :0].eq(0),
                            CrystalGeometry(geometry.cell[1:], geometry.frac_coords[1:, :0]))
                    for key, value in mixed.items():
                        self.assertTrue(torch.isfinite(value).all(), key)
                        torch.testing.assert_close(value, torch.cat((nonempty[key], empty[key])),
                                                   atol=2e-5, rtol=2e-5)
                    spectra = reconstruct_h1_spectra(mixed, [3, 0], [16., 0.])
                    self.assertTrue((spectra['blind_edos'][1] == 0).all())
                    self.assertTrue((spectra['blind_phdos'][1] == 0).all())
                    out = model(z, z.eq(0), geometry)
                    sum(value.square().mean() for value in out.values()).backward()
                    for prefix in ('tok_emb', 'atom_norm', 'encoder.local_layers', 'encoder.global_layers'):
                        grads = [p.grad for name, p in model.named_parameters()
                                 if name.startswith(prefix) and p.grad is not None]
                        self.assertTrue(grads, prefix)
                        self.assertTrue(all(torch.isfinite(g).all() for g in grads), prefix)
                        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0, prefix)

    def test_common_initialization_counts_single_round_and_legacy_state(self):
        common = []
        states = {}
        for route in ('A', 'B+'):
            for layout in ('staged', 'alternating'):
                model, info = self.create(route, layout)
                common.append(info['common_state_sha256'])
                states[route, layout] = info['state_sha256']
                self.assertEqual(model.encoder.update_counts['internal_local_blocks'], 4)
        self.assertEqual(len(set(common)), 1)
        for route in ('A', 'B+'):
            self.assertEqual(states[route, 'staged'], states[route, 'alternating'])
            staged, _ = self.create(route, 'staged', 1)
            alternate, _ = self.create(route, 'alternating', 1)
            z, geometry = fixture()
            with torch.no_grad():
                a, b = staged(z, z.eq(0), geometry), alternate(z, z.eq(0), geometry)
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        legacy = {key: value for key, value in parameters().items() if not key.startswith('local_')}
        torch.manual_seed(52); original = Transformer(**legacy)
        torch.manual_seed(52); explicit = Transformer(**legacy, local_route='none')
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[key], atol=0, rtol=0)

    def test_conflicting_geometry_paths_and_atomic_number_boundary(self):
        for flag in ('use_g1', 'use_g2', 'use_periodic_manybody', 'use_macro_lattice'):
            with self.assertRaisesRegex(ValueError, 'only through'):
                Transformer(**parameters(), **{flag: True})
        z = torch.tensor([[118]])
        geometry = CrystalGeometry(torch.eye(3)[None] * 15, torch.zeros(1, 1, 3))
        for route in ('A', 'B+'):
            for layout in ('staged', 'alternating'):
                model, _ = self.create(route, layout)
                with torch.no_grad():
                    self.assertTrue(torch.isfinite(model(z, z.eq(0), geometry)['edos']).all())
                    for bad_number in (-1, 0, 119):
                        with self.subTest(route=route, layout=layout, number=bad_number):
                            with self.assertRaisesRegex(ValueError, '1,118'):
                                # Zero is legal only when explicitly masked
                                # as padding, not as a valid atomic number.
                                model(torch.full_like(z, bad_number), torch.zeros_like(z, dtype=torch.bool), geometry)


if __name__ == '__main__':
    unittest.main()
