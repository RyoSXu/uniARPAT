"""Physical guarantees of the learned-spring phonon readout.

Synthetic crystals only: analytic zone-boundary frequencies, zero acoustic
modes at Γ, 3N mode count, exact mass scaling, E(3) and padding invariance,
and finite gradients through imaginary modes.
"""
import math
import unittest

import torch

from model.phonon_head import (
    CM_PER_SQRT_EV_A2_AMU, PhononSpringHead, binned_modes, harmonic_frequencies, mesh_sizes,
)
from tests.model.test_local_global_encoder import fixture
from utils.crystal import CrystalGeometry
from utils.periodic_geometry import build_periodic_records

DTYPE = torch.float64


def cubic_records(k_par, k_perp, a=3.0):
    shifts = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=DTYPE)
    unit = shifts
    eye = torch.eye(3, dtype=DTYPE)
    outer = unit[:, :, None] * unit[:, None, :]
    blocks = k_perp * eye + (k_par - k_perp) * outer
    zeros = torch.zeros(6, dtype=torch.long)
    return blocks, zeros, zeros, shifts


def frequency(lam):
    lam = torch.as_tensor(lam, dtype=DTYPE)
    return CM_PER_SQRT_EV_A2_AMU * lam / torch.sqrt(lam.abs() + (1 / CM_PER_SQRT_EV_A2_AMU) ** 2)


def head_inputs(cell, frac, z):
    mask = z == 0
    geometry = CrystalGeometry(cell, frac)
    records = build_periodic_records(geometry, mask, radius=6.0)
    records = records.to(dtype=DTYPE)
    return records, mask, geometry


class TestPhononHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(DTYPE)

    def tearDown(self):
        torch.set_default_dtype(self.old_dtype)

    def test_simple_cubic_zone_boundary_matches_analytic_dispersion(self):
        k_par, k_perp, mass = 2.0, 0.5, 10.0
        blocks, dst, src, delta = cubic_records(k_par, k_perp)
        q = torch.tensor([[0.5, 0.0, 0.0]], dtype=DTYPE)
        nu = harmonic_frequencies(blocks, dst, src, delta, torch.tensor([mass]), q)
        expected = torch.sort(frequency([4 * k_perp / mass, 4 * k_perp / mass, 4 * k_par / mass])).values
        torch.testing.assert_close(nu[0], expected, rtol=1e-10, atol=1e-8)

    def test_acoustic_modes_vanish_at_gamma(self):
        z, geometry = fixture(DTYPE)
        records, mask, geometry = head_inputs(geometry.cell, geometry.frac_coords, z)
        head = PhononSpringHead(16, 6.0)
        springs = head.springs(torch.randn(2, 4, 16), records)
        unit = records.vectors / records.distances[:, None]
        blocks = springs[:, 1, None, None] * torch.eye(3) + (springs[:, 0] - springs[:, 1])[:, None, None] \
            * unit[:, :, None] * unit[:, None, :]
        chosen = records.batch == 0
        delta = geometry.frac_coords[0, records.src[chosen]] - geometry.frac_coords[0, records.dst[chosen]] \
            + records.shifts[chosen].to(DTYPE)
        masses = head.masses[z[0, :3]]
        nu = harmonic_frequencies(blocks[chosen], records.dst[chosen], records.src[chosen], delta, masses,
                                  torch.zeros(1, 3))
        self.assertEqual(nu.shape, (1, 9))
        self.assertTrue((nu[0].abs().sort().values[:3] < 1e-3).all())
        self.assertGreater(float(nu[0].abs().max()), 1.0)

    def test_mass_scaling_and_mode_count(self):
        blocks, dst, src, delta = cubic_records(3.0, 1.0)
        q = torch.rand(7, 3, generator=torch.Generator().manual_seed(0), dtype=DTYPE)
        light = harmonic_frequencies(blocks, dst, src, delta, torch.tensor([4.0]), q)
        heavy = harmonic_frequencies(blocks, dst, src, delta, torch.tensor([16.0]), q)
        torch.testing.assert_close(heavy, light / 2, rtol=1e-4, atol=1e-3)
        modes = binned_modes(light, torch.linspace(-5000, 5000, 101, dtype=DTYPE), torch.tensor(5.0))
        self.assertAlmostEqual(float(modes.sum()), 3.0, places=8)

    def test_negative_springs_give_imaginary_modes_with_finite_gradients(self):
        springs = torch.tensor([-1.5, 0.5], dtype=DTYPE, requires_grad=True)
        blocks, dst, src, delta = cubic_records(springs[0], springs[1])
        nu = harmonic_frequencies(blocks, dst, src, delta, torch.tensor([10.0]), torch.tensor([[0.5, 0.25, 0.0]]))
        self.assertLess(float(nu.min()), -100.0)
        modes = binned_modes(nu, torch.linspace(-1000, 1000, 41, dtype=DTYPE), torch.tensor(20.0))
        (modes * torch.arange(40, dtype=DTYPE)).sum().backward()
        self.assertTrue(torch.isfinite(springs.grad).all())
        self.assertGreater(float(springs.grad.abs().sum()), 0.0)

    def test_head_is_e3_translation_and_padding_invariant(self):
        z, geometry = fixture(DTYPE)
        head = PhononSpringHead(16, 6.0)
        memory = torch.randn(2, 4, 16, generator=torch.Generator().manual_seed(1), dtype=DTYPE)
        centers = torch.linspace(-280 + 9.84375, 980 - 9.84375, 64, dtype=DTYPE)
        records, mask, geom = head_inputs(geometry.cell, geometry.frac_coords, z)
        base = head(memory, records, z, mask, geom, centers)
        self.assertTrue(torch.isfinite(base).all())
        torch.testing.assert_close(base.exp().sum(-1), torch.ones(2, dtype=DTYPE), rtol=0, atol=1e-4)

        angle = 0.7
        rotation = torch.tensor([[math.cos(angle), -math.sin(angle), 0.], [math.sin(angle), math.cos(angle), 0.],
                                 [0., 0., -1.]], dtype=DTYPE)
        moved = geometry.frac_coords + torch.tensor([0.13, -0.27, 0.41], dtype=DTYPE)
        moved = torch.where(mask[..., None], geometry.frac_coords, moved)
        records2, mask2, geom2 = head_inputs(geometry.cell @ rotation, moved, z)
        torch.testing.assert_close(head(memory, records2, z, mask2, geom2, centers), base, rtol=1e-7, atol=1e-7)

        single_z, single_geometry = z[1:, :2], CrystalGeometry(geometry.cell[1:], geometry.frac_coords[1:, :2])
        records3, mask3, geom3 = head_inputs(single_geometry.cell, single_geometry.frac_coords, single_z)
        torch.testing.assert_close(head(memory[1:, :2], records3, single_z, mask3, geom3, centers), base[1:],
                                   rtol=1e-9, atol=1e-9)

    def test_gradients_reach_atom_states_and_broadening(self):
        z, geometry = fixture(DTYPE)
        head = PhononSpringHead(16, 6.0)
        memory = torch.randn(2, 4, 16, dtype=DTYPE, requires_grad=True)
        records, mask, geom = head_inputs(geometry.cell, geometry.frac_coords, z)
        centers = torch.linspace(0, 1200, 8, dtype=DTYPE)
        logits = head(memory, records, z, mask, geom, centers)
        target = torch.softmax(torch.randn(2, 8, dtype=DTYPE), -1)
        (-(target * torch.log_softmax(logits, -1)).sum()).backward()
        self.assertTrue(torch.isfinite(memory.grad).all() and memory.grad.abs().sum() > 0)
        self.assertTrue(torch.isfinite(head.width.grad) and head.width.grad != 0)

    def test_mesh_is_denser_for_small_cells_and_capped(self):
        self.assertEqual(mesh_sizes(3.0 * torch.eye(3).numpy(), 20.0, 512), (7, 7, 7))
        self.assertEqual(mesh_sizes(25.0 * torch.eye(3).numpy(), 20.0, 512), (1, 1, 1))
        small = mesh_sizes(1.0 * torch.eye(3).numpy(), 20.0, 512)
        self.assertLessEqual(small[0] * small[1] * small[2], 512)


if __name__ == "__main__":
    unittest.main()
