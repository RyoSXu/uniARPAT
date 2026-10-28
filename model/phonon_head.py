"""Learned Born–von Kármán springs and exact harmonic phonon frequencies.

Every periodic neighbor record inside the local radius receives a longitudinal
and a transverse spring constant (eV/Å²) from the final invariant atom states
and the pair distance; springs vanish smoothly at the radius. A pair block is
K = k_par r̂r̂ᵀ + k_perp (I − r̂r̂ᵀ); its force constant is −K and the on-site
block is the neighbor sum, so a uniform translation costs no energy. The
mass-weighted dynamical matrix is diagonalized on a Γ-free shifted
Monkhorst–Pack mesh and the frequencies are binned onto the phDOS grid with
one learned Gaussian width.

Only the normalized spectral shape leaves the head, as logits. Current phDOS
labels hold a material-dependent fraction of 3N inside the window, so the
existing per-atom total head keeps the amount.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.periodic_neighbors import g2_quintic_cutoff, g2_rbf_features, g2_rbf_params

# sqrt(eV / (Å² · amu)) in cm⁻¹; the factor phonopy calls VaspToCm.
CM_PER_SQRT_EV_A2_AMU = 521.471
FREQUENCY_SMOOTHING = (1.0 / CM_PER_SQRT_EV_A2_AMU) ** 2
RADIAL_CHANNELS = 16


def atomic_mass_table(max_z=118):
    """Standard atomic masses (amu) indexed by Z; index 0 is padding."""
    from pymatgen.core.periodic_table import Element

    masses = torch.full((max_z + 1,), float("nan"), dtype=torch.float64)
    masses[0] = 0.
    for z in range(1, max_z + 1):
        try:
            value = Element.from_Z(z).atomic_mass
        except ValueError:
            continue
        if value is not None:
            masses[z] = float(value)
    return masses


def mesh_sizes(cell, q_length, max_points):
    """Per-axis mesh sizes ⌈q_length / interplanar spacing⌉, capped in total."""
    cell = np.asarray(cell, dtype=np.float64)
    volume = abs(float(np.linalg.det(cell)))
    sizes = []
    for axis in range(3):
        first, second = (k for k in range(3) if k != axis)
        spacing = volume / float(np.linalg.norm(np.cross(cell[first], cell[second])))
        sizes.append(max(1, math.ceil(q_length / spacing - 1e-9)))
    while int(np.prod(sizes)) > max_points:
        sizes[int(np.argmax(sizes))] -= 1
    return tuple(sizes)


def shifted_mesh(sizes, device, dtype):
    """Γ-free Monkhorst–Pack points (k + 1/2) / n in reciprocal-lattice units."""
    axes = [(torch.arange(n, device=device, dtype=dtype) + 0.5) / n for n in sizes]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def harmonic_frequencies(blocks, dst, src, frac_delta, masses, q_frac):
    """Frequencies in cm⁻¹ for one crystal; imaginary modes are negative.

    ``blocks`` [E,3,3] are springs K for records centred on ``dst`` with the
    neighbor image of ``src`` at fractional displacement ``frac_delta`` [E,3]
    (neighbor minus centre). Indices are compact atom indices, ``masses`` [N]
    in amu, ``q_frac`` [Q,3] in reciprocal-lattice units. Returns [Q, 3N].
    """
    n = masses.numel()
    flat = blocks.reshape(-1, 9)
    off = -flat * (masses[dst] * masses[src]).rsqrt()[:, None]
    theta = 2 * math.pi * (q_frac @ frac_delta.T)
    pair = dst * n + src
    real = flat.new_zeros(q_frac.shape[0], n * n, 9).index_add_(1, pair, torch.cos(theta)[..., None] * off)
    imag = flat.new_zeros(q_frac.shape[0], n * n, 9).index_add_(1, pair, torch.sin(theta)[..., None] * off)
    onsite = flat.new_zeros(n * n, 9).index_add_(0, dst * n + dst, flat / masses[dst][:, None])
    real = real + onsite

    def square(x):
        return x.view(-1, n, n, 3, 3).permute(0, 1, 3, 2, 4).reshape(-1, 3 * n, 3 * n)

    matrix = torch.complex(square(real), square(imag))
    eigen = torch.linalg.eigvalsh(0.5 * (matrix + matrix.conj().transpose(-1, -2)))
    return CM_PER_SQRT_EV_A2_AMU * eigen / torch.sqrt(eigen.abs() + FREQUENCY_SMOOTHING)


def binned_modes(frequencies, edges, sigma):
    """Expected modes per bin and cell under Gaussian broadening, q-averaged."""
    z = (edges[None, :] - frequencies.reshape(-1, 1)) / (sigma * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf(z))
    return (cdf[:, 1:] - cdf[:, :-1]).sum(0) / frequencies.shape[0]


def window_logits(modes, total_modes, floor):
    """Shape logits from in-window modes per cell.

    Normalizing by the fixed 3N instead of the in-window sum keeps gradients
    finite when every mode of a crystal leaves the window.
    """
    return torch.log(modes / total_modes[:, None] + floor)


class PhononSpringHead(nn.Module):
    def __init__(self, d_model, radius, *, atom_channels=64, hidden_channels=128, q_length=20.0,
                 max_q_points=512, initial_width_bins=1.0, min_width_bins=0.25, shape_floor=1e-6):
        super().__init__()
        if not initial_width_bins > min_width_bins > 0:
            raise ValueError("initial broadening must exceed its positive floor")
        self.radius = float(radius)
        self.q_length, self.max_q_points = float(q_length), int(max_q_points)
        self.min_width_bins, self.shape_floor = float(min_width_bins), float(shape_floor)
        self.atom = nn.Linear(d_model, atom_channels)
        self.edge = nn.Sequential(nn.Linear(2 * atom_channels + RADIAL_CHANNELS, hidden_channels),
                                  nn.SiLU(), nn.Linear(hidden_channels, 2))
        self.width = nn.Parameter(torch.tensor(math.log(math.expm1(initial_width_bins - min_width_bins))))
        self.register_buffer("masses", atomic_mass_table().to(torch.get_default_dtype()))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.atom.weight)
        nn.init.zeros_(self.atom.bias)
        nn.init.xavier_normal_(self.edge[0].weight)
        nn.init.zeros_(self.edge[0].bias)
        nn.init.normal_(self.edge[2].weight, std=1e-3)
        with torch.no_grad():
            self.edge[2].bias.copy_(torch.tensor([1.0, 0.1]))

    def springs(self, memory, records):
        """[E, 2] longitudinal and transverse springs, symmetric in the two atoms."""
        atoms = self.atom(memory)
        left, right = atoms[records.batch, records.dst], atoms[records.batch, records.src]
        centers, width = g2_rbf_params(self.radius, RADIAL_CHANNELS, device=memory.device, dtype=memory.dtype)
        radial = g2_rbf_features(records.distances, self.radius, centers, width)
        features = torch.cat((left + right, left * right, radial), -1)
        return self.edge(features) * g2_quintic_cutoff(records.distances, self.radius)[:, None]

    def forward(self, memory, records, atomic_numbers, padding_mask, geometry, centers):
        if records is None or records.radius != self.radius:
            raise ValueError("phonon head requires the encoder's periodic records")
        if centers is None:
            raise ValueError("phonon head requires phDOS bin centers")
        centers = centers.to(device=memory.device, dtype=memory.dtype)
        if centers.ndim == 2:
            if not torch.equal(centers, centers[:1].expand_as(centers)):
                raise ValueError("one shared phDOS grid per batch required")
            centers = centers[0]
        spacing = centers[1:] - centers[:-1]
        if centers.ndim != 1 or len(centers) < 2 or not torch.allclose(spacing, spacing[:1].expand_as(spacing)):
            raise ValueError("uniform phDOS bins required")
        bin_width = spacing[0]
        edges = torch.cat((centers[:1] - bin_width / 2, centers + bin_width / 2))
        sigma = bin_width * (self.min_width_bins + F.softplus(self.width))

        springs = self.springs(memory, records)
        unit = records.vectors / records.distances[:, None]
        outer = unit[:, :, None] * unit[:, None, :]
        eye = torch.eye(3, device=memory.device, dtype=memory.dtype)
        blocks = springs[:, 1, None, None] * eye + (springs[:, 0] - springs[:, 1])[:, None, None] * outer

        valid = ~padding_mask
        masses = self.masses.to(memory.dtype)[atomic_numbers]
        if not torch.isfinite(masses[valid]).all() or (masses[valid] <= 0).any():
            raise ValueError("every atom needs a standard atomic mass")
        compact = valid.long().cumsum(-1) - 1
        frac = geometry.frac_coords.to(memory.dtype)
        cells = geometry.cell.detach().to("cpu", torch.float64).numpy()
        counts = valid.sum(-1).tolist()
        order = torch.argsort(records.batch, stable=True)
        per_crystal = torch.bincount(records.batch, minlength=len(counts)).tolist()
        meshes, spectra, start = {}, [], 0
        for b, n in enumerate(counts):
            if n == 0:
                raise ValueError("phonon head needs at least one atom per crystal")
            chosen = order[start:start + per_crystal[b]]
            start += per_crystal[b]
            dst, src = records.dst[chosen], records.src[chosen]
            delta = frac[b, src] - frac[b, dst] + records.shifts[chosen].to(memory.dtype)
            sizes = mesh_sizes(cells[b], self.q_length, self.max_q_points)
            if sizes not in meshes:
                meshes[sizes] = shifted_mesh(sizes, memory.device, memory.dtype)
            nu = harmonic_frequencies(blocks[chosen], compact[b, dst], compact[b, src], delta,
                                      masses[b][valid[b]], meshes[sizes])
            spectra.append(binned_modes(nu, edges, sigma))
        total = 3.0 * valid.sum(-1).to(memory.dtype)
        return window_logits(torch.stack(spectra), total, self.shape_floor)
