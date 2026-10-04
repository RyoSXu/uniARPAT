import os
from torch.utils.data import Dataset
import numpy as np
import torch
import json
from pathlib import Path
from utils.crystal import CRYSTAL_SCHEMA, CrystalGeometry
from utils.crystal_cache import PackedCrystalStore
from utils.paths import REFERENCE_ROOT, input_path


class Dos_Dataset(Dataset):
    def __init__(self, data_dir="./data", split='train', dos_minmax = False, dos_zscore=False, scale_factor=1.0, apply_log=False, smear=0, choice=[], augment=False, disp_sigma=0.01, disp_clip=0.03, dos_sumnorm=False, edos_edges=None, phdos_edges=None, coords="auto", **kwargs) -> None:
        super().__init__()
        self.split = split
        # Sum normalization replaces min-max normalization when selected.
        # min/max slots are reused as (0, sum) so oracle denorm pred*(max-min)+min
        # == pred*sum keeps working UNCHANGED in all eval code. mean/std dummies.
        self.dos_sumnorm = bool(dos_sumnorm)
        if self.dos_sumnorm:
            dos_minmax = False
            dos_zscore = False
        # Coordinate displacement is applied only to training samples.
        # No rotation augmentation is applied here. Explicit matrix inputs
        # preserve Cartesian frames; model E(3) invariance is a separate claim.
        self.augment = bool(augment) and (split == 'train')
        self.disp_sigma = float(disp_sigma)
        self.disp_clip = float(disp_clip)
        self.smear = smear
        self.data_dir = str(input_path(data_dir))+"/"+split+"/"

        manifest_path = Path(input_path(data_dir)) / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        schema = manifest.get("schema")
        if schema is not None and schema != CRYSTAL_SCHEMA:
            raise ValueError(f"unsupported data schema: {schema}")
        self.crystal_format = schema == CRYSTAL_SCHEMA
        if self.crystal_format:
            self.crystals = PackedCrystalStore(input_path(data_dir), split)
            self.structure_indices = np.arange(len(self.crystals))
            self.atom_counts = self.crystals.n_atoms.clone()
            data_len = len(self.crystals)
        else:
            self.elements = self.get_elements()
            self.positions = self.get_positions()
            self.atom_counts = (self.elements[:, 2:] != 0).sum(-1)
            data_len = len(self.elements)
            self.structure_indices = np.arange(data_len)
        # Coverage masks are optional because older caches do not provide them.
        self.masks_available = (
            os.path.exists(os.path.join(self.data_dir, f"edos_mask_{split}.npy"))
            and os.path.exists(os.path.join(self.data_dir, f"phdos_mask_{split}.npy")))

        if self.split == 'test_cif':
            self.edos_tgtdos = torch.zeros((data_len, 128), dtype=torch.float32)
            self.phdos_tgtdos = torch.zeros((data_len, 64), dtype=torch.float32)
        else:
            self.edos_tgtdos = self.get_dos_data(prefix="edos_tgtdos")
            self.phdos_tgtdos = self.get_dos_data(prefix="phdos_tgtdos")

        self.edos_mask = self.get_mask_data(prefix="edos_mask")
        self.phdos_mask = self.get_mask_data(prefix="phdos_mask")
        # Valence counts are optional metadata for blind scale supervision.
        self.nvalence = self.get_nvalence()
        for name in ("edos_tgtdos", "phdos_tgtdos", "edos_mask", "phdos_mask", "nvalence"):
            value = getattr(self, name)
            if value is not None and len(value) != data_len:
                raise ValueError(f"{name} row count differs from structures")
        if self.crystal_format and (not self.masks_available or self.nvalence is None):
            raise ValueError("crystal cache requires coverage masks and valence counts")
        # Bin centers in physical units are metadata, not target labels.
        # Coordinates are grid constants, not labels: eDOS is relative to Fermi
        # energy and phDOS is relative to zero frequency. They are attached only
        # when the output lengths match the production grid.
        self.coords_mode = coords if isinstance(coords, str) else "auto"
        try:
            self.edos_x, self.phdos_x = self.get_grid_coords(
                data_dir, edos_edges, phdos_edges)
        except AssertionError:
            if self.coords_mode == "on":
                raise
            self.edos_x, self.phdos_x = None, None

        self.edos_mean = torch.mean(self.edos_tgtdos, dim=1, keepdim=True).float()
        self.edos_std = torch.std(self.edos_tgtdos, dim=1, keepdim=True).float()
        self.edos_min = torch.min(self.edos_tgtdos, dim=1, keepdim=True).values.float()
        self.edos_max = torch.max(self.edos_tgtdos, dim=1, keepdim=True).values.float()

        self.phdos_mean = torch.mean(self.phdos_tgtdos, dim=1, keepdim=True).float()
        self.phdos_std = torch.std(self.phdos_tgtdos, dim=1, keepdim=True).float()
        self.phdos_min = torch.min(self.phdos_tgtdos, dim=1, keepdim=True).values.float()
        self.phdos_max = torch.max(self.phdos_tgtdos, dim=1, keepdim=True).values.float()

        if scale_factor != 1.0:
            self.edos_tgtdos = self.edos_tgtdos * scale_factor
            self.phdos_tgtdos = self.phdos_tgtdos * scale_factor

        if apply_log:
            self.edos_tgtdos = torch.log1p(self.edos_tgtdos)
            self.phdos_tgtdos = torch.log1p(self.phdos_tgtdos)

        if dos_zscore:
            self.edos_tgtdos = (self.edos_tgtdos - self.edos_mean) / (self.edos_std + 1e-8)
            self.phdos_tgtdos = (self.phdos_tgtdos - self.phdos_mean) / (self.phdos_std + 1e-8)

        if dos_minmax:
            self.edos_tgtdos = (self.edos_tgtdos - self.edos_min) / (self.edos_max - self.edos_min + 1e-8)
            self.phdos_tgtdos = (self.phdos_tgtdos - self.phdos_min) / (self.phdos_max - self.phdos_min + 1e-8)

        if self.dos_sumnorm:
            e_sum = self.edos_tgtdos.sum(dim=1, keepdim=True)
            p_sum = self.phdos_tgtdos.sum(dim=1, keepdim=True)
            self.edos_tgtdos = self.edos_tgtdos / (e_sum + 1e-12)
            self.phdos_tgtdos = self.phdos_tgtdos / (p_sum + 1e-12)
            self.edos_min = torch.zeros_like(e_sum)
            self.edos_max = e_sum.float()
            self.phdos_min = torch.zeros_like(p_sum)
            self.phdos_max = p_sum.float()
            self.edos_mean = torch.ones_like(e_sum)
            self.edos_std = torch.ones_like(e_sum)
            self.phdos_mean = torch.ones_like(p_sum)
            self.phdos_std = torch.ones_like(p_sum)

        if self.crystal_format and split == "train":
            excluded = set(self.crystals.manifest.get("training_exclusions", []))
            allowed = [i for i, material_id in enumerate(self.crystals.material_ids)
                       if str(material_id) not in excluded]
            self._select_rows(allowed)
        if len(choice) != 0:
            self._select_rows(choice)

    def _select_rows(self, choice):
        cholist = torch.tensor(choice, dtype=torch.long)
        self.structure_indices = self.structure_indices[np.asarray(choice)]
        if not self.crystal_format:
            self.elements = self.elements.index_select(0, cholist)
            self.positions = self.positions.index_select(0, cholist)
        for name in ("atom_counts", "edos_tgtdos", "phdos_tgtdos", "edos_mask",
                     "phdos_mask", "nvalence", "edos_mean", "edos_std", "edos_min",
                     "edos_max", "phdos_mean", "phdos_std", "phdos_min", "phdos_max"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.index_select(0, cholist))

    @property
    def material_ids(self):
        if self.crystal_format:
            return self.crystals.material_ids[self.structure_indices]
        return np.load(os.path.join(self.data_dir, f"{self.split}_index.npy"), allow_pickle=False)[self.structure_indices]

    def __len__(self):
        return len(self.atom_counts)

    def __getitem__(self, index):
        index = min(index, self.__len__() - 1)
        if self.crystal_format:
            elements, geometry = self.crystals[int(self.structure_indices[index])]
            pos = CrystalGeometry(geometry.cell, geometry.frac_coords.clone())
        else:
            elements = self.elements[index]
            pos = self.positions[index].reshape(-1, 3).clone()
        if self.augment:
            # Perturb fractional atom rows and wrap them into the unit cell.
            # The first two element slots are non-atom sentinel tokens.
            n_atom = int(self.atom_counts[index])
            noise = np.random.normal(0.0, self.disp_sigma, size=(n_atom, 3))
            noise = np.clip(noise, -self.disp_clip, self.disp_clip)
            if self.crystal_format:
                pos = CrystalGeometry(pos.cell, (pos.frac_coords + torch.from_numpy(noise).float()) % 1.0)
            else:
                pos[2:2 + n_atom] = (pos[2:2 + n_atom] + torch.from_numpy(noise).to(pos.dtype)) % 1.0
        # The base batch has 15 fields; bin centers are appended only for the
        # production grid. Missing masks are converted to all-ones downstream.
        items = [
            elements,                      # [0] atomic numbers (legacy: two sentinels)
            pos,                           # [1] explicit geometry or legacy rows
            self.edos_tgtdos[index],        # [2]
            self.phdos_tgtdos[index],       # [3]
            self.edos_mean[index],          # [4]
            self.edos_std[index],           # [5]
            self.edos_min[index],           # [6]
            self.edos_max[index],           # [7]
            self.phdos_mean[index],         # [8]
            self.phdos_std[index],          # [9]
            self.phdos_min[index],          # [10]
            self.phdos_max[index],          # [11]
            self.edos_mask[index] if self.edos_mask is not None else None,   # [12]
            self.phdos_mask[index] if self.phdos_mask is not None else None,  # [13]
            self.nvalence[index] if self.nvalence is not None else None,      # [14] H1 N_val
        ]
        if self.edos_x is not None and self.phdos_x is not None:
            items += [
                self.edos_x.clone(),  # [15] Q1 eDOS bin centers (eV @ Fermi)
                self.phdos_x.clone(),  # [16] Q1 phDOS bin centers (cm^-1 @ nu=0)
            ]
        return items

    def get_elements(self):
        filename = os.path.join(self.data_dir, f"elements_{self.split}.npy")
        return torch.from_numpy(np.load(filename)).long()

    def get_positions(self):
        filename = os.path.join(self.data_dir, f"positions_{self.split}.npy")
        return torch.from_numpy(np.load(filename)).float()

    def get_dos_data(self, prefix):
        filename = os.path.join(self.data_dir, f"{prefix}_{self.split}.npy")
        array = np.load(filename, allow_pickle=False)
        if self.crystal_format and array.dtype != np.float32:
            raise ValueError("crystal DOS arrays must be stored in float32")
        return torch.from_numpy(array).float()

    def get_mask_data(self, prefix):
        if not self.masks_available:
            return None
        filename = os.path.join(self.data_dir, f"{prefix}_{self.split}.npy")
        array = np.load(filename, allow_pickle=False)
        if self.crystal_format and array.dtype != np.bool_:
            raise ValueError("crystal coverage masks must be stored as boolean")
        return torch.from_numpy(array).bool()

    def get_nvalence(self):
        filename = os.path.join(self.data_dir, f"nvalence_{self.split}.npy")
        if not os.path.exists(filename):
            return None
        array = np.load(filename, allow_pickle=False)
        if self.crystal_format and (array.dtype != np.float32 or not np.isfinite(array).all()
                                    or (array <= 0).any()):
            raise ValueError("crystal valence counts require positive finite float32 values")
        return torch.from_numpy(array).float()

    @staticmethod
    def _production_edges():
        """Frozen production anchor E0+P0 from grids.json (C2b verdict)."""
        import json
        with open(REFERENCE_ROOT / "grids.json") as f:
            grids = json.load(f)
        return (np.asarray(grids["E0"], dtype=np.float64),
                np.asarray(grids["P0"], dtype=np.float64))

    def get_grid_coords(self, data_dir, edos_edges, phdos_edges):
        if edos_edges is None or phdos_edges is None:
            _e0, _p0 = self._production_edges()
            if edos_edges is None:
                edos_edges = _e0
            if phdos_edges is None:
                phdos_edges = _p0
        edos_edges = np.asarray(edos_edges, dtype=np.float64)
        phdos_edges = np.asarray(phdos_edges, dtype=np.float64)
        assert len(edos_edges) - 1 == self.edos_tgtdos.shape[1], \
            f"Q1 edos bins {len(edos_edges)-1} != targets {self.edos_tgtdos.shape[1]}"
        assert len(phdos_edges) - 1 == self.phdos_tgtdos.shape[1], \
            f"Q1 phdos bins {len(phdos_edges)-1} != targets {self.phdos_tgtdos.shape[1]}"
        edos_x = torch.tensor((edos_edges[:-1] + edos_edges[1:]) / 2, dtype=torch.float32)
        phdos_x = torch.tensor((phdos_edges[:-1] + phdos_edges[1:]) / 2, dtype=torch.float32)
        return edos_x, phdos_x

if __name__ == "__main__":
    test = Dos_Dataset(data_dir="./data/train4ARPAT", split="train")
    print(test.__getitem__(15))
