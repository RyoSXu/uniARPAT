import os
from torch.utils.data import Dataset
import numpy as np
import torch


class Dos_Dataset(Dataset):
    def __init__(self, data_dir="./data", split='train', dos_minmax = False, dos_zscore=False, scale_factor=1.0, apply_log=False, smear=0, choice=[], augment=False, disp_sigma=0.01, disp_clip=0.03, dos_sumnorm=False, edos_edges=None, phdos_edges=None, coords="auto", **kwargs) -> None:
        super().__init__()
        self.split = split
        # C2.1: SumNorm replaces minmax (mutually exclusive; sumnorm wins if both).
        # min/max slots are reused as (0, sum) so oracle denorm pred*(max-min)+min
        # == pred*sum keeps working UNCHANGED in all eval code. mean/std dummies.
        self.dos_sumnorm = bool(dos_sumnorm)
        if self.dos_sumnorm:
            dos_minmax = False
            dos_zscore = False
        # C1.4: train-only stochastic displacement (valid/test stay deterministic).
        # Rotation is intentionally ABSENT: the 82-format carries no orientation
        # (lattice scalars + frac coords; build_cell convention fixed), so any
        # global rotation maps to bit-identical inputs => provably no-op.
        # Corollary: the model is E(3)-invariant by representation blindness
        # (translation/rotation/reflection all collapse), just not equivariant.
        self.augment = bool(augment) and (split == 'train')
        self.disp_sigma = float(disp_sigma)
        self.disp_clip = float(disp_clip)
        self.smear = smear
        self.data_dir = data_dir+"/"+split+"/"
        
        self.elements  = self.get_elements()  #size (__len__, src_len)
        self.positions = self.get_positions() #size (__len__, src_len*3)
        # C2.3: coverage masks (optional files; v1 cache lacks them -> None).
        self.masks_available = (
            os.path.exists(os.path.join(self.data_dir, f"edos_mask_{split}.npy"))
            and os.path.exists(os.path.join(self.data_dir, f"phdos_mask_{split}.npy")))
        data_len = self.positions.shape[0]

        if self.split == 'test_cif':
            self.edos_tgtdos = torch.zeros((data_len, 128), dtype=torch.float32)
            self.phdos_tgtdos = torch.zeros((data_len, 64), dtype=torch.float32)
        else:
            self.edos_tgtdos = self.get_dos_data(prefix="edos_tgtdos")
            self.phdos_tgtdos = self.get_dos_data(prefix="phdos_tgtdos")

        self.edos_mask = self.get_mask_data(prefix="edos_mask")
        self.phdos_mask = self.get_mask_data(prefix="phdos_mask")
        # H1: per-sample N_valence sidecar (Z0 frozen; v1 cache lacks it -> None).
        self.nvalence = self.get_nvalence()
        # E9-P0 Q1: task bin centers in physical units (Design-E section 9).
        # eDOS eV @ Fermi=0 (E0), phDOS cm^-1 @ nu=0 (P0); constant per grid,
        # returned per-sample so the model stays grid-agnostic (future warp
        # grids only change the dataset, never the model). Coordinates are
        # grid constants, NOT labels: no leakage.
        # coords="auto" (default): production grids (128/64) attach [15]/[16];
        # non-production grids (e.g. C2b E1/E2/P1/P2) fall back to legacy
        # 15-item batch so off-path runs never crash. coords="on" asserts.
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

        if len(choice) != 0:
            cholist = torch.Tensor(choice).long()
            self.elements = self.elements.index_select(dim=0, index=cholist)
            self.positions = self.positions.index_select(dim=0, index=cholist)
            self.edos_tgtdos = self.edos_tgtdos.index_select(dim=0, index=cholist)
            self.phdos_tgtdos = self.phdos_tgtdos.index_select(dim=0, index=cholist)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, index):
        index = min(index, self.__len__() - 1)
        pos = self.positions[index].reshape(-1, 3).clone() \
            if torch.is_tensor(self.positions[index]) else self.positions[index].reshape(-1, 3).copy()
        if self.augment:
            # C1.4 phonon displacement: frac rows only, periodic wrap.
            # NOTE: elements[0:2] are sentinels (126/127, nonzero) -> count from [2:].
            el = self.elements[index]
            n_atom = int(((el[2:] != 0).sum()).item()) if torch.is_tensor(el) \
                else int((el[2:] != 0).sum())
            n_atom = max(0, min(n_atom, pos.shape[0] - 2))
            noise = np.random.normal(0.0, self.disp_sigma, size=(n_atom, 3))
            noise = np.clip(noise, -self.disp_clip, self.disp_clip)
            if torch.is_tensor(pos):
                pos[2:2 + n_atom] = (pos[2:2 + n_atom] + torch.from_numpy(noise).to(pos.dtype)) % 1.0
            else:
                pos[2:2 + n_atom] = (pos[2:2 + n_atom] + noise) % 1.0
        # 返回 15 个基础元素；Q1 坐标按需附后（生产网格才有，无文件/非生产网格为None，下游转None）。
        # [0-11] legacy 12 元组，[12-13] C2.3 掩膜（无文件时为None，下游转全1），[14] H1 N_val。
        items = [
            self.elements[index],           # [0]
            pos.reshape(-1, 3),             # [1] (82,3; 与原格式一致)
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
        return torch.from_numpy(np.load(filename)).float()

    def get_mask_data(self, prefix):
        if not self.masks_available:
            return None
        filename = os.path.join(self.data_dir, f"{prefix}_{self.split}.npy")
        return torch.from_numpy(np.load(filename)).bool()

    def get_nvalence(self):
        filename = os.path.join(self.data_dir, f"nvalence_{self.split}.npy")
        if not os.path.exists(filename):
            return None
        return torch.from_numpy(np.load(filename)).float()

    @staticmethod
    def _production_edges():
        """Frozen production anchor E0+P0 from grids.json (C2b verdict)."""
        import json
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(here, "..", "data", "grids_c2b", "grids.json")
        with open(os.path.normpath(cand)) as f:
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
