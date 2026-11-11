"""crystal-q1-v1 train/valid -> Mat2Spec inputs. This module never opens the test split.

Upstream (Mat2Spec_Codes/Mat2Spec/data.py, CIF_Dataset.__getitem__) builds, per crystal:
  * atom features: CGCNN ``atom_init.json`` rows (92 values) looked up by atomic number;
  * neighbours: ``Structure.get_all_neighbors(radius=8)``, sorted by distance, first 12 kept;
    crystals with fewer than 12 neighbours are padded with index 0 and distance radius+1;
  * edge features: Gaussian expansion of the distance on ``arange(0, 8.2, 0.2)`` (41 values),
    ``exp(-(d-mu)^2/0.2^2)``;
  * a 103-long element-fraction vector (``global_feature``);
  * spectral-clustering groups (``cluster``) that the model never uses -> not built here.
The same quantities are produced here from the packed arrays, then batched on the GPU.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ALLOWED_SPLITS = ("train", "valid")
TASK_BINS = {"edos": 128, "phdos": 64}
RADIUS, MAX_NUM_NBR, GAUSS_STEP = 8.0, 12, 0.2
N_ELEMENTS = 103  # upstream ELEM_Encoder: H..Lr in atomic-number order
GRAPH_SCHEMA = "mat2spec-graphs-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(data_root, split):
    """Read one frozen split, verify every array against the manifest, apply the train view."""
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {ALLOWED_SPLITS}; the test split is never read here")
    root = Path(data_root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "crystal-q1-v1":
        raise ValueError("crystal-q1-v1 data required")

    def array(name):
        relative = f"{split}/{name}"
        if file_sha256(root / relative) != manifest["files"][relative]["sha256"]:
            raise ValueError(f"frozen data payload changed: {relative}")
        return np.load(root / relative, allow_pickle=False)

    ids = array(f"{split}_index.npy")
    counts = array(f"n_atoms_{split}.npy").astype(np.int64)
    offsets = array(f"atom_offsets_{split}.npy")
    numbers = array(f"atomic_numbers_{split}.npy").astype(np.int64)
    cell = array(f"cell_{split}.npy")
    frac = array(f"frac_coords_{split}.npy")
    spectra = {task: array(f"{task}_tgtdos_{split}.npy") for task in TASK_BINS}
    if not np.array_equal(np.diff(offsets), counts) or offsets[-1] != len(numbers):
        raise ValueError("atom offsets disagree with atom counts")
    keep = np.ones(len(ids), bool)
    if split == "train":
        keep = ~np.isin(ids, manifest.get("training_exclusions", []))
    if int(keep.sum()) != manifest["splits"][split]["training_n"]:
        raise ValueError("effective view differs from the manifest")
    rows = np.flatnonzero(keep)
    atom_rows = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in rows])
    kept_counts = counts[rows]
    for task, bins in TASK_BINS.items():
        if spectra[task].shape != (len(ids), bins):
            raise ValueError(f"{task}: unexpected grid")
    return {
        "split": split, "material_ids": ids[rows], "n_atoms": kept_counts,
        "offsets": np.concatenate([[0], np.cumsum(kept_counts)]),
        "atomic_numbers": numbers[atom_rows], "cell": cell[rows], "frac_coords": frac[atom_rows],
        "edos": spectra["edos"][rows], "phdos": spectra["phdos"][rows],
        "manifest_sha256": file_sha256(root / "manifest.json"),
    }


def crystal_neighbors(cell, frac, numbers, radius=RADIUS, max_num_nbr=MAX_NUM_NBR):
    """First ``max_num_nbr`` periodic neighbours by distance; upstream padding when fewer exist."""
    from pymatgen.core import Lattice, Structure

    structure = Structure(Lattice(np.asarray(cell, float)), list(map(int, numbers)),
                          np.asarray(frac, float), coords_are_cartesian=False)
    centers, points, _, distances = structure.get_neighbor_list(radius)
    order = np.lexsort((np.arange(len(centers)), distances, centers))  # stable within a centre
    centers, points, distances = centers[order], points[order], distances[order]
    n = len(numbers)
    index = np.zeros((n, max_num_nbr), np.int64)          # upstream pads with atom 0 ...
    dist = np.full((n, max_num_nbr), radius + 1.0)         # ... at distance radius + 1
    first = np.searchsorted(centers, np.arange(n))
    rank = np.arange(len(centers)) - first[centers]
    sel = rank < max_num_nbr
    index[centers[sel], rank[sel]] = points[sel]
    dist[centers[sel], rank[sel]] = distances[sel]
    return index, dist


def literal_upstream_neighbors(cell, frac, numbers, radius=RADIUS, max_num_nbr=MAX_NUM_NBR):
    """The upstream statements, kept verbatim for the equivalence test (slow)."""
    from pymatgen.core import Lattice, Structure

    crystal = Structure(Lattice(np.asarray(cell, float)), list(map(int, numbers)),
                        np.asarray(frac, float), coords_are_cartesian=False)
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[1], nbr)) + [radius + 1.] * (max_num_nbr - len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1], nbr[:max_num_nbr])))
    return np.array(nbr_fea_idx), np.array(nbr_fea)


def build_graphs(data, cache_path=None, progress=None):
    """Neighbour index/distance tables for one split, cached with a content fingerprint."""
    from importlib.metadata import version

    key = hashlib.sha256()
    for name in ("cell", "frac_coords", "atomic_numbers", "n_atoms"):
        key.update(np.ascontiguousarray(data[name]).tobytes())
    fingerprint = {"schema": GRAPH_SCHEMA, "geometry_sha256": key.hexdigest(), "radius": RADIUS,
                   "max_num_nbr": MAX_NUM_NBR, "manifest_sha256": data["manifest_sha256"],
                   "split": data["split"]}
    if cache_path is not None and Path(cache_path).is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            stored = json.loads(str(archive["fingerprint"]))
            stored.pop("pymatgen", None)
            if stored != fingerprint:
                raise ValueError(f"graph cache does not match the data: {cache_path}")
            return archive["nbr_index"], archive["nbr_dist"]
    offsets = data["offsets"]
    index = np.empty((offsets[-1], MAX_NUM_NBR), np.int64)
    dist = np.empty((offsets[-1], MAX_NUM_NBR), np.float64)
    for i in range(len(data["n_atoms"])):
        a, b = offsets[i], offsets[i + 1]
        index[a:b], dist[a:b] = crystal_neighbors(
            data["cell"][i], data["frac_coords"][a:b], data["atomic_numbers"][a:b])
        if progress and (i + 1) % progress == 0:
            print(f"  graphs {data['split']}: {i + 1}/{len(data['n_atoms'])}", flush=True)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(cache_path) + ".tmp.npz")
        np.savez(temporary, nbr_index=index, nbr_dist=dist,
                 fingerprint=json.dumps({**fingerprint, "pymatgen": version("pymatgen")}))
        temporary.replace(cache_path)
    return index, dist


def load_atom_init(path):
    """CGCNN element table -> [101, 92] float tensor indexed by atomic number (row 0 unused)."""
    table = json.loads(Path(path).read_text())
    width = len(next(iter(table.values())))
    out = torch.zeros(max(map(int, table)) + 1, width)
    for key, value in table.items():
        out[int(key)] = torch.tensor(value, dtype=torch.float32)
    return out


class PackedGraphs:
    """All crystals of one split on one device; ``batch(rows)`` returns what Mat2Spec.forward reads."""

    def __init__(self, data, nbr_index, nbr_dist, atom_init, device):
        numbers = torch.as_tensor(data["atomic_numbers"])
        if int(numbers.max()) >= len(atom_init) or int(numbers.max()) > N_ELEMENTS:
            raise ValueError("an element is missing from the CGCNN/Mat2Spec element tables")
        self.device = device
        self.n = len(data["n_atoms"])
        self.material_ids = data["material_ids"]
        self.counts = torch.as_tensor(data["n_atoms"], device=device)
        self.starts = torch.as_tensor(data["offsets"][:-1], device=device)
        self.numbers = numbers.to(device)
        self.atom_init = atom_init.to(device)
        self.nbr_index = torch.as_tensor(nbr_index, device=device)
        self.nbr_dist = torch.as_tensor(nbr_dist, device=device)            # float64, as upstream
        self.centres = torch.arange(0, RADIUS + GAUSS_STEP, GAUSS_STEP, dtype=torch.float64,
                                    device=device)                           # 41 centres
        if len(self.centres) != 41:
            raise ValueError("Gaussian filter must have 41 centres")
        owner = torch.repeat_interleave(torch.arange(self.n), torch.as_tensor(data["n_atoms"]))
        composition = torch.zeros(self.n, N_ELEMENTS)
        composition.index_put_((owner, numbers - 1), torch.ones(len(numbers)), accumulate=True)
        self.composition = (composition / composition.sum(1, keepdim=True)).to(device)

    def __len__(self):
        return self.n

    def batch(self, rows, y):
        """rows: 1-D long tensor of crystal rows; y: [len(rows), bins] label fed to the label branch."""
        rows = rows.to(self.device)
        counts = self.counts[rows]
        graph = torch.repeat_interleave(torch.arange(len(rows), device=self.device), counts)
        first = torch.cumsum(counts, 0) - counts                      # batch offset of each crystal
        local = torch.arange(int(counts.sum()), device=self.device) - first[graph]
        atoms = self.starts[rows][graph] + local                      # rows in the packed atom table
        neighbours = self.nbr_index[atoms] + first[graph][:, None]    # batch-level neighbour index
        source = torch.arange(len(atoms), device=self.device)[:, None].expand_as(neighbours)
        distance = self.nbr_dist[atoms].reshape(-1, 1)
        edge_attr = torch.exp(-(distance - self.centres) ** 2 / GAUSS_STEP ** 2).float()
        return SimpleNamespace(
            x=self.atom_init[self.numbers[atoms]],
            edge_index=torch.stack([source.reshape(-1), neighbours.reshape(-1)]),
            edge_attr=edge_attr, batch=graph, global_feature=self.composition[rows],
            y=y, num_graphs=len(rows))
