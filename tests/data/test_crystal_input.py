"""Complete-atom storage, native matrix geometry and training-batch contracts."""
from copy import deepcopy
import json
import logging
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from model.model import basemodel
from model.transformer import Transformer
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.bucket_batch import crystal_collate, split_crystal_batch, automatic_pair_budget
from utils.crystal import CrystalGeometry, atom_padding_mask, structure_to_crystal_inputs
from utils.crystal_cache import PackedCrystalStore, crystal_data_contract
from utils.dataset import Dos_Dataset
from utils.paths import CRYSTAL_Q1_DATA_ROOT, Q1_DATA_ROOT
from utils.periodic_geometry import build_periodic_records
from utils.relative_features import build_cell_from_lattice


def sample(n, seed=0):
    generator = torch.Generator().manual_seed(seed)
    atoms = torch.randint(1, 20, (n,), generator=generator)
    geometry = CrystalGeometry(torch.eye(3) * 20,
                               torch.rand(n, 3, generator=generator))
    e = torch.rand(8, generator=generator); e /= e.sum()
    p = torch.rand(4, generator=generator); p /= p.sum()
    one, zero = torch.ones(1), torch.zeros(1)
    return [atoms, geometry, e, p, one, one, zero, one * 4,
            one, one, zero, one * 3 * n * 0.2 / 19.6875,
            torch.ones(8, dtype=torch.bool), torch.ones(4, dtype=torch.bool),
            torch.tensor(float(n * 4))]


def tiny_model():
    return Transformer(token_num=128, d_model=16, nhead=4, edos_num=8, phdos_num=4,
        num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32,
        dropout=0.0, activation="gelu", head_type="legacy", atom_feat_mode="z_only_proj",
        scale_mode="eta")


def training_model():
    model = basemodel(logging.getLogger("crystal-test"), loss_form="sumnorm_klw",
        metrics_list=[], lambda_ph=1.0, grad_clip=0.0,
        sub_model={"transformer": dict(token_num=128, d_model=16, nhead=4,
            edos_num=8, phdos_num=4, num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=32, dropout=0.0, activation="gelu", head_type="legacy",
            atom_feat_mode="z_only_proj", scale_mode="eta")},
        optimizer={"transformer": {"type": "AdamW", "params": {"lr": 0.0}}},
        lr_scheduler={})
    model.to(torch.device("cpu"))
    return model


class TestCrystalInput(unittest.TestCase):
    def test_native_geometry_matches_legacy_and_preserves_reflection(self):
        pos = torch.tensor([[[4., 5., 1/6], [80., 85., 75.],
                             [0., 0., 0.], [0.3, 0.2, 0.1]]], dtype=torch.float64)
        cell, frac = build_cell_from_lattice(pos)
        mask = torch.zeros(1, 2, dtype=torch.bool)
        legacy = build_periodic_records(pos, mask, 6.0)
        native = build_periodic_records(CrystalGeometry(cell, frac), mask, 6.0)
        for name in ("batch", "dst", "src", "shifts", "vectors", "distances"):
            torch.testing.assert_close(getattr(native, name), getattr(legacy, name), rtol=0, atol=0)
        q = torch.diag(torch.tensor([-1., 1., 1.], dtype=torch.float64))
        reflected = build_periodic_records(CrystalGeometry(cell @ q, frac), mask, 6.0)
        self.assertLess(float(torch.det(cell @ q)), 0)
        torch.testing.assert_close(reflected.shifts, native.shifts)
        torch.testing.assert_close(reflected.vectors, native.vectors @ q)
        torch.testing.assert_close(reflected.distances, native.distances)

    def test_variable_padding_and_split_keep_all_192_atoms(self):
        batch = crystal_collate([sample(3), sample(192, 1), sample(7, 2)])
        self.assertEqual(batch[0].shape, (3, 192))
        self.assertEqual((~atom_padding_mask(batch[1], batch[0].eq(0))).sum(-1).tolist(), [3, 192, 7])
        chunks = split_crystal_batch(batch, automatic_pair_budget(batch))
        self.assertEqual([int(row.ne(0).sum()) for c in chunks for row in c[0]], [3, 192, 7])
        for chunk in chunks:
            self.assertLessEqual(len(chunk[0]) * chunk[0].shape[1] ** 2, automatic_pair_budget(batch))
        singleton = crystal_collate([sample(192)])
        self.assertEqual(len(split_crystal_batch(singleton, automatic_pair_budget(singleton))), 1)
        with self.assertRaisesRegex(ValueError, "exceeds pair budget"):
            split_crystal_batch(singleton, 100)

    def test_model_ignores_nan_in_masked_atom_coordinates(self):
        model = tiny_model().eval()
        batch = crystal_collate([sample(2), sample(12, 1)])
        mask = batch[0].eq(0)
        poisoned = batch[1].frac_coords.clone()
        poisoned[mask] = float("nan")
        with torch.no_grad():
            expected = model(batch[0], mask, batch[1])
            actual = model(batch[0], mask, CrystalGeometry(batch[1].cell, poisoned))
        for name in ("edos", "phdos", "eta"):
            torch.testing.assert_close(actual[name], expected[name])

    def test_forward_backward_uses_atoms_after_80_without_new_parameters(self):
        model = tiny_model()
        params = sum(p.numel() for p in model.parameters())
        batch = crystal_collate([sample(192)])
        geometry = CrystalGeometry(batch[1].cell.requires_grad_(),
                                   batch[1].frac_coords.requires_grad_())
        seen = []
        def capture(m, inp, out):
            out.retain_grad()
            seen.append(out)
        handle = model.atom_proj.register_forward_hook(capture)
        outputs = model(batch[0], batch[0].eq(0), geometry)
        (outputs["edos"].square().mean() + outputs["phdos"].square().mean()
         + outputs["eta"].square().mean()).backward()
        handle.remove()
        self.assertEqual([v.shape[1] for v in seen], [192])
        self.assertEqual(params, sum(p.numel() for p in model.parameters()))
        self.assertTrue(torch.isfinite(geometry.frac_coords.grad).all())
        self.assertGreater(float(seen[0].grad[:, 80:].abs().sum()), 0)

    def test_accumulated_gradient_matches_full_batch_sample_weighting(self):
        torch.manual_seed(123)
        full = training_model()
        split = training_model()
        split.model["transformer"].load_state_dict(deepcopy(full.model["transformer"].state_dict()))
        batch = crystal_collate([sample(83, 1), sample(96, 2), sample(128, 3)])
        before = deepcopy(split.optimizer["transformer"].state_dict())
        loss_full = full.train_one_step(batch, 0, pair_budget=3 * 128 ** 2)
        loss_split = split.train_one_step(batch, 0, pair_budget=128 ** 2)
        self.assertEqual(split.last_microbatch_sizes, [1, 1, 1])
        self.assertAlmostEqual(loss_full["loss"], loss_split["loss"], places=5)
        for (name, p), (other, q) in zip(full.model["transformer"].named_parameters(),
                                        split.model["transformer"].named_parameters()):
            self.assertEqual(name, other)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=3e-4)
        self.assertFalse(before["state"])
        self.assertTrue(all(int(v["step"]) == 1 for v in split.optimizer["transformer"].state.values()))

    def test_evaluation_split_preserves_sample_metrics(self):
        model = training_model()
        model.model["transformer"].eval()
        batch = crystal_collate([sample(83, 1), sample(96, 2), sample(128, 3)])
        with torch.no_grad():
            full = model.test_one_step(batch, pair_budget=3 * 128 ** 2)
            split = model.test_one_step(batch, pair_budget=128 ** 2)
        for name in full:
            self.assertAlmostEqual(full[name], split[name], places=5)

    def test_cif_converter_preserves_full_explicit_frame(self):
        frac = np.random.default_rng(2).random((192, 3))
        structure = Structure(Lattice([[-30, 0, 0], [0, 30, 0], [0, 0, 30]]), ["Si"] * 192, frac)
        atoms, geometry = structure_to_crystal_inputs(structure, primitive=False)
        self.assertEqual(len(atoms), 192)
        self.assertLess(float(torch.det(geometry.cell)), 0)
        np.testing.assert_array_equal(geometry.frac_coords.numpy(), frac.astype(np.float32))

    def test_cif_default_uses_standard_primitive_and_cache_frame(self):
        structure = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.4), ["Si"], [[0, 0, 0]])
        atoms, geometry = structure_to_crystal_inputs(structure)
        self.assertEqual(len(atoms), 1)
        self.assertAlmostEqual(abs(float(torch.det(geometry.cell))), structure.volume / 4, places=4)
        torch.testing.assert_close(geometry.cell.triu(1), torch.zeros(3, 3))

    def test_checkpoint_refuses_cross_data_revision_before_loading_weights(self):
        model = tiny_model()
        optimizer = torch.optim.AdamW(model.parameters())
        contract = {"schema": "crystal-q1-v1", "manifest_sha256": "version-a"}
        checkpoint = build_ablation_checkpoint(1, "M1", 42, False, model, optimizer,
                                              0.0, data_contract=contract)
        restore_ablation_checkpoint(checkpoint, model, optimizer, False, data_contract=contract)
        with self.assertRaisesRegex(ValueError, "data contract"):
            restore_ablation_checkpoint(checkpoint, model, optimizer, False)
        with self.assertRaisesRegex(ValueError, "data contract"):
            restore_ablation_checkpoint(checkpoint, model, optimizer, False,
                data_contract={"schema": "crystal-q1-v1", "manifest_sha256": "version-b"})

    def test_h1_scale_supervision_counts_complete_atoms(self):
        class FixedHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.eta = torch.nn.Parameter(torch.tensor([0.4, 0.6]))
            def forward(self, src, mask, geometry, *args):
                return {"edos": self.eta.sum() * torch.zeros(len(src), 8),
                        "phdos": self.eta.sum() * torch.zeros(len(src), 4),
                        "eta": self.eta[None].expand(len(src), -1)}
        model = training_model()
        model.model["transformer"] = FixedHead()
        model.optimizer["transformer"] = torch.optim.SGD(model.model["transformer"].parameters(), lr=0)
        batch = crystal_collate([sample(83), sample(192)])
        result = model.train_one_step(batch, 0)
        target = torch.tensor([[0.2, 0.09375 / 83], [0.2, 0.09375 / 192]])
        expected = (torch.tensor([0.4, 0.6]) - target).square().mean()
        self.assertAlmostEqual(result["loss_eta"], float(expected), places=6)


@unittest.skipUnless((CRYSTAL_Q1_DATA_ROOT / "manifest.json").is_file(), "complete Q1 cache unavailable")
class TestCompleteQ1Cache(unittest.TestCase):
    def test_all_repaired_structures_have_native_full_atom_records(self):
        manifest = json.loads((CRYSTAL_Q1_DATA_ROOT / "manifest.json").read_text())
        stores = {split: PackedCrystalStore(CRYSTAL_Q1_DATA_ROOT, split)
                  for split in ("train", "valid", "test")}
        indices = {split: {str(material_id): i for i, material_id in enumerate(store.material_ids)}
                   for split, store in stores.items()}
        for repair in manifest["repairs"]:
            store = stores[repair["split"]]
            atoms, geometry = store[indices[repair["split"]][repair["material_id"]]]
            self.assertEqual(len(atoms), repair["n_atoms"])
            mask = torch.zeros(1, len(atoms), dtype=torch.bool)
            records = build_periodic_records(CrystalGeometry(geometry.cell[None],
                                             geometry.frac_coords[None]), mask, 6.0)
            self.assertEqual(set(records.src.tolist()), set(range(len(atoms))))
            self.assertEqual(set(records.dst.tolist()), set(range(len(atoms))))
            self.assertTrue((records.distances < 6.0).all())

    def test_full_storage_dtype_order_labels_and_repairs(self):
        manifest = json.loads((CRYSTAL_Q1_DATA_ROOT / "manifest.json").read_text())
        self.assertEqual(len(manifest["repairs"]), 55)
        self.assertEqual(sum("valence" in r for r in manifest["repairs"]), 54)
        self.assertEqual(len(manifest["training_exclusions"]), 2)
        total_atoms = 0
        for split in ("train", "valid", "test"):
            store = PackedCrystalStore(CRYSTAL_Q1_DATA_ROOT, split)
            total_atoms += int(store.n_atoms.sum())
            np.testing.assert_array_equal(store.material_ids,
                np.load(Q1_DATA_ROOT / split / f"{split}_index.npy", allow_pickle=False))
            for name in ("edos_tgtdos", "phdos_tgtdos", "edos_mask", "phdos_mask"):
                a = np.load(CRYSTAL_Q1_DATA_ROOT / split / f"{name}_{split}.npy", allow_pickle=False)
                b = np.load(Q1_DATA_ROOT / split / f"{name}_{split}.npy", allow_pickle=False)
                np.testing.assert_array_equal(a, b)
                self.assertEqual(a.dtype, np.bool_ if "mask" in name else np.float32)
        self.assertEqual(total_atoms, 275049)
        crystal_data_contract(CRYSTAL_Q1_DATA_ROOT, normalization="sumnorm")

    def test_training_quality_filter_and_subset_metadata_alignment(self):
        dataset = Dos_Dataset(data_dir=str(CRYSTAL_Q1_DATA_ROOT), split="train", dos_sumnorm=True)
        self.assertEqual(len(dataset), 18704)
        self.assertNotIn("mp-aaabilxy", dataset.material_ids)
        self.assertNotIn("mp-aaabidtd", dataset.material_ids)
        subset = Dos_Dataset(data_dir=str(CRYSTAL_Q1_DATA_ROOT), split="train", dos_sumnorm=True,
                             choice=[7, 0])
        np.testing.assert_array_equal(subset.material_ids, dataset.material_ids[[7, 0]])
        for i, row in enumerate([7, 0]):
            for field in range(2, 15):
                torch.testing.assert_close(subset[i][field], dataset[row][field])

    def test_invalid_offsets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "manifest.json").write_bytes((CRYSTAL_Q1_DATA_ROOT / "manifest.json").read_bytes())
            (directory / "valid").mkdir()
            for name in ("cell", "frac_coords", "atomic_numbers", "atom_offsets", "n_atoms"):
                source = CRYSTAL_Q1_DATA_ROOT / "valid" / f"{name}_valid.npy"
                (directory / "valid" / source.name).write_bytes(source.read_bytes())
            (directory / "valid/valid_index.npy").write_bytes((CRYSTAL_Q1_DATA_ROOT / "valid/valid_index.npy").read_bytes())
            offsets = np.load(directory / "valid/atom_offsets_valid.npy")
            offsets[1] += 1
            np.save(directory / "valid/atom_offsets_valid.npy", offsets)
            with self.assertRaisesRegex(ValueError, "offsets disagree"):
                PackedCrystalStore(directory, "valid")


if __name__ == "__main__":
    unittest.main()
