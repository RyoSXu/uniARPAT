"""B7 CIF blind-inference contracts; CPU-only and independent of Q1/checkpoints."""

import tempfile
import unittest
from pathlib import Path
import json

import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from b7_cif_infer import process_cif

from utils.b7_cif_inference import (
    DELTA_EDOS,
    DELTA_PHDOS,
    checkpoint_state,
    load_b7_grids,
    predict_b7_blind,
    structure_to_b7_inputs,
)


def _silicon():
    return Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


class _FixedB7(torch.nn.Module):
    def forward(self, inp, mask, pos):
        assert inp.shape == (1, 82) and pos.shape == (1, 82, 3)
        return {
            "edos": torch.zeros(1, 128, device=inp.device),
            "phdos": torch.zeros(1, 64, device=inp.device),
            "eta": torch.tensor([[0.5, 0.25]], device=inp.device),
        }


class TestB7CifInference(unittest.TestCase):
    def test_structure_contract_valence_and_grids(self):
        src, pos, metadata = structure_to_b7_inputs(_silicon())
        self.assertEqual(tuple(src.shape), (82,))
        self.assertEqual(tuple(pos.shape), (82, 3))
        self.assertEqual(src[:4].tolist(), [126, 127, 14, 14])
        self.assertAlmostEqual(float(pos[0, 2]), 1.0 / 5.43, places=6)
        self.assertEqual(metadata["n_atoms"], 2)
        self.assertEqual(metadata["n_valence"], 8.0)
        edos_x, phdos_x = load_b7_grids()
        self.assertEqual(edos_x.shape, (128,))
        self.assertEqual(phdos_x.shape, (64,))
        self.assertAlmostEqual(edos_x[0], -6.0 + DELTA_EDOS / 2.0)
        self.assertAlmostEqual(phdos_x[0], -280.0 + DELTA_PHDOS / 2.0)

    def test_h1_blind_formula(self):
        src, pos, metadata = structure_to_b7_inputs(_silicon())
        result = predict_b7_blind(_FixedB7(), src, pos, metadata["n_valence"], torch.device("cpu"))
        self.assertTrue(np.allclose(result["edos"], (8.0 * 0.25 / DELTA_EDOS) / 128.0))
        self.assertTrue(np.allclose(result["phdos"], (3.0 * 2.0 * 0.5 / DELTA_PHDOS) / 64.0))
        self.assertAlmostEqual(float(result["edos"].sum()), result["edos_sum"], places=5)
        self.assertAlmostEqual(float(result["phdos"].sum()), result["phdos_sum"], places=6)

    def test_invalid_structure_and_checkpoint_are_rejected(self):
        disordered = Structure(Lattice.cubic(4.0), [{"Si": 0.5, "Ge": 0.5}], [[0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "ordered"):
            structure_to_b7_inputs(disordered)
        too_long = Structure(Lattice.cubic(30.0), ["Si"] * 81, np.zeros((81, 3)))
        with self.assertRaisesRegex(ValueError, "at most 80"):
            structure_to_b7_inputs(too_long)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not_b7.pth"
            torch.save({"model_name": "M1", "epoch": 32, "seed": 42, "model": {}}, path)
            with self.assertRaisesRegex(ValueError, "not frozen B7"):
                checkpoint_state(path)

    def test_process_cif_writes_auditable_blind_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cif_path = directory / "silicon.cif"
            CifWriter(_silicon()).write_file(cif_path)
            result = process_cif(
                cif_path, _FixedB7(), torch.device("cpu"),
                {"model_name": "M1", "epoch": 33, "seed": 42}, directory / "out",
            )
            with np.load(result["spectra_npz"]) as arrays:
                self.assertEqual(arrays["edos"].shape, (128,))
                self.assertEqual(arrays["phdos"].shape, (64,))
            metadata = json.loads(Path(result["metadata_json"]).read_text())
            self.assertEqual(metadata["checkpoint"]["epoch"], 33)
            self.assertIn("no label-derived scale", metadata["model_contract"])


if __name__ == "__main__":
    unittest.main()
