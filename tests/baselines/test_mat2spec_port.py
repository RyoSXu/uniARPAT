"""Mat2Spec baseline: port fidelity, data adapter, train-view/test-split boundaries and scoring."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "baselines/mat2spec"
DATA = ROOT / "data/crystal_q1_v1"
REFERENCE = ROOT / "results/runs/phdos_head_20261106/analysis/exports/h2_s42"
HAVE_DATA = (DATA / "valid/valid_index.npy").is_file()
HAVE_UPSTREAM = (BASE / "upstream/Mat2Spec_Codes/Mat2Spec/Mat2Spec.py").is_file()

from baselines.mat2spec.adapter import data as D  # noqa: E402
from baselines.mat2spec.adapter import project_metrics as PM  # noqa: E402
from baselines.mat2spec.adapter import train as T  # noqa: E402


def toy_split(seed=0, crystals=6):
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 5, crystals)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    return {"split": "train", "material_ids": np.array([f"toy-{i}" for i in range(crystals)]),
            "n_atoms": counts, "offsets": offsets,
            "atomic_numbers": rng.integers(1, 90, offsets[-1]),
            "cell": np.stack([np.eye(3) * rng.uniform(3, 6) for _ in range(crystals)]).astype(np.float32),
            "frac_coords": rng.random((offsets[-1], 3)).astype(np.float32),
            "edos": rng.random((crystals, 128)).astype(np.float32),
            "phdos": rng.random((crystals, 64)).astype(np.float32), "manifest_sha256": "toy"}


def toy_graphs(data):
    warnings.simplefilter("ignore")
    index, dist = D.build_graphs(data)
    table = torch.randn(101, 92, generator=torch.Generator().manual_seed(0))
    return D.PackedGraphs(data, index, dist, table, torch.device("cpu")), index, dist, table


class PortRecord(unittest.TestCase):
    @unittest.skipUnless(HAVE_UPSTREAM, "upstream clone not present")
    def test_recorded_diff_is_the_whole_difference(self):
        result = subprocess.run(["diff", "-u", "upstream/Mat2Spec_Codes/Mat2Spec/Mat2Spec.py",
                                 "port/mat2spec_model.py"], cwd=BASE, capture_output=True, text=True)
        body = lambda text: [line for line in text.splitlines() if not line.startswith(("---", "+++"))]  # noqa: E731
        self.assertEqual(body(result.stdout), body((BASE / "port/mat2spec_model.diff").read_text()))
        changed = [line for line in body(result.stdout) if line.startswith("-")]
        self.assertEqual(len(changed), 9)   # 7 documented edits; edit 2 removes three import lines
        self.assertEqual((BASE / "upstream/Mat2Spec_Codes/Mat2Spec/pytorch_stats_loss.py").read_bytes(),
                         (BASE / "port/pytorch_stats_loss.py").read_bytes())

    @unittest.skipUnless(HAVE_UPSTREAM, "upstream clone not present")
    def test_upstream_files_match_recorded_hashes(self):
        for line in (BASE / "UPSTREAM_SHA256.txt").read_text().splitlines():
            digest, name = line.split("  ", 1)
            self.assertEqual(D.file_sha256(BASE / "upstream" / name), digest, name)


class PortedLayer(unittest.TestCase):
    def test_gat_layer_matches_explicit_pyg15_semantics(self):
        """flow=target_to_source: centre i = edge_index[0]; softmax and sum are grouped by centre."""
        from baselines.mat2spec.port.mat2spec_model import GAT_Crystal

        torch.manual_seed(0)
        nodes, heads, width = 7, 4, 16
        layer = GAT_Crystal(width, width, width, heads).eval()
        x, edge_attr = torch.randn(nodes, width), torch.randn(nodes * 3, width)
        centre = torch.arange(nodes).repeat_interleave(3)
        neighbour = torch.randint(0, nodes, (nodes * 3,))
        out = layer(x, torch.stack([centre, neighbour]), edge_attr)
        softplus = torch.nn.functional.softplus
        xi = softplus(torch.cat([x[centre], edge_attr], -1) @ layer.W).view(-1, heads, width)
        xj = softplus(torch.cat([x[neighbour], edge_attr], -1) @ layer.W).view(-1, heads, width)
        alpha = softplus(layer.bn1(softplus((torch.cat([xi, xj], -1) * layer.att).sum(-1))))
        expected = torch.zeros(nodes, heads, width)
        for node in range(nodes):
            rows = centre == node
            expected[node] = (xj[rows] * torch.softmax(alpha[rows], 0)[..., None]).sum(0)
        self.assertTrue(torch.allclose(out, expected.mean(1) + layer.bias, atol=1e-5))

    def test_feature_branch_prediction_ignores_labels(self):
        from baselines.mat2spec.port import mat2spec_model as M2S

        data = toy_split()
        graphs, *_ = toy_graphs(data)
        args = SimpleNamespace(task="phdos", label_scaling="normalized_sum", loss="KL", eval_batch_size=4)
        margs = T.model_args(args)
        torch.manual_seed(0)
        net = M2S.Mat2Spec(margs, None).to(M2S.device)
        graphs = D.PackedGraphs(data, *D.build_graphs(data), torch.randn(101, 92), M2S.device)
        labels = torch.rand(len(graphs), 64, device=M2S.device)
        first = T.predict(net, graphs, labels, args, margs, None, None)
        second = T.predict(net, graphs, torch.zeros_like(labels), args, margs, None, None)
        self.assertTrue(torch.allclose(first, second, rtol=0, atol=1e-6))   # GPU scatter order noise only
        other = T.predict(net, graphs, labels.flip(0) * 50, args, margs, None, None)
        self.assertTrue(torch.allclose(first, other, rtol=0, atol=1e-6))
        self.assertTrue(torch.allclose(first.sum(1), torch.ones(len(graphs), device=first.device), atol=1e-5))


class Adapter(unittest.TestCase):
    def test_test_split_is_refused(self):
        with self.assertRaises(ValueError):
            D.load_split(DATA, "test")
        for path in (BASE / "adapter").glob("*.py"):
            text = path.read_text()
            self.assertNotIn("_test.npy", text, path.name)
            self.assertNotIn('"test"', text.replace('{ALLOWED_SPLITS}; the test split', ''), path.name)

    def test_batch_equals_per_crystal_assembly(self):
        data = toy_split()
        graphs, index, dist, table = toy_graphs(data)
        rows = torch.tensor([4, 1, 5])
        batch = graphs.batch(rows, torch.zeros(3, 64))
        centres = np.arange(0, D.RADIUS + D.GAUSS_STEP, D.GAUSS_STEP)
        x, edges, attr, owner, start = [], [], [], [], 0
        for position, row in enumerate(rows.tolist()):
            a, b = data["offsets"][row], data["offsets"][row + 1]
            x.append(table[torch.as_tensor(data["atomic_numbers"][a:b])])
            source = np.repeat(np.arange(b - a), D.MAX_NUM_NBR) + start     # upstream format_adj_matrix
            edges.append(np.stack([source, index[a:b].reshape(-1) + start]))
            attr.append(np.exp(-(dist[a:b][..., None] - centres) ** 2 / D.GAUSS_STEP ** 2).reshape(-1, 41))
            owner += [position] * (b - a)
            start += b - a
        self.assertTrue(torch.equal(batch.x, torch.cat(x)))
        self.assertTrue(np.array_equal(batch.edge_index.numpy(), np.concatenate(edges, 1)))
        self.assertTrue(np.allclose(batch.edge_attr.numpy(), np.concatenate(attr), atol=1e-7))
        self.assertEqual(batch.batch.tolist(), owner)
        composition = batch.global_feature[0].numpy()
        a, b = data["offsets"][4], data["offsets"][5]
        expected = np.bincount(data["atomic_numbers"][a:b] - 1, minlength=103) / (b - a)
        self.assertTrue(np.allclose(composition, expected))

    def test_label_scaling_is_finite_for_zero_spectra_and_train_fitted(self):
        labels = torch.tensor([[0., 0., 0.], [1., 3., 0.]])
        for scaling in ("normalized_sum", "normalized_max"):
            self.assertTrue(torch.isfinite(T.scale_labels(labels, scaling)).all())
        self.assertTrue(torch.allclose(T.scale_labels(labels, "normalized_sum")[1], torch.tensor([.25, .75, 0.])))
        mean, std, constant = T.fit_standardizer(labels)
        self.assertEqual(constant, 1)
        self.assertTrue(torch.isfinite(T.scale_labels(labels, "standardized", mean, std)).all())
        self.assertTrue(torch.allclose(std[:2], labels[:, :2].std(0)))     # unbiased, as upstream
        per_atom = T.label_units(np.array([[2., 4.]], np.float32), np.array([2]), "per_atom")
        self.assertTrue(torch.equal(per_atom, torch.tensor([[1., 2.]])))

    @unittest.skipUnless(HAVE_DATA, "crystal-q1-v1 arrays not present")
    def test_frozen_views_and_literal_upstream_neighbours(self):
        warnings.simplefilter("ignore")
        train, valid = D.load_split(DATA, "train"), D.load_split(DATA, "valid")
        self.assertEqual((len(train["n_atoms"]), len(valid["n_atoms"])), (18704, 2313))
        manifest = json.loads((DATA / "manifest.json").read_text())
        self.assertFalse(set(train["material_ids"]) & set(manifest["training_exclusions"]))
        self.assertEqual(int(train["n_atoms"].sum()), 220413)
        def geometry(row):
            a, b = valid["offsets"][row], valid["offsets"][row + 1]
            return valid["cell"][row], valid["frac_coords"][a:b], valid["atomic_numbers"][a:b]

        # Sparse crystals with fewer than 12 neighbours inside 8 A exercise upstream's padding branch.
        padded = [row for row in range(len(valid["n_atoms"]))
                  if (D.crystal_neighbors(*geometry(row))[1] > D.RADIUS).any()]
        self.assertGreaterEqual(len(padded), 1)
        rng = np.random.default_rng(1)
        for row in list(rng.choice(len(valid["n_atoms"]), 25, replace=False)) + padded:
            fast, literal = D.crystal_neighbors(*geometry(row)), D.literal_upstream_neighbors(*geometry(row))
            self.assertTrue(np.array_equal(fast[0], literal[0]))
            self.assertTrue(np.array_equal(fast[1], literal[1]))


class Scoring(unittest.TestCase):
    def test_zero_targets_and_oracle(self):
        target = np.array([[0., 0., 0., 0.], [1., 3., 0., 0.]], np.float32)
        shape = np.array([[.25, .25, .25, .25], [.5, .5, 0., 0.]], np.float32)
        oracle = PM.oracle_reconstruction(shape, target)
        self.assertTrue(np.array_equal(oracle, np.array([[0, 0, 0, 0], [2, 2, 0, 0]], np.float32)))
        scores = PM.shape_scores(shape, target)
        self.assertTrue(np.isnan(scores["tv"][0]))
        self.assertAlmostEqual(float(scores["tv"][1]), .25)
        self.assertAlmostEqual(float(scores["w1"][1]), .25 / 4)

    def test_energy_matches_definition(self):
        normalizer = {"energy": 2.0, "bin_width": .5, "bins": 4}
        prediction = np.array([[1., 2., 0., 1.]], np.float32)
        target = np.array([[1., 0., 0., 3.]], np.float32)
        energy = PM.material_scores(prediction, target, np.array([2]), normalizer)["energy"][0]
        self.assertAlmostEqual(float(energy), .5 * ((2 / 2) ** 2 + (2 / 2) ** 2) / 2.0, places=6)

    @unittest.skipUnless(Path(f"{REFERENCE}_spectra.npz").is_file() and HAVE_DATA, "h2 export not present")
    def test_reproduces_h2_summary(self):
        rows = PM.selfcheck_reference_export(f"{REFERENCE}_spectra.npz", f"{REFERENCE}_summary.json",
                                             PM.load_protocol(ROOT / PM.DEFAULT_PROTOCOL), DATA)
        self.assertTrue(all(row[4] for row in rows), [row for row in rows if not row[4]])
        joint = PM.rows_value(rows, "balanced_blind_energy")
        self.assertAlmostEqual(joint, 0.2697682, places=6)


@unittest.skipUnless(HAVE_DATA and torch.cuda.is_available(), "needs data and a GPU")
class TrainResume(unittest.TestCase):
    def test_short_run_resumes_and_refuses_changed_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = [sys.executable, str(BASE / "adapter/train.py"), "--task", "phdos", "--label_scaling",
                    "standardized", "--loss", "MAE", "--run_dir", f"{tmp}/run", "--limit_train", "256",
                    "--limit_valid", "64", "--batch_size", "64",
                    "--graph_cache", str(ROOT / "results/runs/mat2spec_baseline_20261110/graphs")]
            quiet = dict(capture_output=True, text=True)
            self.assertEqual(subprocess.run(base + ["--epochs", "4"], **quiet).returncode, 0)
            status = json.loads(Path(f"{tmp}/run/execution/status.json").read_text())
            self.assertEqual((status["state"], status["epoch"]), ("completed", 4))
            self.assertEqual(subprocess.run(base + ["--epochs", "4"], **quiet).returncode, T.EXIT_CONTRACT)
            self.assertEqual(subprocess.run(base + ["--epochs", "8", "--resume"], **quiet).returncode,
                             T.EXIT_CONTRACT)
            self.assertEqual(json.loads(Path(f"{tmp}/run/execution/status.json").read_text())["state"], "completed")
            self.assertEqual(subprocess.run(base + ["--epochs", "4", "--resume"], **quiet).returncode, 0)


if __name__ == "__main__":
    unittest.main()
