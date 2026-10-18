"""Width formulas, matched initialization, compatibility and decision boundaries."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from model.periodic_geometry_cache import GIB, PeriodicGeometryCache
from model.periodic_messages import InvariantMessageBlock, PeriodicLocalMessage, prepare_shared_message_geometry
from model.transformer import Transformer
from tests.model.test_local_global_encoder import fixture
from tests.pipeline.test_global_norm import model_params
from tests.pipeline.test_local_module import exports as module_exports
from tests.pipeline.test_local_structure_controls import predict
from tools.checks import check_local_width as checker
from tools.checks.check_long_budget import captured_config, learning_rate_trace
from tools.eval import local_width_compare as runner
from tools.eval.lg_structure_diagnosis import pair_errors
from tools.eval.local_global_compare import training_command
from tools.train import build_arg_parser
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.checkpoint_selection import BLIND_R2_METRIC
from utils.crystal import CrystalGeometry
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import (
    _is_local_parameter, capture_training_rng, comparison_contract, initialize_comparison_model,
    validate_local_global_config,
)
from utils.local_structure_experiment import FROZEN
from utils.local_width_experiment import (
    WIDE_WIDTHS, WIDTH_SCHEMA, local_width_config, narrow_reference_parameters, validate_width_model,
)
from utils.periodic_geometry import build_periodic_records
from utils.paths import file_sha256
from utils.zproj_alignment import state_hash


def configuration():
    return local_width_config(ExperimentConfig(**{
        **FROZEN, "local_route": "I3", "selection_metric": BLIND_R2_METRIC}), "unit_width")


def parameters():
    return {**model_params("W1"), "local_scalar_channels": 128, "local_hidden_channels": 256}


def direct_block(block, scalar, g):
    """Original full concatenation formula, independent of affine factorization."""
    flat = scalar.flatten(0, 1)
    edge = block.edge_mlp(torch.cat((flat[g.receiver], flat[g.sender], g.radial), -1)) * g.cutoff[:, None]
    radial = torch.zeros_like(flat).index_add(0, g.receiver, edge).view_as(scalar)
    radial = radial / g.degree.to(scalar.dtype).clamp_min(1).sqrt()[..., None]
    left, right = g.pairs
    q = (g.unit[left] * g.unit[right]).sum(-1).clamp(-1, 1)
    angles = torch.stack((q, q.square(), q**3, q**4), -1)
    forward = torch.cat((flat[g.receiver[left]], flat[g.sender[left]], flat[g.sender[right]],
                         g.radial[left], g.radial[right], angles), -1)
    backward = torch.cat((flat[g.receiver[left]], flat[g.sender[right]], flat[g.sender[left]],
                          g.radial[right], g.radial[left], angles), -1)
    pair = .5 * (block.angle_mlp(forward) + block.angle_mlp(backward))
    pair = pair * (g.cutoff[left] * g.cutoff[right])[:, None]
    angular = torch.zeros_like(flat).index_add(0, g.receiver[left], pair).view_as(scalar)
    angular = angular / g.pair_count.to(scalar.dtype).clamp_min(1).sqrt()[..., None]
    counts = torch.stack((g.degree.to(scalar.dtype).log1p(), g.effective_count), -1)
    return block.norm(scalar + block.update(torch.cat((scalar, radial, angular, counts), -1)))


class TestLocalWidth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_wide_factorization_matches_direct_formula_and_input_parameter_geometry_gradients(self):
        for dtype, tolerance in ((torch.float32, 4e-6), (torch.float64, 2e-12)):
            z, geometry = fixture(dtype)
            mask = z.eq(0)
            block = InvariantMessageBlock(pair_chunk=2, recompute_angles=False,
                scalar_channels=128, hidden_channels=256).to(dtype)
            original = deepcopy(block)
            scalar = torch.randn((*z.shape, 128), dtype=dtype)
            weight = torch.randn_like(scalar)
            results = []
            for selected, direct in ((block, False), (original, True)):
                x = scalar.clone().requires_grad_()
                cell = geometry.cell.clone().requires_grad_()
                coords = geometry.frac_coords.clone().requires_grad_()
                records = build_periodic_records(CrystalGeometry(cell, coords), mask, 6.)
                prepared = prepare_shared_message_geometry(records, "A")
                out = direct_block(selected, x, prepared) if direct else selected(x, prepared)
                grads = torch.autograd.grad((out * weight).sum(), (x, cell, coords, *selected.parameters()))
                results.append((out, grads))
            torch.testing.assert_close(results[0][0], results[1][0], atol=tolerance, rtol=tolerance)
            for actual, expected in zip(results[0][1], results[1][1]):
                torch.testing.assert_close(actual, expected, atol=10*tolerance, rtol=10*tolerance)

    def test_all_nonlocal_initial_tensors_match_and_entire_local_branch_is_fresh(self):
        params = parameters()
        baseline_params = narrow_reference_parameters(params)
        narrow = Transformer(**baseline_params)
        baseline_info = initialize_comparison_model(narrow, baseline_params, 42)
        wide = Transformer(**params)
        rng = torch.get_rng_state()
        info = initialize_comparison_model(wide, params, 42)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        self.assertEqual(info["narrow_initialization"], baseline_info)
        validate_width_model(wide, params, require_identity=True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            canonical = Transformer(**params)
        for name, value in wide.state_dict().items():
            source = canonical.state_dict()[name] if _is_local_parameter(name) else narrow.state_dict()[name]
            torch.testing.assert_close(value, source, atol=0, rtol=0)
            self.assertNotEqual(value.data_ptr(), source.data_ptr())
        key = "encoder.local_layers.0.message.local_output.bias"
        self.assertEqual(wide.state_dict()[key].shape, narrow.state_dict()[key].shape)
        self.assertFalse(torch.equal(wide.state_dict()[key], narrow.state_dict()[key]))
        self.assertEqual(info["parameter_counts"]["matched_nonlocal"],
                         info["reference_parameter_counts"]["matched_nonlocal"])
        self.assertGreater(info["parameter_counts"]["local_message"],
                           info["reference_parameter_counts"]["local_message"])

    def test_assembly_cli_contract_and_legacy_shapes_are_isolated(self):
        cfg = configuration()
        parsed = ExperimentConfig.from_args(build_arg_parser().parse_args(training_command(cfg)[2:]))
        self.assertEqual(asdict(parsed), asdict(cfg))
        assembled = captured_config(cfg)
        params = assembled["model"]["params"]["sub_model"]["transformer"]
        self.assertEqual((params["local_scalar_channels"], params["local_hidden_channels"]), WIDE_WIDTHS)
        self.assertEqual(params["d_model"], 512)
        reference_cfg = replace(cfg, local_route="I3")
        narrow_assembled = captured_config(reference_cfg)
        self.assertEqual(runner.restore_narrow_assembly(assembled), narrow_assembled)
        self.assertEqual(comparison_contract(cfg, params)["schema"], WIDTH_SCHEMA)
        for change in ({"seed": 43}, {"epochs": 36}, {"use_amp": True}, {"batch_size": 64},
                       {"selection_metric": "balanced_oracle_mae"}):
            with self.assertRaises(ValueError):
                validate_local_global_config(replace(cfg, **change))
        for invalid in ({}, {"local_scalar_channels": 64, "local_hidden_channels": 128}):
            with self.assertRaises(ValueError):
                Transformer(**{**model_params("W1"), **invalid})
        with self.assertRaises(ValueError):
            Transformer(**{**model_params("I3"), "local_scalar_channels": 128, "local_hidden_channels": 256})
        for widths in ((True, 256), (128, 0), (1.5, 256)):
            with self.assertRaises(ValueError):
                PeriodicLocalMessage("A", scalar_channels=widths[0], hidden_channels=widths[1])
        with self.assertRaises(ValueError):
            PeriodicLocalMessage("B+", scalar_channels=128, hidden_channels=256)
        self.assertEqual(sum(p.numel() for p in PeriodicLocalMessage("A").parameters()), 261568)
        self.assertEqual(sum(p.numel() for p in PeriodicLocalMessage("B+").parameters()), 184772)

    def test_wide_cache_preserves_training_outputs_gradients_rng_and_padding(self):
        params = {**parameters(), "dropout": .05}
        model = Transformer(**params).train()
        initialize_comparison_model(model, params, 42)
        z, geometry = fixture()
        rng = torch.get_rng_state()
        outputs = []
        for cached in (False, True):
            torch.set_rng_state(rng)
            model.zero_grad(set_to_none=True)
            model.enable_local_geometry_cache(enabled=cached, max_bytes=2**20)
            out = predict(model, z, geometry)
            sum(value.square().mean() for value in out.values()).backward()
            grads = {name: p.grad.clone() for name, p in model.named_parameters()}
            self.assertTrue(all(torch.isfinite(p).all() for p in grads.values()))
            outputs.append((out, grads, torch.get_rng_state()))
        for actual, expected in zip(outputs[0][:2], outputs[1][:2]):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                torch.testing.assert_close(actual[key], expected[key], atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(outputs[0][2], outputs[1][2], atol=0, rtol=0)
        cache = model._local_geometry_cache
        self.assertEqual(cache.configuration()["workspace_guard"], "local-pairs-width-v1")
        h = torch.randn((*z.shape, 16))
        h[z.eq(0)] = float("nan")
        records = build_periodic_records(geometry, z.eq(0), 6.)
        hidden = model.encoder(h, records, z.eq(0))
        self.assertTrue(torch.isfinite(hidden).all())
        self.assertTrue(hidden[z.eq(0)].eq(0).all())

    def test_workspace_guard_scales_without_changing_cached_geometry(self):
        z, geometry = fixture()
        requirements = []
        for widths in ((64, 128), WIDE_WIDTHS):
            cache = PeriodicGeometryCache("A", "cpu", max_bytes=2**20, local_blocks=4,
                scalar_channels=widths[0], hidden_channels=widths[1])
            with patch.object(cache, "_reserve_workspace") as reserve:
                records, prepared = cache.get(geometry, z.eq(0), radius=6.)
            requirements.append(reserve.call_args.args[1])
            torch.testing.assert_close(records.vectors, build_periodic_records(geometry, z.eq(0), 6.).vectors)
            if widths == (64, 128):
                self.assertEqual(cache.configuration()["workspace_guard"], "local-pairs-v1")
                self.assertNotIn("scalar_channels", cache.configuration())
                pairs = len(prepared.pairs[0])
                self.assertEqual(requirements[-1], 4*GIB + pairs*1024*4*4)
        self.assertEqual(requirements[1] - 4*GIB, 2*(requirements[0] - 4*GIB))

    def test_wide_recovery_rejects_narrow_and_foreign_selection_contracts(self):
        cfg, params = configuration(), parameters()
        contract = comparison_contract(cfg, params)
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters())
        payload = build_ablation_checkpoint(1, "M1", 42, False, model, opt, .5,
            model_contract=contract, training_rng=capture_training_rng())
        restore_ablation_checkpoint(payload, model, opt, False, model_contract=contract)
        for other in (comparison_contract(replace(cfg, local_route="I3"), narrow_reference_parameters(params)),
                      {**contract, "selection_metric": "balanced_oracle_mae"}):
            with self.assertRaisesRegex(ValueError, "different model contract"):
                restore_ablation_checkpoint(payload, model, opt, False, model_contract=other)

    def test_capacity_choice_and_baseline_adoption_are_independent(self):
        original = module_exports()
        # Candidate beats A64 but remains worse than I1.
        data = {"W1": original["I2"], "A64": original["I1"], "I1": original["I3"]}
        report = runner.comparison_report(data, draws=20)
        self.assertTrue(report["decisions"]["width_upgrade_conditions_met"])
        self.assertFalse(report["decisions"]["baseline_upgrade_conditions_met"])
        self.assertFalse(report["decisions"]["automatic_followup_training"])
        self.assertEqual(report["bootstrap"]["seed"], 20261016)
        for key in report["joint_blind"].values():
            self.assertTrue(key["bootstrap"]["same_indices_for_tasks_and_models"])
        # A score improvement may still violate either task's failure protection.
        worse = deepcopy(data)
        worse["W1"][1].iloc[:2, worse["W1"][1].columns.get_loc("blind_r2_phdos")] = -.1
        self.assertFalse(runner.comparison_report(worse, draws=20)["decisions"]["width_upgrade_conditions_met"])

    def preflight_fixture(self):
        cfg, params = configuration(), parameters()
        assembled = {"model": {"params": {"sub_model": {"transformer": params}}}}
        saved = {"config": runner.restore_narrow_assembly(assembled), "cli": {"data_contract": {}},
                 "runtime": {"local_global_init": {"state": "narrow"},
                             "structure_training_rng": [{"state_sha256": "rng"}]}}
        refs = {"A64": {"data_order": {}}, "I1": {}}
        entry = {"configuration": asdict(cfg), "assembled_configuration": assembled,
            "contract": comparison_contract(cfg, params), "initialization": {
                "narrow_initialization": saved["runtime"]["local_global_init"],
                "matched_nonlocal_state_sha256": "common"}, "training_start_rng_sha256": "rng",
            "data_order": {}, "checkpoint_roundtrip": {"weights_optimizer_rng_equal": True,
                "foreign_contracts_rejected": True}, "narrow_checkpoint_rejected": True,
            "symmetry": {"torch.float32": {}, "torch.float64": {}}, "costs": [1, 2, 3],
            "geometry_cache": {"workspace_guard": "local-pairs-width-v1"}}
        report = {"schema": runner.PREFLIGHT_SCHEMA, "passed": True, "references": refs, "code_sha256": {},
            "data_contract": {}, "learning_rate_35": learning_rate_trace(35, cfg.lr), "controls": {"W1": entry},
            "narrow_probe": {"initialization": saved["runtime"]["local_global_init"],
                             "matched_nonlocal_state_sha256": "common", "costs": [1, 2, 3]}}
        return cfg, saved, refs, report

    def test_preflight_rejects_source_recipe_width_rng_or_recovery_drift(self):
        cfg, saved, refs, report = self.preflight_fixture()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "reference_identity", return_value=(saved, None, refs)), \
                patch.object(checker, "code_fingerprints", return_value={}):
            path = Path(directory) / "preflight.json"
            path.write_text(json.dumps(report))
            self.assertEqual(runner.verify_preflight(path, Path(directory), cfg), report)
            for key in ("code_sha256", "references", "learning_rate_35", "narrow_probe"):
                changed = deepcopy(report)
                changed[key] = {"changed": True}
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    runner.verify_preflight(path, Path(directory), cfg)
            for key in ("training_start_rng_sha256", "initialization", "checkpoint_roundtrip", "geometry_cache"):
                changed = deepcopy(report)
                changed["controls"]["W1"][key] = {}
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    runner.verify_preflight(path, Path(directory), cfg)

    def test_partial_runs_or_source_drift_stop_before_training_or_export(self):
        cfg = configuration()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(checker, "code_fingerprints", return_value={}), \
                patch.object(runner, "summarize", return_value=[{"status": "partial"}]), \
                patch.object(runner.norm, "execute") as run, patch.object(runner, "export") as export:
            with self.assertRaisesRegex(ValueError, "automatic resume"):
                runner.execute(cfg, {"code_sha256": {}}, Path(directory))
            with self.assertRaisesRegex(ValueError, "source changed"):
                runner.analyze(cfg, {"code_sha256": {"old": "hash"}}, Path(directory))
            run.assert_not_called()
            export.assert_not_called()

    def test_reported_common_hash_excludes_every_local_message_tensor(self):
        params = parameters()
        model = Transformer(**params)
        info = initialize_comparison_model(model, params, 42)
        expected = {key: value for key, value in model.state_dict().items() if not _is_local_parameter(key)}
        self.assertEqual(info["matched_nonlocal_state_sha256"], state_hash(expected))
        self.assertTrue(np.isfinite(info["parameter_count"]))

    def test_reference_authentication_retains_historical_sources_and_detects_export_drift(self):
        cfg = replace(configuration(), local_route="I3")
        params = narrow_reference_parameters(parameters())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "analysis").mkdir()
            run = root / ("ablation_m1" + cfg.tag)
            run.mkdir()
            for name in ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth"):
                (run / name).write_text(name)
            history = root / "history.csv"
            history.write_text("complete")
            common = {"I1": {"run_dir": "i1", "data_order": {}},
                      "I2": {"evidence_dir": "old", "export_files_sha256": {}}}
            hashes = {}
            for label in ("I3", "I1"):
                for suffix in runner.EXPORT_SUFFIXES:
                    path = root / "analysis" / f"{label}_{suffix}"
                    path.write_text(path.name)
                    hashes[path.name] = file_sha256(path)
                    if label == "I1":
                        common["I2"]["export_files_sha256"][path.name] = hashes[path.name]
            saved = {"config": {"route": "I3"}, "cli": {"data_contract": {}},
                     "runtime": {"local_global_init": {"init": "historical"}}}
            contract = comparison_contract(cfg, params)
            pre = {"schema": runner.narrow.PREFLIGHT_SCHEMA, "passed": True,
                "code_sha256": {"old_source": "immutable"}, "data_contract": {}, "references": common,
                "controls": {"I3": {"contract": contract, "configuration": asdict(cfg),
                    "assembled_configuration": saved["config"], "initialization": saved["runtime"]["local_global_init"],
                    "data_order": {}}}}
            artifact = {"run_dir": str(run), **{name: file_sha256(run / name) for name in
                ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth")},
                "history_path": str(history), "history_sha256": file_sha256(history)}
            report = {"schema": "local-module-analysis-v1", "analysis_complete": True,
                "code_sha256": pre["code_sha256"], "data_contract": {},
                "run_artifacts": {"I3": artifact, "I1": common["I1"]}}
            (root / "preflight_final.json").write_text(json.dumps(pre))
            (root / "analysis/report.json").write_text(json.dumps(report))
            verified = {"passed": True, "protocol_and_completed_checkpoints_passed": True,
                "preflight_sha256": file_sha256(root / "preflight_final.json"),
                "input_artifacts_sha256": {**hashes, "report.json": file_sha256(root / "analysis/report.json")}}
            (root / "analysis/verification_codex.json").write_text(json.dumps(verified))
            (root / "queue.json").write_text(json.dumps({"schema": "local-module-queue-v1",
                "status": "complete", "arms": {"I3": "complete"}}))
            with patch.object(runner.narrow, "reference_identity", return_value=(None, None, common)), \
                    patch.object(runner, "load_run", return_value=(saved, params, contract)), \
                    patch.object(runner, "verify_completed") as complete, \
                    patch.object(runner, "repository_path", return_value=history):
                _, actual, identities = runner.reference_identity(root)
                complete.assert_called_once()
                self.assertEqual(actual, cfg)
                self.assertEqual(len(identities["A64"]["exports"]), 6)
                self.assertTrue(identities["A64"]["exports"]["A64_summary.json"]["source"].endswith("I3_summary.json"))
                (root / "analysis/I3_valid_spectra.npz").write_text("drift")
                with self.assertRaisesRegex(ValueError, "exports differ"):
                    runner.reference_identity(root)

    def test_analysis_infers_only_wide_and_reuses_both_verified_references(self):
        ids = ["a", "b", "c", "d", "e"]
        keys = [((14, 1),)] * 3 + [((8, 1),)] * 2
        targets = np.array([[.1, .9], [.4, .6], [.6, .4], [.2, .8], [.8, .2]])

        def save_split(out, label, split):
            prediction = targets * .8 + .1
            np.savez(out / f"{label}_{split}_spectra.npz", material_ids=np.asarray(ids),
                tgt_shape_edos=targets, tgt_shape_phdos=targets, pred_shape_edos=prediction, pred_shape_phdos=prediction)
            pd.DataFrame({"material_id": ids}).to_csv(out / f"{label}_{split}_samples.csv", index=False)
            (out / f"{label}_{'train_' if split == 'train' else ''}summary.json").write_text("{}")
            return {(group, task): pair_errors(prediction, targets, keys, ids)
                    for group in ("absolute", "reduced") for task in ("edos", "phdos")}

        def fake_pairs(spectrum, samples, dataset):
            with np.load(spectrum) as values:
                return {(group, task): pair_errors(values["pred_shape_" + task], values["tgt_shape_" + task], keys, ids)
                        for group in ("absolute", "reduced") for task in ("edos", "phdos")}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "frozen"
            frozen.mkdir()
            identities = {}
            for label in ("A64", "I1"):
                for split in ("train", "valid"):
                    save_split(frozen, label, split)
                identities[label] = {"evidence_dir": str(frozen), "exports": {
                    f"{label}_{suffix}": {"source": str(frozen / f"{label}_{suffix}"),
                        "sha256": file_sha256(frozen / f"{label}_{suffix}")} for suffix in runner.EXPORT_SUFFIXES}}
            run = root / "wide"
            run.mkdir()
            for name in ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth"):
                (run / name).write_text(name)
            history = root / "history.csv"
            history.write_text("complete")
            saved = {"config": {}}
            preflight = {"references": identities, "code_sha256": {}, "data_contract": {},
                         "controls": {"W1": {"initialization": {}}}}
            builder = MagicMock()
            datasets = {split: MagicMock() for split in ("train", "valid")}
            datasets["train"].__len__.return_value = 18704
            datasets["valid"].__len__.return_value = 2313
            builder.get_dataset.side_effect = lambda **kw: datasets[kw["split"]]
            with patch.object(checker, "code_fingerprints", return_value={}), \
                    patch.object(runner, "reference_identity", return_value=(saved, None, identities)), \
                    patch.object(runner, "verify_completed"), \
                    patch.object(runner, "run_directory", return_value=run), \
                    patch.object(runner, "repository_path", return_value=history), \
                    patch.object(runner, "load_run", return_value=(saved, None, None)), \
                    patch.object(runner, "ConfigBuilder", return_value=builder), \
                    patch.object(runner, "export", side_effect=lambda run, samples, summary, **kw:
                        save_split(samples.parent, "W1", "valid")) as valid_export, \
                    patch.object(runner.bplus, "export_train", side_effect=lambda run, out, label:
                        save_split(out, label, "train")) as train_export, \
                    patch.object(runner.bplus, "spectrum_pairs", side_effect=fake_pairs), \
                    patch.object(runner, "load", return_value=({}, None)), \
                    patch.object(runner, "comparison_report", return_value={}):
                report = runner.analyze(configuration(), preflight, root / "analysis")
            valid_export.assert_called_once()
            train_export.assert_called_once()
            self.assertEqual(train_export.call_args.args[2], "W1")
            self.assertEqual([call.kwargs["split"] for call in builder.get_dataset.call_args_list], ["train", "valid"])
            for identity in identities.values():
                for name, entry in identity["exports"].items():
                    self.assertEqual(file_sha256(root / "analysis" / name), entry["sha256"])
            self.assertTrue(report["structural_diagnosis"]["diagnostic_only"])
            self.assertEqual(len(report["structural_diagnosis"]["summaries"]), 24)
            self.assertEqual(len(list((root / "analysis").glob("*_pairs.csv"))), 24)


if __name__ == "__main__":
    unittest.main()
