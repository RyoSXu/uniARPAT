"""Candidate-1 pilot contracts: RNG reset shim + resource gate + verdict.

CPU, synthetic and mock only: no Q1 cache, no GPU, no training run and no real
checkpoint file. ``TestRngResetFlow`` exercises the actual ``train_and_eval``
control flow with a tiny synthetic Transformer and a data loader that stops at
the first iteration; ``TestResourceGatePureFunctions`` covers the resource
tool's decision logic without touching a device or a dataset; the
``joint_content_pilot_verdict`` tests cover the valid-only three-arm verdict
preflight, pure decision and CLI surface.
"""

import hashlib
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DistributedSampler, TensorDataset

import run_ablation_experiments as runner
from model.transformer import Transformer
from utils.builder import ConfigBuilder
from utils.experiment_config import ExperimentConfig

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "tools" / "eval") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools" / "eval"))

import edos_error_attribution as attribution  # noqa: E402
import joint_content_resource_gate as gate  # noqa: E402
import joint_content_pilot_verdict as verdict  # noqa: E402
from joint_content_pilot_verdict import (  # noqa: E402
    BOOTSTRAP_SEED,
    COMPOSITION_PAIR_COLUMNS,
    COMPOSITION_PAIR_METRICS,
    EXPECTED_VALID_COUNT,
    METRIC_COLUMNS,
    auxiliary_radial_classification,
    build_auxiliary_readouts,
    build_comparison,
    build_composition_pair_table,
    composition_pair_readout_rows,
    evaluate_joint_verdict,
    load_and_validate_checkpoint,
    paired_metric_summary,
    validate_arm_config,
    validate_arm_artifact_paths,
    validate_checkpoint,
    validate_cross_arm_configs,
    validate_sample_frames,
    win_conditions_met,
)


def _tiny_transformer(use_g2=False, g2_content_mode=None, num_encoder_layers=1):
    """Small synthetic Transformer; decoupled from any production dimension."""
    extra = {}
    if g2_content_mode is not None:
        extra["g2_content_mode"] = g2_content_mode
    return Transformer(
        token_num=64, d_model=16, nhead=2, edos_num=4, phdos_num=2,
        num_encoder_layers=num_encoder_layers, num_decoder_layers=1,
        dim_feedforward=32, dropout=0.0, activation="gelu",
        normalize_before=False, decoupled_decoder=True,
        use_gated_cross_attn=False, head_type="legacy", predict_scale=False,
        use_g2=use_g2, scale_mode="none", **extra,
    )


class _StopAfterOrdering(Exception):
    """Sentinel: the loader records the first iteration attempt and stops."""


class _RecordingLoader:
    """Loader stand-in that logs when data iteration is attempted."""

    def __init__(self, events):
        self.events = events

    def __len__(self):
        return 1

    def __iter__(self):
        self.events.append("data_iter")
        raise _StopAfterOrdering()


class TestRngAlignment(unittest.TestCase):
    """Cross-architecture RNG alignment after an init checkpoint load."""

    def test_reset_realigns_torch_stream_across_architectures(self):
        # Without the shim, construction consumes different numbers of draws,
        # so the post-construction stream is already misaligned.
        runner.setup_ablation_seed(42)
        _tiny_transformer(use_g2=False)
        control_draw = torch.rand(6)
        runner.setup_ablation_seed(42)
        _tiny_transformer(use_g2=True, g2_content_mode="joint")
        joint_draw = torch.rand(6)
        self.assertFalse(torch.equal(control_draw, joint_draw))

        draws = {}
        for name, kwargs in (
            ("control", dict(use_g2=False)),
            ("radial", dict(use_g2=True, g2_content_mode="radial")),
            ("joint", dict(use_g2=True, g2_content_mode="joint")),
        ):
            runner.setup_ablation_seed(42)
            _tiny_transformer(**kwargs)
            runner.setup_ablation_seed(42)
            draws[name] = (random.random(), float(np.random.rand()), torch.rand(6))

        self.assertEqual(draws["control"][0], draws["joint"][0])
        self.assertEqual(draws["control"][1], draws["joint"][1])
        self.assertTrue(torch.equal(draws["control"][2], draws["joint"][2]))
        self.assertTrue(torch.equal(draws["radial"][2], draws["joint"][2]))

    def test_reset_after_init_load_aligns_stream(self):
        runner.setup_ablation_seed(7)
        b7_state = _tiny_transformer(use_g2=False).state_dict()

        def sample(build, mode):
            runner.setup_ablation_seed(7)
            arm = build()
            gate.load_initial_state(arm, mode, b7_state, 1)
            optimizer = torch.optim.AdamW(arm.parameters(), lr=1e-3)
            gate.verify_initialized(arm, mode, b7_state, optimizer, 1)
            runner.setup_ablation_seed(7)
            return torch.rand(5)

        control = sample(lambda: _tiny_transformer(use_g2=False), "control")
        radial = sample(
            lambda: _tiny_transformer(use_g2=True, g2_content_mode="radial"),
            "radial")
        joint = sample(
            lambda: _tiny_transformer(use_g2=True, g2_content_mode="joint"),
            "joint")
        self.assertTrue(torch.equal(control, joint))
        self.assertTrue(torch.equal(radial, joint))

    def test_distributed_sampler_epoch_order_is_independent_of_global_rng(self):
        dataset = TensorDataset(torch.arange(23))

        def order(epoch, global_seed):
            runner.setup_ablation_seed(global_seed)
            sampler = DistributedSampler(
                dataset, num_replicas=1, rank=0, shuffle=True, seed=0)
            sampler.set_epoch(epoch)
            return list(sampler)

        self.assertEqual(order(3, 1), order(3, 999))
        self.assertNotEqual(order(3, 1), order(4, 1))


class TestRngResetFlow(unittest.TestCase):
    """The runner's control flow: early rejection and reset ordering."""

    def setUp(self):
        repo = Path(__file__).resolve().parents[1]
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.root / "configs").mkdir()
        (self.root / "configs/default.yaml").write_text(
            (repo / "configs/default.yaml").read_text())
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)

    def _run_flow(self, tag, reset):
        events = []
        built = {}
        real_transformer = Transformer
        real_setup = runner.setup_ablation_seed

        def seed_spy(seed):
            events.append(("seed", seed))
            real_setup(seed)

        def transformer_spy(**kwargs):
            tiny = dict(kwargs)
            tiny.update(d_model=16, nhead=2, num_encoder_layers=1,
                        num_decoder_layers=1, dim_feedforward=32,
                        dropout=0.0, edos_num=4, phdos_num=2)
            built["model"] = real_transformer(**tiny)
            return built["model"]

        class _Builder(ConfigBuilder):
            def get_dataloader(self, *args, **kwargs):
                return _RecordingLoader(events)

        def load_spy(path, *args, **kwargs):
            events.append(("init_load", str(path)))
            return {"model": built["model"].state_dict()}

        cfg = ExperimentConfig(
            model_name="M1", epochs=1, batch_size=2, tag=tag, data_dir="unused",
            init_ckpt="synthetic-b7.pth", reset_rng_after_init=reset,
            skip_test_eval=True)
        with patch.object(runner, "ConfigBuilder", _Builder), \
                patch("model.model.Transformer", side_effect=transformer_spy), \
                patch.object(torch, "load", side_effect=load_spy), \
                patch.object(runner, "setup_ablation_seed", side_effect=seed_spy), \
                patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(runner, "logger", Mock()):
            with self.assertRaises(_StopAfterOrdering):
                runner.train_and_eval(cfg)
        return events

    @staticmethod
    def _positions(events, kind):
        return [i for i, event in enumerate(events) if event == kind]

    def test_reset_runs_after_init_load_and_before_first_iteration(self):
        events = self._run_flow("_rngon", True)
        seeds = self._positions(events, ("seed", 42))
        loads = [i for i, e in enumerate(events) if isinstance(e, tuple)
                 and e[0] == "init_load"]
        data = self._positions(events, "data_iter")
        self.assertEqual(len(seeds), 2)
        self.assertEqual(len(loads), 1)
        self.assertEqual(len(data), 1)
        self.assertLess(loads[0], seeds[1])
        self.assertLess(seeds[1], data[0])
        config = yaml.safe_load(
            (self.root / "output/ablation_m1_rngon/config_used.yaml").read_text())
        self.assertTrue(config["cli"]["reset_rng_after_init"])

    def test_without_flag_seed_runs_once_and_before_init_load(self):
        events = self._run_flow("_rngoff", False)
        seeds = self._positions(events, ("seed", 42))
        loads = [i for i, e in enumerate(events) if isinstance(e, tuple)
                 and e[0] == "init_load"]
        self.assertEqual(len(seeds), 1)
        self.assertLess(seeds[0], loads[0])
        config = yaml.safe_load(
            (self.root / "output/ablation_m1_rngoff/config_used.yaml").read_text())
        self.assertFalse(config["cli"]["reset_rng_after_init"])

    def test_reset_without_init_ckpt_rejected_before_dirs_or_data(self):
        cfg = ExperimentConfig(model_name="M1", tag="_rngbad", epochs=1,
                               reset_rng_after_init=True)
        builder = Mock()
        builder.get_dataloader.side_effect = AssertionError("data entry reached")
        with patch.object(runner, "ConfigBuilder", return_value=builder) as ctor:
            with self.assertRaises(ValueError):
                runner.train_and_eval(cfg)
            ctor.assert_not_called()
            builder.get_dataloader.assert_not_called()
        self.assertFalse((self.root / "output/ablation_m1_rngbad").exists())
        self.assertFalse((self.root / "results/history_m1_rngbad.csv").exists())

    def test_cli_and_config_default_flag(self):
        parser = runner.build_arg_parser()
        cfg = ExperimentConfig.from_args(parser.parse_args(["--model", "M1"]))
        self.assertFalse(cfg.reset_rng_after_init)
        cfg = ExperimentConfig.from_args(parser.parse_args(
            ["--model", "M1", "--reset_rng_after_init",
             "--init_ckpt", "x.pth"]))
        self.assertTrue(cfg.reset_rng_after_init)
        self.assertEqual(cfg.init_ckpt, "x.pth")
        self.assertFalse(ExperimentConfig().reset_rng_after_init)


class TestResourceGatePureFunctions(unittest.TestCase):
    """Checkpoint keys, loading modes, comparisons and device rejection."""

    def test_expected_g2_key_sets_match_synthetic_model(self):
        self.assertEqual(gate.expected_g2_keys(0, "radial"), set())
        self.assertEqual(len(gate.expected_g2_keys(1, "radial")), 9)
        self.assertEqual(len(gate.expected_g2_keys(1, "joint")), 15)
        self.assertEqual(len(gate.expected_g2_keys(2, "joint")), 30)
        model = _tiny_transformer(
            use_g2=True, g2_content_mode="joint", num_encoder_layers=2)
        actual = {k for k in model.state_dict() if k.startswith(gate.G2_PREFIX)}
        self.assertEqual(actual, gate.expected_g2_keys(2, "joint"))
        with self.assertRaises(ValueError):
            gate.expected_g2_keys(1, "bogus")

    def test_verify_b7_checkpoint_identity(self):
        state = {"encoder.weight": torch.zeros(2)}
        good = {"epoch": 33, "model_name": "M1", "seed": 42, "model": state}
        out = gate.verify_b7_checkpoint(
            good, "abc", expected_sha256="abc", expected_keys=1)
        self.assertIs(out, state)
        g2_state = dict(state)
        g2_state["encoder.g2_msgs.0.alpha"] = torch.zeros(())
        bad_cases = [
            (good, "bad", 1),
            ({**good, "epoch": 34}, "abc", 1),
            ({**good, "model_name": "M2"}, "abc", 1),
            ({**good, "seed": 7}, "abc", 1),
            ({"epoch": 33, "model_name": "M1", "seed": 42, "model": g2_state},
             "abc", 1),
            (good, "abc", 2),
            ({"epoch": 33, "model_name": "M1", "seed": 42}, "abc", 1),
        ]
        for payload, sha256, expected_keys in bad_cases:
            with self.assertRaises(ValueError):
                gate.verify_b7_checkpoint(
                    payload, sha256, expected_sha256="abc",
                    expected_keys=expected_keys)

    def test_verify_b7_run_config(self):
        config = {"cli": {
            "model": "M1", "epochs": 35, "batch_size": 32, "seed": 42,
            "norm": "sumnorm", "scale_mode": "eta",
        }}
        self.assertEqual(gate.verify_b7_run_config(config)["model"], "M1")
        for key, bad_value in (("epochs", 10), ("norm", "minmax"),
                               ("scale_mode", "none")):
            bad = {"cli": dict(config["cli"], **{key: bad_value})}
            with self.assertRaises(ValueError):
                gate.verify_b7_run_config(bad)
        with self.assertRaises(ValueError):
            gate.verify_b7_run_config({"cli": dict(config["cli"], use_g2=False)})

    def test_load_initial_state_contract_per_mode(self):
        b7_state = _tiny_transformer(use_g2=False, num_encoder_layers=2).state_dict()

        control = _tiny_transformer(use_g2=False, num_encoder_layers=2)
        gate.load_initial_state(control, "control", b7_state, 2)

        radial = _tiny_transformer(
            use_g2=True, g2_content_mode="radial", num_encoder_layers=2)
        gate.load_initial_state(radial, "radial", b7_state, 2)
        self.assertEqual(
            {k for k in radial.state_dict() if k.startswith(gate.G2_PREFIX)},
            gate.expected_g2_keys(2, "radial"))

        joint = _tiny_transformer(
            use_g2=True, g2_content_mode="joint", num_encoder_layers=2)
        gate.load_initial_state(joint, "joint", b7_state, 2)
        optimizer = torch.optim.AdamW(joint.parameters(), lr=1e-3)
        gate.verify_initialized(joint, "joint", b7_state, optimizer, 2)

        # a G2-bearing radial checkpoint must never initialize a joint arm
        with self.assertRaises(ValueError):
            other = _tiny_transformer(
                use_g2=True, g2_content_mode="joint", num_encoder_layers=2)
            gate.load_initial_state(other, "joint", radial.state_dict(), 2)

    def test_verify_initialized_rejects_alpha_shared_weight_optimizer(self):
        b7_state = _tiny_transformer(use_g2=False).state_dict()
        joint = _tiny_transformer(use_g2=True, g2_content_mode="joint")
        gate.load_initial_state(joint, "joint", b7_state, 1)
        optimizer = torch.optim.AdamW(joint.parameters(), lr=1e-3)
        gate.verify_initialized(joint, "joint", b7_state, optimizer, 1)

        with torch.no_grad():
            joint.encoder.g2_msgs[0].alpha.fill_(0.25)
        with self.assertRaises(ValueError):
            gate.verify_initialized(joint, "joint", b7_state, optimizer, 1)

        dirty = _tiny_transformer(use_g2=True, g2_content_mode="joint")
        gate.load_initial_state(dirty, "joint", b7_state, 1)
        params = dict(dirty.named_parameters())
        shared_name = next(k for k in b7_state if k in params)
        with torch.no_grad():
            params[shared_name].add_(1.0)
        dirty_optimizer = torch.optim.AdamW(dirty.parameters(), lr=1e-3)
        with self.assertRaises(ValueError) as shared_ctx:
            gate.verify_initialized(dirty, "joint", b7_state, dirty_optimizer, 1)
        self.assertIn("shared weight", str(shared_ctx.exception))

        clean = _tiny_transformer(use_g2=True, g2_content_mode="joint")
        gate.load_initial_state(clean, "joint", b7_state, 1)
        clean_optimizer = torch.optim.AdamW(clean.parameters(), lr=1e-3)
        clean_optimizer.state[clean.encoder.g2_msgs[0].alpha] = {
            "step": torch.tensor(0.0)}
        with self.assertRaises(ValueError) as opt_ctx:
            gate.verify_initialized(clean, "joint", b7_state, clean_optimizer, 1)
        self.assertIn("optimizer", str(opt_ctx.exception))

    def test_build_comparison_ratios(self):
        arms = {
            "control": {"cold": {"gpu_time_ms": 100.0, "wall_time_ms": 110.0,
                                 "peak_alloc_mb": 1000.0, "peak_reserved_mb": 1200.0}},
            "radial": {"cold": {"gpu_time_ms": 120.0, "wall_time_ms": 132.0,
                                "peak_alloc_mb": 1100.0, "peak_reserved_mb": 1320.0}},
            "joint": {"cold": {"gpu_time_ms": 150.0, "wall_time_ms": 165.0,
                               "peak_alloc_mb": 1200.0, "peak_reserved_mb": 1440.0}},
        }
        comparison = gate.build_comparison(arms)
        self.assertAlmostEqual(comparison["joint_vs_control"]["time_ratio"], 1.5)
        self.assertAlmostEqual(comparison["joint_vs_radial"]["time_ratio"], 1.25)
        self.assertAlmostEqual(comparison["radial_vs_control"]["time_ratio"], 1.2)
        self.assertAlmostEqual(
            comparison["joint_vs_control"]["peak_alloc_ratio"], 1.2)
        self.assertAlmostEqual(
            comparison["joint_vs_control"]["wall_time_ratio"], 1.5)
        arms["control"]["cold"]["gpu_time_ms"] = 0.0
        self.assertIsNone(
            gate.build_comparison(arms)["joint_vs_control"]["time_ratio"])

    def test_verify_same_batch(self):
        arms = {
            mode: {"batch": {"batch_sha256": "same", "batch_size": 32}}
            for mode in gate.ARMS
        }
        self.assertEqual(gate.verify_same_batch(arms), "same")
        arms["joint"]["batch"]["batch_sha256"] = "different"
        with self.assertRaises(ValueError):
            gate.verify_same_batch(arms)
        arms["joint"]["batch"]["batch_sha256"] = "same"
        arms["radial"]["batch"]["batch_size"] = 31
        with self.assertRaises(ValueError):
            gate.verify_same_batch(arms)

    def test_device_gates(self):
        self.assertEqual(
            gate.require_v100("Tesla V100-SXM2-32GB"), "Tesla V100-SXM2-32GB")
        with self.assertRaises(RuntimeError):
            gate.require_v100("Tesla A100-SXM4-80GB")
        gate.require_cuda_device(torch.device("cuda:0"))
        with self.assertRaises(RuntimeError):
            gate.require_cuda_device(torch.device("cpu"))

    def test_batch_digest_stable_and_sensitive(self):
        base = (torch.arange(4).reshape(2, 2), "mpid", None)
        same = (torch.arange(4).reshape(2, 2), "mpid", None)
        other = (torch.arange(4).reshape(2, 2) + 1, "mpid", None)
        self.assertEqual(gate.batch_digest(base), gate.batch_digest(same))
        self.assertNotEqual(gate.batch_digest(base), gate.batch_digest(other))

    def test_file_sha256(self):
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(b"uniarpat")
            path = handle.name
        self.addCleanup(os.remove, path)
        expected = hashlib.sha256(b"uniarpat").hexdigest()
        self.assertEqual(gate.file_sha256(path), expected)


def _arm_config(arm, num_layers=1):
    """Valid synthetic ``config_used.yaml`` payload for one pilot arm."""
    use_g2 = arm != "control"
    mode = "joint" if arm == "joint" else "radial"
    cli = {
        "model": "M1", "epochs": 10, "batch_size": 32, "seed": 42,
        "norm": "sumnorm", "scale_mode": "eta",
        "use_g2": use_g2, "g2_content_mode": mode,
        "skip_test_eval": True, "use_amp": False, "use_bucket_batch": False,
        "reset_rng_after_init": True,
        "init_ckpt": "output/ablation_m1_e9ctl/checkpoint_best.pth",
    }
    effective = {
        "model": {
            "params": {
                "use_amp": False,
                "sub_model": {
                    "transformer": {
                        "use_g2": use_g2, "g2_content_mode": mode,
                        "num_encoder_layers": num_layers,
                    }
                },
            }
        }
    }
    return {"cli": cli, "config": effective}


def _comparison(edos_oracle=0.0, edos_oracle_fail=0.0, edos_blind=0.03,
                edos_blind_fail=0.0, phdos_oracle=0.0, phdos_oracle_fail=0.0,
                phdos_blind=0.0, phdos_blind_fail=0.0):
    """Minimal joint-vs-control comparison; defaults are a full win."""
    return {
        "edos_oracle": {"delta_median": edos_oracle, "delta_fail": edos_oracle_fail},
        "edos_blind": {"delta_median": edos_blind, "delta_fail": edos_blind_fail},
        "phdos_oracle": {"delta_median": phdos_oracle,
                         "delta_fail": phdos_oracle_fail},
        "phdos_blind": {"delta_median": phdos_blind, "delta_fail": phdos_blind_fail},
    }


def _sample_frames(mpids=("a", "b", "c", "d"), roughness=None):
    columns = {"mpid": list(mpids)}
    for column in METRIC_COLUMNS.values():
        columns[column] = [0.1, 0.2, 0.3, 0.4]
    columns["edos_spectral_roughness"] = (
        list(roughness) if roughness is not None else [0.1, 0.2, 0.6, 0.8])
    frame = pd.DataFrame(columns)
    return {arm: frame.copy() for arm in ("control", "radial", "joint")}


def _composition_pairs_frame(mpids=("a", "b", "c", "d")):
    """Synthetic composition-pair long frame in fixed pair-then-arm order."""
    rows = []
    for arm, predicted in (("control", 0.1), ("radial", 0.2), ("joint", 0.3)):
        for position in range(0, len(mpids), 2):
            rows.append({
                "arm": arm,
                "sample_index_a": position,
                "sample_index_b": position + 1,
                "mpid_a": mpids[position],
                "mpid_b": mpids[position + 1],
                "target_tv": 0.5,
                "predicted_tv": predicted,
                "contrast_error_tv": predicted + 0.05,
                "contrast_cosine": 1.0 - predicted,
            })
    return pd.DataFrame(rows, columns=list(COMPOSITION_PAIR_COLUMNS))


class TestJointContentVerdictPreflight(unittest.TestCase):
    """Config/checkpoint preflight rejects before any formal result is written."""

    def test_valid_configs_pass_and_cross_arm_equality(self):
        configs = {arm: _arm_config(arm) for arm in ("control", "radial", "joint")}
        for arm in configs:
            self.assertEqual(validate_arm_config(arm, configs[arm])["use_g2"],
                             arm != "control")
        validate_cross_arm_configs(configs)

    def test_cross_arm_unrelated_difference_rejected(self):
        configs = {arm: _arm_config(arm) for arm in ("control", "radial", "joint")}
        configs["radial"]["cli"]["batch_size"] = 16
        with self.assertRaises(ValueError):
            validate_cross_arm_configs(configs)

    def test_arm_mode_and_flag_rejections(self):
        mutations = (
            ("control", "use_g2", True),
            ("control", "g2_content_mode", "joint"),
            ("radial", "g2_content_mode", "joint"),
            ("joint", "use_g2", False),
            ("joint", "g2_content_mode", "radial"),
            ("radial", "skip_test_eval", False),
            ("radial", "use_amp", True),
            ("radial", "use_bucket_batch", True),
            ("radial", "reset_rng_after_init", False),
            ("radial", "init_ckpt", "output/ablation_m1_other/checkpoint_best.pth"),
            ("radial", "epochs", 35),
            ("radial", "batch_size", 16),
            ("radial", "seed", 7),
            ("radial", "norm", "minmax"),
            ("radial", "scale_mode", "none"),
        )
        for arm, key, value in mutations:
            config = _arm_config(arm)
            config["cli"][key] = value
            with self.assertRaises(ValueError, msg=f"{arm}.{key}={value}"):
                validate_arm_config(arm, config)

    def test_effective_config_disagreement_rejected(self):
        config = _arm_config("control")
        transformer = (config["config"]["model"]["params"]["sub_model"]["transformer"])
        transformer["use_g2"] = True
        with self.assertRaises(ValueError):
            validate_arm_config("control", config)

    def test_missing_init_ckpt_rejected(self):
        config = _arm_config("control")
        config["cli"].pop("init_ckpt")
        with self.assertRaises(ValueError):
            validate_arm_config("control", config)

    def test_latest_checkpoint_file_name_enforced_before_load(self):
        with self.assertRaises(ValueError):
            load_and_validate_checkpoint(
                "control", "output/ablation_m1_jcctl/checkpoint_best.pth",
                _arm_config("control"))

    def test_frozen_tag_and_sibling_artifact_paths(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root) / "ablation_m1_jcjoint"
            run_dir.mkdir()
            checkpoint = run_dir / "checkpoint_latest.pth"
            config = run_dir / "config_used.yaml"
            validate_arm_artifact_paths("joint", checkpoint, config)
            with self.assertRaises(ValueError):
                validate_arm_artifact_paths(
                    "joint", Path(root) / "wrong/checkpoint_latest.pth", config)
            with self.assertRaises(ValueError):
                validate_arm_artifact_paths(
                    "joint", checkpoint, Path(root) / "other/config_used.yaml")

    def test_checkpoint_epoch_and_key_set(self):
        radial_state = {
            key: torch.zeros(()) for key in gate.expected_g2_keys(1, "radial")}
        payload = {"epoch": 10, "model_name": "M1", "use_amp": False,
                   "model": radial_state}
        validate_checkpoint(payload, "radial", _arm_config("radial"))
        validate_checkpoint(
            {"epoch": 10, "model_name": "M1", "use_amp": False, "model": {}},
            "control", _arm_config("control"))

        bad_payloads = (
            {"epoch": 9, "model_name": "M1", "use_amp": False, "model": radial_state},
            {"epoch": 10, "model_name": "M2", "use_amp": False, "model": radial_state},
            {"epoch": 10, "model_name": "M1", "use_amp": True, "model": radial_state},
            {"epoch": 10, "model_name": "M1", "use_amp": False, "model": {}},
            {"epoch": 10, "model_name": "M1", "use_amp": False,
             "model": {k: v for k, v in radial_state.items()
                       if not k.endswith("W_o.bias")}},
        )
        for payload in bad_payloads:
            with self.assertRaises(ValueError):
                validate_checkpoint(payload, "radial", _arm_config("radial"))


class TestJointContentVerdictSamples(unittest.TestCase):
    """Same-order, equal-count valid arms are required."""

    def test_sample_order_count_and_ids(self):
        self.assertEqual(EXPECTED_VALID_COUNT, 2313)
        frames = _sample_frames()
        ids = validate_sample_frames(frames, expected_count=4)
        self.assertEqual(list(ids), ["a", "b", "c", "d"])

        reversed_frames = _sample_frames()
        reversed_frames["joint"] = reversed_frames["joint"].iloc[::-1].reset_index(
            drop=True)
        with self.assertRaisesRegex(ValueError, "IDs/order"):
            validate_sample_frames(reversed_frames, expected_count=4)

        with self.assertRaises(ValueError):
            validate_sample_frames(frames, expected_count=5)

        with self.assertRaises(ValueError):
            validate_sample_frames(
                frames, expected_count=4, expected_ids=np.array(["a", "b", "c", "x"]))

    def test_missing_metric_column_rejected(self):
        frames = _sample_frames()
        missing = METRIC_COLUMNS["phdos_blind"]
        for arm in frames:
            frames[arm] = frames[arm].drop(columns=missing)
        with self.assertRaises(ValueError):
            validate_sample_frames(frames, expected_count=4)


class TestJointContentVerdictDecision(unittest.TestCase):
    """Threshold boundaries, explainable states and auxiliary isolation."""

    def test_win_boundaries_pass(self):
        cases = (
            {"edos_blind": 0.02},
            {"edos_oracle": 0.0},
            {"phdos_oracle": -0.02},
            {"phdos_blind": -0.02},
            {"edos_blind": 0.02, "edos_oracle": 0.0,
             "phdos_oracle": -0.02, "phdos_blind": -0.02},
        )
        for overrides in cases:
            decision = evaluate_joint_verdict(_comparison(**overrides))
            self.assertEqual(decision["verdict"], "win", overrides)
            self.assertTrue(all(decision["win_conditions"].values()))

    def test_fail_exactly_one_point_is_not_win(self):
        for metric, kwargs in (
            ("blind_edos_fail_lt_0.01", {"edos_blind_fail": 0.01}),
            ("oracle_phdos_fail_lt_0.01", {"phdos_oracle_fail": 0.01}),
            ("blind_phdos_fail_lt_0.01", {"phdos_blind_fail": 0.01}),
        ):
            decision = evaluate_joint_verdict(_comparison(**kwargs))
            self.assertNotEqual(decision["verdict"], "win", metric)
            self.assertFalse(decision["win_conditions"][metric])

    def test_explainable_non_win_states(self):
        self.assertEqual(
            evaluate_joint_verdict(_comparison(
                edos_blind=0.0))["verdict"], "tie")
        self.assertEqual(
            evaluate_joint_verdict(_comparison(
                phdos_blind=-0.03))["verdict"], "degraded_or_guard_failed")
        partial = evaluate_joint_verdict(
            _comparison(edos_blind=0.03, edos_oracle=-0.001))
        self.assertEqual(partial["verdict"], "primary_insufficient")
        self.assertFalse(partial["guard_breaches"]["edos_oracle_median_le_-0.02"])

    def test_technical_incomplete(self):
        self.assertEqual(
            evaluate_joint_verdict(
                _comparison(), technical_incomplete=True)["verdict"],
            "technical_incomplete")
        non_finite = _comparison()
        non_finite["edos_blind"]["delta_median"] = float("nan")
        self.assertEqual(
            evaluate_joint_verdict(non_finite)["verdict"], "technical_incomplete")
        self.assertFalse(win_conditions_met({"verdict": "technical_incomplete",
                                             "win_conditions": {}}))

    def test_auxiliary_radial_does_not_change_primary(self):
        control = evaluate_joint_verdict(_comparison(edos_blind=0.0))  # tie
        radial_win = evaluate_joint_verdict(_comparison())
        classification = auxiliary_radial_classification(control, radial_win)
        self.assertTrue(classification["joint_wins_vs_radial"])
        self.assertTrue(classification["only_beats_radial"])
        # the primary verdict object is untouched by the radial classification
        self.assertEqual(control["verdict"], "tie")

        radial_tie = evaluate_joint_verdict(_comparison(edos_blind=0.0))
        self.assertFalse(
            auxiliary_radial_classification(control, radial_tie)["only_beats_radial"])
        radial_win_primary_win = auxiliary_radial_classification(
            evaluate_joint_verdict(_comparison()), radial_win)
        self.assertTrue(radial_win_primary_win["joint_wins_vs_radial"])
        self.assertFalse(radial_win_primary_win["only_beats_radial"])

    def test_paired_bootstrap_is_reproducible(self):
        reference = np.array([0.2, -0.1, 0.4, 0.3, 0.5, 0.1])
        candidate = reference + np.array([0.05, 0.1, -0.02, 0.03, 0.0, 0.01])
        first = paired_metric_summary(reference, candidate, "edos_blind",
                                      replicates=200, seed=BOOTSTRAP_SEED)
        second = paired_metric_summary(reference, candidate, "edos_blind",
                                       replicates=200, seed=BOOTSTRAP_SEED)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["delta_median"], 0.04, places=12)


class TestJointContentVerdictCli(unittest.TestCase):
    """The CLI is valid-only: no split and no test entry is exposed."""

    def test_no_split_or_test_options(self):
        parser = verdict.build_arg_parser()
        options = " ".join(parser._option_string_actions).lower()
        self.assertNotIn("split", options)
        self.assertNotIn("test", options)

    def test_cli_requires_all_arms_and_bootstrap_default(self):
        parser = verdict.build_arg_parser()
        args = parser.parse_args([
            "--control-checkpoint", "c.pth", "--control-config", "c.yaml",
            "--radial-checkpoint", "r.pth", "--radial-config", "r.yaml",
            "--joint-checkpoint", "j.pth", "--joint-config", "j.yaml",
            "--output-prefix", "results/out",
        ])
        self.assertEqual(args.bootstrap, 2000)
        self.assertEqual(args.device, "auto")


class TestJointContentVerdictAuxiliary(unittest.TestCase):
    """Auxiliary readouts explain only and record gaps explicitly."""

    def test_auxiliary_gaps_and_no_verdict_effect(self):
        frames = _sample_frames()
        comparisons = {
            name: build_comparison(frames[reference], frames[candidate],
                                   replicates=50, seed=BOOTSTRAP_SEED)
            for name, reference, candidate in [
                ("joint_vs_control", "control", "joint"),
                ("joint_vs_radial", "radial", "joint"),
                ("radial_vs_control", "control", "radial"),
            ]
        }
        decision_before = evaluate_joint_verdict(comparisons["joint_vs_control"])
        composition = _composition_pairs_frame()
        with tempfile.TemporaryDirectory() as root:
            frame, gaps = build_auxiliary_readouts(
                frames, comparisons, root, composition_pairs=composition)
        decision_after = evaluate_joint_verdict(comparisons["joint_vs_control"])
        self.assertEqual(decision_before, decision_after)
        readouts = {gap["readout"] for gap in gaps}
        self.assertNotIn("composition_pair_table", readouts)
        self.assertIn("spectral_support_quartile", readouts)
        self.assertIn("roughness_train_p90", set(frame["kind"]))
        self.assertIn("roughness_other", set(frame["population"]))
        summary = frame[frame["kind"] == "composition_pair_table"]
        self.assertEqual(len(summary), len(verdict.COMPARISONS)
                         * len(COMPOSITION_PAIR_METRICS))
        self.assertTrue(summary["reference_fail_pct"].isna().all())
        self.assertTrue(summary["candidate_fail_pct"].isna().all())
        self.assertTrue(summary["delta_fail_pp"].isna().all())
        joint_vs_control = summary[
            (summary["comparison"] == "joint_vs_control")
            & (summary["metric"] == "predicted_tv")]
        self.assertAlmostEqual(
            float(joint_vs_control["delta_median"].iloc[0]), 0.2)


class TestCompositionPairTable(unittest.TestCase):
    """Frozen pair IDs, contrast formula and target reproduction; no Q1."""

    BINS = 128

    def _fixture(self, root, *, oracle_offset=0.0, duplicate=False,
                 bad_mpid=False):
        results = Path(root) / "results"
        results.mkdir(parents=True, exist_ok=True)
        valid = Path(root) / "data/train4ARPAT/valid"
        valid.mkdir(parents=True, exist_ok=True)
        mpids = np.array(["m0", "m1", "m2", "m3"])
        np.save(valid / "valid_index.npy", mpids)
        target = np.zeros((4, self.BINS), dtype=np.float32)
        for index in range(4):
            target[index, index] = 1.0
        np.save(valid / "edos_tgtdos_valid.npy", target)

        pairs = [
            {"sample_index_a": 0, "sample_index_b": 1, "mpid_a": "m0",
             "mpid_b": "m1", "oracle_target_tv": 1.0 + oracle_offset},
            {"sample_index_a": 2, "sample_index_b": 3, "mpid_a": "m2",
             "mpid_b": "m3", "oracle_target_tv": 1.0},
        ]
        if duplicate:
            pairs.append(dict(pairs[0]))
        if bad_mpid:
            pairs[0]["mpid_a"] = "wrong"
        pd.DataFrame(pairs).to_csv(
            results / "edos_spectral_support_q1_valid_pairs.csv", index=False)

        shape_paths = {}
        for arm in verdict.ARM_ORDER:
            array = np.zeros((4, self.BINS), dtype=np.float32)
            array[1, 1] = 1.0
            array[2, 2] = 1.0
            array[3, 3] = 1.0
            array[0, {"control": 0, "radial": 2, "joint": 3}[arm]] = 1.0
            path = Path(root) / f"{arm}.npy"
            np.save(path, array)
            shape_paths[arm] = path
        frames = {arm: pd.DataFrame({"mpid": mpids}) for arm in verdict.ARM_ORDER}
        return shape_paths, frames

    def test_pair_order_ids_and_formula(self):
        with tempfile.TemporaryDirectory() as root:
            shape_paths, frames = self._fixture(root)
            frame = build_composition_pair_table(
                shape_paths, frames, repo_root=root, expected_count=4)
        self.assertEqual(list(frame.columns), list(COMPOSITION_PAIR_COLUMNS))
        self.assertEqual(len(frame), 6)
        self.assertEqual(list(frame["arm"]), ["control", "radial", "joint"] * 2)
        self.assertEqual(list(frame["sample_index_a"]), [0, 0, 0, 2, 2, 2])
        self.assertEqual(list(frame["sample_index_b"]), [1, 1, 1, 3, 3, 3])
        self.assertEqual(list(frame["mpid_a"]), ["m0"] * 3 + ["m2"] * 3)
        self.assertEqual(list(frame["mpid_b"]), ["m1"] * 3 + ["m3"] * 3)
        self.assertTrue(np.allclose(frame["target_tv"], 1.0))
        control = frame[frame["arm"] == "control"].iloc[0]
        self.assertAlmostEqual(control["predicted_tv"], 1.0)
        self.assertAlmostEqual(control["contrast_error_tv"], 0.0)
        self.assertAlmostEqual(control["contrast_cosine"], 1.0)
        radial = frame[frame["arm"] == "radial"].iloc[0]
        self.assertAlmostEqual(radial["predicted_tv"], 1.0)
        self.assertAlmostEqual(radial["contrast_error_tv"], 1.0)
        self.assertAlmostEqual(radial["contrast_cosine"], 0.5)
        joint = frame[frame["arm"] == "joint"].iloc[0]
        self.assertAlmostEqual(joint["contrast_cosine"], 0.5)

    def test_target_reproduction_tolerance(self):
        with tempfile.TemporaryDirectory() as root:
            shape_paths, frames = self._fixture(root, oracle_offset=0.5)
            with self.assertRaisesRegex(ValueError, "oracle_target_tv"):
                build_composition_pair_table(
                    shape_paths, frames, repo_root=root, expected_count=4)

    def test_rejects_duplicate_pairs_and_bad_mpid(self):
        for kwargs in ({"duplicate": True}, {"bad_mpid": True}):
            with tempfile.TemporaryDirectory() as root:
                shape_paths, frames = self._fixture(root, **kwargs)
                with self.assertRaises(ValueError):
                    build_composition_pair_table(
                        shape_paths, frames, repo_root=root, expected_count=4)

    def test_shape_validation(self):
        good = np.full((2, self.BINS), 1.0 / self.BINS)
        validated = attribution.validate_edos_shape_sumnorm(good, 2)
        self.assertEqual(validated.shape, (2, self.BINS))
        bad = (
            np.full(self.BINS, 1.0 / self.BINS),
            np.full((3, self.BINS), 1.0 / self.BINS),
            np.full((2, 64), 1.0 / 64),
            np.full((2, self.BINS), np.nan),
            -good,
            np.full((2, self.BINS), 1.0 / 64),
        )
        for array in bad:
            with self.assertRaises(ValueError):
                attribution.validate_edos_shape_sumnorm(array, 2)

    def test_variable_final_batch_is_concatenated(self):
        first = np.full((32, self.BINS), 1.0 / self.BINS)
        final = np.full((9, self.BINS), 1.0 / self.BINS)
        arrays = attribution.finalize_evaluation_arrays({
            "metric": [0.1, 0.2],
            "edos_shape_sumnorm": [first, final],
        })
        self.assertEqual(arrays["metric"].shape, (2,))
        self.assertEqual(arrays["edos_shape_sumnorm"].shape, (41, self.BINS))
        attribution.validate_edos_shape_sumnorm(arrays["edos_shape_sumnorm"], 41)


class TestJointContentVerdictOrchestration(unittest.TestCase):
    """Mocked end-to-end staging never reads Q1 or touches a device."""

    def _artifacts(self, root):
        checkpoints = {}
        configs = {}
        for arm in verdict.ARM_ORDER:
            run_dir = Path(root) / verdict.ARM_OUTPUT_DIRS[arm]
            run_dir.mkdir()
            checkpoints[arm] = run_dir / "checkpoint_latest.pth"
            configs[arm] = run_dir / "config_used.yaml"
            mode = "control" if arm == "control" else arm
            state = {
                key: torch.zeros(())
                for key in gate.expected_g2_keys(1, mode)
            }
            torch.save({
                "epoch": 10, "model_name": "M1", "use_amp": False,
                "model": state,
            }, checkpoints[arm])
            configs[arm].write_text(yaml.safe_dump(_arm_config(arm)))
        return checkpoints, configs

    @staticmethod
    def _audit_writer(*args, **kwargs):
        del kwargs
        prefix = Path(args[2])
        arm = prefix.name
        n = EXPECTED_VALID_COUNT
        base = np.linspace(0.1, 0.4, n)
        offset = {"control": 0.0, "radial": 0.01, "joint": 0.03}[arm]
        frame = pd.DataFrame({
            "mpid": [f"synthetic-{i}" for i in range(n)],
            "edos_spectral_roughness": np.linspace(0.1, 0.6, n),
            **{column: base + offset for column in METRIC_COLUMNS.values()},
        })
        frame.to_csv(prefix.with_name(f"{arm}_samples.csv"), index=False)

    @staticmethod
    def _composition_frame():
        """Synthetic long frame so the mocked flow never reads Q1 data."""
        rows = [
            {
                "arm": arm, "sample_index_a": 0, "sample_index_b": 1,
                "mpid_a": "synthetic-0", "mpid_b": "synthetic-1",
                "target_tv": 0.5, "predicted_tv": 0.1,
                "contrast_error_tv": 0.1, "contrast_cosine": 0.9,
            }
            for arm in verdict.ARM_ORDER
        ]
        return pd.DataFrame(rows, columns=list(COMPOSITION_PAIR_COLUMNS))

    def test_success_moves_complete_formal_set(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoints, configs = self._artifacts(root)
            prefix = Path(root) / "results/joint_content_pilot_q1_valid"
            expected_ids = np.array(
                [f"synthetic-{i}" for i in range(EXPECTED_VALID_COUNT)])
            with patch.object(verdict, "file_sha256", return_value=verdict.B7_CKPT_SHA256), \
                    patch.object(verdict, "run_audit", side_effect=self._audit_writer), \
                    patch.object(verdict.np, "load", return_value=expected_ids), \
                    patch.object(verdict, "build_composition_pair_table",
                                 return_value=self._composition_frame()), \
                    patch.object(verdict, "build_auxiliary_readouts",
                                 return_value=(pd.DataFrame(), [])):
                paths, decision = verdict.run_verdict(
                    checkpoints, configs, prefix, torch.device("cpu"), replicates=20)
            self.assertEqual(decision["verdict"], "win")
            self.assertTrue(all(path.exists() for path in paths.values()))
            self.assertTrue(paths["composition_pairs"].exists())
            payload = json.loads(paths["summary"].read_text())
            self.assertEqual(payload["verdict"]["verdict"], "win")
            self.assertEqual(payload["split"], "Q1 valid")
            self.assertEqual(
                payload["auxiliary"]["composition_pairs_csv"],
                str(paths["composition_pairs"]))
            self.assertNotIn(
                "composition_pair_table",
                {gap["readout"] for gap in payload["auxiliary"]["gaps"]})

    def test_worker_failure_leaves_no_formal_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoints, configs = self._artifacts(root)
            prefix = Path(root) / "results/joint_content_pilot_q1_valid"

            def fail_on_joint(*args, **kwargs):
                if Path(args[2]).name == "joint":
                    raise RuntimeError("synthetic audit failure")
                return self._audit_writer(*args, **kwargs)

            with patch.object(verdict, "file_sha256", return_value=verdict.B7_CKPT_SHA256), \
                    patch.object(verdict, "run_audit", side_effect=fail_on_joint):
                with self.assertRaisesRegex(RuntimeError, "synthetic audit failure"):
                    verdict.run_verdict(
                        checkpoints, configs, prefix, torch.device("cpu"), replicates=20)
            self.assertFalse(any(
                path.exists() for path in verdict.formal_paths(prefix).values()))


if __name__ == "__main__":
    unittest.main()
