"""Train I2 only, then compare full train/valid structural differences with I1."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tools.eval import global_norm_compare as norm
from tools.eval.legacy_backbone_compare import run_directory
from tools.eval.lg_structure_diagnosis import dump, pair_errors, paired_group_interval, summarize_pairs
from tools.eval.lg_train_subset_diagnosis import composition_keys_by_position
from tools.eval.lg_valid_export import load_run, select_best
from tools.eval.local_global_compare import summarize, training_command
from tools.train import configure_local_global_execution, evaluate_split
from utils.bucket_batch import crystal_collate
from utils.builder import ConfigBuilder
from utils.i1_bplus_experiment import DEFAULT_I1_REFERENCE, I1_PREFLIGHT, i1_bplus_config
from utils.local_global_experiment import comparison_contract, validate_local_global_config
from utils.paths import file_sha256, input_path, repository_path
from utils.experiment_config import ExperimentConfig

LABELS = ("I2", "I1")
SCOPE = ("full B+ versus radial C1 on the corrected I1 backbone; seed42, 35 epochs, "
         "crystal_q1_v1; common initial tensors matched; capacity also changes; "
         "train/valid only; intervals conditional on selected checkpoints, not training-seed uncertainty")


def reference_identity(reference_dir):
    saved, params, contract = load_run(reference_dir)
    cfg = ExperimentConfig(**{**contract["training"],
                             "tag": reference_dir.name.removeprefix("ablation_m1")})
    validate_local_global_config(cfg)
    if cfg.local_route != "I1" or comparison_contract(cfg, params) != contract:
        raise ValueError("B+ refinement requires the frozen corrected I1 reference")
    data_dir = Path(input_path(cfg.data_dir))
    manifest_path = data_dir / "manifest.json"
    data_contract = saved["cli"]["data_contract"]
    manifest = json.loads(manifest_path.read_text())
    if (file_sha256(manifest_path) != data_contract["manifest_sha256"]
            or manifest["schema"] != data_contract["schema"]
            or cfg.norm != data_contract["normalization"] or cfg.pair_budget != data_contract["pair_budget"]):
        raise ValueError("current data manifest differs from I1")
    for name, entry in manifest["files"].items():
        if Path(name).parts[0] in ("train", "valid") and file_sha256(data_dir / name) != entry["sha256"]:
            raise ValueError("train/valid payload differs from I1: " + name)
    # The selected best and complete latest checkpoint must bind the same data,
    # contract and exact 35-epoch history. No test loader or test arrays opened.
    picked = select_best(reference_dir, cfg.selection_metric, saved)
    latest = torch.load(reference_dir / "checkpoint_latest.pth", map_location="cpu", weights_only=True)
    if (picked["history"].epoch.tolist() != list(range(1, 36)) or latest.get("epoch") != 35
            or latest.get("model_contract") != contract
            or any(checkpoint.get("data_contract") != data_contract
                   for checkpoint in (latest, picked["checkpoint"]))):
        raise ValueError("I1 has not completed its exact, common-data training protocol")
    rng = saved["runtime"]["structure_training_rng"][0]
    if rng["from_epoch"] != 1 or rng["method"] != "seed-reset-before-loop":
        raise ValueError("I1 must have a recorded seed-reset training start")
    preflight_path = ROOT / I1_PREFLIGHT
    old_preflight = json.loads(preflight_path.read_text())
    old_entry = old_preflight["controls"]["I1"]
    if (old_preflight.get("schema") != "attention-projection-preflight-v1"
            or old_preflight.get("passed") is not True
            or old_entry["configuration"] != asdict(cfg)
            or old_entry["initialization"] != saved["runtime"]["local_global_init"]
            or old_entry["training_start_rng_sha256"] != rng["state_sha256"]
            or old_preflight["data_contract"] != data_contract):
        raise ValueError("I1 differs from its historical successful preflight")
    history_path = repository_path(f"results/reports/history_m1{cfg.tag}.csv")
    identities = {"I1": {"run_dir": str(reference_dir.resolve()),
        "files": {name: file_sha256(reference_dir / name) for name in
                  ("config_used.yaml", "checkpoint_best.pth", "checkpoint_latest.pth")},
        "history_path": str(history_path), "history_sha256": file_sha256(history_path),
        "preflight_path": str(preflight_path), "preflight_sha256": file_sha256(preflight_path)}}
    return saved, cfg, identities


def verify_preflight(path, reference_dir, cfg):
    from tools.checks.check_i1_bplus import code_fingerprints

    report = json.loads(path.read_text())
    saved, _, references = reference_identity(reference_dir)
    if (report.get("schema") != "i1-bplus-preflight-v1" or report.get("passed") is not True
            or report.get("code_sha256") != code_fingerprints()
            or report.get("references") != references or set(report.get("controls", {})) != {"I2"}
            or report["controls"]["I2"]["configuration"] != asdict(cfg)
            or report["data_contract"] != saved["cli"]["data_contract"]
            or report["controls"]["I2"]["initialization"]["i1_initialization"]
                != saved["runtime"]["local_global_init"]
            or report["controls"]["I2"]["training_start_rng_sha256"]
                != saved["runtime"]["structure_training_rng"][0]["state_sha256"]):
        raise ValueError("I2 source, reference, configuration or initialization changed after preflight")
    return report


def spectrum_pairs(spectrum_path, samples_path, dataset):
    frame = pd.read_csv(samples_path)
    lookup = {str(value): i for i, value in enumerate(dataset.material_ids)}
    keys = composition_keys_by_position(dataset, [lookup[value] for value in frame.material_id])
    result = {}
    with np.load(spectrum_path) as spectra:
        if spectra["material_ids"].tolist() != frame.material_id.tolist():
            raise ValueError("spectrum and material order differ")
        for grouping in ("absolute", "reduced"):
            for task in ("edos", "phdos"):
                result[grouping, task] = pair_errors(spectra[f"pred_shape_{task}"],
                    spectra[f"tgt_shape_{task}"], keys[grouping], frame.material_id.tolist())
    return result


def group_summary(frame, task):
    # Match the established I1 diagnosis: near-zero eDOS target differences
    # are recorded separately, with one fixed threshold on target values.
    excluded = frame.target_tv < 1e-6 if task == "edos" else np.zeros(len(frame), dtype=bool)
    primary = frame.loc[~excluded].reset_index(drop=True)
    grouped = primary.groupby("composition")[["difference_error", "target_tv"]].mean()
    if not len(grouped) or grouped.target_tv.mean() <= 0:
        raise ValueError("structural comparison needs nonzero target differences")
    zero = primary.assign(difference_error=primary.target_tv)
    return {**summarize_pairs(primary), "near_zero_target_pairs": int(np.sum(excluded)),
            "group_mean_difference_error": float(grouped.difference_error.mean()),
            "zero_response_group_mean_error": float(grouped.target_tv.mean()),
            "error_ratio": float(grouped.difference_error.mean() / grouped.target_tv.mean()),
            "model_minus_zero_response": paired_group_interval(zero, primary)}, primary


def export_train(run, out, label):
    saved, _, contract = load_run(run)
    before = {name: file_sha256(run / name) for name in ("config_used.yaml", "checkpoint_best.pth")}
    picked = select_best(run, contract["selection_metric"], saved)
    builder = ConfigBuilder(**saved["config"])
    dataset = builder.get_dataset(split="train", dos_minmax=True, dos_sumnorm=True)
    if len(dataset) != 18704 or saved["config"]["dataset"]["train"].get("augment", False):
        raise ValueError("train export requires the unchanged complete effective train view")
    ids = [str(value) for value in dataset.material_ids]
    model = builder.get_model()
    transformer = model.model["transformer"]
    transformer.load_state_dict(picked["checkpoint"]["model"], strict=True)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    configure_local_global_execution(transformer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=crystal_collate)
    started = time.monotonic()
    metrics = evaluate_split(model, loader, sumnorm=True, return_sample_level=True,
                             return_spectra=True, sample_ids=ids, pair_budget=saved["cli"].get("pair_budget"))
    frame, spectra = metrics.pop("sample_df"), metrics.pop("sample_spectra")
    if (frame.material_id.tolist() != ids or frame.material_id.duplicated().any()
            or not np.isfinite(frame.select_dtypes("number").to_numpy()).all()
            or any(not np.isfinite(value).all() for value in spectra.values())):
        raise ValueError("non-finite or misaligned complete train export")
    samples_path, spectrum_path = out / f"{label}_train_samples.csv", out / f"{label}_train_spectra.npz"
    frame.to_csv(samples_path, index=False)
    np.savez(spectrum_path, material_ids=np.asarray(ids, dtype=str), **spectra)
    if any(file_sha256(run / name) != digest for name, digest in before.items()):
        raise ValueError("source checkpoint or configuration changed during frozen inference")
    result = spectrum_pairs(spectrum_path, samples_path, dataset)
    dump(out / f"{label}_train_summary.json", {"best_epoch": int(picked["checkpoint"]["epoch"]),
         "samples": len(ids), "elapsed_s": time.monotonic() - started,
         "scope": "frozen selected checkpoint; eval/no_grad; complete train", "metrics": metrics})
    del model, transformer, dataset, loader, picked
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def assert_paired_targets(left, right):
    with np.load(left) as a, np.load(right) as b:
        for key in ("material_ids", "tgt_shape_edos", "tgt_shape_phdos"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError("I2 and I1 frozen export targets/IDs differ: " + key)


def analyze(cfg, preflight, out):
    from tools.checks.check_i1_bplus import code_fingerprints

    if preflight["code_sha256"] != code_fingerprints():
        raise ValueError("I2 source changed during training")
    saved, _, identities = reference_identity(Path(preflight["references"]["I1"]["run_dir"]))
    if identities != preflight["references"] or saved["cli"]["data_contract"] != preflight["data_contract"]:
        raise ValueError("I1 reference changed during I2 training")
    report = norm.analyze(cfg, preflight, out, labels=LABELS, scope=SCOPE)
    runs = {"I2": run_directory(cfg), "I1": Path(identities["I1"]["run_dir"])}
    pairs = {}
    for label, run in runs.items():
        for (grouping, task), table in export_train(run, out, label).items():
            table.to_csv(out / f"{label}_train_{grouping}_{task}_pairs.csv", index=False)
            pairs[label, "train", grouping, task] = table
        for grouping in ("absolute", "reduced"):
            for task in ("edos", "phdos"):
                pairs[label, "valid", grouping, task] = pd.read_csv(
                    out / f"{label}_valid_{grouping}_{task}_pairs.csv")
    structure = {"scope": "normalized signed spectral differences; groups equally weighted",
                 "primary": "valid_absolute_edos", "bootstrap": {"draws": 2000,
                 "unit": "composition group", "seed": 42, "training_seed_uncertainty": False},
                 "summaries": {}, "I2_minus_I1": {}}
    for split in ("train", "valid"):
        assert_paired_targets(out / f"I2_{split}_spectra.npz", out / f"I1_{split}_spectra.npz")
        for grouping in ("absolute", "reduced"):
            for task in ("edos", "phdos"):
                filtered = {}
                for label in LABELS:
                    summary, filtered[label] = group_summary(pairs[label, split, grouping, task], task)
                    structure["summaries"][f"{label}_{split}_{grouping}_{task}"] = summary
                structure["I2_minus_I1"][f"{split}_{grouping}_{task}"] = paired_group_interval(
                    filtered["I1"], filtered["I2"])
    report.update(schema="i1-bplus-analysis-v1", analysis_complete=True, structural_generalization=structure,
                  parameter_counts=preflight["controls"]["I2"]["initialization"]["parameter_counts"],
                  reference_parameter_counts=preflight["controls"]["I2"]["initialization"]["reference_parameter_counts"])
    if reference_identity(Path(identities["I1"]["run_dir"]))[2] != identities:
        raise ValueError("I1 files changed during analysis")
    if preflight["code_sha256"] != code_fingerprints():
        raise ValueError("I2 source changed during analysis")
    dump(out / "report.json", report)
    return report


def execute(cfg, preflight, out):
    return norm.execute(cfg, preflight, out, analyze_fn=analyze, queue_schema="i1-bplus-queue-v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / DEFAULT_I1_REFERENCE)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--out", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--summarize", action="store_true")
    action.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    _, reference, _ = reference_identity(args.reference)
    cfg = i1_bplus_config(reference, args.tag_prefix)
    if args.summarize:
        print(json.dumps(summarize([cfg]), ensure_ascii=False, indent=2))
    elif args.run or args.analyze:
        if args.preflight is None or args.out is None:
            parser.error("--run/--analyze require --preflight and --out")
        preflight = verify_preflight(args.preflight, args.reference, cfg)
        if args.run:
            execute(cfg, preflight, args.out)
        else:
            analyze(cfg, preflight, args.out / "analysis")
    else:
        print(shlex.join(training_command(cfg)))


if __name__ == "__main__":
    main()
