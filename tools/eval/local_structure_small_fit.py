"""Bounded train-only fitting check of C1 same-composition eDOS differences.

Two disposable copies of the existing C1 best checkpoint use identical inputs,
fresh AdamW states and dropout RNG. The control computes the same auxiliary
forward/backward with zero weight; the candidate adds the existing signed-pair
loss. This is a warm-start fit witness, not a validation/generalization trial.
No production checkpoint, model contract, split or core formula is modified.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from tools.eval.lg_structure_diagnosis import pair_errors
from tools.eval.lg_train_subset_diagnosis import composition_keys_by_position
from tools.eval.lg_valid_export import load_run, select_best
from tools.train import configure_local_global_execution, setup_ablation_seed
from utils.bucket_batch import crystal_collate
from utils.builder import ConfigBuilder
from utils.paths import file_sha256
from utils.zproj_alignment import rng_digest, rng_snapshot, state_hash

DEFAULT_RUN = ROOT / "results/runs/ablation_m1_lsc_attrib20261009_k2_g6_c1_staged_s42_e35"
DEFAULT_PAIRS = ROOT / "results/runs/stage_readout_diagnosis_c1c2_20261010/pairs_train_absolute.csv"
ARMS = ("control", "candidate")
NEAR_ZERO = 1e-6


def select_pairs(frame, atom_counts, count, seed=42, max_atoms=40):
    """Uniform composition-group sampling, independent of labels/predictions."""
    columns = ["i", "j", "left_id", "right_id", "composition"]
    if not set(columns).issubset(frame):
        raise ValueError("pair table lacks material identities or composition")
    if count < 1 or max_atoms < 1 or frame[columns].duplicated().any():
        raise ValueError("invalid selection budget or duplicate pairs")
    counts = np.asarray(atom_counts)
    indices = frame[["i", "j"]].to_numpy(int)
    if indices.min() < 0 or indices.max() >= len(counts) or (indices[:, 0] == indices[:, 1]).any():
        raise ValueError("pair indices do not address distinct train materials")
    eligible = frame.loc[(counts[indices] <= max_atoms).all(1), columns]
    groups = sorted(eligible.composition.unique())
    if len(groups) < count:
        raise ValueError("too few eligible composition groups")
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(groups, size=count, replace=False).tolist())
    rows = []
    for key in chosen:
        group = eligible[eligible.composition == key].sort_values(["left_id", "right_id"])
        rows.append(group.iloc[int(rng.integers(len(group)))])
    selected = pd.DataFrame(rows).reset_index(drop=True)
    if len(set(selected.i) | set(selected.j)) != 2 * count:
        raise ValueError("selected groups share materials")
    return selected, {"eligible_pairs": len(eligible), "eligible_groups": len(groups)}


def fit_metrics(prediction, target, keys, ids):
    frame = pair_errors(prediction, target, keys, ids)
    informative = frame.target_tv > NEAR_ZERO
    kept = frame.loc[informative]
    if not len(kept):
        raise ValueError("no informative spectral differences in the selected materials")
    error = float(kept.difference_error.mean())
    zero_error = float(kept.target_tv.mean())
    ratio = error / zero_error
    half_improved = float((kept.skill >= .5).mean())
    return {
        "pairs": len(frame), "informative_pairs": len(kept),
        "zero_response_error": zero_error, "pair_error_mean": error,
        "pair_error_over_zero": ratio, "pair_skill_median": float(kept.skill.median()),
        "positive_skill_pairs": int((kept.skill > 0).sum()),
        "half_error_pairs": int((kept.skill >= .5).sum()),
        "strong_fit_pairs": int((kept.skill >= .8).sum()),
        "direction_cosine_mean": float(kept.direction_cosine.mean())
        if kept.direction_cosine.notna().any() else None,
        "individual_edos_tv_mean": float(.5 * np.abs(prediction - target).sum(1).mean()),
        "fit_demonstrated": bool(ratio <= .2 and half_improved >= .75),
    }, frame


def check_initial_metrics(left, right):
    """Exact discrete checks plus FP32 tolerances for scalar summaries.

    Cosine is more sensitive when the starting predicted gap is close to zero.
    Initial weights and RNG are checked separately and must match exactly.
    """
    if left.keys() != right.keys():
        raise ValueError("initial metric fields differ")
    deltas = {}
    for key in left:
        a, b = left[key], right[key]
        if a is None or b is None or isinstance(a, (int, bool)):
            if a != b:
                raise ValueError(f"initial discrete metric differs: {key}")
        else:
            tolerance = 1e-5 if key == "direction_cosine_mean" else 1e-6
            delta = abs(a - b)
            if not np.isfinite(delta) or delta > tolerance:
                raise ValueError(f"initial FP32 metric differs: {key}")
            deltas[key] = delta
    return {"passed": True, "absolute_deltas": deltas,
            "tolerance": "1e-6 scalar summaries; 1e-5 direction cosine; exact counts/flags"}


def evaluate(model, batch, keys, ids):
    transformer = model.model["transformer"]
    was_training = transformer.training
    transformer.eval()
    try:
        with torch.no_grad():
            data = model.data_preprocess(batch)
            outputs = transformer(data[0], data[2], data[1], data[-2], data[-1])
            edos = outputs["edos"].squeeze(1).float().softmax(-1).cpu().numpy()
            phdos = outputs["phdos"].squeeze(1).float().softmax(-1).cpu().numpy()
            target_e = data[3].cpu().numpy()
            target_p = data[4].cpu().numpy()
        metrics, pairs = fit_metrics(edos, target_e, keys, ids)
        metrics["individual_phdos_tv_mean"] = float(.5 * np.abs(phdos - target_p).sum(1).mean())
        return metrics, pairs, edos, phdos
    finally:
        transformer.train(was_training)


def new_model(saved, weights, arm, args):
    model = ConfigBuilder(**deepcopy(saved["config"])).get_model()
    transformer = model.model["transformer"]
    transformer.load_state_dict(weights, strict=True)
    model.to(torch.device("cuda"))
    configure_local_global_execution(transformer)
    # This isolated diagnostic invokes existing loss code directly; it does not
    # change the production runner's historical pair-auxiliary admission rules.
    model.pair_aux_arm = arm
    model.pair_ratio = args.pair_ratio
    model.pair_aux_calibrated = False
    for group in model.optimizer["transformer"].param_groups:
        group["lr"] = args.lr
    if model.optimizer["transformer"].state:
        raise ValueError("diagnostic AdamW must start with empty state")
    return model


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run(args):
    if args.out.exists():
        raise FileExistsError(f"refusing existing output directory: {args.out}")
    if not torch.cuda.is_available() or args.seed != 42:
        raise ValueError("this bounded diagnostic requires CUDA and seed 42")
    if args.steps < 1 or args.eval_every < 1 or args.max_wall_s <= 0 or args.lr <= 0:
        raise ValueError("invalid update/time budget")
    torch.set_num_threads(1)
    setup_ablation_seed(args.seed)
    start = time.monotonic()
    saved, params, contract = load_run(args.run_dir)
    if params.get("local_route") != "C1" or saved["cli"]["norm"] != "sumnorm":
        raise ValueError("the fitting check requires an existing C1 SumNorm run")
    checkpoint_path = args.run_dir / "checkpoint_best.pth"
    immutable = {str(p.relative_to(ROOT)): file_sha256(p)
                 for p in (checkpoint_path, args.run_dir / "config_used.yaml", args.pair_table)}
    picked = select_best(args.run_dir, contract["selection_metric"], saved)
    weights = picked["checkpoint"]["model"]
    epoch = int(picked["checkpoint"]["epoch"])
    del picked
    initial_hash = state_hash(weights)
    dataset = ConfigBuilder(**deepcopy(saved["config"])).get_dataset(
        split="train", dos_minmax=True, dos_sumnorm=True)
    if dataset.augment:
        raise ValueError("fitting diagnostic requires immutable structures")
    train_root = Path(dataset.data_dir).resolve()
    for path in sorted(train_root.iterdir()):
        if path.is_file():
            immutable[str(path.relative_to(ROOT))] = file_sha256(path)
    immutable["data/crystal_q1_v1/manifest.json"] = file_sha256(ROOT / "data/crystal_q1_v1/manifest.json")
    selected, eligibility = select_pairs(pd.read_csv(args.pair_table), dataset.atom_counts,
                                        args.pairs, args.seed, args.max_atoms)
    rows = selected.i.tolist() + selected.j.tolist()
    ids = [str(dataset.material_ids[i]) for i in rows]
    if ids != selected.left_id.tolist() + selected.right_id.tolist():
        raise ValueError("pair materials do not match the current train partition")
    keys = composition_keys_by_position(dataset, rows)["absolute"]
    if any(keys[i] != keys[i + args.pairs] for i in range(args.pairs)):
        raise ValueError("selected pair does not have identical absolute composition")
    if len(set(keys)) != args.pairs:
        raise ValueError("fitting batch must contain one pair per composition")
    batch = crystal_collate([dataset[i] for i in rows])
    pair_budget = int(len(rows) * batch[0].shape[1] ** 2)
    args.out.mkdir(parents=True)
    selected.to_csv(args.out / "selected_train_pairs.csv", index=False)
    np.savez(args.out / "targets.npz", material_ids=np.asarray(ids),
             edos=batch[2].numpy(), phdos=batch[3].numpy())
    report = {
        "scope": "train-only, warm-start disposable C1 fitting; no valid/test evaluation",
        "seed": args.seed, "source_epoch": epoch, "initial_state_hash": initial_hash,
        "selection": {**eligibility, "pairs": args.pairs, "max_atoms": args.max_atoms,
                      "policy": "uniform groups then one pair/group; labels/predictions unused"},
        "batch": {"materials": len(rows), "atoms": int(batch[0].ne(0).sum()),
                  "max_atoms": int(batch[0].shape[1]), "pair_budget": pair_budget},
        "training": {"steps_per_arm": args.steps, "lr": args.lr, "schedule": "constant",
                     "optimizer": "fresh AdamW; betas/weight_decay from source configuration",
                     "dropout": params["dropout"], "pair_gradient_ratio": args.pair_ratio,
                     "pair_weight_policy": "calibrate once in eval mode on the locked train batch",
                     "control": "original objective + zero-weight identical pair forward/backward",
                     "candidate": "original objective + existing edos_pair_contrast_loss",
                     "main_and_pair_batches": "same full selected-material batch; one update per step"},
        "criterion": "pair error <=20% of zero-response error and >=75% pairs halve error",
        "device": torch.cuda.get_device_name(), "immutable_inputs": immutable,
        "probe": {}, "arms": {}, "complete": False,
    }
    calibration = None
    probe_start = time.monotonic()
    for arm in ARMS:
        model = new_model(saved, weights, arm, args)
        if state_hash(model.model["transformer"].state_dict()) != initial_hash:
            raise ValueError("probe did not start from the source weights")
        if calibration is None:
            model._calibrate_edos_pair_weight(batch)
            calibration = dict(model.pair_aux_calibration)
        model.pair_lambda = calibration["lambda"]
        model.pair_aux_calibrated = True
        setup_ablation_seed(args.seed)
        model.model["transformer"].train()
        times = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(8):
            torch.cuda.synchronize()
            before = time.monotonic()
            values = model.train_one_step(batch, step, pair_batch=batch, pair_budget=pair_budget)
            torch.cuda.synchronize()
            times.append(time.monotonic() - before)
            if not np.isfinite(values["loss"]):
                raise ValueError("non-finite disposable probe loss")
        report["probe"][arm] = {"step_s_median": float(np.median(times[2:])),
                                "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report["calibration"] = calibration
    estimate = args.steps * sum(x["step_s_median"] for x in report["probe"].values()) * 1.25 + 60
    report["probe"].update(estimated_two_arm_s=estimate, wall_s=time.monotonic() - probe_start)
    write_json(args.out / "report.json", report)
    print(json.dumps({"probe": report["probe"], "batch": report["batch"],
                      "calibration": calibration}), flush=True)
    if not args.probe_only and estimate > args.max_wall_s:
        raise RuntimeError("predicted fitting time exceeds the authorized short-run cap")
    if not args.probe_only:
        training_start = time.monotonic()
        for arm in ARMS:
            model = new_model(saved, weights, arm, args)
            model.pair_lambda = calibration["lambda"]
            model.pair_aux_calibrated = True
            transformer = model.model["transformer"]
            if state_hash(transformer.state_dict()) != initial_hash:
                raise ValueError("fitting arms did not start with identical source weights")
            setup_ablation_seed(args.seed)
            rng_start = rng_digest(rng_snapshot())
            baseline, _, _, _ = evaluate(model, batch, keys, ids)
            trace = [{"step": 0, **baseline}]
            arm_start = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            transformer.train()
            for step in range(1, args.steps + 1):
                if time.monotonic() - training_start > args.max_wall_s:
                    raise RuntimeError("short-run wall cap reached; fitting comparison incomplete")
                values = model.train_one_step(batch, step, pair_batch=batch, pair_budget=pair_budget)
                if not all(np.isfinite(x) for x in values.values()):
                    raise ValueError("non-finite fitting loss")
                if step % args.eval_every == 0 or step == args.steps:
                    metrics, pairs, edos, phdos = evaluate(model, batch, keys, ids)
                    trace.append({"step": step, "loss": values["loss"], **metrics})
                    pd.DataFrame(trace).to_csv(args.out / f"trace_{arm}.csv", index=False)
                    pairs.to_csv(args.out / f"pairs_{arm}_s{step}.csv", index=False)
                    np.savez(args.out / f"predictions_{arm}_s{step}.npz", edos=edos, phdos=phdos)
                    print(json.dumps({"arm": arm, "step": step, **metrics}), flush=True)
            entry = {"initial": baseline, "final": metrics, "rng_start": rng_start,
                     "rng_end": rng_digest(rng_snapshot()),
                     "final_state_hash": state_hash(transformer.state_dict()),
                     "wall_s": time.monotonic() - arm_start,
                     "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20}
            if entry["final_state_hash"] == initial_hash:
                raise ValueError("diagnostic did not update weights")
            report["arms"][arm] = entry
            write_json(args.out / "report.json", report)
            del model, transformer
            gc.collect()
            torch.cuda.empty_cache()
        if report["arms"]["control"]["rng_start"] != report["arms"]["candidate"]["rng_start"]:
            raise ValueError("arm starting RNG states differ")
        if report["arms"]["control"]["rng_end"] != report["arms"]["candidate"]["rng_end"]:
            raise ValueError("control and candidate consumed different dropout RNG")
        report["initial_metrics_check"] = check_initial_metrics(
            report["arms"]["control"]["initial"], report["arms"]["candidate"]["initial"])
        report["complete"] = True
    for name, expected in immutable.items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"source input/checkpoint changed: {name}")
    report["source_state_unchanged"] = state_hash(weights) == initial_hash
    if not report["source_state_unchanged"]:
        raise ValueError("loaded source state was mutated")
    report["source_files_unchanged"] = True
    report["wall_s"] = time.monotonic() - start
    report["artifacts"] = {p.name: file_sha256(p) for p in sorted(args.out.iterdir())
                           if p.is_file() and p.name != "report.json"}
    write_json(args.out / "report.json", report)
    print(json.dumps({"complete": report["complete"], "wall_s": report["wall_s"],
                      "source_files_unchanged": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--pair-table", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--pair-ratio", type=float, default=1.0)
    parser.add_argument("--max-wall-s", type=float, default=900)
    parser.add_argument("--probe-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
