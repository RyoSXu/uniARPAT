"""Bounded C1 train-only comparison of the joint objective and pair loss alone.

Keep the archived 16 pairs, best-epoch-34 weights, fresh AdamW, dropout stream
and 600-step budget. Both arms execute the production joint forward followed
by the existing pair forward. The pair-only arm backpropagates the first loss
with zero weight; no production model or training interface is changed.
"""
from __future__ import annotations

import argparse
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

from tools.eval.lg_train_subset_diagnosis import composition_keys_by_position
from tools.eval.local_structure_hard_pair import (
    ARCHIVED_FIT, DEFAULT_PAIRS, DEFAULT_RUN, atom_set_diff, load_context)
from tools.eval.local_structure_small_fit import check_initial_metrics, evaluate, new_model, write_json
from tools.train import setup_ablation_seed
from utils.bucket_batch import crystal_collate
from utils.paths import file_sha256
from utils.zproj_alignment import rng_digest, rng_snapshot, state_hash

HARD_DIAGNOSIS = ROOT / "results/runs/hard_pair_diagnosis_c1_s42_20261012"
ARMS = ("joint", "pair_only")
STEPS = 600
EVAL_EVERY = 100


def objective_step(model, batch, step, pair_budget, objective, audit=False):
    """One update with two matched forwards and the selected loss coefficients.

    Reuse the production accumulation gate to zero the entire original loss,
    including eta/gamma supervision, without copying its implementation.
    Zero gradients keep AdamW's parameter participation and weight decay equal
    between arms; heads outside the pair path receive only weight decay.
    """
    if objective not in ARMS:
        raise ValueError("objective must be joint or pair_only")
    if model.use_amp or model.use_mask or model.grad_clip != 0:
        raise ValueError("this fixed diagnostic requires FP32, no mask and no clipping")
    if objective == "joint":
        if model.pair_aux_arm != "control":
            raise ValueError("joint objective requires the zero-weight pair arm")
        return model.train_one_step(batch, step, pair_batch=batch, pair_budget=pair_budget)
    if model.pair_aux_arm != "candidate" or not model.pair_aux_calibrated:
        raise ValueError("pair-only objective requires its fixed calibrated pair weight")
    # There is one full batch. This call computes the original objective, zeros
    # its gradient and defers the optimizer step to the pair backward below.
    values = model.train_one_step(batch, step, pair_budget=pair_budget,
                                  _accumulation=(True, False, 0.0))
    if audit:
        norms = [float(p.grad.detach().abs().max())
                 for p in model.model["transformer"].parameters() if p.grad is not None]
        if not norms or max(norms) != 0.0:
            raise ValueError("the supposedly removed joint objective leaked gradients")
        values["removed_joint_gradient_max"] = max(norms)
    joint_unweighted = values["loss"]
    loss, _, _, _ = model._pair_forward_loss(batch)
    weighted = model.pair_lambda * loss
    weighted.backward()
    if audit:
        if not any(p.grad is not None and float(p.grad.detach().abs().max()) > 0
                   for p in model.model["transformer"].parameters()):
            raise ValueError("pair-only objective has no update gradient")
    model.optimizer["transformer"].step()
    values.update(loss=float(weighted.detach()), joint_loss_unweighted=joint_unweighted,
                  loss_edos_pair=float(loss.detach()))
    return values


def correct_archived_stages(out):
    """Recompute the atom metrics from saved tensors, preserving old artifacts."""
    report = json.loads((HARD_DIAGNOSIS / "model_report.json").read_text())
    focus = pd.read_csv(HARD_DIAGNOSIS / "focus_pairs.csv")
    corrected, rows = {}, []
    with np.load(HARD_DIAGNOSIS / "stage_tensors.npz", allow_pickle=False) as tensors:
        for state, stages in report["stage_diffs"].items():
            corrected[state] = {}
            for pair in focus.to_dict("records"):
                tag = pair["tag"]
                entry = {}
                row = {"state": state, "pair": tag}
                for stage in ("atom_local_last", "atom_global_last"):
                    left = tensors[f"{state}__{pair['left_id']}__{stage}"]
                    right = tensors[f"{state}__{pair['right_id']}__{stage}"]
                    metric = atom_set_diff(left, right)
                    old = stages["pairs"][tag][stage]["relative_matched_rms"]
                    ratio = old / metric["relative_matched_rms"]
                    if not np.isclose(ratio, np.sqrt(left.shape[1]), rtol=2e-6):
                        raise ValueError("archived distance discrepancy is not sqrt(feature width)")
                    entry[stage] = metric
                    row[f"{stage}_old_rel"] = old
                    row[f"{stage}_corrected_rel"] = metric["relative_matched_rms"]
                row["decoder_edos_rel"] = stages["pairs"][tag]["decoder_edos_tokens"]["relative_rms"]
                corrected[state][tag] = entry
                rows.append(row)
    write_json(out / "archived_stage_correction.json", {
        "source": str(HARD_DIAGNOSIS.relative_to(ROOT)), "units": "per-feature-rms-v2",
        "original_evidence_unchanged": True, "stages": corrected})
    pd.DataFrame(rows).to_csv(out / "archived_stage_correction.csv", index=False)


def run(args):
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.out}")
    if not torch.cuda.is_available() or args.seed != 42 or args.lr != 5e-5:
        raise ValueError("this authorized diagnostic requires CUDA, seed 42 and lr=5e-5")
    if args.max_wall_s <= 0:
        raise ValueError("invalid wall cap")
    args.pairs, args.max_atoms, args.pair_ratio = 16, 40, 1.0
    torch.set_num_threads(1)
    setup_ablation_seed(args.seed)
    start = time.monotonic()
    archived = json.loads((ARCHIVED_FIT / "report.json").read_text())
    saved, params, _, weights, epoch, dataset, selected, eligibility = load_context(args)
    initial_hash = state_hash(weights)
    if initial_hash != archived["initial_state_hash"] or epoch != 34:
        raise ValueError("source weights differ from the archived common best ep34")
    expected_selection = pd.read_csv(ARCHIVED_FIT / "selected_train_pairs.csv")
    pd.testing.assert_frame_equal(selected, expected_selection)
    rows = selected.i.tolist() + selected.j.tolist()
    ids = [str(dataset.material_ids[i]) for i in rows]
    if ids != selected.left_id.tolist() + selected.right_id.tolist():
        raise ValueError("train material identities do not match the locked selection")
    keys = composition_keys_by_position(dataset, rows)["absolute"]
    if len(set(keys)) != 16 or any(keys[i] != keys[i + 16] for i in range(16)):
        raise ValueError("batch must contain one exact-composition pair per group")
    batch = crystal_collate([dataset[i] for i in rows])
    with np.load(ARCHIVED_FIT / "targets.npz", allow_pickle=False) as targets:
        if (not np.array_equal(targets["material_ids"], ids)
                or not np.array_equal(targets["edos"], batch[2].numpy())
                or not np.array_equal(targets["phdos"], batch[3].numpy())):
            raise ValueError("locked train labels or material ordering changed")
    pair_budget = int(len(rows) * batch[0].shape[1] ** 2)
    protected = dict(archived["immutable_inputs"])
    for directory in ("model", "utils"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            protected[str(path.relative_to(ROOT))] = file_sha256(path)
    for path in (Path(__file__), ROOT / "tools/train.py", ARCHIVED_FIT / "report.json",
                 ARCHIVED_FIT / "selected_train_pairs.csv", ARCHIVED_FIT / "targets.npz",
                 HARD_DIAGNOSIS / "artifact_manifest.json"):
        protected[str(path.relative_to(ROOT))] = file_sha256(path)
    old_manifest = json.loads((HARD_DIAGNOSIS / "artifact_manifest.json").read_text())
    for name, digest in old_manifest["files"].items():
        protected[str((HARD_DIAGNOSIS / name).relative_to(ROOT))] = digest
    for name, digest in protected.items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f"protected input/evidence changed before experiment: {name}")
    pair_lambda = float(archived["calibration"]["lambda"])
    args.out.mkdir(parents=True)
    selected.to_csv(args.out / "selected_train_pairs.csv", index=False)
    np.savez(args.out / "targets.npz", material_ids=np.asarray(ids),
             edos=batch[2].numpy(), phdos=batch[3].numpy())
    correct_archived_stages(args.out)
    (args.out / "executed_tool.py.txt").write_text(Path(__file__).read_text())
    report = {
        "scope": "train-only warm-start loss isolation; no valid/test; fixed endpoint",
        "seed": args.seed, "source_epoch": epoch, "initial_state_hash": initial_hash,
        "selection": {**eligibility, "policy": "reuse archived label-independent selection"},
        "batch": {"materials": len(rows), "atoms": int(batch[0].ne(0).sum()),
                  "max_atoms": int(batch[0].shape[1]), "pair_budget": pair_budget},
        "training": {"steps_per_arm": STEPS, "eval_every": EVAL_EVERY, "lr": args.lr,
                     "schedule": "constant", "dropout": params["dropout"], "dtype": "float32",
                     "pair_lambda": pair_lambda, "pair_weight_policy": "reuse archived fixed lambda",
                     "joint": "1 * complete original joint objective + 0 * pair loss",
                     "pair_only": "0 * complete original joint objective + lambda * pair loss",
                     "parameter_policy": "all trainable; zero backward retains common AdamW weight decay",
                     "forwards": "same complete batch, joint then pair, twice per update"},
        "criteria": {"aggregate": "E_mean <= .2*E0_mean and >=75% pairs halve error",
                     "hard_pair": "fixed step600 E <= .2*E0 for each H1/H2, reported separately",
                     "interpretation": "success supports this objective intervention; no unique cause or generalization"},
        "immutable_inputs": protected, "device": torch.cuda.get_device_name(),
        "probe": {}, "arms": {}, "complete": False,
    }
    write_json(args.out / "report.json", report)

    def make_model(objective):
        model = new_model(saved, weights, "control" if objective == "joint" else "candidate", args)
        model.pair_lambda, model.pair_aux_calibrated = pair_lambda, True
        transformer = model.model["transformer"]
        if (state_hash(transformer.state_dict()) != initial_hash
                or not all(p.requires_grad and p.dtype == torch.float32
                           for p in transformer.parameters())):
            raise ValueError("weights, trainability or precision differ from the common start")
        return model

    for arm in ARMS:
        model = make_model(arm)
        optimizer = model.optimizer["transformer"]
        if len(optimizer.param_groups) != 1:
            raise ValueError("diagnostic expects a single shared optimizer group")
        settings = {k: optimizer.param_groups[0][k] for k in ("lr", "betas", "weight_decay", "eps")}
        report["training"]["optimizer"] = settings
        setup_ablation_seed(args.seed)
        model.model["transformer"].train()
        times = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(8):
            torch.cuda.synchronize()
            before = time.monotonic()
            values = objective_step(model, batch, step, pair_budget, arm, audit=step == 0)
            torch.cuda.synchronize()
            times.append(time.monotonic() - before)
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError("nonfinite disposable probe")
        report["probe"][arm] = {"step_s_median": float(np.median(times[2:])),
                                "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20}
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
    estimate = STEPS * sum(x["step_s_median"] for x in report["probe"].values()) * 1.25 + 60
    report["probe"]["estimated_two_arm_s"] = estimate
    write_json(args.out / "report.json", report)
    print(json.dumps({"probe": report["probe"], "batch": report["batch"]}), flush=True)
    if estimate > args.max_wall_s:
        raise RuntimeError("predicted fitting cost exceeds the authorized cap")

    fitting_start = time.monotonic()
    for arm in ARMS:
        model = make_model(arm)
        transformer = model.model["transformer"]
        setup_ablation_seed(args.seed)
        start_rng = rng_snapshot()
        baseline, initial_pairs, initial_edos, initial_phdos = evaluate(model, batch, keys, ids)
        initial_pairs.to_csv(args.out / f"pairs_{arm}_s0.csv", index=False)
        np.savez(args.out / f"predictions_{arm}_s0.npz", edos=initial_edos, phdos=initial_phdos)
        trace = [{"step": 0, **baseline}]
        arm_start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        transformer.train()
        for step in range(1, STEPS + 1):
            if time.monotonic() - fitting_start > args.max_wall_s:
                raise RuntimeError("bounded fitting exceeded wall cap; comparison incomplete")
            values = objective_step(model, batch, step, pair_budget, arm, audit=step == 1)
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError("nonfinite fitting loss")
            if step % EVAL_EVERY == 0:
                metrics, pairs, edos, phdos = evaluate(model, batch, keys, ids)
                trace.append({"step": step, **values, **metrics})
                pd.DataFrame(trace).to_csv(args.out / f"trace_{arm}.csv", index=False)
                pairs.to_csv(args.out / f"pairs_{arm}_s{step}.csv", index=False)
                np.savez(args.out / f"predictions_{arm}_s{step}.npz", edos=edos, phdos=phdos)
                print(json.dumps({"arm": arm, "step": step, **metrics}), flush=True)
        end_rng = rng_snapshot()
        final_hash = state_hash(transformer.state_dict())
        if final_hash == initial_hash:
            raise ValueError("diagnostic did not update weights")
        state_file = f"diagnostic_state_{arm}_s600.pth"
        torch.save({"diagnostic_only": True, "objective": arm, "step": STEPS,
                    "source_initial_hash": initial_hash, "pair_lambda": pair_lambda,
                    "model": transformer.state_dict(),
                    "optimizer": model.optimizer["transformer"].state_dict(),
                    "rng_start": start_rng, "rng_end": end_rng}, args.out / state_file)
        report["arms"][arm] = {
            "initial": baseline, "final": metrics, "rng_start": rng_digest(start_rng),
            "rng_end": rng_digest(end_rng), "final_state_hash": final_hash,
            "state_file": state_file, "state_sha256": file_sha256(args.out / state_file),
            "wall_s": time.monotonic() - arm_start,
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        }
        write_json(args.out / "report.json", report)
        del model, transformer
        gc.collect()
        torch.cuda.empty_cache()
    left, right = (report["arms"][arm] for arm in ARMS)
    if left["rng_start"] != right["rng_start"] or left["rng_end"] != right["rng_end"]:
        raise ValueError("arms consumed different dropout streams")
    report["initial_metrics_check"] = check_initial_metrics(left["initial"], right["initial"])
    for name, digest in protected.items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f"protected input/evidence changed during experiment: {name}")
    if state_hash(weights) != initial_hash:
        raise ValueError("source state changed")
    focus = pd.read_csv(HARD_DIAGNOSIS / "focus_pairs.csv")
    focus_rows = []
    for arm in ARMS:
        pairs = pd.read_csv(args.out / f"pairs_{arm}_s600.csv")
        for pair in focus.to_dict("records"):
            matched = pairs[(pairs.left_id == pair["left_id"]) & (pairs.right_id == pair["right_id"])]
            if len(matched) != 1:
                raise ValueError("focus pair missing from fixed-endpoint evidence")
            focus_rows.append({"arm": arm, "pair": pair["tag"], **matched.iloc[0].to_dict(),
                               "strong_fit": bool(matched.iloc[0].skill >= .8)})
    pd.DataFrame(focus_rows).to_csv(args.out / "focus_results.csv", index=False)
    report.update(complete=True, source_state_unchanged=True, source_files_unchanged=True,
                  wall_s=time.monotonic() - start)
    report["artifacts"] = {p.name: file_sha256(p) for p in sorted(args.out.iterdir())
                           if p.is_file() and p.name != "report.json"}
    write_json(args.out / "report.json", report)
    print(json.dumps({"complete": True, "wall_s": report["wall_s"],
                      "focus_results": focus_rows}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--pair-table", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-wall-s", type=float, default=900)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
