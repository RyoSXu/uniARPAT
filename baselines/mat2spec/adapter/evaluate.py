"""Export Mat2Spec valid predictions and compare them with a project export on the same materials.

  selfcheck : re-score the project's h2 export with this code and compare with its stored summary.
  export    : full-valid predictions of one trained arm -> spectra npz, per-material csv, summary json.
  report    : tables for the arms under --root against the reference export (paired, per material).

Comparability (details in docs/design/design-2026-11-10-mat2spec-baseline.md):
  shape / oracle : comparable for every arm (SumNorm shape; oracle multiplies it by the label sum).
  blind          : Standard-MAE arm directly; "hybrid" = SumNorm-KL shape x total of the Standard-MAE
                   prediction, the rescaling the Mat2Spec README/paper uses. SumNorm arms alone have no total.
Only train/valid are ever loaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.mat2spec.adapter import data as D  # noqa: E402
from baselines.mat2spec.adapter import project_metrics as PM  # noqa: E402
from baselines.mat2spec.adapter import train as T  # noqa: E402

REFERENCE = "results/runs/phdos_head_20261106/analysis/exports/h2_s42"
ARMS = {"sumkl": ("normalized_sum", "KL"), "stdmae": ("standardized", "MAE"),
        "sumwd": ("normalized_sum", "WD"), "maxmse": ("normalized_max", "MSE")}


def export(run_dir, checkpoint, device, allow_precheck=False):
    run_dir = Path(run_dir)
    stored = json.loads((run_dir / "fingerprint.json").read_text())
    config = stored["config"]
    if config["limit_valid"] or (config["limit_train"] and not allow_precheck):
        raise ValueError("refusing to export a precheck run trained or scored on a subset")
    state = torch.load(run_dir / f"{checkpoint}.pth", map_location=device, weights_only=False)
    if json.loads(json.dumps(state["fingerprint"])) != stored:
        raise ValueError("checkpoint and run fingerprint differ")
    args = SimpleNamespace(**config, eval_batch_size=512, data_root=str(ROOT / "data/crystal_q1_v1"))
    task = args.task
    protocol = PM.load_protocol(ROOT / PM.DEFAULT_PROTOCOL)
    if D.file_sha256(ROOT / PM.DEFAULT_PROTOCOL) != stored["protocol_sha256"]:
        raise ValueError("protocol changed since training")
    valid = D.load_split(args.data_root, "valid")
    nbr_index, nbr_dist = D.build_graphs(valid, run_dir.parent / "graphs/graphs_valid.npz")
    graphs = D.PackedGraphs(valid, nbr_index, nbr_dist, D.load_atom_init(T.ATOM_INIT), device)
    labels = T.label_units(valid[task], valid["n_atoms"], args.target_unit).to(device)

    from baselines.mat2spec.port import mat2spec_model as M2S

    margs = T.model_args(args)
    net = M2S.Mat2Spec(margs, None).to(device)
    net.load_state_dict(state["model"])
    prediction = T.predict(net, graphs, labels, args, margs, state["mean"].to(device),
                           state["std"].to(device)).cpu().numpy()
    target = valid[task]
    atoms = valid["n_atoms"]
    blind = None
    if args.label_scaling == "standardized":
        blind = (prediction * (atoms[:, None] if args.target_unit == "per_atom" else 1.0)).astype(np.float32)
    frame, summary = PM.score_task(task, target, atoms, protocol, pred_shape=prediction,
                                   pred_blind=blind, material_ids=valid["material_ids"])
    arrays = {"material_ids": valid["material_ids"], "n_atoms": atoms,
              f"pred_shape_{task}": PM.sumnorm(prediction).astype(np.float32),
              f"tgt_shape_{task}": PM.sumnorm(target).astype(np.float32),
              f"pred_oracle_{task}": PM.oracle_reconstruction(prediction, target),
              f"tgt_{task}": target.astype(np.float32)}
    if blind is not None:
        arrays[f"pred_blind_{task}"] = blind
    out = run_dir / "exports"
    out.mkdir(exist_ok=True)
    np.savez(out / f"{checkpoint}_spectra.npz", **arrays)
    frame.to_csv(out / f"{checkpoint}_samples.csv", index=False)
    summary.update({"run_dir": str(run_dir), "checkpoint": checkpoint, "checkpoint_epoch": int(state["epoch"]),
                    "selection_criterion": state["metrics"].get("project_score_name")
                    if checkpoint == "best_project" else "upstream_mae_x_ori",
                    "config": config, "split": "valid", "precheck_subset_train": bool(config["limit_train"])})
    T.write_json(out / f"{checkpoint}_summary.json", summary)
    return summary


def load_export(prefix):
    with np.load(f"{prefix}_spectra.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def describe(summary, task, mode):
    keys = (("E mean", f"{mode}_energy_{task}_mean"), ("E median", f"{mode}_energy_{task}_median"),
            ("R2 median", f"{mode}_r2_{task}_median"), ("R2 P10", f"{mode}_r2_{task}_p10"),
            ("fail %", f"{mode}_fail_rate_{task}"))
    return {label: summary[key] for label, key in keys if key in summary}


def report(root, checkpoint, reference, out_dir):
    root, out_dir = Path(root), Path(out_dir)
    protocol = PM.load_protocol(ROOT / PM.DEFAULT_PROTOCOL)
    ref = load_export(reference)
    atoms = PM.valid_atom_counts(ref["material_ids"])
    result = {"reference": reference, "checkpoint": checkpoint, "split": "valid", "tasks": {}}
    blind_energy = {"h2": {}, "stdmae": {}, "hybrid": {}}
    lines = []
    for task in PM.TASKS:
        target = ref[f"tgt_{task}"]
        ref_frame, ref_summary = PM.score_task(task, target, atoms, protocol, pred_shape=ref[f"pred_shape_{task}"],
                                               pred_blind=ref[f"pred_blind_{task}"])
        blind_energy["h2"][task] = ref_frame[f"blind_energy_{task}"].to_numpy()
        rows = {"h2": {"summary": ref_summary}}
        exports = {}
        for arm in ARMS:
            prefix = root / f"{task}_{arm}/exports/{checkpoint}"
            if not Path(f"{prefix}_spectra.npz").is_file():
                continue
            arrays = load_export(prefix)
            if arrays["material_ids"].tolist() != ref["material_ids"].tolist():
                raise ValueError(f"{task}_{arm}: materials are not the reference valid cohort in order")
            if not np.allclose(arrays[f"tgt_{task}"], target, rtol=3e-6, atol=2e-6):
                raise ValueError(f"{task}_{arm}: targets differ from the reference export")
            exports[arm] = arrays
            frame, summary = PM.score_task(task, target, atoms, protocol, pred_shape=arrays[f"pred_shape_{task}"],
                                           pred_blind=arrays.get(f"pred_blind_{task}"))
            entry = {"summary": summary, "paired_vs_h2": {"oracle_energy": PM.paired_difference(
                frame[f"oracle_energy_{task}"], ref_frame[f"oracle_energy_{task}"])}}
            if f"blind_energy_{task}" in frame:
                entry["paired_vs_h2"]["blind_energy"] = PM.paired_difference(
                    frame[f"blind_energy_{task}"], ref_frame[f"blind_energy_{task}"])
                blind_energy[arm][task] = frame[f"blind_energy_{task}"].to_numpy()
            rows[f"mat2spec_{arm}"] = entry
        if "sumkl" in exports and "stdmae" in exports:
            total = exports["stdmae"][f"pred_blind_{task}"].astype(np.float64).sum(-1, keepdims=True)
            hybrid = (exports["sumkl"][f"pred_shape_{task}"] * total).astype(np.float32)
            frame, summary = PM.score_task(task, target, atoms, protocol,
                                           pred_shape=exports["sumkl"][f"pred_shape_{task}"], pred_blind=hybrid)
            blind_energy["hybrid"][task] = frame[f"blind_energy_{task}"].to_numpy()
            rows["mat2spec_hybrid"] = {"summary": summary, "paired_vs_h2": {"blind_energy": PM.paired_difference(
                frame[f"blind_energy_{task}"], ref_frame[f"blind_energy_{task}"])}}
        result["tasks"][task] = rows
        lines += [f"### {task} (valid {len(target)}; shape metrics on {ref_summary[f'shape_samples_{task}']} non-zero targets)", "",
                  "| model | mode | E mean | E median | R2 median | R2 P10 | fail % | shape KL | shape W1 | shape TV | dE mean vs h2 [95% CI] |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for name, entry in rows.items():
            summary = entry["summary"]
            for mode in ("blind", "oracle"):
                cells = describe(summary, task, mode)
                if not cells:
                    continue
                paired = entry.get("paired_vs_h2", {}).get(f"{mode}_energy")
                delta = f"{paired['mean']:+.4f} [{paired['ci95'][0]:+.4f}, {paired['ci95'][1]:+.4f}]" if paired else ""
                lines.append(f"| {name} | {mode} | " + " | ".join(f"{cells[k]:.4f}" for k in cells) + " | "
                             + " | ".join(f"{summary[f'shape_{m}_{task}_mean']:.4f}" for m in ("kl", "w1", "tv"))
                             + f" | {delta} |")
        lines.append("")
    result["joint_blind"] = {}
    for name, per_task in blind_energy.items():
        if set(per_task) == set(PM.TASKS):
            joint = .5 * (per_task["edos"] + per_task["phdos"])
            entry = {"balanced_blind_energy": float(.5 * (per_task["edos"].mean() + per_task["phdos"].mean()))}
            if name != "h2":
                entry["paired_vs_h2"] = PM.paired_difference(
                    joint, .5 * (blind_energy["h2"]["edos"] + blind_energy["h2"]["phdos"]))
            result["joint_blind"][name] = entry
    lines += ["### joint blind mean E (0.5 eDOS + 0.5 phDOS)", ""] + [
        f"- {name}: {entry['balanced_blind_energy']:.4f}" + (
            f" (vs h2 {entry['paired_vs_h2']['mean']:+.4f}, 95% CI [{entry['paired_vs_h2']['ci95'][0]:+.4f}, "
            f"{entry['paired_vs_h2']['ci95'][1]:+.4f}])" if "paired_vs_h2" in entry else "")
        for name, entry in result["joint_blind"].items()]
    out_dir.mkdir(parents=True, exist_ok=True)
    T.write_json(out_dir / f"report_{checkpoint}.json", result)
    (out_dir / f"report_{checkpoint}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selfcheck")
    p = sub.add_parser("export")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", default="best_project", choices=("best_project", "best_upstream"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--allow_precheck", action="store_true", help="pipeline test on a train-subset run")
    p = sub.add_parser("report")
    p.add_argument("--root", required=True)
    p.add_argument("--checkpoint", default="best_project", choices=("best_project", "best_upstream"))
    p.add_argument("--reference", default=str(ROOT / REFERENCE))
    p.add_argument("--out_dir", default=None)
    args = parser.parse_args(argv)
    if args.command == "selfcheck":
        rows = PM.selfcheck_reference_export(f"{ROOT / REFERENCE}_spectra.npz", f"{ROOT / REFERENCE}_summary.json",
                                             PM.load_protocol(ROOT / PM.DEFAULT_PROTOCOL), ROOT / "data/crystal_q1_v1")
        for key, value, stored, gap, passed in rows:
            print(f"{'OK  ' if passed else 'FAIL'} {key:44s} recomputed {value:.9g} stored {stored:.9g} |d| {gap:.1e}")
        return 0 if all(row[4] for row in rows) else 1
    if args.command == "export":
        summary = export(args.run_dir, args.checkpoint, torch.device(args.device), args.allow_precheck)
        print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=2))
        return 0
    report(args.root, args.checkpoint, args.reference, args.out_dir or Path(args.root) / "analysis")
    return 0


if __name__ == "__main__":
    sys.exit(main())
