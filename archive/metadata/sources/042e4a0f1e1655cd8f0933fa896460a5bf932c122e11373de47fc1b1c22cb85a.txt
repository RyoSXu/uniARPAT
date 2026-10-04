"""C2.1b: read-only validation loss and gradient attribution for B7.

This tool deliberately loads only the Q1 ``valid`` split.  It never constructs
a test loader, calls an optimizer, or writes a checkpoint.  Its output can
motivate one new loss hypothesis, but cannot be used as a test-set result.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# Direct ``python tools/eval/...`` execution otherwise puts tools/eval rather
# than the repository root on sys.path.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.builder import ConfigBuilder


def sumnorm_loss_components(pred_raw, target, cov, use_mask, huber_delta):
    """Return unreduced KL, W1, Huber terms using production loss semantics."""
    if use_mask:
        eff = cov.clone()
        eff[eff.sum(dim=-1) == 0] = True
        pred_raw = pred_raw.masked_fill(~eff, float("-inf"))
    logp = F.log_softmax(pred_raw, dim=-1)
    inner = target * (target.clamp_min(1e-12).log() - logp)
    selected = eff if use_mask else torch.ones_like(target, dtype=torch.bool)
    kl = torch.where(selected, inner, torch.zeros_like(inner)).sum(dim=-1)
    prob = logp.exp()
    if use_mask:
        weight = eff.float()
        denom = weight.sum(dim=-1).clamp_min(1)
        w1 = ((prob.cumsum(dim=-1) - target.cumsum(dim=-1)).abs() * weight).sum(dim=-1) / denom
        huber = (F.huber_loss(prob, target, reduction="none", delta=huber_delta) * weight).sum(dim=-1) / denom
    else:
        w1 = (prob.cumsum(dim=-1) - target.cumsum(dim=-1)).abs().mean(dim=-1)
        huber = F.huber_loss(prob, target, reduction="none", delta=huber_delta).mean(dim=-1)
    return {"kl": kl, "w1": w1, "huber": huber, "prob": prob}


def _eta_loss(outputs, edos_max, phdos_max, nvalence, valid_atoms, delta_edos, delta_phdos):
    """Reproduce B7's H1 auxiliary loss without touching model state."""
    if "eta" not in outputs or nvalence is None:
        return None
    eta_true = (phdos_max.clamp_min(1e-12) * delta_phdos /
                (3.0 * valid_atoms.unsqueeze(-1))).clamp(0.0, 1.0)
    gamma_true = (edos_max.clamp_min(1e-12) * delta_edos /
                  nvalence.reshape(-1, 1).clamp_min(1e-12)).clamp(0.0, 1.0)
    target = torch.cat([eta_true, gamma_true], dim=-1)
    finite = torch.isfinite(nvalence.reshape(-1, 1))
    sq_error = (outputs["eta"] - target) ** 2
    return torch.where(finite.expand_as(sq_error), sq_error,
                       torch.zeros_like(sq_error)).mean()


def _normalized_r2(prob, target):
    residual = (prob - target).pow(2).sum(dim=-1)
    centered = target - target.mean(dim=-1, keepdim=True)
    return 1.0 - residual / centered.pow(2).sum(dim=-1).clamp_min(1e-12)


def _target_features(target, task):
    entropy = -(target * target.clamp_min(1e-12).log()).sum(dim=-1) / math.log(target.shape[-1])
    features = {
        "entropy": entropy,
        "peak_mass": target.amax(dim=-1),
        "total_variation": (target[:, 1:] - target[:, :-1]).abs().sum(dim=-1),
    }
    if task == "phdos":
        # P0's first 15 bins have centers below zero.  Negative frequency is a
        # coordinate; this reports its target mass, not a negative DOS value.
        features["negative_frequency_mass"] = target[:, :15].sum(dim=-1)
    return features


def _grad_vector(loss, probes):
    gradients = torch.autograd.grad(loss, list(probes.values()), retain_graph=True,
                                    allow_unused=True)
    out = {}
    for (name, parameter), grad in zip(probes.items(), gradients):
        if grad is None:
            out[name] = torch.zeros_like(parameter).reshape(-1)
        else:
            out[name] = grad.detach().reshape(-1)
    return out


def _cosine(x, y):
    denom = x.norm() * y.norm()
    return float((torch.dot(x, y) / denom.clamp_min(1e-30)).item())


def _summary(series, include_mean=True):
    a = np.asarray(series, dtype=np.float64)
    result = {"median": float(np.median(a)), "p10": float(np.quantile(a, 0.10)),
              "p90": float(np.quantile(a, 0.90))}
    if include_mean:
        result["mean"] = float(np.mean(a))
    return result


def _stratum_summary(samples):
    rows = []
    for task in ("edos", "phdos"):
        task_df = samples[samples["task"] == task]
        for feature in ("entropy", "peak_mass", "total_variation", "negative_frequency_mass", "n_atoms"):
            if feature not in task_df:
                continue
            feature_df = task_df.dropna(subset=[feature])
            if feature_df.empty:
                continue
            lo, hi = feature_df[feature].quantile([0.25, 0.75])
            low_r2 = feature_df.loc[feature_df[feature] <= lo, "normalized_r2"].median()
            high_r2 = feature_df.loc[feature_df[feature] >= hi, "normalized_r2"].median()
            rows.append({"task": task, "feature": feature, "q1_threshold": float(lo),
                         "q3_threshold": float(hi), "q1_median_normalized_r2": float(low_r2),
                         "q4_median_normalized_r2": float(high_r2),
                         "q4_minus_q1": float(high_r2 - low_r2)})
    return pd.DataFrame(rows)


def _require_finite(frame, name, ignore=()):
    numeric = frame.drop(columns=list(ignore), errors="ignore").select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError(f"{name} contains non-finite values")


def run_audit(args):
    with open(args.config) as f:
        saved = yaml.safe_load(f)
    config = saved["config"]
    config["dataset"]["valid"]["data_dir"] = args.data_dir
    builder = ConfigBuilder(**config)
    dataset = builder.get_dataset(split="valid", dos_minmax=True, dos_sumnorm=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model = builder.get_model()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.model["transformer"].load_state_dict(state, strict=True)
    model.to(device)
    transformer = model.model["transformer"]
    transformer.eval()

    probes = {
        "encoder_last_ffn": transformer.encoder.layers[-1].linear2.weight,
        "decoder_last_ffn": transformer.decoder.layers[-1].linear2.weight,
    }
    total_batches = len(loader)
    selected = np.linspace(0, total_batches - 1, num=min(args.gradient_batches, total_batches), dtype=int)
    gradient_batches = set(int(v) for v in selected)
    samples = []
    gradients = []
    offset = 0

    for batch_index, batch in enumerate(loader):
        do_gradient = batch_index in gradient_batches
        context = torch.enable_grad() if do_gradient else torch.no_grad()
        with context:
            (inp, pos, mask, target_e, target_p, _mean_e, _std_e, _min_e, max_e,
             _mean_p, _std_p, _min_p, max_p, cov_e, cov_p, nvalence, x_e, x_p) = model.data_preprocess(batch)
            outputs = transformer(inp, mask, pos, x_e, x_p)
            component_e = sumnorm_loss_components(outputs["edos"], target_e, cov_e, model.use_mask, model.huber_delta)
            component_p = sumnorm_loss_components(outputs["phdos"], target_p, cov_p, model.use_mask, model.huber_delta)
            total_e = component_e["kl"].mean() + model.w_w1 * component_e["w1"].mean() + model.w_huber * component_e["huber"].mean()
            total_p = component_p["kl"].mean() + model.w_w1 * component_p["w1"].mean() + model.w_huber * component_p["huber"].mean()
            valid_atoms = (~mask[:, 2:]).sum(dim=-1).float().clamp_min(1.0)
            eta = _eta_loss(outputs, max_e, max_p, nvalence, valid_atoms,
                            model.delta_edos, model.delta_phdos)

            for task, comp, target in (("edos", component_e, target_e), ("phdos", component_p, target_p)):
                prob = comp["prob"]
                features = _target_features(target, task)
                base = {
                    "sample_index": np.arange(offset, offset + target.shape[0]),
                    "task": task,
                    "loss_kl": comp["kl"].detach().cpu().numpy(),
                    "loss_w1": comp["w1"].detach().cpu().numpy(),
                    "loss_huber": comp["huber"].detach().cpu().numpy(),
                    "loss_total": (comp["kl"] + model.w_w1 * comp["w1"] + model.w_huber * comp["huber"]).detach().cpu().numpy(),
                    "normalized_r2": _normalized_r2(prob, target).detach().cpu().numpy(),
                    "l1_error": (prob - target).abs().sum(dim=-1).detach().cpu().numpy(),
                    "n_atoms": valid_atoms.detach().cpu().numpy(),
                }
                for name, value in features.items():
                    base[name] = value.detach().cpu().numpy()
                samples.extend(pd.DataFrame(base).to_dict(orient="records"))

            if do_gradient:
                losses = {
                    "edos_kl": component_e["kl"].mean(),
                    "edos_w1": component_e["w1"].mean(),
                    "edos_huber": component_e["huber"].mean(),
                    "edos_total": total_e,
                    "phdos_kl": component_p["kl"].mean(),
                    "phdos_w1": component_p["w1"].mean(),
                    "phdos_huber": component_p["huber"].mean(),
                    "phdos_total": total_p,
                }
                if eta is not None:
                    losses["eta"] = eta
                vectors = {name: _grad_vector(loss, probes) for name, loss in losses.items()}
                row = {"batch_index": batch_index, "batch_size": int(target_e.shape[0])}
                for loss_name, probe_vectors in vectors.items():
                    for probe_name, vector in probe_vectors.items():
                        row[f"norm_{probe_name}_{loss_name}"] = float(vector.norm().item())
                for probe_name in probes:
                    pairs = [("edos_total", "phdos_total"), ("edos_kl", "edos_w1"),
                             ("edos_kl", "edos_huber"), ("phdos_kl", "phdos_w1"),
                             ("phdos_kl", "phdos_huber")]
                    if "eta" in vectors:
                        pairs += [("eta", "edos_total"), ("eta", "phdos_total")]
                    for left, right in pairs:
                        row[f"cos_{probe_name}_{left}_vs_{right}"] = _cosine(
                            vectors[left][probe_name], vectors[right][probe_name])
                gradients.append(row)
        offset += target_e.shape[0]
        if (batch_index + 1) % 10 == 0 or batch_index + 1 == total_batches:
            print(f"processed {batch_index + 1}/{total_batches} valid batches", flush=True)

    samples_df = pd.DataFrame(samples)
    gradients_df = pd.DataFrame(gradients)
    strata_df = _stratum_summary(samples_df)
    # This wide table contains phDOS-only negative-frequency mass; its eDOS
    # rows are intentionally N/A, while every quantity applicable to a row
    # must remain finite.
    _require_finite(samples_df, "sample attribution", ignore=("negative_frequency_mass",))
    _require_finite(gradients_df, "gradient attribution")
    _require_finite(strata_df, "stratum attribution")
    os.makedirs(args.results_dir, exist_ok=True)
    samples_path = os.path.join(args.results_dir, "c2_1b_valid_samples.csv")
    gradients_path = os.path.join(args.results_dir, "c2_1b_valid_gradient_batches.csv")
    strata_path = os.path.join(args.results_dir, "c2_1b_valid_strata.csv")
    samples_df.to_csv(samples_path, index=False)
    gradients_df.to_csv(gradients_path, index=False)
    strata_df.to_csv(strata_path, index=False)

    loss_summary = {}
    for task in ("edos", "phdos"):
        frame = samples_df[samples_df["task"] == task]
        loss_summary[task] = {column: _summary(frame[column])
                              for column in ("loss_kl", "loss_w1", "loss_huber", "loss_total", "l1_error")}
        # Near-flat target spectra make a per-sample R² arithmetic mean
        # ill-conditioned; retain robust quantiles only.
        loss_summary[task]["normalized_r2"] = _summary(frame["normalized_r2"], include_mean=False)
    grad_summary = {column: _summary(gradients_df[column]) for column in gradients_df.columns
                    if column.startswith("norm_") or column.startswith("cos_")}
    report = {
        "purpose": "C2.1b read-only validation loss attribution; not a training or test result",
        "split": "valid",
        "n_samples": int(len(dataset)),
        "n_batches": total_batches,
        "gradient_batches": sorted(gradient_batches),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else None,
        "device": str(device),
        "loss_summary": loss_summary,
        "gradient_summary": grad_summary,
        "outputs": {"samples": samples_path, "gradient_batches": gradients_path, "strata": strata_path},
    }
    report_path = os.path.join(args.results_dir, "c2_1b_valid_attribution_summary.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="output/ablation_m1_e9ctl/checkpoint_best.pth")
    parser.add_argument("--config", default="output/ablation_m1_e9ctl/config_used.yaml")
    parser.add_argument("--data-dir", default="./data/train4ARPAT")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-batches", type=int, default=32)
    parser.add_argument("--device", default="auto")
    run_audit(parser.parse_args())


if __name__ == "__main__":
    main()
