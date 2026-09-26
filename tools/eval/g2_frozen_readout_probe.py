"""冻结 G2 encoder，训练原 eDOS 读出预测同组成结构谱差。"""

from __future__ import annotations

import argparse
import copy
from functools import reduce
from math import gcd
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_ablation_experiments import setup_ablation_seed
from tools.eval.edos_slope_pilot_verdict import paired_bootstrap_interval
from tools.eval.g2_structure_path_audit import (
    COUNTS, FeatureProbe, contrast_metrics, file_hash, load_split,
)
from utils.ablation_checkpoint import atomic_torch_save
from utils.builder import ConfigBuilder
from utils.metrics import per_sample_spectral_metrics

EPOCHS = 10
PAIR_BATCH = 16
SEED = 20260926
SOURCE = REPO_ROOT / "output/ablation_m1_g2edge"
DESIGN = REPO_ROOT / "docs/design/design-g2-frozen-readout-probe.md"


def composition_keys(elements):
    """返回绝对计数和约化计数组，均只使用元素输入。"""
    absolute, reduced = [], []
    for row in np.asarray(elements):
        z, counts = np.unique(row[2:][row[2:] != 0], return_counts=True)
        divisor = reduce(gcd, counts.tolist())
        absolute.append("|".join(f"{a}:{b}" for a, b in zip(z, counts)))
        reduced.append("|".join(f"{a}:{b // divisor}" for a, b in zip(z, counts)))
    return np.asarray(absolute), np.asarray(reduced)


def prepare_pair_plan(reference, elements, ids):
    frame = reference.loc[reference.arm == "edge"].copy()
    frame = frame[["split", "group", "sample_index_a", "sample_index_b", "mpid_a", "mpid_b"]]
    keys = {split: composition_keys(elements[split]) for split in ("train", "valid")}
    train_reduced = set(keys["train"][1])
    rows = []
    for row in frame.to_dict("records"):
        split, a, b = row["split"], row["sample_index_a"], row["sample_index_b"]
        absolute, reduced = keys[split]
        if absolute[a] != row["group"] or absolute[b] != row["group"]:
            raise ValueError("pair composition does not match frozen cache")
        if ids[split][a] != row["mpid_a"] or ids[split][b] != row["mpid_b"]:
            raise ValueError("pair IDs do not match frozen cache")
        row["reduced_group"] = reduced[a]
        row["primary_valid"] = split == "valid" and reduced[a] not in train_reduced
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["split", "sample_index_a", "sample_index_b"]).reset_index(drop=True)
    if result.duplicated(["split", "sample_index_a", "sample_index_b"]).any():
        raise ValueError("duplicate material pair")
    if set(result.loc[result.primary_valid, "reduced_group"]) & train_reduced:
        raise ValueError("primary validation composition leaked into encoder training")
    return result


def group_weights(groups):
    groups = pd.Series(groups)
    counts = groups.value_counts()
    return (len(groups) / (len(counts) * groups.map(counts))).to_numpy(dtype=np.float32)


def shuffled_donors(indices, group_of, members, rng):
    """每次调用使用全组均匀置换，保留固定点及 batch 内的一致映射。"""
    mapping = {}
    for group in sorted({group_of[int(index)] for index in indices}):
        source = np.asarray(members[group])
        mapping.update(zip(source.tolist(), rng.permutation(source).tolist()))
    return np.asarray([mapping[int(index)] for index in indices], dtype=np.int64)


class FrozenEdosReadout(nn.Module):
    """复制既有 eDOS 调用链；不持有 encoder 或源参数的共享引用。"""

    def __init__(self, transformer):
        super().__init__()
        unsupported = ("decoupled_decoder", "use_gated_cross_attn", "q1_coord", "r1a_point", "r1b_coord", "c5_moe")
        if any(getattr(transformer, name, False) for name in unsupported) or transformer.edos_energy is not None:
            raise ValueError("probe only supports the frozen G2 legacy readout")
        self.decoder = copy.deepcopy(transformer.decoder)
        self.head = copy.deepcopy(transformer.edos_out_head)
        self.query = nn.Parameter(transformer.edos_query_embed.detach().clone())
        self.target = nn.Parameter(transformer.edos_tgt.detach().clone())
        self.requires_grad_(True)
        self.eval()

    def forward(self, memory, mask):
        batch = len(memory)
        decoded, _ = self.decoder(
            self.target.unsqueeze(0).expand(batch, -1, -1), memory,
            memory_key_padding_mask=mask,
            query_pos=self.query.unsqueeze(0).expand(batch, -1, -1),
        )
        return self.head(decoded.transpose(1, 2)).squeeze(1).float().softmax(-1)


def contrast_mse(pred_a, pred_b, target_a, target_b):
    error = (pred_a - pred_b) - (target_a - target_b)
    return (error * error.shape[-1]).square().mean(-1)


def positive_logit_control(targets, pairs, steps=1000):
    """目标／softmax 损失正对照，不代表 decoder 的优化已充分。"""
    selection = np.asarray(pairs[:16], dtype=np.int64)
    indices, inverse = np.unique(selection, return_inverse=True)
    pair_indices = torch.tensor(inverse.reshape(-1, 2), dtype=torch.long)
    target = targets[indices].detach().cpu()
    logits = nn.Parameter(torch.zeros_like(target))
    optimizer = torch.optim.Adam([logits], lr=0.1)

    def objective():
        p = logits.softmax(-1)
        a, b = pair_indices.T
        return contrast_mse(p[a], p[b], target[a], target[b]).mean()

    initial = float(objective().detach())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        optimizer.step()
    final = float(objective().detach())
    return dict(initial_mse=initial, final_mse=final, final_to_initial=final / max(initial, 1e-12),
                steps=steps, passed=initial > 0 and final <= initial * 0.01)


def extract_cache(model, datasets, ids, plan, device):
    transformer = model.model["transformer"]
    probe = FeatureProbe(transformer)
    cache = {}
    try:
        with torch.no_grad():
            for split in ("train", "valid"):
                pairs = plan.loc[plan.split == split]
                order = sorted(set(pairs.sample_index_a) | set(pairs.sample_index_b))
                loader = DataLoader(Subset(datasets[split], order), batch_size=32, shuffle=False, num_workers=0)
                memories, targets, originals, records = [], [], [], []
                offset = 0
                for batch in loader:
                    processed = model.data_preprocess(batch)
                    outputs, features = probe.forward(processed)
                    keep = ~processed[2][:, 2:]
                    probabilities = outputs["edos"].softmax(-1)
                    for local in range(len(keep)):
                        index = order[offset + local]
                        memories.append(features["encoder"][local, keep[local]].float().cpu().clone())
                        targets.append(processed[3][local].float().cpu().clone())
                        originals.append(probabilities[local].float().cpu().clone())
                        records.append(dict(sample_index=index, mpid=str(ids[split][index]),
                                            oracle_min=float(processed[7][local].item()),
                                            oracle_scale=float((processed[8][local] - processed[7][local]).item()),
                                            blind_scale=float((processed[15][local].reshape(()) * outputs["eta"][local, 1] / model.delta_edos).item())))
                    offset += len(keep)
                cache[split] = dict(memories=memories, target=torch.stack(targets), original=torch.stack(originals),
                                    records=pd.DataFrame(records), local_of={index: i for i, index in enumerate(order)})
                print(f"cached {split}: {offset} materials", flush=True)
    finally:
        probe.close()
    return cache


def memory_batch(cache, indices, device):
    tensors = [cache["memories"][int(i)] for i in indices]
    length = max(len(value) for value in tensors)
    memory = torch.zeros(len(tensors), length, tensors[0].shape[-1], dtype=torch.float32, device=device)
    mask = torch.ones(len(tensors), length, dtype=torch.bool, device=device)
    for i, value in enumerate(tensors):
        memory[i, :len(value)] = value.to(device)
        mask[i, :len(value)] = False
    return memory, mask


def predict(readout, cache, device):
    result = []
    readout.eval()
    with torch.no_grad():
        for start in range(0, len(cache["memories"]), 32):
            indices = range(start, min(start + 32, len(cache["memories"])))
            result.append(readout(*memory_batch(cache, indices, device)).cpu())
    return torch.cat(result)


def local_pairs(plan, cache):
    return np.asarray([[cache["local_of"][int(a)], cache["local_of"][int(b)]]
                       for a, b in plan[["sample_index_a", "sample_index_b"]].to_numpy()], dtype=np.int64)


def train_arm(readout, cache, plan, arm, device, run_dir):
    pairs = local_pairs(plan, cache)
    weights = torch.tensor(group_weights(plan.reduced_group), device=device)
    target = cache["target"].to(device)
    members, group_of = {}, {}
    for group, frame in plan.groupby("group"):
        members[group] = sorted({cache["local_of"][int(index)] for index in
                                 set(frame.sample_index_a) | set(frame.sample_index_b)})
        group_of.update({index: group for index in members[group]})
    order_rng, donor_rng = np.random.default_rng(SEED), np.random.default_rng(SEED + 1)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=5e-5, betas=(0.9, 0.99), weight_decay=0.01)
    readout.eval()
    history, updates = [], 0
    start = time.monotonic()
    history_path = run_dir / f"{arm}_history.csv"
    for epoch in range(1, EPOCHS + 1):
        order = order_rng.permutation(len(pairs))
        loss_sum, count, max_grad = 0.0, 0, 0.0
        for offset in range(0, len(order), PAIR_BATCH):
            selected = order[offset:offset + PAIR_BATCH]
            batch_pairs = pairs[selected]
            indices = batch_pairs.reshape(-1)
            donors = shuffled_donors(indices, group_of, members, donor_rng) if arm == "shuffled" else indices
            p = readout(*memory_batch(cache, donors, device)).reshape(len(selected), 2, -1)
            a, b = torch.tensor(batch_pairs, device=device).T
            loss = (contrast_mse(p[:, 0], p[:, 1], target[a], target[b]) * weights[selected]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite {arm} objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            loss_sum += float(loss.detach()) * len(selected)
            count += len(selected)
            max_grad = max(max_grad, float(norm))
        history.append(dict(arm=arm, epoch=epoch, updates=updates, objective=loss_sum / count,
                            max_gradient_norm=max_grad, elapsed_seconds=time.monotonic() - start))
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"{arm} epoch {epoch}/{EPOCHS}: objective={loss_sum / count:.6f}, updates={updates}", flush=True)
    atomic_torch_save({"readout": readout.state_dict(), "arm": arm, "epochs": EPOCHS,
                       "updates": updates, "source_checkpoint_sha256": file_hash(SOURCE / "checkpoint_latest.pth")},
                      str(run_dir / f"{arm}_final.pth"))
    return history


def metric_tables(predictions, caches, plan):
    pair_rows, sample_frames = [], []
    for arm, split_predictions in predictions.items():
        for split, prediction in split_predictions.items():
            cache = caches[split]
            records = cache["records"].copy()
            target = cache["target"]
            scale = torch.tensor(records.oracle_scale.to_numpy(), dtype=torch.float32).unsqueeze(-1)
            minimum = torch.tensor(records.oracle_min.to_numpy(), dtype=torch.float32).unsqueeze(-1)
            blind = torch.tensor(records.blind_scale.to_numpy(), dtype=torch.float32).unsqueeze(-1)
            truth = target * scale + minimum
            records["r2_edos_oracle"] = per_sample_spectral_metrics((prediction * scale + minimum).clamp_min(0), truth)["r2"].numpy()
            records["r2_edos_blind"] = per_sample_spectral_metrics((prediction * blind).clamp_min(0), truth)["r2"].numpy()
            records["split"], records["arm"] = split, arm
            sample_frames.append(records)
            for row in plan.loc[plan.split == split].to_dict("records"):
                a, b = cache["local_of"][row["sample_index_a"]], cache["local_of"][row["sample_index_b"]]
                metrics = contrast_metrics(prediction[a].numpy(), prediction[b].numpy(), target[a].numpy(), target[b].numpy())
                metrics["contrast_mse"] = float(contrast_mse(prediction[a], prediction[b], target[a], target[b]))
                pair_rows.append(dict(arm=arm, **row, **metrics))
    return pd.concat(sample_frames, ignore_index=True), pd.DataFrame(pair_rows)


def verdict_from_comparisons(comparisons, positive_control):
    def supports(population, margin, require_ci):
        rows = comparisons[comparisons.population == population]
        return len(rows) == 3 and bool((rows.relative_improvement >= margin).all()) and (
            not require_ci or bool((rows.ci_low > 0).all()))
    if not positive_control["passed"]:
        return "invalid_positive_control"
    if supports("valid_unseen_composition", 0.05, True):
        return "heldout_recoverability_supported"
    if supports("train", 0.10, False):
        return "train_learning_only"
    return "readout_intervention_not_supported"


def summarize_results(samples, pairs, positive_control):
    populations = {
        "train": pairs[pairs.split == "train"],
        "valid_unseen_composition": pairs[pairs.primary_valid],
        "valid_all": pairs[pairs.split == "valid"],
    }
    summaries, comparisons = [], []
    for population, frame in populations.items():
        grouped = {}
        for arm, rows in frame.groupby("arm"):
            group = rows.groupby("reduced_group")[["contrast_error_tv", "contrast_mse", "target_tv"]].mean()
            grouped[arm] = group
            selected = set(rows.sample_index_a) | set(rows.sample_index_b)
            split = "train" if population == "train" else "valid"
            subset = samples[(samples.arm == arm) & (samples.split == split) & samples.sample_index.isin(selected)]
            row = dict(population=population, arm=arm, n_pairs=len(rows), n_groups=len(group), n_samples=len(subset),
                       error_tv_group_mean=float(group.contrast_error_tv.mean()),
                       error_mse_group_mean=float(group.contrast_mse.mean()),
                       target_tv_group_mean=float(group.target_tv.mean()))
            for metric in ("contrast_error_tv", "target_tv", "predicted_tv", "contrast_cosine"):
                row[metric + "_pair_median"] = float(rows[metric].median())
            for metric in ("r2_edos_oracle", "r2_edos_blind"):
                row[metric + "_median"] = float(subset[metric].median())
                row[metric + "_fail_pct"] = float(100 * (subset[metric] < 0).mean())
            summaries.append(row)
        matched = grouped["matched"].contrast_error_tv
        for reference in ("original", "shuffled", "zero_contrast"):
            baseline = grouped["original"].target_tv if reference == "zero_contrast" else grouped[reference].contrast_error_tv
            if not baseline.index.equals(matched.index):
                raise ValueError("comparison composition groups do not match")
            low, high = paired_bootstrap_interval(matched.to_numpy(), baseline.to_numpy(), "mean", seed=SEED)
            improvement = float(baseline.mean() - matched.mean())
            comparisons.append(dict(population=population, reference=reference, n_groups=len(matched),
                                    baseline_error=float(baseline.mean()), matched_error=float(matched.mean()),
                                    improvement=improvement, relative_improvement=improvement / max(float(baseline.mean()), 1e-12),
                                    ci_low=low, ci_high=high))
    comparisons = pd.DataFrame(comparisons)
    return pd.DataFrame(summaries), comparisons, verdict_from_comparisons(comparisons, positive_control)


def run_probe(run_dir, output_prefix, device):
    run_dir, output_prefix = Path(run_dir).resolve(), Path(output_prefix).resolve()
    os.chdir(REPO_ROOT)
    if run_dir.exists() or list(output_prefix.parent.glob(output_prefix.name + "*")):
        raise FileExistsError("run or result prefix already exists; refusing to overwrite")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    setup_ablation_seed(42)
    reference_path = REPO_ROOT / "results/g2_structure_path_q1_pairs.csv"
    config_path, checkpoint_path = SOURCE / "config_used.yaml", SOURCE / "checkpoint_latest.pth"
    protected = [reference_path, config_path, checkpoint_path, DESIGN, Path(__file__)]
    config = yaml.safe_load(config_path.read_text())
    cli = config["cli"]
    if (cli["model"], cli["norm"], cli["use_g2"], cli["epochs"], cli["seed"]) != ("M1", "sumnorm", True, 10, 42):
        raise ValueError("unexpected frozen source configuration")
    builder = ConfigBuilder(**copy.deepcopy(config["config"]))
    datasets, ids, elements = {}, {}, {}
    for split in ("train", "valid"):
        datasets[split] = load_split(builder, split)
        index = REPO_ROOT / f"data/train4ARPAT/{split}/{split}_index.npy"
        ids[split] = np.load(index).astype(str)
        elements[split] = datasets[split].elements.numpy()
        if len(ids[split]) != COUNTS[split] or len(datasets[split]) != COUNTS[split]:
            raise ValueError("frozen Q1 split sizes do not match")
        protected.append(index)
        for name in ("elements", "positions", "edos_tgtdos", "nvalence"):
            protected.append(REPO_ROOT / f"data/train4ARPAT/{split}/{name}_{split}.npy")
    plan = prepare_pair_plan(pd.read_csv(reference_path), elements, ids)
    train_plan = plan.loc[plan.split == "train"]
    primary = plan.loc[plan.primary_valid]
    observed = (len(train_plan), len(plan) - len(train_plan), len(primary), primary.reduced_group.nunique())
    if observed != (2591, 320, 163, 117) or train_plan.reduced_group.nunique() != 1198:
        raise ValueError(f"preregistered populations do not match: {observed}")
    hashes = {str(path.relative_to(REPO_ROOT)): file_hash(path) for path in protected}
    run_dir.mkdir(parents=True, exist_ok=False)
    plan.to_csv(run_dir / "pair_plan.csv", index=False)
    metadata = dict(status="running", design=str(DESIGN.relative_to(REPO_ROOT)), source_epoch=10,
                    epochs=EPOCHS, pair_batch=PAIR_BATCH, seed=SEED, device=str(device),
                    input_sha256=hashes, encoder_trained=False, test_evaluated=False,
                    validation_used_for_selection=False, comparison_factor="training feature-target correspondence")
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    start = time.monotonic()
    model = builder.get_model()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint.get("epoch"), checkpoint.get("seed"), checkpoint.get("model_name")) != (10, 42, "M1") or checkpoint.get("use_amp", False):
        raise ValueError("unexpected source checkpoint metadata")
    transformer = model.model["transformer"]
    transformer.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device)
    transformer.eval().requires_grad_(False)
    caches = extract_cache(model, datasets, ids, plan, device)
    prototype = FrozenEdosReadout(transformer).cpu()
    del model, transformer, datasets, builder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # DataFrames are stored as records so the cache remains weights_only-loadable.
    serial_cache = {split: {**cache, "records": cache["records"].to_dict("records")} for split, cache in caches.items()}
    cache_path = run_dir / "frozen_features.pt"
    atomic_torch_save(serial_cache, str(cache_path))
    cache_hash = file_hash(cache_path)
    positive = positive_logit_control(caches["train"]["target"], local_pairs(train_plan, caches["train"]))
    metadata["positive_control"] = positive
    print(f"positive logit control: {positive}", flush=True)
    if not positive["passed"]:
        raise RuntimeError("positive logit control did not meet its preregistered gate")
    original = prototype.to(device)
    for split, cache in caches.items():
        reproduced = predict(original, cache, device)
        maximum = float(0.5 * (reproduced - cache["original"]).abs().sum(-1).max())
        if maximum > 1e-5:
            raise ValueError(f"cached readout failed source equivalence: {split} {maximum}")
        metadata[f"{split}_equivalence_max_tv"] = maximum
        print(f"cached readout equivalence {split}: max TV={maximum:.3g}", flush=True)
    prototype.cpu()
    predictions = {"original": {split: cache["original"] for split, cache in caches.items()}}
    history = []
    for arm in ("matched", "shuffled"):
        setup_ablation_seed(42)
        readout = copy.deepcopy(prototype).to(device)
        history.extend(train_arm(readout, caches["train"], train_plan, arm, device, run_dir))
        predictions[arm] = {split: predict(readout, cache, device) for split, cache in caches.items()}
        del readout
        if device.type == "cuda":
            torch.cuda.empty_cache()
    samples, pairs = metric_tables(predictions, caches, plan)
    summary, comparisons, verdict = summarize_results(samples, pairs, positive)
    for path, expected in hashes.items():
        if file_hash(REPO_ROOT / path) != expected:
            raise RuntimeError(f"protected input changed: {path}")
    if file_hash(cache_path) != cache_hash:
        raise RuntimeError("frozen feature cache changed")
    tables = dict(samples=samples, pairs=pairs, summary=summary, comparisons=comparisons, history=pd.DataFrame(history), plan=plan)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
            raise FloatingPointError(f"nonfinite {name} results")
        with output_prefix.with_name(output_prefix.name + "_" + name + ".csv").open("x") as stream:
            frame.to_csv(stream, index=False)
    metadata.update(status="complete", verdict=verdict, elapsed_seconds=time.monotonic() - start,
                    cache_sha256=cache_hash, updates_per_arm={arm: rows[-1]["updates"] for arm in ("matched", "shuffled")
                                                           if (rows := [row for row in history if row["arm"] == arm])},
                    trainable_parameters=sum(p.numel() for p in prototype.parameters()),
                    limitations="Frozen representation and one training seed; probe is not a production accuracy candidate.")
    with output_prefix.with_suffix(".json").open("x") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    (run_dir / "completed.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(comparisons.to_string(index=False), flush=True)
    print(f"verdict: {verdict}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "output/g2_frozen_readout_q1")
    parser.add_argument("--output-prefix", type=Path, default=REPO_ROOT / "results/g2_frozen_readout_q1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_probe(args.run_dir, args.output_prefix, torch.device(args.device))


if __name__ == "__main__":
    main()
