"""固定配对任务，比较 G2 encoder 更新与上轮冻结 encoder 的读出训练。"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
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

from model.heads import global_masked_pool
from run_ablation_experiments import setup_ablation_seed
from tools.eval.edos_slope_pilot_verdict import paired_bootstrap_interval
from tools.eval.g2_frozen_readout_probe import (
    EPOCHS, PAIR_BATCH, SEED, SOURCE, FrozenEdosReadout, contrast_mse,
    group_weights, local_pairs, memory_batch, metric_tables, prepare_pair_plan,
    summarize_results,
)
from tools.eval.g2_structure_path_audit import COUNTS, file_hash, load_split
from utils.ablation_checkpoint import atomic_torch_save
from utils.builder import ConfigBuilder

DESIGN = REPO_ROOT / "docs/design/design-g2-encoder-adaptation-probe.md"
PRIOR = REPO_ROOT / "results/g2_frozen_readout_q1"
PRIOR_RUN = REPO_ROOT / "output/g2_frozen_readout_q1"


def split_encoder_inputs(kwargs):
    """观察真实 encoder 输入，去除 padding 并将边映射到紧凑原子索引。"""
    mask, edges = kwargs["src_key_padding_mask"], kwargs["g2_edges"]
    rows = []
    for i, padded in enumerate(mask):
        keep = ~padded
        atom_indices = torch.where(keep)[0]
        remap = torch.full_like(padded, -1, dtype=torch.long)
        remap[atom_indices] = torch.arange(len(atom_indices), device=mask.device)
        selected = edges["batch"] == i
        dst, src = remap[edges["dst"][selected]], remap[edges["src"][selected]]
        if not len(atom_indices) or (dst < 0).any() or (src < 0).any():
            raise ValueError("empty structure or G2 edge touches padding")
        row = dict(atom_src=kwargs["src"][i, keep],
                   rel_diss=kwargs["rel_diss"][i][keep][:, keep],
                   rel_dirs=kwargs["rel_dirs"][i][keep][:, keep],
                   edge_dst=dst, edge_src=src, edge_dist=edges["distances"][selected])
        rows.append({name: tensor.detach().cpu().clone() for name, tensor in row.items()})
    return rows


def encoder_batch(inputs, indices, device):
    """重复材料各自获得独立 batch 编号，边和原子输入只来自已冻结缓存。"""
    rows = [inputs[int(i)] for i in indices]
    lengths = [len(row["atom_src"]) for row in rows]
    length, batch = max(lengths), len(rows)
    src = torch.zeros(batch, length, rows[0]["atom_src"].shape[-1])
    mask = torch.ones(batch, length, dtype=torch.bool)
    distances = torch.zeros(batch, length, length)
    directions = torch.zeros(batch, length, length, 3)
    for i, (row, n) in enumerate(zip(rows, lengths)):
        src[i, :n], mask[i, :n] = row["atom_src"], False
        distances[i, :n, :n], directions[i, :n, :n] = row["rel_diss"], row["rel_dirs"]
    edges = {name: torch.cat([row[key] for row in rows]).to(device)
             for name, key in (("dst", "edge_dst"), ("src", "edge_src"), ("distances", "edge_dist"))}
    edges["batch"] = torch.cat([torch.full_like(row["edge_src"], i) for i, row in enumerate(rows)]).to(device)
    return dict(src=src.to(device), src_key_padding_mask=mask.to(device),
                rel_diss=distances.to(device), rel_dirs=directions.to(device), g2_edges=edges)


class AdaptiveEdosProbe(nn.Module):
    """只复制现有 encoder／读出／H1，H1 仅用于最终盲推理评估。"""

    def __init__(self, transformer):
        super().__init__()
        if (not transformer.use_g2 or transformer.use_g1 or transformer.use_macro_lattice
                or transformer.scale_mode != "eta"):
            raise ValueError("probe requires the original G2 M1 carrier")
        self.encoder = copy.deepcopy(transformer.encoder).requires_grad_(True)
        self.readout = FrozenEdosReadout(transformer)
        self.eta_head = copy.deepcopy(transformer.eta_head).requires_grad_(False)
        self.eval()

    def forward(self, batch, with_gamma=False):
        memory = self.encoder(**batch)
        mask = batch["src_key_padding_mask"]
        prediction = self.readout(memory, mask)
        if with_gamma:
            gamma = self.eta_head(global_masked_pool(memory, mask))[:, 1]
            return prediction, gamma
        return prediction


def clip_and_step(probe, readout_optimizer, encoder_optimizer):
    """保持原读出裁剪范围，避免 encoder 梯度额外缩放读出更新。"""
    readout_norm = torch.nn.utils.clip_grad_norm_(probe.readout.parameters(), 1., error_if_nonfinite=True)
    encoder_norm = torch.nn.utils.clip_grad_norm_(probe.encoder.parameters(), 1., error_if_nonfinite=True)
    readout_optimizer.step()
    encoder_optimizer.step()
    return float(readout_norm), float(encoder_norm)


def extract_inputs(model, datasets, caches, device):
    transformer = model.model["transformer"]
    captured = []

    def observe(module, args, kwargs):
        captured.extend(split_encoder_inputs(kwargs))

    handle = transformer.encoder.register_forward_pre_hook(observe, with_kwargs=True)
    result = {}
    try:
        with torch.no_grad():
            for split, cache in caches.items():
                order = cache["records"].sample_index.tolist()
                for local, index in enumerate(order):
                    if cache["local_of"][index] != local:
                        raise ValueError("frozen cache ordering mismatch")
                loader = DataLoader(Subset(datasets[split], order), batch_size=32, shuffle=False, num_workers=0)
                rows, factors, offset = [], [], 0
                for batch in loader:
                    processed = model.data_preprocess(batch)
                    captured.clear()
                    output = transformer(processed[0], processed[2], processed[1], processed[-2], processed[-1])
                    n = len(captured)
                    torch.testing.assert_close(processed[3].cpu(), cache["target"][offset:offset + n], rtol=0, atol=0)
                    tv = .5 * (output["edos"].softmax(-1).cpu() - cache["original"][offset:offset + n]).abs().sum(-1)
                    if float(tv.max()) > 1e-5:
                        raise ValueError("source checkpoint no longer reproduces prior predictions")
                    rows.extend(captured)
                    factors.append((processed[15].reshape(-1) / model.delta_edos).float().cpu())
                    offset += n
                result[split] = dict(inputs=rows, blind_factor=torch.cat(factors))
                print(f"cached encoder inputs {split}: {offset} materials", flush=True)
    finally:
        handle.remove()
    return result


@torch.no_grad()
def predict(probe, inputs, device):
    probabilities, gamma = [], []
    for start in range(0, len(inputs["inputs"]), 32):
        indices = range(start, min(start + 32, len(inputs["inputs"])))
        p, g = probe(encoder_batch(inputs["inputs"], indices, device), with_gamma=True)
        probabilities.append(p.cpu())
        gamma.append(g.cpu())
    return torch.cat(probabilities), torch.cat(gamma) * inputs["blind_factor"]


def gradient_equivalence(probe, inputs, cache, plan, device):
    """第一训练 batch 对比旧缓存读出梯度；不执行任何参数更新。"""
    pairs = local_pairs(plan, cache)
    selected = np.random.default_rng(SEED).permutation(len(pairs))[:PAIR_BATCH]
    indices = pairs[selected].reshape(-1)
    a, b = torch.tensor(pairs[selected], device=device).T
    target = cache["target"].to(device)
    weights = torch.tensor(group_weights(plan.reduced_group)[selected], device=device)
    parameters = list(probe.readout.parameters())
    probe.encoder.requires_grad_(False)
    try:
        predictions = [probe.readout(*memory_batch(cache, indices, device)),
                       probe(encoder_batch(inputs["inputs"], indices, device))]
    finally:
        probe.encoder.requires_grad_(True)
    gradients, losses = [], []
    for prediction in predictions:
        p = prediction.reshape(len(selected), 2, -1)
        loss = (contrast_mse(p[:, 0], p[:, 1], target[a], target[b]) * weights).mean()
        gradients.append(torch.cat([g.detach().reshape(-1) for g in torch.autograd.grad(loss, parameters)]))
        losses.append(float(loss.detach()))
    relative = float((gradients[0] - gradients[1]).norm() / gradients[0].norm().clamp_min(1e-12))
    loss_error = abs(losses[0] - losses[1]) / max(abs(losses[0]), 1e-12)
    if relative > 1e-3 or loss_error > 1e-5:
        raise ValueError(f"readout gradient equivalence failed: {relative=}, {loss_error=}")
    return dict(readout_gradient_relative_rms=relative, loss_relative_error=loss_error,
                cached_loss=losses[0], recomputed_loss=losses[1])


def train_joint(probe, inputs, cache, plan, device, run_dir):
    pairs = local_pairs(plan, cache)
    target = cache["target"].to(device)
    weights = torch.tensor(group_weights(plan.reduced_group), device=device)
    options = dict(lr=5e-5, betas=(.9, .99), weight_decay=.01)
    readout_optimizer = torch.optim.AdamW(probe.readout.parameters(), **options)
    encoder_optimizer = torch.optim.AdamW(probe.encoder.parameters(), **options)
    order_rng = np.random.default_rng(SEED)
    history, updates, start = [], 0, time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        order = order_rng.permutation(len(pairs))
        loss_sum, count, max_readout, max_encoder = 0., 0, 0., 0.
        for offset in range(0, len(order), PAIR_BATCH):
            selected = order[offset:offset + PAIR_BATCH]
            indices = pairs[selected].reshape(-1)
            p = probe(encoder_batch(inputs["inputs"], indices, device)).reshape(len(selected), 2, -1)
            a, b = torch.tensor(pairs[selected], device=device).T
            loss = (contrast_mse(p[:, 0], p[:, 1], target[a], target[b]) * weights[selected]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite joint objective")
            readout_optimizer.zero_grad(set_to_none=True)
            encoder_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            r_norm, e_norm = clip_and_step(probe, readout_optimizer, encoder_optimizer)
            updates += 1
            loss_sum += float(loss.detach()) * len(selected)
            count += len(selected)
            max_readout, max_encoder = max(max_readout, r_norm), max(max_encoder, e_norm)
        history.append(dict(arm="joint", epoch=epoch, updates=updates, objective=loss_sum / count,
                            max_readout_gradient_norm=max_readout, max_encoder_gradient_norm=max_encoder,
                            elapsed_seconds=time.monotonic() - start))
        pd.DataFrame(history).to_csv(run_dir / "joint_history.csv", index=False)
        print(f"joint epoch {epoch}/{EPOCHS}: objective={loss_sum / count:.6f}, updates={updates}", flush=True)
    atomic_torch_save(dict(probe=probe.state_dict(), epochs=EPOCHS, updates=updates,
                           source_checkpoint_sha256=file_hash(SOURCE / "checkpoint_latest.pth")),
                      str(run_dir / "joint_final.pth"))
    return pd.DataFrame(history)


def verdict_from_comparisons(comparisons):
    def supports(population, margin, require_ci):
        rows = comparisons.loc[comparisons.population == population]
        expected = {"matched", "original", "shuffled", "zero_contrast"}
        return (len(rows) == 4 and set(rows.reference) == expected
                and bool((rows.relative_improvement >= margin).all())
                and (not require_ci or bool((rows.ci_low > 0).all())))
    if supports("valid_unseen_composition", .05, True):
        return "heldout_joint_adaptation_supported"
    if supports("train", .10, False):
        return "joint_train_learning_only"
    return "joint_adaptation_not_supported"


def summarize_joint(samples, pairs):
    # 复用原汇总口径；丢弃它针对旧 matched 臂生成的比较和结论。
    summary, _, _ = summarize_results(samples, pairs, {"passed": True})
    comparisons = []
    for population, selected in (("train", pairs.split == "train"),
                                 ("valid_unseen_composition", pairs.primary_valid),
                                 ("valid_all", pairs.split == "valid")):
        grouped = pairs.loc[selected].groupby(["arm", "reduced_group"])[["contrast_error_tv", "target_tv"]].mean()
        joint = grouped.loc["joint", "contrast_error_tv"]
        for reference in ("matched", "original", "shuffled", "zero_contrast"):
            baseline = grouped.loc["original", "target_tv"] if reference == "zero_contrast" else grouped.loc[reference, "contrast_error_tv"]
            if not joint.index.equals(baseline.index):
                raise ValueError("composition groups differ across arms")
            low, high = paired_bootstrap_interval(joint.to_numpy(), baseline.to_numpy(), "mean", seed=SEED)
            improvement = float(baseline.mean() - joint.mean())
            comparisons.append(dict(population=population, reference=reference, n_groups=len(joint),
                                    baseline_error=float(baseline.mean()), joint_error=float(joint.mean()),
                                    improvement=improvement, relative_improvement=improvement / max(float(baseline.mean()), 1e-12),
                                    ci_low=low, ci_high=high))
    comparisons = pd.DataFrame(comparisons)
    return summary, comparisons, verdict_from_comparisons(comparisons)


def run_probe(run_dir, output_prefix, device):
    run_dir, output_prefix = Path(run_dir).resolve(), Path(output_prefix).resolve()
    os.chdir(REPO_ROOT)
    if run_dir.exists() or list(output_prefix.parent.glob(output_prefix.name + "*")):
        raise FileExistsError("run or result prefix already exists; refusing to overwrite")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    previous = json.loads(PRIOR.with_suffix(".json").read_text())
    compatibility_path = PRIOR.with_name(PRIOR.name + "_cache_compatibility.json")
    compatibility = json.loads(compatibility_path.read_text())
    cache_path = PRIOR_RUN / "frozen_features_portable.pt"
    if file_hash(cache_path) != compatibility["portable_cache_sha256"]:
        raise ValueError("portable frozen feature cache changed")
    protected = [cache_path, compatibility_path, DESIGN, Path(__file__),
                 PRIOR.with_suffix(".json"), PRIOR_RUN / "matched_final.pth", PRIOR_RUN / "shuffled_final.pth"]
    protected.extend(PRIOR.with_name(PRIOR.name + suffix + ".csv") for suffix in
                     ("_plan", "_samples", "_pairs", "_summary", "_comparisons", "_history"))
    for name, expected in previous["input_sha256"].items():
        if name == "tools/eval/g2_frozen_readout_probe.py":
            executed = PRIOR_RUN / "script_executed.py"
            if file_hash(executed) != expected or file_hash(REPO_ROOT / name) != compatibility["current_script_sha256"]:
                raise ValueError("previous executed script or current portable helper changed")
            protected.extend([executed, REPO_ROOT / name])
        else:
            path = REPO_ROOT / name
            if file_hash(path) != expected:
                raise ValueError(f"previous registered input changed: {name}")
            protected.append(path)
    protected.extend(REPO_ROOT / name for name in (
        "tools/eval/g2_structure_path_audit.py", "tools/eval/edos_slope_pilot_verdict.py",
        "model/transformer.py", "model/heads.py", "model/model.py", "utils/rp_encoding.py",
        "utils/g2_periodic_edges.py", "utils/relative_features.py", "utils/ablation_checkpoint.py",
        "utils/builder.py", "utils/metrics.py", "utils/atom_feature.py", "datasets/dataset.py", "run_ablation_experiments.py"))
    hashes = {str(path.relative_to(REPO_ROOT)): file_hash(path) for path in protected}
    config = yaml.safe_load((SOURCE / "config_used.yaml").read_text())
    cli = config["cli"]
    if (cli["model"], cli["norm"], cli["use_g2"], cli["epochs"], cli["seed"]) != ("M1", "sumnorm", True, 10, 42):
        raise ValueError("unexpected source configuration")
    setup_ablation_seed(42)
    builder = ConfigBuilder(**copy.deepcopy(config["config"]))
    datasets = {split: load_split(builder, split) for split in ("train", "valid")}
    ids = {split: np.load(REPO_ROOT / f"data/train4ARPAT/{split}/{split}_index.npy").astype(str) for split in datasets}
    if any(len(datasets[split]) != COUNTS[split] or len(ids[split]) != COUNTS[split] for split in datasets):
        raise ValueError("unexpected Q1 sizes")
    plan = prepare_pair_plan(pd.read_csv(REPO_ROOT / "results/g2_structure_path_q1_pairs.csv"),
                             {split: data.elements.numpy() for split, data in datasets.items()}, ids)
    pd.testing.assert_frame_equal(plan, pd.read_csv(PRIOR.with_name(PRIOR.name + "_plan.csv")))
    caches = torch.load(cache_path, map_location="cpu", weights_only=True)
    for split, cache in caches.items():
        cache["records"] = pd.DataFrame(cache["records"])
        if not np.array_equal(ids[split][cache["records"].sample_index], cache["records"].mpid):
            raise ValueError("frozen material IDs mismatch")
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, run_dir / "script_executed.py")
    shutil.copyfile(DESIGN, run_dir / "design_executed.md")
    plan.to_csv(run_dir / "pair_plan.csv", index=False)
    metadata = dict(status="running", design=str(DESIGN.relative_to(REPO_ROOT)), input_sha256=hashes,
                    source_epoch=10, epochs=EPOCHS, pair_batch=PAIR_BATCH, seed=SEED, device=str(device),
                    encoder_trained=True, elemental_embedding_trained=False, h1_trained=False,
                    test_evaluated=False, validation_used_for_selection=False,
                    comparison_factor="encoder parameters enabled for optimization",
                    optimizer=dict(type="AdamW", lr=5e-5, betas=[.9, .99], weight_decay=.01,
                                   clip_norm=1., separate_encoder_and_readout=True),
                    torch_version=str(torch.__version__), cuda_version=torch.version.cuda)
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    start = time.monotonic()
    model = builder.get_model()
    checkpoint = torch.load(SOURCE / "checkpoint_latest.pth", map_location="cpu", weights_only=True)
    if (checkpoint.get("epoch"), checkpoint.get("seed"), checkpoint.get("model_name")) != (10, 42, "M1") or checkpoint.get("use_amp", False):
        raise ValueError("unexpected source checkpoint metadata")
    transformer = model.model["transformer"]
    transformer.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device)
    transformer.eval().requires_grad_(False)
    inputs = extract_inputs(model, datasets, caches, device)
    atomic_torch_save(inputs, str(run_dir / "encoder_inputs.pt"))
    input_cache_hash = file_hash(run_dir / "encoder_inputs.pt")
    probe = AdaptiveEdosProbe(transformer).to(device)
    initial_state = {name: value.detach().cpu().clone() for name, value in probe.state_dict().items()}
    del model, transformer, datasets, builder
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    equivalence = {}
    for split in inputs:
        prediction, blind_scale = predict(probe, inputs[split], device)
        max_tv = float(.5 * (prediction - caches[split]["original"]).abs().sum(-1).max())
        expected_scale = torch.tensor(caches[split]["records"].blind_scale.to_numpy(), dtype=torch.float32)
        scale_error = float(((blind_scale - expected_scale).abs() / expected_scale.abs().clamp_min(1e-12)).max())
        if max_tv > 1e-5 or scale_error > 1e-5:
            raise ValueError(f"initial full-chain equivalence failed: {split} {max_tv=} {scale_error=}")
        equivalence[split] = dict(max_prediction_tv=max_tv, max_relative_blind_scale_error=scale_error)
        print(f"initial equivalence {split}: {equivalence[split]}", flush=True)
    train_plan = plan.loc[plan.split == "train"]
    equivalence["first_train_batch"] = gradient_equivalence(probe, inputs["train"], caches["train"], train_plan, device)
    print(f"readout gradient equivalence: {equivalence['first_train_batch']}", flush=True)
    (run_dir / "preflight.json").write_text(json.dumps(equivalence, indent=2) + "\n")
    setup_ablation_seed(42)
    history = train_joint(probe, inputs["train"], caches["train"], train_plan, device, run_dir)
    predictions, updated_caches, scales = {"joint": {}}, {}, {}
    for split, cached_inputs in inputs.items():
        prediction, blind_scale = predict(probe, cached_inputs, device)
        predictions["joint"][split] = prediction
        scales[split] = blind_scale
        updated_caches[split] = {**caches[split], "records": caches[split]["records"].copy()}
        updated_caches[split]["records"]["blind_scale"] = blind_scale.numpy()
    atomic_torch_save(dict(predictions=predictions["joint"], blind_scale=scales), str(run_dir / "joint_predictions.pt"))
    joint_samples, joint_pairs = metric_tables(predictions, updated_caches, plan)
    samples = pd.concat([pd.read_csv(PRIOR.with_name(PRIOR.name + "_samples.csv")), joint_samples], ignore_index=True)
    pairs = pd.concat([pd.read_csv(PRIOR.with_name(PRIOR.name + "_pairs.csv")), joint_pairs], ignore_index=True)
    summary, comparisons, verdict = summarize_joint(samples, pairs)
    changes = {}
    for prefix in ("encoder", "readout", "eta_head"):
        before = torch.cat([v.reshape(-1) for k, v in initial_state.items() if k.startswith(prefix + ".")])
        after = torch.cat([v.detach().cpu().reshape(-1) for k, v in probe.state_dict().items() if k.startswith(prefix + ".")])
        changes[prefix] = dict(changed_values=int((before != after).sum()),
                               relative_l2=float((after - before).norm() / before.norm().clamp_min(1e-12)))
    if changes["eta_head"]["changed_values"] or not all(changes[p]["changed_values"] > 0 for p in ("encoder", "readout")):
        raise RuntimeError("unexpected parameter update boundary")
    for path, expected in hashes.items():
        if file_hash(REPO_ROOT / path) != expected:
            raise RuntimeError(f"protected input changed: {path}")
    if file_hash(run_dir / "encoder_inputs.pt") != input_cache_hash:
        raise RuntimeError("encoder input cache changed")
    tables = dict(samples=samples, pairs=pairs, summary=summary, comparisons=comparisons, history=history, plan=plan)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
            raise FloatingPointError(f"nonfinite {name} results")
        with output_prefix.with_name(output_prefix.name + "_" + name + ".csv").open("x") as stream:
            frame.to_csv(stream, index=False)
    metadata.update(status="complete", verdict=verdict, elapsed_seconds=time.monotonic() - start,
                    updates=int(history.iloc[-1].updates), equivalence=equivalence, parameter_changes=changes,
                    encoder_input_cache_sha256=input_cache_hash,
                    trainable_parameters={key: sum(p.numel() for p in getattr(probe, key).parameters()) for key in ("encoder", "readout")},
                    peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                    limitations="One initialization and training seed; paired contrast objective does not anchor absolute spectra; phDOS not validated.")
    with output_prefix.with_suffix(".json").open("x") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    (run_dir / "completed.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(comparisons.to_string(index=False), flush=True)
    print(f"verdict: {verdict}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "output/g2_encoder_adaptation_q1")
    parser.add_argument("--output-prefix", type=Path, default=REPO_ROOT / "results/g2_encoder_adaptation_q1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_probe(args.run_dir, args.output_prefix, torch.device(args.device))


if __name__ == "__main__":
    main()
