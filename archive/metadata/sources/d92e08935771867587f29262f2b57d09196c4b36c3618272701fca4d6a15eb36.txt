"""固定16个train材料对，检验冻结／联合G2实际网络能否充分拟合谱差。"""

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
from torch.utils.data import DataLoader, Subset
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_ablation_experiments import setup_ablation_seed
from tools.eval.g2_encoder_adaptation_probe import (
    AdaptiveEdosProbe, clip_and_step, encoder_batch, gradient_equivalence, split_encoder_inputs,
)
from tools.eval.g2_frozen_readout_probe import (
    SEED, SOURCE, composition_keys, contrast_mse, local_pairs, memory_batch,
    metric_tables, positive_logit_control,
)
from tools.eval.g2_structure_path_audit import (
    COUNTS, contrast_metrics, file_hash, load_split, matched_atom_rms,
)
from utils.ablation_checkpoint import atomic_torch_save
from utils.builder import ConfigBuilder

DESIGN = REPO_ROOT / "docs/design/design-g2-small-fit-probe.md"
PLAN = REPO_ROOT / "results/g2_frozen_readout_q1_plan.csv"
N_PAIRS, STEPS, EVAL_EVERY = 16, 2000, 100


def select_train_pairs(plan, n_pairs=N_PAIRS, seed=SEED):
    """按组均匀抽样、组内均匀抽一对；顺序及额外标签列不影响选样。"""
    columns = ["split", "group", "sample_index_a", "sample_index_b", "mpid_a", "mpid_b",
               "reduced_group", "primary_valid"]
    frame = plan.loc[plan.split == "train", columns].copy()
    groups = np.asarray(sorted(frame.reduced_group.unique()))
    if len(groups) < n_pairs or frame.duplicated(["sample_index_a", "sample_index_b"]).any():
        raise ValueError("insufficient groups or duplicate pairs")
    rng, rows = np.random.default_rng(seed), []
    for group in rng.choice(groups, size=n_pairs, replace=False):
        members = frame.loc[frame.reduced_group == group].sort_values(["sample_index_a", "sample_index_b"])
        rows.append(members.iloc[int(rng.integers(len(members)))])
    selected = pd.DataFrame(rows).sort_values(["sample_index_a", "sample_index_b"]).reset_index(drop=True)
    if len(set(selected.sample_index_a) | set(selected.sample_index_b)) != 2 * n_pairs or selected.primary_valid.any():
        raise ValueError("selected materials overlap or contain validation entries")
    return selected


def pair_scores(prediction, target, pairs):
    """逐对残余相对零谱差参考；均值不能替代全部材料对门槛。"""
    rows = []
    for i, (a, b) in enumerate(pairs):
        row = contrast_metrics(prediction[a].numpy(), prediction[b].numpy(), target[a].numpy(), target[b].numpy())
        mse = float(contrast_mse(prediction[a], prediction[b], target[a], target[b]))
        zero_mse = float(((target[a] - target[b]) * target.shape[-1]).square().mean())
        informative = zero_mse > 1e-12 and row["target_tv"] > 1e-12
        mse_ratio, tv_ratio = mse / max(zero_mse, 1e-12), row["contrast_error_tv"] / max(row["target_tv"], 1e-12)
        passed = (mse_ratio <= .01 and tv_ratio <= .1) if informative else (mse <= 1e-12 and row["contrast_error_tv"] <= 1e-6)
        rows.append(dict(pair_index=i, **row, contrast_mse=mse, zero_mse=zero_mse,
                         mse_ratio=mse_ratio, tv_ratio=tv_ratio, informative=informative, passed=passed))
    return pd.DataFrame(rows)


def memory_set_rms(left, right):
    """decoder不接收独立元素标签，因此允许所有原子之间匹配。"""
    dummy = np.zeros(len(left), dtype=np.int64)
    return matched_atom_rms(left.numpy(), right.numpy(), dummy, dummy)


def input_audit(cache, inputs, pairs):
    rows = []
    for i, (a, b) in enumerate(pairs):
        left, right = inputs[a], inputs[b]
        distance_difference = float((left["rel_diss"].flatten().sort().values - right["rel_diss"].flatten().sort().values).abs().max())
        edges_equal = torch.equal(left["edge_dist"].sort().values, right["edge_dist"].sort().values)
        rms = memory_set_rms(cache["memories"][a], cache["memories"][b])
        source_tv = float(.5 * (cache["original"][a] - cache["original"][b]).abs().sum())
        rows.append(dict(pair_index=i, n_atoms=len(left["atom_src"]),
                         distance_multiset_max_difference=distance_difference,
                         g2_distance_multisets_equal=edges_equal, edges_a=len(left["edge_dist"]), edges_b=len(right["edge_dist"]),
                         memory_permutation_relative_rms=rms, exact_memory_alias=rms == 0.,
                         source_prediction_tv=source_tv, source_output_distinct=source_tv > 1e-5))
    return pd.DataFrame(rows)


def extract_selected(model, dataset, ids, selection):
    transformer = model.model["transformer"]
    order = sorted({int(i) for i in set(selection.sample_index_a) | set(selection.sample_index_b)})
    inputs, memories, targets, originals, factors, records = [], [], [], [], [], []

    def observe(module, args, kwargs, output):
        inputs.extend(split_encoder_inputs(kwargs))
        for i, keep in enumerate(~kwargs["src_key_padding_mask"]):
            memories.append(output[i, keep].detach().cpu().clone())

    handle = transformer.encoder.register_forward_hook(observe, with_kwargs=True)
    try:
        with torch.no_grad():
            offset = 0
            for batch in DataLoader(Subset(dataset, order), batch_size=32, shuffle=False, num_workers=0):
                processed = model.data_preprocess(batch)
                output = transformer(processed[0], processed[2], processed[1], processed[-2], processed[-1])
                targets.append(processed[3].float().cpu())
                originals.append(output["edos"].float().softmax(-1).cpu())
                factor = (processed[15].reshape(-1) / model.delta_edos).float()
                factors.append(factor.cpu())
                for i in range(len(processed[0])):
                    index = order[offset + i]
                    records.append(dict(sample_index=index, mpid=str(ids[index]),
                                        oracle_min=float(processed[7][i]), oracle_scale=float(processed[8][i] - processed[7][i]),
                                        blind_scale=float(factor[i] * output["eta"][i, 1])))
                offset += len(processed[0])
    finally:
        handle.remove()
    cache = dict(memories=memories, target=torch.cat(targets), original=torch.cat(originals),
                 records=pd.DataFrame(records), local_of={index: i for i, index in enumerate(order)})
    return cache, inputs, torch.cat(factors)


def checkpoint_payload(probe, arm, step):
    return dict(probe=probe.state_dict(), arm=arm, step=step)


def configure_encoder_scope(probe, arm):
    """只控制可更新参数，保留跨冻结层的反向传播。"""
    if arm not in ("frozen", "joint", "g2_only", "non_g2", "last_layer"):
        raise ValueError("unknown small-fit arm")
    if arm in ("g2_only", "non_g2", "last_layer") and not len(getattr(probe.encoder, "g2_msgs", [])):
        raise ValueError(f"{arm} requires existing G2 message modules")
    if arm == "last_layer" and not len(getattr(probe.encoder, "layers", [])):
        raise ValueError("last_layer requires an existing encoder layer")
    probe.zero_grad(set_to_none=True)
    probe.encoder.requires_grad_(arm in ("joint", "non_g2"))
    if arm == "g2_only":
        probe.encoder.g2_msgs.requires_grad_(True)
    elif arm == "non_g2":
        probe.encoder.g2_msgs.requires_grad_(False)
    elif arm == "last_layer":
        probe.encoder.layers[-1].requires_grad_(True)


def fit_arm(probe, arm, cache, batch, pairs, device, run_dir, steps=STEPS, eval_every=EVAL_EVERY):
    configure_encoder_scope(probe, arm)
    probe.eval()
    memory, mask = memory_batch(cache, range(len(cache["target"])), device)
    target = cache["target"].to(device)
    a, b = torch.tensor(pairs, device=device).T
    options = dict(lr=5e-5, betas=(.9, .99), weight_decay=.01)
    readout_optimizer = torch.optim.AdamW(probe.readout.parameters(), **options)
    encoder_parameters = [p for p in probe.encoder.parameters() if p.requires_grad]
    encoder_optimizer = torch.optim.AdamW(encoder_parameters, **options) if encoder_parameters else None
    history, trajectories, first_fit, start = [], [], None, time.monotonic()
    r_norm, e_norm = 0., 0.

    def forward():
        return probe.readout(memory, mask) if arm == "frozen" else probe(batch)

    def evaluate(step):
        nonlocal first_fit
        with torch.no_grad():
            scores = pair_scores(forward().cpu(), cache["target"], pairs)
        row = dict(arm=arm, step=step, mean_mse=float(scores.contrast_mse.mean()),
                   mean_tv=float(scores.contrast_error_tv.mean()),
                   aggregate_mse_ratio=float(scores.contrast_mse.mean() / max(scores.zero_mse.mean(), 1e-12)),
                   aggregate_tv_ratio=float(scores.contrast_error_tv.mean() / max(scores.target_tv.mean(), 1e-12)),
                   max_mse_ratio=float(scores.mse_ratio.max()), max_tv_ratio=float(scores.tv_ratio.max()),
                   n_pairs_passed=int(scores.passed.sum()), all_pairs_passed=bool(scores.passed.all()),
                   readout_gradient_norm=r_norm, encoder_gradient_norm=e_norm,
                   elapsed_seconds=time.monotonic() - start)
        if row["all_pairs_passed"] and first_fit is None:
            first_fit = step
            atomic_torch_save(checkpoint_payload(probe, arm, step), str(run_dir / f"{arm}_first_fit.pth"))
        history.append(row)
        trajectories.append(scores.assign(arm=arm, step=step))
        pd.DataFrame(history).to_csv(run_dir / f"{arm}_history.csv", index=False)
        pd.concat(trajectories, ignore_index=True).to_csv(run_dir / f"{arm}_pair_history.csv", index=False)
        print(f"{arm} step {step}/{steps}: MSE/zero={row['aggregate_mse_ratio']:.5f}, "
              f"TV/zero={row['aggregate_tv_ratio']:.5f}, passed={row['n_pairs_passed']}/{len(pairs)}", flush=True)

    evaluate(0)
    for step in range(1, steps + 1):
        p = forward()
        loss = contrast_mse(p[a], p[b], target[a], target[b]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite small-fit objective")
        readout_optimizer.zero_grad(set_to_none=True)
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if encoder_optimizer is None:
            r_norm = float(torch.nn.utils.clip_grad_norm_(probe.readout.parameters(), 1., error_if_nonfinite=True))
            readout_optimizer.step()
        else:
            r_norm, e_norm = clip_and_step(probe, readout_optimizer, encoder_optimizer)
        if step % eval_every == 0 or step == steps:
            evaluate(step)
    atomic_torch_save(checkpoint_payload(probe, arm, steps), str(run_dir / f"{arm}_final.pth"))
    return pd.DataFrame(history), pd.concat(trajectories, ignore_index=True), first_fit


def fit_verdict(first_fit):
    frozen, joint = first_fit["frozen"] is not None, first_fit["joint"] is not None
    if frozen and joint:
        return "both_networks_fit_small_train_set"
    if frozen:
        return "frozen_readout_fit_supported"
    if joint:
        return "joint_adaptation_fit_only"
    return "small_set_fit_not_demonstrated"


def state_change(before, module):
    after = module.state_dict()
    changed, delta, base = 0, 0., 0.
    for name, value in before.items():
        current = after[name].detach().cpu()
        changed += int((current != value).sum())
        delta += float((current.float() - value.float()).square().sum())
        base += float(value.float().square().sum())
    return dict(changed_values=changed, relative_l2=(delta / max(base, 1e-24)) ** .5)


def run_probe(run_dir, output_prefix, device):
    os.chdir(REPO_ROOT)
    run_dir, output_prefix = Path(run_dir).resolve(), Path(output_prefix).resolve()
    if run_dir.exists() or list(output_prefix.parent.glob(output_prefix.name + "*")):
        raise FileExistsError("run or result prefix already exists; refusing to overwrite")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    prior_path = REPO_ROOT / "results/g2_encoder_adaptation_q1.json"
    prior = json.loads(prior_path.read_text())
    protected = [DESIGN, Path(__file__), PLAN, prior_path, SOURCE / "config_used.yaml", SOURCE / "checkpoint_latest.pth"]
    protected.extend(sorted((REPO_ROOT / "data/train4ARPAT/train").glob("*.npy")))
    protected.extend(REPO_ROOT / name for name in (
        "data/grids_c2b/grids.json", "tools/eval/g2_encoder_adaptation_probe.py",
        "tools/eval/g2_frozen_readout_probe.py", "tools/eval/g2_structure_path_audit.py",
        "model/transformer.py", "model/model.py", "model/heads.py", "utils/relative_features.py",
        "utils/g2_periodic_edges.py", "utils/rp_encoding.py", "utils/atom_feature.py", "utils/metrics.py",
        "utils/builder.py", "utils/ablation_checkpoint.py", "datasets/dataset.py", "run_ablation_experiments.py"))
    hashes = {str(path.relative_to(REPO_ROOT)): file_hash(path) for path in protected}
    for name, observed in hashes.items():
        expected = prior["input_sha256"].get(name, observed)
        if observed != expected:
            raise ValueError(f"previous registered source changed: {name}")
    config = yaml.safe_load((SOURCE / "config_used.yaml").read_text())
    cli = config["cli"]
    if (cli["model"], cli["norm"], cli["use_g2"], cli["epochs"], cli["seed"], cli["augment"]) != ("M1", "sumnorm", True, 10, 42, False):
        raise ValueError("unexpected source configuration")
    selection = select_train_pairs(pd.read_csv(PLAN))
    setup_ablation_seed(42)
    builder = ConfigBuilder(**copy.deepcopy(config["config"]))
    dataset = load_split(builder, "train")
    ids = np.load(REPO_ROOT / "data/train4ARPAT/train/train_index.npy").astype(str)
    if len(dataset) != COUNTS["train"] or len(ids) != COUNTS["train"]:
        raise ValueError("unexpected Q1 train sizes")
    absolute, reduced = composition_keys(dataset.elements.numpy())
    for row in selection.itertuples():
        a, b = row.sample_index_a, row.sample_index_b
        if (absolute[a], absolute[b], reduced[a], reduced[b], ids[a], ids[b]) != (
                row.group, row.group, row.reduced_group, row.reduced_group, row.mpid_a, row.mpid_b):
            raise ValueError("selected IDs or compositions mismatch")
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, run_dir / "script_executed.py")
    shutil.copyfile(DESIGN, run_dir / "design_executed.md")
    selection.to_csv(run_dir / "selection.csv", index=False)
    metadata = dict(status="running", design=str(DESIGN.relative_to(REPO_ROOT)), input_sha256=hashes,
                    source_epoch=10, initialization_seed=42, selection_seed=SEED, steps_per_arm=STEPS,
                    evaluation_interval=EVAL_EVERY, n_pairs=len(selection), n_materials=2 * len(selection),
                    split="train", valid_evaluated=False, test_evaluated=False, device=str(device),
                    comparison_factor="encoder parameters enabled for optimization", dropout=False,
                    optimizer=dict(type="AdamW", lr=5e-5, betas=[.9, .99], weight_decay=.01,
                                   independent_clip_norm=1.),
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
    cache, inputs, blind_factor = extract_selected(model, dataset, ids, selection)
    pairs = local_pairs(selection, cache)
    audit = input_audit(cache, inputs, pairs).merge(selection.reset_index(names="pair_index"), on="pair_index", validate="one_to_one")
    audit.to_csv(run_dir / "input_audit.csv", index=False)
    print(f"input audit: {int(audit.source_output_distinct.sum())}/{len(audit)} source predictions distinguish pairs; "
          f"{int(audit.exact_memory_alias.sum())} exact memory aliases", flush=True)
    serial_cache = {**cache, "records": cache["records"].to_dict("records")}
    atomic_torch_save(dict(cache=serial_cache, inputs=inputs, blind_factor=blind_factor), str(run_dir / "train_cache.pt"))
    cache_hash = file_hash(run_dir / "train_cache.pt")
    prototype = AdaptiveEdosProbe(transformer).to(device)
    batch = encoder_batch(inputs, range(len(inputs)), device)
    with torch.no_grad():
        p, gamma = prototype(batch, with_gamma=True)
        maximum_tv = float(.5 * (p.cpu() - cache["original"]).abs().sum(-1).max())
        expected_scale = torch.tensor(cache["records"].blind_scale.to_numpy(), dtype=torch.float32)
        scale_error = float(((gamma.cpu() * blind_factor - expected_scale).abs() / expected_scale.abs().clamp_min(1e-12)).max())
    if maximum_tv > 1e-5 or scale_error > 1e-5:
        raise ValueError("initial cached encoder chain or blind scale failed equivalence")
    equivalence = dict(max_prediction_tv=maximum_tv, max_relative_blind_scale_error=scale_error,
                       **gradient_equivalence(prototype, {"inputs": inputs}, cache, selection, device))
    positive = positive_logit_control(cache["target"], pairs)
    if not positive["passed"]:
        raise RuntimeError("independent logit positive control failed")
    (run_dir / "preflight.json").write_text(json.dumps(dict(equivalence=equivalence, logit_control=positive), indent=2) + "\n")
    print(f"equivalence passed: {equivalence}; logit ratio={positive['final_to_initial']:.3g}", flush=True)
    prototype.cpu()
    del model, transformer, dataset, builder
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    original_samples, original_pairs = metric_tables({"original": {"train": cache["original"]}}, {"train": cache}, selection)
    sample_frames, final_pair_frames, histories, trajectories = [original_samples], [original_pairs], [], []
    first_fits, changes, final_predictions, final_scales = {}, {}, {}, {}
    for arm in ("frozen", "joint"):
        setup_ablation_seed(42)
        probe = copy.deepcopy(prototype).to(device)
        history, trajectory, first_fit = fit_arm(probe, arm, cache, batch, pairs, device, run_dir)
        histories.append(history)
        trajectories.append(trajectory)
        first_fits[arm] = first_fit
        with torch.no_grad():
            prediction, gamma = probe(batch, with_gamma=True)
        prediction, scale = prediction.cpu(), gamma.cpu() * blind_factor
        final_predictions[arm], final_scales[arm] = prediction, scale
        arm_cache = {**cache, "records": cache["records"].assign(blind_scale=scale.numpy())}
        samples, final_pairs = metric_tables({arm: {"train": prediction}}, {"train": arm_cache}, selection)
        sample_frames.append(samples)
        final_pair_frames.append(final_pairs)
        changes[arm] = {name: state_change(getattr(prototype, name).state_dict(), getattr(probe, name))
                        for name in ("encoder", "readout", "eta_head")}
        if changes[arm]["eta_head"]["changed_values"] or (arm == "frozen" and changes[arm]["encoder"]["changed_values"]):
            raise RuntimeError("frozen parameter boundary violated")
        if not changes[arm]["readout"]["changed_values"] or (arm == "joint" and not changes[arm]["encoder"]["changed_values"]):
            raise RuntimeError("expected parameters did not update")
        del probe
        if device.type == "cuda":
            torch.cuda.empty_cache()
    history, samples = pd.concat(histories, ignore_index=True), pd.concat(sample_frames, ignore_index=True)
    trajectory = pd.concat(trajectories, ignore_index=True).merge(selection.reset_index(names="pair_index"), on="pair_index", validate="many_to_one")
    atomic_torch_save(dict(predictions=final_predictions, blind_scale=final_scales), str(run_dir / "final_predictions.pt"))
    summary = []
    for arm in ("original", "frozen", "joint"):
        scores = pair_scores(cache["original"] if arm == "original" else final_predictions[arm], cache["target"], pairs)
        subset = samples.loc[samples.arm == arm]
        row = dict(arm=arm, n_pairs=len(scores), informative_pairs=int(scores.informative.sum()),
                   n_materials=len(subset), n_pairs_passed=int(scores.passed.sum()),
                   mean_mse=float(scores.contrast_mse.mean()), mean_tv=float(scores.contrast_error_tv.mean()),
                   aggregate_mse_ratio=float(scores.contrast_mse.mean() / max(scores.zero_mse.mean(), 1e-12)),
                   aggregate_tv_ratio=float(scores.contrast_error_tv.mean() / max(scores.target_tv.mean(), 1e-12)),
                   max_mse_ratio=float(scores.mse_ratio.max()), max_tv_ratio=float(scores.tv_ratio.max()),
                   first_fit_step=first_fits.get(arm) if first_fits.get(arm) is not None else -1,
                   final_all_pairs_passed=bool(scores.passed.all()))
        for metric in ("r2_edos_oracle", "r2_edos_blind"):
            row[metric + "_median"] = float(subset[metric].median())
            row[metric + "_fail_pct"] = float((subset[metric] < 0).mean() * 100)
        summary.append(row)
    for name, expected in hashes.items():
        if file_hash(REPO_ROOT / name) != expected:
            raise RuntimeError(f"protected input changed: {name}")
    if file_hash(run_dir / "train_cache.pt") != cache_hash:
        raise RuntimeError("small train cache changed")
    tables = dict(selection=selection, input_audit=audit, history=history, pair_history=trajectory,
                  samples=samples, final_pairs=pd.concat(final_pair_frames, ignore_index=True), summary=pd.DataFrame(summary))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
            raise FloatingPointError(f"nonfinite {name} results")
        with output_prefix.with_name(output_prefix.name + "_" + name + ".csv").open("x") as stream:
            frame.to_csv(stream, index=False)
    metadata.update(status="complete", verdict=fit_verdict(first_fits), first_fit_step=first_fits,
                    equivalence=equivalence, positive_logit_control=positive, parameter_changes=changes,
                    cache_sha256=cache_hash, elapsed_seconds=time.monotonic() - start,
                    peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                    limitations="Training-set fitting witness only; one selection and initialization; pure contrast objective does not constrain shared spectra.")
    with output_prefix.with_suffix(".json").open("x") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    (run_dir / "completed.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(tables["summary"].to_string(index=False), flush=True)
    print(f"verdict: {metadata['verdict']}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "output/g2_small_fit_q1")
    parser.add_argument("--output-prefix", type=Path, default=REPO_ROOT / "results/g2_small_fit_q1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_probe(args.run_dir, args.output_prefix, torch.device(args.device))


if __name__ == "__main__":
    main()
