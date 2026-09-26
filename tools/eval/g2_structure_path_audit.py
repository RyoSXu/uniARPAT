"""冻结 G2 的 train/valid 结构信息通路及内存残差关闭诊断。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import copy
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch
from torch.utils.data import DataLoader, Subset
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_ablation_experiments import setup_ablation_seed
from tools.eval.edos_slope_pilot_verdict import paired_bootstrap_interval
from tools.eval.structure_signal_diagnostic import structure_signature
from utils.builder import ConfigBuilder
from utils.metrics import per_sample_spectral_metrics

COUNTS = {"train": 18706, "valid": 2313}
METRICS = tuple(f"r2_{task}_{mode}" for task in ("edos", "phdos") for mode in ("oracle", "blind"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_rms(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not left.size:
        raise ValueError("relative RMS requires equal nonempty shapes")
    scale = np.sqrt(0.5 * (np.mean(left**2) + np.mean(right**2)))
    return float(np.sqrt(np.mean((left - right)**2)) / max(scale, 1e-12))


def matched_atom_rms(left, right, left_elements, right_elements):
    """Compare atom sets without imposing an arbitrary atom ordering."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    left_elements, right_elements = np.asarray(left_elements), np.asarray(right_elements)
    if not np.array_equal(np.sort(left_elements), np.sort(right_elements)):
        raise ValueError("atom matching requires identical element counts")
    if left.shape != right.shape or left.shape[0] != len(left_elements) or not left.size:
        raise ValueError("atom feature shapes do not match element counts")
    squared_error = 0.0
    for element in np.unique(left_elements):
        a, b = left[left_elements == element], right[right_elements == element]
        costs = cdist(a, b, metric="sqeuclidean")
        rows, columns = linear_sum_assignment(costs)
        squared_error += costs[rows, columns].sum()
    scale = np.sqrt(0.5 * (np.mean(left**2) + np.mean(right**2)))
    return float(np.sqrt(squared_error / left.size) / max(scale, 1e-12))


def contrast_metrics(pred_a, pred_b, target_a, target_b):
    predicted = np.asarray(pred_a, dtype=np.float64) - np.asarray(pred_b, dtype=np.float64)
    target = np.asarray(target_a, dtype=np.float64) - np.asarray(target_b, dtype=np.float64)
    denominator = np.linalg.norm(predicted) * np.linalg.norm(target)
    return {
        "target_tv": float(0.5 * np.abs(target).sum()),
        "predicted_tv": float(0.5 * np.abs(predicted).sum()),
        "contrast_error_tv": float(0.5 * np.abs(predicted - target).sum()),
        "contrast_cosine": float(np.dot(predicted, target) / denominator) if denominator > 1e-12 else 0.0,
    }


@contextmanager
def zero_g2_residuals(transformer):
    modules = list(transformer.encoder.g2_msgs)
    saved = [module.alpha.detach().clone() for module in modules]
    try:
        with torch.no_grad():
            for module in modules:
                module.alpha.zero_()
        yield
    finally:
        with torch.no_grad():
            for module, value in zip(modules, saved):
                module.alpha.copy_(value)


class FeatureProbe:
    """Observe existing modules; leave their returned tensors unchanged."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.features = {}
        self.handles = [transformer.encoder.register_forward_hook(self._encoder),
                        transformer.decoder.register_forward_hook(self._decoder)]
        for index, module in enumerate(getattr(transformer.encoder, "g2_msgs", [])):
            self.handles.append(module.register_forward_hook(self._residual(index)))

    def _encoder(self, module, args, output):
        self.features["encoder"] = output.detach()

    def _decoder(self, module, args, output):
        if output[0].shape[1] == 128:
            self.features["decoder"] = output[0].detach()

    def _residual(self, index):
        def capture(module, args, output):
            keep = self.keep.unsqueeze(-1)
            base = (args[0].float().square() * keep).sum(dim=(1, 2))
            difference = ((output.float() - args[0].float()).square() * keep).sum(dim=(1, 2))
            self.features[f"residual_{index}"] = (difference / base.clamp_min(1e-20)).sqrt()
        return capture

    def forward(self, processed):
        inp, pos, mask = processed[:3]
        self.keep = ~mask[:, 2:]
        self.features = {}
        output = self.transformer(inp, mask, pos, processed[-2], processed[-1])
        return output, dict(self.features)

    def close(self):
        for handle in self.handles:
            handle.remove()


def load_split(builder, split):
    if split not in COUNTS:
        raise ValueError("this audit only permits train/valid")
    return builder.get_dataset(split=split, dos_minmax=True, dos_sumnorm=True)


def make_pair_plan(dataset, ids, split):
    elements = dataset.elements.numpy()
    positions = dataset.positions.numpy().reshape(len(ids), -1, 3)
    groups = defaultdict(list)
    group_of = {}
    for index, row in enumerate(elements):
        values, counts = np.unique(row[2:][row[2:] != 0], return_counts=True)
        group = "|".join(f"{z}:{n}" for z, n in zip(values, counts))
        groups[group].append(index)
        group_of[index] = group
    pairs = []
    for group, indices in groups.items():
        signatures = {i: structure_signature(elements[i], positions[i]) for i in indices} if len(indices) > 1 else {}
        for a, b in combinations(indices, 2):
            if signatures[a] != signatures[b]:
                pairs.append(dict(group=group, sample_index_a=a, sample_index_b=b,
                                  mpid_a=ids[a], mpid_b=ids[b], single_element="|" not in group,
                                  structure_match=-1))
    pairs = pd.DataFrame(pairs)
    if split == "valid":
        reference = pd.read_csv(REPO_ROOT / "results/edos_spectral_support_q1_valid_pairs.csv")
        keys = ["sample_index_a", "sample_index_b", "mpid_a", "mpid_b"]
        if set(map(tuple, pairs[keys].to_numpy())) != set(map(tuple, reference[keys].to_numpy())):
            raise ValueError("valid material pairs do not reproduce the existing 320-pair evidence")
        pairs = pairs.drop(columns="structure_match").merge(reference[keys + ["structure_match"]], on=keys, validate="one_to_one")
        target = dataset.edos_tgtdos.numpy().astype(np.float64)
        target_tv = 0.5 * np.abs(target[reference.sample_index_a] - target[reference.sample_index_b]).sum(-1)
        if not np.allclose(target_tv, reference.oracle_target_tv, atol=2e-6, rtol=0):
            raise ValueError("valid target spectral contrasts do not reproduce the reference")
    order = [index for group in sorted(groups) for index in groups[group]]
    return groups, group_of, order, pairs


def spectral_batch_metrics(outputs, processed, model):
    inp, pos, mask, target_e, target_p = processed[:5]
    e_min, e_max, p_min, p_max = processed[7], processed[8], processed[11], processed[12]
    nval = processed[15]
    if nval is None or not torch.isfinite(nval).all():
        raise ValueError("Q1 blind evaluation requires finite N_valence")
    pe, pp = outputs["edos"].float().softmax(-1), outputs["phdos"].float().softmax(-1)
    truth_e, truth_p = target_e * (e_max - e_min) + e_min, target_p * (p_max - p_min) + p_min
    natoms = (~mask[:, 2:]).sum(-1, keepdim=True).float()
    predictions = {
        "edos_oracle": (pe * (e_max - e_min) + e_min).clamp_min(0),
        "phdos_oracle": (pp * (p_max - p_min) + p_min).clamp_min(0),
        "edos_blind": (pe * nval.reshape(-1, 1) * outputs["eta"][:, 1:2] / model.delta_edos).clamp_min(0),
        "phdos_blind": (pp * 3 * natoms * outputs["eta"][:, 0:1] / model.delta_phdos).clamp_min(0),
    }
    metrics = {"r2_" + key: per_sample_spectral_metrics(value, truth_e if key.startswith("edos") else truth_p)["r2"].cpu().numpy()
               for key, value in predictions.items()}
    metrics["shape_l1_edos"] = (pe - target_e).abs().sum(-1).cpu().numpy()
    return metrics, pe.cpu().numpy(), pp.cpu().numpy()


def numerical_control(probe, processed, original):
    repeated, _ = probe.forward(processed)
    permuted = list(processed)
    permutation = torch.arange(processed[0].shape[1] - 2, device=processed[0].device).flip(0)
    for field in (0, 1, 2):
        permuted[field] = processed[field].clone()
        permuted[field][:, 2:] = processed[field][:, 2:][:, permutation]
    changed, _ = probe.forward(permuted)
    errors = {}
    for label, outputs in (("repeat", repeated), ("permutation", changed)):
        for task in ("edos", "phdos"):
            tv = 0.5 * (outputs[task].softmax(-1) - original[task].softmax(-1)).abs().sum(-1)
            errors[f"{label}_{task}_max_tv"] = float(tv.max())
    if max(errors.values()) > 1e-5:
        raise ValueError(f"numerical controls failed: {errors}")
    return errors


def evaluate_model(model, dataset, ids, split, arms, batch_size, plan):
    groups, group_of, order, pair_plan = plan
    pair_groups = {key: rows.to_dict("records") for key, rows in pair_plan.groupby("group")}
    loader = DataLoader(Subset(dataset, order), batch_size=batch_size, shuffle=False, num_workers=0)
    shapes = {arm: np.empty((len(ids), 128), dtype=np.float32) for arm in arms}
    pending = {arm: {} for arm in arms}
    samples, pairs, interventions, controls = [], [], [], {}
    target = dataset.edos_tgtdos.numpy()
    transformer = model.model["transformer"]
    probe = FeatureProbe(transformer)
    offset = 0
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                processed = model.data_preprocess(batch)
                indices = order[offset:offset + len(processed[0])]
                keep = (~processed[2][:, 2:]).cpu().numpy()
                elements = processed[0][:, 2:].cpu().numpy()
                captured = {}
                for arm in arms:
                    context = zero_g2_residuals(transformer) if arm == "edge_zero" else nullcontext()
                    with context:
                        outputs, features = probe.forward(processed)
                        if batch_index == 0:
                            controls[arm] = numerical_control(probe, processed, outputs)
                    metrics, pe, pp = spectral_batch_metrics(outputs, processed, model)
                    memory = features["encoder"].cpu().numpy()
                    decoder = features["decoder"].cpu().numpy()
                    captured[arm] = (memory, decoder, pe, pp, metrics)
                    shapes[arm][indices] = pe
                    records = pd.DataFrame(dict(split=split, arm=arm, sample_index=indices,
                                                mpid=ids[indices], **metrics))
                    for key, value in features.items():
                        if key.startswith("residual_"):
                            records[key] = value.cpu().numpy()
                    samples.append(records)
                    for local, index in enumerate(indices):
                        group = group_of[index]
                        if group not in pair_groups:
                            continue
                        pending[arm][index] = (elements[local, keep[local]],
                                              memory[local, keep[local]].copy(), decoder[local].copy())
                        if index == groups[group][-1]:
                            for pair in pair_groups[group]:
                                a, b = pair["sample_index_a"], pair["sample_index_b"]
                                za, ha, da = pending[arm][a]
                                zb, hb, db = pending[arm][b]
                                pairs.append(dict(split=split, arm=arm, **pair,
                                                  encoder_relative_rms=matched_atom_rms(ha, hb, za, zb),
                                                  decoder_relative_rms=relative_rms(da, db),
                                                  **contrast_metrics(shapes[arm][a], shapes[arm][b], target[a], target[b])))
                            for member in groups[group]:
                                pending[arm].pop(member, None)
                if "edge_zero" in arms:
                    on, off = captured["edge"], captured["edge_zero"]
                    for local, index in enumerate(indices):
                        interventions.append(dict(
                            split=split, sample_index=index, mpid=ids[index],
                            encoder_relative_rms=relative_rms(on[0][local, keep[local]], off[0][local, keep[local]]),
                            decoder_relative_rms=relative_rms(on[1][local], off[1][local]),
                            edos_tv=float(0.5 * np.abs(on[2][local] - off[2][local]).sum()),
                            phdos_tv=float(0.5 * np.abs(on[3][local] - off[3][local]).sum()),
                            **{metric + "_on_minus_zero": float(on[4][metric][local] - off[4][metric][local]) for metric in METRICS},
                        ))
                offset += len(indices)
                if batch_index % 100 == 0 or offset == len(ids):
                    print(f"{split} {'/'.join(arms)} {offset}/{len(ids)}", flush=True)
    finally:
        probe.close()
    if any(pending.values()) or offset != len(ids):
        raise RuntimeError("incomplete evaluation or pair cache")
    return pd.concat(samples, ignore_index=True), pd.DataFrame(pairs), pd.DataFrame(interventions), controls


def cluster_median_interval(frame, column, replicates=2000):
    groups = [part[column].to_numpy() for _, part in frame.groupby("group")]
    rng = np.random.default_rng(20260926)
    medians = [np.median(np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]))
               for _ in range(replicates)]
    return [float(x) for x in np.quantile(medians, [0.025, 0.975])]


def summarize(samples, pairs):
    summaries, comparisons, pair_summaries, residuals = [], [], [], []
    for (split, arm), frame in samples.groupby(["split", "arm"]):
        for metric in METRICS:
            values = frame[metric].to_numpy()
            summaries.append(dict(split=split, arm=arm, metric=metric, n=len(values),
                                  median=float(np.median(values)), fail_pct=float(100 * (values < 0).mean())))
        for column in frame.columns:
            if column.startswith("residual_") and frame[column].notna().all():
                residuals.append(dict(split=split, arm=arm, layer=int(column.split("_")[-1]),
                                      median=float(frame[column].median()), p90=float(frame[column].quantile(.9)),
                                      maximum=float(frame[column].max())))
    for split in COUNTS:
        for left, right in (("control", "edge"), ("edge_zero", "edge")):
            a = samples[(samples.split == split) & (samples.arm == left)].sort_values("sample_index")
            b = samples[(samples.split == split) & (samples.arm == right)].sort_values("sample_index")
            if not np.array_equal(a.mpid, b.mpid):
                raise ValueError("sample comparison IDs do not match")
            for metric in METRICS:
                x, y = a[metric].to_numpy(), b[metric].to_numpy()
                row = dict(split=split, comparison=f"{right}_minus_{left}", metric=metric,
                           delta_median=float(np.median(y) - np.median(x)),
                           delta_fail_pp=float(100 * ((y < 0).mean() - (x < 0).mean())))
                if split == "valid":
                    row["median_ci_low"], row["median_ci_high"] = paired_bootstrap_interval(x, y, seed=20260926)
                    low, high = paired_bootstrap_interval((x < 0).astype(float), (y < 0).astype(float), "mean", seed=20260926)
                    row["fail_ci_low_pp"], row["fail_ci_high_pp"] = low * 100, high * 100
                comparisons.append(row)
    for (split, arm), frame in pairs.groupby(["split", "arm"]):
        subsets = {"all": frame, "single_element": frame[frame.single_element],
                   "multiple_elements": frame[~frame.single_element]}
        if split == "valid":
            subsets["multiple_elements_unmatched"] = frame[(~frame.single_element) & (frame.structure_match == 0)]
        for label, rows in subsets.items():
            if rows.empty:
                continue
            row = dict(split=split, arm=arm, population=label, n_pairs=len(rows), n_groups=rows.group.nunique())
            for metric in ("target_tv", "predicted_tv", "contrast_error_tv", "contrast_cosine", "encoder_relative_rms", "decoder_relative_rms"):
                row[metric + "_median"] = float(rows[metric].median())
                row[metric + "_group_equal_median"] = float(rows.groupby("group")[metric].median().median())
            pair_summaries.append(row)
    pair_comparisons = []
    keys = ["split", "group", "sample_index_a", "sample_index_b"]
    for left, right in (("control", "edge"), ("edge_zero", "edge")):
        a, b = pairs[pairs.arm == left], pairs[pairs.arm == right]
        joined = a.merge(b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
        for split, rows in joined.groupby("split"):
            rows = rows.copy()
            for metric in ("predicted_tv", "contrast_error_tv", "encoder_relative_rms", "decoder_relative_rms"):
                rows[metric + "_delta"] = rows[metric + "_b"] - rows[metric + "_a"]
                row = dict(split=split, comparison=f"{right}_minus_{left}", metric=metric,
                           paired_delta_median=float(rows[metric + "_delta"].median()))
                if split == "valid":
                    row["cluster_ci_low"], row["cluster_ci_high"] = cluster_median_interval(rows, metric + "_delta")
                pair_comparisons.append(row)
    return {"summary": pd.DataFrame(summaries), "comparisons": pd.DataFrame(comparisons),
            "pair_summary": pd.DataFrame(pair_summaries), "pair_comparisons": pd.DataFrame(pair_comparisons),
            "residuals": pd.DataFrame(residuals)}


def validate_history(samples, tag):
    history = pd.read_csv(REPO_ROOT / f"results/history_m1_{tag}.csv")
    reference = history.loc[history.epoch == 10].iloc[0]
    actual = samples[samples.split == "valid"]
    for task in ("edos", "phdos"):
        values = actual[f"r2_{task}_oracle"]
        for actual_value, expected in ((values.median(), reference[f"r2_{task}_median"]),
                                       (100 * (values < 0).mean(), reference[f"fail_rate_{task}"])):
            if abs(actual_value - expected) > 2e-5:
                raise ValueError(f"{tag} valid {task} does not reproduce epoch 10 history")


def run_audit(output_prefix, device, batch_size=32):
    output_prefix = Path(output_prefix).resolve()
    os.chdir(REPO_ROOT)
    names = ("samples", "pairs", "interventions", "summary", "comparisons", "pair_summary", "pair_comparisons", "residuals")
    paths = {key: output_prefix.with_name(output_prefix.name + "_" + key + ".csv") for key in names}
    paths["metadata"] = output_prefix.with_suffix(".json")
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("audit output exists; choose a new output prefix")
    setup_ablation_seed(42)
    configs, checkpoints, hashes = {}, {}, {}
    for tag in ("g2ctl", "g2edge"):
        root = REPO_ROOT / f"output/ablation_m1_{tag}"
        configs[tag] = yaml.safe_load((root / "config_used.yaml").read_text())
        for name in ("checkpoint_latest.pth", "config_used.yaml"):
            hashes[str((root / name).relative_to(REPO_ROOT))] = file_hash(root / name)
        checkpoints[tag] = root / "checkpoint_latest.pth"
    copies = [copy.deepcopy(configs[tag]) for tag in ("g2ctl", "g2edge")]
    for index, config in enumerate(copies):
        if config["cli"].pop("use_g2") != bool(index):
            raise ValueError("unexpected G2 arm flag")
        config["config"]["model"]["params"]["sub_model"]["transformer"].pop("use_g2")
    if copies[0] != copies[1]:
        raise ValueError("paired effective configurations differ beyond use_g2")
    cli = configs["g2ctl"]["cli"]
    if cli["norm"] != "sumnorm" or cli["use_mask"] or cli["augment"] or cli["epochs"] != 10:
        raise ValueError("unexpected frozen training configuration")
    builder = ConfigBuilder(**copy.deepcopy(configs["g2ctl"]["config"]))
    datasets, ids, plans = {}, {}, {}
    manifest = json.loads((REPO_ROOT / "data/train4ARPAT/manifest.json").read_text())
    for split, count in COUNTS.items():
        datasets[split] = load_split(builder, split)
        index_path = REPO_ROOT / f"data/train4ARPAT/{split}/{split}_index.npy"
        ids[split] = np.load(index_path).astype(str)
        if len(datasets[split]) != count or len(ids[split]) != count or manifest["splits"][split]["n"] != count:
            raise ValueError("frozen Q1 counts do not match")
        plans[split] = make_pair_plan(datasets[split], ids[split], split)
        hashes[str(index_path.relative_to(REPO_ROOT))] = file_hash(index_path)
        for name in ("elements", "positions", "edos_tgtdos", "phdos_tgtdos", "edos_mask", "phdos_mask", "nvalence"):
            path = REPO_ROOT / f"data/train4ARPAT/{split}/{name}_{split}.npy"
            hashes[str(path.relative_to(REPO_ROOT))] = file_hash(path)
    for path in (REPO_ROOT / "data/train4ARPAT/manifest.json", REPO_ROOT / "data/grids_c2b/grids.json"):
        hashes[str(path.relative_to(REPO_ROOT))] = file_hash(path)
    reference_path = REPO_ROOT / "results/edos_spectral_support_q1_valid_pairs.csv"
    hashes[str(reference_path.relative_to(REPO_ROOT))] = file_hash(reference_path)
    all_samples, all_pairs, all_interventions = [], [], []
    controls, alphas = {}, {}
    start = time.monotonic()
    for tag, arms in (("g2ctl", ("control",)), ("g2edge", ("edge", "edge_zero"))):
        model = ConfigBuilder(**copy.deepcopy(configs[tag]["config"])).get_model()
        checkpoint = torch.load(checkpoints[tag], map_location="cpu", weights_only=True)
        if (checkpoint.get("model_name"), checkpoint.get("epoch"), checkpoint.get("seed")) != ("M1", 10, 42) or checkpoint.get("use_amp", False):
            raise ValueError("expected frozen M1 epoch 10 seed 42 FP32 checkpoint")
        transformer = model.model["transformer"]
        transformer.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        model.to(device)
        transformer.eval().requires_grad_(False)
        original_alphas = [module.alpha.detach().clone() for module in getattr(transformer.encoder, "g2_msgs", [])]
        alphas[tag] = [float(value) for value in original_alphas]
        local_samples = []
        for split in COUNTS:
            samples, pairs, interventions, checks = evaluate_model(model, datasets[split], ids[split], split, arms, batch_size, plans[split])
            all_samples.append(samples)
            local_samples.append(samples[samples.arm == arms[0]])
            all_pairs.append(pairs)
            if not interventions.empty:
                all_interventions.append(interventions)
            controls[f"{tag}_{split}"] = checks
        validate_history(pd.concat(local_samples), tag)
        if any(not torch.equal(value, module.alpha) for value, module in zip(original_alphas, getattr(transformer.encoder, "g2_msgs", []))):
            raise RuntimeError("G2 alpha values were not restored")
        del transformer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    samples = pd.concat(all_samples, ignore_index=True).sort_values(["split", "arm", "sample_index"])
    pairs = pd.concat(all_pairs, ignore_index=True).sort_values(["split", "arm", "sample_index_a", "sample_index_b"])
    interventions = pd.concat(all_interventions, ignore_index=True).sort_values(["split", "sample_index"])
    for frame in (samples.drop(columns=[c for c in samples if c.startswith("residual_")]), pairs, interventions):
        if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError("non-finite diagnostic output")
    tables = dict(samples=samples, pairs=pairs, interventions=interventions, **summarize(samples, pairs))
    for path, before in hashes.items():
        if file_hash(REPO_ROOT / path) != before:
            raise RuntimeError(f"input changed during audit: {path}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for key, frame in tables.items():
        with paths[key].open("x") as stream:
            frame.to_csv(stream, index=False)
    metadata = dict(epoch=10, seed=42, device=str(device), batch_size=batch_size, counts=COUNTS,
                    checkpoint_and_input_sha256=hashes, script_sha256=file_hash(__file__),
                    g2_alpha=alphas, numerical_controls=controls, elapsed_seconds=time.monotonic() - start,
                    checkpoint_written=False, optimizer_called=False, test_evaluated=False,
                    output_paths={key: str(path) for key, path in paths.items()},
                    interpretation="Frozen-model dependence only; hidden distances are not information or causal accuracy gains.")
    with paths["metadata"].open("x") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    print(tables["comparisons"].to_string(index=False), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, default=REPO_ROOT / "results/g2_structure_path_q1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(2)
    run_audit(args.output_prefix, torch.device(args.device), args.batch_size)


if __name__ == "__main__":
    main()
