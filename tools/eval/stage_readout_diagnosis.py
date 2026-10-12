"""Frozen-model stage readout diagnosis of same-composition eDOS shape gaps.

Read-only with respect to the trained models: the C1 (primary) and C2 (reference)
best checkpoints run in eval mode, FP32, parameters frozen; only small diagnostic
linear readouts are fitted on train pairs. The three probed stages keep the
spectral position that the earlier pooled mean/RMS decoder summary discarded:

* ``local_last``  - complete outer atom state after the final local update;
* ``global_last`` - atom state after the final global update;
* ``decoder_edos`` - eDOS decoder output tokens at every energy grid point.

Readouts map representation *differences* to the unit-sum eDOS shape difference
and are linear without intercept, so swapping the two materials exactly negates
the prediction and identical representations predict a zero gap. Differential
features may be divided by a train-estimated scale but are never centered.
Regularization is chosen only by composition-grouped cross-validation on train
with preprocessing fitted inside each training fold; every pair of one
composition stays in one fold and composition groups weigh equally in fitting
and scoring. The locked readout is then evaluated on valid. Train/valid only,
seed 42, no test access and no model update.
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
from sklearn.model_selection import GroupKFold

from tools.eval.lg_structure_diagnosis import dump, loader, pair_errors, paired_group_interval
from tools.eval.lg_train_subset_diagnosis import composition_keys_by_position, same_composition_pairs
from tools.eval.lg_valid_export import load_run, select_best
from tools.eval.local_structure_compare import verify_completed_control, verify_preflight
from tools.eval.local_structure_diagnosis import grouped_mean_frame, run_directory
from tools.train import evaluate_split
from utils.builder import ConfigBuilder
from utils.crystal import atom_padding_mask
from utils.local_structure_experiment import (
    DEFAULT_REFERENCE,
    load_control_reference,
    structure_control_configs,
)
from utils.paths import file_sha256, repository_path
from utils.zproj_alignment import state_hash

ATTRIBUTION = ROOT / "results/runs/local_structure_attribution_20261009"
PRIOR_ANALYSIS = ATTRIBUTION / "analysis"
PREFLIGHT_PATH = ATTRIBUTION / "preflight_r1.json"
LABELS = ("C1", "C2")  # C1 primary, C2 reference
GROUPINGS = ("absolute", "reduced")
ATOM_STAGES = ("local_last", "global_last")
STAGES = (*ATOM_STAGES, "decoder_edos")
STAGE_DESCRIPTIONS = {
    "local_last": "complete outer atom state after the final local update",
    "global_last": "atom state after the final global update",
    "decoder_edos": "eDOS decoder output token at every energy grid position",
}
TASK = "edos"
SEED = 42
NEAR_ZERO_TV = 1e-6
SCALE_FLOOR = 1e-6
ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14)
CV_SPLITS = 5
MAX_ALPHA_EXTENSIONS = 3
SOLVE_CHUNK = 8
SCALE_CHUNK = 256
METHODS = (*STAGES, "model", "zero_response")


def pair_index_table(keys, ids):
    """All same-composition pairs of the given keys, in deterministic order."""
    pairs, groups = same_composition_pairs(keys)
    frame = pd.DataFrame({
        "i": [int(i) for i, _ in pairs],
        "j": [int(j) for _, j in pairs],
        "left_id": [str(ids[i]) for i, _ in pairs],
        "right_id": [str(ids[j]) for _, j in pairs],
        "composition": [repr(keys[i]) for i, _ in pairs],
    })
    sizes = {repr(key): len(members) for key, members in groups.items() if len(members) > 1}
    return frame, sizes


def participant_rows(frames):
    rows = set()
    for frame in frames.values():
        rows.update(frame.i.to_numpy(int).tolist())
        rows.update(frame.j.to_numpy(int).tolist())
    return sorted(rows)


def pooled_mean_rms(value, mask):
    """Permutation-invariant mean and RMS over valid atom slots only."""
    valid = (~mask).to(value.dtype)[..., None]
    count = valid.sum(1).clamp_min(1)
    mean = (value * valid).sum(1) / count
    second = (value.square() * valid).sum(1) / count
    return torch.cat((mean, second.sqrt()), -1)


class StageExtractor:
    """Read-only hooks for the three frozen stages; hooks are removable."""

    def __init__(self, transformer):
        encoder = transformer.encoder
        if getattr(encoder, "local_layers", None) is None:
            raise ValueError("stage diagnosis requires the local/global encoder")
        picked = {}
        for kind, index in encoder.execution_order:
            picked[kind] = index
        self.stage_modules = {
            "local_last": encoder.local_layers[picked["local"]],
            "global_last": encoder.global_layers[picked["global"]],
        }
        self.features = {name: [] for name in STAGES}
        self.mask = None
        self.decoder_call = 0
        self.hooks = [transformer.register_forward_pre_hook(self._begin),
                      transformer.decoder.register_forward_hook(self._decoder)]
        for name, module in self.stage_modules.items():
            self.hooks.append(module.register_forward_hook(self._atom(name)))

    def _begin(self, module, args):
        self.mask = atom_padding_mask(args[2], args[1])
        self.decoder_call = 0

    def _atom(self, name):
        def hook(module, args, output):
            self.features[name].append(
                pooled_mean_rms(output, self.mask).detach().float().cpu().numpy())
        return hook

    def _decoder(self, module, args, output):
        # The shared decoder runs eDOS first, then phDOS; keep eDOS tokens only.
        if self.decoder_call == 0:
            value = output[0]
            if value.dim() != 3:
                raise ValueError("eDOS decoder output must keep its energy positions")
            self.features["decoder_edos"].append(value.detach().float().cpu().numpy())
        self.decoder_call += 1

    def stacked(self):
        out = {}
        for name in STAGES:
            chunks = self.features[name]
            if not chunks:
                raise ValueError(f"missing stage features: {name}")
            out[name] = np.concatenate(chunks, axis=0)
        return out

    def close(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


def feature_scale(values, rows, chunk=SCALE_CHUNK):
    """RMS scale per feature over the given materials; no centering."""
    rows = np.asarray(rows, int)
    if not len(rows):
        raise ValueError("cannot estimate a feature scale without materials")
    if values.ndim == 2:
        picked = values[rows].astype(np.float64)
        scale = np.sqrt(np.square(picked).mean(0))
    else:
        total = np.zeros(values.shape[1:], dtype=np.float64)
        for start in range(0, len(rows), chunk):
            picked = values[rows[start:start + chunk]].astype(np.float64)
            total += np.square(picked).sum(0)
        scale = np.sqrt(total / len(rows))
    return np.maximum(scale, SCALE_FLOOR)


def group_equal_weights(group_labels):
    """Pair weights so each composition group carries equal total weight (mean 1)."""
    group_labels = np.asarray(group_labels)
    _, inverse, counts = np.unique(group_labels, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights * (len(weights) / weights.sum())


def group_mean(values, group_labels, mask=None):
    values = np.asarray(values, dtype=np.float64)
    group_labels = np.asarray(group_labels)
    if mask is not None:
        values, group_labels = values[mask], group_labels[mask]
    if not len(values):
        return None
    frame = pd.DataFrame({"composition": group_labels, "value": values})
    return float(frame.groupby("composition").value.mean().mean())


def atom_delta(features, left, right, scale):
    # Scale first, then difference: pair gaps equal differences of per-material
    # readouts bit-for-bit and never mix float32 subtraction into float64 fits.
    scaled = features.astype(np.float64) / np.asarray(scale, dtype=np.float64)
    return scaled[left] - scaled[right]


def atom_ridge_path(x, y, alphas, weights):
    """Weighted multi-output ridge for one design; one Gram matrix, many alphas."""
    root = np.sqrt(weights)[:, None]
    xw = x * root
    yw = y * root
    gram = xw.T @ xw
    rhs = xw.T @ yw
    solved = {}
    for alpha in alphas:
        solved[alpha] = np.linalg.solve(
            gram + alpha * np.eye(gram.shape[0]), rhs)
    return solved


def decoder_ridge_path(features, left, right, scale, y, alphas, weights,
                       chunk=SOLVE_CHUNK):
    """Per-energy-position weighted ridge; positions processed in chunks."""
    left = np.asarray(left, int)
    right = np.asarray(right, int)
    n_pos = features.shape[1]
    n_features = features.shape[2]
    solved = {alpha: np.empty((n_pos, n_features), dtype=np.float64) for alpha in alphas}
    root = np.sqrt(weights)[:, None, None].astype(np.float64)
    y_root = y * np.sqrt(weights)[:, None]
    for start in range(0, n_pos, chunk):
        stop = min(start + chunk, n_pos)
        block = features[:, start:stop, :].astype(np.float64) / np.asarray(
            scale, dtype=np.float64)[start:stop][None]
        delta = block[left] - block[right]
        weighted = delta * root
        batched = np.swapaxes(weighted, 0, 1)  # batch over energy positions
        gram = np.matmul(np.swapaxes(batched, 1, 2), batched)
        rhs = np.einsum("nck,nc->ck", weighted, y_root[:, start:stop])
        for alpha in alphas:
            system = gram + alpha * np.eye(n_features)[None]
            solved[alpha][start:stop] = np.linalg.solve(
                system, rhs[..., None])[..., 0]
    return solved


def stage_path(stage, features, left, right, scale, y, alphas, weights):
    if stage in ATOM_STAGES:
        return atom_ridge_path(atom_delta(features, left, right, scale), y, alphas, weights)
    return decoder_ridge_path(features, left, right, scale, y, alphas, weights)


def predict_pairs(stage, features, left, right, scale, weights):
    if stage in ATOM_STAGES:
        return atom_delta(features, left, right, scale) @ weights
    left = np.asarray(left, int)
    right = np.asarray(right, int)
    n_pos = features.shape[1]
    out = np.empty((len(left), n_pos), dtype=np.float64)
    for start in range(0, n_pos, SOLVE_CHUNK):
        stop = min(start + SOLVE_CHUNK, n_pos)
        block = features[:, start:stop, :].astype(np.float64) / np.asarray(
            scale, dtype=np.float64)[start:stop][None]
        delta = block[left] - block[right]
        out[:, start:stop] = np.einsum("nck,ck->nc", delta, weights[start:stop])
    return out


def predict_materials(stage, features, scale, weights):
    if stage in ATOM_STAGES:
        return (features.astype(np.float64) / np.asarray(scale, dtype=np.float64)) @ weights
    n_rows, n_pos = features.shape[0], features.shape[1]
    out = np.empty((n_rows, n_pos), dtype=np.float64)
    for start in range(0, n_pos, SOLVE_CHUNK):
        stop = min(start + SOLVE_CHUNK, n_pos)
        block = features[:, start:stop, :].astype(np.float64) / np.asarray(
            scale, dtype=np.float64)[start:stop][None]
        out[:, start:stop] = np.einsum("nck,ck->nc", block, weights[start:stop])
    return out


def pair_targets(shape, frame):
    left = frame.i.to_numpy(int)
    right = frame.j.to_numpy(int)
    target = shape[left] - shape[right]
    tv = 0.5 * np.abs(target).sum(-1)
    return target, tv


def fold_score(pred, target, tv, groups):
    keep = tv >= NEAR_ZERO_TV
    if not keep.any():
        return 0.0
    err = 0.5 * np.abs(pred[keep] - target[keep]).sum(-1)
    return group_mean(err, groups[keep])


class StageFit:
    """Composition-grouped train CV and the locked train fit for one stage."""

    def __init__(self, stage, features, frame, shape, alphas=ALPHAS):
        self.stage = stage
        self.features = features
        self.frame = frame
        self.left = frame.i.to_numpy(int)
        self.right = frame.j.to_numpy(int)
        self.groups = frame.composition.to_numpy()
        self.target, self.tv = pair_targets(shape, frame)
        self.alphas = list(alphas)
        self.extensions = []
        self.folds = []
        unique = np.unique(self.groups)
        n_splits = int(min(CV_SPLITS, len(unique)))
        if n_splits < 2:
            raise ValueError("composition-grouped CV needs at least two groups")
        splitter = GroupKFold(n_splits=n_splits)
        self.n_splits = n_splits
        for train_idx, val_idx in splitter.split(np.zeros(len(self.groups)), groups=self.groups):
            slots = np.unique(np.concatenate([self.left[train_idx], self.right[train_idx]]))
            self.folds.append({
                "train": train_idx, "valid": val_idx,
                "scale": feature_scale(features, slots),
                "weights": group_equal_weights(self.groups[train_idx]),
                "cache": {},
            })

    def _fold_paths(self, fold, alphas):
        """Fit each fold once per alpha set; the design and Gram are alpha-independent."""
        missing = tuple(alpha for alpha in alphas if alpha not in fold["cache"])
        if missing:
            fold["cache"].update(stage_path(
                self.stage, self.features, self.left[fold["train"]], self.right[fold["train"]],
                fold["scale"], self.target[fold["train"]], missing, fold["weights"]))
        return fold["cache"]

    def cv_trace(self, alphas=None):
        wanted = self.alphas if alphas is None else list(alphas)
        for fold in self.folds:
            self._fold_paths(fold, wanted)
        trace = {}
        for alpha in wanted:
            scores = []
            for fold in self.folds:
                weights = fold["cache"][alpha]
                pred = predict_pairs(self.stage, self.features, self.left[fold["valid"]],
                                     self.right[fold["valid"]], fold["scale"], weights)
                scores.append(fold_score(pred, self.target[fold["valid"]],
                                         self.tv[fold["valid"]], self.groups[fold["valid"]]))
            trace[alpha] = float(np.mean(scores))
        return trace

    @staticmethod
    def _best(trace):
        # Exact ties prefer the stronger shrinkage: closer to the zero-response control.
        return min(trace, key=lambda value: (trace[value], -value))

    def select(self):
        """Locked alpha from train CV only; extend the grid only on train evidence."""
        trace = self.cv_trace()
        best = self._best(trace)
        while ((best == self.alphas[0] or best == self.alphas[-1])
               and len(self.extensions) < MAX_ALPHA_EXTENSIONS):
            if best == self.alphas[0]:
                add = [self.alphas[0] * 1e-4, self.alphas[0] * 1e-2]
                self.alphas = add + self.alphas
            else:
                add = [self.alphas[-1] * 1e2, self.alphas[-1] * 1e4]
                self.alphas = self.alphas + add
            self.extensions.extend(add)
            trace.update(self.cv_trace(add))
            best = self._best(trace)
        self.trace = trace
        self.alpha = best
        self.selected_at_grid_edge = bool(best in (self.alphas[0], self.alphas[-1]))
        return best

    def final_fit(self):
        slots = np.unique(np.concatenate([self.left, self.right]))
        self.scale = feature_scale(self.features, slots)
        weights = group_equal_weights(self.groups)
        path = stage_path(self.stage, self.features, self.left, self.right, self.scale,
                          self.target, (self.alpha,), weights)
        self.weights = path[self.alpha]
        return self.weights

    def describe(self):
        input_dim = int(np.prod(self.features.shape[1:]))
        per_position = self.features.shape[2] if self.features.ndim == 3 else None
        readout_params = (input_dim * self.target.shape[1] if self.features.ndim == 2
                          else int(self.features.shape[1] * self.features.shape[2]))
        return {
            "input_dim_per_material": input_dim,
            "readout_parameters": readout_params,
            "energy_positions": (int(self.features.shape[1]) if self.features.ndim == 3 else None),
            "feature_per_energy_dim": per_position,
            "target_dim": int(self.target.shape[1]),
            "intercept": False,
            "scaling": "per-feature RMS over fit materials (no centering)",
            "pair_weights": "composition-group equal, mean weight 1",
            "alpha_grid": [float(value) for value in self.alphas],
            "alpha_extensions": [float(value) for value in self.extensions],
            "selected_alpha": float(self.alpha),
            "selected_at_grid_edge": bool(self.selected_at_grid_edge),
            "cv_splits": self.n_splits,
            "cv_group_mean_pair_error": {str(key): value for key, value in self.trace.items()},
            "fit_scope": "train pairs only; scalers fit on train materials; alpha by "
                         "train-internal composition-grouped CV; locked before valid",
        }


def method_metrics(frame):
    main = frame[frame.target_tv >= NEAR_ZERO_TV]
    near = frame[frame.target_tv < NEAR_ZERO_TV]
    out = {
        "pairs": int(len(frame)),
        "main_pairs": int(len(main)),
        "near_zero_pairs": int(len(near)),
        "composition_groups": int(main.composition.nunique()),
        "target_tv_median": float(main.target_tv.median()) if len(main) else None,
        "predicted_tv_median": float(main.predicted_tv.median()) if len(main) else None,
    }
    if len(main):
        out["group_mean_error"] = grouped_mean_frame(main, "difference_error")
        out["skill"] = grouped_mean_frame(main.assign(value=main.skill), "value")
        out["skill_median"] = float(main.skill.median())
        cosine = main.dropna(subset=["direction_cosine"])
        out["direction_cosine"] = (
            grouped_mean_frame(cosine.assign(value=cosine.direction_cosine), "value")
            if len(cosine) else {"groups": 0, "mean_group": None, "ci95": None})
        out["direction_cosine_median"] = (
            float(cosine.direction_cosine.median()) if len(cosine) else None)
    if len(near):
        out["near_zero"] = {
            "target_tv_max": float(near.target_tv.max()),
            "group_mean_error": grouped_mean_frame(near, "difference_error"),
            "skill": "undefined for near-zero targets; reported separately",
        }
    return out


def compare_frames(left, right, name):
    result = {"comparison": name, "error_delta_right_minus_left": paired_group_interval(left, right)}
    interval = result["error_delta_right_minus_left"]
    if interval.get("ci95") is not None:
        result["right_beats_left"] = bool(interval["ci95"][1] < 0)
    return result


def build_frames(shape, predictions, keys, ids):
    frames = {}
    for name, values in predictions.items():
        frames[name] = pair_errors(values, shape, keys, ids)
    return frames


def identity_checks(stage, features, scale, weights, frame):
    """The readout contract: swap negates, identical representations give zero."""
    left = frame.i.to_numpy(int)[:16]
    right = frame.j.to_numpy(int)[:16]
    forward = predict_pairs(stage, features, left, right, scale, weights)
    backward = predict_pairs(stage, features, right, left, scale, weights)
    self_zero = predict_pairs(stage, features, left, left, scale, weights)
    return {
        "swap_antisymmetry_max_abs": float(np.max(np.abs(forward + backward))) if len(left) else None,
        "identical_representation_max_abs": float(np.max(np.abs(self_zero))) if len(left) else None,
    }


def extraction(model, dataset, rows, ids, device):
    transformer = model.model["transformer"]
    extractor = StageExtractor(transformer)
    tick = time.monotonic()
    try:
        measured = evaluate_split(model, loader(dataset, rows), sumnorm=True,
                                  return_sample_level=True, return_spectra=True,
                                  sample_ids=ids)
    finally:
        extractor.close()
    if extractor.hooks:
        raise ValueError("stage hooks were not removed")
    features = extractor.stacked()
    spectra = measured["sample_spectra"]
    for name in STAGES:
        if features[name].shape[0] != len(ids):
            raise ValueError(f"stage rows differ from evaluated materials: {name}")
    if spectra["pred_shape_edos"].shape[0] != len(ids):
        raise ValueError("spectra rows differ from evaluated materials")
    return features, spectra, time.monotonic() - tick


def prior_agreement(label, ids, pred_shape):
    """Cross-check the recomputed valid predictions against the prior exports."""
    path = PRIOR_ANALYSIS / f"{label}_valid_spectra.npz"
    if not path.exists():
        return {}
    with np.load(path) as prior:
        prior_ids = [str(value) for value in prior["material_ids"]]
        lookup = {mid: index for index, mid in enumerate(prior_ids)}
        positions = [lookup.get(mid) for mid in ids]
        if any(value is None for value in positions):
            raise ValueError("evaluated valid materials are missing from prior exports")
        prior_pred = prior["pred_shape_edos"][np.asarray(positions, int)]
    return {"source": str(path.relative_to(ROOT)), "materials": len(ids),
            "max_abs_pred_shape_difference": float(np.max(np.abs(prior_pred - pred_shape)))}


def pairing_manifest(sizes, frames):
    return {
        grouping: {
            "pairs": int(len(frames[grouping])),
            "composition_groups": int(frames[grouping].composition.nunique()),
            "group_size_histogram": {str(size): int(count) for size, count in
                                     sorted(pd.Series(
                                         list(sizes[grouping].values())).value_counts().items())},
        } for grouping in GROUPINGS
    }


def diagnose(out_dir, smoke=False):
    if (out_dir / "report.json").exists():
        raise FileExistsError(f"refusing completed output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def progress(stage):
        print(json.dumps({"stage": stage, "seconds": round(time.monotonic() - start, 3)}),
              flush=True)

    reference_dir = repository_path(DEFAULT_REFERENCE)
    _, reference_cfg = load_control_reference(reference_dir)
    controls = structure_control_configs(reference_cfg, "attrib20261009")
    preflight = verify_preflight(PREFLIGHT_PATH, reference_dir, controls)

    # Model identities: C1 through the control verification, C2 through its run record.
    identities = {}
    for label in LABELS:
        run = run_directory(label, reference_dir, controls, preflight)
        if label == "C1":
            saved, picked = verify_completed_control(
                next(c for c in controls if c.local_route == label), preflight)
        else:
            saved, _, contract = load_run(run)
            picked = select_best(run, contract["selection_metric"], saved)
        identities[label] = {
            "run": str(run.relative_to(ROOT)),
            "epoch": int(picked["row"].epoch),
            "checkpoint_sha256": file_sha256(run / "checkpoint_best.pth"),
            "data_contract": saved["cli"]["data_contract"],
            "config": saved["config"],
            "model_state": picked["checkpoint"]["model"],
        }
        del picked["checkpoint"], saved
    if identities["C1"]["data_contract"] != identities["C2"]["data_contract"]:
        raise ValueError("C1 and C2 refer to different data contracts")

    builder = ConfigBuilder(**identities["C1"]["config"])
    datasets = {split: builder.get_dataset(split=split, dos_minmax=True, dos_sumnorm=True)
                for split in ("train", "valid")}
    if any(ds.augment for ds in datasets.values()):
        raise ValueError("diagnosis requires unaugmented data")

    # Data identity: verify only train/valid payloads; never open test arrays.
    manifest_path = ROOT / "data/crystal_q1_v1/manifest.json"
    data_manifest = json.loads(manifest_path.read_text())
    verified_files = {}
    for name, entry in data_manifest["files"].items():
        if name.startswith(("train/", "valid/")):
            value = file_sha256(manifest_path.parent / name)
            if value != entry["sha256"]:
                raise ValueError("dataset digest mismatch: " + name)
            verified_files[name] = value
    data_manifest_sha = file_sha256(manifest_path)
    for label in LABELS:
        if identities[label]["data_contract"]["manifest_sha256"] != data_manifest_sha:
            raise ValueError(f"{label} refers to a different data revision")
    progress("identities")

    # Pairing on the complete splits; train representations only for paired materials.
    keys_by_split, ids_by_split, pairs_by_split, sizes_by_split, scope_rows = {}, {}, {}, {}, {}
    for split in ("train", "valid"):
        rows = list(range(len(datasets[split])))
        ids = [str(datasets[split].material_ids[row]) for row in rows]
        keys = composition_keys_by_position(datasets[split], rows)
        if smoke:
            rows, ids, keys = smoke_subset(keys, rows, ids)
        frames, sizes = {}, {}
        for grouping in GROUPINGS:
            frames[grouping], sizes[grouping] = pair_index_table(keys[grouping], ids)
            frames[grouping].to_csv(out_dir / f"pairs_{split}_{grouping}.csv", index=False)
        keys_by_split[split], ids_by_split[split], scope_rows[split] = keys, ids, rows
        pairs_by_split[split], sizes_by_split[split] = frames, sizes
    extraction_rows = {split: participant_rows(pairs_by_split[split]) for split in ("train", "valid")}
    pairing_info = {split: pairing_manifest(sizes_by_split[split], pairs_by_split[split])
                    for split in ("train", "valid")}

    manifest = {
        "scope": "frozen C1/C2 stage readout diagnosis; seed 42; train/valid only; "
                 "readouts linear without intercept; no model parameter update",
        "smoke": smoke,
        "seed": SEED,
        "device": str(device),
        "near_zero_target_tv": NEAR_ZERO_TV,
        "groupings": list(GROUPINGS),
        "stages": STAGE_DESCRIPTIONS,
        "task": TASK,
        "target": "unit-sum eDOS shape difference (sum-normalized spectra)",
        "alpha_grid": [float(value) for value in ALPHAS],
        "data_manifest_sha256": data_manifest_sha,
        "verified_train_valid_files": verified_files,
        "checkpoints": {label: {
            "run": identities[label]["run"], "epoch": identities[label]["epoch"],
            "checkpoint_sha256": identities[label]["checkpoint_sha256"]} for label in LABELS},
        "split_scope": {split: {"dataset_rows": scope_rows[split],
                                "material_ids": ids_by_split[split]}
                        for split in ("train", "valid")},
        "pairing": pairing_info,
        "extraction_rows": {split: extraction_rows[split] for split in ("train", "valid")},
        "extraction_material_ids": {
            split: [ids_by_split[split][row] for row in extraction_rows[split]]
            for split in ("train", "valid")},
        "pair_tables_sha256": {
            f"{split}_{grouping}": file_sha256(out_dir / f"pairs_{split}_{grouping}.csv")
            for split in ("train", "valid") for grouping in GROUPINGS},
        "source_sha256": file_sha256(Path(__file__)),
    }
    dump(out_dir / "manifest.json", manifest)
    progress("manifest")

    # Frozen representation extraction, per label.
    store = {}
    extraction_seconds = {}
    state_reports = {}
    label_states = {}
    for label in LABELS:
        run = repository_path(identities[label]["run"])
        build = ConfigBuilder(**identities[label]["config"])
        model = build.get_model()
        transformer = model.model["transformer"]
        transformer.load_state_dict(identities[label]["model_state"], strict=True)
        del identities[label]["model_state"]
        model.to(device)
        transformer.eval()
        before = state_hash(transformer.state_dict())
        entry = {}
        for split in ("train", "valid"):
            rows = extraction_rows[split]
            ids = [ids_by_split[split][row] for row in rows]
            features, spectra, seconds = extraction(
                model, datasets[split], rows, ids, device)
            keys_subset = {grouping: [keys_by_split[split][grouping][row] for row in rows]
                           for grouping in GROUPINGS}
            frames_subset, _ = {}, {}
            for grouping in GROUPINGS:
                frame, _ = pair_index_table(keys_subset[grouping], ids)
                full = pairs_by_split[split][grouping]
                expected = {(left, right) for left, right in zip(full.left_id, full.right_id)
                            if left in set(ids) and right in set(ids)}
                if set(zip(frame.left_id, frame.right_id)) != expected:
                    raise ValueError("subset pairing differs from the complete-split pairing")
                frames_subset[grouping] = frame
            np.savez(out_dir / f"features_{label}_{split}.npz", material_ids=np.asarray(ids),
                     **features)
            np.savez(out_dir / f"spectra_{label}_{split}.npz", material_ids=np.asarray(ids),
                     pred_shape_edos=spectra["pred_shape_edos"], tgt_shape_edos=spectra["tgt_shape_edos"])
            entry[split] = {"materials": len(ids), "seconds": seconds,
                            "features": f"features_{label}_{split}.npz",
                            "spectra": f"spectra_{label}_{split}.npz"}
            store[label, split] = (features, spectra, frames_subset, ids, keys_subset)
            if split == "valid":
                entry[split]["prior_export_agreement"] = prior_agreement(
                    label, ids, spectra["pred_shape_edos"])
            print(json.dumps({"extracted": label, "split": split, "materials": len(ids),
                              "seconds": round(seconds, 2)}), flush=True)
        after = state_hash(transformer.state_dict())
        if before != after or file_sha256(run / "checkpoint_best.pth") != identities[label]["checkpoint_sha256"]:
            raise ValueError("model or checkpoint mutated during extraction")
        state_reports[label] = {"state_hash_before": before, "state_unchanged": True,
                                "hooks_removed": True}
        label_states[label] = state_reports[label]
        extraction_seconds[label] = entry
        del model, transformer, build
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        progress(f"extract_{label}")
    report = {
        "scope": manifest["scope"],
        "smoke": smoke,
        "seed": SEED,
        "device": str(device),
        "extraction": extraction_seconds,
        "stages": {stage: STAGE_DESCRIPTIONS[stage] for stage in STAGES},
        "models": {},
        "comparisons": {},
    }

    # Readout fitting (train only) and locked evaluation on train/valid.
    frames_by_key = {}
    for label in LABELS:
        label_entry = {
            "run": identities[label]["run"], "epoch": identities[label]["epoch"],
            "checkpoint_sha256": identities[label]["checkpoint_sha256"],
            **label_states[label], "groupings": {},
        }
        for grouping in GROUPINGS:
            group_entry = {"stages": {}, "splits": {}}
            fits = {}
            for stage in STAGES:
                train_features = store[label, "train"][0][stage]
                train_frame = store[label, "train"][2][grouping]
                train_shape = store[label, "train"][1][f"tgt_shape_{TASK}"]
                fit = StageFit(stage, train_features, train_frame, train_shape)
                fit.select()
                fit.final_fit()
                fits[stage] = fit
                group_entry["stages"][stage] = fit.describe()
                checks = identity_checks(stage, train_features, fit.scale, fit.weights, train_frame)
                group_entry["stages"][stage]["identity_checks"] = checks
                if checks["swap_antisymmetry_max_abs"] > 1e-9 or \
                        checks["identical_representation_max_abs"] > 1e-9:
                    raise ValueError("readout violates swap antisymmetry or zero-difference")
            for split in ("train", "valid"):
                features, spectra, frames_subset, ids, keys_subset = store[label, split]
                shape = spectra[f"tgt_shape_{TASK}"]
                predictions = {stage: predict_materials(stage, features[stage], fits[stage].scale,
                                                        fits[stage].weights) for stage in STAGES}
                predictions["model"] = spectra[f"pred_shape_{TASK}"]
                predictions["zero_response"] = np.zeros_like(shape)
                frames = build_frames(shape, predictions, keys_subset[grouping], ids)
                for name, frame in frames.items():
                    frames_by_key[label, grouping, split, name] = frame
                    frame.to_csv(out_dir / f"pairs_{label}_{split}_{grouping}_{name}.csv", index=False)
                main = {name: frame[frame.target_tv >= NEAR_ZERO_TV] for name, frame in frames.items()}
                split_entry = {
                    "pairs": len(frames["model"]),
                    "near_zero_pairs": int((frames["model"].target_tv < NEAR_ZERO_TV).sum()),
                    "methods": {name: method_metrics(frame) for name, frame in frames.items()},
                    "versus_zero": {name: compare_frames(main["zero_response"], main[name],
                                                         f"{name} minus zero_response")
                                    for name in METHODS if name != "zero_response"},
                    "versus_model": {name: compare_frames(main["model"], main[name],
                                                          f"{name} minus model")
                                     for name in STAGES},
                }
                group_entry["splits"][split] = split_entry
            label_entry["groupings"][grouping] = group_entry
            for stage in STAGES:
                del fits[stage]
        report["models"][label] = label_entry
        progress(f"fit_{label}")

    # Stage-to-stage and C1-to-C2 readability differences under this readout protocol.
    for grouping in GROUPINGS:
        report["comparisons"][grouping] = {}
        for split in ("train", "valid"):
            block = {}
            for left, right in ((STAGES[0], STAGES[1]), (STAGES[1], STAGES[2]), (STAGES[0], STAGES[2])):
                left_frame = frames_by_key["C1", grouping, split, left]
                right_frame = frames_by_key["C1", grouping, split, right]
                left_main = left_frame[left_frame.target_tv >= NEAR_ZERO_TV]
                right_main = right_frame[right_frame.target_tv >= NEAR_ZERO_TV]
                block[f"C1_{right}_minus_{left}"] = compare_frames(
                    left_main, right_main, f"{right} minus {left}")
            for stage in STAGES:
                c1 = frames_by_key["C1", grouping, split, stage]
                c2 = frames_by_key["C2", grouping, split, stage]
                c1_main = c1[c1.target_tv >= NEAR_ZERO_TV]
                c2_main = c2[c2.target_tv >= NEAR_ZERO_TV]
                block[f"C2_{stage}_minus_C1"] = compare_frames(c1_main, c2_main,
                                                               f"C2 {stage} minus C1 {stage}")
            report["comparisons"][grouping][split] = block

    report["seconds"] = time.monotonic() - start
    report["peak_vram_mb"] = (torch.cuda.max_memory_allocated() / 2**20
                              if torch.cuda.is_available() else 0.0)
    report["artifact_sha256"] = {path.name: file_sha256(path) for path in sorted(out_dir.iterdir())
                                 if path.is_file() and path.name not in
                                 ("report.json", "report.partial.json")}
    report["limitations"] = [
        "Single training seed (42) and one checkpoint per model; intervals resample "
        "composition groups and do not cover training-seed variation.",
        "The atom stages use permutation-invariant mean/RMS pooling of the outer state; "
        "a negative linear probe on a lossy summary cannot prove the full representation "
        "lacks information.",
        "The decoder stage reads each energy position with its own linear map; cross-energy "
        "linear maps are not tested here.",
        "Stage differences are readability differences under this readout protocol, not "
        "evidence that information was deleted.",
        "Near-zero target gaps (target TV below the recorded threshold) are excluded from "
        "the main metrics and reported separately.",
        "Blind/oracle scale calibration is out of scope; targets and predictions are the "
        "unit-sum eDOS shape space.",
    ]
    dump(out_dir / "report.json", report)
    progress("report")
    print(json.dumps({"complete": str(out_dir), "seconds": report["seconds"]}), flush=True)


def smoke_subset(keys, rows, ids, limit_groups=8):
    """Small deterministic subset: materials of the first few multi-member groups."""
    selected = set()
    for grouping in GROUPINGS:
        _, groups = same_composition_pairs(keys[grouping])
        multi = sorted((members for members in groups.values() if len(members) > 1),
                       key=lambda members: members[0])
        for members in multi[:limit_groups]:
            selected.update(members)
    if len(selected) < 4:
        raise ValueError("smoke subset needs at least four paired materials")
    picked = sorted(selected)
    return ( [rows[i] for i in picked],
             [ids[i] for i in picked],
             {grouping: [keys[grouping][i] for i in picked] for grouping in GROUPINGS} )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="first paired composition groups per split; plumbing and timing only")
    args = parser.parse_args()
    diagnose(args.out_dir, args.smoke)


if __name__ == "__main__":
    main()
