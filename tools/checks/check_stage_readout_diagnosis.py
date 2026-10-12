"""Independent replay of the frozen stage readout diagnosis.

Re-extracts the three frozen stages with separately written hooks and pooling,
re-fits the train-only readouts with a separately written weighted-ridge solver,
replays the composition-grouped CV selection and every reported pair metric, and
verifies manifest fingerprints, material ordering and the frozen model state.
Nothing here imports the diagnosis fitting helpers; shared utilities are only the
established loading, identity and pairing functions. Train/valid only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from tools.eval.lg_structure_diagnosis import loader
from tools.eval.lg_train_subset_diagnosis import composition_keys_by_position, same_composition_pairs
from tools.eval.lg_valid_export import load_run, select_best
from tools.eval.local_structure_compare import verify_completed_control, verify_preflight
from tools.eval.local_structure_diagnosis import run_directory
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
PREFLIGHT_PATH = ATTRIBUTION / "preflight_r1.json"
ATOM_STAGES = ("local_last", "global_last")
STAGES = (*ATOM_STAGES, "decoder_edos")
METHODS = (*STAGES, "model", "zero_response")
GROUPINGS = ("absolute", "reduced")
SPLITS = ("train", "valid")
NEAR_ZERO_TV = 1e-6
SCALE_FLOOR = 1e-6
TOL = (1e-6, 1e-5)          # locked statistics: FP64 solve-path rounding only
TOL_CV = (1e-6, 1e-3)       # CV scores: tiny alphas are ill-conditioned, one sweep each
TOL_PAIR = (1e-6, 1e-5)     # replayed pair columns
TOL_FEATURE = (1e-5, 2e-4)  # GPU forward re-extraction agreement


def check(condition, message, payload, checks):
    checks.append({"name": message, "passed": bool(condition), "detail": str(payload)[:200]})
    if not condition:
        raise ValueError(message)


def assert_close(message, actual, expected, checks, tol=TOL):
    difference = abs(float(actual) - float(expected))
    allowed = tol[0] + tol[1] * abs(float(expected))
    checks.append({"name": message, "passed": difference <= allowed,
                   "abs_diff": difference, "allowed": allowed})
    if difference > allowed:
        raise ValueError(f"{message}: |{actual} - {expected}| = {difference} > {allowed}")


def assert_array_close(message, actual, expected, checks, tol=TOL_FEATURE):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise ValueError(f"{message}: shape {actual.shape} != {expected.shape}")
    difference = float(np.max(np.abs(actual - expected))) if expected.size else 0.0
    allowed = tol[0] + tol[1] * (float(np.max(np.abs(expected))) if expected.size else 0.0)
    checks.append({"name": message, "passed": difference <= allowed,
                   "max_abs_diff": difference, "allowed": allowed})
    if difference > allowed:
        raise ValueError(f"{message}: max |difference| = {difference} > {allowed}")


class ReplayExtractor:
    """Separately written read-only hooks; mean/RMS pooling masks padding."""

    def __init__(self, transformer):
        encoder = transformer.encoder
        last = {kind: index for kind, index in encoder.execution_order}
        self.mask = None
        self.call = 0
        self.values = {name: [] for name in STAGES}
        self.hooks = [transformer.register_forward_pre_hook(self.begin)]
        self.hooks.append(encoder.local_layers[last["local"]].register_forward_hook(
            self.atom("local_last")))
        self.hooks.append(encoder.global_layers[last["global"]].register_forward_hook(
            self.atom("global_last")))
        self.hooks.append(transformer.decoder.register_forward_hook(self.decoder))

    def begin(self, module, args):
        self.mask = atom_padding_mask(args[2], args[1])
        self.call = 0

    def atom(self, name):
        def hook(module, args, output):
            keep = (~self.mask).unsqueeze(-1).to(output.dtype)
            count = keep.sum(1).clamp_min(1)
            picked = output * keep
            mean = picked.sum(1) / count
            square = (picked * picked).sum(1) / count
            self.values[name].append(torch.cat([mean, square.sqrt()], dim=1)
                                     .detach().float().cpu().numpy())
        return hook

    def decoder(self, module, args, output):
        if self.call == 0:
            self.values["decoder_edos"].append(output[0].detach().float().cpu().numpy())
        self.call += 1

    def stacked(self):
        return {name: np.concatenate(chunks, axis=0) for name, chunks in self.values.items()}

    def close(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


def replay_scale(values, rows):
    rows = np.asarray(rows, int)
    if values.ndim == 2:
        picked = values[rows].astype(np.float64)
        return np.maximum(np.sqrt(np.square(picked).mean(0)), SCALE_FLOOR)
    total = np.zeros(values.shape[1:], dtype=np.float64)
    for row in rows:
        total += np.square(values[row].astype(np.float64))
    return np.maximum(np.sqrt(total / len(rows)), SCALE_FLOOR)


def replay_weights(group_labels):
    group_labels = np.asarray(group_labels)
    counts = pd.Series(group_labels).value_counts()
    weights = np.array([1.0 / counts[label] for label in group_labels])
    return weights * (len(weights) / weights.sum())


def replay_group_mean(values, group_labels):
    frame = pd.DataFrame({"composition": np.asarray(group_labels),
                          "value": np.asarray(values, dtype=np.float64)})
    return float(frame.groupby("composition").value.mean().mean())


def replay_group_values(values, group_labels):
    frame = pd.DataFrame({"composition": np.asarray(group_labels),
                          "value": np.asarray(values, dtype=np.float64)})
    return frame.groupby("composition").value.mean().to_numpy()


def replay_bootstrap(values, seed=42, draws=2000):
    grouped = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(grouped, len(grouped), replace=True).mean()
                     for _ in range(draws)])
    return np.percentile(boot, [2.5, 97.5]).tolist()


def replay_fit(stage, values, left, right, scale, target, alpha, weights):
    """Weighted ridge from explicit normal equations; energy positions one by one."""
    root = np.sqrt(weights)
    scale = np.asarray(scale, dtype=np.float64)
    if stage in ATOM_STAGES:
        scaled = values.astype(np.float64) / scale
        design = scaled[left] - scaled[right]
        weighted = design * root[:, None]
        gram = weighted.T @ weighted
        rhs = weighted.T @ (target * root[:, None])
        return np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)
    n_pos, n_features = values.shape[1], values.shape[2]
    fitted = np.empty((n_pos, n_features), dtype=np.float64)
    for position in range(n_pos):
        block = values[:, position, :].astype(np.float64) / scale[position]
        design = block[left] - block[right]
        weighted = design * root[:, None]
        gram = weighted.T @ weighted
        rhs = weighted.T @ (target[:, position] * root)
        fitted[position] = np.linalg.solve(gram + alpha * np.eye(n_features), rhs)
    return fitted


def replay_predict(stage, values, scale, fitted):
    """Per-material readout spectra; pair gaps are differences of these."""
    scale = np.asarray(scale, dtype=np.float64)
    if stage in ATOM_STAGES:
        return values.astype(np.float64) / scale @ fitted
    n_rows, n_pos = values.shape[0], values.shape[1]
    out = np.empty((n_rows, n_pos), dtype=np.float64)
    for position in range(n_pos):
        out[:, position] = (values[:, position, :].astype(np.float64)
                            / scale[position]) @ fitted[position]
    return out


def replay_pair_delta(stage, values, left, right, scale, fitted):
    """Readout prediction of the pair spectral difference."""
    scale = np.asarray(scale, dtype=np.float64)
    if stage in ATOM_STAGES:
        scaled = values.astype(np.float64) / scale
        return (scaled[left] - scaled[right]) @ fitted
    n_pos = values.shape[1]
    out = np.empty((len(left), n_pos), dtype=np.float64)
    for position in range(n_pos):
        block = values[:, position, :].astype(np.float64) / scale[position]
        out[:, position] = (block[left] - block[right]) @ fitted[position]
    return out


def replay_pair_table(keys, ids):
    pairs, _ = same_composition_pairs(keys)
    return pd.DataFrame({
        "i": [int(i) for i, _ in pairs],
        "j": [int(j) for _, j in pairs],
        "left_id": [str(ids[i]) for i, _ in pairs],
        "right_id": [str(ids[j]) for _, j in pairs],
        "composition": [repr(keys[i]) for i, _ in pairs],
    })


def replay_pair_metrics(pred, shape, frame):
    """Independent restatement of the signed pair-difference formulas."""
    left = frame.i.to_numpy(int)
    right = frame.j.to_numpy(int)
    target = shape[left] - shape[right]
    predicted = pred[left] - pred[right]
    tv = 0.5 * np.abs(target).sum(-1)
    error = 0.5 * np.abs(predicted - target).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        skill = np.where(tv > 1e-10, 1.0 - error / tv, np.nan)
        denom = np.linalg.norm(target, axis=1) * np.linalg.norm(predicted, axis=1)
        cosine = np.where(denom > 1e-12, (predicted * target).sum(-1) / denom, np.nan)
    return pd.DataFrame({
        "left_id": frame.left_id.to_numpy(), "right_id": frame.right_id.to_numpy(),
        "composition": frame.composition.to_numpy(), "target_tv": tv,
        "predicted_tv": 0.5 * np.abs(predicted).sum(-1),
        "difference_error": error, "skill": skill, "direction_cosine": cosine})


def replay_interval(left, right):
    delta = right.difference_error.to_numpy() - left.difference_error.to_numpy()
    grouped = pd.DataFrame({"composition": left.composition.to_numpy(),
                            "delta": delta}).groupby("composition").delta.mean().to_numpy()
    return {"groups": int(len(grouped)), "mean_group_error_delta": float(grouped.mean()),
            "ci95": replay_bootstrap(grouped)}


def verify(diagnosis_dir, out, skip_extraction=False):
    checks = []
    report = json.loads((diagnosis_dir / "report.json").read_text())
    manifest = json.loads((diagnosis_dir / "manifest.json").read_text())

    # Fingerprints: source, checkpoints, artifacts, data manifest and payloads.
    check(file_sha256(ROOT / "tools/eval/stage_readout_diagnosis.py") == manifest["source_sha256"],
          "diagnosis source unchanged since the run", manifest["source_sha256"], checks)
    data_manifest = ROOT / "data/crystal_q1_v1/manifest.json"
    check(file_sha256(data_manifest) == manifest["data_manifest_sha256"],
          "data manifest unchanged", manifest["data_manifest_sha256"], checks)
    for name, digest in manifest["verified_train_valid_files"].items():
        check(file_sha256(data_manifest.parent / name) == digest,
              f"train/valid payload unchanged: {name}", name, checks)
    for name, digest in report["artifact_sha256"].items():
        path = diagnosis_dir / name
        check(path.exists() and file_sha256(path) == digest,
              f"artifact unchanged: {name}", name, checks)

    # Model identities and frozen state.
    reference_dir = repository_path(DEFAULT_REFERENCE)
    _, reference_cfg = load_control_reference(reference_dir)
    controls = structure_control_configs(reference_cfg, "attrib20261009")
    preflight = verify_preflight(PREFLIGHT_PATH, reference_dir, controls)
    labels = list(manifest["checkpoints"])
    identities = {}
    for label in labels:
        run = run_directory(label, reference_dir, controls, preflight)
        if label == "C1":
            saved, picked = verify_completed_control(
                next(c for c in controls if c.local_route == label), preflight)
        else:
            saved, _, contract = load_run(run)
            picked = select_best(run, contract["selection_metric"], saved)
        check(file_sha256(run / "checkpoint_best.pth")
              == manifest["checkpoints"][label]["checkpoint_sha256"],
              f"checkpoint unchanged: {label}", label, checks)
        check(int(picked["row"].epoch) == manifest["checkpoints"][label]["epoch"],
              f"checkpoint epoch unchanged: {label}", label, checks)
        identities[label] = (saved, picked["checkpoint"]["model"])
        del picked["checkpoint"]

    # Pairing and material ordering over the complete splits.
    builder = ConfigBuilder(**identities["C1"][0]["config"])
    datasets = {split: builder.get_dataset(split=split, dos_minmax=True, dos_sumnorm=True)
                for split in SPLITS}
    keys_by_split, rows_by_split, ids_by_split = {}, {}, {}
    for split in SPLITS:
        scope = manifest["split_scope"][split]
        rows = [int(row) for row in scope["dataset_rows"]]
        ids = [str(datasets[split].material_ids[row]) for row in rows]
        check(ids == scope["material_ids"],
              f"analysis scope ids match the dataset: {split}", split, checks)
        if not manifest["smoke"]:
            check(rows == list(range(len(datasets[split]))),
                  f"analysis scope is the complete split: {split}", split, checks)
        keys = composition_keys_by_position(datasets[split], rows)
        extraction_rows = [int(row) for row in manifest["extraction_rows"][split]]
        check([ids[row] for row in extraction_rows]
              == manifest["extraction_material_ids"][split],
              f"material order matches the dataset: {split}", split, checks)
        keys_by_split[split], rows_by_split[split], ids_by_split[split] = keys, extraction_rows, ids
        for grouping in GROUPINGS:
            saved = pd.read_csv(diagnosis_dir / f"pairs_{split}_{grouping}.csv")
            replayed = replay_pair_table(keys[grouping], ids)
            check(list(saved.left_id) == list(replayed.left_id)
                  and list(saved.right_id) == list(replayed.right_id)
                  and list(saved.composition) == list(replayed.composition),
                  f"analysis-scope pair table reproduced: {split} {grouping}", grouping, checks)
            check(len(saved) == manifest["pairing"][split][grouping]["pairs"],
                  f"pair count recorded: {split} {grouping}", grouping, checks)

    # Frozen feature re-extraction with the separately written hooks.
    if not skip_extraction:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for label in labels:
            saved, model_state = identities[label]
            build = ConfigBuilder(**saved["config"])
            model = build.get_model()
            transformer = model.model["transformer"]
            transformer.load_state_dict(model_state, strict=True)
            del model_state
            model.to(device)
            transformer.eval()
            before = state_hash(transformer.state_dict())
            check(before == report["models"][label]["state_hash_before"],
                  f"frozen model state unchanged: {label}", label, checks)
            for split in SPLITS:
                rows = rows_by_split[split]
                ids = [ids_by_split[split][row] for row in rows]
                extractor = ReplayExtractor(transformer)
                with torch.no_grad():
                    measured = evaluate_split(model, loader(datasets[split], rows), sumnorm=True,
                                              return_sample_level=True, return_spectra=True,
                                              sample_ids=ids)
                extractor.close()
                check(not extractor.hooks, f"hooks removed after replay: {label} {split}",
                      label, checks)
                replayed = extractor.stacked()
                with np.load(diagnosis_dir / f"features_{label}_{split}.npz") as stored:
                    check(list(stored["material_ids"]) == ids,
                          f"feature material order: {label} {split}", split, checks)
                    for stage in STAGES:
                        assert_array_close(f"re-extracted {stage}: {label} {split}",
                                           replayed[stage], stored[stage], checks)
                with np.load(diagnosis_dir / f"spectra_{label}_{split}.npz") as stored:
                    assert_array_close(f"re-extracted pred shape: {label} {split}",
                                       measured["sample_spectra"]["pred_shape_edos"],
                                       stored["pred_shape_edos"], checks)
            check(before == state_hash(transformer.state_dict()),
                  f"model state unchanged during replay: {label}", label, checks)
            del model, transformer, build
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Independent refit on train only, then locked metrics on train and valid.
    for label in labels:
        features = {split: {} for split in SPLITS}
        shapes = {}
        model_predictions = {}
        for split in SPLITS:
            with np.load(diagnosis_dir / f"features_{label}_{split}.npz") as stored:
                for stage in STAGES:
                    features[split][stage] = stored[stage]
            with np.load(diagnosis_dir / f"spectra_{label}_{split}.npz") as stored:
                shapes[split] = stored["tgt_shape_edos"]
                model_predictions[split] = stored["pred_shape_edos"]
        for grouping in GROUPINGS:
            tables = {}
            for split in SPLITS:
                keys = [keys_by_split[split][grouping][row] for row in rows_by_split[split]]
                ids = [ids_by_split[split][row] for row in rows_by_split[split]]
                tables[split] = replay_pair_table(keys, ids)
            train_table = tables["train"]
            left = train_table.i.to_numpy(int)
            right = train_table.j.to_numpy(int)
            groups = train_table.composition.to_numpy()
            target = shapes["train"][left] - shapes["train"][right]
            tv = 0.5 * np.abs(target).sum(-1)
            unique = np.unique(groups)
            splitter = GroupKFold(n_splits=int(min(5, len(unique))))
            for stage in STAGES:
                stage_record = report["models"][label]["groupings"][grouping]["stages"][stage]
                check(stage_record["fit_scope"].startswith("train pairs only"),
                      f"fit scope is train-only: {label} {grouping} {stage}", stage, checks)
                values = features["train"][stage]
                folds = []
                for train_idx, val_idx in splitter.split(np.zeros(len(groups)), groups=groups):
                    slots = np.unique(np.concatenate([left[train_idx], right[train_idx]]))
                    folds.append((train_idx, val_idx, replay_scale(values, slots)))
                trace = {}
                for alpha in [float(key) for key in stage_record["alpha_grid"]]:
                    scores = []
                    for train_idx, val_idx, scale in folds:
                        fitted = replay_fit(stage, values, left[train_idx], right[train_idx],
                                            scale, target[train_idx], alpha,
                                            replay_weights(groups[train_idx]))
                        pred = replay_pair_delta(stage, values, left[val_idx], right[val_idx],
                                                 scale, fitted)
                        keep = tv[val_idx] >= NEAR_ZERO_TV
                        error = 0.5 * np.abs(pred[keep] - target[val_idx][keep]).sum(-1)
                        scores.append(replay_group_mean(error, groups[val_idx][keep]))
                    trace[str(alpha)] = float(np.mean(scores))
                for key, value in stage_record["cv_group_mean_pair_error"].items():
                    assert_close(f"cv score {label} {grouping} {stage} alpha={key}",
                                 trace[key], value, checks, tol=TOL_CV)
                selected = min(trace, key=lambda key: (trace[key], -float(key)))
                check(float(selected) == stage_record["selected_alpha"],
                      f"selected alpha reproduced: {label} {grouping} {stage}", stage, checks)
                checks.append({"name": f"identity contract {label} {grouping} {stage}",
                               "passed": bool(
                                   stage_record["identity_checks"]["swap_antisymmetry_max_abs"]
                                   <= 1e-9 and stage_record["identity_checks"][
                                       "identical_representation_max_abs"] <= 1e-9)})
                # Locked train-only fit; valid is only evaluated afterwards.
                slots = np.unique(np.concatenate([left, right]))
                scale = replay_scale(values, slots)
                fitted = replay_fit(stage, values, left, right, scale, target,
                                    stage_record["selected_alpha"], replay_weights(groups))
                zero = {split: np.zeros_like(shapes[split]) for split in SPLITS}
                stage_pred = {split: replay_predict(stage, features[split][stage],
                                                    scale, fitted) for split in SPLITS}
                method_predictions = {
                    **{name: {split: stage_pred[split] for split in SPLITS} for name in [stage]},
                    "model": model_predictions,
                    "zero_response": zero,
                }
                for name, predictions in method_predictions.items():
                    for split in SPLITS:
                        frame = replay_pair_metrics(predictions[split], shapes[split], tables[split])
                        saved_frame = pd.read_csv(
                            diagnosis_dir / f"pairs_{label}_{split}_{grouping}_{name}.csv")
                        assert_array_close(f"pair error column: {label} {split} {grouping} {name}",
                                           frame.difference_error, saved_frame.difference_error,
                                           checks, tol=TOL_PAIR)
                        recorded = report["models"][label]["groupings"][grouping][
                            "splits"][split]["methods"][name]
                        main = frame[frame.target_tv >= NEAR_ZERO_TV]
                        assert_close(f"group mean error: {label} {split} {grouping} {name}",
                                     replay_group_mean(main.difference_error, main.composition),
                                     recorded["group_mean_error"]["mean_group"], checks)
                        assert_close(f"group mean skill: {label} {split} {grouping} {name}",
                                     replay_group_mean(main.skill.to_numpy(), main.composition),
                                     recorded["skill"]["mean_group"], checks)
                        assert_close(f"skill median: {label} {split} {grouping} {name}",
                                     float(main.skill.median()), recorded["skill_median"], checks)
                        cosine = main.dropna(subset=["direction_cosine"])
                        expected_cosine = (float(cosine.direction_cosine.median())
                                           if len(cosine) else None)
                        if expected_cosine is None:
                            check(recorded["direction_cosine_median"] is None,
                                  f"undefined cosine at zero response: {label} {split} "
                                  f"{grouping} {name}", name, checks)
                        else:
                            assert_close(f"cosine median: {label} {split} {grouping} {name}",
                                         expected_cosine, recorded["direction_cosine_median"],
                                         checks)
                        assert_close(f"near-zero count: {label} {split} {grouping} {name}",
                                     len(frame) - len(main), recorded["near_zero_pairs"], checks)
                        assert_array_close(f"group mean error ci: {label} {split} {grouping} {name}",
                                           replay_bootstrap(
                                               replay_group_values(main.difference_error,
                                                                   main.composition)),
                                           recorded["group_mean_error"]["ci95"], checks, tol=TOL)
                    # Locked comparisons: stage against the strict zero response and model.
                    for split in SPLITS:
                        frames = {key: replay_pair_metrics(predictions[split], shapes[split],
                                                           tables[split])
                                  for key, predictions in method_predictions.items()}
                        keep = {key: value[value.target_tv >= NEAR_ZERO_TV]
                                for key, value in frames.items()}
                        if name != "zero_response":
                            versus = report["models"][label]["groupings"][grouping][
                                "splits"][split]["versus_zero"][name]
                            interval = replay_interval(keep["zero_response"], keep[name])
                            assert_close(f"versus zero delta: {label} {split} {grouping} {name}",
                                         interval["mean_group_error_delta"],
                                         versus["error_delta_right_minus_left"][
                                             "mean_group_error_delta"], checks)
                            assert_array_close(f"versus zero ci: {label} {split} {grouping} {name}",
                                               interval["ci95"],
                                               versus["error_delta_right_minus_left"]["ci95"],
                                               checks, tol=TOL)
                        if name in STAGES:
                            versus_model = report["models"][label]["groupings"][grouping][
                                "splits"][split]["versus_model"][name]
                            interval = replay_interval(keep["model"], keep[name])
                            assert_close(f"versus model delta: {label} {split} {grouping} {name}",
                                         interval["mean_group_error_delta"],
                                         versus_model["error_delta_right_minus_left"][
                                             "mean_group_error_delta"], checks)

    out.write_text(json.dumps({
        "scope": "independent replay of the frozen stage readout diagnosis; train/valid only",
        "passed": True, "checks": len(checks), "entries": checks,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "checks": len(checks)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-extraction", action="store_true",
                        help="replay fits and metrics from the saved features only")
    args = parser.parse_args()
    verify(args.diagnosis_dir, args.out, args.skip_extraction)


if __name__ == "__main__":
    main()
