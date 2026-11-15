"""Train-fitted, per-atom spectral energy normalization for joint blind learning.

This is an opt-in protocol. Legacy H1 reconstructions and scores are unchanged.
The normalizer is a task-level training statistic, not a per-material variance
or an electron/mode-count sum rule. All bins, including zero targets, contribute.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from utils.paths import file_sha256, input_path

TASKS = ("edos", "phdos")
ENERGY_METRIC = "balanced_blind_energy"
BALANCE_SCHEMA = "joint-blind-spectral-energy-v1"
BALANCE_SCHEMA_V2 = "joint-blind-spectral-energy-v2"
REFERENCE_MODES = ("none", "fixed")


def label_contract(value):
    """Label contract of a view manifest or protocol; absence means the frozen v1 labels."""
    return value.get("label_contract", "v1")


def fit_task_normalizer(target, atoms, bin_width):
    """Fit C=mean_train integral[(DOS/N)^2] and a numerical mass reference."""
    target = np.asarray(target, dtype=np.float64)
    atoms = np.asarray(atoms, dtype=np.float64)
    if (target.ndim != 2 or not len(target) or target.shape[1] < 2
            or atoms.shape != (len(target),) or not np.isfinite(target).all()
            or (target < 0).any() or not np.isfinite(atoms).all() or (atoms <= 0).any()
            or not math.isfinite(bin_width) or bin_width <= 0):
        raise ValueError("finite nonnegative spectra, positive atom counts and bin width required")
    intensive = target / atoms[:, None]
    energy = float(np.square(intensive).sum(-1).mean() * bin_width)
    mass = float(np.median(intensive.sum(-1) * bin_width))
    if energy <= 0 or mass <= 0 or not math.isfinite(energy + mass):
        raise ValueError("training task needs positive finite energy and median mass")
    return {"energy": energy, "mass_per_atom_reference": mass,
            "bin_width": float(bin_width), "bins": target.shape[1]}


def validate_balance_protocol(protocol):
    if isinstance(protocol, dict) and protocol.get("schema") == BALANCE_SCHEMA_V2:
        return _validate_balance_protocol_v2(protocol)
    if (not isinstance(protocol, dict) or protocol.get("schema") != BALANCE_SCHEMA
            or "label_contract" in protocol
            or protocol.get("fit_split") != "train"
            or protocol.get("task_weights") != {"edos": .5, "phdos": .5}
            or set(protocol.get("tasks", {})) != set(TASKS)):
        raise ValueError("invalid joint spectral energy protocol")
    for task in TASKS:
        entry = protocol["tasks"][task]
        if any(not math.isfinite(float(entry.get(key, float("nan")))) or float(entry[key]) <= 0
               for key in ("energy", "mass_per_atom_reference", "bin_width")):
            raise ValueError("positive finite train-fitted task normalizers required")
        if not isinstance(entry.get("bins"), int) or entry["bins"] < 2:
            raise ValueError("task bin counts required")
    reference = protocol.get("reference", {})
    count = reference.get("samples", 0)
    if not isinstance(count, int) or count < 1 or reference.get("split") != "valid":
        raise ValueError("a fixed complete-valid reference is required")
    for task in TASKS:
        failed = reference.get(f"failures_{task}", -1)
        score = float(reference.get(f"r2_{task}_median", float("nan")))
        if not isinstance(failed, int) or not 0 <= failed <= count or not math.isfinite(score):
            raise ValueError("finite per-task reference R2 and failure counts required")
    if (protocol.get("failure_increase_limit_pp") != 0.
            or protocol.get("r2_drop_limit") != .02):
        raise ValueError("fixed dual-task guards require zero failure increase and R2 drop <= .02")
    return protocol


def _validate_balance_protocol_v2(protocol):
    """v2 labels: same normalizer and guard definitions; the guard reference may be absent.

    Without a reference ("none") every epoch is eligible; the first v2 run of the working
    point is selected this way and its valid export then becomes the fixed reference.
    """
    if (protocol.get("label_contract") != "v2" or protocol.get("fit_split") != "train"
            or protocol.get("task_weights") != {"edos": .5, "phdos": .5}
            or set(protocol.get("tasks", {})) != set(TASKS)):
        raise ValueError("invalid v2 joint spectral energy protocol")
    view = protocol.get("view", {})
    if not isinstance(view.get("valid_split"), str) or not isinstance(view.get("partial_labels"), bool):
        raise ValueError("v2 protocol must name its valid split and partial-label mode")
    for task in TASKS:
        entry = protocol["tasks"][task]
        if any(not math.isfinite(float(entry.get(key, float("nan")))) or float(entry[key]) <= 0
               for key in ("energy", "mass_per_atom_reference", "bin_width")):
            raise ValueError("positive finite train-fitted task normalizers required")
        if not isinstance(entry.get("bins"), int) or entry["bins"] < 2:
            raise ValueError("task bin counts required")
        if not isinstance(protocol.get("fit_samples", {}).get(task), int) or protocol["fit_samples"][task] < 1:
            raise ValueError("per-task labelled train counts required")
    reference = protocol.get("reference", {})
    count = reference.get("samples", 0)
    if (reference.get("mode") not in REFERENCE_MODES or reference.get("split") != "valid"
            or not isinstance(count, int) or count < 1):
        raise ValueError("a complete-valid cohort and a reference mode are required")
    if reference["mode"] == "fixed":
        if not isinstance(reference.get("label"), str):
            raise ValueError("a fixed reference must be labelled")
        for task in TASKS:
            failed = reference.get(f"failures_{task}", -1)
            score = float(reference.get(f"r2_{task}_median", float("nan")))
            if not isinstance(failed, int) or not 0 <= failed <= count or not math.isfinite(score):
                raise ValueError("finite per-task reference R2 and failure counts required")
    if (protocol.get("failure_increase_limit_pp") != 0.
            or protocol.get("r2_drop_limit") != .02):
        raise ValueError("fixed dual-task guards require zero failure increase and R2 drop <= .02")
    return protocol


def reconstruct_balanced_spectra(outputs, n_atoms, protocol):
    """Target-free reconstruction: q * N * mass_reference * softplus(u) / dx.

    Outputs must contain raw shape logits and two raw mass logits, ordered e/p.
    No N_valence, target total, coverage mask or upper mass bound is consumed.
    """
    import torch
    import torch.nn.functional as F

    validate_balance_protocol(protocol)
    e, p, mass_logits = (outputs[key] for key in ("edos", "phdos", "mass_logits"))
    if (e.ndim != 2 or p.ndim != 2 or p.shape[0] != e.shape[0]
            or mass_logits.shape != (e.shape[0], 2)):
        raise ValueError("aligned [batch, bins] logits and [batch, 2] mass logits required")
    atoms = torch.as_tensor(n_atoms, device=e.device, dtype=e.dtype).reshape(-1, 1)
    if (atoms.shape != (len(e), 1) or not torch.isfinite(atoms).all() or (atoms < 0).any()
            or not all(torch.isfinite(value).all() for value in (e, p, mass_logits))):
        raise ValueError("finite aligned logits and nonnegative atom counts required")
    result = {}
    for i, (task, logits) in enumerate(zip(TASKS, (e, p))):
        entry = protocol["tasks"][task]
        if logits.shape[1] != entry["bins"]:
            raise ValueError("spectral logits differ from the frozen production grid")
        shape = logits.softmax(-1)
        mass = F.softplus(mass_logits[:, i:i + 1]) * entry["mass_per_atom_reference"]
        prediction = shape * atoms * mass / entry["bin_width"]
        if not torch.isfinite(prediction).all():
            raise ValueError("non-finite reconstructed spectra")
        result[f"shape_{task}"] = shape
        result[f"blind_{task}"] = prediction
    return result


def normalized_spectral_error(prediction, target, n_atoms, task_normalizer):
    """Per-material integrated squared error / N^2 / train task energy.

    Zero targets are scored normally. There is no target-dependent denominator,
    epsilon, loss clipping or per-material normalization by a near-zero spectrum.
    """
    import torch

    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("prediction and target must have matching [batch, bins] shapes")
    atoms = torch.as_tensor(n_atoms, dtype=prediction.dtype, device=prediction.device).reshape(-1)
    if (atoms.shape != (len(prediction),) or not torch.isfinite(atoms).all() or (atoms <= 0).any()
            or not torch.isfinite(target).all() or (target < 0).any()
            or not torch.isfinite(prediction).all() or (prediction < 0).any()):
        raise ValueError("finite nonnegative spectra and positive per-material atom counts required")
    energy, width = (float(task_normalizer[key]) for key in ("energy", "bin_width"))
    if not all(math.isfinite(value) and value > 0 for value in (energy, width)):
        raise ValueError("positive train energy and bin width required")
    if prediction.shape[1] != task_normalizer["bins"]:
        raise ValueError("normalization grid mismatch")
    error = ((prediction - target) / atoms[:, None]).square().sum(-1) * (width / energy)
    if not torch.isfinite(error).all():
        raise ValueError("non-finite normalized spectral error")
    return error


def balanced_energy_score(metrics):
    values = [float(metrics[f"blind_energy_{task}_mean"]) for task in TASKS]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("finite nonnegative mean blind errors required for both tasks")
    return .5 * (values[0] + values[1])


def balanced_checkpoint_eligible(metrics, protocol):
    """Both tasks must pass independently; percentage values are in pp units."""
    validate_balance_protocol(protocol)
    ref = protocol["reference"]
    if float(metrics["selection_samples"]) != ref["samples"]:
        raise ValueError("checkpoint guards require the same complete valid cohort")
    if ref.get("mode") == "none":
        return True
    for task in TASKS:
        rate, r2 = (float(metrics[key]) for key in
                    (f"blind_fail_rate_{task}", f"blind_r2_{task}_median"))
        if not math.isfinite(rate + r2) or not 0 <= rate <= 100:
            raise ValueError("finite complete-valid R2 and failure percentages required")
        observed_count = rate * ref["samples"] / 100
        if not math.isclose(observed_count, round(observed_count), rel_tol=0, abs_tol=1e-8):
            raise ValueError("failure rate must represent an integer count on the fixed valid cohort")
        if (round(observed_count) > ref[f"failures_{task}"]
                or r2 < ref[f"r2_{task}_median"] - protocol["r2_drop_limit"] - 1e-12):
            return False
    return True


def energy_selection_contract(protocol):
    validate_balance_protocol(protocol)
    return {"checkpoint_selection": {
        "schema": "joint-blind-energy-selection-v1", "metric": ENERGY_METRIC,
        "mode": "min", "split": "valid", "task_weights": {"edos": .5, "phdos": .5},
        "formula": "0.5*mean(blind_energy_edos) + 0.5*mean(blind_energy_phdos)",
        "tie_break": "earliest_epoch", "reference": deepcopy(protocol["reference"]),
        "failure_increase_limit_pp": 0., "r2_drop_limit": .02,
        "no_eligible_epoch": "no checkpoint_best; retain latest; report no_eligible_best",
        "diagnostics_only": ["oracle_mae", "same_composition_spectral_difference"],
    }}


def selected_energy_history_row(history, protocol):
    """Re-derive eligibility and score. Return None if every epoch is ineligible."""
    required = {"epoch", "balanced_score", ENERGY_METRIC, "selection_samples",
                "checkpoint_eligible"} | {f"{prefix}_{task}{suffix}" for task in TASKS
        for prefix, suffix in (("blind_energy", "_mean"), ("blind_r2", "_median"),
                               ("blind_fail_rate", ""))}
    if history.empty or not required.issubset(history.columns) or history.epoch.duplicated().any():
        raise ValueError("incomplete or duplicated joint blind energy history")
    epochs = history.epoch.to_numpy(float)
    if not np.isfinite(epochs).all() or (epochs < 1).any() or (epochs != np.floor(epochs)).any():
        raise ValueError("finite positive integer history epochs required")
    if sorted(epochs.tolist()) != list(range(1, int(epochs.max()) + 1)):
        raise ValueError("balanced selection requires an unbroken history from epoch one")
    records = history.to_dict("records")
    expected = np.asarray([balanced_energy_score(row) for row in records])
    eligible = np.asarray([balanced_checkpoint_eligible(row, protocol) for row in records])
    stored = history.checkpoint_eligible.to_numpy()
    if not all(isinstance(value, (bool, np.bool_)) for value in stored):
        raise ValueError("checkpoint eligibility must be stored as booleans")
    if (not np.array_equal(stored, eligible)
            or not np.allclose(history.balanced_score, expected, rtol=1e-12, atol=1e-12)
            or not np.allclose(history[ENERGY_METRIC], expected, rtol=1e-12, atol=1e-12)):
        raise ValueError("recorded joint energy score or eligibility differs from the protocol")
    feasible = history.loc[eligible].sort_values(["balanced_score", "epoch"], kind="stable")
    return None if feasible.empty else feasible.iloc[0]


# Reference controls select by their own blind objective. The frozen task
# guards are recorded for every epoch but do not filter these selections.
REFERENCE_ENERGY_METRICS = {
    "unguarded_blind_energy": {"edos": .5, "phdos": .5},
    "edos_blind_energy": {"edos": 1., "phdos": 0.},
    "phdos_blind_energy": {"edos": 0., "phdos": 1.},
}


def reference_energy_score(metrics, metric):
    weights = REFERENCE_ENERGY_METRICS[metric]
    values = {task: float(metrics[f"blind_energy_{task}_mean"]) for task in TASKS if weights[task]}
    if not all(math.isfinite(value) and value >= 0 for value in values.values()):
        raise ValueError("finite nonnegative mean blind errors required for the selected tasks")
    return sum(weights[task] * value for task, value in values.items())


def reference_selection_contract(protocol, metric):
    validate_balance_protocol(protocol)
    weights = REFERENCE_ENERGY_METRICS[metric]
    terms = [f"{weight:g}*mean(blind_energy_{task})" for task, weight in weights.items() if weight]
    return {"checkpoint_selection": {
        "schema": "reference-blind-energy-selection-v1", "metric": metric,
        "mode": "min", "split": "valid", "task_weights": dict(weights),
        "formula": " + ".join(terms), "tie_break": "earliest_epoch",
        "reference": deepcopy(protocol["reference"]),
        "task_guards": "recorded per epoch as checkpoint_eligible; not applied to selection",
        "diagnostics_only": ["oracle_mae", "same_composition_spectral_difference"],
    }}


def selected_reference_history_row(history, protocol, metric):
    """Re-derive the unguarded reference score; guards must still match the record."""
    required = {"epoch", "balanced_score", metric, ENERGY_METRIC, "selection_samples",
                "checkpoint_eligible"} | {f"{prefix}_{task}{suffix}" for task in TASKS
        for prefix, suffix in (("blind_energy", "_mean"), ("blind_r2", "_median"),
                               ("blind_fail_rate", ""))}
    if history.empty or not required.issubset(history.columns) or history.epoch.duplicated().any():
        raise ValueError("incomplete or duplicated reference blind energy history")
    epochs = history.epoch.to_numpy(float)
    if not np.isfinite(epochs).all() or sorted(epochs.tolist()) != list(range(1, int(epochs.max()) + 1)):
        raise ValueError("reference selection requires an unbroken history from epoch one")
    records = history.to_dict("records")
    expected = np.asarray([reference_energy_score(row, metric) for row in records])
    joint = np.asarray([balanced_energy_score(row) for row in records])
    eligible = np.asarray([balanced_checkpoint_eligible(row, protocol) for row in records])
    stored = history.checkpoint_eligible.to_numpy()
    if not all(isinstance(value, (bool, np.bool_)) for value in stored):
        raise ValueError("checkpoint eligibility must be stored as booleans")
    if (not np.array_equal(stored, eligible)
            or not np.allclose(history.balanced_score, expected, rtol=1e-12, atol=1e-12)
            or not np.allclose(history[metric], expected, rtol=1e-12, atol=1e-12)
            or not np.allclose(history[ENERGY_METRIC], joint, rtol=1e-12, atol=1e-12)):
        raise ValueError("recorded reference score, joint energy or guard flag differs from the protocol")
    return history.sort_values(["balanced_score", "epoch"], kind="stable").iloc[0]


def load_balance_protocol(path, data_dir=None):
    protocol = validate_balance_protocol(json.loads(input_path(path).read_text()))
    if data_dir is not None:
        root = input_path(data_dir)
        view = label_contract(json.loads((root / "manifest.json").read_text()))
        if view != label_contract(protocol):
            raise ValueError(f"label contract mismatch: {label_contract(protocol)} protocol on a {view} view")
        if protocol["schema"] == BALANCE_SCHEMA_V2:
            return _check_protocol_v2_data(protocol, root)
        if file_sha256(root / "manifest.json") != protocol["data_manifest_sha256"]:
            raise ValueError("spectral protocol and data manifest differ")
        for name, digest in protocol["train_sha256"].items():
            if file_sha256(root / name) != digest:
                raise ValueError(f"train-fitted normalizer source changed: {name}")
        if file_sha256(root.parent / 'reference/grids.json') != protocol['grids_sha256']:
            raise ValueError('spectral protocol grid definitions changed')
        manifest = json.loads((root / 'manifest.json').read_text())
        ids = np.load(root / 'train/train_index.npy', allow_pickle=False)
        keep = ~np.isin(ids, manifest.get('training_exclusions', []))
        atoms = np.load(root / 'train/n_atoms_train.npy', allow_pickle=False)[keep]
        if int(keep.sum()) != protocol['fit_samples']:
            raise ValueError('train calibration cohort changed')
        for task in TASKS:
            target = np.load(root / 'train' / f'{task}_tgtdos_train.npy', allow_pickle=False)[keep]
            recomputed = fit_task_normalizer(target, atoms, protocol['tasks'][task]['bin_width'])
            if any(not math.isclose(float(value), float(protocol['tasks'][task][key]), rel_tol=1e-12, abs_tol=0)
                   for key, value in recomputed.items()):
                raise ValueError('stored constants are not the frozen train-only normalizers')
    return protocol


def _verified_array(root, manifest, relative):
    path = root / relative
    digest = file_sha256(path)
    if digest != manifest["files"][relative]["sha256"]:
        raise ValueError(f"view payload differs from its manifest: {relative}")
    return np.load(path, allow_pickle=False), digest


def _fit_v2(root, manifest):
    """C_t on effective train rows that carry a task-t label (W1: missing labels are not zeros)."""
    if label_contract(manifest) != "v2" or manifest.get("schema") != "crystal-q1-v1":
        raise ValueError("a v2 crystal view is required")
    partial = bool(manifest.get("partial_labels", False))
    hashes = {}
    ids, hashes["train/train_index.npy"] = _verified_array(root, manifest, "train/train_index.npy")
    atoms, hashes["train/n_atoms_train.npy"] = _verified_array(root, manifest, "train/n_atoms_train.npy")
    keep = ~np.isin(ids, manifest.get("training_exclusions", []))
    tasks, counts = {}, {}
    for task in TASKS:
        grid = manifest["grid"][task]
        lo, hi = (float(v) for v in grid["range"])
        target, hashes[f"train/{task}_tgtdos_train.npy"] = _verified_array(
            root, manifest, f"train/{task}_tgtdos_train.npy")
        rows = keep.copy()
        if partial:
            has, hashes[f"train/has_{task}_train.npy"] = _verified_array(root, manifest, f"train/has_{task}_train.npy")
            rows &= has.astype(bool)
        if target.shape[1] != int(grid["bins"]):
            raise ValueError("view targets differ from the manifest grid")
        tasks[task] = fit_task_normalizer(target[rows], atoms[rows], (hi - lo) / int(grid["bins"]))
        counts[task] = int(rows.sum())
    return tasks, counts, hashes, partial


def _valid_cohort(root, manifest, valid_split, partial):
    """Selection cohort; with partial labels each task is scored on its own labelled rows (W1)."""
    if valid_split not in manifest["splits"] or valid_split == "train":
        raise ValueError(f"unknown public valid split: {valid_split}")
    ids, digest = _verified_array(root, manifest, f"{valid_split}/{valid_split}_index.npy")
    if partial:
        for task in TASKS:
            has, _ = _verified_array(root, manifest, f"{valid_split}/has_{task}_{valid_split}.npy")
            if not has.any():
                raise ValueError(f"the selection valid split has no {task} labels")
    return len(ids), digest


def _check_protocol_v2_data(protocol, root):
    manifest_path = root / "manifest.json"
    if file_sha256(manifest_path) != protocol["data_manifest_sha256"]:
        raise ValueError("spectral protocol and data manifest differ")
    manifest = json.loads(manifest_path.read_text())
    tasks, counts, hashes, partial = _fit_v2(root, manifest)
    if hashes != protocol["train_sha256"] or counts != protocol["fit_samples"]:
        raise ValueError("train calibration cohort changed")
    if partial != protocol["view"]["partial_labels"]:
        raise ValueError("partial-label mode differs from the view")
    for task in TASKS:
        if any(not math.isclose(float(value), float(protocol["tasks"][task][key]), rel_tol=1e-12, abs_tol=0)
               for key, value in tasks[task].items()):
            raise ValueError("stored constants are not the frozen train-only normalizers")
    samples, digest = _valid_cohort(root, manifest, protocol["view"]["valid_split"], partial)
    if samples != protocol["reference"]["samples"] or digest != protocol["view"]["valid_index_sha256"]:
        raise ValueError("valid selection cohort changed")
    return protocol


def prepare_balance_protocol_v2(data_dir, valid_split="valid"):
    """v2 protocol: C fitted on labelled effective-train rows, grid from the view, no guard reference yet."""
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    tasks, counts, hashes, partial = _fit_v2(root, manifest)
    samples, digest = _valid_cohort(root, manifest, valid_split, partial)
    protocol = {"schema": BALANCE_SCHEMA_V2, "label_contract": "v2", "fit_split": "train",
                "fit_samples": counts, "tasks": tasks, "task_weights": {"edos": .5, "phdos": .5},
                "data_manifest_sha256": file_sha256(root / "manifest.json"), "train_sha256": hashes,
                "view": {"valid_split": valid_split, "valid_index_sha256": digest, "partial_labels": partial,
                         "grid": deepcopy(manifest["grid"])},
                "reference": {"mode": "none", "split": "valid", "samples": samples},
                "failure_increase_limit_pp": 0., "r2_drop_limit": .02,
                "task_weighting": "W1: each task averaged over its labelled rows, then 0.5/0.5",
                "scientific_scope": "joint blind full-window per-atom squared error on v2 labels; "
                                    "C fitted on labelled effective-train rows; no guard reference "
                                    "(every epoch eligible) until a v2 working-point export is bound"}
    return validate_balance_protocol(protocol)


def prepare_balance_protocol(data_dir, analysis_dir, reference="A64"):
    """Fit on effective train only; bind independent fixed valid failure guards.

    Existing full-valid prediction exports are checked against the frozen labels
    and their explicit sample scores. No checkpoint is loaded or reselected.
    """
    import pandas as pd
    import torch

    root, analysis_dir = Path(data_dir), Path(analysis_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "crystal-q1-v1":
        raise ValueError("complete crystal-q1-v1 data required")
    fingerprints = {}

    def array(split, name):
        path = root / split / name
        relative = path.relative_to(root).as_posix()
        digest = file_sha256(path)
        if digest != manifest["files"][relative]["sha256"]:
            raise ValueError(f"frozen data payload changed: {relative}")
        fingerprints[relative] = digest
        return np.load(path, allow_pickle=False)

    ids = array("train", "train_index.npy")
    keep = ~np.isin(ids, manifest.get("training_exclusions", []))
    atoms = array("train", "n_atoms_train.npy")
    offsets = array("train", "atom_offsets_train.npy")
    if (not np.array_equal(np.diff(offsets), atoms) or len(set(ids)) != len(ids)
            or int(keep.sum()) != manifest["splits"]["train"]["training_n"]):
        raise ValueError("effective train IDs/complete atom counts differ from manifest")
    grids_path = root.parent / "reference/grids.json"
    grids = json.loads(grids_path.read_text())
    tasks = {}
    for task, grid in zip(TASKS, ("E0", "P0")):
        target = array("train", f"{task}_tgtdos_train.npy")[keep]
        widths = np.diff(np.asarray(grids[grid], float))
        if len(widths) != target.shape[1] or not np.allclose(widths, widths[0], rtol=0, atol=1e-12):
            raise ValueError("fixed uniform E0/P0 grids required")
        lo, hi = (-6., 6.) if task == 'edos' else (-280., 980.)
        if not np.allclose(grids[grid], np.linspace(lo, hi, target.shape[1] + 1), rtol=0, atol=1e-12):
            raise ValueError('production E0/P0 boundaries changed')
        tasks[task] = fit_task_normalizer(target, atoms[keep], float(widths[0]))
    train_hashes = deepcopy(fingerprints)
    valid_ids = array("valid", "valid_index.npy")
    if len(set(valid_ids)) != len(valid_ids):
        raise ValueError("unique complete valid IDs required")
    samples_path = analysis_dir / f"{reference}_valid_samples.csv"
    spectra_path = analysis_dir / f"{reference}_valid_spectra.npz"
    source_path = analysis_dir / "report.json"
    source = json.loads(source_path.read_text())
    if source["data_contract"]["manifest_sha256"] != file_sha256(root / "manifest.json"):
        raise ValueError("reference experiment and current data manifest differ")
    bound_exports = source["run_artifacts"][reference].get("exports", {})
    for path in (samples_path, spectra_path):
        if path.name in bound_exports and file_sha256(path) != bound_exports[path.name]["sha256"]:
            raise ValueError("reference export differs from the bound experiment")
    frame = pd.read_csv(samples_path)
    with np.load(spectra_path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    if (frame.material_id.duplicated().any() or set(frame.material_id) != set(valid_ids)
            or frame.material_id.tolist() != values["material_ids"].tolist()):
        raise ValueError("reference must cover every valid material exactly once")
    order = pd.Index(valid_ids).get_indexer(frame.material_id)
    ref = {"label": reference, "split": "valid", "samples": len(frame),
           "samples_sha256": file_sha256(samples_path), "spectra_sha256": file_sha256(spectra_path),
           "experiment_report_sha256": file_sha256(source_path),
           "selected_epoch": int(source["runs"][reference]["best_epoch"]),
           "checkpoint_sha256": source["run_artifacts"][reference]["files"]["checkpoint_best.pth"]}
    # Reuse the current project's explicit epsilon-R2 implementation in FP64.
    from utils.metrics import per_sample_spectral_metrics

    for task in TASKS:
        target = array("valid", f"{task}_tgtdos_valid.npy")[order].astype(float)
        if not np.allclose(target, values[f"tgt_{task}"], rtol=3e-6, atol=2e-6):
            raise ValueError("reference target spectra differ from the frozen valid labels")
        metrics = per_sample_spectral_metrics(torch.from_numpy(values[f"pred_blind_{task}"]).double(),
                                             torch.from_numpy(target))
        r2 = metrics["r2"].numpy()
        recorded = frame[f"blind_r2_{task}"].to_numpy()
        if (not np.array_equal(r2 < 0, recorded < 0)
                or abs(np.median(r2) - np.median(recorded)) > 3e-6):
            raise ValueError("reference blind R2/failures differ from exported sample scores")
        ref[f"failures_{task}"] = int((recorded < 0).sum())
        ref[f"r2_{task}_median"] = float(np.median(recorded))
    protocol = {"schema": BALANCE_SCHEMA, "fit_split": "train", "fit_samples": int(keep.sum()),
                "tasks": tasks, "task_weights": {"edos": .5, "phdos": .5},
                "data_manifest_sha256": file_sha256(root / "manifest.json"),
                "train_sha256": train_hashes, "valid_guard_sha256": {
                    name: digest for name, digest in fingerprints.items() if name.startswith("valid/")},
                "grids_sha256": file_sha256(grids_path), "reference": ref,
                "failure_increase_limit_pp": 0., "r2_drop_limit": .02,
                "scientific_scope": "joint blind full-window per-atom squared error; no coverage mask; "
                                    "train task-energy normalization; same-score loss and checkpoint selection; "
                                    "no claim of equal task difficulty or equal parameter gradients"}
    return validate_balance_protocol(protocol)
