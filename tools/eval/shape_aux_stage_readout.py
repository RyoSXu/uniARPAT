"""Frozen stage readout for the spectral-shape-auxiliary checkpoint.

Read-only. The epoch-25 complete-A64 checkpoint stays in eval mode and FP32.
Train absolute-composition pairs fit a no-intercept linear readout at three
stages; valid pairs only score the locked readout. eDOS and phDOS are separate.
Test arrays are never opened.
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

from tools.eval.lg_structure_diagnosis import dump, loader
from tools.eval.lg_valid_export import load_run, select_best
from tools.eval.stage_readout_diagnosis import (
    ATOM_STAGES,
    NEAR_ZERO_TV,
    StageFit,
    build_frames,
    compare_frames,
    identity_checks,
    method_metrics,
    pair_index_table,
    pooled_mean_rms,
    predict_materials,
)
from tools.train import evaluate_split
from utils.builder import ConfigBuilder
from utils.crystal import atom_padding_mask
from utils.paths import file_sha256, repository_path
from utils.zproj_alignment import state_hash
from tools.eval.lg_train_subset_diagnosis import composition_keys_by_position, same_composition_pairs

DEFAULT_RUN = "results/runs/ablation_m1_sbal_shape20261025_shape_a64_s42_e35"
PRIOR_SPECTRA = "results/runs/spectral_shape_aux_20261025/analysis/candidate_valid_spectra.npz"
EXPECTED_EPOCH = 25
EXPECTED_SCHEMA = "joint-spectral-balance-shape-model-v1"
TASKS = ("edos", "phdos")
DECODER_STAGE = {"edos": "decoder_edos", "phdos": "decoder_phdos"}
STAGE_DESCRIPTIONS = {
    "local_last": "pooled mean and RMS of the atom state after the final local update",
    "global_last": "pooled mean and RMS of the atom state after the final global update",
    "decoder_edos": "eDOS decoder token at every energy grid position",
    "decoder_phdos": "phDOS decoder token at every frequency grid position",
}
SEPARATION_FLOOR = 1e-3
GROUPING = "absolute"
SEED = 42
BOOTSTRAP_DRAWS = 2000
ACTIONS = {
    "premise_absent": "模型输出已优于零响应；本诊断不建议改结构",
    "local_message": "下一轮看局部消息",
    "global_mixing": "下一轮看全局混合",
    "readout_head": "下一轮看最终读出",
    "decoder_or_readout": "下一轮看 decoder 或读出",
    "stop_no_bottleneck": "三层都分开但线性探针都读不出；不指定某一层为瓶颈",
    "unresolved": "事先规则没有覆盖这种组合；只保留数字",
}


def task_stages(task):
    if task not in DECODER_STAGE:
        raise ValueError(f"unknown task: {task}")
    return (*ATOM_STAGES, DECODER_STAGE[task])


def relative_pair_l2(features, left, right):
    """Scale-free pair distance of the vector actually shown to the readout."""
    flat = np.asarray(features, dtype=np.float64).reshape(len(features), -1)
    left = np.asarray(left, int)
    right = np.asarray(right, int)
    gap = flat[left] - flat[right]
    denom = 0.5 * (np.linalg.norm(flat[left], axis=1) + np.linalg.norm(flat[right], axis=1))
    return np.linalg.norm(gap, axis=1) / np.maximum(denom, 1e-12)


def group_mean_interval(values, groups, seed=SEED, draws=BOOTSTRAP_DRAWS):
    frame = pd.DataFrame({"composition": np.asarray(groups), "value": np.asarray(values, dtype=np.float64)})
    grouped = frame.groupby("composition").value.mean().to_numpy()
    if not len(grouped):
        return {"groups": 0, "mean_group": None, "ci95": None}
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(grouped, len(grouped), replace=True).mean() for _ in range(draws)])
    return {"groups": int(len(grouped)), "mean_group": float(grouped.mean()),
            "ci95": np.percentile(boot, [2.5, 97.5]).tolist()}


def separation_class(interval, floor=SEPARATION_FLOOR):
    """Lock the identical-versus-separated cut before looking at valid scores."""
    if interval.get("ci95") is None:
        return "uncertain"
    low, high = interval["ci95"]
    if high < floor:
        return "indistinguishable"
    if low > floor:
        return "separated"
    return "uncertain"


def beats_zero(comparison):
    return bool(comparison.get("right_beats_left", False))


def decide(local_distance, global_distance, decoder_distance, local_beats, global_beats,
           decoder_beats, model_beats):
    """First matching rule wins. Valid absolute-composition inputs only."""
    distances = (local_distance, global_distance, decoder_distance)
    stage_beats = (local_beats, global_beats, decoder_beats)
    if model_beats:
        action = "premise_absent"
    elif local_distance == "indistinguishable":
        action = "local_message"
    elif local_distance == "separated" and global_distance == "indistinguishable":
        action = "global_mixing"
    elif local_beats and global_beats and not model_beats:
        action = "readout_head" if decoder_beats else "decoder_or_readout"
    elif all(value == "separated" for value in distances) and not any(stage_beats):
        action = "stop_no_bottleneck"
    else:
        action = "unresolved"
    if action not in ACTIONS:
        raise ValueError(f"unknown action: {action}")
    return {"action": action, "meaning": ACTIONS[action]}


class ShapeAuxStageExtractor:
    """Removable hooks for the last local layer, last global layer, and both decoder calls."""

    def __init__(self, transformer):
        encoder = transformer.encoder
        if getattr(encoder, "local_layers", None) is None:
            raise ValueError("stage diagnosis requires the local/global encoder")
        if getattr(transformer, "decoupled_decoder", False) or hasattr(transformer, "atom_phdos_head"):
            raise ValueError("this diagnosis requires the shared eDOS-then-phDOS decoder")
        picked = {}
        for kind, index in encoder.execution_order:
            picked[kind] = index
        self.stage_modules = {
            "local_last": encoder.local_layers[picked["local"]],
            "global_last": encoder.global_layers[picked["global"]],
        }
        self.features = {name: [] for name in STAGE_DESCRIPTIONS}
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
            if output.dim() != 3:
                raise ValueError(f"{name} must be an atom-state tensor")
            self.features[name].append(pooled_mean_rms(output, self.mask).detach().float().cpu().numpy())
        return hook

    def _decoder(self, module, args, output):
        if self.decoder_call >= 2:
            raise ValueError("shared decoder ran more than twice in one forward")
        name = ("decoder_edos", "decoder_phdos")[self.decoder_call]
        value = output[0]
        if value.dim() != 3:
            raise ValueError(f"{name} must keep its spectral positions")
        self.features[name].append(value.detach().float().cpu().numpy())
        self.decoder_call += 1

    def stacked(self):
        out = {}
        for name in STAGE_DESCRIPTIONS:
            chunks = self.features[name]
            if not chunks:
                raise ValueError(f"missing stage features: {name}")
            out[name] = np.concatenate(chunks, axis=0)
        return out

    def close(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


def _participant_rows(frame):
    return sorted(set(frame.i.to_numpy(int).tolist()).union(frame.j.to_numpy(int).tolist()))


def smoke_subset(keys, rows, ids, limit_groups=8):
    """Materials from the first multi-member absolute groups, in index order."""
    _, groups = same_composition_pairs(keys)
    multi = sorted((members for members in groups.values() if len(members) > 1),
                   key=lambda members: members[0])
    selected = set()
    for members in multi[:limit_groups]:
        selected.update(members)
    if len(selected) < 4:
        raise ValueError("smoke subset needs at least four paired materials")
    picked = sorted(selected)
    return ([rows[i] for i in picked], [ids[i] for i in picked], [keys[i] for i in picked])


def _load_shape_aux(run_dir):
    saved, _params, contract = load_run(run_dir)
    if contract.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"diagnosis is locked to {EXPECTED_SCHEMA}")
    picked = select_best(run_dir, contract["selection_metric"], saved)
    epoch = int(picked["row"].epoch)
    if epoch != EXPECTED_EPOCH:
        raise ValueError(f"best epoch {epoch} != locked epoch {EXPECTED_EPOCH}")
    checkpoint = picked["checkpoint"]
    if int(checkpoint["epoch"]) != EXPECTED_EPOCH:
        raise ValueError("checkpoint epoch differs from the locked epoch")
    return saved, contract, checkpoint


def _verify_train_valid(saved, checkpoint):
    """Hash the manifest and train/valid payloads. Do not open test arrays."""
    data_dir = repository_path(saved["config"]["dataset"]["valid"]["data_dir"])
    manifest_path = data_dir / "manifest.json"
    recorded = checkpoint.get("data_contract") or saved["cli"].get("data_contract")
    manifest_sha = file_sha256(manifest_path)
    if recorded is None or recorded.get("manifest_sha256") != manifest_sha:
        raise ValueError("checkpoint manifest hash differs from the current manifest")
    manifest = json.loads(manifest_path.read_text())
    verified = {}
    for name, entry in manifest["files"].items():
        if not name.startswith(("train/", "valid/")):
            continue
        digest = file_sha256(manifest_path.parent / name)
        if digest != entry["sha256"]:
            raise ValueError("dataset digest mismatch: " + name)
        verified[name] = digest
    if not any(name.startswith("train/") for name in verified) or not any(
            name.startswith("valid/") for name in verified):
        raise ValueError("train/valid payload verification is incomplete")
    return data_dir, manifest_sha, verified


def _prior_agreement(ids, pred_shape, tgt_shape, task):
    path = repository_path(PRIOR_SPECTRA)
    if not path.exists():
        return {"present": False}
    pred_key = f"pred_shape_{task}"
    tgt_key = f"tgt_shape_{task}"
    with np.load(path) as prior:
        if pred_key not in prior.files or tgt_key not in prior.files:
            return {"present": True, "compared": False, "files": list(prior.files)}
        lookup = {str(value): index for index, value in enumerate(prior["material_ids"])}
        positions = [lookup.get(mid) for mid in ids]
        if any(value is None for value in positions):
            raise ValueError("extracted valid materials are missing from the prior export")
        index = np.asarray(positions, int)
        pred_gap = np.max(np.abs(prior[pred_key][index] - pred_shape))
        tgt_gap = np.max(np.abs(prior[tgt_key][index] - tgt_shape))
    return {"present": True, "compared": True, "source": str(path.relative_to(ROOT)),
            "materials": len(ids), "max_abs_pred_shape_difference": float(pred_gap),
            "max_abs_target_shape_difference": float(tgt_gap)}


def _separation_report(features, frame):
    values = relative_pair_l2(features, frame.i.to_numpy(int), frame.j.to_numpy(int))
    interval = group_mean_interval(values, frame.composition.to_numpy())
    interval["pair_median"] = float(np.median(values))
    interval["floor"] = SEPARATION_FLOOR
    interval["class"] = separation_class(interval)
    return interval


def diagnose(out_dir, smoke=False, run_dir=None):
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
        print(json.dumps({"stage": stage, "seconds": round(time.monotonic() - start, 3)}), flush=True)

    run = repository_path(run_dir or DEFAULT_RUN)
    saved, contract, checkpoint = _load_shape_aux(run)
    data_dir, manifest_sha, verified_files = _verify_train_valid(saved, checkpoint)
    checkpoint_sha = file_sha256(run / "checkpoint_best.pth")
    model_state = checkpoint["model"]
    del checkpoint

    builder = ConfigBuilder(**saved["config"])
    datasets = {split: builder.get_dataset(split=split, dos_minmax=True, dos_sumnorm=True)
                for split in ("train", "valid")}
    if any(ds.augment for ds in datasets.values()):
        raise ValueError("diagnosis requires unaugmented data")

    keys_by_split, ids_by_split, frames_by_split, scope_rows = {}, {}, {}, {}
    for split in ("train", "valid"):
        rows = list(range(len(datasets[split])))
        ids = [str(datasets[split].material_ids[row]) for row in rows]
        keys = composition_keys_by_position(datasets[split], rows)[GROUPING]
        if smoke:
            rows, ids, keys = smoke_subset(keys, rows, ids)
        frame, sizes = pair_index_table(keys, ids)
        frame.to_csv(out_dir / f"pairs_{split}_{GROUPING}.csv", index=False)
        keys_by_split[split], ids_by_split[split] = keys, ids
        frames_by_split[split], scope_rows[split] = frame, rows
        if frame.empty:
            raise ValueError(f"{split} has no absolute-composition pairs")
        del sizes
    extraction_rows = {split: _participant_rows(frames_by_split[split]) for split in ("train", "valid")}
    progress("identities")

    build = ConfigBuilder(**saved["config"])
    model = build.get_model()
    transformer = model.model["transformer"]
    transformer.load_state_dict(model_state, strict=True)
    del model_state
    model.to(device)
    transformer.eval()
    before = state_hash(transformer.state_dict())
    store = {}
    extraction = {}
    for split in ("train", "valid"):
        rows = extraction_rows[split]
        ids = [ids_by_split[split][row] for row in rows]
        dataset_rows = [scope_rows[split][row] for row in rows]
        extractor = ShapeAuxStageExtractor(transformer)
        tick = time.monotonic()
        try:
            measured = evaluate_split(model, loader(datasets[split], dataset_rows), sumnorm=True,
                                      return_sample_level=True, return_spectra=True, sample_ids=ids)
        finally:
            extractor.close()
        if extractor.hooks:
            raise ValueError("stage hooks were not removed")
        features = extractor.stacked()
        spectra = measured["sample_spectra"]
        for name, value in features.items():
            if value.shape[0] != len(ids):
                raise ValueError(f"{name} rows differ from evaluated materials")
        if spectra["pred_shape_edos"].shape[0] != len(ids):
            raise ValueError("spectra rows differ from evaluated materials")
        subset_keys = [keys_by_split[split][row] for row in rows]
        subset_frame, _ = pair_index_table(subset_keys, ids)
        full = frames_by_split[split]
        expected = {(left, right) for left, right in zip(full.left_id, full.right_id)
                    if left in set(ids) and right in set(ids)}
        if set(zip(subset_frame.left_id, subset_frame.right_id)) != expected:
            raise ValueError("subset pairing differs from the complete-split pairing")
        np.savez(out_dir / f"features_{split}.npz", material_ids=np.asarray(ids), **features)
        np.savez(out_dir / f"spectra_{split}.npz", material_ids=np.asarray(ids),
                 **{key: spectra[key] for key in spectra if key.startswith(("pred_shape_", "tgt_shape_"))})
        entry = {"materials": len(ids), "seconds": time.monotonic() - tick,
                 "pairs": int(len(subset_frame)),
                 "composition_groups": int(subset_frame.composition.nunique())}
        if split == "valid":
            for task in TASKS:
                entry[f"prior_{task}"] = _prior_agreement(
                    ids, spectra[f"pred_shape_{task}"], spectra[f"tgt_shape_{task}"], task)
                gap = entry[f"prior_{task}"].get("max_abs_pred_shape_difference")
                tgt_gap = entry[f"prior_{task}"].get("max_abs_target_shape_difference")
                if gap is not None and (gap > 2e-4 or tgt_gap > 1e-5):
                    raise ValueError(f"recomputed {task} shapes differ from the frozen export")
        store[split] = {"features": features, "spectra": spectra, "frame": subset_frame,
                        "ids": ids, "keys": subset_keys}
        extraction[split] = entry
        print(json.dumps({"extracted": split, **{k: entry[k] for k in ("materials", "pairs", "seconds")}}),
              flush=True)
    after = state_hash(transformer.state_dict())
    if before != after or file_sha256(run / "checkpoint_best.pth") != checkpoint_sha:
        raise ValueError("model or checkpoint mutated during extraction")
    del model, transformer, build
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    progress("extract")

    report = {
        "scope": "frozen shape-aux epoch-25 stage readout; absolute composition; "
                 "train fit, valid decision; no parameter update; test arrays not opened",
        "smoke": smoke,
        "seed": SEED,
        "device": str(device),
        "run": str(run.relative_to(ROOT)),
        "epoch": EXPECTED_EPOCH,
        "checkpoint_sha256": checkpoint_sha,
        "schema": EXPECTED_SCHEMA,
        "data_dir": str(data_dir),
        "data_manifest_sha256": manifest_sha,
        "verified_train_valid_files": verified_files,
        "state_hash_before": before,
        "state_unchanged": True,
        "separation_floor": SEPARATION_FLOOR,
        "near_zero_target_tv": NEAR_ZERO_TV,
        "stages": STAGE_DESCRIPTIONS,
        "primary_task": "edos",
        "extraction": extraction,
        "tasks": {},
    }
    for task in TASKS:
        stages = task_stages(task)
        fits = {}
        described = {}
        for stage in stages:
            fit = StageFit(stage, store["train"]["features"][stage], store["train"]["frame"],
                           store["train"]["spectra"][f"tgt_shape_{task}"])
            fit.select()
            fit.final_fit()
            fits[stage] = fit
            described[stage] = fit.describe()
            checks = identity_checks(stage, store["train"]["features"][stage], fit.scale, fit.weights,
                                     store["train"]["frame"])
            described[stage]["identity_checks"] = checks
            if checks["swap_antisymmetry_max_abs"] > 1e-9 or checks["identical_representation_max_abs"] > 1e-9:
                raise ValueError("readout violates swap antisymmetry or zero-difference")
        task_entry = {"stages": described, "splits": {}}
        for split in ("train", "valid"):
            features = store[split]["features"]
            spectra = store[split]["spectra"]
            frame = store[split]["frame"]
            shape = spectra[f"tgt_shape_{task}"]
            predictions = {stage: predict_materials(stage, features[stage], fits[stage].scale, fits[stage].weights)
                           for stage in stages}
            predictions["model"] = spectra[f"pred_shape_{task}"]
            predictions["zero_response"] = np.zeros_like(shape)
            frames = build_frames(shape, predictions, store[split]["keys"], store[split]["ids"])
            main = {name: value[value.target_tv >= NEAR_ZERO_TV] for name, value in frames.items()}
            methods = {}
            versus = {}
            for name, value in frames.items():
                value.to_csv(out_dir / f"pairs_{task}_{split}_{name}.csv", index=False)
                methods[name] = method_metrics(value)
            for name in (*stages, "model"):
                versus[name] = compare_frames(main["zero_response"], main[name], f"{name} minus zero_response")
            separation = {stage: _separation_report(features[stage], frame) for stage in stages}
            task_entry["splits"][split] = {
                "pairs": int(len(frames["model"])),
                "near_zero_pairs": int((frames["model"].target_tv < NEAR_ZERO_TV).sum()),
                "methods": methods,
                "versus_zero": versus,
                "separation": separation,
            }
        valid = task_entry["splits"]["valid"]
        decision = decide(
            valid["separation"]["local_last"]["class"],
            valid["separation"]["global_last"]["class"],
            valid["separation"][DECODER_STAGE[task]]["class"],
            beats_zero(valid["versus_zero"]["local_last"]),
            beats_zero(valid["versus_zero"]["global_last"]),
            beats_zero(valid["versus_zero"][DECODER_STAGE[task]]),
            beats_zero(valid["versus_zero"]["model"]),
        )
        task_entry["decision"] = decision
        report["tasks"][task] = task_entry
        for stage in stages:
            del fits[stage]
        progress(f"fit_{task}")

    report["seconds"] = time.monotonic() - start
    report["peak_vram_mb"] = (torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0)
    report["limitations"] = [
        "Single training seed and the selected epoch-25 checkpoint; group intervals do not cover training-seed variation.",
        "Atom stages use permutation-invariant mean/RMS pooling. A failed linear probe on that summary does not prove the full atom state lacks information.",
        "A smaller feature norm is not treated as collapse. Collapse means the relative-L2 interval lies below the pre-registered 1e-3 floor.",
        "Decoder readouts are per spectral position. Cross-position linear maps are not tested.",
        "The target is the unit-sum shape difference, not the physical spectrum that includes the predicted total.",
        "Near-zero target gaps are excluded from probe metrics and reported separately. Representation distance uses every absolute-composition pair.",
    ]
    dump(out_dir / "report.json", report)
    progress("report")
    print(json.dumps({"complete": str(out_dir), "seconds": report["seconds"],
                      "edos": report["tasks"]["edos"]["decision"]["action"],
                      "phdos": report["tasks"]["phdos"]["decision"]["action"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="first absolute-composition groups only; plumbing check, not a scientific result")
    args = parser.parse_args()
    diagnose(args.out_dir, smoke=args.smoke, run_dir=args.run_dir)


if __name__ == "__main__":
    main()
