"""Complete-cache and full-size ZP forward/backward checks, without updates.

Reads structural train inputs only for GPU profiling; never loads a checkpoint,
trains an epoch, updates parameters, or evaluates test predictions.
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
import torch

from model.transformer import Transformer
from utils.b7_cif_inference import B7_TRANSFORMER_PARAMS
from utils.bucket_batch import crystal_collate, split_crystal_batch, automatic_pair_budget
from utils.crystal import CrystalGeometry
from utils.crystal_cache import PackedCrystalStore, crystal_data_contract
from utils.paths import CRYSTAL_Q1_DATA_ROOT, Q1_DATA_ROOT, file_sha256


def profile(model, chunks, device, repeats):
    def run():
        model.zero_grad(set_to_none=True)
        total = sum(len(chunk[0]) for chunk in chunks)
        for atoms, geometry in chunks:
            atoms, geometry = atoms.to(device), geometry.to(device)
            outputs = model(atoms, atoms.eq(0), geometry)
            if not all(torch.isfinite(outputs[name]).all() for name in ("edos", "phdos", "eta")):
                raise AssertionError("non-finite full-model output")
            loss = sum(outputs[name].square().mean() for name in ("edos", "phdos", "eta"))
            (loss * len(atoms) / total).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise AssertionError("non-finite or absent full-model gradient")
    for _ in range(2):
        run()
    torch.cuda.synchronize(device)
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        run()
        torch.cuda.synchronize(device)
        timings.append(1000 * (time.perf_counter() - start))
    return {"forward_backward_finite": True, "updates": 0,
            "computation_batch_sizes": [len(c[0]) for c in chunks],
            "atom_widths": [c[0].shape[1] - (2 if torch.is_tensor(c[1]) else 0) for c in chunks],
            "median_ms": float(np.median(timings)), "timings_ms": timings,
            "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 1024 ** 2}


def check(out, repeats=3):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    contract = crystal_data_contract(CRYSTAL_Q1_DATA_ROOT, normalization="sumnorm")
    manifest = json.loads((CRYSTAL_Q1_DATA_ROOT / "manifest.json").read_text())
    store = PackedCrystalStore(CRYSTAL_Q1_DATA_ROOT, "train")
    old_elements = np.load(Q1_DATA_ROOT / "train/elements_train.npy", allow_pickle=False)
    old_positions = np.load(Q1_DATA_ROOT / "train/positions_train.npy", allow_pickle=False)
    largest = int(store.n_atoms.argmax())
    # The largest is quality-isolated; here only its structural fields are read.
    small = [i for i, n in enumerate(store.n_atoms.tolist()) if n <= 16][:31]
    chosen = small + [largest]
    mixed = crystal_collate([list(store[i]) for i in chosen])
    mixed_chunks = split_crystal_batch(mixed, automatic_pair_budget(mixed))
    worst = crystal_collate([list(store[largest]) for _ in range(32)])
    worst_chunks = split_crystal_batch(worst, automatic_pair_budget(worst))
    report = {"data_contract": contract, "splits": manifest["splits"],
              "training_exclusions": manifest["training_exclusions"],
              "structure_repairs": len(manifest["repairs"]),
              "valence_repairs": sum("valence" in r for r in manifest["repairs"]),
              "test_use": "cache identity/integrity only; no predictions or selection",
              "largest_structural_case": {"id": str(store.material_ids[largest]),
                                          "n_atoms": int(store.n_atoms[largest])},
              "source_sha256": {"checker": file_sha256(__file__),
                                "transformer": file_sha256(ROOT / "model/transformer.py")}}
    if not torch.cuda.is_available():
        raise RuntimeError("GPU resource acceptance requires CUDA; no CPU timing substituted")
    device = torch.device("cuda")
    params = {**B7_TRANSFORMER_PARAMS, "atom_feat_mode": "z_only_proj"}
    model = Transformer(**params).to(device).train()
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    report["hardware"] = torch.cuda.get_device_name(device)
    report["precision"] = "FP32; no AMP"
    report["model_parameters"] = sum(p.numel() for p in model.parameters())
    baseline = [torch.from_numpy(old_elements[chosen]).long(),
                torch.from_numpy(old_positions[chosen].reshape(32, 82, 3)).float()]
    report["fixed80_historical"] = profile(model, [baseline], device, repeats)
    model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    report["complete_mixed"] = profile(model, mixed_chunks, device, repeats)
    model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    report["complete_worst32x192"] = profile(model, worst_chunks, device, repeats)
    if any(not torch.equal(value.cpu(), initial[name]) for name, value in model.state_dict().items()):
        raise AssertionError("acceptance check changed model weights or buffers")
    report["weights_unchanged"] = True
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"refusing existing acceptance report: {out}")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("positive repeat count required")
    check(args.out, args.repeats)


if __name__ == "__main__":
    main()
