"""Train one Mat2Spec model on the frozen crystal-q1-v1 train view, select on the full valid split.

One process = one arm (task x label scaling x loss). Mirrors upstream train_Mat2Spec.py:
AdamW(lr, weight_decay=1e-2), StepLR(step=ceil(n_train/batch)*(epochs/4), gamma=0.5) stepped every
iteration, seed 1, shuffle, last partial batch kept, loss = compute_loss() from the ported model.
Differences from upstream are listed in baselines/mat2spec/README.md.

Exit codes: 0 completed, 3 non-finite loss/prediction, 4 CUDA out of memory, 5 contract mismatch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.mat2spec.adapter import data as D  # noqa: E402
from baselines.mat2spec.adapter import project_metrics as PM  # noqa: E402

HERE = Path(__file__).resolve().parent
SOURCE_FILES = [HERE.parent / "port/mat2spec_model.py", HERE.parent / "port/pytorch_stats_loss.py",
                HERE / "data.py", HERE / "train.py", HERE / "project_metrics.py"]
ATOM_INIT = HERE.parent / "upstream_cgcnn/atom_init.json"
SCALING_LOSSES = {"normalized_sum": ("KL", "WD"), "standardized": ("MAE",), "normalized_max": ("MAE", "MSE")}
EXIT_NONFINITE, EXIT_OOM, EXIT_CONTRACT = 3, 4, 5


class ContractError(RuntimeError):
    pass


class NonFiniteError(RuntimeError):
    pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=("edos", "phdos"))
    p.add_argument("--label_scaling", required=True, choices=tuple(SCALING_LOSSES))
    p.add_argument("--loss", required=True, choices=("KL", "WD", "MAE", "MSE"))
    p.add_argument("--run_dir", required=True)
    p.add_argument("--epochs", type=int, default=200)          # upstream --num-epochs
    p.add_argument("--batch_size", type=int, default=256)      # upstream train default (eDOS scripts)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1)              # upstream random_num
    p.add_argument("--target_unit", choices=("per_atom", "cell"), default="per_atom")
    p.add_argument("--data_root", default=str(ROOT / "data/crystal_q1_v1"))
    p.add_argument("--protocol", default=str(ROOT / PM.DEFAULT_PROTOCOL))
    p.add_argument("--graph_cache", default=None, help="directory for graphs_{split}.npz (default: run_dir/..)")
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit_train", type=int, default=0, help="precheck only: first N train crystals")
    p.add_argument("--limit_valid", type=int, default=0, help="precheck only: first N valid crystals")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)
    if args.loss not in SCALING_LOSSES[args.label_scaling]:
        p.error(f"{args.label_scaling} supports losses {SCALING_LOSSES[args.label_scaling]}")
    if args.epochs % 4:
        p.error("epochs must be divisible by 4 (upstream halves the lr four times)")
    return args


def model_args(args):
    """Upstream argparse defaults plus the per-script overrides (see SCRIPTS/train_*.sh)."""
    return SimpleNamespace(
        num_layers=3, num_neurons=128, num_heads=4, concat_comp=False, train=True,
        label_scaling=args.label_scaling, Mat2Spec_loss_type=args.loss,
        Mat2Spec_input_dim=128, Mat2Spec_label_dim=D.TASK_BINS[args.task], Mat2Spec_latent_dim=128,
        Mat2Spec_emb_size=512, Mat2Spec_keep_prob=0.5, Mat2Spec_scale_coeff=1.0, Mat2Spec_K=10,
        ablation_LE=False, ablation_CL=False)


def fingerprint(args):
    from importlib.metadata import version

    import torch_geometric

    config = {key: getattr(args, key) for key in
              ("task", "label_scaling", "loss", "epochs", "batch_size", "lr", "seed", "target_unit",
               "limit_train", "limit_valid")}
    return {"config": config,
            "model_args": {k: v for k, v in vars(model_args(args)).items() if k != "train"},
            "data_manifest_sha256": D.file_sha256(Path(args.data_root) / "manifest.json"),
            "protocol_sha256": D.file_sha256(args.protocol),
            "atom_init_sha256": D.file_sha256(ATOM_INIT),
            "sources_sha256": {path.relative_to(ROOT).as_posix(): D.file_sha256(path) for path in SOURCE_FILES},
            "versions": {"torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
                         "pymatgen": version("pymatgen")}}


def write_json(path, payload):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


class Status:
    def __init__(self, run_dir):
        self.path = Path(run_dir) / "execution/status.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = {}

    def update(self, **fields):
        self.state.update(fields, pid=os.getpid(), heartbeat_unix=time.time(),
                          heartbeat=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json(self.path, self.state)


def log(run_dir, message):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(Path(run_dir) / "execution/train.log", "a") as handle:
        handle.write(line + "\n")


def label_units(spectra, n_atoms, target_unit):
    """The spectrum the model works in: cell DOS divided by N (default) or the stored cell DOS."""
    spectra = torch.as_tensor(spectra, dtype=torch.float32)
    if target_unit == "per_atom":
        spectra = spectra / torch.as_tensor(n_atoms, dtype=torch.float32)[:, None]
    return spectra


def fit_standardizer(train_labels):
    """Upstream DATA_normalizer (per-bin mean / unbiased std) fitted on the train view only.

    Bins whose train std is exactly zero get std=1 so that (y-mean)/std stays finite; the
    prediction for such a bin then is the (zero) train mean plus the model output.
    """
    mean = train_labels.mean(0)
    std = train_labels.std(0)
    constant = std == 0
    return mean, torch.where(constant, torch.ones_like(std), std), int(constant.sum())


def scale_labels(labels, scaling, mean=None, std=None):
    """Loss-space target. normalized_sum/max use upstream's valid-phase epsilon so all-zero spectra are finite."""
    if scaling == "standardized":
        return (labels - mean) / std
    if scaling == "normalized_max":
        return labels / (labels.max(1, keepdim=True)[0] + 1e-8)
    return labels / (labels.sum(1, keepdim=True) + 1e-8)


@torch.no_grad()
def predict(net, graphs, labels, args, margs, mean, std):
    """Deterministic feature-branch prediction (z = mu, dropout off), as upstream test_Mat2Spec.py.

    Returns the prediction in label units before any rescaling:
      normalized_sum -> softmax shape; normalized_max -> max-normalised shape;
      standardized   -> de-standardised spectrum clipped at zero (upstream clips the same way).
    The label branch still receives ``labels`` (the model's forward needs them) but the returned
    feature-branch output does not depend on them (covered by a test).
    """
    net.eval()
    margs.train = False
    chunks = []
    for start in range(0, len(graphs), args.eval_batch_size):
        rows = torch.arange(start, min(start + args.eval_batch_size, len(graphs)))
        out = net(graphs.batch(rows, labels[rows]))["feat_out"]
        if args.label_scaling == "normalized_sum":
            out = torch.softmax(out, dim=1)
        elif args.label_scaling == "standardized":
            out = (out * std + mean).clamp_min(0)
        chunks.append(out)
    prediction = torch.cat(chunks)
    if not torch.isfinite(prediction).all():
        raise NonFiniteError("non-finite valid prediction")
    return prediction


def valid_metrics(prediction, valid, args, protocol):
    """Upstream selection quantity (mae_x_ori) plus project-comparable valid scores."""
    task = args.task
    target_cell = valid[task]
    atoms = valid["n_atoms"]
    labels = label_units(target_cell, atoms, args.target_unit)
    pred = prediction.cpu()
    if args.label_scaling == "normalized_sum":
        restored = pred * labels.sum(1, keepdim=True)
    elif args.label_scaling == "normalized_max":
        restored = pred * labels.max(1, keepdim=True)[0]
    else:
        restored = pred
    metrics = {"upstream_mae_x_ori": float((restored - labels).abs().mean())}
    blind = None
    if args.label_scaling == "standardized":
        blind = pred.numpy() * (atoms[:, None] if args.target_unit == "per_atom" else 1.0)
    _, summary = PM.score_task(task, target_cell, atoms, protocol, pred_shape=pred.numpy(), pred_blind=blind)
    for key in ("oracle_energy_{t}_mean", "oracle_r2_{t}_median", "oracle_fail_rate_{t}",
                "blind_energy_{t}_mean", "blind_r2_{t}_median", "blind_fail_rate_{t}",
                "shape_kl_{t}_mean", "shape_w1_{t}_mean", "shape_tv_{t}_mean"):
        name = key.format(t=task)
        if name in summary:
            metrics[name] = summary[name]
    mode = "blind" if blind is not None else "oracle"
    metrics["project_score"] = summary[f"{mode}_energy_{task}_mean"]
    metrics["project_score_name"] = f"{mode}_energy_{task}_mean"
    return metrics


def save_checkpoint(path, payload):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(args):
    run_dir = Path(args.run_dir)
    marker = run_dir / "fingerprint.json"
    latest_path = run_dir / "latest.pth"
    print_fp = fingerprint(args)
    if marker.exists():
        if not args.resume:
            raise ContractError(f"{run_dir} already holds a run; pass --resume or choose a new --run_dir")
        if json.loads(marker.read_text()) != json.loads(json.dumps(print_fp)):
            raise ContractError("fingerprint differs from the run being resumed (config/data/source changed)")
    run_dir.mkdir(parents=True, exist_ok=True)
    status = Status(run_dir)
    if not marker.exists():
        write_json(marker, print_fp)
    log(run_dir, f"=== launch pid={os.getpid()} resume={args.resume} args={vars(args)}")
    status.update(state="preparing", epoch=0, epochs=args.epochs)

    device = torch.device(args.device)
    protocol = PM.load_protocol(args.protocol)
    cache_dir = Path(args.graph_cache) if args.graph_cache else run_dir.parent / "graphs"
    atom_init = D.load_atom_init(ATOM_INIT)
    splits, graphs, labels = {}, {}, {}
    for split, limit in (("train", args.limit_train), ("valid", args.limit_valid)):
        data = D.load_split(args.data_root, split)
        nbr_index, nbr_dist = D.build_graphs(data, cache_dir / f"graphs_{split}.npz", progress=5000)
        if limit:
            data = {**data, "material_ids": data["material_ids"][:limit], "n_atoms": data["n_atoms"][:limit],
                    "offsets": data["offsets"][:limit + 1], "edos": data["edos"][:limit],
                    "phdos": data["phdos"][:limit],
                    "atomic_numbers": data["atomic_numbers"][:data["offsets"][limit]]}
            nbr_index, nbr_dist = nbr_index[:data["offsets"][-1]], nbr_dist[:data["offsets"][-1]]
        splits[split] = data
        graphs[split] = D.PackedGraphs(data, nbr_index, nbr_dist, atom_init, device)
        labels[split] = label_units(data[args.task], data["n_atoms"], args.target_unit).to(device)
    n_train = len(graphs["train"])
    log(run_dir, f"train {n_train} crystals, valid {len(graphs['valid'])} crystals, task {args.task}")

    mean, std, constant_bins = fit_standardizer(labels["train"])
    log(run_dir, f"standardizer fitted on train: {constant_bins} constant bins (std set to 1)")

    # Seeds as upstream (random / numpy / torch all set to 1). Imported here so the port's
    # module-level device probe happens after argument checks.
    import random

    from baselines.mat2spec.port import mat2spec_model as M2S

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    margs = model_args(args)
    net = M2S.Mat2Spec(margs, None).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_epoch = math.ceil(n_train / args.batch_size)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(steps_per_epoch * (args.epochs / 4)), 0.5)
    shuffle = torch.Generator().manual_seed(args.seed)
    history, best = [], {"upstream": None, "project": None}
    start_epoch = 0
    if args.resume and latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        shuffle.set_state(state["shuffle_state"].cpu())     # map_location moved the CPU RNG states
        torch.set_rng_state(state["torch_rng"].cpu())
        if state.get("cuda_rng") is not None and device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
        history, best, start_epoch = state["history"], state["best"], state["epoch"]
        log(run_dir, f"resumed from latest.pth at epoch {start_epoch}")
    parameters = sum(p.numel() for p in net.parameters())
    log(run_dir, f"parameters {parameters}; steps/epoch {steps_per_epoch}; lr halves every {args.epochs // 4} epochs")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        started = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        net.train()
        margs.train = True
        order = torch.randperm(n_train, generator=shuffle)
        sums = np.zeros(5)
        for step in range(steps_per_epoch):
            rows = order[step * args.batch_size:(step + 1) * args.batch_size]
            if len(rows) < 2:
                raise ContractError("a batch of one crystal makes upstream's contrastive loss undefined")
            y = labels["train"][rows]
            output = net(graphs["train"].batch(rows, y))
            total, nll, nll_x, kl, contrastive, _, _ = M2S.compute_loss(
                scale_labels(y, args.label_scaling, mean, std), output, None, margs)
            if not torch.isfinite(total):
                raise NonFiniteError(f"non-finite training loss at epoch {epoch} step {step + 1}")
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            scheduler.step()
            sums += [float(v) for v in (total, nll, nll_x, kl, contrastive)]
            if step % 200 == 0:
                status.update(state="running", epoch=epoch, step=step + 1)
        train_time = time.time() - started
        prediction = predict(net, graphs["valid"], labels["valid"], args, margs, mean, std)
        metrics = valid_metrics(prediction, splits["valid"], args, protocol)
        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0],
               **dict(zip(("train_total", "train_nll_label", "train_nll_feature", "train_latent_kl",
                           "train_contrastive"), (sums / steps_per_epoch).tolist())),
               **metrics, "train_time_s": train_time, "epoch_time_s": time.time() - started,
               "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else 0.0}
        history.append(row)
        saved = []
        for name, key in (("upstream", "upstream_mae_x_ori"), ("project", "project_score")):
            if best[name] is None or row[key] < best[name]["score"]:        # ties keep the earlier epoch
                best[name] = {"epoch": epoch, "score": row[key], "criterion": key}
                save_checkpoint(run_dir / f"best_{name}.pth", {
                    "model": net.state_dict(), "epoch": epoch, "metrics": row, "mean": mean.cpu(),
                    "std": std.cpu(), "fingerprint": print_fp})
                saved.append(name)
        save_checkpoint(latest_path, {
            "model": net.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "shuffle_state": shuffle.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "history": history, "best": best, "epoch": epoch, "mean": mean.cpu(), "std": std.cpu(),
            "fingerprint": print_fp})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        log(run_dir, f"epoch {epoch}/{args.epochs} loss {row['train_total']:.5f} "
                     f"valid mae_x_ori {row['upstream_mae_x_ori']:.6g} {row['project_score_name']} "
                     f"{row['project_score']:.5f} time {row['epoch_time_s']:.1f}s "
                     f"vram {row['peak_vram_mb']:.0f}MB best saved: {saved or '-'}")
        status.update(state="running", epoch=epoch, step=steps_per_epoch, best=best)
    status.update(state="completed", epoch=args.epochs, best=best)
    log(run_dir, f"completed: best {best}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    status_path = Path(args.run_dir) / "execution/status.json"

    def fail(state, code, error):
        if status_path.parent.exists():
            previous = json.loads(status_path.read_text()) if status_path.exists() else {}
            write_json(status_path, {**previous, "state": state, "error": str(error), "pid": os.getpid(),
                                     "heartbeat_unix": time.time()})
        print(f"[{state}] {error}", file=sys.stderr, flush=True)
        return code

    try:
        return run(args)
    except NonFiniteError as error:
        return fail("failed_nonfinite", EXIT_NONFINITE, error)
    except torch.cuda.OutOfMemoryError as error:
        return fail("failed_oom", EXIT_OOM, error)
    except ContractError as error:
        # A refused launch must not overwrite the status of the run that owns this directory.
        print(f"[failed_contract] {error}", file=sys.stderr, flush=True)
        return EXIT_CONTRACT


if __name__ == "__main__":
    sys.exit(main())
