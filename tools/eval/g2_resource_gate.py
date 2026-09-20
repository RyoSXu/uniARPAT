#!/usr/bin/env python3
"""G2a V100 resource measurement tool (B7 vs G2, batch 32, same batch).

Compares single-step peak VRAM (allocated and reserved) and latency of B7 and G2a
under the exact production recipe (M1, batch 32, SumNorm KL/W1/Huber, H1 eta/gamma,
dropout 0.05, seed 42) on Tesla V100-SXM2-32GB.

It reports peak memory and full-training-step latency under the fixed G2a
configuration. Fixed batch 32 OOM blocks a pilot; reported cost ratios have no
automatic stop threshold unless the user explicitly approves one. Neither batch
size, cutoff nor top-k is altered.
"""

import argparse
import json
import os
import sys
import time
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datasets.dataset import Dos_Dataset
from model.model import basemodel
from run_ablation_experiments import MODEL_CONFIGS
from torch.utils.data import DataLoader
from utils.builder import ConfigBuilder
from utils.experiment_config import ExperimentConfig


def build_production_yaml_cfg(use_g2: bool, r_cut: float = 5.5, data_dir: str = "./data/train4ARPAT"):
    """Construct identical production YAML config matching run_ablation_experiments.py."""
    cfg = ExperimentConfig(
        model_name="M1",
        epochs=10,
        batch_size=32,
        lr=5e-5,
        seed=42,
        data_dir=data_dir,
        norm="sumnorm",
        use_mask=False,
        dropout=0.05,
        scale_mode="eta",
        scale_sup_w=1.0,
        eta_sup_w=1.0,
        delta_edos=0.09375,
        delta_phdos=19.6875,
        use_g1=False,
        use_g2=use_g2,
        g2_r_cut=r_cut,
    )

    with open("configs/default.yaml") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    yaml_cfg["model"]["params"]["sub_model"]["transformer"].update(MODEL_CONFIGS["M1"]["transformer_params"])
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["edos_num"] = cfg.edos_num
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["phdos_num"] = cfg.phdos_num
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["atom_feat_mode"] = cfg.atom_feat
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["energy_code"] = cfg.energy_code
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["use_macro_lattice"] = bool(cfg.use_macro_lattice)
    for _k, _v in (("tv_w", cfg.tv_w), ("grad_w", cfg.grad_w), ("peak_w", cfg.peak_w),
                   ("tail_w", cfg.tail_w), ("tail_start", cfg.tail_start)):
        yaml_cfg["model"]["params"][_k] = _v
    yaml_cfg["model"]["params"]["use_mask"] = bool(cfg.use_mask)
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["scale_mode"] = cfg.scale_mode
    yaml_cfg["model"]["params"]["scale_sup_w"] = float(cfg.scale_sup_w)
    yaml_cfg["model"]["params"]["eta_sup_w"] = float(cfg.eta_sup_w)
    yaml_cfg["model"]["params"]["delta_edos"] = float(cfg.delta_edos)
    yaml_cfg["model"]["params"]["delta_phdos"] = float(cfg.delta_phdos)
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["scalar_mode"] = cfg.scalar_mode
    yaml_cfg["model"]["params"]["scalar_sup_w"] = float(cfg.scalar_sup_w)
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["use_g1"] = bool(cfg.use_g1)
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["g1_r_cut"] = float(cfg.g1_r_cut)
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["g1_max_neighbors"] = int(cfg.g1_max_neighbors)
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["use_g2"] = bool(cfg.use_g2)
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["g2_r_cut"] = float(r_cut)
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["q1_coord"] = bool(cfg.q1_coord)
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["q1_hidden"] = int(cfg.q1_hidden)
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["q2_fourier"] = bool(cfg.q2_fourier)
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["c5_moe"] = bool(cfg.c5_moe)
    yaml_cfg["model"]["params"]["c5_moe_balance_w"] = float(cfg.c5_moe_balance_w)
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["r1a_point"] = bool(cfg.r1a_point or cfg.r1b_coord)
    yaml_cfg["model"]["params"]["sub_model"]["transformer"]["r1b_coord"] = bool(cfg.r1b_coord)
    if cfg.dropout is not None:
        yaml_cfg["model"]["params"]["sub_model"]["transformer"]["dropout"] = float(cfg.dropout)
    yaml_cfg["model"]["params"]["loss_form"] = "sumnorm_klw"
    yaml_cfg["model"]["params"]["dos_minmax"] = True
    yaml_cfg["model"]["params"]["save_best"] = "balanced_score"
    yaml_cfg["dataset"]["train"]["data_dir"] = cfg.data_dir
    yaml_cfg["dataset"]["valid"]["data_dir"] = cfg.data_dir
    yaml_cfg["dataset"]["test"]["data_dir"] = cfg.data_dir
    yaml_cfg["dataset"]["train"]["augment"] = bool(cfg.augment)
    yaml_cfg["dataset"]["train"]["disp_sigma"] = float(cfg.disp_sigma)
    return yaml_cfg


def measure_arm(arm_name: str, use_g2: bool, r_cut: float, data_dir: str,
                num_steady_steps: int = 5, device_str: str = "cuda:0"):
    """Measure single-step and steady-state resource metrics for a single arm."""
    device = torch.device(device_str)
    assert device.type == "cuda", "V100 resource gate must run on CUDA"

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Load dataset and prepare deterministic batch 0..num_steady_steps+1
    ds = Dos_Dataset(data_dir=data_dir, split="train", dos_minmax=True, dos_sumnorm=True)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    batches = []
    for i, b in enumerate(loader):
        batches.append(b)
        if i >= num_steady_steps + 1:
            break

    batch0 = batches[0]
    n_atom_batch0 = int((batch0[0][:, 2:82] != 0).sum().item())

    # Build model
    torch.manual_seed(42)
    yaml_cfg = build_production_yaml_cfg(use_g2=use_g2, r_cut=r_cut, data_dir=data_dir)
    builder = ConfigBuilder(**yaml_cfg)
    model = builder.get_model()
    model.to(device)
    model.model["transformer"].train()

    total_params = sum(p.numel() for p in model.model["transformer"].parameters() if p.requires_grad)

    # 1. Measure Cold Step 0 on Batch 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_ev_cold = torch.cuda.Event(enable_timing=True)
    end_ev_cold = torch.cuda.Event(enable_timing=True)
    t0_wall_cold = time.perf_counter()
    start_ev_cold.record()

    out_cold = model.train_one_step(batch0, step=0)

    end_ev_cold.record()
    torch.cuda.synchronize()
    t1_wall_cold = time.perf_counter()

    cold_time_ms = start_ev_cold.elapsed_time(end_ev_cold)
    cold_wall_ms = (t1_wall_cold - t0_wall_cold) * 1000.0
    cold_peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    cold_peak_res_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    # Verify finite loss
    for k in ("loss", "loss_edos", "loss_phdos", "loss_eta"):
        assert k in out_cold, f"Missing {k}"
        assert torch.isfinite(torch.tensor(out_cold[k])), f"{k} not finite: {out_cold[k]}"

    # Verify G2 alpha gradient
    if use_g2:
        alpha = model.model["transformer"].encoder.g2_msgs[0].alpha
        assert alpha.grad is not None and torch.isfinite(alpha.grad).all(), \
            "G2 alpha has no finite gradient"

    # 2. Warm step on the EXACT SAME Batch 0 (to isolate CUDA kernel latency from JIT/driver overhead)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_ev_warm0 = torch.cuda.Event(enable_timing=True)
    end_ev_warm0 = torch.cuda.Event(enable_timing=True)
    t0_wall_warm0 = time.perf_counter()
    start_ev_warm0.record()

    out_warm0 = model.train_one_step(batch0, step=1)

    end_ev_warm0.record()
    torch.cuda.synchronize()
    t1_wall_warm0 = time.perf_counter()

    warm0_time_ms = start_ev_warm0.elapsed_time(end_ev_warm0)
    warm0_wall_ms = (t1_wall_warm0 - t0_wall_warm0) * 1000.0
    warm0_peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    warm0_peak_res_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    # 3. Steady-state across distinct batches (batches 1..num_steady_steps)
    step_times_ms = []
    step_alloc_mbs = []
    for s_idx in range(1, num_steady_steps + 1):
        b = batches[s_idx]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()

        out_s = model.train_one_step(b, step=s_idx)

        end_ev.record()
        torch.cuda.synchronize()

        step_times_ms.append(start_ev.elapsed_time(end_ev))
        step_alloc_mbs.append(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    steady_mean_time_ms = sum(step_times_ms) / len(step_times_ms)
    steady_peak_alloc_mb = max(step_alloc_mbs)

    metrics = {
        "arm": arm_name,
        "use_g2": use_g2,
        "r_cut": r_cut,
        "trainable_params": total_params,
        "batch_size": 32,
        "batch0_atoms": n_atom_batch0,
        "cold_step0": {
            "gpu_time_ms": cold_time_ms,
            "wall_time_ms": cold_wall_ms,
            "peak_alloc_mb": cold_peak_alloc_mb,
            "peak_reserved_mb": cold_peak_res_mb,
            "loss": out_cold["loss"],
            "loss_edos": out_cold["loss_edos"],
            "loss_phdos": out_cold["loss_phdos"],
            "loss_eta": out_cold["loss_eta"],
        },
        "warm_batch0": {
            "gpu_time_ms": warm0_time_ms,
            "wall_time_ms": warm0_wall_ms,
            "peak_alloc_mb": warm0_peak_alloc_mb,
            "peak_reserved_mb": warm0_peak_res_mb,
            "loss": out_warm0["loss"],
        },
        "steady_state": {
            "num_steps": num_steady_steps,
            "mean_gpu_time_ms": steady_mean_time_ms,
            "min_gpu_time_ms": min(step_times_ms),
            "max_gpu_time_ms": max(step_times_ms),
            "step_times_ms": step_times_ms,
            "peak_alloc_mb": steady_peak_alloc_mb,
        }
    }
    return metrics


def run_isolated_process(arm_name: str, use_g2: bool, r_cut: float, data_dir: str,
                         num_steps: int, tmp_json: str):
    """Execute measure_arm in a clean subprocess to prevent any VRAM cross-contamination."""
    import subprocess
    cmd = [
        sys.executable, __file__,
        "--worker",
        "--arm_name", arm_name,
        "--use_g2" if use_g2 else "--no-use_g2",
        "--r_cut", str(r_cut),
        "--data_dir", data_dir,
        "--num_steps", str(num_steps),
        "--out_json", tmp_json,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Worker {arm_name} failed with code {res.returncode}:")
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        raise RuntimeError(f"Worker {arm_name} crashed (possible OOM or CUDA error)")
    with open(tmp_json) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help="Internal worker mode")
    ap.add_argument("--arm_name", default="B7")
    ap.add_argument("--use_g2", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--r_cut", type=float, default=5.5)
    ap.add_argument("--data_dir", default="./data/train4ARPAT")
    ap.add_argument("--num_steps", type=int, default=5)
    ap.add_argument("--out_json", default="./results/g2_resource_gate_v100.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.worker:
        metrics = measure_arm(
            arm_name=args.arm_name,
            use_g2=args.use_g2,
            r_cut=args.r_cut,
            data_dir=args.data_dir,
            num_steady_steps=args.num_steps,
            device_str=args.device,
        )
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        return

    # Master runner: runs B7 and G2a in separate clean subprocesses
    assert torch.cuda.is_available(), "CUDA is required for V100 gate"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"================================================================")
    print(f"  V100 RESOURCE GATE: B7 vs G2a")
    print(f"  Device: {gpu_name}")
    print(f"  Batch size: 32 (Q1 train) | Fixed Cutoff: {args.r_cut} A")
    print(f"  Requirement: no OOM at batch 32; report VRAM and latency ratios")
    print(f"================================================================")

    os.makedirs("./results", exist_ok=True)
    tmp_b7 = "./results/_tmp_gate_b7.json"
    tmp_g2 = "./results/_tmp_gate_g2.json"

    print("\n[1/2] Measuring B7 (control, use_g2=False) in clean process...")
    b7_res = run_isolated_process("B7", False, args.r_cut, args.data_dir, args.num_steps, tmp_b7)

    print("\n[2/2] Measuring G2a (experimental, use_g2=True) in clean process...")
    g2_res = run_isolated_process("G2a", True, args.r_cut, args.data_dir, args.num_steps, tmp_g2)

    # Compute comparison ratios
    # 1. Warm single-step Batch 0
    time_batch0_b7 = b7_res["warm_batch0"]["gpu_time_ms"]
    time_batch0_g2 = g2_res["warm_batch0"]["gpu_time_ms"]
    time_ratio_batch0 = time_batch0_g2 / time_batch0_b7

    mem_batch0_b7 = b7_res["warm_batch0"]["peak_alloc_mb"]
    mem_batch0_g2 = g2_res["warm_batch0"]["peak_alloc_mb"]
    mem_ratio_batch0 = mem_batch0_g2 / mem_batch0_b7

    # 2. Steady state across 5 batches
    time_steady_b7 = b7_res["steady_state"]["mean_gpu_time_ms"]
    time_steady_g2 = g2_res["steady_state"]["mean_gpu_time_ms"]
    time_ratio_steady = time_steady_g2 / time_steady_b7

    mem_steady_b7 = b7_res["steady_state"]["peak_alloc_mb"]
    mem_steady_g2 = g2_res["steady_state"]["peak_alloc_mb"]
    mem_ratio_steady = mem_steady_g2 / mem_steady_b7

    # 3. Cold step 0
    time_cold_b7 = b7_res["cold_step0"]["gpu_time_ms"]
    time_cold_g2 = g2_res["cold_step0"]["gpu_time_ms"]
    time_ratio_cold = time_cold_g2 / time_cold_b7

    mem_cold_b7 = b7_res["cold_step0"]["peak_alloc_mb"]
    mem_cold_g2 = g2_res["cold_step0"]["peak_alloc_mb"]
    mem_ratio_cold = mem_cold_g2 / mem_cold_b7

    # The workers returning successfully establishes the only resource gate:
    # fixed batch-32 execution completed without OOM.  Cost ratios are evidence,
    # not an autonomous scientific stop rule.
    verdict = "MEASURED_NO_OOM"

    summary = {
        "device": gpu_name,
        "batch_size": 32,
        "r_cut": args.r_cut,
        "b7": b7_res,
        "g2": g2_res,
        "comparison": {
            "batch0_warm": {
                "b7_time_ms": time_batch0_b7,
                "g2_time_ms": time_batch0_g2,
                "time_ratio": time_ratio_batch0,
                "time_diff_pct": (time_ratio_batch0 - 1.0) * 100.0,
                "b7_peak_alloc_mb": mem_batch0_b7,
                "g2_peak_alloc_mb": mem_batch0_g2,
                "mem_ratio": mem_ratio_batch0,
                "mem_diff_pct": (mem_ratio_batch0 - 1.0) * 100.0,
                "measured_without_oom": True,
            },
            "steady_state": {
                "b7_mean_time_ms": time_steady_b7,
                "g2_mean_time_ms": time_steady_g2,
                "time_ratio": time_ratio_steady,
                "time_diff_pct": (time_ratio_steady - 1.0) * 100.0,
                "b7_peak_alloc_mb": mem_steady_b7,
                "g2_peak_alloc_mb": mem_steady_g2,
                "mem_ratio": mem_ratio_steady,
                "mem_diff_pct": (mem_ratio_steady - 1.0) * 100.0,
                "measured_without_oom": True,
            },
            "batch0_cold": {
                "b7_time_ms": time_cold_b7,
                "g2_time_ms": time_cold_g2,
                "time_ratio": time_ratio_cold,
                "b7_peak_alloc_mb": mem_cold_b7,
                "g2_peak_alloc_mb": mem_cold_g2,
                "mem_ratio": mem_ratio_cold,
            },
        },
      "verdict": verdict,
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Also save a flat CSV for audit
    csv_path = args.out_json.replace(".json", ".csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "metric_scope", "gpu_time_ms", "peak_alloc_mb", "peak_reserved_mb", "loss"])
        writer.writerow(["B7", "cold_step0", f"{time_cold_b7:.1f}", f"{mem_cold_b7:.1f}", f"{b7_res['cold_step0']['peak_reserved_mb']:.1f}", f"{b7_res['cold_step0']['loss']:.4f}"])
        writer.writerow(["B7", "warm_batch0", f"{time_batch0_b7:.1f}", f"{mem_batch0_b7:.1f}", f"{b7_res['warm_batch0']['peak_reserved_mb']:.1f}", f"{b7_res['warm_batch0']['loss']:.4f}"])
        writer.writerow(["B7", "steady_state_mean", f"{time_steady_b7:.1f}", f"{mem_steady_b7:.1f}", "-", "-"])
        writer.writerow(["G2a", "cold_step0", f"{time_cold_g2:.1f}", f"{mem_cold_g2:.1f}", f"{g2_res['cold_step0']['peak_reserved_mb']:.1f}", f"{g2_res['cold_step0']['loss']:.4f}"])
        writer.writerow(["G2a", "warm_batch0", f"{time_batch0_g2:.1f}", f"{mem_batch0_g2:.1f}", f"{g2_res['warm_batch0']['peak_reserved_mb']:.1f}", f"{g2_res['warm_batch0']['loss']:.4f}"])
        writer.writerow(["G2a", "steady_state_mean", f"{time_steady_g2:.1f}", f"{mem_steady_g2:.1f}", "-", "-"])

    # Clean up temp files
    for p in (tmp_b7, tmp_g2):
        if os.path.exists(p):
            os.remove(p)

    print("\n" + "="*64)
    print("  RESOURCE GATE RESULTS SUMMARY")
    print("="*64)
    print(f"Device: {gpu_name}")
    print(f"B7 Parameters:  {b7_res['trainable_params']:,}")
    print(f"G2a Parameters: {g2_res['trainable_params']:,} (+{(g2_res['trainable_params']-b7_res['trainable_params'])/b7_res['trainable_params']*100:.2f}%)")
    print("-" * 64)
    print(f"[1] Single-step on identical Batch 0 (warm):")
    print(f"    Peak VRAM: B7 = {mem_batch0_b7:.1f} MB | G2a = {mem_batch0_g2:.1f} MB")
    print(f"    VRAM Ratio: {mem_ratio_batch0:.4f}x (+{(mem_ratio_batch0-1)*100:.2f}%)")
    print(f"    Step Time: B7 = {time_batch0_b7:.1f} ms | G2a = {time_batch0_g2:.1f} ms")
    print(f"    Time Ratio: {time_ratio_batch0:.4f}x (+{(time_ratio_batch0-1)*100:.2f}%)")
    print("-" * 64)
    print(f"[2] Steady-state across {args.num_steps} distinct batches:")
    print(f"    Peak VRAM: B7 = {mem_steady_b7:.1f} MB | G2a = {mem_steady_g2:.1f} MB")
    print(f"    VRAM Ratio: {mem_ratio_steady:.4f}x (+{(mem_ratio_steady-1)*100:.2f}%)")
    print(f"    Mean Time: B7 = {time_steady_b7:.1f} ms | G2a = {time_steady_g2:.1f} ms")
    print(f"    Time Ratio: {time_ratio_steady:.4f}x (+{(time_ratio_steady-1)*100:.2f}%)")
    print("=" * 64)
    print(f"VERDICT: {verdict}")
    print(f"Detailed output saved to: {args.out_json} and {csv_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
