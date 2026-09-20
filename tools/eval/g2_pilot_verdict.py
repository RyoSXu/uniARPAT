"""G2a Pilot Evaluation & Verdict Tool.

Evaluates test set under both oracle and blind protocols for _g2ctl and _g2edge,
computes spectral metrics (median R2, failure rate), oracle-blind gap distribution,
Cv MAE, training cost (time, VRAM), and checks pre-registered accuracy criteria.

Saves:
  results/test_m1_{tag}_blind_summary.csv
  results/samples_m1_{tag}_blind_test.csv
Outputs formatted comparison table and formal verdict.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.builder import ConfigBuilder
from utils.metrics import per_sample_spectral_metrics


def evaluate_tag_blind(tag: str, device: torch.device):
    """Run blind and oracle evaluation for a trained checkpoint."""
    clean_tag = tag.lstrip("_")
    model_name = f"m1_{clean_tag}"
    save_dir = os.path.join(REPO_ROOT, "output", f"ablation_{model_name}")
    ckpt_path = os.path.join(save_dir, "checkpoint_best.pth")
    cfg_used_path = os.path.join(save_dir, "config_used.yaml")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    if not os.path.exists(cfg_used_path):
        raise FileNotFoundError(f"Config used not found at {cfg_used_path}")

    with open(cfg_used_path) as f:
        full_cfg = yaml.load(f, Loader=yaml.FullLoader)
    yaml_cfg = full_cfg["config"]

    builder = ConfigBuilder(**yaml_cfg)
    test_loader = builder.get_dataloader(
        split="test", dos_minmax=True, batch_size=32, dos_sumnorm=True
    )
    model = builder.get_model()
    model.to(device)

    ck = torch.load(ckpt_path, map_location="cpu")
    best_epoch = ck.get("epoch", None) if isinstance(ck, dict) else None
    st = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.model["transformer"].load_state_dict(st)
    model.to(device)
    model.model["transformer"].eval()

    total_params = sum(p.numel() for p in model.model["transformer"].parameters())
    trainable_params = sum(
        p.numel() for p in model.model["transformer"].parameters() if p.requires_grad
    )

    D_e = float(model.params.get("delta_edos", 0.09375))
    D_p = float(model.params.get("delta_phdos", 19.6875))

    recs = []
    with torch.no_grad():
        for batch in test_loader:
            out = model.data_preprocess(batch)
            (
                inp, pos, mask, edos_tgt, phdos_tgt,
                edos_m, edos_s, edos_min, edos_max,
                phdos_m, phdos_s, phdos_min, phdos_max,
                edos_cov, phdos_cov, nval, edos_x, phdos_x
            ) = out

            outs = model.model["transformer"](inp, mask, pos, edos_x, phdos_x)
            pe = F.softmax(outs["edos"], dim=-1)
            pp = F.softmax(outs["phdos"], dim=-1)
            Se = edos_max - edos_min
            Sp = phdos_max - phdos_min
            t_e = edos_tgt * Se + edos_min
            t_p = phdos_tgt * Sp + phdos_min

            # Oracle reconstruction
            o_e = torch.clamp(pe * Se + edos_min, min=0.0)
            o_p = torch.clamp(pp * Sp + phdos_min, min=0.0)

            # Blind reconstruction via H1 eta/gamma heads
            atom_len = pos.shape[1] - 2
            nat = (~mask[:, 2:2 + atom_len]).sum(dim=-1).float().clamp_min(1.0)
            eg = outs["eta"]  # [:, 0]=eta_ph, [:, 1]=gamma_e
            Sph_hat = 3.0 * nat.unsqueeze(-1) * eg[:, 0:1] / D_p
            b_p = torch.clamp(pp * Sph_hat, min=0.0)
            eta_true = (phdos_max.clamp_min(1e-12) * D_p / (3.0 * nat.unsqueeze(-1))).clamp(0, 1)

            if nval is not None:
                nv = nval.reshape(-1, 1).float()
                finite = torch.isfinite(nv).squeeze(-1)
                Se_hat = torch.where(
                    finite.unsqueeze(-1),
                    nv.clamp_min(1e-12) * eg[:, 1:2] / D_e,
                    torch.zeros_like(Se),
                )
                gam_true = torch.where(
                    finite.unsqueeze(-1),
                    (edos_max.clamp_min(1e-12) * D_e / nv.clamp_min(1e-12)).clamp(0, 1),
                    torch.zeros_like(Se),
                )
            else:
                finite = torch.zeros(inp.shape[0], dtype=torch.bool, device=inp.device)
                Se_hat = torch.zeros_like(Se)
                gam_true = torch.zeros_like(Se)

            b_e = torch.clamp(pe * Se_hat, min=0.0)

            m_oe = per_sample_spectral_metrics(o_e, t_e)
            m_op = per_sample_spectral_metrics(o_p, t_p)
            m_be = per_sample_spectral_metrics(b_e, t_e)
            m_bp = per_sample_spectral_metrics(b_p, t_p)

            B = inp.shape[0]
            for i in range(B):
                recs.append({
                    "r2_edos_oracle": m_oe["r2"][i].item(),
                    "r2_edos_blind": m_be["r2"][i].item(),
                    "mae_edos_oracle": m_oe["mae"][i].item(),
                    "mae_edos_blind": m_be["mae"][i].item(),
                    "r2_phdos_oracle": m_op["r2"][i].item(),
                    "r2_phdos_blind": m_bp["r2"][i].item(),
                    "mae_phdos_oracle": m_op["mae"][i].item(),
                    "mae_phdos_blind": m_bp["mae"][i].item(),
                    "eta_true": eta_true[i].item(),
                    "eta_pred": eg[i, 0].item(),
                    "gamma_true": gam_true[i].item(),
                    "gamma_pred": eg[i, 1].item(),
                    "natoms": nat[i].item(),
                    "nval": float(nval.reshape(-1)[i].item()) if nval is not None else float("nan"),
                    "nval_finite": bool(finite[i].item()),
                })

    df_samples = pd.DataFrame(recs)

    def q(s, p):
        return float(np.quantile(s, p))

    summary = {
        "tag": tag,
        "best_epoch": best_epoch,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "n": len(df_samples),
    }

    for arm, o, b in (("edos", "r2_edos_oracle", "r2_edos_blind"),
                      ("phdos", "r2_phdos_oracle", "r2_phdos_blind")):
        gap = df_samples[o] - df_samples[b]
        summary.update({
            f"oracle_{arm}_med": float(df_samples[o].median()),
            f"oracle_{arm}_fail": float((df_samples[o] < 0).mean() * 100),
            f"blind_{arm}_med": float(df_samples[b].median()),
            f"blind_{arm}_fail": float((df_samples[b] < 0).mean() * 100),
            f"gap_{arm}_p50": q(gap, 0.5),
            f"gap_{arm}_p90": q(gap, 0.9),
            f"gap_{arm}_p99": q(gap, 0.99),
            f"gap_{arm}_mean": float(gap.mean()),
        })

    summary["eta_mae"] = float((df_samples["eta_pred"] - df_samples["eta_true"]).abs().mean())
    fin = df_samples[df_samples["nval_finite"]]
    summary["gamma_mae"] = float((fin["gamma_pred"] - fin["gamma_true"]).abs().mean()) if len(fin) > 0 else float("nan")

    # Read official test summary for Cv MAE and existing oracle metrics
    test_summary_path = os.path.join(REPO_ROOT, "results", f"test_{model_name}_summary.csv")
    if os.path.exists(test_summary_path):
        df_ts = pd.read_csv(test_summary_path)
        if "cv_mae" in df_ts.columns:
            summary["cv_mae"] = float(df_ts["cv_mae"].iloc[0])

    # Read training history for cost metrics
    hist_path = os.path.join(REPO_ROOT, "results", f"history_{model_name}.csv")
    if os.path.exists(hist_path):
        df_h = pd.read_csv(hist_path)
        if "epoch_time_s" in df_h.columns:
            summary["epoch_time_mean_s"] = float(df_h["epoch_time_s"].mean())
            summary["epoch_time_total_s"] = float(df_h["epoch_time_s"].sum())
        if "peak_vram_mb" in df_h.columns:
            summary["peak_vram_mb"] = float(df_h["peak_vram_mb"].max())

    # Save blind CSVs
    blind_summary_path = os.path.join(REPO_ROOT, "results", f"test_{model_name}_blind_summary.csv")
    pd.DataFrame([summary]).to_csv(blind_summary_path, index=False)
    samples_blind_path = os.path.join(REPO_ROOT, "results", f"samples_{model_name}_blind_test.csv")
    df_samples.to_csv(samples_blind_path, index=False)

    return summary


def compare_arms(ctl_res: dict, exp_res: dict):
    """Compare control arm against experimental arm and evaluate decision rules."""
    delta_e_orc_med = exp_res["oracle_edos_med"] - ctl_res["oracle_edos_med"]
    delta_e_orc_fail = exp_res["oracle_edos_fail"] - ctl_res["oracle_edos_fail"]
    delta_p_orc_med = exp_res["oracle_phdos_med"] - ctl_res["oracle_phdos_med"]
    delta_p_orc_fail = exp_res["oracle_phdos_fail"] - ctl_res["oracle_phdos_fail"]

    delta_e_bld_med = exp_res["blind_edos_med"] - ctl_res["blind_edos_med"]
    delta_e_bld_fail = exp_res["blind_edos_fail"] - ctl_res["blind_edos_fail"]
    delta_p_bld_med = exp_res["blind_phdos_med"] - ctl_res["blind_phdos_med"]
    delta_p_bld_fail = exp_res["blind_phdos_fail"] - ctl_res["blind_phdos_fail"]

    delta_cv = exp_res.get("cv_mae", float("nan")) - ctl_res.get("cv_mae", float("nan"))
    time_ratio = (
        exp_res.get("epoch_time_mean_s", 1.0) / ctl_res.get("epoch_time_mean_s", 1.0)
        if "epoch_time_mean_s" in ctl_res and "epoch_time_mean_s" in exp_res
        else float("nan")
    )
    vram_ratio = (
        exp_res.get("peak_vram_mb", 1.0) / ctl_res.get("peak_vram_mb", 1.0)
        if "peak_vram_mb" in ctl_res and "peak_vram_mb" in exp_res
        else float("nan")
    )

    print("\n" + "=" * 90)
    print("                      G2a PILOT (10 EPOCHS) PAIRWISE COMPARISON")
    print("=" * 90)
    print(f"{'Metric':<32} | {'Control (_g2ctl)':<18} | {'G2a (_g2edge)':<18} | {'Delta / Ratio':<16}")
    print("-" * 90)
    print(f"{'Best Epoch':<32} | {ctl_res.get('best_epoch', 'N/A')!s:<18} | {exp_res.get('best_epoch', 'N/A')!s:<18} | -")
    print(f"{'Trainable Parameters':<32} | {ctl_res.get('trainable_params', 0):<18,d} | {exp_res.get('trainable_params', 0):<18,d} | {exp_res.get('trainable_params', 0) - ctl_res.get('trainable_params', 0):+,d}")
    print(f"{'eDOS Oracle Med R2 / Fail':<32} | {ctl_res['oracle_edos_med']:.4f} / {ctl_res['oracle_edos_fail']:.2f}% | {exp_res['oracle_edos_med']:.4f} / {exp_res['oracle_edos_fail']:.2f}% | {delta_e_orc_med:+.4f} / {delta_e_orc_fail:+.2f}pt")
    print(f"{'phDOS Oracle Med R2 / Fail':<32} | {ctl_res['oracle_phdos_med']:.4f} / {ctl_res['oracle_phdos_fail']:.2f}% | {exp_res['oracle_phdos_med']:.4f} / {exp_res['oracle_phdos_fail']:.2f}% | {delta_p_orc_med:+.4f} / {delta_p_orc_fail:+.2f}pt")
    print("-" * 90)
    print(f"{'eDOS Blind Med R2 / Fail':<32} | {ctl_res['blind_edos_med']:.4f} / {ctl_res['blind_edos_fail']:.2f}% | {exp_res['blind_edos_med']:.4f} / {exp_res['blind_edos_fail']:.2f}% | {delta_e_bld_med:+.4f} / {delta_e_bld_fail:+.2f}pt")
    print(f"{'phDOS Blind Med R2 / Fail':<32} | {ctl_res['blind_phdos_med']:.4f} / {ctl_res['blind_phdos_fail']:.2f}% | {exp_res['blind_phdos_med']:.4f} / {exp_res['blind_phdos_fail']:.2f}% | {delta_p_bld_med:+.4f} / {delta_p_bld_fail:+.2f}pt")
    print(f"{'eDOS Gap p50 / p90 / p99':<32} | {ctl_res['gap_edos_p50']:.3f} / {ctl_res['gap_edos_p90']:.3f} / {ctl_res['gap_edos_p99']:.3f} | {exp_res['gap_edos_p50']:.3f} / {exp_res['gap_edos_p90']:.3f} / {exp_res['gap_edos_p99']:.3f} | -")
    print(f"{'phDOS Gap p50 / p90 / p99':<32} | {ctl_res['gap_phdos_p50']:.3f} / {ctl_res['gap_phdos_p90']:.3f} / {ctl_res['gap_phdos_p99']:.3f} | {exp_res['gap_phdos_p50']:.3f} / {exp_res['gap_phdos_p90']:.3f} / {exp_res['gap_phdos_p99']:.3f} | -")
    print(f"{'eta MAE / gamma MAE':<32} | {ctl_res['eta_mae']:.4f} / {ctl_res['gamma_mae']:.4f} | {exp_res['eta_mae']:.4f} / {exp_res['gamma_mae']:.4f} | {exp_res['eta_mae'] - ctl_res['eta_mae']:+.4f} / {exp_res['gamma_mae'] - ctl_res['gamma_mae']:+.4f}")
    print(f"{'Cv MAE':<32} | {ctl_res.get('cv_mae', float('nan')):.4f} | {exp_res.get('cv_mae', float('nan')):.4f} | {delta_cv:+.4f}")
    print("-" * 90)
    print(f"{'Mean Epoch Time (s)':<32} | {ctl_res.get('epoch_time_mean_s', float('nan')):.1f}s | {exp_res.get('epoch_time_mean_s', float('nan')):.1f}s | {time_ratio:.3f}x ({(time_ratio - 1) * 100:+.1f}%)")
    print(f"{'Peak VRAM (MB)':<32} | {ctl_res.get('peak_vram_mb', float('nan')):.0f} MB | {exp_res.get('peak_vram_mb', float('nan')):.0f} MB | {vram_ratio:.3f}x ({(vram_ratio - 1) * 100:+.1f}%)")
    print("=" * 90)

    # Pre-registered verdict checks:
    has_win_task = (delta_e_orc_med >= 0.02) or (delta_p_orc_med >= 0.02)
    no_fail_worse = (delta_e_orc_fail < 1.0) and (delta_p_orc_fail < 1.0)
    no_regress_other = (delta_e_orc_med >= -0.02) and (delta_p_orc_med >= -0.02)

    is_win = has_win_task and no_fail_worse and no_regress_other
    is_tie = (abs(delta_e_orc_med) < 0.02 and abs(delta_p_orc_med) < 0.02 and
              abs(delta_e_orc_fail) < 1.0 and abs(delta_p_orc_fail) < 1.0)

    print("\nVERDICT ANALYSIS:")
    print(f"  Condition 1: At least one task med R2 >= +0.02: {has_win_task} (eDOS: {delta_e_orc_med:+.4f}, phDOS: {delta_p_orc_med:+.4f})")
    print(f"  Condition 2: Neither fail rate worsens >= 1.0pt: {no_fail_worse} (eDOS: {delta_e_orc_fail:+.2f}pt, phDOS: {delta_p_orc_fail:+.2f}pt)")
    print(f"  Condition 3: Neither task med R2 drops < -0.02: {no_regress_other}")
    print(f"  Statistical tie line (|d_med| < 0.02 & |d_fail| < 1.0pt): {is_tie}")

    if is_win:
        verdict = "WIN"
        action = "G2a qualifies for 35-epoch confirmation (Q1 M1x35 _g2longctl vs _g2longedge)."
    elif is_tie:
        verdict = "PARK"
        action = "G2a is within statistical tie line (|d_med| < 0.02 and |d_fail| < 1.0pt). Park G2a (default off, no scanning)."
    else:
        verdict = "PARK / DEAD"
        action = "G2a does not achieve win and shows regression or failed bounds. Park G2a (default off, no scanning)."

    print(f"\nFINAL VERDICT: [{verdict}]")
    print(f"ACTION: {action}\n")

    return {
        "verdict": verdict,
        "action": action,
        "is_win": is_win,
        "is_tie": is_tie,
        "delta_e_orc_med": delta_e_orc_med,
        "delta_e_orc_fail": delta_e_orc_fail,
        "delta_p_orc_med": delta_p_orc_med,
        "delta_p_orc_fail": delta_p_orc_fail,
        "delta_e_bld_med": delta_e_bld_med,
        "delta_e_bld_fail": delta_e_bld_fail,
        "delta_p_bld_med": delta_p_bld_med,
        "delta_p_bld_fail": delta_p_bld_fail,
        "time_ratio": time_ratio,
        "vram_ratio": vram_ratio,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G2a Pilot Verdict & Comparison")
    parser.add_argument("--ctl_tag", type=str, default="_g2ctl", help="Control arm tag")
    parser.add_argument("--exp_tag", type=str, default="_g2edge", help="Experimental arm tag")
    parser.add_argument("--single_tag", type=str, default="", help="Evaluate single tag only")
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {dev}")

    if args.single_tag:
        res = evaluate_tag_blind(args.single_tag, dev)
        print(f"Summary for {args.single_tag}: {res}")
    else:
        print(f"Evaluating Control Arm ({args.ctl_tag})...")
        ctl = evaluate_tag_blind(args.ctl_tag, dev)
        print(f"Evaluating Experimental Arm ({args.exp_tag})...")
        exp = evaluate_tag_blind(args.exp_tag, dev)
        compare_arms(ctl, exp)
