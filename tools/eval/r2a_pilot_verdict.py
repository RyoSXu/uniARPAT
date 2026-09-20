#!/usr/bin/env python3
"""R2a paired verdict: decoder-depth non-inferiority plus measured cost.

Blind metrics are evaluated through the shared H1 evaluator used by prior
pilots.  R2a's decision rule is deliberately different from an accuracy-win
experiment: it accepts a lower-cost decoder only when both tasks remain within
the pre-registered negative tie boundary.
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "../.."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from g2_pilot_verdict import evaluate_tag_blind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctl_tag", default="_r2a6ctl")
    parser.add_argument("--exp_tag", default="_r2a3")
    parser.add_argument("--out_json", default="./results/r2a_pilot_verdict.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    control = evaluate_tag_blind(args.ctl_tag, device)
    experiment = evaluate_tag_blind(args.exp_tag, device)

    de_med = experiment["oracle_edos_med"] - control["oracle_edos_med"]
    de_fail = experiment["oracle_edos_fail"] - control["oracle_edos_fail"]
    dp_med = experiment["oracle_phdos_med"] - control["oracle_phdos_med"]
    dp_fail = experiment["oracle_phdos_fail"] - control["oracle_phdos_fail"]
    time_ratio = experiment["epoch_time_mean_s"] / control["epoch_time_mean_s"]
    mem_ratio = experiment["peak_vram_mb"] / control["peak_vram_mb"]
    noninferior = de_med >= -0.02 and dp_med >= -0.02 and de_fail < 1.0 and dp_fail < 1.0
    cost_reduced = time_ratio < 1.0 or mem_ratio < 1.0
    verdict = "ACCEPT_LOW_COST_CARRIER" if noninferior and cost_reduced else "PARK"
    out = {
        "control_tag": args.ctl_tag,
        "experiment_tag": args.exp_tag,
        "control": control,
        "experiment": experiment,
        "deltas": {
            "oracle_edos_med": de_med, "oracle_edos_fail_pt": de_fail,
            "oracle_phdos_med": dp_med, "oracle_phdos_fail_pt": dp_fail,
            "blind_edos_med": experiment["blind_edos_med"] - control["blind_edos_med"],
            "blind_edos_fail_pt": experiment["blind_edos_fail"] - control["blind_edos_fail"],
            "blind_phdos_med": experiment["blind_phdos_med"] - control["blind_phdos_med"],
            "blind_phdos_fail_pt": experiment["blind_phdos_fail"] - control["blind_phdos_fail"],
            "cv_mae": experiment["cv_mae"] - control["cv_mae"],
        },
        "cost": {"epoch_time_ratio": time_ratio, "peak_vram_ratio": mem_ratio},
        "noninferior": noninferior,
        "cost_reduced": cost_reduced,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
