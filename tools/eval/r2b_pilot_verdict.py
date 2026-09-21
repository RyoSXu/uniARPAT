#!/usr/bin/env python3
"""R2b atom-additive phDOS pilot verdict against the R2a carrier."""
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
    parser.add_argument("--ctl_tag", default="_r2a3")
    parser.add_argument("--exp_tag", default="_apdossum")
    parser.add_argument("--out_json", default="./results/r2b_pilot_verdict.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    control = evaluate_tag_blind(args.ctl_tag, device)
    experiment = evaluate_tag_blind(args.exp_tag, device)
    deltas = {
        "oracle_edos_med": experiment["oracle_edos_med"] - control["oracle_edos_med"],
        "oracle_edos_fail_pt": experiment["oracle_edos_fail"] - control["oracle_edos_fail"],
        "oracle_phdos_med": experiment["oracle_phdos_med"] - control["oracle_phdos_med"],
        "oracle_phdos_fail_pt": experiment["oracle_phdos_fail"] - control["oracle_phdos_fail"],
        "blind_edos_med": experiment["blind_edos_med"] - control["blind_edos_med"],
        "blind_edos_fail_pt": experiment["blind_edos_fail"] - control["blind_edos_fail"],
        "blind_phdos_med": experiment["blind_phdos_med"] - control["blind_phdos_med"],
        "blind_phdos_fail_pt": experiment["blind_phdos_fail"] - control["blind_phdos_fail"],
        "cv_mae": experiment["cv_mae"] - control["cv_mae"],
    }
    ph_win = deltas["oracle_phdos_med"] >= 0.02 and deltas["oracle_phdos_fail_pt"] < 1.0
    e_protected = deltas["oracle_edos_med"] >= -0.02 and deltas["oracle_edos_fail_pt"] < 1.0
    verdict = "WIN" if ph_win and e_protected else "PARK"
    out = {
        "control_tag": args.ctl_tag,
        "experiment_tag": args.exp_tag,
        "control": control,
        "experiment": experiment,
        "deltas": deltas,
        "phdos_win": ph_win,
        "edos_protected": e_protected,
        "verdict": verdict,
        "action": "35-epoch confirmation" if verdict == "WIN" else "park; no head scan or confirmation",
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
