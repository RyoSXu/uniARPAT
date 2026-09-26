"""从已完成的小样本正式CSV绘图，不重新训练或重新判定结果。"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(prefix):
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    if metadata["status"] != "complete":
        raise ValueError("only completed runs can be plotted")
    history = pd.read_csv(prefix.with_name(prefix.name + "_history.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    specs = [("max_mse_ratio", "Worst-pair MSE / zero-contrast MSE", .01),
             ("max_tv_ratio", "Worst-pair TV / zero-contrast TV", .1),
             ("n_pairs_passed", "Pairs passing both gates", metadata["n_pairs"])]
    colors = {"frozen": "#2563eb", "joint": "#c2410c", "g2_only": "#15803d", "non_g2": "#7e22ce", "last_layer": "#c0267b"}
    labels = {"frozen": "Frozen encoder", "joint": "Encoder + readout", "g2_only": "G2 messages + readout",
              "non_g2": "Non-G2 encoder + readout", "last_layer": "Last encoder layer + readout"}
    for axis, (metric, title, threshold) in zip(axes, specs):
        for arm, rows in history.groupby("arm", sort=False):
            axis.plot(rows.step, rows[metric], color=colors[arm], label=labels[arm], linewidth=2)
        axis.axhline(threshold, color="#64748b", linestyle="--", linewidth=1, label="Required gate")
        if metric != "n_pairs_passed":
            axis.set_yscale("log")
        else:
            axis.set_ylim(-.5, metadata["n_pairs"] + .5)
            axis.set_yticks(range(0, metadata["n_pairs"] + 1, 4))
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Optimizer steps on the same 16 pairs")
        axis.grid(alpha=.15)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle("G2: fitting 16 fixed training pairs (no validation / test evaluation)", fontsize=12)
    path = prefix.with_name(prefix.name + "_learning.png")
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=Path(__file__).resolve().parents[2] / "results/g2_small_fit_q1")
    print(plot_results(parser.parse_args().prefix))
