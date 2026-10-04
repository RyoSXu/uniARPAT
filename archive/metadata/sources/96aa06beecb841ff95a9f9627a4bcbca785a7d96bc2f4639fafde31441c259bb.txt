"""Plot radius audit coverage and cost from its reproducible summary artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.audit_dir / "coverage_cost.png"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    manifest = json.loads((args.audit_dir / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("audit must be complete before plotting")
    with (args.audit_dir / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    result = json.loads((args.audit_dir / "summary.json").read_text())
    tolerance = min(manifest["shell_tolerances_A"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for split, color in (("train", "#1676a5"), ("valid", "#c04e26")):
        selected = [row for row in rows if row["split"] == split and float(row["shell_tolerance_A"]) == tolerance]
        radii = np.array([float(row["radius_A"]) for row in selected])
        values = lambda key: np.array([float(row[key]) for row in selected])
        axes[0,0].plot(radii, values("degree_p50"), "o-", color=color, label=f"{split}: median")
        axes[0,0].plot(radii, values("degree_p99"), "--", color=color, label=f"{split}: p99")
        axes[0,1].plot(radii, 100*values("shell1_structures_fully_covered_fraction"), "o-", color=color,
                       label=f"{split}: first shell")
        axes[0,1].plot(radii, 100*values("shell3_structures_fully_covered_fraction"), "--", color=color,
                       label=f"{split}: third shell")
        base = np.where(radii == 5.5)[0]
        if len(base) != 1:
            raise ValueError("plot requires the 5.5 A reference candidate")
        edges, pairs = values("total_edges"), values("potential_unordered_angle_pairs")
        axes[1,0].plot(radii, edges/edges[base[0]], "o-", color=color, label=f"{split}: records")
        axes[1,0].plot(radii, pairs/pairs[base[0]], "--", color=color, label=f"{split}: possible angle pairs")
        timing = result["benchmark"][split]["seconds_per_structure"]
        axes[1,1].plot(radii, [1000*timing[str(radius)]["mean"] for radius in radii], "o-", color=color,
                       label=split)
    labels = (("Periodic neighbors per atom", "Number of directed records"),
              ("Structures with complete shell coverage", "Structures (%)"),
              ("Structural cost proxies", "Ratio to 5.5 A"),
              ("CPU enumeration on fixed samples", "Mean time per structure (ms)"))
    for ax, (title, ylabel) in zip(axes.ravel(), labels):
        ax.set_title(title)
        ax.set_xlabel("Radius (A)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    axes[0,1].set_ylim(0, 101)
    fig.suptitle(f"Q1 train/valid: geometry coverage and cost\n"
                 f"Diagnostic shell tolerance {tolerance:g} A; CPU single thread; model cost unmeasured", fontsize=13)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
