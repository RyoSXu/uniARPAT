"""Export the verified A/B/B+ CPU comparison as a static scientific figure."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.eval.element_identity_preflight import sha256_file
from tools.eval.periodic_geometry_acceptance import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    verification = json.loads((args.out_dir/"verification.json").read_text())
    manifest = json.loads((args.out_dir/"manifest.json").read_text())
    summary = json.loads((args.out_dir/"summary.json").read_text())
    assert verification["status"] == "pass"
    assert sha256_file(args.out_dir/"manifest.json") == verification["main_manifest_sha256"]
    for name in ("comparison.png", "comparison.svg", "figure.json"):
        if (args.out_dir/name).exists():
            raise FileExistsError(f"refusing to overwrite {name}")
    routes = ("A", "B", "B+")
    colors = ("#4477AA", "#999999", "#228833")
    panels = [
        ("Local parameters", "thousand parameters", [manifest["parameter_counts"][r]/1000 for r in routes], "{:.1f}"),
        ("Forward + backward (no update)", "median milliseconds per structure", [summary["comparison"][r]["forward_plus_backward_seconds"]["median"]*1000 for r in routes], "{:.1f}"),
        ("Fresh-process peak memory", "MiB (same largest-pair structure)", [verification["isolated"][r]["peak_rss_MiB"] for r in routes], "{:.0f}"),
        ("Controlled angular counterexample", "seeds with a resolved difference / 5", [summary["comparison"][r]["controlled_counterexample"]["resolved_seeds"] for r in routes], "{:.0f}/5"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.7), layout="constrained")
    fig.suptitle("A / B / enhanced B+: untrained local-message comparison", fontsize=15)
    for ax, (title, ylabel, values, pattern) in zip(axes.flat, panels):
        bars = ax.bar(routes, values, color=colors, width=.55)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values)*1.22 if max(values) else 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+max(values)*.035,
                    pattern.format(value), ha="center", fontsize=11)
    fig.supxlabel("CPU: 16 fixed Q1 train/valid structures, 3 timing repeats per route/structure.\n"
                  "Counterexample: float64, cutoff=1 ablation, threshold 1e-10; native near-cutoff weakness remains.\n"
                  "Local-feature checks establish neither DOS accuracy nor complete-model invariance.", fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig(args.out_dir/f"comparison.{suffix}", dpi=180)
    plt.close(fig)
    write_json(args.out_dir/"figure.json", {"main_manifest_sha256": sha256_file(args.out_dir/"manifest.json"),
               "verification_sha256": sha256_file(args.out_dir/"verification.json"),
               "source_sha256": sha256_file(Path(__file__)),
               "panels": [{"title": title, "ylabel": ylabel, "values": dict(zip(routes, values))}
                          for title, ylabel, values, _ in panels],
               "artifacts_sha256": {name: sha256_file(args.out_dir/name) for name in ("comparison.png", "comparison.svg")}})
    print(args.out_dir/"comparison.png")


if __name__ == "__main__":
    main()
