"""ZP100 (z_only_proj) 训练监控：等待目标 epoch 并打印一次健康快照。

用法（从仓库根目录）：

* ``python3 tools/eval/eid_zproj_monitor.py`` —— 立即打印当前快照；
* ``python3 tools/eval/eid_zproj_monitor.py --wait-epoch N`` —— 轮询直到 history
  记录到第 N 轮（或训练结束/异常），打印快照后退出；
* ``python3 tools/eval/eid_zproj_monitor.py --wait-finish`` —— 轮询直到日志出现结束标记、
  进程退出或异常。

退出码：0 健康；2 进程退出但未达目标；3 history 出现非有限值；4 日志出现异常关键词；
5 等待超时。快照包含 epoch、训练损失与分量、valid 指标、最佳 epoch、耗时、显存与产物更新。
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "results/history_m1_eidzproj100_s42.csv"
LOG = ROOT / "results/train_m1_eidzproj100_s42.log"
SAVE_DIR = ROOT / "output/ablation_m1_eidzproj100_s42"
TAG_PATTERN = "eidzproj100_s42"
END_MARKER = "Test loader and automatic test evaluation skipped"
BAD_PATTERNS = ("Traceback", "CUDA out of memory", "non-finite", "loss is nan",
                "RuntimeError", "ValueError", "Killed")

WATCH_COLUMNS = (
    "epoch", "train_loss", "train_loss_edos", "train_loss_phdos", "train_loss_eta",
    "train_loss_shape_e", "train_loss_shape_p", "train_loss_scale_e", "train_loss_scale_p",
    "balanced_score", "r2_edos_median", "r2_phdos_median",
    "fail_rate_edos", "fail_rate_phdos", "mae_edos_median", "mae_phdos_median",
    "epoch_time_s", "peak_vram_mb",
)


def read_history() -> list[dict]:
    if not HISTORY.exists():
        return []
    with HISTORY.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def process_alive() -> list[int]:
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            cmdline = (Path("/proc") / entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if TAG_PATTERN in cmdline and "run_ablation_experiments" in cmdline:
            pids.append(int(entry))
    return sorted(pids)


def log_tail(n: int = 3) -> list[str]:
    if not LOG.exists():
        return []
    lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]


def snapshot() -> tuple[str, bool, str]:
    rows = read_history()
    pids = process_alive()
    problems = []
    log_text = LOG.read_text(encoding="utf-8", errors="replace") if LOG.exists() else ""
    for pattern in BAD_PATTERNS:
        if pattern in log_text:
            problems.append(f"log contains {pattern!r}")
    out = [f"time={datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
           f"pids={pids or 'none'} log={LOG.relative_to(ROOT)}"]
    if rows:
        last = rows[-1]
        nonfinite = []
        for key, value in last.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number != number or abs(number) == float("inf"):
                nonfinite.append(key)
        if nonfinite:
            problems.append(f"nonfinite history columns: {nonfinite}")
        best = min(rows, key=lambda r: float(r["balanced_score"]))
        out.append("last epoch: " + " ".join(
            f"{key}={last.get(key)}" for key in WATCH_COLUMNS if key in last))
        out.append(f"best so far: epoch={best['epoch']} balanced_score={best['balanced_score']}")
        times = [float(r["epoch_time_s"]) for r in rows]
        vrams = [float(r["peak_vram_mb"]) for r in rows]
        out.append(f"progress: epochs={len(rows)} mean_epoch_time={sum(times)/len(times):.1f}s "
                   f"peak_vram={max(vrams):.0f}MiB")
    else:
        out.append("history: no rows yet")
    out.append(f"history: exists={HISTORY.exists()} "
               f"mtime={datetime.datetime.fromtimestamp(HISTORY.stat().st_mtime):%H:%M:%S}"
               if HISTORY.exists() else "history: missing")
    for name in ("checkpoint_latest.pth", "checkpoint_best.pth", "config_used.yaml",
                 "zproj_align.json"):
        path = SAVE_DIR / name
        out.append(f"artifact {name}: " + (
            f"{path.stat().st_size} bytes, mtime="
            f"{datetime.datetime.fromtimestamp(path.stat().st_mtime):%H:%M:%S}"
            if path.exists() else "absent"))
    out.append("log tail:")
    out.extend(f"  {line}" for line in log_tail())
    if problems:
        out.append("PROBLEMS: " + "; ".join(problems))
    finished = END_MARKER in log_text
    out.append(f"finished={finished}")
    return "\n".join(out), not problems, ("finished" if finished else
                                          ("no_process" if not pids else "running"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-epoch", type=int, default=None)
    parser.add_argument("--wait-finish", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--poll", type=int, default=60)
    args = parser.parse_args()

    deadline = time.time() + args.timeout
    target = args.wait_epoch
    while True:
        rows = read_history()
        reached = target is not None and len(rows) >= target
        text, healthy, state = snapshot()
        finished = state == "finished"
        if target is None and not args.wait_finish:
            print(text)
            return
        if reached or finished:
            print(text)
            sys.exit(0 if healthy else 4)
        if state == "no_process":
            print(text)
            print(f"training process exited before reaching the target (state={state})")
            sys.exit(2)
        if not healthy:
            print(text)
            sys.exit(4)
        if time.time() > deadline:
            print(text)
            print(f"wait timeout after {args.timeout}s (state={state})")
            sys.exit(5)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
