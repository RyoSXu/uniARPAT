"""Run Mat2Spec arms one after another, detached from any agent session.

  setsid nohup python baselines/mat2spec/adapter/run_queue.py --root <root> --arms a,b,c \
      > <root>/execution/queue.out 2>&1 &

Each arm is one ``train.py`` process writing ``<root>/<arm>/execution/status.json``. The queue
  * refuses to start while another queue for the same root is alive (pid lock);
  * skips arms whose status is ``completed``; resumes arms that already have a fingerprint;
  * stops at the first arm that exits non-zero (non-finite, OOM, contract) and never retries or
    changes parameters;
  * records progress in ``<root>/execution/queue_status.json`` (state: running / completed / stopped).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = {"sumkl": ("normalized_sum", "KL"), "stdmae": ("standardized", "MAE"),
        "sumwd": ("normalized_sum", "WD"), "maxmse": ("normalized_max", "MSE")}
EXIT_MEANING = {3: "non-finite loss or prediction", 4: "CUDA out of memory", 5: "contract mismatch"}


def write_json(path, payload):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def alive(pid):
    try:
        os.kill(pid, 0)
    except (OSError, TypeError):
        return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--arms", required=True, help="comma list of <task>_<arm>, e.g. edos_sumkl,phdos_stdmae")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target_unit", default="per_atom", choices=("per_atom", "cell"))
    parser.add_argument("--precheck_limit_train", type=int, default=0,
                        help="precheck only: train on the first N crystals (such runs cannot be exported)")
    args = parser.parse_args(argv)
    root = Path(args.root)
    arms = args.arms.split(",")
    for arm in arms:
        task, _, kind = arm.partition("_")
        if task not in ("edos", "phdos") or kind not in ARMS:
            parser.error(f"unknown arm {arm}")
    execution = root / "execution"
    execution.mkdir(parents=True, exist_ok=True)
    lock = execution / "queue.pid"
    if lock.exists() and alive(int(lock.read_text().strip() or 0)):
        print(f"another queue is running for {root} (pid {lock.read_text().strip()})", file=sys.stderr)
        return 2
    lock.write_text(str(os.getpid()))
    status = {"state": "running", "pid": os.getpid(), "arms": {arm: "pending" for arm in arms},
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "queue_args": vars(args)}
    status_path = execution / "queue_status.json"

    def save(**fields):
        status.update(fields, heartbeat=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json(status_path, status)

    save()
    for arm in arms:
        task, _, kind = arm.partition("_")
        run_dir = root / arm
        arm_status = run_dir / "execution/status.json"
        if arm_status.exists() and json.loads(arm_status.read_text()).get("state") == "completed":
            # A finished arm only counts if it was trained with the recipe this queue asks for.
            wanted = {"task": task, "label_scaling": ARMS[kind][0], "loss": ARMS[kind][1], "epochs": args.epochs,
                      "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
                      "target_unit": args.target_unit, "limit_train": args.precheck_limit_train, "limit_valid": 0}
            if json.loads((run_dir / "fingerprint.json").read_text())["config"] != wanted:
                status["arms"][arm] = "failed(exit 5: completed with a different recipe)"
                save(state="stopped", current=None)
                print(f"[queue] {arm} was completed with a different recipe; queue stopped", flush=True)
                return 5
            status["arms"][arm] = "completed"
            save()
            continue
        command = [sys.executable, str(HERE / "train.py"), "--task", task, "--label_scaling", ARMS[kind][0],
                   "--loss", ARMS[kind][1], "--run_dir", str(run_dir), "--graph_cache", str(root / "graphs"),
                   "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--lr", str(args.lr),
                   "--seed", str(args.seed), "--target_unit", args.target_unit]
        if args.precheck_limit_train:
            command += ["--limit_train", str(args.precheck_limit_train)]
        if (run_dir / "fingerprint.json").exists():
            command.append("--resume")
        status["arms"][arm] = "running"
        save(current=arm, command=command)
        print(f"[queue] {time.strftime('%H:%M:%S')} start {arm}: {' '.join(command)}", flush=True)
        code = subprocess.call(command)
        if code:
            status["arms"][arm] = f"failed(exit {code}: {EXIT_MEANING.get(code, 'see log')})"
            save(state="stopped", current=None)
            print(f"[queue] {arm} failed with exit {code}; queue stopped, nothing retried", flush=True)
            return code
        status["arms"][arm] = "completed"
        save(current=None)
        print(f"[queue] {time.strftime('%H:%M:%S')} finished {arm}", flush=True)
    save(state="completed")
    print("[queue] all arms completed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
