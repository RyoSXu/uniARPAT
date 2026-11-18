"""Single-owner detached supervisor for a prepared dos_v1 baseline.

Launch only in the authorized execution chat. No automatic retries or recipe changes.
An interrupted process may be resumed explicitly with --resume under the same contract.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.paths import file_sha256

BAD = re.compile(r'out of memory|CUDA error|Traceback \(most recent call last\)|'
                 r'non[- ]finite|(?:Loss:|eDOS:|phDOS:)\s*[-+]?(?:nan|inf)\b', re.I)


def supervise(out, resume=False):
    OUT = Path(out).resolve()
    execution = OUT / 'execution'
    execution.mkdir(exist_ok=True)
    with (execution / 'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = json.loads((OUT / 'report.json').read_text())
        epochs = report['configuration']['epochs']
        check = json.loads((OUT / 'preflight.json').read_text())
        if not check['passed'] or check['preparation_sha256'] != file_sha256(OUT / 'report.json'):
            raise ValueError('preflight does not certify this preparation')
        if any(file_sha256(p) != h for p, h in report['input_sha256'].items()):
            raise ValueError('prepared inputs changed')
        run = Path(report['run_dir'])
        latest = run / 'checkpoint_latest.pth'
        if resume != latest.exists():
            raise ValueError('--resume must be explicit exactly when latest exists')
        if not resume and (run.exists() or Path(report['history']).exists()):
            raise FileExistsError('fresh run has existing artifacts')
        stamp = str(time.time_ns())
        log_path = execution / f'train_{stamp}.log'
        state = dict(status='starting', supervisor_pid=os.getpid(), log=str(log_path),
                     started_unix=time.time(), resume=resume)

        def save(**fields):
            state.update(fields, heartbeat_unix=time.time())
            temp = execution / 'state.tmp'
            temp.write_text(json.dumps(state, indent=2) + '\n')
            temp.replace(execution / 'state.json')

        with log_path.open('w') as log:
            env = dict(os.environ)
            env.update(report['environment'], PYTHONUNBUFFERED='1')
            process = subprocess.Popen(report['command'], cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, env=env, start_new_session=True)
            save(status='running', training_pid=process.pid)

            def stop(signum, frame):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                save(status='interrupted', reason=f'supervisor signal {signum}')
                raise SystemExit(128 + signum)

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            try:
                with log_path.open() as reader:
                    while True:
                        # Scan only this launch's new bytes; never scan historical logs.
                        new = reader.read()
                        bad = BAD.search(new)
                        rows = []
                        history = Path(report['history'])
                        if history.exists():
                            with history.open() as stream:
                                rows = list(csv.DictReader(stream))
                            for row in rows:
                                for key in ('balanced_score', 'train_loss',
                                            'blind_energy_edos_mean', 'blind_energy_phdos_mean'):
                                    if key in row and not math.isfinite(float(row[key])):
                                        raise ValueError(f'non-finite history {key}')
                        if bad:
                            raise RuntimeError(f'anomaly in new log: {bad.group()}')
                        code = process.poll()
                        if code is not None:
                            # One more pass consumes trailing output after process exit.
                            tail = reader.read()
                            if BAD.search(tail) or code != 0:
                                raise RuntimeError(f'training exited {code}: {tail[-1000:]}')
                            if not rows or int(rows[-1]['epoch']) != epochs:
                                raise RuntimeError(f'process ended without {epochs} completed epochs')
                            if not latest.exists() or not (run / 'checkpoint_best.pth').exists():
                                raise RuntimeError('completed run lacks checkpoints')
                            save(status='training_complete', completed_epochs=epochs, returncode=code)
                            return
                        save(completed_epochs=int(rows[-1]['epoch']) if rows else 0)
                        time.sleep(10)
            except Exception as error:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                save(status='stopped_error', reason=str(error))
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    supervise(args.preparation_dir, args.resume)
