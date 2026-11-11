"""Download PhononDB (NIMS MDR) archives for spectral-store-v1 (design 2026-11-11 step 1c).

Index: atztogo/phonondb mdr/phonondb/README.md, parsed to raw_phonondb/index.json
(Materials Project integer id, formula, MDR serial, URL). Archives listed in
raw_phonondb/priority_serials.json (same reduced formula as a legacy phDOS label without an
MP source) are downloaded first, then all remaining ones; every archive is kept, including
those duplicating MP phonons. Anonymous HTTPS downloads only.

Resume-safe: a finished archive is raw_phonondb/zips/<serial>.zip (written via .partial and
checked to be a valid zip). Failures are retried later and recorded, never dropped. State
and heartbeat: execution/spectral_store_phonondb/state.json; log next to it.

    setsid python tools/data/spectral_store/fetch_phonondb.py --workers 4 &
"""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

from common import ROOT, STORE

RAW = STORE / 'raw_phonondb'
ZIPS = RAW / 'zips'
STATE_DIR = ROOT / 'execution/spectral_store_phonondb'


def now():
    return datetime.now(timezone.utc).isoformat()


def fetch(session, serial, url, retries=6):
    target = ZIPS / f'{serial}.zip'
    if target.exists():
        return serial, 'exists', None
    partial = target.with_suffix('.partial')
    error = None
    for attempt in range(retries):
        try:
            with session.get(url, timeout=600, stream=True) as response:
                if response.status_code != 200:
                    error = f'HTTP {response.status_code}'
                    time.sleep(min(30 * (attempt + 1), 300))
                    continue
                with open(partial, 'wb') as stream:
                    for block in response.iter_content(1 << 16):
                        stream.write(block)
            with zipfile.ZipFile(partial) as archive:
                if 'phonopy_params.yaml.xz' not in archive.namelist():
                    error = 'no phonopy_params.yaml.xz'
                    break
            partial.rename(target)
            return serial, 'downloaded', None
        except (requests.RequestException, zipfile.BadZipFile, OSError) as exc:
            error = f'{type(exc).__name__}: {str(exc)[:120]}'
            time.sleep(min(30 * (attempt + 1), 300))
    partial.unlink(missing_ok=True)
    return serial, 'failed', error


def write_state(state):
    state['heartbeat'] = now()
    tmp = STATE_DIR / 'state.json.tmp'
    tmp.write_text(json.dumps(state, indent=1) + '\n')
    tmp.replace(STATE_DIR / 'state.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--passes', type=int, default=3, help='full passes over still-missing archives')
    args = parser.parse_args()
    ZIPS.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    index = {r['serial']: r['url'] for r in json.loads((RAW / 'index.json').read_text())}
    priority = [s for s in json.loads((RAW / 'priority_serials.json').read_text()) if s in index]
    order = priority + sorted(set(index) - set(priority))
    state = {'task': 'phonondb_download', 'pid': __import__('os').getpid(), 'started': now(), 'status': 'running',
             'total': len(order), 'priority': len(priority), 'failed': {}}
    session = requests.Session()
    session.headers['User-Agent'] = 'spectral-store-fetch/1.0'
    for npass in range(args.passes):
        todo = [s for s in order if not (ZIPS / f'{s}.zip').exists()]
        state.update({'pass': npass + 1, 'todo_at_pass_start': len(todo)})
        write_state(state)
        if not todo:
            break
        done = 0
        with ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(fetch, session, s, index[s]) for s in todo]
            for future in as_completed(futures):
                serial, status, error = future.result()
                done += 1
                if status == 'failed':
                    state['failed'][serial] = error
                else:
                    state['failed'].pop(serial, None)
                if done % 25 == 0 or done == len(todo):
                    have = sum((ZIPS / f'{s}.zip').exists() for s in order)
                    state.update({'downloaded': have, 'priority_downloaded':
                                  sum((ZIPS / f'{s}.zip').exists() for s in priority)})
                    write_state(state)
                    print(f'{now()} pass {npass + 1} {done}/{len(todo)} have {have}/{len(order)} '
                          f'failed {len(state["failed"])}', flush=True)
    have = sum((ZIPS / f'{s}.zip').exists() for s in order)
    state.update({'downloaded': have, 'status': 'finished' if have == len(order) else 'finished_with_failures',
                  'finished': now()})
    write_state(state)
    print('done', json.dumps({k: state[k] for k in ('downloaded', 'total', 'status')}), flush=True)


if __name__ == '__main__':
    main()
