"""Fetch JARVIS-DFT phonon DOS pages for spectral-store-v1 (design 2026-11-11 step 1d).

Candidates: raw_jarvis/candidate_jids.json -- JVASP ids of dft_3d (12-12-2022) whose
reduced formula equals that of a legacy phDOS label without an MP source; ids with phonon
modes first, then the rest. Each page is read with jarvis.db.webpages.Webpage (the path of
the legacy pipeline) and the phonon DOS fields are kept verbatim (frequencies, intensity,
phonon modes, minimum finite-difference mode); band-structure arrays are not kept.
Anonymous HTTPS only.

Output: raw_jarvis/pages/part-XXXX.jsonl.gz ({jid, fetched, fields | null, error}),
resume-safe by jid. State and heartbeat: execution/spectral_store_jarvis/state.json.

    setsid python tools/data/spectral_store/fetch_jarvis.py --workers 4 &
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from common import ROOT, STORE

RAW = STORE / 'raw_jarvis'
PAGES = RAW / 'pages'
STATE_DIR = ROOT / 'execution/spectral_store_jarvis'
KEEP = ('phonon_dos_frequencies', 'phonon_dos_intensity', 'phonon_modes', 'min_fd_phonon_mode')
SHARD = 200


def now():
    return datetime.now(timezone.utc).isoformat()


def fetch(jid, retries=5):
    from jarvis.db.webpages import Webpage
    error = None
    for attempt in range(retries):
        try:
            data = Webpage(jid=jid).get_dft_phonon_dos()
            fields = {k: data.get(k) for k in KEEP} if isinstance(data, dict) else None
            if fields and fields.get('phonon_dos_frequencies'):
                return {'jid': jid, 'fetched': now(), 'fields': fields, 'error': None}
            return {'jid': jid, 'fetched': now(), 'fields': None, 'error': 'no phonon dos on page'}
        except Exception as exc:  # recorded and retried; never dropped
            error = f'{type(exc).__name__}: {str(exc)[:120]}'
            time.sleep(min(20 * (attempt + 1), 120))
    return {'jid': jid, 'fetched': now(), 'fields': None, 'error': f'failed: {error}'}


def done_ids():
    out = {}
    for path in sorted(PAGES.glob('part-*.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                r = json.loads(line)
                out[r['jid']] = r['error']
    return out


def write_state(state):
    state['heartbeat'] = now()
    tmp = STATE_DIR / 'state.json.tmp'
    tmp.write_text(json.dumps(state, indent=1) + '\n')
    tmp.replace(STATE_DIR / 'state.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    PAGES.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    candidates = json.loads((RAW / 'candidate_jids.json').read_text())
    order = candidates['with_modes'] + [j for j in candidates['any'] if j not in set(candidates['with_modes'])]
    previous = done_ids()
    # retry only transport failures; "no phonon dos" is a final answer
    todo = [j for j in order if j not in previous or (previous[j] or '').startswith('failed')]
    state = {'task': 'jarvis_fetch', 'pid': os.getpid(), 'started': now(), 'status': 'running',
             'total': len(order), 'with_modes': len(candidates['with_modes']), 'todo': len(todo)}
    write_state(state)
    shard = len(list(PAGES.glob('part-*.jsonl.gz')))
    buffer, counts = [], {'with_dos': 0, 'no_dos': 0, 'failed': 0}

    def flush():
        nonlocal shard, buffer
        if buffer:
            path = PAGES / f'part-{shard:04d}.jsonl.gz'
            with gzip.open(path.with_suffix('.partial'), 'wt') as stream:
                stream.writelines(json.dumps(r) + '\n' for r in buffer)
            path.with_suffix('.partial').rename(path)
            shard += 1
            buffer = []

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch, j) for j in todo]
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            buffer.append(r)
            key = 'with_dos' if r['fields'] else ('failed' if (r['error'] or '').startswith('failed') else 'no_dos')
            counts[key] += 1
            if len(buffer) >= SHARD:
                flush()
            if i % 25 == 0 or i == len(todo):
                state.update({'processed': i, **counts})
                write_state(state)
                print(f'{now()} {i}/{len(todo)} {counts}', flush=True)
    flush()
    state.update({'status': 'finished' if counts['failed'] == 0 else 'finished_with_failures', 'finished': now(),
                  **counts})
    write_state(state)
    print('done', json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
