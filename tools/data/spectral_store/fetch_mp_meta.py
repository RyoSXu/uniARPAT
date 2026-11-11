"""Fetch Materials Project metadata for spectral-store-v1 (P1a; design 2026-11-10 P0).

Pages of three REST endpoints are saved verbatim (gzip JSON) so the build is
reproducible offline:

- materials/summary: material_id, structure, nsites, deprecated, builder_meta
  (license, database_version), formula_pretty;
- materials/electronic_structure: material_id and dos (canonical DOS task_id
  per material, band gap summary);
- materials/phonon: identifier, phonon_method, sum_rules_breaking;
- materials/core: material_id, task_ids (phonon identifiers are task ids, not
  material ids, so this list maps phonon and DOS tasks to materials);
- materials/core (core_calc_types): material_id, calc_types (task id -> calc type,
  e.g. "GGA NSCF Uniform" or "GGA Static"; used for the eDOS fallback-task audit).

Spectra themselves come from the local public Delta tables, not from the API.
The API key is read from MP_API_KEY and never written anywhere. Resume-safe:
existing complete pages are skipped. Usage:

    python tools/data/spectral_store/fetch_mp_meta.py --out data/spectral_store_v1/raw_mp
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = 'https://api.materialsproject.org'
ENDPOINTS = {
    'summary': ('/materials/summary/', 'material_id,structure,nsites,deprecated,builder_meta,formula_pretty'),
    'electronic_structure': ('/materials/electronic_structure/', 'material_id,dos'),
    'phonon': ('/materials/phonon/', 'identifier,phonon_method,sum_rules_breaking'),
    'core': ('/materials/core/', 'material_id,task_ids,deprecated_tasks'),
    'core_calc_types': ('/materials/core/', 'material_id,calc_types'),
}
LIMIT = 1000


def get(session, path, params, retries=12):
    for attempt in range(retries):
        try:
            response = session.get(BASE + path, params=params, timeout=300)
            if response.status_code == 200:
                return response.json()
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(10 * (attempt + 1), 120))
                continue
            raise RuntimeError(f'{path}: HTTP {response.status_code}: {response.text[:300]}')
        except requests.RequestException:
            time.sleep(min(10 * (attempt + 1), 120))
    raise RuntimeError(f'{path}: failed after {retries} attempts')


def fetch(name, out, session):
    path, fields = ENDPOINTS[name]
    directory = out / name
    directory.mkdir(parents=True, exist_ok=True)
    first = get(session, path, {'_fields': fields, '_limit': 1, '_skip': 0})
    total = int(first['meta']['total_doc'])
    pages = (total + LIMIT - 1) // LIMIT
    for page in range(pages):
        target = directory / f'page_{page:05d}.json.gz'
        if target.exists():
            continue
        data = get(session, path, {'_fields': fields, '_limit': LIMIT, '_skip': page * LIMIT})
        partial = target.with_suffix('.partial')
        with gzip.open(partial, 'wt') as stream:
            json.dump(data, stream)
        partial.rename(target)
        print(f'{name} page {page + 1}/{pages}', flush=True)
    return {'total_doc': total, 'pages': pages, 'api_version': first['meta'].get('api_version'),
            'fields': fields, 'limit': LIMIT}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--endpoint', action='append', choices=sorted(ENDPOINTS))
    args = parser.parse_args()
    key = os.environ.get('MP_API_KEY')
    if not key:
        raise SystemExit('MP_API_KEY is not set')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers['X-API-KEY'] = key
    meta_path = out / 'fetch_meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for name in args.endpoint or ['phonon', 'electronic_structure', 'summary', 'core']:
        info = fetch(name, out, session)
        info['finished_at'] = datetime.now(timezone.utc).isoformat()
        meta[name] = info
        meta_path.write_text(json.dumps(meta, indent=2) + '\n')
    print('done', json.dumps(meta))


if __name__ == '__main__':
    main()
