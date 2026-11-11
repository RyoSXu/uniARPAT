"""Local second copy of spectral-store-v1 with a SHA-256 manifest (design 2026-11-11 step 1a).

Copies the core tables (materials, geometries, spectra_*, sealed/, legacy_splits/, views/,
raw metadata pages and finalize report) to a backup directory on another disk and writes
backup_manifest.json (path, bytes, sha256). Incremental: files whose size and mtime match
an existing manifest entry are not copied again; every file is re-hashed at the target.
Nothing is uploaded anywhere.

    python tools/data/spectral_store/backup_store.py --dest /root/spectral_store_backup/v1-20261111
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from common import STORE, file_sha256

INCLUDE = ['materials.parquet', 'geometries.parquet', 'spectra_edos.parquet', 'spectra_phdos.parquet',
           'spectra_legacy.parquet', 'finalize_report.json', 'sealed', 'legacy_splits', 'views', 'raw_mp',
           'raw_mp_core', 'raw_phonondb', 'raw_jarvis', 'verify']
VERSION = 'spectral-store-v1.0'


def files(root: Path):
    for name in INCLUDE:
        path = root / name
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from sorted(p for p in path.rglob('*') if p.is_file() and not p.name.endswith('.partial'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dest', required=True)
    parser.add_argument('--version', default=VERSION)
    args = parser.parse_args()
    dest = Path(args.dest)
    if dest.resolve().is_relative_to(STORE.resolve()):
        raise SystemExit('backup must be outside the store')
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / 'backup_manifest.json'
    old = json.loads(manifest_path.read_text())['files'] if manifest_path.exists() else {}
    entries, copied = {}, 0
    for source in files(STORE):
        rel = str(source.relative_to(STORE))
        stat = source.stat()
        target = dest / rel
        prev = old.get(rel)
        if not (prev and target.exists() and prev['bytes'] == stat.st_size and prev['mtime'] == stat.st_mtime):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
            prev = None
        digest = prev['sha256'] if prev else file_sha256(target)
        if not prev and digest != file_sha256(source):
            raise SystemExit(f'hash mismatch after copy: {rel}')
        entries[rel] = {'bytes': stat.st_size, 'mtime': stat.st_mtime, 'sha256': digest}
    manifest = {'store_version': args.version, 'source': str(STORE), 'updated': datetime.now(timezone.utc).isoformat(),
                'files': entries, 'total_bytes': sum(e['bytes'] for e in entries.values())}
    manifest_path.write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({'files': len(entries), 'copied': copied, 'total_GB': round(manifest['total_bytes'] / 1e9, 2)}))


if __name__ == '__main__':
    main()
