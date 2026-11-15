"""Freeze the v2 train normalizers of a v2 view (no guard reference yet); never launches training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import file_sha256, input_path
from utils.spectral_balance import load_balance_protocol, prepare_balance_protocol_v2

SOURCES = ('utils/spectral_balance.py', 'utils/partial_labels.py', 'utils/dataset.py', 'model/model.py',
           'tools/train.py', 'tools/eval/spectral_balance_prepare_v2.py')


def prepare(data_dir, out, valid_split):
    data_dir, out = Path(data_dir), Path(out)
    if out.exists():
        raise FileExistsError(f"use a new output directory: {out}")
    protocol = prepare_balance_protocol_v2(data_dir, valid_split)
    hashes = {str(data_dir / 'manifest.json'): protocol['data_manifest_sha256'],
              **{str(data_dir / name): digest for name, digest in protocol['train_sha256'].items()},
              **{str(ROOT / name): file_sha256(ROOT / name) for name in SOURCES}}
    if any(file_sha256(path) != digest for path, digest in hashes.items()):
        raise RuntimeError('an input changed while preparing the protocol')
    out.mkdir(parents=True)
    path = out / 'protocol.json'
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if load_balance_protocol(path, data_dir) != protocol:
        raise RuntimeError('written protocol does not reload against its view')
    report = {'schema': 'joint-spectral-balance-preparation-v2', 'training_started': False,
              'data_dir': str(data_dir.resolve()), 'valid_split': valid_split,
              'normalizers_fit_on': 'effective train rows carrying each label',
              'fit_samples': protocol['fit_samples'],
              'C': {task: protocol['tasks'][task]['energy'] for task in protocol['tasks']},
              'reference': protocol['reference'], 'input_sha256': hashes}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, help='v2 view, e.g. data/spectral_store_v1/views/s1_v2')
    parser.add_argument('--valid-split', required=True, help='dual-labelled selection split (valid or valid_q1)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = prepare(input_path(args.data_dir), input_path(args.out), args.valid_split)
    print(json.dumps({k: report[k] for k in ('fit_samples', 'C', 'reference')}, indent=1))


if __name__ == '__main__':
    main()
