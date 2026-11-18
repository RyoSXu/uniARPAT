"""Bounded train-only cohort census and disposable comparison of two compute budgets."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from utils.crystal_cache import PackedCrystalStore
from utils.periodic_geometry import build_periodic_records
from utils.paths import file_sha256

OUT = ROOT / 'results/runs/dos_v1_h2_e100_throughput_20261116'
DATA = ROOT / 'data/datasets/dos_v1/views/dos_v1'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def census():
    torch.set_num_threads(1)
    store = PackedCrystalStore(DATA, 'train')
    atoms = store.n_atoms.numpy()
    density = atoms / np.abs(np.linalg.det(store.cell.numpy()))
    rng = np.random.default_rng(20261116)
    candidates = np.unique(np.concatenate((np.argsort(density)[-64:],
        rng.choice(len(atoms), 64, replace=False), np.argsort(atoms)[-32:])))
    rows = []
    for i in candidates:
        _, geometry = store[int(i)]
        from utils.crystal import CrystalGeometry
        records = build_periodic_records(
            CrystalGeometry(geometry.cell[None], geometry.frac_coords[None]),
            torch.zeros((1, int(atoms[i])), dtype=torch.bool), 6.)
        degree = torch.bincount(records.dst, minlength=int(atoms[i]))
        pairs = int((degree * (degree-1) // 2).sum())
        rows.append(dict(row=int(i), material_id=str(store.material_ids[i]), atoms=int(atoms[i]),
            degree_max=int(degree.max()), angular_pairs=pairs, pairs_per_atom=pairs/int(atoms[i])))
    ranked = sorted(rows, key=lambda row: row['pairs_per_atom'], reverse=True)
    dense = [row['row'] for row in ranked[:4]]
    # Dense crystals embedded in ordinary batch32, rather than a synthetic batch
    # of 32 identical worst-density cells which does not describe the sampler.
    ordinary = [int(i) for i in rng.permutation(len(atoms)) if int(i) not in dense][:31]
    cohorts = dict(train_manifest_sha256=file_sha256(DATA / 'manifest.json'),
        scope='bounded census: top64 number density + uniform64 + largest32; no valid/test',
        candidate_rows=rows, stress_batches=[
            dict(name='densest_one_plus_31_ordinary', rows=dense[:1]+ordinary),
            dict(name='densest_four_plus_28_ordinary', rows=dense+ordinary[:28])])
    write(OUT / 'cohorts.json', cohorts)
    print(f'census complete: {len(rows)} candidates, densest={ranked[0]}', flush=True)


def main():
    if OUT.exists():
        raise FileExistsError(f'fresh throughput directory required: {OUT}')
    OUT.mkdir(parents=True)
    census()
    results = []
    for budget in (51200, 76800):
        prep = OUT / f'pb{budget}'
        command = [sys.executable, str(ROOT / 'tools/eval/dos_baseline_prepare.py'),
            '--out', str(prep), '--tag', f'_sbal_dosv1h2e100pb{budget}_20261116_s42',
            '--pair-budget', str(budget), '--epochs', '100', '--cohorts', str(OUT / 'cohorts.json')]
        with (OUT / f'pb{budget}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write(OUT / 'decision.json', dict(status='stopped_error', failed_budget=budget,
                returncode=result.returncode, automatic_retry=False))
            raise RuntimeError(f'preflight failed; stop without retry: {prep}')
        results.append(json.loads((prep / 'preflight.json').read_text()))
        print(f"pb{budget}: {results[-1]['warm_materials_per_second']:.2f} materials/s, "
              f"peak {results[-1]['peak_allocated_mib']/1024:.2f} GiB", flush=True)
    ratio = results[1]['warm_materials_per_second'] / results[0]['warm_materials_per_second']
    # A short timing comparison is a resource choice, not evidence of model quality.
    choose_large = ratio >= 1.05 and results[1]['peak_allocated_mib'] <= 21*1024
    chosen = 76800 if choose_large else 51200
    write(OUT / 'decision.json', dict(status='complete', selected_budget=chosen,
        selected_preparation=str(OUT / f'pb{chosen}'), throughput_ratio_76800_over_51200=ratio,
        rule='select76800 only if >=5% faster and peak <=21 GiB; otherwise retain51200',
        full_training_started=False, results=results))
    print(f'selected pb{chosen}, ratio={ratio:.4f}', flush=True)


if __name__ == '__main__':
    main()
