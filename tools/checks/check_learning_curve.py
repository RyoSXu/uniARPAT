"""Disposable real-batch checks for the LR2 E+S learning-curve points; never trains.

Per point: production assembly, seed-42 initial tensors equal to the adopted
control, subset rows (whole groups, nested, repeated to the full-train epoch size),
valid view unchanged, contract record, two disposable updates, an lr=0
objective/evaluation check, and timed warm-cache batches for the epoch cost.
No checkpoint is written; test is never loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.checks.check_structure_generalization import HISTORICAL_EPOCH_SECONDS, TIMED_BATCHES, build, timed
from tools.train import configure_local_global_execution, evaluate_local_global_split
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import comparison_contract
from utils.paths import file_sha256, input_path
from utils.spectral_balance import ENERGY_METRIC
from utils.spectral_balance_experiment import validate_balance_config
from utils.train_subset import train_view

FULL_TRAIN = 18704
VALID = 2313


def check_point(point, control, view):
    cfg = ExperimentConfig(**point['configuration'])
    validate_balance_config(cfg)
    builder, model, params, initialization, _ = build(cfg)
    transformer = model.model['transformer']
    contract = comparison_contract(cfg, params)
    record = contract.get('train_subset', {})
    if record.get('fraction') != point['fraction'] or record.get('file_sha256') != file_sha256(point['subset_path']):
        raise ValueError(f"{point['name']}: contract does not bind its subset file")
    if initialization['state_sha256'] != control['initialization']['state_sha256']:
        raise ValueError(f"{point['name']}: seed-42 initial tensors differ from the adopted control")
    loader = builder.get_dataloader(split='train', batch_size=cfg.batch_size, dos_minmax=True, dos_sumnorm=True)
    dataset = loader.dataset
    ids = [str(m) for m in dataset.material_ids]
    counts = Counter(ids)
    subset = json.loads(Path(point['subset_path']).read_text())
    if set(counts) != set(subset['material_ids']) or set(counts.values()) != {subset['repeat']}:
        raise ValueError(f"{point['name']}: training rows are not the subset repeated {subset['repeat']} times")
    if abs(len(dataset) - FULL_TRAIN) > 0.01 * FULL_TRAIN:
        raise ValueError(f"{point['name']}: epoch size {len(dataset)} departs from the full-train step count")
    # Whole groups only: every composition of the subset keeps all its train-view members.
    member = set(subset['material_ids'])
    inside, outside = set(), set()
    for material, key in zip(*view):
        (inside if material in member else outside).add(key)
    if inside & outside:
        raise ValueError(f"{point['name']}: a composition group straddles the subset boundary")
    valid = builder.get_dataloader(split='valid', batch_size=cfg.batch_size, dos_minmax=True, dos_sumnorm=True)
    if len(valid.dataset) != VALID or valid.dataset.train_subset is not None:
        raise ValueError('valid view must stay complete')
    model.to(torch.device('cuda'))
    configure_local_global_execution(transformer)
    loader.sampler.set_epoch(0)
    batches = [next(iter(loader))]
    torch.cuda.reset_peak_memory_stats()
    timings = timed(model, loader, cfg, TIMED_BATCHES, passes=2)
    peak = torch.cuda.max_memory_allocated() / 1024**2
    transformer.eval()
    for group in model.optimizer['transformer'].param_groups:
        group['lr'] = 0.
    before = model.train_one_step(batches[0], 0, pair_budget=cfg.pair_budget)
    evaluated = evaluate_local_global_split(model, batches[:1], pair_budget=cfg.pair_budget)
    auxiliary = .5 * (before['loss_shape_e'] + before['loss_shape_p'])
    if not np.isclose(before['loss'] - auxiliary, evaluated[ENERGY_METRIC], rtol=3e-6, atol=1e-7):
        raise ValueError(f"{point['name']}: training and evaluation objectives differ")
    model.to(torch.device('cpu'))
    del model
    torch.cuda.empty_cache()
    return {'name': point['name'], 'configuration': asdict(cfg), 'train_subset': record,
            'epoch_materials': len(dataset), 'unique_materials': len(counts), 'groups': len(inside),
            'initialization': initialization, 's_per_batch_cold_warm': timings,
            'batches_per_epoch': len(loader),
            'estimated_train_seconds_per_epoch_warm': timings[-1] * len(loader),
            'historical_full_train_epoch_seconds': HISTORICAL_EPOCH_SECONDS,
            'peak_allocated_mib': peak,
            'loss_evaluation_difference': abs(before['loss'] - auxiliary - evaluated[ENERGY_METRIC])}


def check(preparation, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'use a new preflight output directory: {out}')
    saved = json.loads(Path(preparation).read_text())
    if any(file_sha256(p) != digest for p, digest in saved['input_sha256'].items()):
        raise ValueError('preparation inputs or source changed; prepare a new isolated trial')
    if not torch.cuda.is_available():
        raise ValueError('real-batch preflight requires CUDA')
    torch.set_num_threads(1)
    view = train_view(input_path(ExperimentConfig(**saved['points'][0]['configuration']).data_dir))
    small, large = (set(json.loads(Path(p['subset_path']).read_text())['material_ids']) for p in saved['points'])
    if not small < large:
        raise ValueError('the 25 % subset must be strictly inside the 50 % subset')
    results = [check_point(point, saved['completed_control'], view) for point in saved['points']]
    report = {'schema': 'lr2-es-learning-curve-preflight-v1', 'passed': True,
              'preparation_sha256': file_sha256(preparation), 'gpu': torch.cuda.get_device_name(),
              'nested': True, 'points': results, 'full_training_started': False,
              'scope': f'{TIMED_BATCHES} timed disposable updates (cold and warm cache) and an lr=0 objective '
                       'check per point; no checkpoint or test'}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = check(input_path(args.preparation), input_path(args.out))
    for point in report['points']:
        print(point['name'], point['unique_materials'], point['epoch_materials'],
              [round(s, 3) for s in point['s_per_batch_cold_warm']], round(point['peak_allocated_mib']))


if __name__ == '__main__':
    main()
