"""Disposable real-batch checks for the phDOS head depth arms; never trains.

Per arm: production assembly, every non-head initial tensor equal to the depth-6
reference built with the same seed, the declared head depth and parameter count,
the contract record, timed disposable updates (cold and warm cache) and an lr=0
objective/evaluation check. No checkpoint is written; test is never loaded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
from utils.zproj_alignment import state_hash

HEAD = 'phdos_out_head.'


def body(state):
    return {key: value for key, value in state.items() if not key.startswith(HEAD)}


def reference_body(arm, control):
    cfg = replace(ExperimentConfig(**arm['configuration']), phdos_head_layers=None,
                  tag=arm['configuration']['tag'] + '_reference')
    _, model, _, initialization, _ = build(cfg)
    if initialization['state_sha256'] != control['initialization']['state_sha256']:
        raise ValueError('depth-6 reference no longer reproduces the adopted R42 initial tensors')
    return state_hash(body(model.model['transformer'].state_dict()))


def check_arm(arm, control):
    cfg = ExperimentConfig(**arm['configuration'])
    validate_balance_config(cfg)
    expected_body = reference_body(arm, control)
    builder, model, params, initialization, _ = build(cfg)
    transformer = model.model['transformer']
    state = transformer.state_dict()
    if state_hash(body(state)) != expected_body:
        raise ValueError(f"{arm['name']}: non-head initial tensors differ from the R42 initial state")
    layers = sum(1 for key in state if key.startswith(HEAD) and key.endswith('.weight'))
    if layers != arm['phdos_head_layers'] or params.get('phdos_head_layers') != arm['phdos_head_layers']:
        raise ValueError(f"{arm['name']}: constructed head depth differs from the arm")
    contract = comparison_contract(cfg, params)
    if contract.get('phdos_head', {}).get('layers') != arm['phdos_head_layers']:
        raise ValueError(f"{arm['name']}: contract does not record the head depth")
    loader = builder.get_dataloader(split='train', batch_size=cfg.batch_size, dos_minmax=True, dos_sumnorm=True)
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
        raise ValueError(f"{arm['name']}: training and evaluation objectives differ")
    model.to(torch.device('cpu'))
    del model
    torch.cuda.empty_cache()
    return {'name': arm['name'], 'configuration': asdict(cfg), 'phdos_head': contract['phdos_head'],
            'initialization': initialization, 'non_head_state_sha256': expected_body,
            'parameters': sum(v.numel() for v in state.values() if v.is_floating_point()),
            'head_parameters': sum(v.numel() for k, v in state.items() if k.startswith(HEAD)),
            's_per_batch_cold_warm': timings, 'batches_per_epoch': len(loader),
            'estimated_train_seconds_per_epoch_warm': timings[-1] * len(loader),
            'historical_full_train_epoch_seconds': HISTORICAL_EPOCH_SECONDS, 'peak_allocated_mib': peak,
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
    results = [check_arm(arm, saved['completed_control']) for arm in saved['arms']]
    report = {'schema': 'lr2-es-phdos-head-preflight-v1', 'passed': True,
              'preparation_sha256': file_sha256(preparation), 'gpu': torch.cuda.get_device_name(),
              'arms': results, 'full_training_started': False,
              'scope': f'{TIMED_BATCHES} timed disposable updates (cold and warm cache) and an lr=0 objective '
                       'check per arm; no checkpoint or test'}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = check(input_path(args.preparation), input_path(args.out))
    for arm in report['arms']:
        print(arm['name'], arm['head_parameters'], arm['parameters'],
              [round(s, 3) for s in arm['s_per_batch_cold_warm']], round(arm['peak_allocated_mib']))


if __name__ == '__main__':
    main()
