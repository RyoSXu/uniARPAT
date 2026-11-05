"""Disposable real-batch checks for the LR2 E+S reference controls; never trains.

Per arm: production assembly and initializer, initial-tensor identity against the
adopted control, training data order, two disposable updates (cost, finite
gradients), lr=0 objective/evaluation consistency, and for single-task arms
that the unsupervised task readout receives no gradient. No checkpoint is
written and test is never loaded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.checks.check_long_budget import captured_config
from tools.train import configure_local_global_execution, evaluate_local_global_split, setup_ablation_seed
from utils.builder import ConfigBuilder
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import comparison_contract, initialize_comparison_model
from utils.paths import file_sha256, input_path
from utils.spectral_balance import ENERGY_METRIC
from utils.spectral_balance_experiment import validate_balance_config


def order_digest(loader):
    loader.sampler.set_epoch(0)
    return hashlib.sha256(np.asarray(list(loader.sampler), dtype=np.int64).tobytes()).hexdigest()


def assembled(cfg):
    """Capture the runner assembly under a throwaway tag and remove its empty directory."""
    probe = replace(cfg, tag=cfg.tag.replace('_sbal_', '_sbal_refctlcheck_', 1))
    run = ROOT / f'results/runs/ablation_{probe.model_name.lower()}{probe.tag}'
    existed = run.exists()
    try:
        return captured_config(probe)
    finally:
        if not existed and run.is_dir() and not any(run.iterdir()):
            run.rmdir()


def check_arm(arm, control):
    cfg = ExperimentConfig(**arm['configuration'])
    validate_balance_config(cfg)
    setup_ablation_seed(cfg.seed)
    assembly = assembled(cfg)
    builder = ConfigBuilder(**assembly)
    model = builder.get_model()
    transformer = model.model['transformer']
    params = assembly['model']['params']['sub_model']['transformer']
    initialization = initialize_comparison_model(transformer, params, cfg.seed)
    contract = comparison_contract(cfg, params)
    seed42_lr2 = cfg.local_route == 'LR2' and cfg.seed == 42
    if seed42_lr2 and initialization['state_sha256'] != control['initialization']['state_sha256']:
        raise ValueError(f"{arm['name']}: seed-42 LR2 initial tensors differ from the adopted control")
    if cfg.local_route == 'LR2' and cfg.seed != 42 and initialization['state_sha256'] == control['initialization']['state_sha256']:
        raise ValueError(f"{arm['name']}: a seed replicate must not reuse the seed-42 initial tensors")
    if cfg.local_route == 'S0' and (transformer.encoder.local_layers or transformer._local_geometry_cache is not None):
        raise ValueError('S0 must own no local layers or geometry cache')
    model.to(torch.device('cuda'))
    configure_local_global_execution(transformer)
    loader = builder.get_dataloader(split='train', batch_size=cfg.batch_size, dos_minmax=True,
                                    dos_sumnorm=True, sampler_seed=cfg.data_order_seed)
    historical = builder.get_dataloader(split='train', batch_size=cfg.batch_size, dos_minmax=True, dos_sumnorm=True)
    order, historical_order = order_digest(loader), order_digest(historical)
    if (cfg.data_order_seed is None) != (order == historical_order):
        raise ValueError(f"{arm['name']}: data order does not follow its declared seed")
    loader.sampler.set_epoch(0)
    iterator = iter(loader)
    batches = [next(iterator), next(iterator)]
    transformer.train()
    timings, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for index, batch in enumerate(batches):
        torch.cuda.synchronize()
        started = time.perf_counter()
        losses.append(model.train_one_step(batch, index, pair_budget=cfg.pair_budget))
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        if not all(np.isfinite(value) for value in losses[-1].values()):
            raise ValueError(f"{arm['name']}: non-finite disposable update")
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in transformer.parameters()):
            raise ValueError(f"{arm['name']}: non-finite gradient")
    peak = torch.cuda.max_memory_allocated() / 1024**2
    transformer.eval()
    for group in model.optimizer['transformer'].param_groups:
        group['lr'] = 0.
    before = model.train_one_step(batches[0], 0, pair_budget=cfg.pair_budget)
    evaluated = evaluate_local_global_split(model, batches[:1], pair_budget=cfg.pair_budget)
    if cfg.single_task is None:
        objective = evaluated[ENERGY_METRIC]
        auxiliary = .5 * (before['loss_shape_e'] + before['loss_shape_p'])
    else:
        objective = evaluated[f'blind_energy_{cfg.single_task}_mean']
        auxiliary = before['loss_shape_e' if cfg.single_task == 'edos' else 'loss_shape_p']
    if not np.isclose(before['loss'] - auxiliary, objective, rtol=3e-6, atol=1e-7):
        raise ValueError(f"{arm['name']}: training and evaluation objectives differ")
    unsupervised = {}
    if cfg.single_task is not None:
        other = 'phdos' if cfg.single_task == 'edos' else 'edos'
        for name, parameter in transformer.named_parameters():
            if name.startswith(f'{other}_'):
                grad = parameter.grad
                unsupervised[name] = 0. if grad is None else float(grad.detach().abs().max())
        if not unsupervised or any(value != 0. for value in unsupervised.values()):
            raise ValueError(f"{arm['name']}: the unsupervised task readout received a gradient")
    model.to(torch.device('cpu'))
    del model
    torch.cuda.empty_cache()
    return {'name': arm['name'], 'configuration': asdict(cfg), 'model_contract_schema': contract['schema'],
            'reference_control': contract.get('reference_control'), 'initialization': initialization,
            'data_order_sha256': order, 'historical_order_sha256': historical_order,
            'update_seconds': timings, 'peak_allocated_mib': peak, 'losses': losses,
            'loss_evaluation_difference': abs(before['loss'] - auxiliary - objective),
            'unsupervised_readout_max_abs_grad': max(unsupervised.values()) if unsupervised else None,
            'batch_atoms': [int(batch[0].ne(0).sum()) for batch in batches]}


def check(preparation, out, names=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'use a new preflight output directory: {out}')
    saved = json.loads(Path(preparation).read_text())
    if any(file_sha256(p) != digest for p, digest in saved['input_sha256'].items()):
        raise ValueError('preparation inputs or source changed; prepare a new isolated trial')
    if not torch.cuda.is_available():
        raise ValueError('real-batch preflight requires CUDA')
    torch.set_num_threads(1)
    arms = [arm for arm in saved['arms'] if names is None or arm['name'] in names]
    if names is not None and {arm['name'] for arm in arms} != set(names):
        raise ValueError('unknown arm name requested')
    results = [check_arm(arm, saved['completed_control']) for arm in arms]
    report = {'schema': 'lr2-es-reference-controls-preflight-v1', 'passed': True,
              'preparation_sha256': file_sha256(preparation), 'gpu': torch.cuda.get_device_name(),
              'arms': results, 'full_training_started': False,
              'scope': 'two disposable updates and an lr=0 objective check per arm; no checkpoint or test'}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--arms', nargs='*', default=None)
    args = parser.parse_args()
    report = check(input_path(args.preparation), input_path(args.out), args.arms)
    for arm in report['arms']:
        print(arm['name'], [round(t, 3) for t in arm['update_seconds']], round(arm['peak_allocated_mib']),
              arm['loss_evaluation_difference'])


if __name__ == '__main__':
    main()
