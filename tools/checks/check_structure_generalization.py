"""Disposable real-batch checks for the LR2 E+S generalization arms; never trains.

Per arm: production assembly and initializer (seed-42 tensors must equal the
adopted control), geometry cache setting, training-only perturbation (G) or
eval-path identity (D), two disposable updates, lr=0 objective/evaluation
consistency, and a timed run of extra batches against a warm-cache LR2
baseline to estimate the epoch cost. No checkpoint is written; test is never loaded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.checks.check_reference_controls import assembled
from tools.train import configure_local_global_execution, evaluate_local_global_split, setup_ablation_seed
from utils.builder import ConfigBuilder
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import comparison_contract, initialize_comparison_model
from utils.paths import file_sha256, input_path
from utils.spectral_balance import ENERGY_METRIC
from utils.spectral_balance_experiment import validate_balance_config

HISTORICAL_EPOCH_SECONDS = 417.  # LR2 E+S median epoch with the warm geometry cache
TIMED_BATCHES = 20
EVAL_TOLERANCE = 1e-4  # absolute, on batch metrics of order 0.1-1


def build(cfg):
    setup_ablation_seed(cfg.seed)
    assembly = assembled(cfg)
    builder = ConfigBuilder(**assembly)
    model = builder.get_model()
    params = assembly['model']['params']['sub_model']['transformer']
    initialization = initialize_comparison_model(model.model['transformer'], params, cfg.seed)
    return builder, model, params, initialization, assembly


def timed(model, loader, cfg, count, passes=1):
    transformer = model.model['transformer']
    transformer.train()
    seconds = []
    for _ in range(passes):
        loader.sampler.set_epoch(0)
        iterator = iter(loader)
        batches = [next(iterator) for _ in range(count)]
        torch.cuda.synchronize()
        started = time.perf_counter()
        for index, batch in enumerate(batches):
            losses = model.train_one_step(batch, index, pair_budget=cfg.pair_budget)
            if not all(np.isfinite(value) for value in losses.values()):
                raise ValueError('non-finite disposable update')
        torch.cuda.synchronize()
        seconds.append((time.perf_counter() - started) / count)
    return seconds


def baseline_seconds(arm):
    """Warm-cache seconds per batch of the unregularized adopted recipe on the same batches."""
    cfg = replace(ExperimentConfig(**arm['configuration']), strain_augment=False, local_drop_prob=None,
                  tag=arm['configuration']['tag'] + '_base')
    builder, model, _, _, _ = build(cfg)
    model.to(torch.device('cuda'))
    configure_local_global_execution(model.model['transformer'])
    loader = builder.get_dataloader(split='train', batch_size=cfg.batch_size, dos_minmax=True, dos_sumnorm=True)
    cold, warm = timed(model, loader, cfg, TIMED_BATCHES, passes=2)
    model.to(torch.device('cpu'))
    del model
    torch.cuda.empty_cache()
    return {'cold_cache_s_per_batch': cold, 'warm_cache_s_per_batch': warm}


def check_arm(arm, control, baseline):
    cfg = ExperimentConfig(**arm['configuration'])
    validate_balance_config(cfg)
    builder, model, params, initialization, assembly = build(cfg)
    transformer = model.model['transformer']
    contract = comparison_contract(cfg, params)
    if contract.get('regularization', {}).get('kind') != ('strain_augment' if cfg.strain_augment else 'local_drop'):
        raise ValueError(f"{arm['name']}: contract does not record its regularization arm")
    same = initialization['state_sha256'] == control['initialization']['state_sha256']
    if (cfg.seed == 42) != same:
        raise ValueError(f"{arm['name']}: initial tensors do not follow the declared seed")
    model.to(torch.device('cuda'))
    configure_local_global_execution(transformer, augment=cfg.strain_augment)
    cache_enabled = transformer._local_geometry_cache is not None and bool(
        transformer._local_geometry_cache.configuration().get('enabled'))
    if cache_enabled == cfg.strain_augment:
        raise ValueError(f"{arm['name']}: geometry cache must be off exactly for strain augmentation")
    loader = builder.get_dataloader(split='train', batch_size=cfg.batch_size, dos_minmax=True,
                                    dos_sumnorm=True, sampler_seed=cfg.data_order_seed)
    if bool(loader.dataset.strain_augment) != cfg.strain_augment:
        raise ValueError(f"{arm['name']}: training dataset perturbation flag differs")
    if builder.get_dataloader(split='valid', batch_size=cfg.batch_size, dos_minmax=True,
                              dos_sumnorm=True).dataset.strain_augment:
        raise ValueError('valid inputs must never be perturbed')
    loader.sampler.set_epoch(0)
    iterator = iter(loader)
    batches = [next(iterator), next(iterator)]
    perturbation = None
    if cfg.strain_augment:
        plain = replace(cfg, strain_augment=False)
        plain_builder = ConfigBuilder(**{**assembly, 'dataset': {**assembly['dataset'], 'train': {
            **assembly['dataset']['train'], 'strain_augment': False}}})
        plain_loader = plain_builder.get_dataloader(split='train', batch_size=plain.batch_size, dos_minmax=True,
                                                    dos_sumnorm=True, sampler_seed=plain.data_order_seed)
        plain_loader.sampler.set_epoch(0)
        reference = next(iter(plain_loader))
        ratio = (torch.det(batches[0][1].cell.double()) / torch.det(reference[1].cell.double()))
        perturbation = {'volume_ratio_min': float(ratio.min()), 'volume_ratio_max': float(ratio.max()),
                        'labels_identical': bool(torch.equal(batches[0][2], reference[2])
                                                 and torch.equal(batches[0][3], reference[3]))}
        if not perturbation['labels_identical'] or not (.96 < perturbation['volume_ratio_min']
                                                        and perturbation['volume_ratio_max'] < 1.04):
            raise ValueError(f"{arm['name']}: perturbation changed labels or exceeded its strain bound")
        if torch.equal(batches[0][1].cell, reference[1].cell):
            raise ValueError(f"{arm['name']}: training geometry was not perturbed")
    eval_identity = None
    if cfg.local_drop_prob is not None:
        # Same module and warm geometry cache with the drop switched off. GPU evaluation is not
        # bitwise repeatable (the CPU unit test checks exact identity), so compare against repeat noise.
        transformer.eval()

        def evaluated_metrics(model=model):
            return evaluate_local_global_split(model, batches[:1], pair_budget=cfg.pair_budget)

        def largest(left, right):
            return max(abs(left[k] - right[k]) for k in left if isinstance(left[k], float))

        with torch.no_grad():
            evaluated_metrics()  # warm the geometry cache
            on, repeat = evaluated_metrics(), evaluated_metrics()
            transformer.encoder.local_drop_prob = None
            off = evaluated_metrics()
            transformer.encoder.local_drop_prob = cfg.local_drop_prob
        eval_identity = {'drop_on_vs_off': largest(on, off), 'repeat_noise': largest(on, repeat)}
        if eval_identity['drop_on_vs_off'] > EVAL_TOLERANCE:
            raise ValueError(f"{arm['name']}: eval output depends on the training-only drop {eval_identity}")
    torch.cuda.reset_peak_memory_stats()
    # With the cache on (D), the second pass over the same batches is the steady-state cost.
    timings = timed(model, loader, cfg, TIMED_BATCHES, passes=1 if cfg.strain_augment else 2)
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
    ratio = timings[-1] / baseline['warm_cache_s_per_batch']
    return {'name': arm['name'], 'configuration': asdict(cfg), 'regularization': contract['regularization'],
            'initialization': initialization, 'geometry_cache_enabled': cache_enabled,
            'perturbation': perturbation, 'eval_identity_max_abs': eval_identity,
            's_per_batch': timings[-1], 'cost_ratio_vs_warm_lr2': ratio,
            'estimated_epoch_seconds': HISTORICAL_EPOCH_SECONDS * ratio, 'peak_allocated_mib': peak,
            'loss_evaluation_difference': abs(before['loss'] - auxiliary - evaluated[ENERGY_METRIC])}


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
    baseline = baseline_seconds(arms[0])
    results = [check_arm(arm, saved['completed_control'], baseline) for arm in arms]
    report = {'schema': 'lr2-es-structure-generalization-preflight-v1', 'passed': True,
              'preparation_sha256': file_sha256(preparation), 'gpu': torch.cuda.get_device_name(),
              'baseline': baseline, 'arms': results, 'full_training_started': False,
              'scope': f'{TIMED_BATCHES} timed disposable updates and an lr=0 objective check per arm; '
                       'no checkpoint or test'}
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
    print('baseline warm s/batch', round(report['baseline']['warm_cache_s_per_batch'], 3))
    for arm in report['arms']:
        print(arm['name'], round(arm['s_per_batch'], 3), round(arm['cost_ratio_vs_warm_lr2'], 2),
              round(arm['estimated_epoch_seconds']), round(arm['peak_allocated_mib']))


if __name__ == '__main__':
    main()
