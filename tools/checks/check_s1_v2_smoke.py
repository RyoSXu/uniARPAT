"""Disposable checks of the h2 recipe on a v2 partial-label view; never trains or writes a checkpoint.

Assembles the h2 configuration on the view through the runner, runs a few disposable updates on
real partial-label batches, checks with lr=0 that the W1 training objective equals the evaluator's
joint energy on the same batch, then scores the untrained model on the selection split (and any
requested public split) with the per-source report. Test is never loaded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.checks.check_structure_generalization import build
from tools.train import configure_local_global_execution, evaluate_local_global_split
from utils.builder import ConfigBuilder
from utils.experiment_config import ExperimentConfig
from utils.partial_labels import batch_labels
from utils.paths import input_path
from utils.source_report import grouped_report
from utils.spectral_balance import ENERGY_METRIC
from utils.spectral_balance_experiment import validate_balance_config

H2 = dict(model_name='M1', epochs=35, batch_size=32, lr=5e-05, seed=42, edos_num=128, phdos_num=64,
          decoder_layers=6, atom_feat='z_only_proj', norm='sumnorm', skip_test_eval=True,
          scale_mode='per_atom', spectral_shape_aux=True, phdos_head_layers=2,
          local_route='LR2', local_layout='staged', local_depth=2, global_depth=6,
          selection_metric='balanced_blind_energy', q1_coord=True)
UPDATES = 4


def score_split(model, assembly, split, cfg, view):
    config = {**assembly, 'dataset': {**assembly['dataset'],
                                      'valid': {**assembly['dataset']['valid'], 'view_split': split}}}
    loader = ConfigBuilder(**config).get_dataloader(split='valid', batch_size=cfg.batch_size,
                                                    dos_minmax=True, dos_sumnorm=True)
    ids = [str(v) for v in loader.dataset.material_ids]
    metrics = evaluate_local_global_split(model, loader, return_sample_level=True, pair_budget=cfg.pair_budget,
                                          sample_ids=[ids[i] for i in loader.sampler])
    frame = metrics.pop('sample_df')
    report = grouped_report(frame, view, split)
    return {'samples': len(frame),
            'labelled': {t: metrics.get(f'labelled_{t}') for t in ('edos', 'phdos')},
            ENERGY_METRIC: metrics[ENERGY_METRIC],
            'blind_energy_mean': {t: metrics[f'blind_energy_{t}_mean'] for t in ('edos', 'phdos')},
            'groups': report['tasks']}


def check(data_dir, protocol, out, extra_splits=()):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'use a new check output directory: {out}')
    if not torch.cuda.is_available():
        raise ValueError('real-batch check requires CUDA')
    torch.set_num_threads(1)
    cfg = ExperimentConfig(**H2, data_dir=str(data_dir), spectral_balance_config=str(protocol),
                           tag='_sbal_s1v2smoke20261112_h2_s42')
    validate_balance_config(cfg)
    builder, model, params, initialization, assembly = build(cfg)
    view_split = assembly['dataset']['valid'].get('view_split')
    if view_split != model.spectral_balance['view']['valid_split']:
        raise ValueError('runner did not bind the protocol selection split to the valid role')
    transformer = model.model['transformer']
    model.to(torch.device('cuda'))
    configure_local_global_execution(transformer)
    loader = builder.get_dataloader(split='train', batch_size=cfg.batch_size, dos_minmax=True, dos_sumnorm=True)
    loader.sampler.set_epoch(0)
    iterator = iter(loader)
    batches = [next(iterator) for _ in range(UPDATES)]
    label_counts = [batch_labels(b).sum(0).tolist() for b in batches]
    if not any(c[0] < cfg.batch_size or c[1] < cfg.batch_size for c in label_counts):
        raise ValueError('no single-label rows in the checked batches')
    transformer.train()
    losses = []
    for index, batch in enumerate(batches):
        losses.append(model.train_one_step(batch, index, pair_budget=cfg.pair_budget))
        if not all(np.isfinite(v) for v in losses[-1].values()):
            raise ValueError('non-finite disposable update')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in transformer.parameters()):
            raise ValueError('non-finite gradient')
    transformer.eval()
    for group in model.optimizer['transformer'].param_groups:
        group['lr'] = 0.
    before = model.train_one_step(batches[0], 0, pair_budget=cfg.pair_budget)
    evaluated = evaluate_local_global_split(model, batches[:1], pair_budget=cfg.pair_budget)
    auxiliary = .5 * (before['loss_shape_e'] + before['loss_shape_p'])
    difference = abs(before['loss'] - auxiliary - evaluated[ENERGY_METRIC])
    if not np.isclose(before['loss'] - auxiliary, evaluated[ENERGY_METRIC], rtol=3e-6, atol=1e-7):
        raise ValueError(f'training and evaluation objectives differ by {difference}')
    view = Path(data_dir)
    report = {'schema': 's1-v2-h2-smoke-v1', 'passed': True, 'training_started': False,
              'configuration': asdict(cfg), 'initialization': initialization,
              'valid_view_split': view_split, 'train_batches': len(loader),
              'label_counts_per_batch': label_counts, 'losses': losses,
              'loss_evaluation_difference': difference,
              'untrained_scores': {split: score_split(model, assembly, split, cfg, view)
                                   for split in (view_split, *extra_splits)},
              'scope': f'{UPDATES} disposable updates and an lr=0 objective check; untrained-model scores '
                       'only exercise the evaluator and report; no checkpoint or test'}
    out.mkdir(parents=True)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--extra-splits', nargs='*', default=[], help='other public splits to score, e.g. dev')
    args = parser.parse_args()
    report = check(input_path(args.data_dir), input_path(args.protocol), input_path(args.out), args.extra_splits)
    print(json.dumps({k: report[k] for k in ('valid_view_split', 'label_counts_per_batch',
                                             'loss_evaluation_difference')}, indent=1))
    for split, scores in report['untrained_scores'].items():
        print(split, scores['samples'], scores['labelled'], scores[ENERGY_METRIC])


if __name__ == '__main__':
    main()
