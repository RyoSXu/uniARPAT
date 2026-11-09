"""Disposable real-batch checks for the MACE frozen-feature arm M1; never trains.

Checks: production assembly with the feature directory; the h2 reference built
from the same configuration without features reproduces the frozen h2 initial
tensors; every M1 tensor except the zero projection equals that state; the
initial forward pass on real train and valid batches equals h2 within the
h2 repeat noise (the features enter through a zero projection); the projection receives a non-zero
gradient; the contract records the features; timed disposable updates and an
lr=0 objective/evaluation check. No checkpoint is written; test is never loaded.
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

from tools.checks.check_structure_generalization import TIMED_BATCHES, build, timed
from tools.train import configure_local_global_execution, evaluate_local_global_split
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import comparison_contract
from utils.paths import file_sha256, input_path
from utils.spectral_balance import ENERGY_METRIC
from utils.spectral_balance_experiment import validate_balance_config
from utils.zproj_alignment import state_hash

PROJECTION = 'external_atom_proj.'


def body(state):
    return {key: value for key, value in state.items() if not key.startswith(PROJECTION)}


def first_batches(builder, cfg):
    train = builder.get_dataloader(split='train', batch_size=cfg.batch_size, dos_minmax=True, dos_sumnorm=True)
    valid = builder.get_dataloader(split='valid', batch_size=cfg.batch_size, dos_minmax=True, dos_sumnorm=True)
    train.sampler.set_epoch(0)
    return train, [next(iter(train)), next(iter(valid))]


@torch.no_grad()
def outputs(model, batches):
    transformer = model.model['transformer']
    transformer.eval()
    result = []
    for batch in batches:
        inp, pos, mask, *_, ex, px = model.data_preprocess(batch)
        out = transformer(inp, mask, pos, ex, px)
        result.append({key: out[key].detach().float().cpu() for key in ('edos', 'phdos')})
    return result


def check_arm(arm, control):
    cfg = ExperimentConfig(**arm['configuration'])
    validate_balance_config(cfg)
    reference_cfg = replace(cfg, external_atom_features='', tag=cfg.tag + '_reference')
    _, reference, _, reference_init, _ = build(reference_cfg)
    if reference_init['state_sha256'] != control['initialization_state_sha256']:
        raise ValueError('h2 reference no longer reproduces the frozen h2 initial tensors')
    builder, model, params, initialization, _ = build(cfg)
    transformer = model.model['transformer']
    state = transformer.state_dict()
    if state_hash(body(state)) != reference_init['state_sha256']:
        raise ValueError('M1 tensors other than the projection differ from the h2 initial state')
    projection = {k: v for k, v in state.items() if k.startswith(PROJECTION)}
    if set(projection) != {PROJECTION + 'weight', PROJECTION + 'bias'} or any(v.abs().max() != 0 for v in projection.values()):
        raise ValueError('the external projection must exist and start at zero')
    if params.get('external_atom_dim') != arm['external_atom_features']['dim']:
        raise ValueError('model feature width differs from the frozen features')
    contract = comparison_contract(cfg, params)
    if contract.get('external_atom_features', {}).get('features') != arm['external_atom_features']:
        raise ValueError('contract does not record the frozen features')

    device = torch.device('cuda')
    loader, batches = first_batches(builder, cfg)
    if any(len(b) <= 17 or b[17] is None for b in batches):
        raise ValueError('feature batches must carry field 17')
    if any(b[17].shape[:2] != b[0].shape for b in batches):
        raise ValueError('feature field does not align with the atom slots')
    model.to(device)
    reference.to(device)
    configure_local_global_execution(transformer)
    configure_local_global_execution(reference.model['transformer'])
    plain = [b[:17] for b in batches]
    mine, theirs, again = outputs(model, batches), outputs(reference, plain), outputs(reference, plain)

    def gap(left, right):
        return max(float((a[k] - b[k]).abs().max()) for a, b in zip(left, right) for k in a)

    difference, noise = gap(mine, theirs), gap(theirs, again)
    # CUDA scatter kernels may differ run to run; the zero projection must add nothing beyond that.
    if difference > max(2 * noise, 1e-6):
        raise ValueError(f'initial M1 forward differs from h2 ({difference:.3e}; repeat noise {noise:.3e})')
    reference.to(torch.device('cpu'))
    del reference

    transformer.train()
    model.optimizer['transformer'].zero_grad(set_to_none=True)
    inp, pos, mask, *_, ex, px = model.data_preprocess(batches[0])
    out = transformer(inp, mask, pos, ex, px)
    (out['edos'].float().mean() + out['phdos'].float().mean()).backward()
    gradient = float(transformer.external_atom_proj.weight.grad.abs().sum())
    if not np.isfinite(gradient) or gradient == 0.0:
        raise ValueError('the zero projection receives no gradient')
    model.optimizer['transformer'].zero_grad(set_to_none=True)

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
        raise ValueError('training and evaluation objectives differ')
    model.to(torch.device('cpu'))
    del model
    torch.cuda.empty_cache()
    return {'name': arm['name'], 'configuration': asdict(cfg), 'contract': contract['external_atom_features'],
            'initialization': initialization, 'h2_initial_state_sha256': reference_init['state_sha256'],
            'initial_forward_max_abs_difference_vs_h2': difference, 'h2_repeat_forward_noise': noise,
            'projection_gradient_l1': gradient,
            'parameters': sum(v.numel() for v in state.values() if v.is_floating_point()),
            'projection_parameters': sum(v.numel() for v in projection.values()),
            's_per_batch_cold_warm': timings, 'batches_per_epoch': len(loader),
            'estimated_train_seconds_per_epoch_warm': timings[-1] * len(loader), 'peak_allocated_mib': peak,
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
    report = {'schema': 'lr2-es-mace-features-preflight-v1', 'passed': True,
              'preparation_sha256': file_sha256(preparation), 'gpu': torch.cuda.get_device_name(),
              'arms': results, 'full_training_started': False,
              'scope': 'h2 initial-state identity, initial forward equality (within h2 repeat noise) on one train and one valid batch, '
                       f'projection gradient, {TIMED_BATCHES} timed disposable updates (cold and warm cache) and an '
                       'lr=0 objective check; no checkpoint or test'}
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
        print(arm['name'], arm['projection_parameters'], arm['parameters'],
              arm['initial_forward_max_abs_difference_vs_h2'],
              [round(s, 3) for s in arm['s_per_batch_cold_warm']], round(arm['peak_allocated_mib']))


if __name__ == '__main__':
    main()
