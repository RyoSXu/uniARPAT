"""Pre-training verification for the baseline comparison arm.

Bounded checks on the real Q1 data at real model size: matched initialization
against the four arms' recorded reference, finite forward/backward, the shared
blind/oracle evaluator, checkpoint resume (weights, optimizer, RNG, LR
schedule) and representative GPU batch costs. No full resource scan, no
training and no writes outside the report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from model.transformer import Transformer, TransformerEncoder
from tools.train import evaluate_split
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.builder import ConfigBuilder, build_warmup_cosine_scheduler
from utils.experiment_config import ExperimentConfig
from utils.lg_baseline_experiment import (
    BASELINE_CONTRACT_SCHEMA, baseline_contract, initialize_baseline_model,
)
from utils.local_global_experiment import capture_training_rng, restore_training_rng

ARM_RUN = 'results/runs/ablation_m1_lg_s42sel20261005_k2_a_staged_s42/config_used.yaml'
# Frozen recipe of the comparison family; the encoder is the only difference.
RECIPE = {'d_model': 512, 'nhead': 8, 'num_encoder_layers': 6, 'num_decoder_layers': 6,
          'dim_feedforward': 2048, 'dropout': 0.05, 'activation': 'gelu',
          'edos_num': 128, 'phdos_num': 64, 'atom_feat_mode': 'z_only_proj',
          'scale_mode': 'eta', 'decoupled_decoder': False, 'use_gated_cross_attn': False,
          'head_type': 'legacy', 'predict_scale': False, 'q1_coord': True}
TRAINING = dict(epochs=35, batch_size=32, lr=5e-5, seed=42,
                tag='_lg_baseline_q1v1_e35_s42', atom_feat='z_only_proj',
                skip_test_eval=True, lg_baseline=True,
                selection_metric='balanced_oracle_mae')


def baseline_params_from_arm(params: dict) -> dict:
    """Baseline parameters: the arm's values with the encoder route removed."""
    derived = {key: value for key, value in params.items()
               if key not in ('local_route', 'local_layout', 'local_depth', 'lg_baseline')}
    derived.update({'local_route': 'none', 'lg_baseline': True})
    return derived


def finite(values) -> bool:
    return all(np.isfinite(float(value)) for value in values)


def optimizer_difference(left, right):
    """Keys whose AdamW moments differ after a restore."""
    bad = []
    for key, slots in left.items():
        other = right.get(key)
        for name, value in slots.items():
            candidate = None if other is None else other.get(name)
            if candidate is None or not torch.equal(
                    torch.as_tensor(value).cpu(), torch.as_tensor(candidate).cpu()):
                bad.append(f'{key}.{name}')
    return sorted(bad)


def take(loader, batches):
    """First ``batches`` batches plus the number of samples they carry."""
    picked, seen = [], 0
    for batch in loader:
        picked.append(batch)
        seen += len(batch[0])
        if len(picked) == batches:
            break
    if len(picked) != batches:
        raise ValueError(f'expected {batches} batches, got {len(picked)}')
    return picked, seen


def check(out: Path, arm_config: Path, batches: int = 2) -> None:
    if out.exists():
        raise FileExistsError(f'refusing existing report: {out}')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report = {'scope': 'bounded pre-training verification; no training, no resource scan'}

    saved = yaml.safe_load(arm_config.read_text())
    arm_params = saved['config']['model']['params']['sub_model']['transformer']
    arm_common = saved['runtime']['local_global_init']['common_state_sha256']
    if arm_params.get('local_route') != 'A' or arm_params.get('local_layout') != 'staged' \
            or arm_params.get('local_depth') != 2:
        raise ValueError('the reference arm must be the A/staged/K=2 configuration')
    params = baseline_params_from_arm(arm_params)
    wrong = {key: [params.get(key), value] for key, value in RECIPE.items()
             if params.get(key) != value}
    if wrong:
        raise ValueError(f'baseline recipe deviates from the comparison family: {wrong}')
    report['recipe'] = {'matches_comparison_family': True, 'verified_keys': sorted(RECIPE),
                        'source': str(arm_config), 'selection_metric': TRAINING['selection_metric'],
                        'data_contract': saved['cli']['data_contract'],
                        'trainer': saved['config']['trainer'],
                        'optimizer': saved['config']['model']['params']['optimizer'],
                        'lr_scheduler': saved['config']['model']['params']['lr_scheduler']}

    probe = Transformer(**params)
    if not isinstance(probe.encoder, TransformerEncoder):
        raise ValueError('the baseline arm must keep the original encoder type')
    if len(probe.encoder.layers) != RECIPE['num_encoder_layers'] \
            or not hasattr(probe.encoder, 'rp_encoder') \
            or not hasattr(probe.encoder.layers[0], 'rp_proj'):
        raise ValueError('the baseline arm must keep the six-layer encoder with geometry scoring')
    report['encoder'] = {'class': type(probe.encoder).__name__,
                         'layers': len(probe.encoder.layers),
                         'geometry_scoring': 'shared RPEncoding + per-layer rp_proj'}
    init = initialize_baseline_model(probe, params, 42)
    if init['reference_common_state_sha256'] != arm_common:
        raise ValueError('the copied common part does not match the four-arm reference: '
                         f"{init['reference_common_state_sha256']} != {arm_common}")
    report['initialization'] = {**init, 'arms_common_state_sha256': arm_common,
                                'reference_common_state_sha256_matches_arms': True}
    del probe

    config = {**saved['config'], 'model': {**saved['config']['model'],
              'params': {**saved['config']['model']['params'],
                         'sub_model': {'transformer': params}}}}
    builder = ConfigBuilder(**config)
    model = builder.get_model()
    transformer = model.model['transformer']
    again = initialize_baseline_model(transformer, params, 42)
    if again['common_state_sha256'] != init['common_state_sha256'] \
            or again['reference_common_state_sha256'] != init['reference_common_state_sha256']:
        raise ValueError('the copied common part is not reproducible through ConfigBuilder')
    if again['encoder_state_sha256'] == init['encoder_state_sha256']:
        raise ValueError('the original encoder must be independently initialized')
    report['initialization']['builder_rebuild'] = {
        'common_state_sha256_matches': True,
        'encoder_state_sha256_differs': True,
        'encoder_state_sha256': again['encoder_state_sha256'],
        'state_sha256': again['state_sha256'],
        'note': 'encoder init follows the construction RNG; training re-seeds seed=42 before the loop'}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loader = builder.get_dataloader(split='train', dos_minmax=True,
                                          batch_size=32, dos_sumnorm=True)
    valid_loader = builder.get_dataloader(split='valid', dos_minmax=True,
                                          batch_size=32, dos_sumnorm=True)
    train_batches, _ = take(train_loader, batches)

    model.model['transformer'].train()
    losses, grad_norms = [], []
    for step, batch in enumerate(train_batches):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.time()
        values = model.train_one_step(batch, step=step, pair_budget=None)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        entry = {key: float(value) for key, value in values.items()}
        if not finite(entry.values()):
            raise ValueError(f'non-finite training loss: {entry}')
        total = torch.zeros((), device=device)
        for name, parameter in transformer.named_parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise ValueError(f'non-finite gradient: {name}')
                total = total + parameter.grad.detach().float().pow(2).sum()
        entry.update({'wall_s': time.time() - started, 'grad_norm': float(total.sqrt().cpu()),
                      'microbatches': list(model.last_microbatch_sizes),
                      'peak_allocated_mb': (torch.cuda.max_memory_allocated() / 2**20
                                            if device.type == 'cuda' else 0.0)})
        grad_norms.append(entry['grad_norm'])
        losses.append(entry)
    if not finite(grad_norms):
        raise ValueError(f'non-finite gradient norm: {grad_norms}')
    if not all(bool(torch.isfinite(value).all()) for value in transformer.state_dict().values()):
        raise ValueError('non-finite parameter after the bounded updates')
    report['forward_backward'] = {'train_batches': len(train_batches), 'losses': losses,
                                  'grad_norms': grad_norms, 'finite': True}

    order = list(valid_loader.sampler)
    material_ids = [str(value) for value in valid_loader.dataset.material_ids]
    valid_batches, seen = take(valid_loader, batches)
    metrics = evaluate_split(model, valid_batches, sumnorm=True, return_sample_level=True,
                             sample_ids=[material_ids[index] for index in order[:seen]])
    frame = metrics.pop('sample_df')
    if len(frame) != seen or not finite(metrics.values()):
        raise ValueError('invalid blind/oracle evaluation output')
    if metrics['r2_edos_median'] != metrics['oracle_r2_edos_median']:
        raise ValueError('generic metrics must alias the oracle columns')
    report['evaluation'] = {
        'samples': len(frame), 'blind_oracle': True,
        'selected_metrics': {key: float(metrics[key]) for key in (
            'balanced_blind_mae', 'balanced_oracle_mae', 'blind_r2_edos_median',
            'blind_r2_phdos_median', 'oracle_r2_edos_median', 'oracle_r2_phdos_median')},
        'material_id_first': str(frame['material_id'].iloc[0])}

    contract = baseline_contract(ExperimentConfig(**TRAINING), params)
    if contract['schema'] != BASELINE_CONTRACT_SCHEMA:
        raise ValueError('unexpected baseline contract schema')
    payload = build_ablation_checkpoint(1, 'M1', 42, False, transformer,
                                        model.optimizer['transformer'], 0.5,
                                        model_contract=contract,
                                        training_rng=capture_training_rng())
    fresh = builder.get_model()
    fresh.to(device)
    restored = restore_ablation_checkpoint(payload, fresh.model['transformer'],
                                           fresh.optimizer['transformer'], False,
                                           model_contract=contract)
    mismatch_rejected = False
    try:
        restore_ablation_checkpoint(payload, fresh.model['transformer'],
                                    fresh.optimizer['transformer'], False)
    except ValueError:
        mismatch_rejected = True
    if not mismatch_rejected:
        raise ValueError('a mismatched contract must be rejected')
    left, right = transformer.state_dict(), fresh.model['transformer'].state_dict()
    unequal = sorted(key for key in left if not torch.equal(left[key].cpu(), right[key].cpu()))
    opt_unequal = optimizer_difference(model.optimizer['transformer'].state_dict()['state'],
                                       fresh.optimizer['transformer'].state_dict()['state'])
    stream = capture_training_rng()
    restore_training_rng(payload['training_rng'])
    expected = (float(np.random.random()), torch.rand(2))
    restore_training_rng(payload['training_rng'])
    actual = (float(np.random.random()), torch.rand(2))
    restore_training_rng(stream)
    rng_equal = expected[0] == actual[0] and torch.equal(expected[1], actual[1])

    schedule = []
    for start_epoch in (0, 5, 21, 35):
        fresh_optimizer = builder.get_model().optimizer['transformer']
        direct = build_warmup_cosine_scheduler(fresh_optimizer, 35)
        for _ in range(start_epoch):
            direct.step()
        resumed_model = builder.get_model()
        resumed_optimizer = resumed_model.optimizer['transformer']
        restore_ablation_checkpoint(payload, resumed_model.model['transformer'],
                                    resumed_optimizer, False, model_contract=contract)
        resumed = build_warmup_cosine_scheduler(resumed_optimizer, 35)
        for _ in range(start_epoch):
            resumed.step()
        schedule.append({'start_epoch': start_epoch,
                         'fresh_lr': fresh_optimizer.param_groups[0]['lr'],
                         'resumed_lr': resumed_optimizer.param_groups[0]['lr']})
    if any(abs(row['fresh_lr'] - row['resumed_lr']) > 1e-15 for row in schedule):
        raise ValueError(f'LR schedule resume mismatch: {schedule}')
    report['resume'] = {
        'epoch': restored['epoch'], 'optimizer_restored': restored['optimizer_restored'],
        'weights_equal': not unequal, 'unequal_weights': unequal[:5],
        'optimizer_state_equal': not opt_unequal, 'unequal_optimizer': opt_unequal[:5],
        'rng_replay_equal': bool(rng_equal),
        'contract_mismatch_rejected': mismatch_rejected, 'lr_schedule': schedule}

    model.model['transformer'].train()
    costs = []
    for batch in (train_batches[0], train_batches[min(1, len(train_batches) - 1)]):
        model.train_one_step(batch, step=0, pair_budget=None)  # warm the kernels
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.time()
        values = model.train_one_step(batch, step=0, pair_budget=None)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        costs.append({'samples': len(batch[0]),
                      'atoms': int(batch[0].ne(0).sum(-1).sum()),
                      'microbatches': list(model.last_microbatch_sizes),
                      'loss': float(values['loss']), 'wall_s': time.time() - started,
                      'peak_allocated_mb': (torch.cuda.max_memory_allocated() / 2**20
                                            if device.type == 'cuda' else 0.0)})
    report['gpu_batches'] = {'device': str(device), 'note': 'representative, not a resource scan',
                             'costs': costs}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm-config', type=Path, default=ROOT / ARM_RUN)
    parser.add_argument('--batches', type=int, default=2)
    args = parser.parse_args()
    check(args.out, args.arm_config, args.batches)


if __name__ == '__main__':
    main()
