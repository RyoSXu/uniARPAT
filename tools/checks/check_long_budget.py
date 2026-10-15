"""Pre-training verification for the 100-epoch budget comparison arms.

Without training this verifies, per arm: the frozen recipe is unchanged apart
from the budget, the approved 100-epoch warmup-plus-cosine learning-rate
trajectory is in place and differs from the 35-epoch one, the arm starts from
the established untrained initial values with unchanged hashes, the data and
model contracts match the data manifest and isolate the new budget from every
older checkpoint, and a checkpoint round trip restores weights, optimizer state
and RNG streams. Representative real-batch forward/backward is included.
No model formula is changed and no contract is weakened.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from utils.checkpoint_selection import selected_history_row

from tools.checks.check_global_depth import RECIPE
from tools.checks.check_lg_baseline import optimizer_difference, take
import tools.train as runner
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.builder import ConfigBuilder, build_warmup_cosine_scheduler
from utils.crystal_cache import crystal_data_contract
from utils.experiment_config import ExperimentConfig
from utils.lg_baseline_experiment import (
    BASELINE_CONTRACT_SCHEMA, BASELINE_INIT_PROTOCOL,
    baseline_contract, initialize_baseline_model, validate_baseline_config,
)
from utils.local_global_experiment import (
    DEEP_CONTRACT_SCHEMA, DEEP_INIT_PROTOCOL, capture_training_rng,
    comparison_contract, initialize_comparison_model, restore_training_rng,
    validate_local_global_config,
)
from utils.paths import repository_path

BUDGET = {'epochs': 100, 'batch_size': 32, 'lr': 5e-5, 'seed': 42,
          'atom_feat': 'z_only_proj', 'skip_test_eval': True,
          'selection_metric': 'balanced_oracle_mae'}
SHORT_BUDGET = 35
WARMUP_EPOCHS, WARMUP_LR, MIN_LR = 5, 1e-5, 1e-6
# Established untrained initial values; the 100-epoch arms must reproduce them.
EXPECTED_HASHES = {
    'baseline': {'common_state_sha256': '79c996382bf2c3a5af9d30ee280de1c75994a82d8e113e0c56f9b7d39ad61804',
                 'encoder_state_sha256': '927adeaf6ca39f2590b6cd9e4c720e915906d7173a79c01272100390719b1d4a',
                 'state_sha256': '7bb47f4f4eb828d95b055d20b18a0ccfb06d3fecffe106bbf861351655e19793',
                 'reference_common_state_sha256': '8eea1e0c95063424fb2dea2a8a09bdf25952d5ff5e3b748dd9e913f0032a35a6'},
    'deep_g6': {'shared_state_sha256': '9746b5825aab96bab6aaefd05d33accdfb21e97d4e72f9e54c8e147725ad5c34',
                'extended_global_state_sha256': '944ad93d990eca8ba6894b596ffb087334b2b8669fec836c89d77dabecd47345',
                'common_state_sha256': '0e3f9c7bc57bb4a50bffc04ee2284c02a35589adc7848ab6b80d5e4fce94c8f9',
                'reference_common_state_sha256': '8eea1e0c95063424fb2dea2a8a09bdf25952d5ff5e3b748dd9e913f0032a35a6',
                'state_sha256': 'dbdbbdf50a076083b7810687b7c2f6498bad55bb8fa6c8a1a50bcb0c486c9850'},
}
ARMS = (
    {'label': 'baseline', 'tag': '_lg_baseline_q1v1_e100_s42',
     'reference': 'results/runs/ablation_m1_lg_baseline_q1v1_e35_s42/config_used.yaml',
     'contract_schema': BASELINE_CONTRACT_SCHEMA, 'init_protocol': BASELINE_INIT_PROTOCOL,
     'hashes': EXPECTED_HASHES['baseline']},
    {'label': 'deep_g6', 'tag': '_lgd_s42g6e10020261008_k2_g6_bp_staged_s42',
     'reference': 'results/runs/ablation_m1_lgd_s42g6sel20261008_k2_g6_bp_staged_s42/config_used.yaml',
     'contract_schema': DEEP_CONTRACT_SCHEMA, 'init_protocol': DEEP_INIT_PROTOCOL,
     'hashes': EXPECTED_HASHES['deep_g6']},
)


def finite(values) -> bool:
    return all(np.isfinite(float(value)) for value in values)


def arm_config(arm: dict) -> ExperimentConfig:
    """The exact configuration this arm will train with."""
    if arm['label'] == 'baseline':
        cfg = ExperimentConfig(**{**BUDGET, 'tag': arm['tag'], 'lg_baseline': True,
                                  'q1_coord': True})
        validate_baseline_config(cfg)
    else:
        cfg = ExperimentConfig(**{**BUDGET, 'tag': arm['tag'], 'local_route': 'B+',
                                  'local_layout': 'staged', 'local_depth': 2,
                                  'global_depth': 6, 'q1_coord': True})
        validate_local_global_config(cfg)
    return cfg


def learning_rate_trace(epochs: int, base_lr: float) -> list:
    """Learning rate used by each epoch, in the runner's build-and-step order."""
    probe = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(probe.parameters())
    for group in optimizer.param_groups:
        group['lr'] = base_lr
        group['initial_lr'] = base_lr
    scheduler = build_warmup_cosine_scheduler(optimizer, epochs)
    trace = []
    for _ in range(epochs):
        trace.append(float(optimizer.param_groups[0]['lr']))
        scheduler.step()
    trace.append(float(optimizer.param_groups[0]['lr']))
    return trace


def captured_config(cfg: ExperimentConfig) -> dict:
    """The configuration the runner would assemble and persist.

    ``ConfigBuilder.get_model`` writes its resolved data-preprocessing fields
    back into the configuration before the run persists it, so the capture runs
    that write-back with the model construction stubbed out.
    """
    captured = {}

    def record(**kwargs):
        builder = ConfigBuilder(**kwargs)
        with patch('model.model.basemodel', lambda *args, **keywords: object()):
            builder.get_model()
        captured['config'] = deepcopy(kwargs)
        raise RuntimeError('preflight capture stops before data/model construction')

    with patch.object(runner, 'ConfigBuilder', side_effect=record):
        try:
            runner.train_and_eval(cfg)
        except RuntimeError as error:
            if 'preflight capture' not in str(error):
                raise
    return captured['config']


def check(arm: dict, batches: int = 2) -> dict:
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report = {'arm': arm['label'], 'tag': arm['tag'], 'budget': dict(BUDGET),
              'scope': 'bounded pre-training verification; no training, no resource scan'}
    # The runner reads tools/training-defaults.yaml with the process default text
    # encoding. That encoding can degrade to ANSI_X3.4-1968 during GPU training at
    # an unspecified point (observed between the first and later train_one_step
    # calls, independent of num_workers and pin_memory), which is the recorded
    # 2026-10-07 queue incident. PEP 540 UTF-8 mode makes text I/O UTF-8 whatever
    # the locale says, so every process of this experiment must start with it.
    if not sys.flags.utf8_mode:
        raise ValueError('this experiment requires PYTHONUTF8=1 (PEP 540 UTF-8 mode): '
                         'the process default text encoding is not deterministic')
    report['environment'] = {
        'python_utf8_mode': bool(sys.flags.utf8_mode),
        'reason': 'text I/O must not depend on the process locale; see the encoding '
                  'incident note in the experiment log'}
    saved = yaml.safe_load((ROOT / arm['reference']).read_text())
    params = saved['config']['model']['params']['sub_model']['transformer']
    recorded_init = saved['runtime'].get('lg_baseline_init') or saved['runtime']['local_global_init']
    cfg = arm_config(arm)

    # --- New tag identity: nothing may already exist for it. ---
    run_dir = repository_path(f"results/runs/ablation_{cfg.model_name.lower()}{arm['tag']}")
    existing = sorted(path.name for path in Path(run_dir).glob('*')) if Path(run_dir).is_dir() else []
    if existing:
        raise ValueError(f'tag {arm["tag"]} already has artifacts: {existing}; '
                         'verify identity before any rerun')
    report['tag_identity'] = {'run_dir': str(run_dir), 'existing_artifacts': existing,
                              'fresh_start': True}

    # --- Frozen recipe: only the budget and the tag may differ. ---
    # Compare the recorded 35-epoch CLI against the configuration this arm would
    # actually run, field by field, so an omitted scientific knob cannot hide.
    reference_cli = dict(saved['cli'])
    field_names = {field.name for field in fields(ExperimentConfig)}
    cli_diffs = {key: [value, getattr(cfg, key)]
                 for key, value in sorted(reference_cli.items())
                 if key in field_names and getattr(cfg, key) != value}
    recipe_matches = {key: params.get(key) == value for key, value in RECIPE.items()}
    if set(cli_diffs) != {'epochs'}:
        raise ValueError(f'configuration differs from the frozen recipe in {sorted(cli_diffs)}')
    if not all(recipe_matches.values()):
        raise ValueError(f'transformer recipe deviates: {recipe_matches}')
    report['recipe'] = {
        'source': arm['reference'],
        'cli_diffs_against_reference': cli_diffs,
        'transformer_recipe_keys': sorted(RECIPE),
        'transformer_matches': recipe_matches,
        'optimizer': saved['config']['model']['params']['optimizer'],
        'lr_scheduler': saved['config']['model']['params']['lr_scheduler'],
        'trainer': saved['config']['trainer'],
        'dataset': saved['config']['dataset'],
        'data_contract': saved['cli']['data_contract'],
    }

    assembled = captured_config(cfg)
    if assembled != saved['config']:
        changed = sorted(key for key in set(assembled) | set(saved['config'])
                         if assembled.get(key) != saved['config'].get(key))
        raise ValueError(f'assembled configuration differs from the frozen recipe: {changed}')
    leftovers = sorted(path.name for path in Path(run_dir).glob('*')) if Path(run_dir).is_dir() else []
    if leftovers:
        raise ValueError(f'configuration capture left artifacts: {leftovers}')
    report['recipe']['assembled_config_matches_reference'] = True

    # --- Learning-rate trajectory of the approved budget. ---
    trace = learning_rate_trace(BUDGET['epochs'], BUDGET['lr'])
    short = learning_rate_trace(SHORT_BUDGET, BUDGET['lr'])
    shared = min(len(trace), len(short))
    lr_checks = {
        'starts_at_warmup_lr': abs(trace[0] - WARMUP_LR) < 1e-12,
        'reaches_base_lr_after_warmup': abs(trace[WARMUP_EPOCHS] - BUDGET['lr']) < 1e-12,
        'warmup_monotone': all(trace[i] <= trace[i + 1] + 1e-15 for i in range(WARMUP_EPOCHS)),
        'cosine_monotone': all(trace[i] >= trace[i + 1] - 1e-15
                               for i in range(WARMUP_EPOCHS, BUDGET['epochs'])),
        'floors_at_min_lr': abs(trace[BUDGET['epochs']] - MIN_LR) < 1e-12,
        'differs_from_35_epoch_trace': any(abs(trace[i] - short[i]) > 1e-12
                                           for i in range(shared)),
    }
    if not all(lr_checks.values()):
        raise ValueError(f'learning-rate trajectory rejected: {lr_checks}')
    report['learning_rate'] = {
        'schedule': 'warmup 5 epochs linear then cosine to 1e-6',
        'warmup_epochs': WARMUP_EPOCHS, 'warmup_lr': WARMUP_LR, 'min_lr': MIN_LR,
        'checks': lr_checks,
        'trace_100_epoch': {str(i): value for i, value in enumerate(trace)},
        'trace_35_epoch_reference': {str(i): value for i, value in enumerate(short)},
        'note': 'the 100-epoch trajectory is not the 35-epoch one; the first 35 epochs '
                'of this run are not a reproduction of the earlier experiments',
    }

    # --- Initialization: the same established untrained values. ---
    # The runner seeds before model construction; the baseline's old encoder is
    # drawn from that stream, so the seed context is reproduced here too.
    initialize = (lambda model: initialize_baseline_model(model, params, BUDGET['seed'])
                  if arm['label'] == 'baseline'
                  else initialize_comparison_model(model, params, BUDGET['seed']))
    runner.setup_ablation_seed(BUDGET['seed'])
    probe = ConfigBuilder(**assembled).get_model()
    init = initialize(probe.model['transformer'])
    hash_checks = {key: init[key] == value for key, value in arm['hashes'].items()}
    reference_checks = {key: recorded_init.get(key) == value for key, value in arm['hashes'].items()}
    if init['protocol'] != arm['init_protocol']:
        raise ValueError(f'unexpected init protocol: {init["protocol"]}')
    if not all(hash_checks.values()):
        raise ValueError(f'initialization hashes changed: {hash_checks}')
    if not all(reference_checks.values()):
        raise ValueError(f'recorded reference hashes disagree with expected: {reference_checks}')
    report['initialization'] = {
        'protocol': init['protocol'],
        'hashes': {key: init[key] for key in arm['hashes']},
        'matches_established_values': hash_checks,
        'reference_records_agree': reference_checks,
        'source': arm['reference']}
    del probe

    # --- Data and model contracts; budget isolation from older checkpoints. ---
    data_contract = crystal_data_contract(
        repository_path(cfg.data_dir, root=ROOT), normalization=cfg.norm,
        pair_budget=cfg.pair_budget)
    if data_contract != saved['cli']['data_contract']:
        raise ValueError('data contract differs from the frozen recipe')
    contract = (baseline_contract(cfg, params) if arm['label'] == 'baseline'
                else comparison_contract(cfg, params))
    older = saved['model_contract']
    training_diffs = sorted(key for key in set(contract['training']) | set(older['training'])
                            if contract['training'].get(key) != older['training'].get(key))
    if contract['schema'] != arm['contract_schema']:
        raise ValueError(f'unexpected contract schema: {contract["schema"]}')
    if contract['training']['epochs'] != BUDGET['epochs']:
        raise ValueError(f'contract does not carry the budget: {contract["training"]["epochs"]}')
    if contract == older or training_diffs != ['epochs']:
        raise ValueError(f'only the budget may differ between contracts: {training_diffs}')
    report['contracts'] = {
        'schema': contract['schema'], 'init_protocol': contract['init_protocol'],
        'training_epochs': contract['training']['epochs'], 'data_contract': data_contract,
        'differs_from_35_epoch_contract': True,
        'training_keys_differing_from_35_epoch': training_diffs}

    # --- Real batch forward/backward and a checkpoint round trip. ---
    runner.setup_ablation_seed(BUDGET['seed'])
    builder = ConfigBuilder(**assembled)
    model = builder.get_model()
    transformer = model.model['transformer']
    initialize(transformer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_loader = builder.get_dataloader(split='train', dos_minmax=True,
                                          batch_size=BUDGET['batch_size'], dos_sumnorm=True)
    train_batches, _ = take(train_loader, batches)
    transformer.train()
    losses = []
    for step, batch in enumerate(train_batches):
        values = model.train_one_step(batch, step=step, pair_budget=None)
        entry = {key: float(value) for key, value in values.items()}
        if not finite(entry.values()):
            raise ValueError(f'non-finite training loss: {entry}')
        for name, parameter in transformer.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError(f'non-finite gradient: {name}')
        losses.append(entry)
    report['forward_backward'] = {'train_batches': len(train_batches), 'losses': losses,
                                  'finite': True}

    optimizer = model.optimizer['transformer']
    payload = build_ablation_checkpoint(1, cfg.model_name, BUDGET['seed'], False, transformer,
                                        optimizer, 0.5, model_contract=contract,
                                        training_rng=capture_training_rng())
    fresh = builder.get_model()
    fresh.to(device)
    restored = restore_ablation_checkpoint(payload, fresh.model['transformer'],
                                           fresh.optimizer['transformer'], False,
                                           model_contract=contract)
    rejected = {}
    for name, other in (('35_epoch_contract', saved['model_contract']), ('missing', None)):
        try:
            restore_ablation_checkpoint(payload, fresh.model['transformer'],
                                        fresh.optimizer['transformer'], False,
                                        model_contract=other)
        except ValueError:
            rejected[name] = True
        else:
            rejected[name] = False
    if not all(rejected.values()):
        raise ValueError(f'contract isolation failed: {rejected}')
    left, right = transformer.state_dict(), fresh.model['transformer'].state_dict()
    unequal = sorted(key for key in left if not torch.equal(left[key].cpu(), right[key].cpu()))
    opt_unequal = optimizer_difference(optimizer.state_dict()['state'],
                                       fresh.optimizer['transformer'].state_dict()['state'])
    stream = capture_training_rng()
    restore_training_rng(payload['training_rng'])
    expected = (float(np.random.random()), torch.rand(2))
    restore_training_rng(payload['training_rng'])
    actual = (float(np.random.random()), torch.rand(2))
    restore_training_rng(stream)

    # LR resume mirrors the runner: build the scheduler from the configured base
    # learning rate, restore the checkpoint, then fast-forward the schedule.
    schedule = []
    for start_epoch in (0, 5, 21, 35, 100):
        resumed_model = builder.get_model()
        resumed_model.to(device)
        resumed_optimizer = resumed_model.optimizer['transformer']
        for group in resumed_optimizer.param_groups:
            group['lr'] = BUDGET['lr']
            group['initial_lr'] = BUDGET['lr']
        resumed = build_warmup_cosine_scheduler(resumed_optimizer, BUDGET['epochs'])
        restore_ablation_checkpoint(payload, resumed_model.model['transformer'],
                                    resumed_optimizer, False, model_contract=contract)
        for _ in range(start_epoch):
            resumed.step()
        schedule.append({'start_epoch': start_epoch,
                         'resumed_lr': float(resumed_optimizer.param_groups[0]['lr']),
                         'trace_lr': trace[start_epoch]})
    if any(abs(row['resumed_lr'] - row['trace_lr']) > 1e-15 for row in schedule):
        raise ValueError(f'LR schedule resume mismatch: {schedule}')
    report['resume'] = {
        'epoch': restored['epoch'], 'optimizer_restored': restored['optimizer_restored'],
        'weights_equal': not unequal, 'unequal_weights': unequal[:5],
        'optimizer_state_equal': not opt_unequal, 'unequal_optimizer': opt_unequal[:5],
        'rng_replay_equal': bool(expected[0] == actual[0] and torch.equal(expected[1], actual[1])),
        'contract_mismatch_rejected': rejected,
        'lr_schedule_resume': schedule}
    return report


def verify_completed(arm: dict) -> dict:
    """Completion checks after training: records, checkpoints and contracts."""
    cfg = arm_config(arm)
    run_dir = repository_path(f"results/runs/ablation_{cfg.model_name.lower()}{arm['tag']}")
    history_path = repository_path(f'results/reports/history_{cfg.model_name.lower()}{arm["tag"]}.csv')
    if not history_path.is_file():
        raise ValueError(f'missing history for {arm["tag"]}: {history_path}')
    history = pd.read_csv(history_path)
    if len(history) != BUDGET['epochs']:
        raise ValueError(f'{arm["tag"]} has {len(history)} epoch records, expected {BUDGET["epochs"]}')
    epochs = [int(value) for value in history['epoch']]
    if epochs != list(range(1, BUDGET['epochs'] + 1)):
        raise ValueError(f'{arm["tag"]} epoch records are not 1..{BUDGET["epochs"]}: {epochs[:5]}…')
    numbers = history.select_dtypes('number')
    if not np.isfinite(numbers.to_numpy()).all():
        raise ValueError(f'{arm["tag"]} has non-finite epoch records')
    for column in ('lr', 'epoch_time_s', 'peak_vram_mb', 'balanced_oracle_mae',
                   'blind_r2_edos_median', 'oracle_r2_edos_median', 'blind_r2_phdos_median',
                   'oracle_r2_phdos_median', 'blind_fail_rate_edos', 'oracle_fail_rate_edos',
                   'blind_fail_rate_phdos', 'oracle_fail_rate_phdos', 'mae_edos_median',
                   'mae_phdos_median', 'train_loss'):
        if column not in history.columns:
            raise ValueError(f'{arm["tag"]} history is missing the required column {column}')
    trace = learning_rate_trace(BUDGET['epochs'], BUDGET['lr'])
    lr_mismatch = [int(row.epoch) for row in history.itertuples()
                   if abs(float(row.lr) - trace[int(row.epoch) - 1]) > 1e-12]
    if lr_mismatch:
        raise ValueError(f'{arm["tag"]} recorded learning rates differ from the schedule '
                         f'at epochs {lr_mismatch[:5]}')

    saved = yaml.safe_load((run_dir / 'config_used.yaml').read_text())
    contract = saved['model_contract']
    if contract['training']['epochs'] != BUDGET['epochs']:
        raise ValueError(f'{arm["tag"]} contract does not carry the budget')
    recorded_init = saved['runtime'].get('lg_baseline_init') or saved['runtime']['local_global_init']
    init_matches = {key: recorded_init.get(key) == value for key, value in arm['hashes'].items()}
    if not all(init_matches.values()):
        raise ValueError(f'{arm["tag"]} initial values changed during training: {init_matches}')

    checkpoints = {}
    for name in ('checkpoint_best', 'checkpoint_latest'):
        path = run_dir / f'{name}.pth'
        if not path.is_file():
            raise ValueError(f'{arm["tag"]} is missing {name}.pth')
        payload = torch.load(path, map_location='cpu')
        if payload.get('model_contract') != contract:
            raise ValueError(f'{arm["tag"]} {name} carries a different model contract')
        if not isinstance(payload.get('training_rng'), dict):
            raise ValueError(f'{arm["tag"]} {name} has no training RNG state')
        if not torch.isfinite(torch.cat([value.reshape(-1).float()
                                         for value in payload['model'].values()])).all():
            raise ValueError(f'{arm["tag"]} {name} holds non-finite weights')
        checkpoints[name] = {'epoch': int(payload['epoch']),
                             'best_val_score': float(payload['best_val_score']),
                             'bytes': path.stat().st_size}
    if checkpoints['checkpoint_latest']['epoch'] != BUDGET['epochs']:
        raise ValueError(f'{arm["tag"]} latest checkpoint is at epoch '
                         f'{checkpoints["checkpoint_latest"]["epoch"]}')
    best = selected_history_row(history, contract['selection_metric'])
    if int(checkpoints['checkpoint_best']['epoch']) != int(best.epoch):
        raise ValueError(f'{arm["tag"]} best checkpoint epoch {checkpoints["checkpoint_best"]["epoch"]} '
                         f'!= selected history epoch {int(best.epoch)}')
    if not np.isclose(checkpoints['checkpoint_best']['best_val_score'], float(best.balanced_score)):
        raise ValueError(f'{arm["tag"]} best checkpoint score does not match the selected history score')
    return {
        'arm': arm['label'], 'tag': arm['tag'],
        'epoch_records': len(history),
        'epochs_contiguous_1_to_budget': True,
        'all_records_finite': True,
        'learning_rate_matches_schedule': True,
        'best_epoch': int(best.epoch),
        'best_score': float(best.balanced_score),
        'checkpoints': checkpoints,
        'contract_schema': contract['schema'],
        'init_hashes_unchanged': init_matches,
        'epoch_time_s_median': float(history['epoch_time_s'].median()),
        'epoch_time_s_sum': float(history['epoch_time_s'].sum()),
        'peak_vram_mb_median': float(history['peak_vram_mb'].median()),
        'peak_vram_mb_max': float(history['peak_vram_mb'].max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batches', type=int, default=2)
    parser.add_argument('--verify-completed', action='store_true',
                        help='Run the post-training completion checks instead')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'refusing existing report: {args.out}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.verify_completed:
        reports = [verify_completed(arm) for arm in ARMS]
        args.out.write_text(json.dumps(reports, ensure_ascii=False, indent=2, default=float) + '\n')
        print(json.dumps(reports, ensure_ascii=False, indent=2, default=float))
        return
    reports = [check(arm, args.batches) for arm in ARMS]
    args.out.write_text(json.dumps(reports, ensure_ascii=False, indent=2, default=float) + '\n')
    printable = []
    for report in reports:
        entry = {key: value for key, value in report.items() if key != 'learning_rate'}
        entry['learning_rate'] = {key: value for key, value in report['learning_rate'].items()
                                  if key not in ('trace_100_epoch', 'trace_35_epoch_reference')}
        printable.append(entry)
    print(json.dumps(printable, ensure_ascii=False, indent=2, default=float))


if __name__ == '__main__':
    main()
