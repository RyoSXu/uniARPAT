"""Pre-training verification for the decoupled global-depth comparison arms.

Bounded checks on the real Q1 data at real model size: the historical call
order and initialization stay bit-identical, the extended global stack is one
shared initial state for both layouts, real train batches run forward and
backward with finite losses and gradients, both spectra keep E(3) output
invariance on real structures, contracts isolate the depth variants, and a
checkpoint round trip restores weights, optimizer state and RNG streams.
Representative GPU batch costs are reported separately with their accounting
convention. No full resource scan, no training and no writes outside the report.
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

from model.transformer import Transformer
from tools.eval.periodic_geometry_acceptance import e3_operations
from tools.train import configure_local_global_execution, evaluate_split
from utils.ablation_checkpoint import build_ablation_checkpoint, restore_ablation_checkpoint
from utils.builder import ConfigBuilder, build_warmup_cosine_scheduler
from utils.crystal import CrystalGeometry, atom_padding_mask
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import (
    DEEP_CONTRACT_SCHEMA, INIT_PROTOCOL, capture_training_rng,
    comparison_contract, initialize_comparison_model, restore_training_rng,
)
from utils.spectral_inference import reconstruct_h1_spectra

ARM_RUN = 'results/runs/ablation_m1_lg_s42sel20261005_k2_bp_staged_s42/config_used.yaml'
# The other route is re-checked too: its recorded hashes must also be unchanged.
ARM_RUN_A = 'results/runs/ablation_m1_lg_s42sel20261005_k2_a_staged_s42/config_used.yaml'
# Frozen recipe of the comparison family; only the encoder schedule changes.
RECIPE = {'d_model': 512, 'nhead': 8, 'num_encoder_layers': 6, 'num_decoder_layers': 6,
          'dim_feedforward': 2048, 'dropout': 0.05, 'activation': 'gelu',
          'edos_num': 128, 'phdos_num': 64, 'atom_feat_mode': 'z_only_proj',
          'scale_mode': 'eta', 'decoupled_decoder': False, 'use_gated_cross_attn': False,
          'head_type': 'legacy', 'predict_scale': False, 'q1_coord': True}
TRAINING = dict(epochs=35, batch_size=32, lr=5e-5, seed=42, atom_feat='z_only_proj',
                skip_test_eval=True, local_route='B+', local_depth=2, global_depth=6,
                selection_metric='balanced_oracle_mae')
HISTORICAL_ORDER = {
    'staged': (('local', 0), ('local', 1), ('global', 0), ('global', 1)),
    'alternating': (('local', 0), ('global', 0), ('local', 1), ('global', 1)),
}
DEEP_ORDER = {
    'staged': (('local', 0), ('local', 1)) + tuple(('global', i) for i in range(6)),
    'alternating': (('local', 0), ('global', 0), ('local', 1), ('global', 1))
                   + tuple(('global', i) for i in range(2, 6)),
}
COUNTS_DEEP = {'local_calls': 2, 'internal_local_blocks': 4,
               'global_attention': 6, 'global_ffn': 6}


def finite(values) -> bool:
    return all(np.isfinite(float(value)) for value in values)


def take(loader, batches):
    picked, seen = [], 0
    for batch in loader:
        picked.append(batch)
        seen += len(batch[0])
        if len(picked) == batches:
            break
    if len(picked) != batches:
        raise ValueError(f'expected {batches} batches, got {len(picked)}')
    return picked, seen


def optimizer_difference(left, right):
    bad = []
    for key, slots in left.items():
        other = right.get(key)
        for name, value in slots.items():
            candidate = None if other is None else other.get(name)
            if candidate is None or not torch.equal(
                    torch.as_tensor(value).cpu(), torch.as_tensor(candidate).cpu()):
                bad.append(f'{key}.{name}')
    return sorted(bad)


def transformed(geometry, matrix, translation):
    """Physical E(3) transform: rotate the lattice and shift Cartesian atoms."""
    batch = geometry.cell.shape[0]
    cell = geometry.cell @ matrix
    shift = translation.expand(batch, 1, 3)
    frac = geometry.frac_coords + torch.bmm(shift, torch.linalg.inv(cell))
    return CrystalGeometry(cell, frac)


def e3_case(model, transformer, batch, operation, dtype):
    """One physical E(3) transform of a real batch; compare outputs and spectra.

    The transformer and every floating input are converted to ``dtype`` first, so
    the FP64 case is a real double-precision computation and not an FP32 run
    reported in double. Outputs are compared in that working dtype.
    """
    matrix, translation = operation[1], operation[2]
    values = model.data_preprocess(batch)
    inp, mask, nval = values[0], values[2], values[15]
    pos = CrystalGeometry(values[1].cell.to(dtype), values[1].frac_coords.to(dtype))
    ex, px = values[16].to(dtype), values[17].to(dtype)
    atoms = (~atom_padding_mask(pos, mask)).sum(-1)
    reference = transformer(inp, mask, pos, ex, px)
    spectra = reconstruct_h1_spectra(reference, atoms, nval,
                                     delta_edos=model.delta_edos, delta_phdos=model.delta_phdos)
    moved = transformed(pos, torch.as_tensor(matrix, dtype=dtype, device=pos.cell.device),
                        torch.as_tensor(translation, dtype=dtype, device=pos.cell.device))
    actual = transformer(inp, mask, moved, ex, px)
    moved_spectra = reconstruct_h1_spectra(actual, atoms, nval,
                                           delta_edos=model.delta_edos,
                                           delta_phdos=model.delta_phdos)
    rows = {}
    for key in ('edos', 'phdos', 'eta'):
        rows[key] = difference(reference[key], actual[key])
    for key in ('shape_edos', 'shape_phdos', 'blind_edos', 'blind_phdos'):
        rows[key] = difference(spectra[key], moved_spectra[key])
    return rows


def difference(expected, actual):
    scale = float(expected.abs().max())
    worst = float((actual - expected).abs().max())
    return {'max_abs': worst, 'max_rel': float(worst / max(scale, 1e-12))}


def check(out: Path, arm_config: Path, batches: int = 2) -> None:
    if out.exists():
        raise FileExistsError(f'refusing existing report: {out}')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report = {'scope': 'bounded pre-training verification; no training, no resource scan'}

    saved = yaml.safe_load(arm_config.read_text())
    arm_params = saved['config']['model']['params']['sub_model']['transformer']
    recorded = saved['runtime']['local_global_init']
    if arm_params.get('local_route') != TRAINING['local_route'] \
            or arm_params.get('local_layout') != 'staged' or arm_params.get('local_depth') != 2:
        raise ValueError('the reference arm must be the trained route in staged/K=2 configuration')
    deep_params = {**arm_params, 'global_depth': 6}
    wrong = {key: [deep_params.get(key), value] for key, value in RECIPE.items()
             if deep_params.get(key) != value}
    if wrong:
        raise ValueError(f'depth recipe deviates from the comparison family: {wrong}')
    report['recipe'] = {'matches_comparison_family': True, 'verified_keys': sorted(RECIPE),
                        'source': str(arm_config), 'selection_metric': TRAINING['selection_metric'],
                        'training': dict(TRAINING),
                        'data_contract': saved['cli']['data_contract'],
                        'trainer': saved['config']['trainer'],
                        'optimizer': saved['config']['model']['params']['optimizer'],
                        'lr_scheduler': saved['config']['model']['params']['lr_scheduler']}

    # 1) The historical call order and initialization must be untouched, for
    #    both routes: equal-depth arms keep their recorded hashes and order.
    historical = {}
    for route, config_path in (('A', ROOT / ARM_RUN_A), (TRAINING['local_route'], arm_config)):
        config = yaml.safe_load(config_path.read_text())
        params = config['config']['model']['params']['sub_model']['transformer']
        record = config['runtime']['local_global_init']
        entry = {}
        for layout in ('staged', 'alternating'):
            shaped = {**params, 'local_layout': layout}
            model = Transformer(**shaped)
            info = initialize_comparison_model(model, shaped, 42)
            entry[layout] = {
                'execution_order_matches':
                    [list(step) for step in model.encoder.execution_order]
                    == [list(step) for step in HISTORICAL_ORDER[layout]],
                'update_counts_match': info['update_counts'] == record['update_counts'],
                'common_state_sha256_matches':
                    info['common_state_sha256'] == record['common_state_sha256'],
                'state_sha256_matches': info['state_sha256'] == record['state_sha256'],
            }
            del model
        historical[route] = {'config': str(config_path),
                             'recorded_common_state_sha256': record['common_state_sha256'],
                             'recorded_state_sha256': record['state_sha256'],
                             'update_counts': record['update_counts'],
                             'layouts': entry}
    if not all(all(values.values()) for entry in historical.values()
               for values in entry['layouts'].values()):
        raise ValueError(f'historical order/initialization changed: {historical}')
    equal_depth = Transformer(**{**arm_params, 'global_depth': 2})
    initialize_comparison_model(equal_depth, {**arm_params, 'global_depth': 2}, 42)
    if [list(step) for step in equal_depth.encoder.execution_order] != \
            [list(step) for step in HISTORICAL_ORDER['staged']]:
        raise ValueError('an explicit equal global_depth must reproduce the historical order')
    report['historical_semantics'] = {
        'routes': historical,
        'explicit_equal_depth_matches': True,
        'historical_update_counts': {'local_calls': 2, 'internal_local_blocks': 4,
                                     'global_attention': 2, 'global_ffn': 2}}
    del equal_depth

    # 2) Decoupled depth: one shared initial state for both layouts.
    initialized = {}
    for layout in ('staged', 'alternating'):
        params = {**deep_params, 'local_layout': layout}
        model = Transformer(**params)
        initialized[layout] = initialize_comparison_model(model, params, 42)
        del model
    shared_reference_matches = {
        layout: {'shared_is_reference_state':
                 info['shared_state_sha256'] == recorded['state_sha256'],
                 'reference_common_matches_four_arm':
                 info['reference_common_state_sha256'] == recorded['common_state_sha256'],
                 'execution_order': info['execution_order'] == [list(step) for step in DEEP_ORDER[layout]],
                 'update_counts': info['update_counts'] == COUNTS_DEEP}
        for layout, info in initialized.items()}
    for layout, checks in shared_reference_matches.items():
        if not all(checks.values()):
            raise ValueError(f'depth initialization mismatch for {layout}: {checks}')
    identical = {key: initialized['staged'][key] == initialized['alternating'][key]
                 for key in ('shared_state_sha256', 'extended_global_state_sha256',
                             'state_sha256', 'common_state_sha256',
                             'reference_common_state_sha256')}
    if not all(identical.values()):
        raise ValueError(f'the two layouts do not share one initial state: {identical}')
    if initialized['staged']['extended_global_state_sha256'] == initialized['staged']['shared_state_sha256']:
        raise ValueError('extended global layers must be an independent initialization')
    report['depth_initialization'] = {
        'protocol': initialized['staged']['protocol'],
        'shared_init_protocol': initialized['staged']['shared_init_protocol'],
        'reference': initialized['staged']['reference'],
        'shared_state_sha256': initialized['staged']['shared_state_sha256'],
        'extended_global_state_sha256': initialized['staged']['extended_global_state_sha256'],
        'state_sha256': initialized['staged']['state_sha256'],
        'common_state_sha256': initialized['staged']['common_state_sha256'],
        'reference_common_state_sha256': initialized['staged']['reference_common_state_sha256'],
        'n_shared_tensors': initialized['staged']['n_shared_tensors'],
        'n_extended_global_tensors': initialized['staged']['n_extended_global_tensors'],
        'layout_identity': identical,
        'per_layout': shared_reference_matches,
        'update_counts': initialized['staged']['update_counts'],
        'execution_order': {layout: [list(step) for step in DEEP_ORDER[layout]]
                            for layout in DEEP_ORDER}}

    config = {**saved['config'], 'model': {**saved['config']['model'],
              'params': {**saved['config']['model']['params'],
                         'sub_model': {'transformer': deep_params}}}}
    builder = ConfigBuilder(**config)
    model = builder.get_model()
    transformer = model.model['transformer']
    rebuilt = initialize_comparison_model(transformer, deep_params, 42)
    if rebuilt['state_sha256'] != initialized['staged']['state_sha256'] \
            or rebuilt['shared_state_sha256'] != initialized['staged']['shared_state_sha256']:
        raise ValueError('the depth initialization is not reproducible through ConfigBuilder')
    report['depth_initialization']['builder_rebuild_matches'] = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loader = builder.get_dataloader(split='train', dos_minmax=True,
                                          batch_size=32, dos_sumnorm=True)
    valid_loader = builder.get_dataloader(split='valid', dos_minmax=True,
                                          batch_size=32, dos_sumnorm=True)
    train_batches, _ = take(train_loader, batches)

    # 3) Real batches: forward, backward, finite loss and gradients everywhere.
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
        missing = []
        for name, parameter in transformer.named_parameters():
            if parameter.grad is None:
                missing.append(name)
                continue
            if not torch.isfinite(parameter.grad).all():
                raise ValueError(f'non-finite gradient: {name}')
            total = total + parameter.grad.detach().float().pow(2).sum()
        if missing:
            raise ValueError(f'parameters without gradient: {missing[:5]}')
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
                                  'grad_norms': grad_norms, 'finite': True,
                                  'all_parameters_receive_gradients': True}

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

    # 4) E(3) output invariance of both spectra on real structures.
    transformer.eval()
    identity = {name: id(parameter) for name, parameter in transformer.named_parameters()}
    e3_report = {}
    for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-9)):
        transformer.to(dtype)
        worst = {}
        for batch in valid_batches:
            for operation in e3_operations():
                rows = e3_case(model, transformer, batch, operation, dtype)
                for key, values in rows.items():
                    current = worst.get(key, {'max_abs': 0.0, 'max_rel': 0.0})
                    worst[key] = {'max_abs': max(current['max_abs'], values['max_abs']),
                                  'max_rel': max(current['max_rel'], values['max_rel'])}
        offending = {key: values for key, values in worst.items()
                     if values['max_rel'] > tolerance}
        if offending:
            raise ValueError(f'E(3) output invariance exceeded {tolerance} '
                             f'for {str(dtype).split(".")[-1]}: {offending}')
        e3_report[str(dtype).split('.')[-1]] = {
            'operations': [name for name, _, _ in e3_operations()],
            'batches': len(valid_batches), 'tolerance_max_rel': tolerance,
            'observed': worst}
    transformer.to(torch.float32)
    if {name: id(parameter) for name, parameter in transformer.named_parameters()} != identity:
        raise ValueError('dtype conversion replaced parameter objects')
    report['e3_invariance'] = {
        'inputs': 'real valid batches, physical lattice rotation/reflection/inversion/translation',
        'outputs': 'eDOS, phDOS and H1 scale logits plus shape/blind reconstructed spectra',
        'parameter_identity_preserved': True,
        **e3_report}

    # 5) Contract isolation and checkpoint round trip.
    shallow_params = {key: value for key, value in arm_params.items()}
    deep_contract = comparison_contract(
        ExperimentConfig(**{**TRAINING, 'local_layout': 'staged'}), deep_params)
    shallow_contract = comparison_contract(
        ExperimentConfig(**{**TRAINING, 'local_layout': 'staged', 'global_depth': None}),
        shallow_params)
    if deep_contract['schema'] != DEEP_CONTRACT_SCHEMA \
            or shallow_contract['schema'] != 'local-global-model-v1':
        raise ValueError('unexpected contract schema')
    optimizer = model.optimizer['transformer']
    payload = build_ablation_checkpoint(1, 'M1', 42, False, transformer, optimizer, 0.5,
                                        model_contract=deep_contract,
                                        training_rng=capture_training_rng())
    fresh = builder.get_model()
    fresh.to(device)
    restored = restore_ablation_checkpoint(payload, fresh.model['transformer'],
                                           fresh.optimizer['transformer'], False,
                                           model_contract=deep_contract)
    rejected = {}
    for name, contract in (('shallow', shallow_contract), ('missing', None),
                           ('other_layout', comparison_contract(
                               ExperimentConfig(**{**TRAINING, 'local_layout': 'alternating'}),
                               {**deep_params, 'local_layout': 'alternating'}))):
        try:
            restore_ablation_checkpoint(payload, fresh.model['transformer'],
                                        fresh.optimizer['transformer'], False,
                                        model_contract=contract)
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
    rng_equal = expected[0] == actual[0] and torch.equal(expected[1], actual[1])
    schedule = []
    for start_epoch in (0, 5, 21, 35):
        fresh_optimizer = builder.get_model().optimizer['transformer']
        direct = build_warmup_cosine_scheduler(fresh_optimizer, 35)
        for _ in range(start_epoch):
            direct.step()
        resumed_model = builder.get_model()
        restore_ablation_checkpoint(payload, resumed_model.model['transformer'],
                                    resumed_model.optimizer['transformer'], False,
                                    model_contract=deep_contract)
        resumed = build_warmup_cosine_scheduler(resumed_model.optimizer['transformer'], 35)
        for _ in range(start_epoch):
            resumed.step()
        schedule.append({'start_epoch': start_epoch,
                         'fresh_lr': fresh_optimizer.param_groups[0]['lr'],
                         'resumed_lr': resumed_model.optimizer['transformer'].param_groups[0]['lr']})
    if any(abs(row['fresh_lr'] - row['resumed_lr']) > 1e-15 for row in schedule):
        raise ValueError(f'LR schedule resume mismatch: {schedule}')
    report['contract_and_resume'] = {
        'deep_schema': deep_contract['schema'], 'shallow_schema': shallow_contract['schema'],
        'deep_reference': deep_contract['reference'],
        'rejected_cross_contracts': rejected,
        'epoch': restored['epoch'], 'optimizer_restored': restored['optimizer_restored'],
        'weights_equal': not unequal, 'unequal_weights': unequal[:5],
        'optimizer_state_equal': not opt_unequal, 'unequal_optimizer': opt_unequal[:5],
        'rng_replay_equal': bool(rng_equal), 'lr_schedule': schedule}

    # 6) Representative costs with an explicit accounting convention.
    configure_local_global_execution(transformer, augment=False)
    model.model['transformer'].train()
    costs = []
    cases = (('warm', train_batches[0], False),
             ('warm_tail', train_batches[min(1, len(train_batches) - 1)], False),
             ('cold_cache', train_batches[0], True))
    for label, batch, cold in cases:
        cache = getattr(transformer, '_local_geometry_cache', None)
        if not cold:
            for _ in range(2):
                model.train_one_step(batch, step=0, pair_budget=None)
        if cache is not None:
            cache.clear()
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.time()
        values = model.train_one_step(batch, step=0, pair_budget=None)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        costs.append({'case': label, 'samples': len(batch[0]),
                      'atoms': int(batch[0].ne(0).sum(-1).sum()),
                      'microbatches': list(model.last_microbatch_sizes),
                      'loss': float(values['loss']), 'wall_s': time.time() - started,
                      'peak_allocated_mb': (torch.cuda.max_memory_allocated() / 2**20
                                            if device.type == 'cuda' else 0.0),
                      'peak_reserved_mb': (torch.cuda.max_memory_reserved() / 2**20
                                           if device.type == 'cuda' else 0.0)})
    cache = getattr(transformer, '_local_geometry_cache', None)
    report['gpu_batches'] = {
        'device': str(device), 'note': 'representative batches, not a resource scan',
        'accounting': 'PyTorch CUDA allocator counters over the measured step only: '
                      'peak_allocated_mb is max_memory_allocated, peak_reserved_mb is '
                      'max_memory_reserved. Both include the model, the optimizer state '
                      '(AdamW moments already allocated by the earlier updates) and the '
                      'lazily filled geometry cache; neither includes the CUDA context. '
                      'warm/warm_tail run after two warm-up steps on the same batch and '
                      'measure a populated cache; cold_cache clears the cache immediately '
                      'before the measured step, so the step pays the geometry rebuild.',
        'geometry_cache': cache.configuration() if cache is not None else {'enabled': False},
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
