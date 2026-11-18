"""Prepare and preflight the approved dos_v1 h2 baseline; never launch training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from dataclasses import asdict
from tools.checks.check_structure_generalization import build
from tools.eval.local_global_compare import training_command
from tools.train import configure_local_global_execution, evaluate_local_global_split
from utils.crystal_cache import crystal_data_contract
from utils.experiment_config import ExperimentConfig
from utils.partial_labels import batch_labels
from utils.paths import file_sha256
from utils.spectral_balance import ENERGY_METRIC, prepare_balance_protocol_v2
from utils.spectral_balance_experiment import validate_balance_config

DATA = ROOT / 'data/datasets/dos_v1/views/dos_v1'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def prepare(out, tag, pair_budget, epochs=35, cohorts=None):
    OUT = Path(out).resolve()
    if pair_budget < 160 ** 2:
        raise ValueError('pair budget must fit one complete 160-atom crystal')
    if OUT.exists():
        raise FileExistsError(f'fresh preparation required: {OUT}')
    protocol = prepare_balance_protocol_v2(DATA, 'valid')
    manifest = json.loads((DATA / 'manifest.json').read_text())
    public_splits = {p.split('/')[0] for p in manifest['files']}
    if manifest.get('dataset') != 'dos_v1' or public_splits != {'train', 'valid'}:
        raise ValueError('only public dos_v1 train/valid may be prepared')
    data_contract = crystal_data_contract(DATA, normalization='sumnorm', pair_budget=pair_budget)
    cfg = ExperimentConfig(epochs=epochs, batch_size=32, lr=5e-5, seed=42,
        tag=tag, data_dir=str(DATA), pair_budget=pair_budget,
        atom_feat='z_only_proj', skip_test_eval=True, q1_coord=True,
        local_route='LR2', local_layout='staged', local_depth=2, global_depth=6,
        scale_mode='per_atom', spectral_shape_aux=True, phdos_head_layers=2,
        selection_metric=ENERGY_METRIC, spectral_balance_config=str(OUT / 'protocol.json'))
    run = ROOT / f'results/runs/ablation_m1{cfg.tag}'
    history = ROOT / f'results/reports/history_m1{cfg.tag}.csv'
    if run.exists() or history.exists():
        raise FileExistsError('baseline tag is already in use')
    OUT.mkdir(parents=True)
    write(OUT / 'protocol.json', protocol)
    validate_balance_config(cfg)
    # Bind all runtime sources and all public payloads, including valid labels/geometry.
    sources = sorted({*ROOT.glob('model/*.py'), *ROOT.glob('utils/*.py'),
                      ROOT / 'tools/train.py', ROOT / 'tools/training-defaults.yaml',
                      Path(__file__).resolve(), ROOT / 'tools/eval/dos_baseline_execute.py',
                      ROOT / 'tools/checks/check_dos_baseline_throughput.py',
                      ROOT / 'tools/eval/lg_valid_export.py'})
    hashes = {str(p): file_sha256(p) for p in sources}
    hashes.update({str(DATA / p): entry['sha256'] for p, entry in manifest['files'].items()})
    hashes[str(DATA / 'manifest.json')] = file_sha256(DATA / 'manifest.json')
    hashes[str(OUT / 'protocol.json')] = file_sha256(OUT / 'protocol.json')
    if cohorts:
        hashes[str(Path(cohorts).resolve())] = file_sha256(cohorts)
    for source in sources:
        target = OUT / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    report = dict(schema='dos-v1-h2-baseline-preparation-v1', training_started=False, preparation_dir=str(OUT),
        configuration=asdict(cfg), command=training_command(cfg), run_dir=str(run),
        benchmark_cohorts=str(Path(cohorts).resolve()) if cohorts else None,
        history=str(history), data_contract=data_contract, input_sha256=hashes,
        environment={'PYTHONUTF8': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
        selection='full valid; each task on labelled rows; 0.5/0.5; earliest minimum; no old guards',
        fit_samples=protocol['fit_samples'], C={t: protocol['tasks'][t]['energy'] for t in ('edos', 'phdos')})
    write(OUT / 'report.json', report)
    (OUT / 'command.txt').write_text(shlex.join(report['command']) + '\n')
    return report


def preflight(report):
    OUT = Path(report['preparation_dir'])
    if (OUT / 'preflight.json').exists():
        raise FileExistsError('preflight already exists')
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise ValueError('CUDA required')
    cfg = ExperimentConfig(**report['configuration'])
    from utils.builder import build_warmup_cosine_scheduler
    probe_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=cfg.lr)
    scheduler = build_warmup_cosine_scheduler(probe_optimizer, cfg.epochs)
    lr_history = []
    for _ in range(cfg.epochs):
        lr_history.append(probe_optimizer.param_groups[0]['lr'])
        probe_optimizer.step()
        scheduler.step()
    assert np.isclose(probe_optimizer.param_groups[0]['lr'], 1e-6)
    builder, model, params, initialization, assembly = build(cfg)
    transformer = model.model['transformer']
    assert params['phdos_head_layers'] == 2
    model.to(torch.device('cuda'))
    configure_local_global_execution(transformer)
    loader = builder.get_dataloader(split='train', batch_size=32, dos_minmax=True, dos_sumnorm=True)
    valid = builder.get_dataloader(split='valid', batch_size=32, dos_minmax=True, dos_sumnorm=True)
    for split, ld, expected in [('train', loader, (58105, 49184, 24639)),
                                ('valid', valid, (7474, 6342, 3080))]:
        flags = [np.load(DATA / split / f'has_{t}_{split}.npy') for t in ('edos', 'phdos')]
        assert (len(ld.dataset), *(int(v.sum()) for v in flags)) == expected
    torch.cuda.reset_peak_memory_stats()
    loader.sampler.set_epoch(0)
    order = list(loader.sampler)
    # Spread timing batches across the complete production epoch rather than its prefix.
    positions = np.linspace(0, len(loader) - 2, 24, dtype=int)
    batches = [loader.collate_fn([loader.dataset[i] for i in order[p*32:(p+1)*32]]) for p in positions]
    seconds = []
    for _ in range(3):
        transformer.train()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for index, batch in enumerate(batches):
            values = model.train_one_step(batch, index, pair_budget=cfg.pair_budget)
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError('non-finite ordinary batch update')
        torch.cuda.synchronize()
        seconds.append((time.perf_counter() - started) / len(batches))
    # Stress the largest complete crystals through the unchanged computation splitter.
    atoms = np.load(DATA / 'train/n_atoms_train.npy')
    rows = np.argsort(atoms)[-32:]
    largest = loader.collate_fn([loader.dataset[int(i)] for i in rows])
    started = time.perf_counter()
    losses = model.train_one_step(largest, 0, pair_budget=cfg.pair_budget)
    torch.cuda.synchronize()
    if not all(np.isfinite(v) for v in losses.values()):
        raise ValueError('non-finite large-crystal update')
    large_seconds = time.perf_counter() - started
    largest_microbatch_sizes = model.last_microbatch_sizes
    dense_checks = []
    if report.get('benchmark_cohorts'):
        cohorts = json.loads(Path(report['benchmark_cohorts']).read_text())
        for case in cohorts['stress_batches']:
            stress = loader.collate_fn([loader.dataset[int(i)] for i in case['rows']])
            started = time.perf_counter()
            values = model.train_one_step(stress, 0, pair_budget=cfg.pair_budget)
            torch.cuda.synchronize()
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError(f"non-finite stress update: {case['name']}")
            dense_checks.append(dict(name=case['name'], seconds=time.perf_counter()-started,
                microbatch_sizes=model.last_microbatch_sizes))
    # Eval-mode lr=0 compares the same objective with labelled-row aggregation.
    batch = next(iter(valid))
    flags = batch_labels(batch)
    assert flags is not None and not flags.all()
    transformer.eval()
    for group in model.optimizer['transformer'].param_groups:
        group['lr'] = 0.
    losses = model.train_one_step(batch, 0, pair_budget=cfg.pair_budget)
    metrics = evaluate_local_global_split(model, [batch], pair_budget=cfg.pair_budget)
    energy = losses['loss'] - .5 * (losses['loss_shape_e'] + losses['loss_shape_p'])
    if not np.isclose(energy, metrics[ENERGY_METRIC], rtol=1e-4, atol=1e-6):
        raise ValueError('train/evaluation partial-label objectives disagree')
    warm = float(np.median(seconds[1:]))
    result = dict(passed=True, full_training_started=False, timed_batches=len(batches),
        cold_warm_seconds_per_batch=seconds, train_batches=len(loader),
        warm_materials_per_second=32 / warm,
        estimated_train_hours=warm * len(loader) * cfg.epochs / 3600, epochs=cfg.epochs,
        estimate_scope='24 batches spread across one epoch; excludes valid, checkpoint I/O, and full cache occupancy',
        largest_batch_atoms=[int(atoms[i]) for i in rows], largest_batch_seconds=large_seconds,
        largest_microbatch_sizes=largest_microbatch_sizes, pair_budget=cfg.pair_budget,
        dense_checks=dense_checks, timing_batch_positions=positions.tolist(),
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
        parameters=sum(p.numel() for p in transformer.parameters()),
        objective_difference=abs(energy - metrics[ENERGY_METRIC]),
        gpu=torch.cuda.get_device_name(), initialization=initialization,
        lr_per_epoch=lr_history, lr_after_budget=probe_optimizer.param_groups[0]['lr'],
        torch_version=torch.__version__, python=sys.executable,
        preparation_sha256=file_sha256(OUT / 'report.json'))
    if any(file_sha256(p) != h for p, h in report['input_sha256'].items()):
        raise ValueError('inputs changed during preflight')
    write(OUT / 'effective_assembly.json', assembly)
    write(OUT / 'preflight.json', result)
    return result


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUTF8', '1')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--pair-budget', type=int, required=True)
    parser.add_argument('--epochs', type=int, choices=(35, 100), required=True)
    parser.add_argument('--cohorts', type=Path)
    args = parser.parse_args()
    report = prepare(args.out, args.tag, args.pair_budget, args.epochs, args.cohorts)
    try:
        print(json.dumps(preflight(report), indent=2), flush=True)
    except Exception as error:
        write(Path(report['preparation_dir']) / 'preflight.json', dict(
            passed=False, full_training_started=False, error_type=type(error).__name__, error=str(error),
            preparation_sha256=file_sha256(Path(report['preparation_dir']) / 'report.json')))
        raise
