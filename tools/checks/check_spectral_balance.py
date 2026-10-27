"""Two disposable real-batch updates and objective/evaluation consistency check.

No experiment checkpoint is written. Train/valid loaders and the unchanged A64
initializer are reused; test is never evaluated. Results bind source snapshots.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from utils.spectral_balance import ENERGY_METRIC, normalized_spectral_error, reconstruct_balanced_spectra
from utils.spectral_balance_experiment import validate_balance_config


def check(preparation, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'use a new preflight output directory: {out}')
    saved = json.loads(Path(preparation).read_text())
    if any(file_sha256(p) != digest for p, digest in saved['input_sha256'].items()):
        raise ValueError('preparation inputs or source changed; prepare a new isolated trial')
    cfg = ExperimentConfig(**saved['configuration'])
    protocol = validate_balance_config(cfg)
    if not torch.cuda.is_available():
        raise ValueError('real-batch cost preflight requires CUDA')
    torch.set_num_threads(1)
    setup_ablation_seed(cfg.seed)
    # Capture the production assembly, never the training loop.
    assembly = captured_config(cfg)
    builder = ConfigBuilder(**assembly)
    model = builder.get_model()
    params = assembly['model']['params']['sub_model']['transformer']
    initialization = initialize_comparison_model(model.model['transformer'], params, cfg.seed)
    control = saved.get('completed_control')
    if cfg.spectral_shape_aux and (control is None or control['initialization'] != initialization):
        raise ValueError('shape candidate initial tensors differ from the completed pure-energy control')
    model.to(torch.device('cuda'))
    configure_local_global_execution(model.model['transformer'])
    loader = builder.get_dataloader(split='train', batch_size=cfg.batch_size, dos_minmax=True, dos_sumnorm=True)
    iterator = iter(loader)
    batches = [next(iterator), next(iterator)]
    model.model['transformer'].train()
    timings, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for index, batch in enumerate(batches):
        torch.cuda.synchronize()
        started = time.perf_counter()
        losses.append(model.train_one_step(batch, index, pair_budget=cfg.pair_budget))
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        if not all(np.isfinite(value) for value in losses[-1].values()):
            raise ValueError('non-finite disposable training update')
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.model['transformer'].parameters()):
            raise ValueError('non-finite model gradient')
    peak = torch.cuda.max_memory_allocated() / 1024**2
    # lr=0 makes the training wrapper a reproducible loss/gradient evaluator.
    # Eval mode removes dropout when comparing the two official call paths.
    model.model['transformer'].eval()
    for group in model.optimizer['transformer'].param_groups:
        group['lr'] = 0.
    before = model.train_one_step(batches[0], 0, pair_budget=cfg.pair_budget)
    evaluated = evaluate_local_global_split(model, batches[:1], pair_budget=cfg.pair_budget)
    auxiliary = .5*(before['loss_shape_e'] + before['loss_shape_p']) if cfg.spectral_shape_aux else 0.
    if not np.isclose(before['loss'] - auxiliary, evaluated[ENERGY_METRIC], rtol=3e-6, atol=1e-7):
        raise ValueError('training/evaluation normalized blind objectives differ')
    # Measure shared-encoder gradients without tuning weights or taking a step.
    (inp, pos, mask, et, pt, _, _, elo, ehi, _, _, plo, phi, _, _, _, ex, px) = model.data_preprocess(batches[0])
    outputs = model.model['transformer'](inp, mask, pos, ex, px)
    atoms = (~mask).sum(-1)
    spectra = reconstruct_balanced_spectra(outputs, atoms, protocol)
    objective = [normalized_spectral_error(spectra[f'blind_{task}'], target, atoms,
        protocol['tasks'][task]).mean() for task, target in (('edos', et*(ehi-elo)+elo), ('phdos', pt*(phi-plo)+plo))]
    if cfg.spectral_shape_aux:
        from model.losses import nonzero_sumnorm_klw_loss
        for i, (task, target) in enumerate((('edos', et*(ehi-elo)+elo), ('phdos', pt*(phi-plo)+plo))):
            objective[i] = objective[i] + nonzero_sumnorm_klw_loss(outputs[task], target).mean()
    parameters = tuple(model.model['transformer'].encoder.parameters())
    gradients = [torch.autograd.grad(value, parameters, retain_graph=i == 0, allow_unused=True)
                 for i, value in enumerate(objective)]
    norms = [sum(float(g.detach().double().square().sum()) for g in values if g is not None)**.5
             for values in gradients]
    dot = sum(float(a.detach().double().mul(b.detach().double()).sum())
              for a, b in zip(*gradients) if a is not None and b is not None)
    cosine = dot / (norms[0]*norms[1]) if min(norms) > 0 else None
    sources = {str(p.relative_to(ROOT)): file_sha256(p) for directory in ('model', 'utils')
               for p in (ROOT / directory).rglob('*.py')}
    for name in ('tools/train.py', 'tools/checks/check_spectral_balance.py',
                 'tools/eval/spectral_balance_prepare.py', 'tools/eval/lg_valid_export.py'):
        sources[name] = file_sha256(ROOT / name)
    result = {'schema': 'joint-spectral-balance-preflight-v1', 'passed': True,
        'configuration': asdict(cfg), 'model_contract': comparison_contract(cfg, params),
        'initialization': initialization, 'preparation_sha256': file_sha256(preparation),
        'source_sha256': sources, 'assembled_configuration': assembly,
        'scope': 'two disposable updates plus lr=0 objective check; no experiment checkpoint or test evaluation',
        'gpu': torch.cuda.get_device_name(), 'batch_sizes': [len(batch[0]) for batch in batches],
        'batch_atoms': [int(batch[0].ne(0).sum()) for batch in batches],
        'update_seconds': timings, 'peak_allocated_mib': peak, 'losses': losses,
        'loss_evaluation_difference': abs(before['loss'] - auxiliary - evaluated[ENERGY_METRIC]),
        'shape_auxiliary_at_consistency_check': auxiliary,
        'shared_encoder_gradient_diagnostic': {'edos_norm': norms[0], 'phdos_norm': norms[1],
            'cosine': cosine, 'scope': 'one train batch after two disposable updates; not a causal or global balance claim'},
        'full_training_started': False}
    out.mkdir(parents=True, exist_ok=False)
    for name in sources:
        target = out / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (out / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', required=True, help='report.json from spectral_balance_prepare.py')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = check(input_path(args.preparation), input_path(args.out))
    print(json.dumps({key: result[key] for key in ('passed', 'gpu', 'update_seconds', 'peak_allocated_mib',
        'loss_evaluation_difference', 'shared_encoder_gradient_diagnostic')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
