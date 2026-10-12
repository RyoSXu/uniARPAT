"""No-training diagnosis of the five trained arms on one fixed train subset.

Every arm is rebuilt from its own saved configuration and its own best
checkpoint, in eval mode, with no parameter update and no test split access. On
the same fixed train subset the tool reports blind/oracle eDOS and phDOS
R2/MAE/failure rate, the per-block residual strength and encoder output scale,
and the prediction difference of same-composition pairs against their target
difference. Train-subset numbers are listed next to the previously exported
valid numbers of the very same checkpoints so the two splits stay separate.

The subset is a seeded sample of train rows; its material IDs, row indices and
digest are written out so later runs can reuse exactly the same materials.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from archive.code.utils.pair_aux_batches import composition_keys
from tools.eval.lg_valid_export import load_run, select_best
from tools.train import evaluate_split
from utils.builder import ConfigBuilder
from utils.crystal import atom_padding_mask
from utils.paths import repository_path

# label, run directory, stem of the exported valid per-sample CSV
GROUPS = (
    ('baseline', 'results/runs/ablation_m1_lg_baseline_q1v1_e35_s42', 'baseline'),
    ('a_staged', 'results/runs/ablation_m1_lg_s42sel20261005_k2_a_staged_s42', 'a_staged'),
    ('a_alternating', 'results/runs/ablation_m1_lg_s42sel20261005_k2_a_alternating_s42', 'a_alternating'),
    ('bp_staged', 'results/runs/ablation_m1_lg_s42sel20261005_k2_bp_staged_s42', 'bp_staged'),
    ('bp_alternating', 'results/runs/ablation_m1_lg_s42sel20261005_k2_bp_alternating_s42', 'bp_alternating'),
)
VALID_EXPORTS = 'results/runs/lg_baseline_q1v1_20261008/exports'
SUBSET_SEED = 20261008
SUBSET_SIZE = 2048
TASKS = ('edos', 'phdos')
MODES = ('blind', 'oracle')
# Spectral spaces compared between two materials. Shape space is sum-normalized
# and therefore scale free; blind and oracle carry the per-material totals.
SPACES = ('shape', 'blind', 'oracle')


def subset_rows(n_rows: int, size: int, seed: int) -> np.ndarray:
    """Deterministic subset of dataset rows, kept in row order."""
    if not 0 < size <= n_rows:
        raise ValueError(f'subset size {size} outside 1..{n_rows}')
    rows = np.random.default_rng(seed).choice(n_rows, size=size, replace=False)
    return np.sort(rows)


def spectral_summary(frame: pd.DataFrame) -> dict:
    """The reported aggregates: medians plus the R2<0 failure rate in percent."""
    summary = {}
    for mode in MODES:
        for task in TASKS:
            for metric in ('r2', 'mae'):
                summary[f'{mode}_{metric}_{task}_median'] = float(
                    frame[f'{mode}_{metric}_{task}'].median())
            summary[f'{mode}_fail_rate_{task}'] = float(
                (frame[f'{mode}_r2_{task}'] < 0).mean() * 100)
    return summary


def valid_reference(stem: str) -> dict:
    """Recompute the same aggregates from the exported valid per-sample rows."""
    frame = pd.read_csv(repository_path(f'{VALID_EXPORTS}/{stem}_samples.csv'))
    return spectral_summary(frame)


class StateProbe:
    """Per-block update strength and encoder scale, over valid atoms only.

    Each recorded entry keeps the block input and output of one forward pass;
    the residual strength is RMS(output - input) / RMS(input) accumulated over
    every valid atom slot of the probed batches.
    """

    def __init__(self, transformer):
        self.valid = None
        self.order = []
        self.pending = {}
        self.totals = {}
        self.handles = [transformer.register_forward_pre_hook(self._capture_mask),
                        transformer.register_forward_hook(self._end)]
        self.handles.append(transformer.encoder.register_forward_hook(
            self._state_hook('encoder'), with_kwargs=True))
        for name, module in self.blocks(transformer):
            self.order.append(name)
            self.handles.append(module.register_forward_hook(self._state_hook(name), with_kwargs=True))

    @staticmethod
    def blocks(transformer):
        encoder = transformer.encoder
        if getattr(encoder, 'local_layers', None) is not None:
            picked = []
            for kind, index in encoder.execution_order:
                layers = encoder.local_layers if kind == 'local' else encoder.global_layers
                picked.append((f'{kind}[{index}]', layers[index]))
            return picked
        return [(f'layer[{index}]', layer) for index, layer in enumerate(encoder.layers)]

    def _capture_mask(self, module, args):
        self.valid = ~atom_padding_mask(args[2], args[1])
        self.pending = {}

    def _end(self, module, args, output):
        self.flush()

    def _state_hook(self, name):
        def hook(module, args, kwargs, output):
            state = args[0] if args else kwargs['src']
            self.pending[name] = (state.detach(), output.detach())
        return hook

    def flush(self):
        if self.valid is None:
            raise ValueError('probe has no captured mask')
        for name, (state, output) in self.pending.items():
            picked_in = state[self.valid]
            picked_delta = (output - state)[self.valid]
            totals = self.totals.setdefault(name, np.zeros(4))
            totals += np.array([
                float(picked_in.float().pow(2).sum()),
                float(output[self.valid].float().pow(2).sum()),
                float(picked_delta.float().pow(2).sum()),
                int(picked_in.numel())])
        self.pending = {}

    def report(self):
        entries = {}
        for name in ['encoder', *self.order]:
            if name not in self.totals:
                continue
            in_sq, out_sq, delta_sq, count = self.totals[name]
            if count <= 0:
                raise ValueError(f'probe entry {name} saw no valid atom')
            entries[name] = {
                'input_rms': float(np.sqrt(in_sq / count)),
                'output_rms': float(np.sqrt(out_sq / count)),
                'delta_rms': float(np.sqrt(delta_sq / count)),
                'relative_update': float(np.sqrt(delta_sq / in_sq)) if in_sq > 0 else None,
                'valid_atom_slots': int(count),
            }
        return entries

    def close(self):
        for handle in self.handles:
            handle.remove()


def rankdata(values):
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    return ranks


def same_composition_pairs(keys):
    """All within-composition pairs of evaluated positions, plus group sizes."""
    groups = defaultdict(list)
    for position, key in enumerate(keys):
        groups[key].append(position)
    pairs = [pair for members in groups.values() if len(members) > 1
             for pair in combinations(sorted(members), 2)]
    return pairs, groups


def pair_statistics(predictions: dict, targets: dict, keys) -> dict:
    """Same-composition pair differences: prediction against target.

    ``predictions``/``targets`` carry one array per spectral space. Total
    variation is 0.5 * L1, which is bounded by 1 in shape space; in blind/oracle
    space it keeps the physical units of the task. A large target difference
    with a small predicted difference means the prediction barely separates
    those structures; it is not by itself evidence of better or worse spectra.
    """
    pairs, groups = same_composition_pairs(keys)
    result = {'pairs': len(pairs),
              'composition_groups': int(sum(1 for members in groups.values() if len(members) > 1)),
              'largest_group_sizes': sorted(
                  (len(members) for members in groups.values() if len(members) > 1),
                  reverse=True)[:10]}
    if not pairs:
        return result
    left = np.array([pair[0] for pair in pairs])
    right = np.array([pair[1] for pair in pairs])
    for space in SPACES:
        if space not in predictions:
            continue
        target = 0.5 * np.abs(targets[space][left] - targets[space][right]).sum(-1)
        predicted = 0.5 * np.abs(predictions[space][left] - predictions[space][right]).sum(-1)
        entry = {
            'target_tv_median': float(np.median(target)),
            'predicted_tv_median': float(np.median(predicted)),
            'predicted_over_target': (float(np.median(predicted) / np.median(target))
                                      if np.median(target) > 0 else None),
        }
        if len(pairs) > 2:
            entry['predicted_vs_target_spearman'] = float(np.corrcoef(
                rankdata(predicted), rankdata(target))[0, 1])
        result[space] = entry
    return result


def composition_keys_by_position(dataset, order):
    """Absolute and reduced composition keys of the evaluated materials."""
    absolute_keys, reduced_keys = [], []
    for row in order:
        elements, _ = dataset.crystals[int(dataset.structure_indices[row])]
        absolute, reduced = composition_keys(np.asarray(elements), sentinel_count=0)
        absolute_keys.append(absolute)
        reduced_keys.append(reduced)
    return {'absolute': absolute_keys, 'reduced': reduced_keys}


def diagnose(out_dir: Path, subset_size: int, subset_seed: int):
    if out_dir.exists():
        raise FileExistsError(f'refusing existing report directory: {out_dir}')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report = {
        'scope': 'no-training diagnosis in eval mode on one fixed train subset; '
                 'train numbers are not valid numbers and no test split is read',
        'subset': {'size': subset_size, 'seed': subset_seed,
                   'selection': 'numpy.default_rng(seed).choice(train_rows, size, '
                                'replace=False), sorted into train row order'},
        'spaces': {
            'shape': 'sum-normalized spectra; scale free, bounded total variation',
            'blind': 'structure-only scale reconstruction in physical units',
            'oracle': 'target-scale reconstruction in physical units',
        },
        'groups': {},
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    subset = None
    keys_recorded = None
    pair_index = None
    for label, run_dir_name, valid_stem in GROUPS:
        run_dir = repository_path(run_dir_name)
        saved, params, contract = load_run(run_dir)
        picked = select_best(run_dir, contract['selection_metric'], saved)
        checkpoint = picked.pop('checkpoint')

        builder = ConfigBuilder(**saved['config'])
        model = builder.get_model()
        model.model['transformer'].load_state_dict(checkpoint['model'], strict=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        if subset is None:
            full = builder.get_dataset(split='train')
            rows = subset_rows(len(full), subset_size, subset_seed)
            all_ids = [str(value) for value in full.material_ids]
            all_atoms = np.asarray(full.atom_counts, dtype=np.int64)
            subset = {'rows': rows,
                      'material_ids': [all_ids[int(row)] for row in rows],
                      'n_atoms': [int(all_atoms[int(row)]) for row in rows]}
        loader = builder.get_dataloader(split='train', dos_minmax=True, batch_size=32,
                                        dos_sumnorm=True, choice=[int(row) for row in subset['rows']])
        dataset = loader.dataset
        order = list(loader.sampler)
        ids_by_position = [subset['material_ids'][int(index)] for index in order]
        keys_by_position = composition_keys_by_position(dataset, order)
        if keys_recorded is None:
            keys_recorded = keys_by_position
            pair_index = {grouping: [list(pair) for pair in same_composition_pairs(keys)[0]]
                          for grouping, keys in keys_by_position.items()}
        elif keys_by_position != keys_recorded:
            raise ValueError(f'composition keys differ between groups: {label}')
        if len(set(ids_by_position)) != len(ids_by_position):
            raise ValueError(f'duplicate material IDs in the fixed subset: {label}')

        probe = StateProbe(model.model['transformer'])
        metrics = evaluate_split(model, loader, sumnorm=True, return_sample_level=True,
                                 return_spectra=True, sample_ids=ids_by_position)
        probe.close()
        spectra = metrics.pop('sample_spectra')
        frame = metrics.pop('sample_df')
        if len(frame) != len(order):
            raise ValueError(f'evaluated rows differ from the fixed subset: {label}')
        if not np.isfinite(frame.select_dtypes('number').to_numpy()).all():
            raise ValueError(f'non-finite train-subset metrics: {label}')

        frame.to_csv(out_dir / f'train_subset_{label}.csv', index=False)
        for name, values in spectra.items():
            np.save(out_dir / f'spectra_{label}_{name}.npy', values)
        pairs = {}
        for grouping, keys in keys_by_position.items():
            pairs[grouping] = {}
            for task in TASKS:
                physical = spectra[f'tgt_{task}']
                pairs[grouping][task] = pair_statistics(
                    {'shape': spectra[f'pred_shape_{task}'],
                     'blind': spectra[f'pred_blind_{task}'],
                     'oracle': spectra[f'pred_oracle_{task}']},
                    {'shape': spectra[f'tgt_shape_{task}'],
                     'blind': physical, 'oracle': physical},
                    keys)

        reference = valid_reference(valid_stem)
        train_stats = spectral_summary(frame)
        report['groups'][label] = {
            'run_dir': run_dir_name,
            'tag': picked['tag'],
            'best_epoch': int(picked['row'].epoch),
            'best_score': float(picked['row'].balanced_score),
            'contract_schema': contract['schema'],
            'init_protocol': contract['init_protocol'],
            'samples': len(frame),
            'train_subset': train_stats,
            'valid_exported': reference,
            'train_minus_valid': {key: train_stats[key] - reference[key] for key in train_stats},
            'encoder_probe': {
                'definition': 'RMS over valid atom slots of one forward pass; '
                              'relative_update = RMS(output - input) / RMS(input)',
                'blocks': probe.report(),
            },
            'same_composition_pairs': pairs,
        }
        del probe, model, metrics, spectra, frame

    subset_digest = hashlib.sha256('\n'.join(sorted(subset['material_ids'])).encode()).hexdigest()
    pairs_path = out_dir / 'subset_pairs.json'
    pairs_path.write_text(json.dumps({
        'unit': 'evaluated position of the fixed subset',
        'counts': {name: len(value) for name, value in pair_index.items()},
        'pairs': pair_index}) + '\n')
    report['subset'].update({
        'rows': [int(row) for row in subset['rows']],
        'material_ids': subset['material_ids'],
        'n_atoms': subset['n_atoms'],
        'material_id_sha256': subset_digest,
        'pair_counts': {name: len(value) for name, value in pair_index.items()},
        'pairs_file': pairs_path.name,
        'pairs_sha256': hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
    })
    (out_dir / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--subset-size', type=int, default=SUBSET_SIZE)
    parser.add_argument('--subset-seed', type=int, default=SUBSET_SEED)
    args = parser.parse_args()
    report = diagnose(args.out_dir, args.subset_size, args.subset_seed)
    printable = json.loads(json.dumps(report, default=float))
    for key in ('rows', 'material_ids', 'n_atoms'):
        printable['subset'].pop(key, None)
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
