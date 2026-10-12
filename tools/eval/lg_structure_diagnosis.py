"""Read-only structure diagnosis of saved baseline/B+ G6 checkpoints.

No optimizer, parameter update, test loader or training dispatch. Pair errors
measure signed spectral differences, not merely response amplitude. Inference
interventions are sensitivity tests on trained weights, not retrained ablations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader, Subset

from tools.eval.lg_train_subset_diagnosis import (
    StateProbe, composition_keys_by_position, same_composition_pairs, spectral_summary,
)
from tools.eval.lg_valid_export import load_run, select_best
from tools.train import evaluate_split
from utils.bucket_batch import crystal_collate
from utils.builder import ConfigBuilder
from utils.crystal import CrystalGeometry, atom_padding_mask
from utils.paths import file_sha256, repository_path
from utils.periodic_geometry import build_periodic_records
from utils.periodic_neighbors import g2_indegrees, g2_quintic_cutoff, g2_rbf_features, g2_rbf_params
from utils.zproj_alignment import state_hash

RUNS = {
    'baseline35': 'ablation_m1_lg_baseline_q1v1_e35_s42',
    'bp35': 'ablation_m1_lgd_s42g6sel20261008_k2_g6_bp_staged_s42',
    'baseline100': 'ablation_m1_lg_baseline_q1v1_e100_s42',
    'bp100': 'ablation_m1_lgd_s42g6e10020261008_k2_g6_bp_staged_s42',
}
PRIOR_SUBSET = ROOT / 'results/runs/lg_train_subset_diagnosis_20261008/report.json'
TASKS = ('edos', 'phdos')
MODES = ('blind', 'oracle')


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False) + '\n', encoding='utf-8')


def correlation(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    ok = np.isfinite(left) & np.isfinite(right)
    if ok.sum() < 3 or np.ptp(left[ok]) == 0 or np.ptp(right[ok]) == 0:
        return None
    return float(spearmanr(left[ok], right[ok]).statistic)


def pair_errors(prediction, target, keys, ids):
    """Compare signed pair differences; zero-response reference has skill zero."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or len(keys) != len(target) or len(ids) != len(target):
        raise ValueError('pair arrays and labels must align')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('non-finite pair spectra')
    pairs, _ = same_composition_pairs(keys)
    rows = []
    for i, j in pairs:
        actual = target[i].astype(np.float64) - target[j]
        predicted = prediction[i].astype(np.float64) - prediction[j]
        amplitude = .5 * np.abs(actual).sum()
        error = .5 * np.abs(predicted - actual).sum()
        denominator = np.linalg.norm(actual) * np.linalg.norm(predicted)
        rows.append(dict(i=i, j=j, left_id=ids[i], right_id=ids[j],
                         composition=repr(keys[i]), target_tv=float(amplitude),
                         predicted_tv=float(.5 * np.abs(predicted).sum()),
                         difference_error=float(error),
                         skill=float(1 - error / amplitude) if amplitude > 1e-10 else np.nan,
                         direction_cosine=float(predicted @ actual / denominator)
                         if denominator > 1e-12 else np.nan))
    return pd.DataFrame(rows, columns=['i', 'j', 'left_id', 'right_id', 'composition',
                                     'target_tv', 'predicted_tv', 'difference_error',
                                     'skill', 'direction_cosine'])


def summarize_pairs(frame):
    result = {'pairs': len(frame), 'composition_groups': int(frame.composition.nunique())}
    for name in ('target_tv', 'predicted_tv', 'difference_error', 'skill', 'direction_cosine'):
        finite = frame[name].dropna()
        result[name + '_median'] = float(finite.median()) if len(finite) else None
    result['positive_skill_fraction'] = (float((frame.skill.dropna() > 0).mean())
                                         if frame.skill.notna().any() else None)
    return result


def paired_group_interval(left, right):
    """Resample composition groups; dependent pairs are never treated as iid."""
    cols = ['left_id', 'right_id', 'composition']
    if not left[cols].equals(right[cols]):
        raise ValueError('pair identities differ between models')
    values = right.assign(delta=right.difference_error - left.difference_error)
    grouped = values.groupby('composition').delta.mean().to_numpy()
    if not len(grouped):
        return {'groups': 0}
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(grouped, len(grouped), replace=True).mean()
                     for _ in range(2000)])
    return {'groups': len(grouped), 'mean_group_error_delta': float(grouped.mean()),
            'ci95': np.percentile(boot, [2.5, 97.5]).tolist(),
            'scope': 'composition-group mean signed-difference error; single training seed'}


def loader(dataset, rows, batch_size=32):
    return DataLoader(Subset(dataset, [int(i) for i in rows]), batch_size=batch_size,
                      shuffle=False, num_workers=0, collate_fn=crystal_collate)


def geometry_table(dataset, rows, device):
    """Label-free radial support proxy, computed on the current complete cells."""
    keys = composition_keys_by_position(dataset, rows)
    ids = [str(dataset.material_ids[int(i)]) for i in rows]
    result = []
    started = time.monotonic()
    for start in range(0, len(rows), 32):
        indices = rows[start:start + 32]
        crystals = [dataset.crystals[int(dataset.structure_indices[int(i)])] for i in indices]
        width = max(len(z) for z, _ in crystals)
        cells = torch.stack([g.cell for _, g in crystals]).to(device)
        frac = torch.zeros(len(crystals), width, 3, device=device)
        mask = torch.ones(len(crystals), width, dtype=torch.bool, device=device)
        for b, (z, g) in enumerate(crystals):
            frac[b, :len(z)] = g.frac_coords.to(device)
            mask[b, :len(z)] = False
        with torch.no_grad():
            rec = build_periodic_records(CrystalGeometry(cells, frac), mask, radius=6.)
            cut = g2_quintic_cutoff(rec.distances, 6.)
            centers, bandwidth = g2_rbf_params(6., 16, device=device, dtype=cells.dtype)
            radial = g2_rbf_features(rec.distances, 6., centers, bandwidth)
            radial_sum = cells.new_zeros(len(crystals), 16).index_add_(
                0, rec.batch, radial * cut[:, None]).cpu().numpy()
            degree = g2_indegrees(rec.batch, rec.dst, len(crystals), width).cpu().numpy()
            batch, dst, distance, weights = [x.cpu().numpy() for x in
                                             (rec.batch, rec.dst, rec.distances, cut)]
        for b, (z, geometry) in enumerate(crystals):
            select = batch == b
            d, c, receivers = distance[select], weights[select], dst[select]
            effective = np.bincount(receivers, weights=c, minlength=len(z))
            inner = np.bincount(receivers, weights=c * (d < 5), minlength=len(z))
            square = np.bincount(receivers, weights=c * c, minlength=len(z))
            inner_square = np.bincount(receivers, weights=c * c * (d < 5), minlength=len(z))
            pair_mass = .5 * np.maximum(effective ** 2 - square, 0).sum()
            inner_mass = .5 * np.maximum(inner ** 2 - inner_square, 0).sum()
            n = len(z)
            position = start + b
            row = dict(material_id=ids[position], dataset_row=int(indices[b]), n_atoms=n,
                       composition=repr(keys['reduced'][position]),
                       absolute_composition=repr(keys['absolute'][position]),
                       chemical_system=','.join(str(int(x)) for x in torch.unique(z)),
                       volume_per_atom=float(torch.linalg.det(geometry.cell).abs()) / n,
                       degree_mean=float(degree[b, :n].mean()),
                       degree_std=float(degree[b, :n].std()),
                       effective_degree=float(effective.mean()),
                       outer_edge_fraction=float((d >= 5).mean()) if len(d) else 0.,
                       outer_cutoff_fraction=float(c[d >= 5].sum() / c.sum()) if c.sum() else 0.,
                       outer_pair_weight_fraction=float(1 - inner_mass / pair_mass) if pair_mass else 0.)
            row.update({f'rbf{k}': float(value / n) for k, value in enumerate(radial_sum[b])})
            result.append(row)
        if start == 0 or (start + 32) % 2048 == 0:
            print(json.dumps({'geometry_rows': min(start + 32, len(rows)), 'total': len(rows),
                              'seconds': round(time.monotonic() - started, 2)}), flush=True)
    return pd.DataFrame(result)


def attach_support(train, query):
    """Train-only scaling/reference, exact reduced-composition neighbors, no self."""
    columns = [f'rbf{k}' for k in range(16)] + ['degree_std', 'volume_per_atom']
    reference = train[columns].to_numpy(float)
    scale = np.maximum(reference.std(axis=0), 1e-8)
    reference = reference / scale
    lookup = {key: np.asarray(rows) for key, rows in train.groupby('composition').indices.items()}
    systems = train.chemical_system.value_counts()
    ids = train.material_id.to_numpy()
    reference_ids = set(ids)
    output = query.copy()
    distances, counts, nearest = [], [], []
    for _, row in query.iterrows():
        candidates = lookup.get(row.composition, np.array([], dtype=int))
        candidates = candidates[ids[candidates] != row.material_id]
        counts.append(len(candidates))
        if not len(candidates):
            distances.append(np.nan)
            nearest.append('')
        else:
            diff = reference[candidates] - row[columns].to_numpy(float) / scale
            d = np.sqrt((diff * diff).mean(-1))
            winner = int(np.argmin(d))
            distances.append(float(d[winner]))
            nearest.append(str(ids[candidates[winner]]))
    output['same_composition_train_count'] = counts
    output['radial_support_distance'] = distances
    output['nearest_train_id'] = nearest
    output['same_system_train_count'] = [int(systems.get(row.chemical_system, 0)) -
                                        int(row.material_id in reference_ids)
                                        for _, row in query.iterrows()]
    return output


class InformationProbe:
    """Permutation-invariant token mean/second moment, NOT complete information."""
    def __init__(self, transformer):
        self.transformer = transformer
        self.features = defaultdict(list)
        self.handles = [transformer.register_forward_pre_hook(self.begin)]
        self.handles.append(transformer.encoder.register_forward_pre_hook(
            self.encoder_input, with_kwargs=True))
        for name, module in StateProbe.blocks(transformer):
            self.handles.append(module.register_forward_hook(self.atom_hook(name), with_kwargs=True))
        if getattr(transformer.encoder, 'local_layers', None) is not None:
            for i, layer in enumerate(transformer.encoder.local_layers):
                self.handles.append(layer.message.local_input.register_forward_hook(
                    self.atom_hook(f'local[{i}].projected'), with_kwargs=True))
                self.handles.append(layer.message.local_output.register_forward_pre_hook(
                    self.scalar_readout(f'local[{i}].scalar')))
        self.handles.append(transformer.decoder.register_forward_hook(self.decoder_hook))

    def begin(self, module, args):
        self.mask = atom_padding_mask(args[2], args[1])
        self.decoder_call = 0

    def record(self, name, value, mask):
        valid = (~mask).to(value.dtype)[..., None]
        count = valid.sum(1).clamp_min(1)
        mean = (value * valid).sum(1) / count
        second = (value.square() * valid).sum(1) / count
        self.features[name].append(torch.cat((mean, second.sqrt()), -1).detach().float().cpu().numpy())

    def encoder_input(self, module, args, kwargs):
        self.record('input', args[0] if args else kwargs['src'], self.mask)

    def atom_hook(self, name):
        def hook(module, args, kwargs, output):
            self.record(name, output, self.mask)
        return hook

    def scalar_readout(self, name):
        def hook(module, args):
            self.record(name, args[0], self.mask)
        return hook

    def decoder_hook(self, module, args, output):
        name = TASKS[self.decoder_call]
        self.decoder_call += 1
        value = output[0]
        self.record('decoder.' + name, value, torch.zeros(value.shape[:2], dtype=torch.bool,
                                                          device=value.device))

    def close(self):
        for handle in self.handles:
            handle.remove()


@contextmanager
def intervention(transformer, mode):
    """Temporary inference-only changes, restored even when evaluation raises."""
    if mode not in ('identity', 'local_90pct', 'local_off', 'rbf_off', 'high_order_off', 'angular_off'):
        raise ValueError('unknown inference intervention')
    handles = []
    def local_hook(module, args, output):
        factor = .9 if mode == 'local_90pct' else 0.
        return args[0] + factor * (output - args[0])
    def block_hook(module, args):
        state, geometry, harmonics, high = args
        if mode == 'rbf_off':
            geometry = replace(geometry, radial=torch.zeros_like(geometry.radial))
        if mode in ('high_order_off', 'angular_off'):
            high = False
        if mode == 'angular_off':
            harmonics = torch.cat((harmonics[:, :1], torch.zeros_like(harmonics[:, 1:])), -1)
        return state, geometry, harmonics, high
    try:
        for layer in transformer.encoder.local_layers:
            if mode in ('local_90pct', 'local_off'):
                handles.append(layer.register_forward_hook(local_hook))
            elif mode != 'identity':
                if layer.message.route != 'B+':
                    raise ValueError('branch interventions require B+')
                for block in layer.message.blocks:
                    handles.append(block.register_forward_pre_hook(block_hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def feature_pairs(features, pairs):
    result = {}
    for stage, chunks in features.items():
        values = np.concatenate(chunks).astype(np.float64)
        i, j = pairs.i.to_numpy(int), pairs.j.to_numpy(int)
        denominator = np.sqrt(.5 * ((values[i] ** 2).mean(-1) + (values[j] ** 2).mean(-1)))
        distance = np.sqrt(((values[i] - values[j]) ** 2).mean(-1)) / np.maximum(denominator, 1e-12)
        result[stage] = {'relative_pair_distance_median': float(np.median(distance)) if len(i) else None,
                         'distance_vs_target_tv_spearman': correlation(distance, pairs.target_tv),
                         'description': 'mean and per-channel RMS only; no cross-model feature alignment'}
    return result


def strata(frame, metadata, target_shape, thresholds):
    if list(frame.material_id) != list(metadata.material_id):
        raise ValueError('strata metadata must follow evaluated IDs')
    entropy = -(target_shape * np.log(np.maximum(target_shape, 1e-30))).sum(-1)
    conditions = {
        'all': np.ones(len(frame), dtype=bool),
        'composition_seen': metadata.same_composition_train_count.to_numpy() > 0,
        'composition_unseen': metadata.same_composition_train_count.to_numpy() == 0,
        'atoms_le8': metadata.n_atoms.to_numpy() <= 8,
        'atoms_9_32': metadata.n_atoms.between(9, 32).to_numpy(),
        'atoms_gt32': metadata.n_atoms.to_numpy() > 32,
        'degree_high': metadata.degree_mean.to_numpy() > thresholds['degree_q75'],
        'radial_support_far': metadata.radial_support_distance.to_numpy() > thresholds['radial_q75'],
        'radial_support_near': metadata.radial_support_distance.to_numpy() <= thresholds['radial_q25'],
        'target_sharp': entropy < thresholds['entropy_q25'],
        'target_broad': entropy > thresholds['entropy_q75'],
    }
    return {name: {'n': int(selected.sum()), **spectral_summary(frame.loc[selected])}
            for name, selected in conditions.items() if selected.any()}


def structure_matches(dataset, rows, pairs, spectra):
    """Tolerant matches identify provenance questions, never automatic bad labels."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    matcher = StructureMatcher(ltol=.2, stol=.3, angle_tol=5, primitive_cell=True,
                               scale=True, attempt_supercell=False)
    structures = {}
    def get(position):
        if position not in structures:
            z, g = dataset.crystals[int(dataset.structure_indices[int(rows[position])])]
            structures[position] = Structure(g.cell.numpy(), z.numpy(), g.frac_coords.numpy())
        return structures[position]
    matches = []
    for row in pairs.itertuples():
        if matcher.fit(get(row.i), get(row.j)):
            matches.append({'left_id': row.left_id, 'right_id': row.right_id,
                            'edos_target_tv': float(row.target_tv),
                            'phdos_target_tv': float(.5 * np.abs(
                                spectra['tgt_shape_phdos'][row.i] -
                                spectra['tgt_shape_phdos'][row.j]).sum())})
    return {'settings': {'ltol': .2, 'stol': .3, 'angle_tol': 5, 'scale': True},
            'matches': matches,
            'limitation': 'tolerant structural equivalence; spin/smearing/calculation metadata '
                          'are not contained in the active packed cache; not evidence of label errors'}


def diagnose(out, smoke=False):
    if out.exists():
        raise FileExistsError(f'refusing existing output: {out}')
    out.mkdir(parents=True)
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved, _, _ = load_run(ROOT / 'results/runs' / RUNS['bp35'])
    builder = ConfigBuilder(**saved['config'])
    datasets = {split: builder.get_dataset(split=split, dos_minmax=True, dos_sumnorm=True)
                for split in ('train', 'valid')}
    if any(ds.augment for ds in datasets.values()):
        raise ValueError('diagnosis requires unaugmented data')
    prior = json.loads(PRIOR_SUBSET.read_text())['subset']
    train_by_id = {str(mid): i for i, mid in enumerate(datasets['train'].material_ids)}
    rows = {'train': [train_by_id[mid] for mid in prior['material_ids']],
            'valid': list(range(len(datasets['valid'])))}
    if smoke:
        rows = {split: indices[:64] for split, indices in rows.items()}
    manifests = [ROOT / 'data/crystal_q1_v1/manifest.json']
    manifest = json.loads(manifests[0].read_text())
    # Verify only train/valid payloads. Do not hash or open test arrays.
    verified_files = {}
    for name, entry in manifest['files'].items():
        if name.startswith(('train/', 'valid/')):
            value = file_sha256(manifests[0].parent / name)
            if value != entry['sha256']:
                raise ValueError('dataset digest mismatch: ' + name)
            verified_files[name] = value
    report = {'scope': 'read-only saved-checkpoint diagnosis; train/valid only; no training',
              'smoke': smoke, 'seed': 42, 'device': str(device),
              'data_manifest_sha256': file_sha256(manifests[0]),
              'verified_train_valid_files': verified_files,
              'source_sha256': file_sha256(Path(__file__)), 'groups': {},
              'splits': {split: {'rows': ids, 'material_ids': [str(datasets[split].material_ids[i])
                                                             for i in ids]}
                         for split, ids in rows.items()}}
    dump(out / 'manifest.json', report)
    geometry = {}
    train_rows = list(range(min(256, len(datasets['train'])) if smoke else len(datasets['train'])))
    for split, indices in (('train', train_rows), ('valid', rows['valid'])):
        geometry[split] = geometry_table(datasets[split], indices, device)
    reference = geometry['train']
    for split in geometry:
        geometry[split] = attach_support(reference, geometry[split])
        geometry[split].to_csv(out / f'geometry_{split}.csv', index=False)
    # Smoke train rows need not be among the 256 radial-reference rows.
    if smoke:
        train_eval = geometry_table(datasets['train'], rows['train'], device)
        train_eval = attach_support(reference, train_eval)
    else:
        train_eval = geometry['train'].set_index('material_id').loc[
            report['splits']['train']['material_ids']].reset_index()
    meta = {'train': train_eval, 'valid': geometry['valid']}
    train_shape = datasets['train'].edos_tgtdos.numpy()
    train_entropy = -(train_shape * np.log(np.maximum(train_shape, 1e-30))).sum(-1)
    finite_support = geometry['train'].radial_support_distance.dropna()
    thresholds = {'degree_q75': float(reference.degree_mean.quantile(.75)),
                  'radial_q25': float(finite_support.quantile(.25)) if len(finite_support) else 0.,
                  'radial_q75': float(finite_support.quantile(.75)) if len(finite_support) else 0.,
                  'entropy_q25': float(np.quantile(train_entropy, .25)),
                  'entropy_q75': float(np.quantile(train_entropy, .75))}
    report['strata_thresholds_train_only'] = thresholds
    report['radial_proxy'] = '16 cutoff-weighted RBF sums/atom, degree std, volume/atom; '
    report['radial_proxy'] += 'standardized by train std; nearest same reduced composition excluding self; no angular completeness claim'
    report['cutoff'] = {split: {key: float(geometry[split][key].median()) for key in (
        'outer_edge_fraction', 'outer_cutoff_fraction', 'outer_pair_weight_fraction')}
                        for split in geometry}
    pair_tables = {}
    for label, name in RUNS.items():
        if smoke and label.endswith('100'):
            continue
        run = ROOT / 'results/runs' / name
        saved, _, contract = load_run(run)
        if saved['cli']['data_contract']['manifest_sha256'] != report['data_manifest_sha256']:
            raise ValueError('checkpoint refers to a different data revision')
        picked = select_best(run, contract['selection_metric'], saved)
        checkpoint_path = run / 'checkpoint_best.pth'
        checkpoint_digest = file_sha256(checkpoint_path)
        build = ConfigBuilder(**saved['config'])
        model = build.get_model()
        transformer = model.model['transformer']
        transformer.load_state_dict(picked['checkpoint']['model'], strict=True)
        del picked['checkpoint']
        model.to(device)
        transformer.eval()
        before = state_hash(transformer.state_dict())
        group = {'run': str(run.relative_to(ROOT)), 'epoch': int(picked['row'].epoch),
                 'checkpoint_sha256': checkpoint_digest, 'contract': contract,
                 'splits': {}, 'interventions': {}}
        for split in ('train', 'valid'):
            ids = report['splits'][split]['material_ids']
            data = loader(datasets[split], rows[split])
            keys = composition_keys_by_position(datasets[split], rows[split])
            probe = InformationProbe(transformer)
            tick = time.monotonic()
            try:
                measured = evaluate_split(model, data, sumnorm=True, return_sample_level=True,
                                          return_spectra=True, sample_ids=ids)
            finally:
                probe.close()
            frame, spectra = measured['sample_df'], measured['sample_spectra']
            if not np.isfinite(frame.select_dtypes('number').to_numpy()).all():
                raise ValueError('non-finite evaluated sample metrics')
            frame.to_csv(out / f'{label}_{split}_samples.csv', index=False)
            np.savez(out / f'{label}_{split}_spectra.npz', material_ids=np.asarray(ids), **spectra)
            entry = {'n': len(frame), 'metrics': spectral_summary(frame), 'pairs': {},
                     'seconds': time.monotonic() - tick,
                     'strata': strata(frame, meta[split], spectra['tgt_shape_edos'], thresholds)}
            for grouping in ('absolute', 'reduced'):
                entry['pairs'][grouping] = {}
                for task in TASKS:
                    pairs = pair_errors(spectra[f'pred_shape_{task}'], spectra[f'tgt_shape_{task}'],
                                        keys[grouping], ids)
                    pairs.to_csv(out / f'{label}_{split}_{grouping}_{task}_pairs.csv', index=False)
                    pair_tables[label, split, grouping, task] = pairs
                    entry['pairs'][grouping][task] = summarize_pairs(pairs)
                    if grouping == 'reduced' and task == 'edos':
                        entry['feature_pairs'] = feature_pairs(probe.features, pairs)
                        if label == 'baseline35':
                            entry['structure_matches'] = structure_matches(datasets[split], rows[split],
                                                                          pairs, spectra)
            group['splits'][split] = entry
            print(json.dumps({'evaluated': label, 'split': split, 'n': len(ids),
                              'seconds': round(entry['seconds'], 2)}), flush=True)
            del probe, measured, spectra, frame
        if label.startswith('bp'):
            # Fixed label-independent valid sample, shared by both trained B+ models.
            choices = np.sort(np.random.default_rng(42).choice(
                rows['valid'], min(256, len(rows['valid'])), replace=False)).tolist()
            ids = [str(datasets['valid'].material_ids[i]) for i in choices]
            group['intervention_ids'] = ids
            control = None
            repeat = None
            for mode in ('identity', 'local_90pct', 'local_off', 'rbf_off', 'high_order_off', 'angular_off'):
                with intervention(transformer, mode):
                    measured = evaluate_split(model, loader(datasets['valid'], choices), sumnorm=True,
                                              return_sample_level=True, return_spectra=True, sample_ids=ids)
                frame, spectra = measured['sample_df'], measured['sample_spectra']
                if mode == 'identity':
                    control = {key: value.copy() for key, value in spectra.items()}
                    repeat = evaluate_split(model, loader(datasets['valid'], choices), sumnorm=True,
                                            return_spectra=True)['sample_spectra']
                effect = {task: float(np.median(.5 * np.abs(spectra[f'pred_shape_{task}'] -
                              control[f'pred_shape_{task}']).sum(-1))) for task in TASKS}
                group['interventions'][mode] = {'n': len(frame), 'metrics': spectral_summary(frame),
                                                'prediction_shape_tv_vs_identity': effect}
                frame.to_csv(out / f'{label}_intervention_{mode}.csv', index=False)
            with intervention(transformer, 'identity'):
                restored = evaluate_split(model, loader(datasets['valid'], choices), sumnorm=True,
                                          return_spectra=True)['sample_spectra']
            drift = max(float(np.max(np.abs(restored[key] - control[key]))) for key in control)
            for key in control:
                # Physical spectra and unit-sum shapes have different units.
                # GPU index_add can change last bits even in identity repeats.
                np.testing.assert_allclose(repeat[key], control[key], rtol=2e-5, atol=2e-6,
                                           err_msg='identity repeat: ' + key)
                np.testing.assert_allclose(restored[key], control[key], rtol=2e-5, atol=2e-6,
                                           err_msg='intervention restoration: ' + key)
            group['intervention_restore_max_abs'] = drift
            group['identity_repeat_max_abs'] = max(float(np.max(np.abs(repeat[key] - control[key])))
                                                   for key in control)
            group['restore_tolerance'] = {'rtol': 2e-5, 'atol': 2e-6}
        after = state_hash(transformer.state_dict())
        if before != after or file_sha256(checkpoint_path) != checkpoint_digest:
            raise ValueError('model/checkpoint mutated during diagnosis')
        group['state_unchanged'] = True
        report['groups'][label] = group
        dump(out / 'report.partial.json', report)
        del model, transformer, build, group
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report['pair_model_comparisons'] = {}
    for budget in (35, 100):
        if 'bp' + str(budget) not in report['groups']:
            continue
        for split in ('train', 'valid'):
            for grouping in ('absolute', 'reduced'):
                for task in TASKS:
                    left = pair_tables[f'baseline{budget}', split, grouping, task]
                    right = pair_tables[f'bp{budget}', split, grouping, task]
                    report['pair_model_comparisons'][f'e{budget}_{split}_{grouping}_{task}'] = paired_group_interval(left, right)
    report['seconds'] = time.monotonic() - start
    report['peak_vram_mb'] = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.
    report['limitations'] = [
        'Fixed train subset is not the whole train set; comparisons use one training seed.',
        'Only saved MAE-selected checkpoints; earlier R2 peaks cannot be reconstructed from histories.',
        'Feature means/second moments are lossy diagnostic summaries, not measures of total information.',
        'Inference removals can be out of distribution and do not replace retrained C0/C1 controls.',
        'Target-entropy strata are posthoc label diagnostics, never model inputs.',
        'Radial support proxy omits angular and longer-range environment differences.',
    ]
    dump(out / 'report.json', report)
    print(json.dumps({'complete': str(out), 'seconds': report['seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='64 rows/split; two models; plumbing only')
    args = parser.parse_args()
    diagnose(args.out_dir, args.smoke)


if __name__ == '__main__':
    main()
