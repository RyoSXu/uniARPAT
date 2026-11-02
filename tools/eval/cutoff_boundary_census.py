"""Train-geometry census of strict-cutoff exposure and fixed-rotation count flips."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from tools.eval.crystal_model_invariance import record_arrays, write_json
from tools.eval.periodic_geometry_acceptance import compare_records, e3_operations, row_tokens
from utils.bucket_batch import automatic_pair_budget, crystal_collate, split_crystal_batch
from utils.crystal import CrystalGeometry
from utils.crystal_cache import PackedCrystalStore
from utils.paths import file_sha256, input_path
from utils.periodic_geometry import build_periodic_records
from utils.periodic_neighbors import g2_indegrees, g2_quintic_cutoff

RADIUS = 6.
SCAN_RADIUS = 6.02
BANDS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
PROBE_BAND = 1e-4
BATCH_SIZE = 32


def band_key(band):
    return f'band_{band:.0e}'


def summarize_batch(records, mask):
    """Count both sides in the consuming FP32 distances; include exact d=R."""
    valid = ~mask.detach().cpu().numpy()
    batch = records.batch.detach().cpu().numpy()
    dst = records.dst.detach().cpu().numpy()
    distances = records.distances.detach().cpu().numpy()
    if not valid[batch, dst].all():
        raise ValueError('padding cannot receive audited records')
    size, length = valid.shape
    receiver = batch*length+dst
    kept = distances < RADIUS
    weights = g2_quintic_cutoff(records.distances, RADIUS).detach().cpu().numpy()
    degree = np.bincount(receiver[kept], minlength=size*length).reshape(size, length)
    zero_weight = kept & (weights == 0)
    zero_centers = np.bincount(receiver[zero_weight], minlength=size*length).reshape(size, length) > 0
    rows = [{'n_atoms': int(valid[i].sum()), 'records_kept': int(degree[i].sum()),
             'zero_weight_kept_records': int(np.count_nonzero(zero_weight & (batch == i))),
             'centers_with_zero_weight_kept': int((zero_centers[i] & valid[i]).sum())} for i in range(size)]
    difference = abs(distances.astype(np.float64)-RADIUS)
    for band in BANDS:
        near = difference <= band
        centers = np.bincount(receiver[near], minlength=size*length).reshape(size, length) > 0
        for i, row in enumerate(rows):
            selected = near & (batch == i)
            key = band_key(band)
            row.update({f'{key}_inside': int(np.count_nonzero(selected & kept)),
                        f'{key}_exact': int(np.count_nonzero(selected & (distances == RADIUS))),
                        f'{key}_outside': int(np.count_nonzero(selected & (distances > RADIUS))),
                        f'{key}_centers': int((centers[i] & valid[i]).sum())})
    return rows, degree, distances, weights


def degree_change_summary(before, after):
    delta = np.asarray(after, dtype=np.int64)-np.asarray(before, dtype=np.int64)
    return {'changed_centers': int(np.count_nonzero(delta)),
            'max_abs_degree_delta': int(abs(delta).max(initial=0)),
            'net_degree_delta': int(delta.sum()),
            'max_abs_log_degree_delta': float(abs(np.log1p(after)-np.log1p(before)).max(initial=0))}


def geometry_batches(store, rows, positions):
    """Reuse complete-atom batching and its computation budget without labels."""
    for start in range(0, len(positions), BATCH_SIZE):
        chosen = positions[start:start+BATCH_SIZE]
        batch = crystal_collate([store[rows[position]] for position in chosen])
        offset = 0
        for chunk in split_crystal_batch(batch, automatic_pair_budget(batch)):
            count = len(chunk[0])
            yield np.asarray(chosen[offset:offset+count]), chunk[0], chunk[1]
            offset += count


def census(out, data_dir, *, max_wall_s=900):
    out, data_dir = input_path(out), input_path(data_dir)
    if out.exists():
        raise FileExistsError(f'refusing existing census: {out}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA geometry census required')
    torch.set_num_threads(1)
    started = time.monotonic()
    store = PackedCrystalStore(data_dir, 'train')
    excluded = set(store.manifest.get('training_exclusions', []))
    rows = np.array([i for i, mid in enumerate(store.material_ids) if str(mid) not in excluded])
    counts = store.n_atoms.numpy()[rows]
    offsets = np.r_[0, counts.cumsum()]
    rotations = [(name, q) for name, q, _ in e3_operations() if name.startswith('rotation_')]
    inputs = [data_dir/'manifest.json']
    for relative in ('train/cell_train.npy', 'train/frac_coords_train.npy', 'train/atomic_numbers_train.npy',
                     'train/atom_offsets_train.npy', 'train/n_atoms_train.npy', 'train/train_index.npy'):
        path = data_dir/relative
        if file_sha256(path) != store.manifest['files'][relative]['sha256']:
            raise ValueError(f'geometry input changed: {relative}')
        inputs.append(path)
    sources = [ROOT/name for name in ('tools/eval/cutoff_boundary_census.py',
        'tools/eval/crystal_model_invariance.py', 'tools/eval/periodic_geometry_acceptance.py',
        'utils/crystal.py', 'utils/crystal_cache.py', 'utils/bucket_batch.py', 'utils/periodic_geometry.py',
        'utils/periodic_neighbors.py', 'utils/relative_features.py', 'utils/paths.py')]
    out.mkdir(parents=True)
    policy = {'schema': 'train-cutoff-boundary-census-v1', 'split': 'effective train only', 'radius_A': RADIUS,
              'scan_radius_A': SCAN_RADIUS, 'bands_A': list(BANDS), 'dtype': 'float32',
              'probe_band_A': PROBE_BAND, 'rotations': [name for name, _ in rotations],
              'batch_size': BATCH_SIZE, 'max_wall_s': max_wall_s,
              'scan_note': '6.02 is an audit envelope only; production membership is still strict d<6',
              'interpretation': 'proximity is exposure, fixed-rotation count changes are observed sensitivity; neither proves output or accuracy effects',
              'input_sha256': {str(path): file_sha256(path) for path in inputs},
              'source_sha256': {str(path.relative_to(ROOT)): file_sha256(path) for path in sources}}
    write_json(out/'protocol.json', policy)
    for source in sources:
        dest = out/'source_snapshot'/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    stats, distance_blocks, receiver_blocks, weight_blocks = [], [], [], []
    baseline_degrees = np.empty(int(offsets[-1]), dtype=np.int32)
    with (out/'structure_statistics.csv').open('w') as stream:
        writer = None
        with torch.inference_mode():
            for positions, numbers, geometry in geometry_batches(store, rows, list(range(len(rows)))):
                if time.monotonic()-started > max_wall_s:
                    raise TimeoutError('census budget exceeded; no automatic rerun')
                mask = numbers.eq(0).cuda()
                geometry = geometry.to('cuda')
                records = build_periodic_records(geometry, mask, SCAN_RADIUS)
                batch_stats, degree, distances, weights = summarize_batch(records, mask)
                receivers = offsets[positions[records.batch.cpu().numpy()]]+records.dst.cpu().numpy()
                distance_blocks.append(distances)
                receiver_blocks.append(receivers.astype(np.int32))
                weight_blocks.append(weights)
                for local, position in enumerate(positions):
                    row = {'position': int(position), 'train_row': int(rows[position]),
                           'material_id': str(store.material_ids[rows[position]]), **batch_stats[local]}
                    stats.append(row)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    baseline_degrees[offsets[position]:offsets[position+1]] = degree[local, :counts[position]]
                stream.flush()
                if len(stats) % 1024 < len(positions) or len(stats) == len(rows):
                    print(json.dumps({'scanned': len(stats), 'total': len(rows), 'seconds': round(time.monotonic()-started, 1)}), flush=True)
    frame = pd.DataFrame(stats).sort_values('position')
    near_key = band_key(PROBE_BAND)
    probe_positions = frame.loc[frame[[f'{near_key}_{side}' for side in ('inside', 'exact', 'outside')]].sum(axis=1) > 0, 'position'].to_numpy(int)
    degree_archive = np.full((2+len(rotations), int(offsets[-1])), -1, dtype=np.int32)
    degree_archive[0] = baseline_degrees
    probe_rows, changed_edges, group_checks = [], [], []
    with torch.inference_mode():
        for positions, numbers, geometry in geometry_batches(store, rows, probe_positions.tolist()):
            if time.monotonic()-started > max_wall_s:
                raise TimeoutError('rotation probe budget exceeded; no automatic rerun')
            mask = numbers.eq(0).cuda()
            geometry = geometry.to('cuda')
            base = build_periodic_records(geometry, mask, RADIUS)
            base_arrays = record_arrays(base)
            base_degree = g2_indegrees(base.batch, base.dst, *mask.shape).cpu().numpy()
            for local, position in enumerate(positions):
                start, end = offsets[position:position+2]
                degree_archive[1, start:end] = base_degree[local, :counts[position]]
            for rotation_index, (name, q) in enumerate(rotations, 1):
                matrix = torch.as_tensor(q, dtype=torch.float64, device='cuda')
                changed = CrystalGeometry((geometry.cell.double()@matrix).float(), geometry.frac_coords)
                actual = build_periodic_records(changed, mask, RADIUS)
                actual_arrays = record_arrays(actual)
                actual_degree = g2_indegrees(actual.batch, actual.dst, *mask.shape).cpu().numpy()
                expected = {**base_arrays, 'r': base_arrays['r']@q}
                try:
                    check = compare_records(expected, actual_arrays, tolerance=1e-4)
                except AssertionError as error:
                    check = {'status': 'fail', 'reason': str(error)}
                group_checks.append({'rotation': name, 'positions': positions.tolist(), **check})
                _, left, right = np.intersect1d(row_tokens(base_arrays['keys']), row_tokens(actual_arrays['keys']), return_indices=True)
                missing = np.setdiff1d(np.arange(len(base_arrays['d'])), left)
                extra = np.setdiff1d(np.arange(len(actual_arrays['d'])), right)
                missing_counts = np.bincount(base_arrays['keys'][missing, 0], minlength=len(positions))
                extra_counts = np.bincount(actual_arrays['keys'][extra, 0], minlength=len(positions))
                for label, arrays, indices in [('removed', base_arrays, missing), ('added', actual_arrays, extra)]:
                    for edge in indices:
                        key = arrays['keys'][edge]
                        position = positions[key[0]]
                        changed_edges.append({'position': int(position), 'material_id': str(store.material_ids[rows[position]]),
                            'rotation': name, 'change': label, 'dst': int(key[1]), 'src': int(key[2]),
                            'Tx': int(key[3]), 'Ty': int(key[4]), 'Tz': int(key[5]), 'distance_A': float(arrays['d'][edge])})
                for local, position in enumerate(positions):
                    start, end = offsets[position:position+2]
                    n = counts[position]
                    original = base_degree[local, :n]
                    result = actual_degree[local, :n]
                    degree_archive[rotation_index+1, start:end] = result
                    probe_rows.append({'position': int(position), 'material_id': str(store.material_ids[rows[position]]),
                        'rotation': name, 'removed_records': int(missing_counts[local]), 'added_records': int(extra_counts[local]),
                        'census_vs_probe_changed_centers': int(np.count_nonzero(original != baseline_degrees[start:end])),
                        **degree_change_summary(original, result)})
    pd.DataFrame(probe_rows, columns=['position', 'material_id', 'rotation', 'removed_records', 'added_records',
        'census_vs_probe_changed_centers', 'changed_centers', 'max_abs_degree_delta', 'net_degree_delta',
        'max_abs_log_degree_delta']).to_csv(out/'rotation_statistics.csv', index=False)
    pd.DataFrame(changed_edges, columns=['position', 'material_id', 'rotation', 'change', 'dst', 'src',
        'Tx', 'Ty', 'Tz', 'distance_A']).to_csv(out/'changed_edges.csv', index=False)
    np.savez_compressed(out/'distance_archive.npz', distances=np.concatenate(distance_blocks),
        receiver=np.concatenate(receiver_blocks), cutoff_weights=np.concatenate(weight_blocks),
        center_offsets=offsets, train_rows=rows, material_ids=store.material_ids[rows].astype(str))
    np.savez_compressed(out/'degree_probe.npz', degree=degree_archive, probe_positions=probe_positions,
                       row_labels=np.array(['census_r6', 'probe_r6', *[name for name, _ in rotations]]))
    bands = {}
    for band in BANDS:
        key = band_key(band)
        exposed = frame[[f'{key}_{side}' for side in ('inside', 'exact', 'outside')]].sum(axis=1) > 0
        bands[str(band)] = {'inside_records': int(frame[f'{key}_inside'].sum()),
                            'exact_records': int(frame[f'{key}_exact'].sum()),
                            'outside_records': int(frame[f'{key}_outside'].sum()),
                            'structures': int(exposed.sum()), 'structures_pct': float(exposed.mean()*100),
                            'centers': int(frame[f'{key}_centers'].sum())}
    membership = {row['position'] for row in probe_rows if row['removed_records']+row['added_records']}
    flipped = {row['position'] for row in probe_rows if row['changed_centers']}
    context_changed = {row['position'] for row in probe_rows if row['census_vs_probe_changed_centers']}
    report = {'schema': 'train-cutoff-boundary-census-result-v1', 'completed': True,
        'structures': len(rows), 'atoms': int(counts.sum()), 'kept_directed_records': int(frame.records_kept.sum()),
        'bands_A': bands, 'zero_weight_kept_records': int(frame.zero_weight_kept_records.sum()),
        'structures_with_zero_weight_kept': int((frame.zero_weight_kept_records > 0).sum()),
        'rotation_probe': {'selected_structures': len(probe_positions), 'comparisons': len(probe_rows),
            'structures_with_record_changes': len(membership), 'structures_with_degree_changes': len(flipped),
            'degree_changed_pct_of_train': float(len(flipped)/len(rows)*100),
            'census_probe_context_changes': len(context_changed), 'geometry_failures': [row for row in group_checks if row['status'] == 'fail'],
            'group_checks': group_checks},
        'seconds': time.monotonic()-started, 'gpu': torch.cuda.get_device_name(),
        'peak_allocated_MiB': torch.cuda.max_memory_allocated()/1024**2,
        'labels_read': False, 'test_evaluated': False, 'model_forward_or_training': False,
        'limitations': 'finite fixed rotations on exposed structures; broad bands are proximity, not demonstrated numerical failures or accuracy loss',
        'artifact_sha256': {path.name: file_sha256(path) for path in (
            out/'protocol.json', out/'structure_statistics.csv', out/'rotation_statistics.csv',
            out/'changed_edges.csv', out/'distance_archive.npz', out/'degree_probe.npz')}}
    for path, digest in policy['input_sha256'].items():
        if file_sha256(path) != digest:
            raise RuntimeError(f'input changed during census: {path}')
    write_json(out/'report.json', report)
    print(json.dumps({key: value for key, value in report.items() if key not in ('rotation_probe', 'artifact_sha256')}, indent=2), flush=True)
    print(json.dumps({key: value for key, value in report['rotation_probe'].items() if key != 'group_checks'}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--data-dir', default=str(ROOT/'data/crystal_q1_v1'))
    args = parser.parse_args()
    census(args.out, args.data_dir)


if __name__ == '__main__':
    main()
