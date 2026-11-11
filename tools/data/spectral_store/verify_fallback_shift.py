"""Read-only check: why do fallback (Static) eDOS tasks differ from MP's canonical NSCF Uniform DOS?

Log with the pre-registered criteria: docs/logs/log-2026-11-10-fallback-shift-check.md.

Same material set and fallback pick as `verify_store.py fallback` (type_then_volume rule;
materials with a usable canonical task and another owned task; test_q1 materials skipped
before any spectrum is read). The canonical label is always the project label (box_average
of the canonical spectrum on E0). The fallback spectrum is compared under:

- M0  project box_average, no shift (reproduces the existing check)
- M1  box_average, best global shift (oracle: searched against the canonical label)
- M2  piecewise-linear integral binning ("interp binning"), no shift (deployable)
- M3  interp binning, best shift (oracle)
- M4  interp binning, shift = E_F(fallback) - E_F(canonical) (absolute-energy alignment)
- R   canonical coarse-grained onto the fallback's own energy points (at the M3 shift):
      R_floor = E(label, coarse canonical), R_resid = E(coarse canonical, M3 fallback)
- Z   deployable re-zeroing from the fallback spectrum alone (own band edges), interp binning:
      Z_mid (zero at own mid-gap) and Z_vbm (zero at own VBM + c, c fitted on half A only)

Also: band-gap class from the canonical spectrum (analysis) and from the fallback spectrum
(deployable), grid sizes/spacing, spin channels, calc-cell atom counts, and how many
eligible new materials without a canonical task the deployable rule would add.

Writes data/spectral_store_v1/verify/fallback_shift.json and fallback_shift_pairs.parquet
(per pair); refuses to overwrite either.

--posthoc (added after the main run; not pre-registered, see the log section 3) re-reads the
pairs table and writes fallback_shift_posthoc.json / fallback_shift_posthoc_pairs.parquet:

- L        E(project label, exact integral of the same canonical spectrum): label-rule bias
- N2/N3    exact-integral binning on both sides, no shift / best shift (neutral comparison)
- Rn_*     resolution floor / residual in the neutral frame
- D_or     fallback interpolated onto the canonical's own energy points, then box_average
           (label emulation upper bound; uses the canonical grid)
- D_hat    fallback interpolated onto an estimated grid (spacing = kappa x own range / 2000,
           starting at its own lowest energy; kappa fitted on half A), then box_average (deployable) Reads staging tables only; never writes elsewhere.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from multiprocessing import Pool

from common import STORE, box_average, choose_edos_task, load_calc_types, strip_mp
from verify_store import E_CONST, FALLBACK_VARIANTS, GRIDS, _shape

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUT = STORE / 'verify'
STAGING = STORE / 'staging'
EDGES = GRIDS['edos']
NB = len(EDGES) - 1
BIN_W = (EDGES[-1] - EDGES[0]) / NB
WIDTH, ENERGY = E_CONST['edos']
SEED = 20261110
SHIFT_MAX, COARSE, FINE = 4.0, 0.02, 0.002
WINDOW = 12.0  # raw points kept: |x| <= WINDOW eV (E0 edges +-6, shifts +-4, margin)
OCC = 0.01  # states/eV/atom counted as occupied when locating band edges
GAP_SEARCH = 0.3  # the gap must intersect [-0.3, 0.3] eV around the task's own zero
METAL_MAX, SMALL_GAP_MAX = 0.15, 0.5  # canonical-spectrum classes (eV)
OWN_METAL_MIN, OWN_METAL_SPACING = 0.3, 2.5  # own-spectrum metal threshold: max(0.3, 2.5 x spacing)
MAX_ATOMS = 160
TYPE_GROUPS = ('GGA Static', 'GGA+U Static', 'GGA NSCF Uniform', 'GGA+U NSCF Uniform')

SPEC = {}  # task -> (x, y) sorted, windowed; filled before the pair pool forks


# ---------------------------------------------------------------- spectra
def _extract(job):
    path, wanted = job
    out = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=['source_task_id', 'x', 'y']):
        ids = batch.column('source_task_id').to_pylist()
        take = [i for i, t in enumerate(ids) if t in wanted]
        if not take:
            continue
        for r in batch.take(pa.array(take)).to_pylist():
            x, y = np.asarray(r['x'], np.float64), np.asarray(r['y'], np.float64)
            keep = np.isfinite(x) & np.isfinite(y)
            x, y = x[keep], y[keep]
            order = np.argsort(x, kind='stable')
            x, y = x[order], y[order]
            win = np.abs(x) <= WINDOW
            out[r['source_task_id']] = (x[win], y[win], len(x), float(x[0]), float(x[-1]))
    return out


def make_integral(x, y):
    """G(t) = integral of the piecewise-linear interpolant from x[0] to t (0 outside the range)."""
    h = np.diff(x)
    F = np.concatenate([[0.0], np.cumsum(0.5 * h * (y[:-1] + y[1:]))])
    slope = np.where(h > 0, np.diff(y) / np.where(h > 0, h, 1.0), 0.0)

    def G(t):
        t = np.clip(t, x[0], x[-1])
        k = np.clip(np.searchsorted(x, t, side='right') - 1, 0, len(x) - 2)
        u = t - x[k]
        return F[k] + y[k] * u + 0.5 * slope[k] * u * u
    return G


def interp_bins(G, shifts):
    """Bin averages of the spectrum moved by +s (rows = shifts)."""
    t = EDGES[None, :] - np.asarray(shifts, float)[:, None]
    return np.diff(G(t), axis=1) / BIN_W


def box_bins(x, y, shifts):
    """Vectorised common.box_average for several shifts (used for the search only)."""
    shifts = np.asarray(shifts, float)
    X = x[None, :] + shifts[:, None]
    idx = np.searchsorted(EDGES, X, side='right') - 1
    valid = (idx >= 0) & (idx < NB)
    ns = len(shifts)
    flat = np.arange(ns)[:, None] * NB + idx
    cnt = np.bincount(flat[valid], minlength=ns * NB).reshape(ns, NB)
    ysum = np.bincount(flat[valid], weights=np.broadcast_to(y, X.shape)[valid], minlength=ns * NB).reshape(ns, NB)
    same = valid[:, :-1] & (idx[:, :-1] == idx[:, 1:])
    seg = 0.5 * (X[:, 1:] - X[:, :-1]) * (y[None, :-1] + y[None, 1:])
    trap = np.bincount(flat[:, :-1][same], weights=seg[same], minlength=ns * NB).reshape(ns, NB)
    return np.where(cnt > 1, trap / BIN_W, np.where(cnt == 1, ysum, 0.0))


def e_metric(a, b, na, nb):
    return WIDTH * ((np.asarray(a) / na - np.asarray(b) / nb) ** 2).sum(axis=-1) / ENERGY


def tv(a, b):
    return float(.5 * np.abs(_shape(a) - _shape(b)).sum())


def search(fn, a, na, nb):
    coarse = np.arange(-SHIFT_MAX, SHIFT_MAX + 1e-9, COARSE)
    s0 = coarse[int(np.argmin(e_metric(a, fn(coarse), na, nb)))]
    fine = s0 + np.arange(-COARSE, COARSE + 1e-9, FINE)
    return float(fine[int(np.argmin(e_metric(a, fn(fine), na, nb)))])


def gap_info(x, y, n):
    """Largest empty interval between occupied points that intersects [-0.3, 0.3] eV."""
    occ = x[y / max(n, 1) > OCC]
    if len(occ) < 2:
        return None
    lo, hi = occ[:-1], occ[1:]
    cand = np.flatnonzero((hi >= -GAP_SEARCH) & (lo <= GAP_SEARCH))
    if not len(cand):
        return None
    k = cand[int(np.argmax((hi - lo)[cand]))]
    near = x[(x > -2) & (x < 2)]
    spacing = float(np.median(np.diff(near))) if len(near) > 2 else float('nan')
    return {'gap': float(hi[k] - lo[k]), 'vbm': float(lo[k]), 'cbm': float(hi[k]), 'spacing': spacing}


def canon_class(g):
    if g is None:
        return 'unknown'
    return 'metal' if g['gap'] < METAL_MAX else ('small_gap' if g['gap'] < SMALL_GAP_MAX else 'gapped')


def own_gapped(g):
    return g is not None and g['gap'] >= max(OWN_METAL_MIN, OWN_METAL_SPACING * g['spacing'])


def type_group(ctype):
    return ctype if ctype in TYPE_GROUPS else 'other'


# ---------------------------------------------------------------- pairs
def _pair(item):
    material_id, canon, chosen, ctype = item
    xc, yc, _ = SPEC[canon['source_task_id']][:3]
    xs, ys, ns_full = SPEC[chosen['source_task_id']][:3]
    na, nb = canon['n_calc'] or 1, chosen['n_calc'] or 1
    a = box_average(xc, yc, EDGES)[0].astype(np.float64)
    b0, cov0 = box_average(xs, ys, EDGES)
    s1 = search(lambda S: box_bins(xs, ys, S), a, na, nb)
    b1 = box_average(xs + s1, ys, EDGES)[0]
    Gs = make_integral(xs, ys)
    b2 = interp_bins(Gs, [0.0])[0]
    s3 = search(lambda S: interp_bins(Gs, S), a, na, nb)
    b3 = interp_bins(Gs, [s3])[0]
    s4 = float(chosen['efermi'] - canon['efermi'])
    b4 = interp_bins(Gs, [s4])[0]
    # canonical coarse-grained onto the fallback's own energy points (canonical frame, M3 shift)
    t = xs + s3
    d = np.diff(t)
    mids = np.concatenate([[t[0] - d[0] / 2], (t[1:] + t[:-1]) / 2, [t[-1] + d[-1] / 2]])
    Gc = make_integral(xc, yc)
    width = np.diff(mids)
    coarse = np.where(width > 0, (Gc(mids[1:]) - Gc(mids[:-1])) / np.where(width > 0, width, 1), 0.0)
    r = interp_bins(make_integral(t, coarse), [0.0])[0]
    gc, gs = gap_info(xc, yc, na), gap_info(xs, ys, nb)
    rec = {
        'material_id': material_id, 'canonical_task': canon['source_task_id'], 'fallback_task': chosen['source_task_id'],
        'fallback_type': ctype, 'type_group': type_group(ctype),
        'same_functional': chosen['functional'] == canon['functional'],
        'n_calc_canonical': na, 'n_calc_fallback': nb,
        'spin_canonical': canon['spin_channels'], 'spin_fallback': chosen['spin_channels'],
        'npts_canonical': canon['n_points'], 'npts_fallback': chosen['n_points'],
        'spacing_canonical': gc['spacing'] if gc else float('nan'), 'spacing_fallback': gs['spacing'] if gs else float('nan'),
        'empty_bins_m0': int((cov0 == 0).sum()),
        'efermi_canonical': canon['efermi'], 'efermi_fallback': chosen['efermi'],
        'gap_canonical': gc['gap'] if gc else float('nan'), 'vbm_canonical': gc['vbm'] if gc else float('nan'),
        'cbm_canonical': gc['cbm'] if gc else float('nan'),
        'gap_fallback': gs['gap'] if gs else float('nan'), 'vbm_fallback': gs['vbm'] if gs else float('nan'),
        'cbm_fallback': gs['cbm'] if gs else float('nan'),
        'class_canonical': canon_class(gc), 'own_gapped': own_gapped(gs),
        's1': s1, 's3': s3, 's4': s4,
        'R_floor': float(e_metric(a, r, na, na)), 'R_resid': float(e_metric(r, b3, na, nb)),
        'trunc_fallback': chosen['trunc_ratio'], 'npts_fallback_full': ns_full,
    }
    for name, b in (('M0', b0), ('M1', b1), ('M2', b2), ('M3', b3), ('M4', b4)):
        rec[f'E_{name}'] = float(e_metric(a, b, na, nb))
        rec[f'TV_{name}'] = tv(a, b)
    return rec


def _rezero(item):
    """Deployable rules: shift = -(own zero estimate) for own-gapped spectra, 0 for own metals."""
    rec, canon_task, c_vbm = item
    xc, yc = SPEC[canon_task][:2]
    xs, ys = SPEC[rec['fallback_task']][:2]
    na, nb = rec['n_calc_canonical'], rec['n_calc_fallback']
    a = box_average(xc, yc, EDGES)[0].astype(np.float64)
    Gs = make_integral(xs, ys)
    out = {}
    gapped = rec['own_gapped']
    for name, s in (('Z_mid', -(rec['vbm_fallback'] + rec['cbm_fallback']) / 2 if gapped else 0.0),
                    ('Z_vbm', -(rec['vbm_fallback'] + c_vbm) if gapped else 0.0)):
        b = interp_bins(Gs, [s])[0]
        out[f'E_{name}'] = float(e_metric(a, b, na, nb))
        out[f'TV_{name}'] = tv(a, b)
        out[f's_{name}'] = float(s)
    return out


# ---------------------------------------------------------------- summaries
def quant(v, ps=(10, 50, 90)):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    return {str(p): (float(np.percentile(v, p)) if len(v) else None) for p in ps}


def summarise(recs, metrics):
    out = {'n': len(recs)}
    if not recs:
        return out
    for m in metrics:
        e = np.asarray([r[f'E_{m}'] for r in recs])
        out[m] = {'E_median': float(np.median(e)), 'E_P90': float(np.percentile(e, 90)),
                  'TV_median': float(np.median([r[f'TV_{m}'] for r in recs])),
                  'TV_P90': float(np.percentile([r[f'TV_{m}'] for r in recs], 90)),
                  'frac_E_lt_0.0526': float(np.mean(e < 0.0526)), 'frac_E_le_0.15': float(np.mean(e <= 0.15)),
                  'frac_E_ge_0.30': float(np.mean(e >= 0.30))}
    return out


def shift_summary(recs):
    if not recs:
        return {}
    s1, s3, s4 = (np.asarray([r[k] for r in recs]) for k in ('s1', 's3', 's4'))
    return {'s1_quantiles': quant(s1, (5, 25, 50, 75, 95)), 'abs_s1_median': float(np.median(np.abs(s1))),
            's3_quantiles': quant(s3, (5, 25, 50, 75, 95)), 'abs_s3_median': float(np.median(np.abs(s3))),
            'frac_abs_s3_gt_0.1': float(np.mean(np.abs(s3) > 0.1)), 'frac_abs_s3_gt_0.5': float(np.mean(np.abs(s3) > 0.5)),
            'frac_s3_at_search_limit': float(np.mean(np.abs(s3) > SHIFT_MAX - 0.03)),
            'dEF_quantiles': quant(s4, (5, 25, 50, 75, 95)), 'frac_abs_dEF_gt_0.1': float(np.mean(np.abs(s4) > 0.1)),
            'frac_abs_s3_minus_dEF_lt_0.05': float(np.mean(np.abs(s3 - s4) < 0.05)),
            'corr_s3_dEF': float(np.corrcoef(s3, s4)[0, 1]) if len(s3) > 2 else None}


METRICS = ('M0', 'M1', 'M2', 'M3', 'M4')


def by(recs, key):
    groups = {}
    for r in recs:
        groups.setdefault(key(r), []).append(r)
    return dict(sorted(groups.items(), key=lambda kv: -len(kv[1])))


# ---------------------------------------------------------------- deployment counts
def eligible_new():
    """make_splits.assign eligibility for new materials (structure metadata only)."""
    materials = pq.read_table(STORE / 'materials.parquet', columns=[
        'material_id', 'legacy_split', 'reduced_composition', 'in_current_mp', 'deprecated', 'license',
        'structure_status', 'n_atoms', 'cell']).to_pylist()
    exclusion = {m['reduced_composition'] for m in materials if m['legacy_split'] in ('valid', 'test')}
    return {m['material_id']: m for m in materials
            if not m['legacy_split'] and m['in_current_mp'] and not m['deprecated'] and m['license'] == 'BY-C'
            and m['structure_status'] == 'ok' and m['n_atoms'] <= MAX_ATOMS
            and m['reduced_composition'] not in exclusion}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='random subset of pairs (0 = all); smoke tests only')
    parser.add_argument('--posthoc', action='store_true', help='post-hoc analyses (log section 3)')
    args = parser.parse_args()
    if args.posthoc:
        return posthoc(args.workers)
    for name in ('fallback_shift.json', 'fallback_shift_pairs.parquet'):
        if (OUT / name).exists() and not args.limit:
            raise FileExistsError(OUT / name)
    t0 = time.time()
    calc_types = load_calc_types()
    ctype = lambda r: calc_types.get(strip_mp(r['source_task_id']), 'missing')  # noqa: E731
    parts = sorted((STAGING / 'edos').glob('part-*.parquet'))
    cols = ['material_id', 'source_task_id', 'functional', 'status', 'is_canonical', 'calc_volume_per_atom',
            'calc_numbers', 'efermi', 'n_points', 'spin_channels', 'trunc_ratio']
    rows = pa.concat_tables([pq.read_table(f, columns=cols) for f in parts]).to_pylist()
    materials = pq.read_table(STAGING / 'materials.parquet',
                              columns=['material_id', 'legacy_split', 'cell', 'n_atoms']).to_pylist()
    volume = {m['material_id']: abs(float(np.linalg.det(np.asarray(m['cell'], np.float64).reshape(3, 3)))) / m['n_atoms']
              for m in materials if m['cell'] and m['legacy_split'] != 'test'}
    owned = {}
    for r in rows:
        if r['material_id'] in volume:  # test_q1 materials dropped here, before any spectrum is read
            r['n_calc'] = len(r['calc_numbers'] or [])
            del r['calc_numbers']
            owned.setdefault(r['material_id'], []).append(r)
    del rows
    order = FALLBACK_VARIANTS['type_then_volume']
    trials = []
    for material_id, rs in owned.items():
        canon = [r for r in rs if r['is_canonical'] and r['status'] == 'ok']
        if not canon:
            continue
        chosen, _ = choose_edos_task([r for r in rs if not r['is_canonical']], volume[material_id], calc_types, order)
        if chosen is not None:
            trials.append((material_id, canon[0], chosen, ctype(chosen)))
    n_trials_all = len(trials)
    rng = np.random.default_rng(SEED)
    if args.limit:
        trials = [trials[i] for i in sorted(rng.choice(len(trials), args.limit, replace=False))]
    # Materials without a canonical task: the pick the fallback rule would make (for counts)
    eligible = eligible_new()
    phdos_ok = {r['material_id'] for f in sorted((STAGING / 'phdos').glob('part-*.parquet'))
                for r in pq.read_table(f, columns=['material_id', 'status']).to_pylist()
                if r['material_id'] and r['status'] == 'ok'}
    edos_canon_ok = {m for m, rs in owned.items() if any(r['is_canonical'] and r['status'] == 'ok' for r in rs)}
    picks = {}
    for material_id, rs in owned.items():
        if material_id in edos_canon_ok:
            continue
        chosen, kind = choose_edos_task(rs, volume[material_id], calc_types, order)
        if chosen is not None:
            picks[material_id] = chosen
    if args.limit:
        picks = {k: picks[k] for k in sorted(picks)[:args.limit]}
    wanted = {t[1]['source_task_id'] for t in trials} | {t[2]['source_task_id'] for t in trials} | \
        {p['source_task_id'] for p in picks.values()}
    print(f'pairs {len(trials)} (of {n_trials_all}); no-canonical picks {len(picks)}; tasks {len(wanted)}', flush=True)
    with Pool(min(args.workers, len(parts))) as pool:
        for part in pool.imap_unordered(_extract, [(p, wanted) for p in parts]):
            SPEC.update(part)
    print(f'spectra loaded {len(SPEC)} ({time.time() - t0:.0f}s)', flush=True)
    recs = []
    with Pool(args.workers) as pool:
        for i, rec in enumerate(pool.imap(_pair, trials, chunksize=64)):
            recs.append(rec)
            if (i + 1) % 10000 == 0:
                print(f'pairs done {i + 1} ({time.time() - t0:.0f}s)', flush=True)
    # deployable re-zero: c fitted on half A (per type group, own-gapped only), applied everywhere
    half = {r['material_id']: ('A' if h < 0.5 else 'B') for r, h in zip(recs, rng.random(len(recs)))}
    c_vbm = {}
    for group in TYPE_GROUPS + ('other',):
        v = [-r['s3'] - r['vbm_fallback'] for r in recs
             if half[r['material_id']] == 'A' and r['type_group'] == group and r['own_gapped']]
        c_vbm[group] = float(np.median(v)) if v else 0.0
    with Pool(args.workers) as pool:
        z = pool.map(_rezero, [(r, r['canonical_task'], c_vbm[r['type_group']]) for r in recs], chunksize=64)
    for r, extra in zip(recs, z):
        r.update(extra, half=half[r['material_id']])
    # deployment: own-gap class of the picks for materials without a canonical task
    deploy = []
    for material_id, p in picks.items():
        x, y = SPEC[p['source_task_id']][:2]
        g = gap_info(x, y, p['n_calc'] or 1)
        deploy.append({'material_id': material_id, 'type_group': type_group(ctype(p)), 'own_gapped': own_gapped(g),
                       'eligible': material_id in eligible, 'phdos_ok': material_id in phdos_ok,
                       'status_ok': p['status'] == 'ok'})

    report = build_report(recs, deploy, eligible, edos_canon_ok, phdos_ok, c_vbm, n_trials_all, args)
    report['runtime_s'] = round(time.time() - t0, 1)
    if args.limit:
        print(json.dumps(report, indent=1)[:6000])
        return
    OUT.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(recs), OUT / 'fallback_shift_pairs.parquet', compression='zstd')
    (OUT / 'fallback_shift.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1)[:6000])


def build_report(recs, deploy, eligible, edos_canon_ok, phdos_ok, c_vbm, n_trials_all, args):
    zm = METRICS + ('Z_mid', 'Z_vbm')
    report = {
        'protocol': {'materials': 'usable canonical task + another owned task, test_q1 skipped; fallback pick '
                                  'type_then_volume (as verify_store.py fallback)',
                     'pairs': len(recs), 'pairs_available': n_trials_all, 'limit': args.limit, 'seed': SEED,
                     'E': 'width*sum((a/na-b/nb)^2)/energy, E0 grid [-6,6] eV 128 bins, per atom by each task cell',
                     'E_const': [WIDTH, ENERGY], 'shift_search': [SHIFT_MAX, COARSE, FINE],
                     'occ_threshold_states_per_eV_atom': OCC, 'gap_search_window': GAP_SEARCH,
                     'canonical_classes_eV': {'metal': f'<{METAL_MAX}', 'small_gap': f'{METAL_MAX}-{SMALL_GAP_MAX}',
                                              'gapped': f'>={SMALL_GAP_MAX}'},
                     'own_gapped': f'gap >= max({OWN_METAL_MIN}, {OWN_METAL_SPACING} x local spacing)',
                     'Z_vbm_c_fitted_on_half_A': c_vbm},
        'by_type': {}, 'by_type_and_canonical_class': {}, 'by_type_and_own_class': {},
    }
    for g, rs in by(recs, lambda r: r['type_group']).items():
        report['by_type'][g] = {**summarise(rs, METRICS), 'shifts': shift_summary(rs),
                                'R_floor_median': float(np.median([r['R_floor'] for r in rs])),
                                'R_resid_median': float(np.median([r['R_resid'] for r in rs])),
                                'R_floor_P90': float(np.percentile([r['R_floor'] for r in rs], 90)),
                                'R_resid_P90': float(np.percentile([r['R_resid'] for r in rs], 90)),
                                'half_B_Z': summarise([r for r in rs if r['half'] == 'B'], ('M2', 'M3', 'Z_mid', 'Z_vbm')),
                                'grid': {'npts_fallback': dict(Counter(r['npts_fallback'] for r in rs).most_common(4)),
                                         'npts_canonical': dict(Counter(r['npts_canonical'] for r in rs).most_common(4)),
                                         'spacing_fallback_median': float(np.nanmedian([r['spacing_fallback'] for r in rs])),
                                         'spacing_canonical_median': float(np.nanmedian([r['spacing_canonical'] for r in rs])),
                                         'empty_bins_m0_median': float(np.median([r['empty_bins_m0'] for r in rs]))}}
        report['by_type_and_canonical_class'][g] = {
            k: {**summarise(v, METRICS), 'shifts': shift_summary(v)}
            for k, v in by(rs, lambda r: r['class_canonical']).items()}
        report['by_type_and_own_class'][g] = {
            ('own_gapped' if k else 'own_metal'): summarise([r for r in v if r['half'] == 'B'], ('M2', 'Z_mid', 'Z_vbm'))
            for k, v in by(rs, lambda r: r['own_gapped']).items()}
        report['by_type'][g]['own_vs_canonical_class'] = {
            f'{a}|own_gapped={b}': n for (a, b), n in Counter((r['class_canonical'], r['own_gapped']) for r in rs).items()}
        report['by_type'][g]['spin'] = {f'{a}->{b}': {**summarise(v, ('M0', 'M2', 'M3'))}
                                        for (a, b), v in by(rs, lambda r: (r['spin_canonical'], r['spin_fallback'])).items()}
        report['by_type'][g]['calc_atoms_differ'] = summarise(
            [r for r in rs if r['n_calc_canonical'] != r['n_calc_fallback']], ('M0', 'M2', 'M3'))
        report['by_type'][g]['abs_dEF_gt_0.1'] = summarise([r for r in rs if abs(r['s4']) > 0.1], ('M2', 'M3', 'M4'))
        report['by_type'][g]['abs_dEF_le_0.1'] = summarise([r for r in rs if abs(r['s4']) <= 0.1], ('M2', 'M3', 'M4'))
        gapped = [r for r in rs if r['class_canonical'] == 'gapped']
        if gapped:
            z = np.asarray([-r['vbm_canonical'] for r in gapped])
            frac = np.asarray([-r['vbm_canonical'] / r['gap_canonical'] for r in gapped])
            report['by_type'][g]['canonical_zero_in_gap'] = {'zero_minus_vbm_eV': quant(z, (10, 50, 90)),
                                                             'fraction_of_gap': quant(frac, (10, 50, 90))}
            zf = np.asarray([-r['vbm_fallback'] for r in gapped if np.isfinite(r['vbm_fallback'])])
            report['by_type'][g]['fallback_zero_minus_own_vbm_eV'] = quant(zf, (10, 50, 90))
    # deployment counts on eligible new materials
    new_canon = {m for m in edos_canon_ok if m in eligible}
    dep = [d for d in deploy if d['eligible'] and d['status_ok']]
    counts = {'eligible_new_with_canonical_edos': len(new_canon),
              'eligible_new_with_phdos': len(phdos_ok & set(eligible)),
              'eligible_new_dual_canonical': len(new_canon & phdos_ok),
              'eligible_new_without_canonical_with_fallback_pick': len(dep),
              'by_type_and_own_class': {}}
    for (g, gp), v in by(dep, lambda d: (d['type_group'], d['own_gapped'])).items():
        counts['by_type_and_own_class'][f"{g}|{'own_gapped' if gp else 'own_metal'}"] = {
            'materials': len(v), 'with_phdos': sum(d['phdos_ok'] for d in v)}
    report['deployment'] = counts
    return report


# ---------------------------------------------------------------- post-hoc (log section 3)
def _posthoc_pair(item):
    rec, kappa = item
    xc, yc, _, c_lo, c_hi = SPEC[rec['canonical_task']]
    xs, ys, ns_full, s_lo, s_hi = SPEC[rec['fallback_task']]
    na, nb = rec['n_calc_canonical'], rec['n_calc_fallback']
    label = box_average(xc, yc, EDGES)[0].astype(np.float64)
    Gc, Gs = make_integral(xc, yc), make_integral(xs, ys)
    exact_c = interp_bins(Gc, [0.0])[0]
    n3 = search(lambda S: interp_bins(Gs, S), exact_c, na, nb)
    out = {'material_id': rec['material_id'], 'L': float(e_metric(label, exact_c, na, na)),
           'label_area_ratio': float(label.sum() / exact_c.sum()) if exact_c.sum() > 0 else float('nan'),
           'range_ratio': (c_hi - c_lo) / (s_hi - s_lo), 'n3': n3}
    for name, b in (('N2', interp_bins(Gs, [0.0])[0]), ('N3', interp_bins(Gs, [n3])[0])):
        out[f'E_{name}'] = float(e_metric(exact_c, b, na, nb))
        out[f'TV_{name}'] = tv(exact_c, b)
    t = xs + n3
    d = np.diff(t)
    mids = np.concatenate([[t[0] - d[0] / 2], (t[1:] + t[:-1]) / 2, [t[-1] + d[-1] / 2]])
    width = np.diff(mids)
    coarse = np.where(width > 0, (Gc(mids[1:]) - Gc(mids[:-1])) / np.where(width > 0, width, 1), 0.0)
    rc = interp_bins(make_integral(t, coarse), [0.0])[0]
    out['Rn_floor'] = float(e_metric(exact_c, rc, na, na))
    out['Rn_resid'] = float(e_metric(rc, interp_bins(Gs, [n3])[0], na, nb))
    # label emulation: piecewise-linear fallback resampled on a dense grid, then the project rule
    d_or = box_average(xc, np.interp(xc, xs, ys, left=0.0, right=0.0), EDGES)[0]
    step = kappa * (s_hi - s_lo) / 2000.0
    k0 = np.ceil((-WINDOW - s_lo) / step)
    grid = s_lo + step * np.arange(max(k0, 0), np.floor((WINDOW - s_lo) / step) + 1)
    d_hat = box_average(grid, np.interp(grid, xs, ys, left=0.0, right=0.0), EDGES)[0]
    for name, b in (('D_or', d_or), ('D_hat', d_hat)):
        out[f'E_{name}'] = float(e_metric(label, b, na, nb))
        out[f'TV_{name}'] = tv(label, b)
    out['step_hat'] = float(step)
    return out


def posthoc(workers: int):
    for name in ('fallback_shift_posthoc.json', 'fallback_shift_posthoc_pairs.parquet'):
        if (OUT / name).exists():
            raise FileExistsError(OUT / name)
    t0 = time.time()
    recs = pq.read_table(OUT / 'fallback_shift_pairs.parquet', columns=[
        'material_id', 'canonical_task', 'fallback_task', 'n_calc_canonical', 'n_calc_fallback', 'type_group',
        'class_canonical', 'own_gapped', 'half', 'E_M0', 'TV_M0', 'npts_fallback']).to_pylist()
    wanted = {r['canonical_task'] for r in recs} | {r['fallback_task'] for r in recs}
    parts = sorted((STAGING / 'edos').glob('part-*.parquet'))
    with Pool(min(workers, len(parts))) as pool:
        for part in pool.imap_unordered(_extract, [(p, wanted) for p in parts]):
            SPEC.update(part)
    print(f'spectra loaded {len(SPEC)} ({time.time() - t0:.0f}s)', flush=True)
    kappa = {}
    for group in TYPE_GROUPS + ('other',):
        v = [(SPEC[r['canonical_task']][4] - SPEC[r['canonical_task']][3]) /
             (SPEC[r['fallback_task']][4] - SPEC[r['fallback_task']][3])
             for r in recs if r['half'] == 'A' and r['type_group'] == group]
        kappa[group] = float(np.median(v)) if v else 1.0
    with Pool(workers) as pool:
        extra = pool.map(_posthoc_pair, [(r, kappa[r['type_group']]) for r in recs], chunksize=64)
    for r, e in zip(recs, extra):
        r.update(e)
    metrics = ('M0', 'N2', 'N3', 'D_or', 'D_hat')
    report = {'protocol': {'note': 'post-hoc, not pre-registered (log section 3)', 'pairs': len(recs),
                           'kappa_fitted_on_half_A': kappa},
              'by_type': {}, 'D_hat_half_B_by_type_and_own_class': {}, 'D_hat_half_B_by_type_and_canonical_class': {}}
    for g, rs in by(recs, lambda r: r['type_group']).items():
        entry = summarise(rs, metrics)
        for k in ('L', 'Rn_floor', 'Rn_resid', 'label_area_ratio', 'range_ratio', 'step_hat'):
            entry[k] = quant([r[k] for r in rs], (10, 50, 90))
        entry['abs_n3_median'] = float(np.median(np.abs([r['n3'] for r in rs])))
        entry['half_B'] = summarise([r for r in rs if r['half'] == 'B'], ('M0', 'D_or', 'D_hat'))
        report['by_type'][g] = entry
        report['D_hat_half_B_by_type_and_own_class'][g] = {
            ('own_gapped' if k else 'own_metal'): summarise([r for r in v if r['half'] == 'B'], ('M0', 'N2', 'D_or', 'D_hat'))
            for k, v in by(rs, lambda r: r['own_gapped']).items()}
        report['D_hat_half_B_by_type_and_canonical_class'][g] = {
            k: summarise([r for r in v if r['half'] == 'B'], ('M0', 'N2', 'N3', 'D_or', 'D_hat'))
            for k, v in by(rs, lambda r: r['class_canonical']).items()}
    report['runtime_s'] = round(time.time() - t0, 1)
    pq.write_table(pa.Table.from_pylist(extra), OUT / 'fallback_shift_posthoc_pairs.parquet', compression='zstd')
    (OUT / 'fallback_shift_posthoc.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1)[:3000])


if __name__ == '__main__':
    main()
