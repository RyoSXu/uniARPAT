"""Shared definitions for spectral-store-v1 (design 2026-11-10 P0)."""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from math import gcd
from functools import reduce
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

STORE = ROOT / 'data/spectral_store_v1'
LEGACY = ROOT / 'data/crystal_q1_v1'
DELTA = ROOT / 'tools/data/raw/delta_mp'
SPLITS_LEGACY = ('train', 'valid', 'test')
THZ2CM = 33.35641
SYMPREC = 0.1
TRUNC_MIN = 0.5  # Q1 rule: full-spectrum mass / task-cell valence electrons
SCHEMA = 'spectral-store-v1'


def box_average(x, y, edges):
    """Dataset binning rule (identical to tools/data/c2b_grids.box_average, the label code).

    Left-closed bins; >=2 points -> trapezoid of in-bin points / width; one point -> its
    value; none -> 0 with coverage 0. No edge interpolation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    out = np.zeros(len(edges) - 1, dtype=np.float64)
    cov = np.zeros(len(edges) - 1, dtype=np.int8)
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x < edges[i + 1])
        n = int(m.sum())
        if n > 1:
            out[i] = np.trapz(y[m], x[m]) / (edges[i + 1] - edges[i])
            cov[i] = 1
        elif n == 1:
            out[i] = float(y[m][0])
            cov[i] = 1
    return out.astype(np.float32), cov


def exact_bin(x, y, edges):
    """Exact bin average of the piecewise-linear interpolant of (x, y) (label contract v2).

    The sampled points are joined by straight lines; each bin value is the exact integral of
    that function over [edge_i, edge_{i+1}] divided by the bin width. Outside the sampled
    range the function is 0. Duplicate x are averaged. Returns (values float32, coverage
    fraction of each bin inside the sampled range, float32). Unlike box_average, no edge
    segment is dropped, so the binned area equals the area of the interpolant.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    edges = np.asarray(edges, dtype=np.float64)
    width = np.diff(edges)
    if len(x) < 2:
        return np.zeros(len(width), np.float32), np.zeros(len(width), np.float32)
    order = np.argsort(x, kind='stable')
    x, y = x[order], y[order]
    ux, inverse = np.unique(x, return_inverse=True)
    if len(ux) != len(x):
        sums, counts = np.zeros(len(ux)), np.zeros(len(ux))
        np.add.at(sums, inverse, y)
        np.add.at(counts, inverse, 1)
        x, y = ux, sums / counts
    if len(x) < 2:
        return np.zeros(len(width), np.float32), np.zeros(len(width), np.float32)
    cumulative = np.r_[0.0, np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))]
    t = np.clip(edges, x[0], x[-1])
    k = np.clip(np.searchsorted(x, t, side='right') - 1, 0, len(x) - 2)
    dx = t - x[k]
    slope = (y[k + 1] - y[k]) / (x[k + 1] - x[k])
    primitive = cumulative[k] + y[k] * dx + 0.5 * slope * dx * dx
    values = np.diff(primitive) / width
    coverage = (np.minimum(edges[1:], x[-1]) - np.maximum(edges[:-1], x[0])).clip(0) / width
    return values.astype(np.float32), coverage.astype(np.float32)


def classify_label(new, old, mask_old, max_clipped=None):
    """Compare a re-binned spectrum with a frozen label: 'exact', 'differs_only_where_label_lower'
    (spike clipping only lowers values; at most max_clipped such bins when given) or 'differs'."""
    new = np.asarray(new)
    old = np.asarray(old)
    if np.array_equal(new, old):
        return 'exact', 0
    diff = np.abs(new.astype(np.float64) - old.astype(np.float64))
    close = diff <= 1e-6 * np.maximum(1.0, np.abs(old.astype(np.float64)))
    off = int((~close).sum())
    clipped_like = (new > old) & np.asarray(mask_old, bool)
    if off and np.all(close | clipped_like) and (max_clipped is None or off <= max_clipped):
        return 'differs_only_where_label_lower', off
    return 'differs', off


def spike_thresholds(values, mask, quantile=0.999):
    """Per-bin p99.9 of covered values (fit on train rows only in this store)."""
    values, mask = np.asarray(values, dtype=float), np.asarray(mask, dtype=bool)
    return np.array([np.quantile(values[mask[:, j], j], quantile) if mask[:, j].any() else np.inf
                     for j in range(values.shape[1])])


def clip_isolated_spikes(values, mask, thresholds):
    """Dataset spike rule: over-threshold bins that are not part of a wide peak and whose
    larger neighbour is below half the value are clipped to the threshold."""
    values = np.array(values, dtype=np.float32, copy=True)
    mask = np.asarray(mask, dtype=bool)
    left = np.zeros_like(values)
    left[:, 1:] = values[:, :-1]
    right = np.zeros_like(values)
    right[:, :-1] = values[:, 1:]
    over = (values > thresholds) & mask
    wide = np.zeros_like(over)
    wide[:, 1:-1] = over[:, 1:-1] & over[:, :-2] & over[:, 2:]
    clip = over & ~wide & (np.maximum(left, right) < 0.5 * values)
    values[clip] = np.broadcast_to(thresholds, values.shape)[clip].astype(np.float32)
    return values, clip


def composition_key(numbers) -> str:
    z, counts = np.unique(np.asarray(numbers, dtype=int), return_counts=True)
    return ','.join(f'{a}:{c}' for a, c in zip(z, counts))


def reduced_key(key: str) -> str:
    pairs = [tuple(map(int, item.split(':'))) for item in key.split(',')]
    g = reduce(gcd, [c for _, c in pairs])
    return ','.join(f'{a}:{c // g}' for a, c in pairs)


def load_pages(directory: Path):
    for path in sorted(Path(directory).glob('page_*.json.gz')):
        with gzip.open(path, 'rt') as stream:
            yield from json.load(stream)['data']


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def strip_mp(identifier: str) -> str:
    return identifier[3:] if identifier.startswith('mp-') else identifier


# ---------------------------------------------------------------- eDOS task rule
FALLBACK_FUNCTIONALS = ('GGA', 'GGA+U')


def load_calc_types(directory: Path = None) -> dict:
    """MP task id (without 'mp-') -> calc type string, from materials/core calc_types pages."""
    directory = directory or STORE / 'raw_mp_core' / 'core_calc_types'
    out = {}
    for doc in load_pages(directory):
        for task, calc_type in (doc.get('calc_types') or {}).items():
            out[strip_mp(task)] = calc_type
    return out


def choose_edos_task(candidates: list, input_volume: float | None, calc_types: dict | None = None,
                     type_order: tuple = (), allow_fallback: bool = True):
    """Pick a material's eDOS task: (task row, 'canonical' | 'fallback') or (None, None).

    Canonical: MP's own DOS task with status ok. Fallback (only when there is no usable
    canonical task): an owned GGA/GGA+U task with status ok; tasks whose calc type ends
    with an earlier entry of `type_order` come first, then the calculation volume closest
    to the input structure (|log ratio|), then the task id.
    """
    ok = [c for c in candidates if c['status'] == 'ok']
    canonical = [c for c in ok if c['is_canonical']]
    if canonical:
        return canonical[0], 'canonical'
    if not allow_fallback:
        return None, None
    pool = [c for c in ok if c['functional'] in FALLBACK_FUNCTIONALS]
    if not pool:
        return None, None

    def type_rank(c):
        calc_type = (calc_types or {}).get(strip_mp(c['source_task_id']), '')
        return next((i for i, t in enumerate(type_order) if calc_type.endswith(t)), len(type_order))

    def volume_rank(c):
        v = c.get('calc_volume_per_atom')
        if not input_volume or not v:
            return float('inf')
        return abs(float(np.log(v / input_volume)))

    return min(pool, key=lambda c: (type_rank(c), volume_rank(c), c['source_task_id'])), 'fallback'
