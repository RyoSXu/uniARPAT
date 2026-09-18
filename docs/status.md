# Status (living doc — update at end of each session; history goes to logs)

Updated: 2026-09-18 | Baseline: B7 `_e9ctl` (Q1, M1×35 best ep33, e 0.518/5.73% p 0.741/3.50% Cv 0.30)

## Now
- E9-P0: G1 pending → Qc1 parked → Qc2 parked (Q group closed) → L3 closed (three-arm draw; loss track graduated; see log E9P0-L3).
- Data: Q1 clean pool 18,706 / 2,313 / 2,287; legacy cache archived at `data/archive/train4ARPAT_20260916_preQ1/`.
- Default recipe: sumnorm + E0P0 + eta + dropout 0.05.
- Structure cleanup (09-18): root has 3 .py entry points; legacy entry points archived to `tools/legacy/`; docs reorganized to English; dead code removed; ExperimentConfig dataclass + losses.py extracted.

## Next (sequential — one at a time)
1. Commander to decide: C5 fast-track or next E9-P0 candidate (after L3 closure).
2. E9 backlog (do not start early): C5 new readout → warp grids → non-uniform large window → density arm → C1 reunion / D3 / C3 → continuous spectral field.

## Blockers / Watch
- `dataset.py` coords default assertion regression fixed (auto fallback). Smoke test before C2b reruns.
- `output/` ~74 GB checkpoints: do NOT delete any checkpoint without explicit per-item commander approval (irreversible GPU cost). `output/*/config_used.yaml` not tracked; use `results/` as source of truth.
