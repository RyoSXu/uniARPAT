# Work Log: 2026-09-18 — Project Structure Overhaul

## What Changed

### Code Cleanup (Phase 1)
- Added `__init__.py` to `model/`, `datasets/`, `utils/`
- Removed unused `from ipaddress import ip_address` in `utils/misc.py`
- Removed weather/geophysics legacy code from `utils/metrics.py` (lat, weighted_rmse, WRMSE, WACC)
- Removed dead methods from `model/model.py`: `multi_step_predict`, `test_final`, `stat`
- Removed duplicate `self.metric_best = None` assignment
- Cleaned all debug comment cruft from `model/model.py`
- Deleted orphan `results/pred_*.npy` files
- Renamed `configs/config.yaml` → `configs/default.yaml` with template warning

### Code Refactoring (Phase 2)
- Created `utils/experiment_config.py` — `ExperimentConfig` dataclass replaces 38-parameter function signature
- Created `model/losses.py` — extracted all loss functions from `model/model.py`
- Refactored `run_ablation_experiments.py` to use `ExperimentConfig`

### Directory Reorganization (Phase 3)
- `docs/01_开发文档/` → `docs/` root + `docs/design/` + `docs/archive/`
- `docs/02_审查复核/` → `docs/reviews/`
- `docs/03_工作日志/` → `docs/logs/` (renamed 日志-* → log-*)
- `docs/04_前沿探索/` → `docs/research/`
- `tools/getdata/` → `tools/data/` (with `fetch/`, `process/`, `census/` subdirs)
- `results/` archived historical files to `results/archive/`

### Documentation (Phase 4)
- `AGENTS.md` rewritten in English
- `docs/status.md`, `docs/index.md`, `docs/glossary.md` rewritten in English
- All doc paths updated to new structure

## Verdict
Structure overhaul — no model/data changes. All tests should remain green.

## Next
- Run full test suite to verify
- Commander to decide next experiment arm (C5 or E9-P0 continuation)
