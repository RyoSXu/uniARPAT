# Data Pipeline Tools

Scripts for fetching, processing, and auditing the uniARPAT training data.
All scripts are one-shot utilities (run once during data preparation, not during training).

## Directory Structure

| Directory | Purpose |
|-----------|--------|
| `fetch/` | API calls to Materials Project, PhononDB, JARVIS |
| `process/` | Data transformation, cache building, grid construction |
| `census/` | Data auditing, coverage analysis, gap detection |

## Key Scripts

### fetch/
- `fetch_mp_raw.py` — Bulk download from Materials Project API
- `download_phonondb.py` — Download PhononDB phonon spectra
- `a7c_fetch_jarvis.py` — Fetch JARVIS-DFT coverage data

### process/
- `a4_process.py` — Main v2 data processing pipeline
- `a6_split.py` — Stratified train/valid/test split
- `build_v2_cache.py` — Build training cache from processed data
- `q1_rebuild.py` — Rebuild Q1 clean pool cache
- `c2b_grids.py` — Generate energy/frequency grid definitions

### census/
- `census_final.py` — Final data census and statistics
- `a2b_gap.py` — Band gap analysis and filtering
- `a3c_phonondb_coverage.py` — PhononDB coverage mapping
