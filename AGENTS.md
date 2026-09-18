# AGENTS.md — Agent Navigation

> Read `docs/status.md` first (3 min), then this page.
> Work logs go to `docs/logs/`, conclusions go to `docs/backlog.md` and `docs/decisions.md`.

## 1. What Is This

uniARPAT: end-to-end prediction of electronic DOS (eDOS) and phonon DOS (phDOS) from crystal structures.

Current default recipe: M1 shared backbone + sumnorm-KL/W1/Huber + H1 η/γ blind heads + dropout 0.05 + Q1 clean pool.

Baseline: B7 `_e9ctl` (Q1, M1×35 best ep33, e med 0.518 / fail 5.73%, p med 0.741 / fail 3.50%).

## 2. Read-First Checklist

| Task | Read first |
|------|------------|
| Any task | `docs/status.md` → `docs/index.md` → `docs/glossary.md` |
| Run experiments / modify model | `README.md` Quickstart + `docs/backlog.md` Phase table |
| Data pipeline questions | `docs/backlog.md` Q1 section + `index/z0_REPORT.md` D1–D4 |
| Terminology (med/fail/gap/park/pre-Q) | `docs/glossary.md` |

## 3. Key Commands

```bash
python3 -m unittest discover tests        # Full test suite (~34 tests, ~1 min)
python3 run_ablation_experiments.py --model M1 --epochs 35 --tag _e9ctl   # B7 baseline recipe
python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _xxx     # Pilot screening
```

**Forbidden:**
- `--model all --epochs 100` without explicit approval (contaminates results)
- Reusing checkpoints across different normalization schemes (see backlog)

## 4. Directory Map

```
uniARPAT/
├── model/          Backbone + heads + losses
├── datasets/       Dataset class
├── utils/          Features, metrics, config, scheduling
├── tools/
│   ├── data/       Data pipeline (fetch/ process/ census/)
│   ├── eval/       Evaluation & verdict scripts
│   └── legacy/     Archived entry points
├── configs/        default.yaml (template only, overwritten by CLI)
├── tests/          Unit tests
├── docs/           Documentation (see docs/index.md)
├── data/           Training data cache (gitignored)
├── output/         Checkpoints (gitignored, ~74 GB)
├── results/        Experiment CSVs (archive/ for historical)
├── figures/        Publication figures
└── index/          Data indices & Z0 report
```

## 5. Experiment Discipline

- **Single-factor**: one arm, one verdict per experiment.
- **Pilot first**: 10-epoch pilot; only winners proceed to long runs.
- **Equal compute**: control and experiment arms use same epoch budget.
- **Report format**: `e med/fail + p med/fail + (test|valid, epN, oracle|blind, Q1|pre-Q)`
- **Five states**: win (merge) / park (neutral, code kept, default off) / pending (awaiting long run) / dead (disproven, never redo) / closed (paused, can reopen)
- **Draw line**: |Δmed| < 0.02 and |Δfail| < 1pp = draw.

## 6. Session Protocol

**Start of session:**
1. Read `docs/status.md`
2. Claim next task
3. Verify previous verdict is recorded (gate rule)

**End of session (mandatory):**
1. Write work log: `docs/logs/log-YYYY-MM-DD-<topic>.md`
2. Update `docs/status.md`
3. Sync conclusions to `docs/backlog.md`

**Long runs:** use `setsid + nohup`. Artifacts go to `results/` (history_*.csv + test_*_summary.csv + samples_*.csv). Verdict scripts go to `tools/eval/`, never `/tmp`.
```
