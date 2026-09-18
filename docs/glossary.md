# Glossary (use these definitions consistently)

## Data Caliber (口径 — always state when reporting numbers)
- **pre-Q**: Legacy cache 20,040 / 2,477 / 2,471, γ floor 0.002. Used by B5/B6/H1/S1. Archived — do not cite as current.
- **Q1**: Clean pool 18,706 / 2,313 / 2,287, γ floor 0.106+. All experiments from B7 onward use Q1. Note: "Q1" = data quarantine (1,682 samples removed). "Qc1/Qc2" = model coordinate trunks (`--q1_coord` / `--q2_fourier`). Never write bare "Q1" without context.
- **oracle / blind**: Oracle uses ground-truth sum-slots for denormalization. Blind uses η/γ head self-predictions. `gap = oracle_R² − blind_R²`. Report p50/p90/p99 of gap distribution.
- **Number template**: `e med/fail + p med/fail + (test|valid, epN, oracle|blind, Q1|pre-Q)`. Example: `e 0.518/5.73% (test, best ep33, oracle, Q1)`. Use `pt`/`pp` for percentage points.

## Experiment States (五态)
| State | Meaning | Action |
|-------|---------|--------|
| **win** (merge) | med improves beyond draw line, fail doesn't worsen | Merge to default |
| **park** | Draw — code + tests kept, default OFF | Keep behind flag |
| **pending** | Awaiting long run confirmation | Do not start next arm |
| **dead** | Mechanism disproven | Never redo |
| **closed** | Paused, can reopen later | Available for future |

**Draw line**: |Δmed| < 0.02 AND |Δfail| < 1pp = draw.

## Common Abbreviations
- **med / fail**: Median R² / R²<0 failure rate. Mean R² is deprecated.
- **ctl / exp**: Control arm / experiment arm. Always suffix, e.g., `ctl_B7-35`, `ctl_Q1-10`.
- **NBANDS truncation**: MP-DOS insufficient bands → total electrons < 50% → eDOS shape poison. See Q1 D1–D4.
- **E_F misalignment**: Samples with only 0.01–0.1 e⁻/atom in window tail. Main cause of blind gap p99. Routed to S1/Eg queue.
- **sum-slot / Δ**: Box-average sum × Δ recovers area. Δ_e = 0.09375 eV, Δ_p = 19.6875 cm⁻¹.
- **balanced-score**: Valid-set composite score for best-checkpoint selection. Formula in runner. Not comparable across different epoch counts.
