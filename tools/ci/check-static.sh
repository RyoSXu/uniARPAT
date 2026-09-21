#!/usr/bin/env bash
# E7: cache-free static and contract checks used both locally and in CI.
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-python3}"
mapfile -t python_sources < <(git ls-files '*.py')

if [[ ${#python_sources[@]} -eq 0 ]]; then
    echo "No tracked Python sources found." >&2
    exit 1
fi

"$python_bin" -m ruff check "${python_sources[@]}"
"$python_bin" -m compileall -q "${python_sources[@]}"
"$python_bin" -m unittest \
    tests.test_model_heads \
    tests.test_p0_fixes \
    tests.test_thermo_props \
    tests.test_gated_attention \
    tests.test_scale_norm \
    tests.test_g1_graph \
    tests.test_q2_fourier \
    tests.test_c5_token_moe \
    tests.test_c4_amp \
    tests.test_e5_checkpoint_boundary \
    tests.test_c2_1b_loss_attribution \
    tests.test_b7_cif_inference \
    tests.test_d4_phdos_spike_imaginary_audit
