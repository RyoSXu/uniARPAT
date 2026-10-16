#!/usr/bin/env bash
# E7: cache-free static and contract checks used both locally and in CI.
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1
mapfile -t python_sources < <("$python_bin" - <<'PY'
from pathlib import Path

# Include actual moved/new files even while the Git index still has old paths.
root = Path('.')
sources = list(root.glob('*.py'))
for directory in ('model', 'utils', 'tools', 'tests', 'archive/code'):
    sources.extend((root / directory).rglob('*.py'))
print('\n'.join(sorted(str(path) for path in sources if '__pycache__' not in path.parts)))
PY
)

if [[ ${#python_sources[@]} -eq 0 ]]; then
    echo "No Python sources found in the maintained source directories." >&2
    exit 1
fi

"$python_bin" -m ruff check --no-cache "${python_sources[@]}"
# Compile in memory: syntax checks need not leave hundreds of cache files.
"$python_bin" - "${python_sources[@]}" <<'PY'
from pathlib import Path
import sys
import tokenize

paths = [Path(name) for name in sys.argv[1:]]
paths += list(Path('results/reports').rglob('*.py'))
paths += list(Path('archive/results').rglob('*.py'))
for path in paths:
    with tokenize.open(path) as stream:
        compile(stream.read(), str(path), 'exec')
PY
"$python_bin" -m unittest \
    tests.model.test_heads \
    tests.model.test_regressions \
    tests.pipeline.test_thermodynamics \
    tests.model.test_gated_attention \
    tests.model.test_scale_normalization \
    archive.code.tests.test_g1_graph \
    archive.code.tests.test_g2_structure_path_audit \
    archive.code.tests.test_g2_frozen_readout_probe \
    archive.code.tests.test_g2_encoder_adaptation_probe \
    archive.code.tests.test_g2_small_fit_probe \
    archive.code.tests.test_g2_value_fit_probe \
    archive.code.tests.test_g2_joint_content \
    archive.code.tests.test_periodic_manybody \
    archive.code.tests.test_joint_content_pilot \
    archive.code.tests.test_q2_fourier \
    archive.code.tests.test_c5_token_moe \
    tests.pipeline.test_amp \
    tests.pipeline.test_checkpoint_boundary \
    tests.pipeline.test_checkpoint_selection \
    archive.code.tests.test_c2_1b_loss_attribution \
    tests.pipeline.test_b7_cif_inference \
    archive.code.tests.test_d4_phdos_spike_imaginary_audit \
    archive.code.tests.test_d4b_negative_coordinate_provenance_audit \
    archive.code.tests.test_d4c_jarvis_min_fd_semantics_audit \
    archive.code.tests.test_d3a_edos_support_adjudication \
    tests.pipeline.test_repository_paths \
    tests.pipeline.test_repository_policy \
    tests.data.test_periodic_neighbors \
    tests.data.test_crystal_input \
    tests.model.test_periodic_messages \
    tests.model.test_periodic_geometry_cache \
    tests.model.test_local_global_encoder \
    tests.pipeline.test_local_global_comparison \
    tests.pipeline.test_lg_baseline \
    tests.pipeline.test_global_depth \
    tests.pipeline.test_lg_structure_diagnosis \
    tests.pipeline.test_local_structure_controls \
    tests.pipeline.test_local_structure_diagnosis \
    tests.pipeline.test_stage_readout_diagnosis \
    tests.pipeline.test_local_structure_small_fit \
    tests.pipeline.test_local_structure_hard_pair \
    tests.pipeline.test_local_structure_loss_isolation \
    tests.pipeline.test_global_only \
    tests.pipeline.test_legacy_backbone \
    tests.pipeline.test_global_norm \
    tests.pipeline.test_attention_dropout \
    tests.pipeline.test_attention_projection \
    tests.pipeline.test_i1_bplus \
    tests.pipeline.test_i1_a \
    tests.pipeline.test_local_readout \
    tests.pipeline.test_local_module \
    tests.model.test_periodic_messages_gpu

"$python_bin" tools/checks/check_repository.py --worktree
