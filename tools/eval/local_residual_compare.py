"""Compare four internal A additions with no additions, retaining the outer atom residual."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.local_norm_compare import NormProtocol, main
from utils.local_residual_experiment import additive_parameters, residual_config, validate_residual_model

INTERNAL_RESIDUAL = NormProtocol(
    route="LR2", stem="local-residual", config_factory=residual_config,
    reference_parameters=additive_parameters, validate_model=validate_residual_model,
    reference_init_key="additive_initialization", rejected_checkpoint_key="additive_checkpoint_rejected",
    compatibility_key="reference_compatibility",
    execution_fields=(("internal_normalization", ("post-norm",)*4), ("outer_normalization", ("pre-norm",)*2),
                      ("local_readout", ("state",)*2), ("internal_residual_add", (False,)*4)),
    decision_key="no_internal_addition_conditions_met",
    scope=("complete A64 with versus without four explicit internal scalar additions; retain both norm positions, "
           "complete-state readout, output bias and outer atom residual; identical initial tensors, staged L2/G6, "
           "joint losses, seed42, 35 epochs, crystal_q1_v1 train/valid only; equal-task blind median R2 selection"),
    source_files=("utils/local_residual_experiment.py", "tools/eval/local_residual_compare.py",
                  "tools/checks/check_local_residual.py"))


if __name__ == "__main__":
    main(protocol=INTERNAL_RESIDUAL)
