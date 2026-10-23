"""Compare complete A scalar-increment readout against authenticated A64."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.local_norm_compare import NormProtocol, main
from utils.local_increment_experiment import increment_config, state_readout_parameters, validate_increment_model

# Reuse the existing local comparison runner; neither normalization position changes.
INCREMENT_READOUT = NormProtocol(
    route="LR1", stem="local-increment", config_factory=increment_config,
    reference_parameters=state_readout_parameters, validate_model=validate_increment_model,
    reference_init_key="state_readout_initialization", rejected_checkpoint_key="state_readout_checkpoint_rejected",
    compatibility_key="reference_compatibility",
    execution_fields=(("internal_normalization", ("post-norm",)*4),
                      ("outer_normalization", ("pre-norm",)*2), ("local_readout", ("increment",)*2)),
    decision_key="increment_conditions_met",
    scope=("complete A64 projected scalar increment versus complete-state readout; retain output bias, both residuals "
           "and both norm positions; identical initial tensors, staged L2/G6, joint losses, seed42, 35 epochs, "
           "crystal_q1_v1 train/valid only; equal-task blind median R2 selection; selected-checkpoint intervals"),
    source_files=("utils/local_increment_experiment.py", "tools/eval/local_increment_compare.py",
                  "tools/checks/check_local_increment.py"))


if __name__ == "__main__":
    main(protocol=INCREMENT_READOUT)
