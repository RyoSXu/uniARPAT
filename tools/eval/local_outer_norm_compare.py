"""Train only complete A's outer post-norm candidate; reuse authenticated A64."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.local_norm_compare import NormProtocol, main
from utils.local_outer_norm_experiment import outer_norm_config, pre_norm_parameters, validate_outer_norm_model

OUTER_NORM = NormProtocol(
    route="LN2", stem="local-outer-norm", config_factory=outer_norm_config,
    reference_parameters=pre_norm_parameters, validate_model=validate_outer_norm_model,
    reference_init_key="pre_norm_initialization", rejected_checkpoint_key="pre_norm_checkpoint_rejected",
    compatibility_key="reference_compatibility",
    execution_fields=(("internal_normalization", ("post-norm",)*4), ("outer_normalization", ("post-norm",)*2)),
    decision_key="post_norm_conditions_met",
    scope=("complete A64 outer post-norm versus outer pre-norm; internal post-norm unchanged, identical initial tensors; "
           "staged L2/G6, joint losses, seed42, 35 epochs, crystal_q1_v1 train/valid only; "
           "equal-task blind median R2 selection; intervals conditional on selected checkpoints"),
    source_files=("utils/local_outer_norm_experiment.py", "tools/eval/local_outer_norm_compare.py",
                  "tools/checks/check_local_outer_norm.py"))


if __name__ == "__main__":
    main(protocol=OUTER_NORM)
