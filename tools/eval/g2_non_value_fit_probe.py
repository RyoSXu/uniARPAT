"""冻结全部G2消息参数，检验其余encoder与读出能否拟合同一16对train材料。"""

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval.g2_value_fit_probe import run_probe

DESIGN = REPO_ROOT / "docs/design/design-g2-non-value-fit-probe.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "output/g2_non_value_fit_q1")
    parser.add_argument("--output-prefix", type=Path, default=REPO_ROOT / "results/g2_non_value_fit_q1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--finalize-existing", action="store_true", help="仅汇总已完成2000步的现有权重，不执行训练")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_probe(args.run_dir, args.output_prefix, torch.device(args.device), args.finalize_existing,
              arm="non_g2", design=DESIGN, runner_path=Path(__file__),
              comparison_prefix=REPO_ROOT / "results/g2_value_fit_q1")


if __name__ == "__main__":
    main()
