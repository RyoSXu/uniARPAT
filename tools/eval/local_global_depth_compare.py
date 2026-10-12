"""Plan or explicitly execute the decoupled global-depth comparison arms.

Default action prints commands only. Budget, seeds, local depth, global depth
and the valid checkpoint criterion must be supplied. The route is fixed to B+
and both orchestrations are generated from one shared initial state, so a
difference between them is the orchestration at fixed depth. No test loader or
evaluation is created by these arms.
"""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.local_global_compare import summarize, training_command
from tools.train import build_arg_parser, train_and_eval
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import depth_comparison_configs


def main():
    parser = build_arg_parser()
    parser.description = __doc__
    for name in ('epochs', 'batch_size', 'lr', 'local_depth', 'global_depth', 'selection_metric'):
        parser._option_string_actions[f'--{name}'].required = True
    parser.set_defaults(model='M1', atom_feat='z_only_proj', skip_test_eval=True, q1_coord=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--tag_prefix', required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--run', action='store_true',
                        help='Execute the supplied training budget sequentially')
    action.add_argument('--summarize', action='store_true',
                        help='Read existing valid histories without training')
    args = parser.parse_args()
    if args.model != 'M1' or args.local_route != 'none' or args.local_layout is not None:
        parser.error('depth comparison supplies route/layout itself and requires M1')
    configs = depth_comparison_configs(ExperimentConfig.from_args(args), depth=args.local_depth,
                                       global_depth=args.global_depth, seeds=args.seeds,
                                       tag_prefix=args.tag_prefix)
    if args.summarize:
        print(json.dumps(summarize(configs), ensure_ascii=False, indent=2))
    elif args.run:
        for cfg in configs:
            train_and_eval(cfg)
    else:
        for cfg in configs:
            print(shlex.join(training_command(cfg)))


if __name__ == '__main__':
    main()
