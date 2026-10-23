"""Plan or explicitly execute four equal-depth local/global comparison arms.

Default action prints commands only. Budget, seeds, depth and valid checkpoint
criterion must be supplied. No test loader/evaluation is created by these arms.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from tools.train import build_arg_parser, train_and_eval
from utils.experiment_config import ExperimentConfig
from utils.checkpoint_selection import selected_history_row, selection_mode
from utils.local_global_experiment import comparison_configs, comparison_contract
from utils.paths import repository_path


def training_command(cfg: ExperimentConfig) -> list[str]:
    defaults = asdict(ExperimentConfig())
    command = [sys.executable, str(ROOT / 'tools/train.py')]
    explicit = {'epochs', 'batch_size', 'lr', 'seed', 'tag'}
    for key, value in asdict(cfg).items():
        if value is None or (key not in explicit and value == defaults[key]):
            continue
        name = 'model' if key == 'model_name' else key
        if isinstance(value, bool):
            if value:
                command.append(f'--{name}')
        else:
            command.extend((f'--{name}', str(value)))
    return command


def summarize(configs):
    """Read the selected valid epoch for each arm; incomplete arms stay pending."""
    rows = []
    for cfg in configs:
        path = repository_path(f'results/reports/history_{cfg.model_name.lower()}{cfg.tag}.csv')
        row = {'route': cfg.local_route, 'layout': cfg.local_layout, 'depth': cfg.local_depth,
               'seed': cfg.seed, 'tag': cfg.tag, 'selection_metric': cfg.selection_metric}
        if not path.is_file():
            rows.append({**row, 'status': 'pending'})
            continue
        config_path = repository_path(f'results/runs/ablation_{cfg.model_name.lower()}{cfg.tag}/config_used.yaml')
        if not config_path.is_file():
            raise ValueError(f'cannot verify comparison {cfg.tag}: missing configuration {config_path}; '
                             f'history exists at {path}; restore the original config_used.yaml before summarizing')
        with config_path.open() as stream:
            saved = yaml.safe_load(stream)
        params = saved['config']['model']['params']['sub_model']['transformer']
        identity = {'local_route': cfg.local_route, 'local_layout': cfg.local_layout,
                    'local_depth': cfg.local_depth, 'global_depth': cfg.global_depth,
                    'atom_feat_mode': cfg.atom_feat}
        if any(params.get(key) != value for key, value in identity.items()) \
                or saved.get('model_contract') != comparison_contract(cfg, params):
            raise ValueError(f'comparison configuration mismatch: {config_path}')
        history = pd.read_csv(path)
        if history.empty or cfg.selection_metric not in history:
            raise ValueError(f'incomplete comparison metric schema: {path}')
        chosen = selected_history_row(history, cfg.selection_metric, saved['model_contract'].get('spectral_balance'))
        if chosen is None:
            rows.append({**row, 'status': 'no_eligible_best',
                         'completed_epochs': int(history.epoch.max()), 'best_epoch': None})
            continue
        rows.append({**row, 'status': 'complete' if int(history.epoch.max()) >= cfg.epochs else 'partial',
                     'selection_mode': selection_mode(cfg.selection_metric),
                     'completed_epochs': int(history.epoch.max()), 'best_epoch': int(chosen.epoch),
                     **{key: float(chosen[key]) for key in history.columns
                        if key.startswith(('blind_', 'oracle_')) or key in (
                            'balanced_score', 'epoch_time_s', 'peak_vram_mb')}})
    return rows


def main():
    parser = build_arg_parser()
    parser.description = __doc__
    for name in ('epochs', 'batch_size', 'lr', 'local_depth', 'selection_metric'):
        parser._option_string_actions[f'--{name}'].required = True
    parser.set_defaults(model='M1', atom_feat='z_only_proj', skip_test_eval=True, q1_coord=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--tag_prefix', required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--run', action='store_true', help='Execute the supplied training budget sequentially')
    action.add_argument('--summarize', action='store_true', help='Read existing valid histories without training')
    args = parser.parse_args()
    if args.model != 'M1' or args.local_route != 'none' or args.local_layout is not None:
        parser.error('four-arm planning supplies route/layout itself and requires M1')
    configs = comparison_configs(ExperimentConfig.from_args(args), depth=args.local_depth,
                                 seeds=args.seeds, tag_prefix=args.tag_prefix)
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
