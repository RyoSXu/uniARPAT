"""Export per-sample valid results from a run's selected best checkpoint.

Works for the baseline arm and the four local/global arms alike: the saved
configuration rebuilds the model, the saved contract is verified against both
the configuration and the checkpoint, and the selected epoch is re-derived from
the history with the established ``balanced_score`` rule before any export.
Sample rows are labelled with material IDs so runs can be paired.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from tools.train import evaluate_split
from utils.builder import ConfigBuilder
from utils.paths import repository_path


def load_run(run_dir: Path):
    saved = yaml.safe_load((run_dir / 'config_used.yaml').read_text())
    params = saved['config']['model']['params']['sub_model']['transformer']
    contract = saved.get('model_contract')
    if contract is None or contract.get('transformer') != params:
        raise ValueError(f'configuration does not describe its own contract: {run_dir}')
    return saved, params, contract


def select_best(run_dir: Path, selection_metric: str, saved) -> dict:
    """Re-derive the selected epoch with the established balanced_score rule."""
    # config_used.yaml records no tag; the runner derives suffix = model+tag and
    # writes results/runs/ablation_{suffix} plus results/reports/history_{suffix}.
    suffix = run_dir.name.removeprefix('ablation_')
    if not suffix.startswith(str(saved['cli']['model']).lower()):
        raise ValueError(f'run directory does not start with its model name: {run_dir}')
    history_path = repository_path(f'results/reports/history_{suffix}.csv')
    history = pd.read_csv(history_path)
    if history.empty or 'balanced_score' not in history:
        raise ValueError(f'incomplete history for {suffix}')
    if not (history['balanced_score'] == history[selection_metric]).all():
        raise ValueError(f'history selection metric mismatch for {suffix}: {selection_metric}')
    chosen = history.loc[history['balanced_score'].idxmin()]
    checkpoint = torch.load(run_dir / 'checkpoint_best.pth', map_location='cpu')
    if int(checkpoint['epoch']) != int(chosen.epoch):
        raise ValueError(f'best checkpoint epoch {checkpoint["epoch"]} != history {chosen.epoch}')
    if not np.isclose(float(checkpoint['best_val_score']), float(chosen.balanced_score)):
        raise ValueError(f'best checkpoint score does not match the history minimum: {suffix}')
    if checkpoint.get('model_contract') != saved.get('model_contract'):
        raise ValueError(f'checkpoint contract differs from the run configuration: {suffix}')
    return {'checkpoint': checkpoint, 'row': chosen, 'history': history, 'tag': suffix}


def export(run_dir: Path, out: Path, summary_out: Path) -> None:
    if out.exists() or summary_out.exists():
        raise FileExistsError(f'refusing existing export: {out} or {summary_out}')
    torch.set_num_threads(1)
    saved, params, contract = load_run(run_dir)
    selection_metric = contract['selection_metric']
    picked = select_best(run_dir, selection_metric, saved)
    checkpoint = picked['checkpoint']

    builder = ConfigBuilder(**saved['config'])
    model = builder.get_model()
    model.model['transformer'].load_state_dict(checkpoint['model'], strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loader = builder.get_dataloader(split='valid', dos_minmax=True,
                                    batch_size=saved['cli']['batch_size'], dos_sumnorm=True)
    order = list(loader.sampler)
    material_ids = [str(value) for value in loader.dataset.material_ids]
    metrics = evaluate_split(model, loader, sumnorm=True, return_sample_level=True,
                             sample_ids=[material_ids[index] for index in order])
    frame = metrics.pop('sample_df')
    if len(frame) != len(order) or frame['material_id'].duplicated().any():
        raise ValueError(f'sample labels are not a one-to-one valid partition: {run_dir}')
    if not np.isfinite(frame.select_dtypes('number').to_numpy()).all():
        raise ValueError(f'non-finite exported metrics: {run_dir}')

    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    row = picked['row']
    summary = {
        'tag': picked['tag'], 'run_dir': str(run_dir), 'samples': len(frame),
        'contract_schema': contract['schema'], 'init_protocol': contract['init_protocol'],
        'selection_metric': selection_metric,
        'best_epoch': int(row.epoch), 'best_score': float(row.balanced_score),
        'checkpoint_epoch': int(checkpoint['epoch']),
        'epoch_time_s_median': float(picked['history'].epoch_time_s.median()),
        'epoch_time_s_sum': float(picked['history'].epoch_time_s.sum()),
        'peak_vram_mb_median': float(picked['history'].peak_vram_mb.median()),
        'peak_vram_mb_max': float(picked['history'].peak_vram_mb.max()),
        'epochs': int(picked['history'].epoch.max()),
        'recomputed_valid': {key: value for key, value in metrics.items()
                             if key.startswith(('blind_', 'oracle_')) and key.endswith('_median')
                             or key.startswith('balanced_')},
        'history_best_row': {key: float(row[key]) for key in row.index
                             if key.startswith(('blind_', 'oracle_')) or key in (
                                 'balanced_score', 'balanced_oracle_mae', 'balanced_blind_mae')},
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'history_best_row'},
                     ensure_ascii=False, indent=2, default=float))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--summary-out', type=Path, required=True)
    args = parser.parse_args()
    export(args.run_dir, args.out, args.summary_out)


if __name__ == '__main__':
    main()
