"""Analyze the phDOS head depth study with the rule fixed in the 2026-11-06 design.

Each arm (h1: one Conv1d; h2: one hidden layer) is compared with R42, the
depth-6 run with the same seed, initial non-head tensors and data order.
Verdict per arm, on valid blind mean E (arm minus R42):
  worse            joint > +0.004, or phDOS > +0.0099 (3 x its seed range), or no guard-eligible epoch
  better           joint < -0.004
  not_distinguished otherwise
The R42-R44 seed range and mean are reported for context. Train inference on
the selected checkpoints gives the train/valid diagnostic; never used for selection.
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

from tools.eval.phdos_head_prepare import SCHEMA as PREPARATION_SCHEMA
from tools.eval.reference_controls_analyze import TASKS, interval, summarize
from tools.eval.structure_generalization_analyze import REFERENCE_MEAN, REPLICATES, context, export_arms, load
from utils.paths import file_sha256, input_path

DRAWS = 2000
SEED = 20261107
SEED_RANGE = {'joint': 0.001319, 'edos': 0.000651, 'phdos': 0.003289}
CLEAR = 3.
R42_TRAIN = ROOT / 'results/runs/structure_use_audit_20261105/infer'


def verified_preparation(path):
    preparation = json.loads(Path(path).read_text())
    if preparation.get('schema') != PREPARATION_SCHEMA:
        raise ValueError('not a phDOS head preparation report')
    for item, digest in preparation['input_sha256'].items():
        if file_sha256(item) != digest:
            raise ValueError(f'frozen input changed since preparation: {item}')
    return preparation


def verdict(delta, eligible):
    if not eligible or delta['joint'] > CLEAR * SEED_RANGE['joint'] or delta['phdos'] > CLEAR * SEED_RANGE['phdos']:
        return 'worse'
    if delta['joint'] < -CLEAR * SEED_RANGE['joint']:
        return 'better'
    return 'not_distinguished'


def history_reading(arm):
    """Fallback when no epoch passes the guards: best unguarded epoch from the full-valid history."""
    history = pd.read_csv(arm['history'])
    row = history.loc[history.balanced_blind_energy.idxmin()]
    return {'epoch': int(row.epoch), 'joint': float(row.balanced_blind_energy),
            'edos': float(row.blind_energy_edos_mean), 'phdos': float(row.blind_energy_phdos_mean),
            'eligible_epochs': int(history.checkpoint_eligible.sum())}


def train_ratio(samples, valid_summary):
    frame = pd.read_csv(samples)
    return {t: {'train_mean_E': float(frame[f'blind_energy_{t}'].mean()),
                'train_over_valid': float(frame[f'blind_energy_{t}'].mean() / valid_summary[f'{t}_mean_E'])}
            for t in TASKS}


def analyze(preparation_path, out, train_inference=True):
    preparation = verified_preparation(preparation_path)
    out = Path(out)
    if (out / 'report.json').exists():
        raise FileExistsError(f'refusing existing analysis: {out}')
    protocol = json.loads((Path(preparation_path).parent / 'protocol.json').read_text())
    ids, atoms, _, sources = context(preparation)
    exports, status = export_arms(preparation, set(preparation['step1_order']), out)
    values, _, checks = load({**{k: sources[k] for k in REPLICATES}, **exports}, ids, atoms, protocol)
    seed_mean = float(np.mean([values[n]['energy_joint'].mean() for n in REPLICATES]))
    if abs(seed_mean - REFERENCE_MEAN) > 5e-7:
        raise ValueError(f'R42-R44 seed mean changed: {seed_mean}')
    draws = np.random.default_rng(SEED).integers(0, len(ids), size=(DRAWS, len(ids)))
    reference = summarize(values['lr2_s42'])
    report = {'schema': 'lr2-es-phdos-head-analysis-v1', 'valid_materials': len(ids),
              'reference': {'name': 'lr2_s42', 'phdos_head_layers': 6, 'summary': reference,
                            'seed_mean_joint': seed_mean, 'seed_range': SEED_RANGE},
              'bootstrap': {'draws': DRAWS, 'seed': SEED, 'unit': 'material'},
              'export_status': status, 'export_checks': checks, 'arms': {}}
    for arm in preparation['arms']:
        name = arm['name']
        entry = {'phdos_head_layers': arm['phdos_head_layers'], 'export': status[name]}
        if name in values:
            summary = summarize(values[name])
            delta = {'joint': summary['joint_mean_E'] - reference['joint_mean_E'],
                     **{t: summary[f'{t}_mean_E'] - reference[f'{t}_mean_E'] for t in TASKS}}
            entry.update(summary=summary, delta_vs_r42=delta, eligible=True,
                         paired_vs_r42={t: interval(values['lr2_s42'][f'energy_{t}'], values[name][f'energy_{t}'],
                                                    draws) for t in ('joint', *TASKS)})
        else:
            reading = history_reading(arm)
            delta = {t: reading[t] - (reference['joint_mean_E'] if t == 'joint' else reference[f'{t}_mean_E'])
                     for t in ('joint', *TASKS)}
            entry.update(history_reading=reading, delta_vs_r42=delta, eligible=False)
        entry['verdict'] = verdict(delta, entry['eligible'])
        report['arms'][name] = entry
    if train_inference:
        from tools.eval.structure_use_audit import infer

        target = out / 'train_inference'
        runs = [Path(a['run_dir']) for a in preparation['arms'] if a['name'] in exports]
        pending = [run for run in runs if not (target / f'{run.name}_train_samples.csv').exists()]
        if pending:
            infer(pending, 'train', target)
        r42 = Path(preparation['completed_control']['run_dir']).name
        report['reference']['train'] = train_ratio(R42_TRAIN / f'{r42}_train_samples.csv', reference)
        for arm in preparation['arms']:
            path = target / f"{Path(arm['run_dir']).name}_train_samples.csv"
            if path.exists():
                report['arms'][arm['name']]['train'] = train_ratio(path, report['arms'][arm['name']]['summary'])
    verdicts = {name: entry['verdict'] for name, entry in report['arms'].items()}
    if verdicts.get('h1_s42') in ('better', 'not_distinguished'):
        conclusion = 'depth_6_not_needed: a single Conv1d phDOS head is not worse on valid'
    elif verdicts.get('h2_s42') in ('better', 'not_distinguished'):
        conclusion = 'one_hidden_layer_suffices: h1 is worse, h2 is not'
    else:
        conclusion = 'depth_needed: both shallower heads are worse than depth 6'
    report.update(conclusion=conclusion,
                  scope='valid only, blind; single seed 42 per arm against R42 (same seed, non-head initial tensors '
                        'and data order); repaired Q1, 35 epochs; diagnostic, not an adoption')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False,
                                                default=float) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preparation', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--no-train-inference', action='store_true')
    args = parser.parse_args()
    report = analyze(input_path(args.preparation), input_path(args.out), not args.no_train_inference)
    for name, entry in report['arms'].items():
        print(name, entry['verdict'], {k: round(v, 6) for k, v in entry['delta_vs_r42'].items()})
    print(report['conclusion'])


if __name__ == '__main__':
    main()
