"""Analyze the MACE frozen-feature reference M1 with the rule fixed in the 2026-11-08 design.

M1 (h2 + frozen MACE invariant per-atom features) is compared with h2_s42, the
working point with the same seed, initial tensors (the feature projection
starts at zero) and data order. Verdict on valid blind mean E (M1 minus h2):
  effective          joint < -0.004 (about 3 x the seed range)
  too_small          -0.004 <= joint < -0.001319
  not_distinguished  -0.001319 <= joint <= +0.004
  worse              joint > +0.004, or phDOS > +0.0099 (3 x its seed range), or no guard-eligible epoch
Train inference on the selected checkpoint gives the train/valid diagnostic; never used for selection.
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

from tools.eval.mace_features_prepare import SCHEMA as PREPARATION_SCHEMA
from tools.eval.phdos_head_analyze import history_reading, train_ratio
from tools.eval.reference_controls_analyze import TASKS, interval, summarize
from tools.eval.structure_generalization_analyze import context, export_arms, load
from utils.paths import file_sha256, input_path

DRAWS = 2000
SEED = 20261108
SEED_RANGE = {'joint': 0.001319, 'edos': 0.000651, 'phdos': 0.003289}
CLEAR = 3.
ARM = 'm1_s42'
REFERENCE = 'h2_s42'


def verified_preparation(path):
    preparation = json.loads(Path(path).read_text())
    if preparation.get('schema') != PREPARATION_SCHEMA:
        raise ValueError('not a MACE feature preparation report')
    for item, digest in preparation['input_sha256'].items():
        if file_sha256(item) != digest:
            raise ValueError(f'frozen input changed since preparation: {item}')
    return preparation


def verdict(delta, eligible):
    if not eligible or delta['joint'] > CLEAR * SEED_RANGE['joint'] or delta['phdos'] > CLEAR * SEED_RANGE['phdos']:
        return 'worse'
    if delta['joint'] < -CLEAR * SEED_RANGE['joint']:
        return 'effective'
    if delta['joint'] < -SEED_RANGE['joint']:
        return 'too_small'
    return 'not_distinguished'


CONCLUSIONS = {
    'effective': 'pretrained_representation_valuable: run the element-only M0 attribution and design own force pretraining',
    'too_small': 'marginal_value: force-pretrained representation helps too little; lower pretraining priority, move to data',
    'not_distinguished': 'no_measurable_value: force-pretrained representation does not help valid; move to data',
    'worse': 'worse: check feature scale or injection before any conclusion (no automatic retuning)',
}


def analyze(preparation_path, out, train_inference=True):
    preparation = verified_preparation(preparation_path)
    out = Path(out)
    if (out / 'report.json').exists():
        raise FileExistsError(f'refusing existing analysis: {out}')
    control = preparation['completed_control']
    protocol = json.loads((Path(preparation_path).parent / 'protocol.json').read_text())
    head = json.loads(Path(control['head_preparation']).read_text())
    ids, atoms, _, sources = context(head)
    exports, status = export_arms(preparation, set(preparation['step1_order']), out)
    values, _, checks = load({'lr2_s42': sources['lr2_s42'],
                              REFERENCE: (Path(control['samples']), Path(control['spectra'])), **exports},
                             ids, atoms, protocol)
    draws = np.random.default_rng(SEED).integers(0, len(ids), size=(DRAWS, len(ids)))
    reference = summarize(values[REFERENCE])
    if abs(reference['joint_mean_E'] - control['best_score']) > 1e-5:
        raise ValueError('h2 export no longer reproduces its selected score')
    report = {'schema': 'lr2-es-mace-features-analysis-v1', 'valid_materials': len(ids),
              'reference': {'name': REFERENCE, 'epoch': control['selected_epoch'], 'summary': reference,
                            'seed_range': SEED_RANGE},
              'bootstrap': {'draws': DRAWS, 'seed': SEED, 'unit': 'material'},
              'export_status': status, 'export_checks': checks, 'arms': {}}
    for arm in preparation['arms']:
        name = arm['name']
        entry = {'external_atom_features': arm['external_atom_features'], 'export': status[name]}
        if name in values:
            summary = summarize(values[name])
            delta = {'joint': summary['joint_mean_E'] - reference['joint_mean_E'],
                     **{t: summary[f'{t}_mean_E'] - reference[f'{t}_mean_E'] for t in TASKS}}
            entry.update(summary=summary, delta_vs_h2=delta, eligible=True,
                         paired_vs_h2={t: interval(values[REFERENCE][f'energy_{t}'], values[name][f'energy_{t}'],
                                                   draws) for t in ('joint', *TASKS)},
                         task_vs_own_range={t: delta[t] / SEED_RANGE[t] for t in TASKS})
        else:
            reading = history_reading(arm)
            delta = {t: reading[t] - (reference['joint_mean_E'] if t == 'joint' else reference[f'{t}_mean_E'])
                     for t in ('joint', *TASKS)}
            entry.update(history_reading=reading, delta_vs_h2=delta, eligible=False)
        entry['verdict'] = verdict(delta, entry['eligible'])
        report['arms'][name] = entry
    if train_inference:
        from tools.eval.structure_use_audit import infer

        target = out / 'train_inference'
        runs = [Path(a['run_dir']) for a in preparation['arms'] if a['name'] in exports]
        pending = [run for run in runs if not (target / f'{run.name}_train_samples.csv').exists()]
        if pending:
            infer(pending, 'train', target)
        report['reference']['train'] = train_ratio(control['train_samples'], reference)
        for arm in preparation['arms']:
            path = target / f"{Path(arm['run_dir']).name}_train_samples.csv"
            if path.exists():
                report['arms'][arm['name']]['train'] = train_ratio(path, report['arms'][arm['name']]['summary'])
    report.update(conclusion=CONCLUSIONS[report['arms'][ARM]['verdict']],
                  scope='valid only, blind; single seed 42 against h2_s42 (same seed, initial tensors and data '
                        'order; zero-initialized feature projection); repaired Q1, 35 epochs; MACE-MP-0b3 was '
                        'pretrained on MPtrj, which contains valid/test structures but no DOS; diagnostic, not '
                        'an adoption')
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
        print(name, entry['verdict'], {k: round(v, 6) for k, v in entry['delta_vs_h2'].items()})
    print(report['conclusion'])


if __name__ == '__main__':
    main()
