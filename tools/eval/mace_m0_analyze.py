"""Analyze the element-only MACE attribution arm M0 with the rule fixed in the 2026-11-09 design.

M0 (h2 + per-element train-mean MACE descriptors) is compared with h2_s42 and the
completed M1 (h2 + per-atom MACE descriptors); all three share seed, initial
tensors and data order. Primary verdict on valid blind joint mean E, d0 = M0 - h2:
  element_dominant      d0 < -0.004 (M0 alone is effective: element chemistry carries the gain)
  partial               -0.004 <= d0 < -0.001319
  environment_dominant  -0.001319 <= d0 <= +0.004 (M0 not distinguished from h2)
  worse                 d0 > +0.004, or phDOS d0 > +0.0099, or no guard-eligible epoch
Secondary eDOS reading (where M1's gain lies) from material-paired 95% intervals:
  element if M0-h2 excludes 0 and M0-M1 includes 0; environment if the reverse;
  partial if both exclude 0; unresolved if both include 0.
Same-composition skill and train/valid ratios are reported as diagnostics only.
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

from tools.eval.mace_m0_prepare import SCHEMA as PREPARATION_SCHEMA
from tools.eval.phdos_head_analyze import history_reading, train_ratio
from tools.eval.reference_controls_analyze import TASKS, interval, same_composition, summarize
from tools.eval.structure_generalization_analyze import context, export_arms, load
from utils.paths import file_sha256

DRAWS = 2000
SEED = 20261109
SEED_RANGE = {'joint': 0.001319, 'edos': 0.000651, 'phdos': 0.003289}
CLEAR = 3.
ARM = 'm0_s42'
REFERENCE = 'h2_s42'
M1 = 'm1_s42'


def verified_preparation(path):
    preparation = json.loads(Path(path).read_text())
    if preparation.get('schema') != PREPARATION_SCHEMA:
        raise ValueError('not a MACE M0 preparation report')
    for item, digest in preparation['input_sha256'].items():
        if file_sha256(item) != digest:
            raise ValueError(f'frozen input changed since preparation: {item}')
    return preparation


def verdict(delta, eligible):
    if not eligible or delta['joint'] > CLEAR * SEED_RANGE['joint'] or delta['phdos'] > CLEAR * SEED_RANGE['phdos']:
        return 'worse'
    if delta['joint'] < -CLEAR * SEED_RANGE['joint']:
        return 'element_dominant'
    if delta['joint'] < -SEED_RANGE['joint']:
        return 'partial'
    return 'environment_dominant'


def edos_reading(vs_h2, vs_m1):
    h2_clear, m1_clear = vs_h2['excludes_zero'], vs_m1['excludes_zero']
    if h2_clear and not m1_clear:
        return 'element'
    if m1_clear and not h2_clear:
        return 'environment'
    return 'partial' if h2_clear else 'unresolved'


CONCLUSIONS = {
    'element_dominant': 'element_prior: MACE mainly supplies better element chemistry; own structure pretraining is '
                        'weakly motivated; compare cheap element descriptors and move to data',
    'partial': 'both: element chemistry and environment each carry part of the gain; structure pretraining stays '
               'a candidate with a smaller expected return',
    'environment_dominant': 'environment: the gain needs per-atom environment information; pursue richer '
                            'pretrained representations and design own structure pretraining',
    'worse': 'worse: element-only features hurt; check feature scale or injection before any conclusion',
}


def analyze(preparation_path, out, train_inference=True):
    preparation = verified_preparation(preparation_path)
    out = Path(out)
    if (out / 'report.json').exists():
        raise FileExistsError(f'refusing existing analysis: {out}')
    control, m1 = preparation['completed_control'], preparation['completed_m1']
    protocol = json.loads((Path(preparation_path).parent / 'protocol.json').read_text())
    head = json.loads(Path(control['head_preparation']).read_text())
    ids, atoms, keys, sources = context(head)
    exports, status = export_arms(preparation, set(preparation['step1_order']), out)
    values, arrays, checks = load({'lr2_s42': sources['lr2_s42'],
                                   REFERENCE: (Path(control['samples']), Path(control['spectra'])),
                                   M1: (Path(m1['samples']), Path(m1['spectra'])), **exports},
                                  ids, atoms, protocol)
    draws = np.random.default_rng(SEED).integers(0, len(ids), size=(DRAWS, len(ids)))
    reference, m1_summary = summarize(values[REFERENCE]), summarize(values[M1])
    if abs(reference['joint_mean_E'] - control['best_score']) > 1e-5:
        raise ValueError('h2 export no longer reproduces its selected score')
    if abs(m1_summary['joint_mean_E'] - m1['best_score']) > 1e-5:
        raise ValueError('M1 export no longer reproduces its selected score')
    report = {'schema': 'lr2-es-mace-m0-analysis-v1', 'valid_materials': len(ids),
              'reference': {'name': REFERENCE, 'epoch': control['selected_epoch'], 'summary': reference,
                            'seed_range': SEED_RANGE},
              'm1': {'name': M1, 'epoch': m1['selected_epoch'], 'summary': m1_summary,
                     'delta_vs_h2': {'joint': m1_summary['joint_mean_E'] - reference['joint_mean_E'],
                                     **{t: m1_summary[f'{t}_mean_E'] - reference[f'{t}_mean_E'] for t in TASKS}}},
              'bootstrap': {'draws': DRAWS, 'seed': SEED, 'unit': 'material'},
              'export_status': status, 'export_checks': checks, 'arms': {},
              'same_composition': {name: same_composition(arrays[name], ids, keys)
                                   for name in (REFERENCE, M1, *exports)}}
    for arm in preparation['arms']:
        name = arm['name']
        entry = {'external_atom_features': arm['external_atom_features'], 'export': status[name]}
        if name in values:
            summary = summarize(values[name])
            delta = {'joint': summary['joint_mean_E'] - reference['joint_mean_E'],
                     **{t: summary[f'{t}_mean_E'] - reference[f'{t}_mean_E'] for t in TASKS}}
            vs_m1 = {'joint': summary['joint_mean_E'] - m1_summary['joint_mean_E'],
                     **{t: summary[f'{t}_mean_E'] - m1_summary[f'{t}_mean_E'] for t in TASKS}}
            paired_h2 = {t: interval(values[REFERENCE][f'energy_{t}'], values[name][f'energy_{t}'], draws)
                         for t in ('joint', *TASKS)}
            paired_m1 = {t: interval(values[M1][f'energy_{t}'], values[name][f'energy_{t}'], draws)
                         for t in ('joint', *TASKS)}
            m1_delta = report['m1']['delta_vs_h2']
            entry.update(summary=summary, delta_vs_h2=delta, delta_vs_m1=vs_m1, eligible=True,
                         paired_vs_h2=paired_h2, paired_vs_m1=paired_m1,
                         share_of_m1_gain={t: delta[t] / m1_delta[t] for t in ('joint', *TASKS)},
                         edos_reading=edos_reading(paired_h2['edos'], paired_m1['edos']))
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
        report['m1']['train'] = train_ratio(m1['train_samples'], m1_summary)
        for arm in preparation['arms']:
            path = target / f"{Path(arm['run_dir']).name}_train_samples.csv"
            if path.exists():
                report['arms'][arm['name']]['train'] = train_ratio(path, report['arms'][arm['name']]['summary'])
    report.update(conclusion=CONCLUSIONS[report['arms'][ARM]['verdict']],
                  scope='valid only, blind; single seed 42 against h2_s42 and M1 (same seed, initial tensors and '
                        'data order; zero-initialized feature projection); repaired Q1, 35 epochs; element-only '
                        'features are train means of MACE-MP-0b3 descriptors; diagnostic, not an adoption')
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
    from utils.paths import input_path

    report = analyze(input_path(args.preparation), input_path(args.out), not args.no_train_inference)
    for name, entry in report['arms'].items():
        print(name, entry['verdict'], entry.get('edos_reading'), {k: round(v, 6) for k, v in entry['delta_vs_h2'].items()})
    print(report['conclusion'])


if __name__ == '__main__':
    main()
