"""Prepare one LR2 E+S intervention against a completed control; never train."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.local_global_compare import training_command
from tools.eval.spectral_balance_prepare import completed_spectral_control
from utils.experiment_config import ExperimentConfig
from utils.local_global_experiment import training_record
from utils.local_residual_experiment import residual_ablation_contract
from utils.paths import file_sha256, input_path
from utils.spectral_balance import load_balance_protocol
from utils.spectral_balance_experiment import validate_balance_config


def prepare(control_preparation, out, tag_prefix, *, midpoint=False):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'use a new output directory: {out}')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', tag_prefix):
        raise ValueError('isolated alphanumeric tag prefix required')
    if not isinstance(midpoint, bool):
        raise ValueError('midpoint must be boolean')
    original = json.loads(input_path(control_preparation).read_text())
    reference = ExperimentConfig(**original['configuration'])
    protocol = load_balance_protocol(reference.spectral_balance_config, reference.data_dir)
    control = completed_spectral_control(control_preparation, protocol, input_path(reference.data_dir),
                                        shape_aux=True, local_route='LR2' if midpoint else 'I3')
    if midpoint and control['selected_epoch'] != 29:
        raise ValueError('midpoint comparison requires the adopted staged LR2 E+S ep29 control')
    candidate = replace(reference, local_route='LR2', skip_existing=False,
                        local_layout='midpoint' if midpoint else reference.local_layout,
                        tag=f'_sbal_{tag_prefix}_lr2_{"midpoint_" if midpoint else ""}s42_e35')
    validate_balance_config(candidate)
    path = out / 'protocol.json'
    candidate = replace(candidate, spectral_balance_config=str(path.resolve()))
    before, after = training_record(reference), training_record(candidate)
    changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    expected = {'local_layout' if midpoint else 'local_route', 'spectral_balance_config'}
    if changed != expected:
        raise ValueError(f'LR2 E+S changes more than its single intervention and identical protocol path: {changed}')
    run = ROOT / f'results/runs/ablation_{candidate.model_name.lower()}{candidate.tag}'
    history = ROOT / f'results/reports/history_{candidate.model_name.lower()}{candidate.tag}.csv'
    if run.exists() or history.exists():
        raise FileExistsError('LR2 trial requires a fresh tag without existing run/history')
    protocol_source = input_path(reference.spectral_balance_config)
    protocol_bytes = protocol_source.read_bytes()
    sources = [p for directory in ('model', 'utils') for p in (ROOT / directory).rglob('*.py')]
    sources += [ROOT / name for name in ('tools/train.py', 'tools/eval/spectral_residual_prepare.py',
                'tools/eval/spectral_layout_prepare.py',
                'tools/eval/spectral_balance_prepare.py', 'tools/eval/lg_valid_export.py',
                'tools/checks/check_spectral_balance.py', 'tools/checks/check_long_budget.py')]
    hashes = {**control['input_sha256'],
              str(input_path(reference.data_dir) / 'manifest.json'):
                  file_sha256(input_path(reference.data_dir) / 'manifest.json')}
    for name, digest in {**protocol['train_sha256'], **protocol['valid_guard_sha256']}.items():
        hashes[str(input_path(reference.data_dir) / name)] = digest
    hashes.update({str(p): file_sha256(p) for p in sources})
    if any(file_sha256(p) != digest for p, digest in hashes.items()):
        raise ValueError('an input changed during preparation')
    out.mkdir(parents=True, exist_ok=False)
    path.write_bytes(protocol_bytes)
    hashes[str(path.resolve())] = file_sha256(path)
    report = {'schema': 'joint-spectral-balance-residual-preparation-v1', 'training_started': False,
              'configuration': asdict(candidate), 'command': training_command(candidate),
              'completed_control': control, 'reference': protocol['reference'],
              'ablation': residual_ablation_contract(), 'changed_training_fields': sorted(changed),
              'protocol_copy_identical': path.read_bytes() == protocol_bytes,
              'input_sha256': hashes, 'not_a_new_selection_or_adoption': True}
    if midpoint:
        from utils.spectral_layout_experiment import layout_ablation_contract

        report.update(schema='joint-spectral-balance-layout-preparation-v1',
                      ablation=layout_ablation_contract())
    for source in sources:
        target = out / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-preparation', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--tag-prefix', required=True)
    args = parser.parse_args()
    result = prepare(input_path(args.control_preparation), input_path(args.out), args.tag_prefix)
    print(shlex.join(result['command']))


if __name__ == '__main__':
    main()
