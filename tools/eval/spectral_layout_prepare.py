"""Prepare only L1 G1 G2 G3 L2 G4 G5 G6 against completed staged LR2 E+S."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval.spectral_residual_prepare import prepare as prepare_residual
from utils.paths import input_path


def prepare(control_preparation, out, tag_prefix):
    return prepare_residual(control_preparation, out, tag_prefix, midpoint=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-preparation', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--tag-prefix', required=True)
    args = parser.parse_args()
    report = prepare(input_path(args.control_preparation), input_path(args.out), args.tag_prefix)
    print(shlex.join(report['command']))


if __name__ == '__main__':
    main()
