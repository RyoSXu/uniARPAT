"""Disposable D1 train/valid checks, reusing the N1 real-data preflight."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.checks import check_global_norm as norm
from utils.local_structure_experiment import DEFAULT_REFERENCE
from utils.paths import file_sha256


def code_fingerprints():
    return {**norm.code_fingerprints(), **{name: file_sha256(ROOT / name) for name in
        ("utils/attention_dropout_experiment.py", "tools/checks/check_attention_dropout.py",
         "tools/eval/attention_dropout_compare.py")}}


def check(out, reference_dir, tag_prefix):
    return norm.check(out, reference_dir, tag_prefix, route="D1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / DEFAULT_REFERENCE)
    parser.add_argument("--tag-prefix", required=True)
    args = parser.parse_args()
    check(args.out, args.reference, args.tag_prefix)


if __name__ == "__main__":
    main()
