"""Real-batch full A preflight for matched joint-blind local-module selection."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.checks.check_i1_a import check as check_a, code_fingerprints as a_fingerprints
from tools.checks.check_local_readout import code_fingerprints as readout_fingerprints, permutation_case
from tools.eval.local_module_compare import (
    DEFAULT_EVIDENCE, PREFLIGHT_SCHEMA, candidate_config, reference_identity,
)
from utils.paths import file_sha256


def code_fingerprints():
    return {**a_fingerprints(), **readout_fingerprints(), **{
        name: file_sha256(ROOT / name) for name in
        ("tools/checks/check_local_module.py", "tools/eval/local_module_compare.py")}}


def check(out, evidence_dir, tag_prefix):
    check_a(out, evidence_dir, tag_prefix, identity_loader=reference_identity,
            configuration_factory=candidate_config, fingerprint_provider=code_fingerprints,
            report_schema=PREFLIGHT_SCHEMA, permutation_probe=permutation_case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--tag-prefix", required=True)
    args = parser.parse_args()
    check(args.out, args.references, args.tag_prefix)


if __name__ == "__main__":
    main()
