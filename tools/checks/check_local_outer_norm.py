"""Bounded real-batch preflight for complete A's outer post-norm candidate."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.checks.check_local_norm import check
from tools.eval.local_norm_compare import DEFAULT_EVIDENCE, REFERENCE_COMMIT
from tools.eval.local_outer_norm_compare import OUTER_NORM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--reference-commit", default=REFERENCE_COMMIT)
    parser.add_argument("--tag-prefix", required=True)
    args = parser.parse_args()
    check(args.out, args.references, args.tag_prefix, args.reference_commit, protocol=OUTER_NORM)


if __name__ == "__main__":
    main()
