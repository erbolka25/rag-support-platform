"""Run every test suite of the workspace: python scripts/run_tests.py [pytest args].

Each suite runs in its own pytest process from its own directory: both packages keep a
top-level ``tests`` package (tests import helpers as ``tests.<module>``), so they cannot
share one interpreter.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ["packages/rag_core", "apps/support_copilot", "eval"]


def main() -> int:
    failed = []
    for suite in SUITES:
        print(f"\n=== {suite}", flush=True)
        code = subprocess.call([sys.executable, "-m", "pytest", *sys.argv[1:]], cwd=ROOT / suite)
        if code != 0:
            failed.append(suite)
    print("\nAll suites passed." if not failed else f"\nFailed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
