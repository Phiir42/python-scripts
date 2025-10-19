# run_debug.py
"""
Convenience runner for the lipid_analysis pipeline from an IDE (e.g., Spyder).

Place this file NEXT TO the 'lipid_analysis/' package folder, then run it.
It safely amends sys.path, temporarily patches sys.argv to simulate CLI flags,
and invokes lipid_analysis.cli.main().
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Iterator, List


@contextlib.contextmanager
def _patched_argv(new_argv: List[str]) -> Iterator[None]:
    """Temporarily replace sys.argv for argparse-based entrypoints."""
    old = sys.argv[:]
    sys.argv = list(new_argv)
    try:
        yield
    finally:
        sys.argv = old


def _project_root() -> Path:
    """
    Best-effort project root resolution:
    - Prefer the directory of this script if __file__ exists.
    - Fall back to the current working directory (IDE-safe).
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:
        # __file__ may be missing in some IDE run modes
        return Path.cwd()


def main() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Import after sys.path is set so the adjacent package is resolvable.
    from lipid_analysis.cli import main as cli_main

    # ---- Simulate CLI args (edit these paths/flags as needed) ----
    args = [
        "run_debug.py",
        "--config",
        r"C:\Users\clchr\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD4d.py",
        "--root",
        r"C:\Users\clchr\OneDrive - Stanford",
        "--verbose",
    ]

    with _patched_argv(args):
        cli_main()


if __name__ == "__main__":
    main()
