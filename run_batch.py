# run_batch.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import subprocess


def run_one(
    cfg_path: Path,
    project_root: Path,
    run_root: Path,
    verbose: bool,
    no_classify: bool,
    capture: bool,
) -> int:
    """
    Invoke: python -m lipid_analysis --config <cfg> --root <root> [--verbose] [--no_classify]
    Returns the process return code.
    """
    cmd = [
        sys.executable,
        "-m",
        "lipid_analysis",
        "--config",
        str(cfg_path),
        "--root",
        str(run_root),
    ]
    if verbose:
        cmd.append("--verbose")
    if no_classify:
        cmd.append("--no_classify")

    # Stream live output by default (nicer for long runs). Use capture only if requested.
    if capture:
        res = subprocess.run(
            cmd,
            cwd=str(project_root),
            text=True,
            capture_output=True,
        )
        if res.stdout:
            print(res.stdout, end="")
        if res.stderr:
            print(res.stderr, end="")
        return int(res.returncode)
    else:
        # Inherit parent stdio for live printing
        return int(
            subprocess.call(
                cmd,
                cwd=str(project_root),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-run lipid_analysis over a set of config files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"C:\Users\clchr\OneDrive - Stanford"),
        help="Root path override passed to lipid_analysis (--root).",
    )
    parser.add_argument(
        "--pattern",
        default="config_AD*.py",
        help="Glob pattern under the config_files folder to select configs.",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Capture child output and print after completion (default streams live).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass --verbose to lipid_analysis.",
    )
    parser.add_argument(
        "--no-classify",
        action="store_true",
        help="Pass --no_classify to lipid_analysis.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the configs that would be run, then exit.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    pkg_dir = project_root / "lipid_analysis"
    if not pkg_dir.is_dir():
        raise SystemExit(f"[RUN_BATCH] Could not find package at: {pkg_dir}")

    # Configs live under: <RUN_ROOT>/Research Documents/Python Scripts/config_files
    config_dir = args.root / r"Research Documents\Python Scripts\config_files"
    if not config_dir.is_dir():
        raise SystemExit(f"[RUN_BATCH] Config directory not found: {config_dir}")

    config_files = sorted(config_dir.glob(args.pattern))
    if not config_files:
        raise SystemExit(f"[RUN_BATCH] No configs matched: {config_dir}\\{args.pattern}")

    print(f"[RUN_BATCH] Project root: {project_root}")
    print(f"[RUN_BATCH] Using --root:   {args.root}")
    print(f"[RUN_BATCH] Config dir:     {config_dir}")
    print(f"[RUN_BATCH] Matched {len(config_files)} config(s):")
    for p in config_files:
        print("  -", p)

    if args.dry_run:
        print("\n[RUN_BATCH] Dry run only. Exiting.")
        return

    failures: list[Path] = []
    for cfg_path in config_files:
        print(f"\n[RUN_BATCH] Running lipid_analysis with config:\n  {cfg_path}")
        rc = run_one(
            cfg_path=cfg_path,
            project_root=project_root,
            run_root=args.root,
            verbose=args.verbose,
            no_classify=args.no_classify,
            capture=args.capture,
        )
        if rc != 0:
            print(f"[RUN_BATCH] FAILED (rc={rc}) → {cfg_path}")
            failures.append(cfg_path)
        else:
            print(f"[RUN_BATCH] SUCCESS → {cfg_path}")

    if failures:
        print("\n[RUN_BATCH] Failures:")
        for f in failures:
            print("  -", f)
        # non-zero exit to make CI/schedulers aware
        sys.exit(1)
    else:
        print("\n[RUN_BATCH] All runs completed successfully.")


if __name__ == "__main__":
    main()
