from pathlib import Path
import subprocess
import sys

# --- set your run root ONCE here ---
RUN_ROOT = Path(r"D:\OneDrive - Stanford")  # the --root you want

PROJECT_ROOT = Path(__file__).resolve().parent
PKG_DIR = PROJECT_ROOT / "lipid_analysis"
if not PKG_DIR.is_dir():
    raise SystemExit(f"Could not find package at: {PKG_DIR}")

# Build config dir from RUN_ROOT so it works no matter the drive
CONFIG_DIR = RUN_ROOT / r"Research Documents\Python Scripts\config_files"

# Option A: explicitly list files you want (under CONFIG_DIR)
config_files = sorted(CONFIG_DIR.glob("config_AD*.py"))
if not config_files:
    raise SystemExit(f"No configs matched in {CONFIG_DIR}")

# (Optional) sanity check before running
missing = [p for p in config_files if not p.exists()]
if missing:
    print("[RUN_BATCH] Missing config files:")
    for m in missing:
        print(" -", m)
    # You can sys.exit(1) here if you want to fail fast
    # sys.exit(1)

failures = []
for cfg_path in config_files:
    print(f"\n[RUN_BATCH] Running lipid_analysis with config: {cfg_path}")
    cmd = [
        sys.executable, "-m", "lipid_analysis",
        "--config", str(cfg_path),
        "--root", str(RUN_ROOT),
    ]
    try:
        res = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
            cwd=PROJECT_ROOT,  # run from project root; does not affect file locations
        )
        if res.stdout:
            print(res.stdout)
        if res.stderr:
            print(res.stderr)
    except subprocess.CalledProcessError as e:
        print("\n[RUN_BATCH] FAILED")
        print("[RUN_BATCH] Return code:", e.returncode)
        if e.stdout:
            print("\n[STDOUT]\n", e.stdout)
        if e.stderr:
            print("\n[STDERR]\n", e.stderr)
        failures.append(str(cfg_path))

if failures:
    print("\n[RUN_BATCH] Failures:")
    for f in failures:
        print(" -", f)
