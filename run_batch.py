from pathlib import Path
import subprocess, sys

lipid_analysis_path = r"D:\OneDrive - Stanford\Research Documents\Python Scripts\lipid_analysis.py"
# List of config files (full paths or relative paths)
config_files = [
    # r"D:\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD3a.py",
    # r"D:\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD3b.py",
    # r"D:\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD3c.py",
    # r"D:\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD3d.py",
    # r"D:\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD3e.py",
    # r"D:\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD3f.py",
    # r"D:\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD4a.py",
    # r"D:\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD4b.py",
    r"D:\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD4c.py",
    # r"D:\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD4d.py",
    # r"D:\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD4e.py",
    # r"D:\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD4f.py"
]

failures = []
for cfg_path in config_files:
    print(f"\n[RUN_BATCH] Running lipid_analysis with config: {cfg_path}")
    cmd = [sys.executable, lipid_analysis_path, "--config", cfg_path]
    try:
        res = subprocess.run(
            cmd, check=True, text=True, capture_output=True,
            cwd=Path(lipid_analysis_path).parent
        )
        if res.stdout:
            print(res.stdout)
    except subprocess.CalledProcessError as e:
        print("\n[RUN_BATCH] FAILED")
        print("[RUN_BATCH] Return code:", e.returncode)
        if e.stdout: print("\n[STDOUT]\n", e.stdout)
        if e.stderr: print("\n[STDERR]\n", e.stderr)
        failures.append(cfg_path)
        
if failures:
    print("\n[RUN_BATCH] Failures:")
    for f in failures:
        print(" -", f)