import subprocess
import sys

lipid_analysis_path = r"C:\Users\clchr\OneDrive - Stanford\Research Documents\Python Scripts\lipid_analysis.py"
# List of config files (full paths or relative paths)
config_files = [
    r"C:\Users\clchr\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD3a.py",
    r"C:\Users\clchr\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD3b.py",
    r"C:\Users\clchr\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD3c.py",
    r"C:\Users\clchr\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD3d.py",
    r"C:\Users\clchr\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD4a.py",
    r"C:\Users\clchr\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD4b.py",
    r"C:\Users\clchr\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD4c.py",
    r"C:\Users\clchr\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD4d.py"
]

for cfg_path in config_files:
    print(f"\n[RUN_BATCH] Running lipid_analysis with config: {cfg_path}")
    cmd = [
        sys.executable,
        lipid_analysis_path,      # full or relative path to lipid_analysis.py
        "--config",
        cfg_path                  # full or relative path to the config
    ]
    subprocess.run(cmd, check=True)