# run_debug.py (place this NEXT TO the 'lipid_analysis/' folder)
import sys
from pathlib import Path
from lipid_analysis.cli import main

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Simulate CLI args
sys.argv = [
    "run_debug.py",
    "--config", r"C:\Users\clchr\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD4d.py",
    "--root", r"C:\Users\clchr\OneDrive - Stanford",
    "--verbose"
]

main()
