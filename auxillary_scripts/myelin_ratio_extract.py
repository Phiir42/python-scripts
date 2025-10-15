#!/usr/bin/env python3
import re
from pathlib import Path
import pandas as pd

# --------- USER-EDITABLE SETTINGS (test mode) ---------
INPUT_FILE = Path(r"C:\Users\clchr\OneDrive - Stanford\Research Documents\AD Project\2025\AD3a\Hyperspectral_Myelin_AverageFits.xlsx")
OUTPUT_FILE = Path(r"C:\Users\clchr\OneDrive - Stanford\Research Documents\AD Project\2025\Myelin_Ratio_ByClinicogenotype.xlsx")

# Sheet names for output (Excel-safe: no / \ ? * [ ] : )
SHEET_ALL = "All frames"
SHEET_CTRL = "Non-dementia control"
SHEET_AD33 = "AD APOE e3e3"
SHEET_AD44 = "AD APOE e4e4"

# --------- Regex helpers ---------
# Use alphanumeric-only boundaries so tokens like AD33_... still match
_re_ctrl  = re.compile(r"(?<![A-Za-z0-9])(?:Ctrl|Control)(?![A-Za-z0-9])", flags=re.IGNORECASE)
_re_ad33  = re.compile(r"(?<![A-Za-z0-9])AD33(?![A-Za-z0-9])", flags=re.IGNORECASE)
_re_ad44  = re.compile(r"(?<![A-Za-z0-9])AD44(?![A-Za-z0-9])", flags=re.IGNORECASE)
# Layer tokens such as L1..L6 or WM typically appear as standalone segments; \b is fine here,
# but to be consistent we can also adopt the same boundary rule:
_re_layer = re.compile(r"(?<![A-Za-z0-9])(L[1-6]|WM)(?![A-Za-z0-9])", flags=re.IGNORECASE)

def clinicogenotype_from_folder(folder_str: str) -> str:
    if not isinstance(folder_str, str):
        return None
    if _re_ctrl.search(folder_str):
        return "Non-dementia control"
    if _re_ad33.search(folder_str):
        return "AD APOE e3/e3"
    if _re_ad44.search(folder_str):
        return "AD APOE e4/e4"
    return None

def cortical_layer_from_folder(folder_str: str) -> str:
    if not isinstance(folder_str, str):
        return None
    m = _re_layer.search(folder_str)
    if m:
        return m.group(1).upper()
    return None

def process_single_file(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=0)

    required_cols = ["A1", "A2", "A3", "A6", "Folder"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {xlsx_path.name}: {missing}")

    denom = df["A2"] + df["A3"] + df["A6"]
    valid = denom > 0
    df = df.loc[valid].copy()
    df["R"] = df["A1"] / (df["A2"] + df["A3"] + df["A6"])

    df["clinicogenotype"] = df["Folder"].apply(clinicogenotype_from_folder)
    df["cortical_layer"] = df["Folder"].apply(cortical_layer_from_folder)

    if "FrameIndex" not in df.columns:
        df["FrameIndex"] = range(1, len(df) + 1)

    keep_cols = ["FrameIndex", "Folder", "A1", "A2", "A3", "A6", "R", "clinicogenotype", "cortical_layer"]
    return df[keep_cols]

def main():
    # Root directory containing session subfolders (AD3a, AD3b, ..., AD4f, etc.)
    ROOT = Path(r"C:\Users\clchr\OneDrive - Stanford\Research Documents\AD Project\2025")
    pattern = "Hyperspectral_Myelin_AverageFits.xlsx"

    files = sorted(ROOT.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found under {ROOT} matching {pattern}")

    tables = []
    for fp in files:
        try:
            t = process_single_file(fp)
            # Attach useful context columns (non-invasive; can be removed if undesired)
            t["session_id"] = fp.parent.name
            t["file_path"] = str(fp)
            tables.append(t)
        except Exception as e:
            # Non-fatal: skip problematic files but continue
            print(f"Skipping {fp}: {e}")

    if not tables:
        raise RuntimeError("No valid frames were parsed from any file.")

    combined = pd.concat(tables, ignore_index=True)

    with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as xr:
        # Sheet 1: all frames
        combined.to_excel(xr, sheet_name=SHEET_ALL, index=False)

        # Sheets 2–4: split by clinicogenotype
        combined.loc[combined["clinicogenotype"] == "Non-dementia control"].to_excel(
            xr, sheet_name=SHEET_CTRL, index=False
        )
        combined.loc[combined["clinicogenotype"] == "AD APOE e3/e3"].to_excel(
            xr, sheet_name=SHEET_AD33, index=False
        )
        combined.loc[combined["clinicogenotype"] == "AD APOE e4/e4"].to_excel(
            xr, sheet_name=SHEET_AD44, index=False
        )

    print(f"Wrote: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
