#!/usr/bin/env python3
"""
merge_analysis_results.py — Merge Summary sheets from all analysis_results.xlsx into one AD Lipid Statistics.xlsx

This script expects a hard-coded parent directory containing subfolders:
AD3a, AD3b, AD3c, AD3d, AD3e, AD3f, AD4a, AD4b, AD4c, AD4d, AD4e, AD4f.
Each subfolder must contain an analysis_results.xlsx with a "Summary" sheet.
Rows are routed to one of nine sheets in the output file based on:
  - file_name containing: Control → "Control", AD33 → "AD33", AD44 → "AD44"
  - cell_marker: IBA1 → "Microglia", GFAP → "Astrocytes",
                 MAP2_Sigma, TUJ_Ck, TUJ → "Neurons"

Usage:
  - Edit PARENT_DIR to the base path containing the subfolders.
  - Run inside Spyder or any Python IDE: python merge_analysis_results.py
"""

import pathlib
import pandas as pd

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
PARENT_DIR = pathlib.Path(r"C:\Users\clchr\OneDrive - Stanford\Research Documents\AD Project\2025")  # edit as needed
OUTPUT_FILE = PARENT_DIR / "AD Lipid Statistics.xlsx"
# ──────────────────────────────────────────────────────────────────────────────

# Define keywords and mappings
CONDITIONS = ["Control", "AD33", "AD44"]
CELL_MAP = {
    "IBA1": "Microglia",
    "GFAP": "Astrocytes",
    "MAP2_Sigma": "Neurons",
    "TUJ_Ck": "Neurons",
    "TUJ": "Neurons",
}

# Prepare an accumulator for each output sheet
sheet_accumulator: dict[str, list[pd.Series]] = {
    f"{cond} {ctype}": [] for cond in CONDITIONS for ctype in CELL_MAP.values()
}

# Iterate subfolders
for subdir in PARENT_DIR.iterdir():
    if not subdir.is_dir():
        continue
    results_path = subdir / "analysis_results.xlsx"
    if not results_path.exists():
        print(f"Skipping {subdir.name}: no analysis_results.xlsx found.")
        continue

    print(f"Processing {subdir.name}/analysis_results.xlsx...")
    try:
        df_summary = pd.read_excel(results_path, sheet_name="Summary")
    except Exception as e:
        print(f"  Error reading Summary sheet: {e}")
        continue

    # Route each row
    for _, row in df_summary.iterrows():
        fname = str(row.get("file_name", ""))
        cmarker = row.get("cell_marker", "")

        # Determine condition
        cond = next((c for c in CONDITIONS if c in fname), None)
        if cond is None:
            print(f"  Warning: no condition match in '{fname}'")
            continue

        # Determine cell type
        ctype = CELL_MAP.get(cmarker)
        if ctype is None:
            print(f"  Warning: unrecognized cell_marker '{cmarker}'")
            continue

        # Append row to appropriate accumulator list
        sheet_name = f"{cond} {ctype}"
        sheet_accumulator[sheet_name].append(row)

# Build output workbook
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    for sheet_name, rows in sheet_accumulator.items():
        if rows:
            out_df = pd.DataFrame(rows)
        else:
            # If no data, create an empty DataFrame with no rows
            out_df = pd.DataFrame(columns=["file_name", "cell_marker"])

        out_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"✅ Merged results saved to: {OUTPUT_FILE}")
