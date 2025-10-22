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
import re
import pandas as pd

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
PARENT_DIR = pathlib.Path(r"D:\OneDrive - Stanford\Research Documents\AD Project\2025")  # edit as needed
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

# --- Robust condition + cell-marker inference (case/format tolerant) ---
_RE_CTRL = re.compile(r"(?<![A-Za-z0-9])Control(?![A-Za-z0-9])", re.IGNORECASE)
_RE_AD33 = re.compile(r"(?<![A-Za-z0-9])AD33(?![A-Za-z0-9])", re.IGNORECASE)
_RE_AD44 = re.compile(r"(?<![A-Za-z0-9])AD44(?![A-Za-z0-9])", re.IGNORECASE)

def infer_condition(fname: str) -> str | None:
    if _RE_AD33.search(fname): return "AD33"
    if _RE_AD44.search(fname): return "AD44"
    if _RE_CTRL.search(fname): return "Control"
    return None

CELL_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"IBA1", re.IGNORECASE), "Microglia"),
    (re.compile(r"GFAP", re.IGNORECASE), "Astrocytes"),
    (re.compile(r"(MAP2|MAP-?2|MAP2_Sigma)", re.IGNORECASE), "Neurons"),
    (re.compile(r"(TUJ|TUJ[_-]?Ck)", re.IGNORECASE), "Neurons"),
]

def infer_cell_type(cell_marker: str) -> str | None:
    if not isinstance(cell_marker, str):
        return None
    for pat, ctype in CELL_RULES:
        if pat.search(cell_marker):
            return ctype
    return None

# Prepare an accumulator for exactly 9 sheets (3 conditions × 3 cell types)
SHEETS = [f"{cond} {ctype}" for cond in CONDITIONS for ctype in ["Microglia","Astrocytes","Neurons"]]
sheet_accumulator: dict[str, list[pd.Series]] = {name: [] for name in SHEETS}

# Track the union of all columns we see across all Summary sheets
all_cols: set[str] = set()

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

    # ── FILTER BLOCK: drop rows with cell_volume_voxels < 10000 ──────────────
    MIN_VOXELS = 10_000
    if "cell_volume_voxels" in df_summary.columns:
        # Coerce to numeric in case the column comes in as object/strings
        vol = pd.to_numeric(df_summary["cell_volume_voxels"], errors="coerce")
        before = len(df_summary)
        df_summary = df_summary[vol >= MIN_VOXELS].copy()
        removed = before - len(df_summary)
        if removed > 0:
            print(f"  Filtered {removed} row(s) with cell_volume_voxels < {MIN_VOXELS}.")
    else:
        print("  [WARN] 'cell_volume_voxels' not found in Summary; no volume filter applied.")
    # ─────────────────────────────────────────────────────────────────────────

    # Route each row (robust to variants)
    for _, row in df_summary.iterrows():
        fname = str(row.get("file_name", "") or "").strip()
        cmarker = row.get("cell_marker", "")

        cond = infer_condition(fname)
        if cond is None:
            print(f"  [WARN] No condition match in file_name: '{fname}'")
            continue

        ctype = infer_cell_type(cmarker)
        if ctype is None:
            print(f"  [WARN] Unrecognized cell_marker '{cmarker}' (file: '{fname}')")
            continue

        sheet_name = f"{cond} {ctype}"
        sheet_accumulator[sheet_name].append(row)
        all_cols.update(row.index.astype(str))

# Build output workbook with a consistent schema across sheets
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    # Stable column order: put file_name and cell_marker first if they exist
    ordered = list(all_cols) if all_cols else ["file_name", "cell_marker"]
    for pref in ("file_name", "cell_marker"):
        if pref in ordered:
            ordered.remove(pref)
            ordered.insert(0, pref)

    for sheet_name, rows in sheet_accumulator.items():
        if rows:
            out_df = pd.DataFrame(rows)
            out_df = out_df.reindex(columns=ordered)
        else:
            out_df = pd.DataFrame(columns=ordered)

        out_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"✅ Merged results saved to: {OUTPUT_FILE}")
