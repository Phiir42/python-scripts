#!/usr/bin/env python3
# Spyder-friendly version — one Excel with:
#   - 3 per-donor sheets
#   - 3 donor-merged sheets (AllDonors)
#   - 3 genotype+sheet merged sheets (AllGenosAllSheets)
# Includes donor forward-fill, blank-ignoring, and debug prints

import re
from pathlib import Path
import pandas as pd
import numpy as np

# ====================================================
# ---- USER SETTINGS ---------------------------------
# ====================================================
XLSX_PATH = r"D:\OneDrive - Stanford\Research Documents\AD Project\2025\AD_Lipid_Statistics_CorticalLayers.xlsx"
OUTPUT_XLSX = r"D:\OneDrive - Stanford\Research Documents\AD Project\2025\nonzero_summary_all.xlsx"

# Object types to process
OBJECT_TYPES = ["Lipids", "Lipofuscin", "Lipidated Lipofuscin"]

# ====================================================
# ---- CONSTANTS AND HELPERS -------------------------
# ====================================================
LAYER_NAMES = [
    "Layer I", "Layer II", "Layer III", "Layer IV", "Layer V", "Layer VI",
    "White Matter", "WM", "WhiteMatter"
]
CLINICOGENOTYPES = ["Control", "AD33", "AD44"]
CELL_TYPES = ["Microglia", "Astrocytes", "Neurons"]


def infer_labels_from_sheet(sheet_name: str):
    clinicogenotype = next((g for g in CLINICOGENOTYPES if re.search(rf'\b{g}\b', sheet_name, re.I)), None)
    cell_type       = next((c for c in CELL_TYPES        if re.search(rf'\b{c}\b', sheet_name, re.I)), None)
    return clinicogenotype, cell_type


def find_file_name_column(df: pd.DataFrame):
    """Find 'file_name' or 'filename' column (case-insensitive, single or MultiIndex)."""
    if isinstance(df.columns, pd.MultiIndex):
        for tup in df.columns:
            for part in tup:
                if isinstance(part, str) and part.strip().lower() in ("file_name", "filename"):
                    return tup
    else:
        for col in df.columns:
            if isinstance(col, str) and col.strip().lower() in ("file_name", "filename"):
                return col
    return None


def clean_and_ffill_filenames(series: pd.Series) -> pd.Series:
    """Treat empty/whitespace as missing, then forward-fill so blocks inherit the donor."""
    s = series.copy()
    s = s.astype("string")
    s = s.replace(r'^\s*$', pd.NA, regex=True).ffill()
    return s


def extract_donor_from_filename(series: pd.Series) -> pd.Series:
    """Extract donor IDs of form S#### from filenames."""
    def get_donor(x):
        if pd.isna(x):
            return np.nan
        m = re.search(r"(S\d+)", str(x))
        return m.group(1) if m else np.nan
    return series.apply(get_donor)


def melt_layer_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Melt wide layer/object data into long form. Keeps source row index in 'Row'."""
    records = []
    if isinstance(df.columns, pd.MultiIndex):
        for top, sub in df.columns:
            if pd.isna(top):
                continue
            top_s = str(top).strip()
            if top_s in LAYER_NAMES and isinstance(sub, str) and sub.strip():
                vals = pd.to_numeric(df[(top, sub)], errors="coerce")  # blanks -> NaN
                rec = pd.DataFrame({
                    "Row": df.index,
                    "Layer": top_s,
                    "ObjectType": str(sub).strip(),
                    "Value": vals
                })
                records.append(rec)
    else:
        for col in df.columns:
            if not isinstance(col, str):
                continue
            parts = [p.strip() for p in re.split(r"[|>/\\]+", col) if p.strip()]
            if len(parts) >= 2 and parts[0] in LAYER_NAMES:
                vals = pd.to_numeric(df[col], errors="coerce")
                rec = pd.DataFrame({
                    "Row": df.index,
                    "Layer": parts[0],
                    "ObjectType": parts[1],
                    "Value": vals
                })
                records.append(rec)

    if not records:
        return pd.DataFrame(columns=["Row", "Layer", "ObjectType", "Value"])

    out = pd.concat(records, axis=0, ignore_index=True)
    out["ObjectType"] = out["ObjectType"].astype("string").str.strip()
    out["Layer"] = out["Layer"].astype("string").str.strip()
    return out


def compute_nonzero_percent(df_long, object_type, donors_by_row, clinicogenotype, cell_type, sheet_name):
    """
    Compute % nonzero per Layer × Donor, ignoring blanks:
    - n_total = count of rows with Value NOT NaN
    - n_nonzero = subset where Value != 0
    """
    mask = df_long["ObjectType"].str.casefold() == object_type.casefold()
    out = df_long[mask].copy()

    if out.empty:
        print(f"  [DEBUG] No entries for object '{object_type}' in sheet '{sheet_name}'.")
        return pd.DataFrame(columns=[
            "Clinicogenotype", "CellType", "Layer", "Donor", "Sheet",
            "n_total", "n_nonzero", "pct_nonzero"
        ])

    # Map donor by original row index
    out["Donor"] = out["Row"].map(donors_by_row) if isinstance(donors_by_row, pd.Series) else np.nan
    out["Clinicogenotype"] = clinicogenotype
    out["CellType"]        = cell_type
    out["Sheet"]           = sheet_name

    # Valid entries are those with real numeric values
    out["valid"] = out["Value"].notna()
    out["nonzero"] = out["valid"] & (out["Value"] != 0)

    grp_keys = ["Clinicogenotype", "CellType", "Layer", "Donor", "Sheet"]
    summary = (
        out.groupby(grp_keys, dropna=False)
           .agg(n_total=("valid", "sum"), n_nonzero=("nonzero", "sum"))
           .reset_index()
    )

    # Keep groups with ≥1 valid entry
    summary = summary[summary["n_total"] > 0].copy()
    summary["pct_nonzero"] = (summary["n_nonzero"] / summary["n_total"]) * 100.0
    summary["Layer"] = summary["Layer"].replace({"WhiteMatter": "White Matter", "WM": "White Matter"})

    # Debug
    print(f"  [DEBUG] {sheet_name} | {object_type} — groups: {len(summary)}")
    return summary[["Clinicogenotype", "CellType", "Layer", "Donor", "Sheet",
                    "n_total", "n_nonzero", "pct_nonzero"]]


def process_workbook(xlsx_path: str, object_type: str) -> pd.DataFrame:
    xl = pd.ExcelFile(xlsx_path)
    per_sheet = []

    for sheet in xl.sheet_names:
        # Prefer 2-row header (Layer merged top, Object subcolumns second)
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sheet, header=[0, 1])
        except Exception:
            df = pd.read_excel(xlsx_path, sheet_name=sheet, header=0)

        clinicogenotype, cell_type = infer_labels_from_sheet(sheet)
        df_long = melt_layer_object_columns(df)
        if df_long.empty:
            print(f"[DEBUG] Sheet '{sheet}': No layer/object columns detected.")
            continue

        # Locate and clean the file_name column, then forward-fill
        fn_col = find_file_name_column(df)
        if fn_col is not None:
            raw_file_name = df[fn_col]
            file_name_ff = clean_and_ffill_filenames(raw_file_name)
            donors_by_row = extract_donor_from_filename(file_name_ff)
            donors_by_row.index = df.index  # preserve row alignment
        else:
            donors_by_row = pd.Series([np.nan] * len(df.index), index=df.index)

        # Optional debug: nonblank counts per subcolumn BEFORE filtering
        sub_counts_all = (
            df_long.dropna(subset=["Value"])
                  .groupby("ObjectType")
                  .size()
                  .sort_values(ascending=False)
                  .to_dict()
        )
        print(f"\n[DEBUG] Sheet '{sheet}' — nonblank counts per subcolumn: {sub_counts_all}")

        summary = compute_nonzero_percent(
            df_long=df_long,
            object_type=object_type,
            donors_by_row=donors_by_row,
            clinicogenotype=clinicogenotype,
            cell_type=cell_type,
            sheet_name=sheet
        )
        per_sheet.append(summary)

    if not per_sheet:
        return pd.DataFrame(columns=[
            "Clinicogenotype", "CellType", "Layer", "Donor", "Sheet",
            "n_total", "n_nonzero", "pct_nonzero"
        ])

    combined = pd.concat(per_sheet, axis=0, ignore_index=True)
    print(f"[DEBUG] Total rows for '{object_type}': {len(combined)}\n")
    return combined


def donor_merged(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge donors within Clinicogenotype × CellType × Layer × Sheet:
    - Sum n_total and n_nonzero, then recompute pct_nonzero.
    """
    if summary_df.empty:
        return summary_df.copy()
    grp_keys = ["Clinicogenotype", "CellType", "Layer", "Sheet"]
    merged = (
        summary_df.groupby(grp_keys, dropna=False)[["n_total", "n_nonzero"]]
                  .sum()
                  .reset_index()
    )
    merged["pct_nonzero"] = (merged["n_nonzero"] / merged["n_total"]) * 100.0
    return merged[["Clinicogenotype", "CellType", "Layer", "Sheet", "n_total", "n_nonzero", "pct_nonzero"]]


def genotype_and_sheet_merged(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge *across* Clinicogenotype and Sheet (i.e., collapse both),
    after donors have already been merged:
      Output granularity: CellType × Layer (summing n_total and n_nonzero).
    """
    if merged_df.empty:
        return merged_df.copy()
    grp_keys = ["CellType", "Layer"]
    out = (
        merged_df.groupby(grp_keys, dropna=False)[["n_total", "n_nonzero"]]
                 .sum()
                 .reset_index()
    )
    out["pct_nonzero"] = (out["n_nonzero"] / out["n_total"]) * 100.0
    return out[["CellType", "Layer", "n_total", "n_nonzero", "pct_nonzero"]]


# ====================================================
# ---- RUN FOR ALL OBJECT TYPES ----------------------
# ====================================================
OUTPUT_XLSX = Path(OUTPUT_XLSX)
OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    for obj in OBJECT_TYPES:
        print("\n==============================")
        print(f"Processing object type: {obj}")
        print("==============================")
        per_donor_df = process_workbook(XLSX_PATH, obj)
        per_donor_df.to_excel(writer, sheet_name=obj[:31], index=False)
        print(f"  ✅ {len(per_donor_df)} rows written to sheet '{obj}'")

        merged_df = donor_merged(per_donor_df)
        merged_sheet = (obj[:27] + "_AllDonors") if len(obj) > 20 else (obj + "_AllDonors")
        merged_sheet = merged_sheet[:31]
        merged_df.to_excel(writer, sheet_name=merged_sheet, index=False)
        print(f"  ✅ {len(merged_df)} rows written to sheet '{merged_sheet}'")

        # NEW: merge across clinicogenotypes *and* sheets
        gs_df = genotype_and_sheet_merged(merged_df)
        gs_sheet = (obj[:22] + "_AllGenosAllSheets") if len(obj) > 16 else (obj + "_AllGenosAllSheets")
        gs_sheet = gs_sheet[:31]
        gs_df.to_excel(writer, sheet_name=gs_sheet, index=False)
        print(f"  ✅ {len(gs_df)} rows written to sheet '{gs_sheet}'")

print(f"\n✅ All results saved in one file:\n{OUTPUT_XLSX}")
