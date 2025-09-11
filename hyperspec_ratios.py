#!/usr/bin/env python3
"""
Extract 2850/2930 (acyl-chain) ratios from Hyperspectral_Results_*.xlsx files.

- Scans a directory tree for 'Hyperspectral_Results_*.xlsx'
- Reads the 'Raw Data' sheet
- Uses:
    - 'Wavenumber 24' (2850 cm^-1)
    - 'Wavenumber 13' (2930 cm^-1)
- Computes ratio = I2850 / I2930 per droplet
- Preserves 'Category' and 'Location'
- Infers 'condition' and 'cell_type' from filename
- Saves:
    - hyperspectral_ratios_droplets.csv  (all droplets, one row per droplet)
    - hyperspectral_ratios_summary.xlsx  with two sheets:
        * 'Summary'                       (grouped stats)
        * 'Intracellular_Lipid_Points'    (9 columns of individual ratios:
                                           Microglia/Control, AD33, AD44;
                                           Astrocytes/Control, AD33, AD44;
                                           Neurons/Control, AD33, AD44)
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Hardcoded paths for Spyder
# ---------------------------------------------------------------------
INDIR = Path(r"D:/OneDrive - Stanford/Research Documents/AD Project/2025")
OUTDIR = Path(r"D:/OneDrive - Stanford/Research Documents/AD Project/2025/hyperspec_ratios")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
FILE_GLOB = "Hyperspectral_Results_*.xlsx"
SHEET_NAME = "Raw Data"
COL_WN_2850 = "Wavenumber 24"  # 2850 cm^-1
COL_WN_2930 = "Wavenumber 13"  # 2930 cm^-1
MIN_I2930 = 0.0                # drop rows with I2930 <= this value

CONDITIONS = ["Control", "AD33", "AD44"]
CELL_TYPES = ["Microglia", "Astrocytes", "Neurons"]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def infer_condition_and_celltype(name: str):
    """Infer condition and cell type from filename stem."""
    s = name.lower()

    cond = None
    if "control" in s or "ctrl" in s:
        cond = "Control"
    elif "ad33" in s or "e3" in s:
        cond = "AD33"
    elif "ad44" in s or "e4" in s:
        cond = "AD44"

    ctype = None
    if "microglia" in s:
        ctype = "Microglia"
    elif "astro" in s:
        ctype = "Astrocytes"
    elif "neuron" in s:
        ctype = "Neurons"

    return cond, ctype


def load_file(xlsx_path: Path) -> pd.DataFrame:
    """Load one hyperspectral results file and compute droplet-level ratios."""
    raw = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME, engine="openpyxl")
    raw.columns = [str(c).strip() for c in raw.columns]

    required = {COL_WN_2850, COL_WN_2930, "Category", "Location"}
    if not required.issubset(raw.columns):
        missing = ", ".join(sorted(required - set(raw.columns)))
        raise ValueError(f"Missing required columns in {xlsx_path.name}: {missing}")

    i2850 = pd.to_numeric(raw[COL_WN_2850], errors="coerce")
    i2930 = pd.to_numeric(raw[COL_WN_2930], errors="coerce")
    mask = i2930.notna() & (i2930 > MIN_I2930)

    cond, ctype = infer_condition_and_celltype(xlsx_path.stem)

    df = pd.DataFrame({
        "file": xlsx_path.name,
        "category": raw.loc[mask, "Category"],
        "location": raw.loc[mask, "Location"],
        "I2850": i2850[mask],
        "I2930": i2930[mask],
        "condition": cond,
        "cell_type": ctype,
    })
    df["ratio_2850_2930"] = df["I2850"] / df["I2930"]
    return df


def summarize_groups(droplets: pd.DataFrame) -> pd.DataFrame:
    """Group by condition/cell_type/category/location and compute n, mean, median, std, sem."""
    gcols = ["condition", "cell_type", "category", "location"]

    def _agg(vals: pd.Series):
        vals = vals.dropna().to_numpy(dtype=float)
        n = vals.size
        mean = np.mean(vals) if n else np.nan
        median = np.median(vals) if n else np.nan
        std = np.std(vals, ddof=1) if n > 1 else np.nan
        sem = std / math.sqrt(n) if n > 1 else np.nan
        return pd.Series({"n": n, "mean": mean, "median": median, "std": std, "sem": sem})

    return droplets.groupby(gcols)["ratio_2850_2930"].apply(_agg).reset_index()


def make_intracellular_lipid_wide(droplets: pd.DataFrame) -> pd.DataFrame:
    """
    Build a 9-column wide table of individual ratios for *intracellular lipid* droplets:
      Microglia_Control, Microglia_AD33, Microglia_AD44,
      Astrocytes_Control, Astrocytes_AD33, Astrocytes_AD44,
      Neurons_Control, Neurons_AD33, Neurons_AD44

    Columns are padded with NaNs to equalize length.
    """
    # Normalize category/location for robust filtering (accept "Lipid" or "Lipids")
    cat_norm = droplets["category"].astype(str).str.strip().str.lower()
    loc_norm = droplets["location"].astype(str).str.strip().str.lower()

    is_lipid = cat_norm.isin({"lipid", "lipids"})
    is_intra = loc_norm.eq("intracellular")

    df_lipid_intra = droplets.loc[is_lipid & is_intra, ["condition", "cell_type", "ratio_2850_2930"]]

    columns = []
    series_list = []

    for ctype in CELL_TYPES:
        for cond in CONDITIONS:
            mask = (df_lipid_intra["cell_type"] == ctype) & (df_lipid_intra["condition"] == cond)
            s = df_lipid_intra.loc[mask, "ratio_2850_2930"].reset_index(drop=True)
            series_list.append(s)
            columns.append(f"{ctype}_{cond}")

    # Pad columns to the same length
    max_len = max((len(s) for s in series_list), default=0)
    padded = [s.reindex(range(max_len)) for s in series_list]

    wide = pd.concat(padded, axis=1)
    wide.columns = columns
    return wide


# ---------------------------------------------------------------------
# Main execution (Spyder-friendly)
# ---------------------------------------------------------------------
files = sorted(INDIR.rglob(FILE_GLOB))
if not files:
    raise SystemExit(f"No files found in {INDIR} matching {FILE_GLOB}")

all_rows = []
for fp in files:
    try:
        df_one = load_file(fp)
        all_rows.append(df_one)
    except Exception as e:
        print(f"[WARN] Skipped {fp.name}: {e}")

droplets = pd.concat(all_rows, ignore_index=True)

# Save droplets (full detail) as CSV
droplets_path = OUTDIR / "hyperspectral_ratios_droplets.csv"
droplets.to_csv(droplets_path, index=False)

# Build summary and the wide intracellular-lipid sheet
summary = summarize_groups(droplets)
wide_intra_lipid = make_intracellular_lipid_wide(droplets)

# Write both sheets to one Excel file
summary_path = OUTDIR / "hyperspectral_ratios_summary.xlsx"
with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    wide_intra_lipid.to_excel(writer, sheet_name="Intracellular_Lipid_Points", index=False)

print(f"[OK] Droplets CSV: {droplets_path}")
print(f"[OK] Summary XLSX: {summary_path}")
