#!/usr/bin/env python3
"""
make_all_histograms_ad.py

Build histograms (and cumulative histograms) for counts and percentages of:
  - pure lipid
  - lipidated lipofuscin
  - lipofuscin
across cell types (Microglia, Astrocyte, Neuron) and conditions (Control, AD33, AD44).

Input workbook (edit path below):
  D:/OneDrive - Stanford/Research Documents/AD Project/2025/AD Lipid Statistics_prism.xlsx

Outputs:
  PNG files in ./plots (or change OUTPUT_DIR below)

Requirements:
    pip install pandas matplotlib seaborn openpyxl
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# USER SETTINGS
# -----------------------------
EXCEL_FILE = r"D:/OneDrive - Stanford/Research Documents/AD Project/2025/AD Lipid Statistics_prism.xlsx"
OUTPUT_DIR = "plots"  # created if missing

# Which cell types / conditions do we expect as sheet names like "Control Microglia"
CELL_TYPES = ["Microglia", "Astrocytes", "Neurons"]
CONDITIONS = ["Control", "AD33", "AD44"]

# Object types and their column stems in your sheets
# (columns assumed present: *_count and *_percentage)
OBJECTS = {
    "pure_lipid": {
        "nice": "Pure Lipid",
        "count_col": "pure_lipid_count",
        "pct_col": "pure_lipid_percentage",
    },
    "lipid_lipofuscin": {
        "nice": "Lipidated Lipofuscin",
        "count_col": "lipid_lipofuscin_count",
        "pct_col": "lipid_lipofuscin_percentage",
    },
    "lipofuscin": {
        "nice": "Pure Lipofuscin",
        "count_col": "lipofuscin_count",
        "pct_col": "lipofuscin_percentage",
    },
}

# Plot look
FIGSIZE = (6, 4)
DPI = 600
BINS_PERCENTAGES = 100         # number of bins for percentages (0..100)
DROP_ZERO_COUNTS = True        # set True to remove zero-count cells from histograms
DROP_ZERO_PERCENTS = True      # set True to remove zero-percent cells from histograms
STYLE = "white"            # seaborn style
PALETTE = {
    "Control": "blue",      # control blue-based
    "AD33": "magenta",      # AD33 magenta-based
    "AD44": "red"           # AD44 red-based
}

# -----------------------------
# HELPERS
# -----------------------------
def safe_read_sheet(xlsx_path, sheet_name):
    """Read a sheet if it exists; else return None."""
    try:
        return pd.read_excel(xlsx_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"[WARN] Could not load sheet '{sheet_name}': {e}")
        return None

def int_bins_for_counts(series: pd.Series):
    """Create integer-aligned bins for counts if feasible; else fallback."""
    s = series.dropna()
    s_min, s_max = int(s.min()), int(s.max())
    return list(range(s_min, s_max + 2))

def percentage_bins():
    """Fixed bins 0..100 for percentages."""
    return BINS_PERCENTAGES

def make_hist(
    df: pd.DataFrame, value_col: str, hue_col: str, title: str,
    xlabel: str, ylabel: str, outfile: Path, cumulative: bool, bins
):
    """Generic histogram plotter with seaborn."""
    if df is None or df.empty:
        print(f"[SKIP] No data for {outfile.name}")
        return

    plt.figure(figsize=FIGSIZE)
    sns.histplot(
        data=df,
        x=value_col,
        hue=hue_col,
        element="step",
        stat="probability",
        common_norm=False,
        bins=bins,
        cumulative=cumulative,
        palette=PALETTE,
        fill=False,
        linewidth=2
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outfile, dpi=DPI)
    plt.close()
    print(f"[OK] Saved: {outfile}")

# -----------------------------
# MAIN
# -----------------------------
def main():
    sns.set_style(STYLE)
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # Iterate through each cell type, assemble per-condition data from sheets
    # Sheet naming pattern: "<Condition> <CellType>", e.g., "Control Microglia"
    for cell in CELL_TYPES:
        # Load all conditions for this cell type and stack
        stacked = []
        for cond in CONDITIONS:
            sheet = f"{cond} {cell}"
            df = safe_read_sheet(EXCEL_FILE, sheet)
            if df is None:
                continue
            df = df.copy()
            df["condition"] = cond
            stacked.append(df)

        if not stacked:
            print(f"[WARN] No sheets found for cell type: {cell}")
            continue

        data = pd.concat(stacked, ignore_index=True)

        # For each object, make histograms for counts and percentages
        for key, meta in OBJECTS.items():
            # ---- COUNTS ----
            counts_col = meta["count_col"]
            if counts_col in data.columns:
                df_counts = data[["condition", counts_col]].rename(
                    columns={counts_col: "value"}
                )
                if DROP_ZERO_COUNTS:
                    df_counts = df_counts[df_counts["value"] != 0]
                df_counts = df_counts.dropna(subset=["value"])

                bins_counts = int_bins_for_counts(df_counts["value"])

                title = f"{meta['nice']} Count per Cell — {cell}"
                xlabel = "Count per Cell"
                ylabel = "Probability"

                # Non-cumulative
                outfile = outdir / f"{cell}_{key}_count_hist.png"
                make_hist(
                    df_counts, "value", "condition",
                    title, xlabel, ylabel, outfile,
                    cumulative=False, bins=bins_counts
                )

                # Cumulative
                outfile = outdir / f"{cell}_{key}_count_cdf.png"
                make_hist(
                    df_counts, "value", "condition",
                    f"{title} (Cumulative)", xlabel, "Cumulative Probability",
                    outfile, cumulative=True, bins=bins_counts
                )
            else:
                print(f"[WARN] Missing column '{counts_col}' for {cell}/{key}")

            # ---- PERCENTAGES ----
            pct_col = meta["pct_col"]
            if pct_col in data.columns:
                df_pct = data[["condition", pct_col]].rename(
                    columns={pct_col: "value"}
                )
                if DROP_ZERO_PERCENTS:
                    df_pct = df_pct[df_pct["value"] != 0]
                df_pct = df_pct.dropna(subset=["value"])

                bins_pct = percentage_bins()

                title = f"{meta['nice']} Area % of Cell — {cell}"
                xlabel = "Area Percentage (%)"
                ylabel = "Probability"

                # Non-cumulative
                outfile = outdir / f"{cell}_{key}_pct_hist.png"
                make_hist(
                    df_pct, "value", "condition",
                    title, xlabel, ylabel, outfile,
                    cumulative=False, bins=bins_pct
                )

                # Cumulative
                outfile = outdir / f"{cell}_{key}_pct_cdf.png"
                make_hist(
                    df_pct, "value", "condition",
                    f"{title} (Cumulative)", xlabel, "Cumulative Probability",
                    outfile, cumulative=True, bins=bins_pct
                )
            else:
                print(f"[WARN] Missing column '{pct_col}' for {cell}/{key}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
