#!/usr/bin/env python3
"""
Compile and summarize droplet classifications from Hyperspectral_Results_*.xlsx files.

Enhanced to work with sheets structured like:
- 'Classification' sheet containing 'class_label' plus features and droplet ID as 'DropletID'
- 'Peak Fits' (or 'Raw Data') sheet containing metadata columns such as 'Location', 'Cell Marker', 'LAMP2_Coloc'

This script:
1) Recursively scans a data directory for Hyperspectral_Results_*.xlsx
2) Reads the 'Classification' sheet (if present) for the classification label
3) Merges Location / Cell Marker / LAMP2 from 'Peak Fits' (and then 'Raw Data') if missing
4) Infers clinicogenetic condition from filename (AD33/AD44/CTRL/Control/HC)
5) Writes tidy combined rows and aggregates, with optional basic plots and chi-square tests

Usage
-----
python compile_hyperspec_summary.py --data-root <dir> --out <dir> [--make-plots] [--dry-run]
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy.stats import chi2_contingency
except Exception:
    chi2_contingency = None


# --- replace CONDITION_MAP + infer_* with this ---

DEFAULT_CONDITION = "Unknown"

def infer_condition_from_path(path_like) -> str:
    s = str(path_like)
    s = s.replace("\\", "/")
    # treat underscores/dashes as separators
    s = re.sub(r"[_-]+", " ", s)

    if re.search(r"(?i)\bAD\s*3\s*3\b|\bAD\s*33\b|\bAPOE\s*3\s*/\s*3\b|\bAPOE\s*33\b|\bE3\s*E3\b", s):
        return "AD APOE 3/3"
    if re.search(r"(?i)\bAD\s*4\s*4\b|\bAD\s*44\b|\bAPOE\s*4\s*/\s*4\b|\bAPOE\s*44\b|\bE4\s*E4\b", s):
        return "AD APOE 4/4"
    if re.search(r"(?i)\bCTRL\b|\bCONTROL\b|\bHC\b|\bHEALTHY\s*CONTROL\b", s):
        return "Healthy Control"
    # folder shorthands like AD3a/AD4b
    parts = [p.lower() for p in Path(s).parts]
    if any(re.fullmatch(r"ad3[a-z]?", p) for p in parts): return "AD APOE 3/3"
    if any(re.fullmatch(r"ad4[a-z]?", p) for p in parts): return "AD APOE 4/4"
    if any(p in {"ctrl", "control", "hc"} for p in parts): return "Healthy Control"
    return DEFAULT_CONDITION


COL_ALIASES = {
    "droplet_id": ["DropletID", "Lipid ID", "droplet_id", "object_id", "id", "dropletid", "lipidid"],
    "location": ["Location", "location", "compartment", "intra_extra", "intracellular_extracellular"],
    "cell_type": ["Cell Marker", "cell marker", "cell_type", "cell", "marker", "cell_marker", "celltype"],
    "lamp2": ["LAMP2_Coloc", "LAMP2 Coloc", "lamp2", "lamp2_coloc", "lamp2 colocalized",
              "lamp2_colocalized", "lamp2_colocalisation", "lamp2_flag", "lamp2 coloc", "lamp2_coloc."],
    "classification": ["class_label", "classification", "class", "label", "final_class", "droplet_class"],
}

TRUE_TOKENS = {"1", "true", "t", "yes", "y", "coloc", "colocalized", "colocalised"}


def pick_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    # exact match (case insensitive)
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # fuzzy: strip spaces/underscores
    normalized = {re.sub(r"[\s_]+", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[\s_]+", "", cand.lower())
        if key in normalized:
            return normalized[key]
    return None


def normalize_lamp2(value) -> Optional[bool]:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return bool(int(value))
    s = str(value).strip().lower()
    if s in TRUE_TOKENS:
        return True
    if s in {"0", "false", "f", "no", "n", "none"}:
        return False
    if s == "true":
        return True
    if s == "false":
        return False
    return None


def _read_sheet_safe(xls, name):
    try:
        return pd.read_excel(xls, sheet_name=name)
    except Exception:
        return None


def _merge_meta(base: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if meta is None or meta.empty:
        return base

    id_col_base = pick_first_existing_column(base, ["DropletID", "Lipid ID", "droplet_id", "dropletid", "lipidid", "id"])
    id_col_meta = pick_first_existing_column(meta, ["DropletID", "Lipid ID", "droplet_id", "dropletid", "lipidid", "id"])
    if id_col_base is None or id_col_meta is None:
        return base

    lhs = base.copy()
    rhs = meta.copy()

    loc_col = pick_first_existing_column(rhs, COL_ALIASES["location"])
    cell_col = pick_first_existing_column(rhs, COL_ALIASES["cell_type"])
    lamp_col = pick_first_existing_column(rhs, COL_ALIASES["lamp2"])

    keep = {id_col_meta}
    if loc_col: keep.add(loc_col)
    if cell_col: keep.add(cell_col)
    if lamp_col: keep.add(lamp_col)
    rhs2 = rhs[list(keep)].copy()

    rename_map = {}
    if loc_col: rename_map[loc_col] = "_meta_location"
    if cell_col: rename_map[cell_col] = "_meta_cell_type"
    if lamp_col: rename_map[lamp_col] = "_meta_lamp2"
    rhs2 = rhs2.rename(columns=rename_map)

    merged = pd.merge(lhs, rhs2, left_on=id_col_base, right_on=id_col_meta, how="left")

    if "location" in merged.columns and "_meta_location" in merged.columns:
        merged["location"] = merged["location"].fillna(merged["_meta_location"])
    elif "_meta_location" in merged.columns:
        merged["location"] = merged["_meta_location"]

    if "cell_type" in merged.columns and "_meta_cell_type" in merged.columns:
        merged["cell_type"] = merged["cell_type"].fillna(merged["_meta_cell_type"])
    elif "_meta_cell_type" in merged.columns:
        merged["cell_type"] = merged["_meta_cell_type"]

    if "lamp2" in merged.columns and "_meta_lamp2" in merged.columns:
        merged["lamp2"] = merged["lamp2"].combine_first(
            merged["_meta_lamp2"].map(normalize_lamp2)
        )
    elif "_meta_lamp2" in merged.columns:
        merged["lamp2"] = merged["_meta_lamp2"].map(normalize_lamp2)

    for c in ["_meta_location", "_meta_cell_type", "_meta_lamp2", id_col_meta]:
        if c in merged.columns:
            merged = merged.drop(columns=c)

    return merged


def load_classification_sheet(xlsx_path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    sheet_names = xls.sheet_names
    sheet_to_read = "Classification" if "Classification" in sheet_names else sheet_names[-1]
    dfC = pd.read_excel(xls, sheet_name=sheet_to_read)

    # --- build the frame with the right length FIRST ---
    out = pd.DataFrame(index=dfC.index)
    # DropletID first…
    c_id = pick_first_existing_column(dfC, COL_ALIASES["droplet_id"])
    out["DropletID"] = dfC[c_id] if c_id else np.arange(len(dfC))
    
    # now broadcast scalars
    out["file"] = str(xlsx_path)
    out["condition"] = infer_condition_from_path(xlsx_path)

    # location / cell_type / lamp2 / classification (unchanged order is fine now)
    c_loc = pick_first_existing_column(dfC, COL_ALIASES["location"])
    out["location"] = dfC[c_loc].astype(str).str.strip() if c_loc else np.nan

    c_cell = pick_first_existing_column(dfC, COL_ALIASES["cell_type"])
    out["cell_type"] = dfC[c_cell].astype(str).str.strip() if c_cell else np.nan

    c_lamp = pick_first_existing_column(dfC, COL_ALIASES["lamp2"])
    out["lamp2"] = dfC[c_lamp].map(normalize_lamp2) if c_lamp else np.nan

    c_class = pick_first_existing_column(dfC, COL_ALIASES["classification"])
    if not c_class:
        raise ValueError(f"Could not find a 'classification' column in {xlsx_path} (sheet: {sheet_to_read}).")
    out["classification"] = dfC[c_class].astype(str).str.strip()

    # Merge missing metadata from Peak Fits then Raw Data
    df_peak = _read_sheet_safe(xls, "Peak Fits")
    df_raw = _read_sheet_safe(xls, "Raw Data")
    out = _merge_meta(out, df_peak)
    out = _merge_meta(out, df_raw)

    # Standardize NA fallbacks
    out["location"] = out["location"].fillna("unknown")
    out["cell_type"] = out["cell_type"].fillna("unspecified")

    return out


def aggregate_distributions(tidy: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    results = {}

    df = tidy.copy()
    df["cell_type"] = df["cell_type"].fillna("unspecified").replace({"nan": "unspecified"})
    df["location"] = df["location"].fillna("unknown").replace({"nan": "unknown"})
    df["classification"] = df["classification"].fillna("uncertain").replace({"nan": "uncertain"})
    df["condition"] = df["condition"].fillna("Unknown")

    counts = (
        df.groupby(["condition", "cell_type", "classification"], dropna=False)
          .size()
          .rename("count")
          .reset_index()
    )
    results["counts_by_cond_cell_class"] = counts

    total_per_cond_cell = counts.groupby(["condition", "cell_type"])["count"].transform("sum")
    pct_class_given_cond_cell = counts.copy()
    pct_class_given_cond_cell["pct"] = np.where(total_per_cond_cell > 0,
                                                counts["count"] / total_per_cond_cell,
                                                np.nan)
    results["pct_class_given_cond_cell"] = pct_class_given_cond_cell

    total_per_class = counts.groupby(["classification"])["count"].transform("sum")
    pct_cond_cell_given_class = counts.copy()
    pct_cond_cell_given_class["pct"] = np.where(total_per_class > 0,
                                                counts["count"] / total_per_class,
                                                np.nan)
    results["pct_cond_cell_given_class"] = pct_cond_cell_given_class

    overall = (
        df.groupby(["condition", "classification"], dropna=False)
          .size()
          .rename("count")
          .reset_index()
    )
    results["overall_counts_by_cond_class"] = overall

    lamp2_df = df[df["lamp2"] == True].copy()
    L_counts = (
        lamp2_df.groupby(["condition", "classification"], dropna=False)
                .size()
                .rename("count")
                .reset_index()
    )
    results["LAMP2_counts_by_cond_class"] = L_counts

    total_per_cond_L = L_counts.groupby(["condition"])["count"].transform("sum")
    L_pct = L_counts.copy()
    L_pct["pct"] = np.where(total_per_cond_L > 0, L_counts["count"] / total_per_cond_L, np.nan)
    results["LAMP2_pct_class_given_cond"] = L_pct

    return results


def chi_square_tests(results: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out = {}
    if chi2_contingency is None:
        return out

    overall = results["overall_counts_by_cond_class"]
    if not overall.empty:
        overall_pivot = overall.pivot_table(index="condition", columns="classification", values="count", fill_value=0)
        if overall_pivot.shape[0] > 1 and overall_pivot.shape[1] > 1:
            chi2, p, dof, expected = chi2_contingency(overall_pivot.values)
            out["overall_condition_vs_classification"] = pd.DataFrame({"chi2": [chi2], "p_value": [p], "dof": [dof]})

    L_counts = results.get("LAMP2_counts_by_cond_class", pd.DataFrame())
    if not L_counts.empty:
        L_pivot = L_counts.pivot_table(index="condition", columns="classification", values="count", fill_value=0)
        if L_pivot.shape[0] > 1 and L_pivot.shape[1] > 1 and L_pivot.values.sum() > 0:
            chi2, p, dof, expected = chi2_contingency(L_pivot.values)
            out["lamp2_condition_vs_classification"] = pd.DataFrame({"chi2": [chi2], "p_value": [p], "dof": [dof]})

    return out


def make_plots(results: Dict[str, pd.DataFrame], out_dir: Path):
    if plt is None:
        print("matplotlib is not available; skipping plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    df = results["pct_class_given_cond_cell"].copy()
    if not df.empty:
        for (cond, cell), sub in df.groupby(["condition", "cell_type"]):
            classes = sub["classification"].tolist()
            pct = sub["pct"].fillna(0).values

            fig = plt.figure()
            plt.bar(range(len(classes)), pct)
            plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
            plt.ylabel("P(class | condition, cell_type)")
            plt.title(f"{cond} — {cell}")
            plt.tight_layout()
            fig_path = out_dir / f"pct_class_given_{slug(cond)}_{slug(cell)}.png"
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)

    L = results["LAMP2_pct_class_given_cond"]
    if not L.empty:
        for cond, sub in L.groupby("condition"):
            classes = sub["classification"].tolist()
            pct = sub["pct"].fillna(0).values

            fig = plt.figure()
            plt.bar(range(len(classes)), pct)
            plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
            plt.ylabel("P(class | condition) — LAMP2 only")
            plt.title(f"{cond}")
            plt.tight_layout()
            fig_path = out_dir / f"lamp2_pct_class_given_{slug(cond)}.png"
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def main():
    ap = argparse.ArgumentParser(description="Compile hyperspectral droplet classifications across workbooks.")
    ap.add_argument("--data-root", type=Path, required=True, help="Root directory to search for Hyperspectral_Results_*.xlsx files.")
    ap.add_argument("--out", type=Path, required=True, help="Directory to write outputs (CSVs and optional plots).")
    ap.add_argument("--make-plots", action="store_true", help="If set, save basic bar charts (matplotlib).")
    ap.add_argument("--dry-run", action="store_true", help="If set, print which files would be processed and exit.")
    args = ap.parse_args()

    xlsx_files = sorted(args.data_root.rglob("Hyperspectral_Results_*.xlsx"))

    if args.dry_run:
        for p in xlsx_files:
            print(p)
        return

    if not xlsx_files:
        print("No Hyperspectral_Results_*.xlsx files found under:", args.data_root)
        return

    all_rows = []
    for path in xlsx_files:
        try:
            df = load_classification_sheet(path)
            all_rows.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")

    if not all_rows:
        print("No usable classification sheets found.")
        return

    tidy = pd.concat(all_rows, ignore_index=True)
    
    # Debug: show a few paths whose condition stayed Unknown
    unknown_mask = tidy["condition"].eq("Unknown")
    if unknown_mask.any():
        print("[DEBUG] Example paths with Unknown condition:")
        for p in tidy.loc[unknown_mask, "file"].head(10):
            print("  ", p)

    args.out.mkdir(parents=True, exist_ok=True)
    tidy_path = args.out / "combined_classification_rows.csv"
    tidy.to_csv(tidy_path, index=False)
    print("Wrote:", tidy_path)

    results = aggregate_distributions(tidy)

    for key, df in results.items():
        out_csv = args.out / f"{key}.csv"
        df.to_csv(out_csv, index=False)
        print("Wrote:", out_csv)

    chi = chi_square_tests(results)
    for key, df in chi.items():
        out_csv = args.out / f"chisq_{key}.csv"
        df.to_csv(out_csv, index=False)
        print("Wrote:", out_csv)

    if args.make_plots:
        plots_dir = args.out / "plots"
        make_plots(results, plots_dir)
        print("Saved plots to:", plots_dir)


if __name__ == "__main__":
    main()
