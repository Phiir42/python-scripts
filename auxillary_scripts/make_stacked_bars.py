#!/usr/bin/env python3
"""
make_stacked_bars.py — Spyder-friendly figure generator for hyperspectral summary

What this script does
---------------------
- Reads the consolidated CSV produced by run_postclassify.py
  (default: "Hyperspectral_Classification_Summary.csv").
- Builds a tidy percentage table for stacked bars, stratified by:
    * Cell type: Microglia (IBA1), Astrocytes (GFAP), Neurons (TUJ/TUJ_Ck), Extracellular
    * Object type: LAMP2+, Lipid, Lipofuscin, Lipidated Lipofuscin
    * Clinicogenotype: Control, AD33, AD44
  Buckets within stacked bars: myelin_like, TG_unsat, TG_sat, unknown
- Saves the tidy inputs to an Excel file, and produces 16 PNG stacked bar plots.
- Additionally generates **two more sets** of non-stacked bar plots (saved to a
  second output directory):
    1) **Myelin-like object %** per clinicogenotype.
    2) **Unsaturation rate of triglycerides** = 100 * TG_unsat / (TG_unsat + TG_sat).
       (When TG_unsat + TG_sat == 0 for a group, the rate is reported as NaN and
        plotted as 0%.)
  These sets also include Extracellular, yielding 16 plots per set.

How to use in Spyder
--------------------
1) Open this file in Spyder.
2) Adjust the CONFIG block below (paths/colors/fonts) if needed.
3) Press Run ▶

Dependencies: pandas, numpy, matplotlib, openpyxl
"""
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List

# -------------------------
# CONFIG — edit these paths if desired
# -------------------------
# Path to the consolidated CSV from run_postclassify.py
CSV_PATH = r"D:\OneDrive - Stanford\Research Documents\AD Project\2025\Hyperspectral_Classification_Summary.csv"
# Output directory for stacked figures and the Excel summary
OUTPUT_DIR = r"D:\OneDrive - Stanford\Research Documents\AD Project\2025\Figures\Classification_Plots"
# SECOND output directory for the **non-stacked** metric plots
OUTPUT_DIR_METRICS = r"D:\OneDrive - Stanford\Research Documents\AD Project\2025\Figures\Classification_Plots_Isolated"
# Name of the tidy Excel file with the percentages used to draw the bars
PLOT_INPUTS_XLSX = "plot_inputs_stacked_bars.xlsx"

# Clinicogenotype order & bar edge/fill colors
GENO_ORDER = ["Control", "AD33", "AD44"]
# Edge colors (inspired by provided reference image)
EDGE_COLORS = {"Control": "#1536D3", "AD33": "#6B2C91", "AD44": "#9E0B0F"}
# Fill colors for **non-stacked** bars (use genotype colors)
FILL_COLORS = EDGE_COLORS.copy()

# Segment (stack) fill colors for the 4 classification buckets
SEG_COLORS = {
    "myelin_like": "#4C72B0",  # blue
    "TG_unsat":    "#55A868",  # green
    "TG_sat":      "#C44E52",  # red
    "unknown":     "#8172B2",  # purple-gray
}

# Fonts / sizes similar to the sample image
AX_LABELSIZE = 16
TICK_LABELSIZE = 14
TITLE_SIZE = 16
BAR_WIDTH = 0.6
FIGSIZE = (6, 5)
DPI = 150

# -------------------------
# Helper functions
# -------------------------

def _get_col(df: pd.DataFrame, name_like: str) -> Optional[str]:
    """Case-insensitive column finder with underscore/space tolerance."""
    lc = {c.lower(): c for c in df.columns}
    key = name_like.lower()
    if key in lc:
        return lc[key]
    for c in df.columns:
        if c.lower().replace(" ", "_") == key.replace(" ", "_"):
            return c
    return None


def _get_donor_col(df: pd.DataFrame) -> Optional[str]:
    """Try to find a 'donor-like' column name."""
    candidates = ["donor", "donor_id", "case_id", "subject_id", "subject", "case"]
    for name in candidates:
        col = _get_col(df, name)
        if col is not None:
            return col
    return None


def _std_geno(x) -> str:
    if pd.isna(x):
        return "Other"
    s = str(x).strip().lower()
    if "ad44" in s:
        return "AD44"
    if "ad33" in s:
        return "AD33"
    if "control" in s or s == "ctrl":
        return "Control"
    return "Other"


def _extract_donor_token(source_val: object) -> str:
    """
    Extract a donor code from the source_file string.

    Donor is encoded in the *path*, e.g.:
        AD3a\Hyperspectral_Results_AD33_AstrocyteSpectrumCH.xlsx
    where we want "AD3a".

    Heuristic:
    1) Split the path on / and \ and look for components that look like
       donor codes: AD<digits><letter>, e.g. AD3a, AD10b.
       (This will NOT match AD33 / AD44 because those end in digits only.)
    2) If none found in the path components, fall back to a basename-based
       heuristic so we don't crash on odd files.
    """
    if pd.isna(source_val):
        return "Unknown"
    s = str(source_val).strip()
    if not s:
        return "Unknown"

    # Split on both Windows and Unix separators
    parts = [p for p in re.split(r"[\\/]+", s) if p]

    donor_pattern = re.compile(r"AD(\d+)([A-Za-z])", flags=re.IGNORECASE)

    # 1) Prefer donor-style codes from any path component (dirs or filename)
    for part in parts:
        root = os.path.splitext(part)[0]
        m = donor_pattern.search(root)
        if m:
            return m.group(0)

    # 2) Fallback: basename heuristic (for weird paths without AD<digit><letter>)
    base = os.path.basename(s)
    root = os.path.splitext(base)[0]
    tokens = [t for t in re.split(r"[_\s\-]+", root) if t]

    if not tokens:
        return "Unknown"

    generic = {"hyperspectral", "results", "summary", "classification"}
    for t in reversed(tokens):
        if t.lower() not in generic:
            return t

    return tokens[0]


def _truthy(v) -> bool:
    if pd.isna(v):
        return False
    if isinstance(v, (int, float)):
        return v == 1 or v is True
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def _bucket(lbl: object) -> str:
    if pd.isna(lbl):
        return "unknown"
    s = str(lbl).strip().lower()
    if s in {"myelin_like", "myelin-like", "myelinlike"}:
        return "myelin_like"
    if s in {"tg_unsat", "unsaturated_tg", "tg-unsat"}:
        return "TG_unsat"
    if s in {"tg_sat", "saturated_tg", "tg-sat"}:
        return "TG_sat"
    if s in {"uncertain", "unknown", ""}:
        return "unknown"
    return "unknown"


def _cell_type(marker_str: object) -> str:
    """Map marker to cell-type label; fallback to Extracellular if missing/unrecognized."""
    if pd.isna(marker_str):
        return "Extracellular"
    s = str(marker_str).strip()
    if s == "":
        return "Extracellular"
    su = s.upper()
    # Explicit extracellular cues
    if "EXTRA" in su or su in {"EC", "EXTRACELLULAR"}:
        return "Extracellular"
    if "IBA1" in su:
        return "Microglia (IBA1)"
    if "GFAP" in su:
        return "Astrocytes (GFAP)"
    if "TUJ_CK" in su or "TUJ CK" in su or "TUJ" in su:
        return "Neurons (TUJ)"
    # Fallback: treat as extracellular if not a recognized marker
    return "Extracellular"


def _object_type_mask(row: pd.Series, name: str) -> bool:
    cat = str(row["__category__"]).strip().lower()
    if name == "LAMP2+":
        return bool(row["__lamppos__"])
    if name == "Lipid":
        return cat == "lipid"
    if name == "Lipofuscin":
        return cat == "lipofuscin"
    if name == "Lipidated Lipofuscin":
        return cat in {"lipidated lipofuscin", "lipidated_lipofuscin", "lipidated-lipofuscin"}
    return False


# -------------------------
# Core computation
# -------------------------

def compute_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """Return tidy table of percentages for stacked bar plots."""
    col_marker = _get_col(df, "Cell Marker") or _get_col(df, "Cell_Marker") or _get_col(df, "CellMarker")
    col_category = _get_col(df, "Category")
    col_lamp2 = _get_col(df, "LAMP2_Coloc") or _get_col(df, "LAMP2")
    col_label = _get_col(df, "class_label") or _get_col(df, "Class") or _get_col(df, "label")
    col_geno = _get_col(df, "clinicogenotype")

    # Prepare working columns
    work = df.copy()
    work["__marker__"] = work[col_marker].astype(str) if col_marker else ""
    work["__category__"] = work[col_category].astype(str) if col_category else ""
    work["__lamppos__"] = work[col_lamp2].apply(_truthy) if col_lamp2 else False
    work["__bucket__"] = work[col_label].apply(_bucket) if col_label else "unknown"
    work["__geno__"] = work[col_geno].apply(_std_geno) if col_geno else "Other"
    work["__celltype__"] = work["__marker__"].apply(_cell_type)

    object_types = ["LAMP2+", "Lipid", "Lipofuscin", "Lipidated Lipofuscin"]
    cell_types = ["Microglia (IBA1)", "Astrocytes (GFAP)", "Neurons (TUJ)", "Extracellular"]
    geno_order = GENO_ORDER
    buckets = ["myelin_like", "TG_unsat", "TG_sat", "unknown"]

    rows = []
    for cell in cell_types:
        sub_cell = work[work["__celltype__"] == cell]
        for obj in object_types:
            sub_obj = sub_cell[sub_cell.apply(lambda r: _object_type_mask(r, obj), axis=1)]
            for geno in geno_order:
                sub_g = sub_obj[sub_obj["__geno__"] == geno]
                n = len(sub_g)
                if n == 0:
                    perc = {b: 0.0 for b in buckets}
                else:
                    counts = sub_g["__bucket__"].value_counts(dropna=False)
                    perc = {b: 100.0 * counts.get(b, 0) / n for b in buckets}
                row = {"cell_type": cell, "object_type": obj, "clinicogenotype": geno, "N": n}
                row.update(perc)
                rows.append(row)

    return pd.DataFrame(rows)


def compute_percentages_by_donor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return tidy table of percentages for stacked bar plots **per donor**.

    Donor = (donor_code_from_source_file, clinicogenotype)

    donor_code_from_source_file is parsed as the first chunk of the basename
    before '_' / space / '-', e.g. source_file ".../AD3a_region1.nd2" -> "AD3a".
    """
    col_marker = _get_col(df, "Cell Marker") or _get_col(df, "Cell_Marker") or _get_col(df, "CellMarker")
    col_category = _get_col(df, "Category")
    col_lamp2 = _get_col(df, "LAMP2_Coloc") or _get_col(df, "LAMP2")
    col_label = _get_col(df, "class_label") or _get_col(df, "Class") or _get_col(df, "label")
    col_geno = _get_col(df, "clinicogenotype")

    # Try to find a source_file-like column
    col_source = (
        _get_col(df, "source_file")
        or _get_col(df, "SourceFile")
        or _get_col(df, "Source_File")
        or _get_col(df, "source")
        or _get_col(df, "file")
    )

    if col_source is None:
        raise ValueError(
            "Could not find a 'source_file'-like column. "
            "Tried: source_file, SourceFile, Source_File, source, file."
        )

    # Prepare working columns
    work = df.copy()
    work["__marker__"] = work[col_marker].astype(str) if col_marker else ""
    work["__category__"] = work[col_category].astype(str) if col_category else ""
    work["__lamppos__"] = work[col_lamp2].apply(_truthy) if col_lamp2 else False
    work["__bucket__"] = work[col_label].apply(_bucket) if col_label else "unknown"
    work["__geno__"] = work[col_geno].apply(_std_geno) if col_geno else "Other"
    work["__celltype__"] = work["__marker__"].apply(_cell_type)

    work["__source__"] = work[col_source]
    work["__donor_token__"] = work["__source__"].apply(_extract_donor_token)

    object_types = ["LAMP2+", "Lipid", "Lipofuscin", "Lipidated Lipofuscin"]
    cell_types = ["Microglia (IBA1)", "Astrocytes (GFAP)", "Neurons (TUJ)", "Extracellular"]
    buckets = ["myelin_like", "TG_unsat", "TG_sat", "unknown"]

    rows = []
    for cell in cell_types:
        sub_cell = work[work["__celltype__"] == cell]
        for obj in object_types:
            sub_obj = sub_cell[sub_cell.apply(lambda r: _object_type_mask(r, obj), axis=1)]
            if sub_obj.empty:
                continue

            # Donor = (donor_token, genotype)
            sub_obj = sub_obj.copy()
            sub_obj["__donor_key__"] = (
                sub_obj["__donor_token__"].astype(str) + "|" + sub_obj["__geno__"].astype(str)
            )

            for donor_key, sub_d in sub_obj.groupby("__donor_key__"):
                n = len(sub_d)
                # shouldn't happen with groupby, but guard anyway
                if n == 0:
                    continue

                counts = sub_d["__bucket__"].value_counts(dropna=False)
                perc = {b: 100.0 * counts.get(b, 0) / n for b in buckets}

                donor_token = sub_d["__donor_token__"].iloc[0]  # e.g. "AD3a"
                geno_mode = sub_d["__geno__"].mode()
                geno = geno_mode.iat[0] if not geno_mode.empty else "Other"

                row = {
                    "cell_type": cell,
                    "object_type": obj,
                    "donor": donor_token,          # label on the x-axis ("AD3a")
                    "clinicogenotype": geno,       # e.g. "AD33"
                    "N": n,
                }
                row.update(perc)
                rows.append(row)

    return pd.DataFrame(rows)


def compute_metric_tables(plot_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """From the stacked-bar percentage table, compute two metric tables:
    1) myelin-like %
    2) TG unsaturation rate = 100 * TG_unsat / (TG_unsat + TG_sat)

    Works both for genotype-pooled and donor-level tables. If a 'donor'
    column is present, it is preserved in the outputs.
    """
    df = plot_df.copy()

    base_cols = ["cell_type", "object_type", "clinicogenotype", "N"]
    if "donor" in df.columns:
        base_cols = ["cell_type", "object_type", "donor", "clinicogenotype", "N"]

    # Myelin-like % table
    myelin = df[base_cols + ["myelin_like"]].copy()
    myelin.rename(columns={"myelin_like": "myelin_like_pct"}, inplace=True)

    # TG unsaturation rate table
    denom = (df.get("TG_unsat", 0) + df.get("TG_sat", 0))
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = np.where(denom > 0, 100.0 * df.get("TG_unsat", 0) / denom, np.nan)
    tg_rate = df[base_cols].copy()
    tg_rate["tg_unsat_rate_pct"] = rate

    return myelin, tg_rate


# -------------------------
# Plotting
# -------------------------

def make_stacked_plots(plot_df: pd.DataFrame, out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)

    # Matplotlib style
    plt.rcParams.update({
        "figure.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": AX_LABELSIZE,
        "xtick.labelsize": TICK_LABELSIZE,
        "ytick.labelsize": TICK_LABELSIZE,
    })

    object_types = ["LAMP2+", "Lipid", "Lipofuscin", "Lipidated Lipofuscin"]
    cell_types = ["Microglia (IBA1)", "Astrocytes (GFAP)", "Neurons (TUJ)", "Extracellular"]
    geno_order = GENO_ORDER
    buckets = ["myelin_like", "TG_unsat", "TG_sat", "unknown"]

    saved = []
    for cell in cell_types:
        for obj in object_types:
            sub = plot_df[(plot_df["cell_type"] == cell) & (plot_df["object_type"] == obj)]
            # Ensure ordering
            sub = sub.set_index("clinicogenotype").reindex(geno_order).reset_index()
            x = np.arange(len(geno_order))

            fig = plt.figure(figsize=FIGSIZE)
            ax = plt.gca()

            bottoms = np.zeros(len(geno_order))
            for b in buckets:
                vals = sub[b].values if b in sub.columns else np.zeros(len(geno_order))
                ax.bar(
                    x,
                    vals,
                    BAR_WIDTH,
                    bottom=bottoms,
                    label=b.replace("_", " ").title(),
                    color=SEG_COLORS.get(b, "#999999"),
                    edgecolor=[EDGE_COLORS[g] for g in geno_order],
                    linewidth=2,
                )
                bottoms += vals

            ax.set_xticks(x)
            ax.set_xticklabels(geno_order)
            ax.set_ylim(0, 100)
            ax.set_ylabel("Objects (%)")
            ax.set_title(f"{cell} — {obj}", fontsize=TITLE_SIZE)
            ax.legend(
                title="Classification",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0.0,
                fontsize=10,
                title_fontsize=11,
            )
            ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

            fname = f"{cell.replace(' ','_').replace('(','').replace(')','')}_{obj.replace(' ','_')}_stacked.png"
            fpath = os.path.join(out_dir, fname)
            plt.tight_layout()
            plt.savefig(fpath, bbox_inches="tight")
            plt.close(fig)
            saved.append(fpath)

    return saved


def make_metric_bars(metric_df: pd.DataFrame, metric_col: str, ylabel: str, title_suffix: str, out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        "figure.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": AX_LABELSIZE,
        "xtick.labelsize": TICK_LABELSIZE,
        "ytick.labelsize": TICK_LABELSIZE,
    })

    object_types = ["LAMP2+", "Lipid", "Lipofuscin", "Lipidated Lipofuscin"]
    cell_types = ["Microglia (IBA1)", "Astrocytes (GFAP)", "Neurons (TUJ)", "Extracellular"]
    geno_order = GENO_ORDER

    saved = []
    for cell in cell_types:
        for obj in object_types:
            sub = metric_df[(metric_df["cell_type"] == cell) & (metric_df["object_type"] == obj)]
            sub = sub.set_index("clinicogenotype").reindex(geno_order).reset_index()
            x = np.arange(len(geno_order))
            vals = sub[metric_col].fillna(0).values

            fig = plt.figure(figsize=FIGSIZE)
            ax = plt.gca()

            # Bars filled with clinicogenotype colors (not stacked)
            ax.bar(
                x, vals, BAR_WIDTH,
                edgecolor=[EDGE_COLORS[g] for g in geno_order],
                color=[FILL_COLORS[g] for g in geno_order],
                linewidth=2
            )

            ax.set_xticks(x)
            ax.set_xticklabels(geno_order)
            ax.set_ylim(0, 100)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{cell} — {obj} — {title_suffix}", fontsize=TITLE_SIZE)
            ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

            # Save
            obj_tag = obj.replace(' ', '_')
            cell_tag = cell.replace(' ', '_').replace('(', '').replace(')', '')
            fname = f"{cell_tag}_{obj_tag}_{metric_col}.png"
            fpath = os.path.join(out_dir, fname)
            plt.tight_layout()
            plt.savefig(fpath, bbox_inches="tight")
            plt.close(fig)
            saved.append(fpath)

    return saved


def make_stacked_plots_by_donor(plot_df: pd.DataFrame, out_dir: str) -> List[str]:
    """Stacked bars per donor (x-axis = donor, stacks = classification buckets)."""
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        "figure.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": AX_LABELSIZE,
        "xtick.labelsize": TICK_LABELSIZE,
        "ytick.labelsize": TICK_LABELSIZE,
    })

    object_types = ["LAMP2+", "Lipid", "Lipofuscin", "Lipidated Lipofuscin"]
    cell_types = ["Microglia (IBA1)", "Astrocytes (GFAP)", "Neurons (TUJ)", "Extracellular"]
    buckets = ["myelin_like", "TG_unsat", "TG_sat", "unknown"]
    geno_rank = {g: i for i, g in enumerate(GENO_ORDER)}

    saved = []
    for cell in cell_types:
        for obj in object_types:
            sub = plot_df[(plot_df["cell_type"] == cell) & (plot_df["object_type"] == obj)]
            if sub.empty:
                continue

            # Order donors by genotype, then donor name
            sub = sub.copy()
            sub["__geno_rank__"] = sub["clinicogenotype"].map(geno_rank).fillna(len(GENO_ORDER))
            sub = sub.sort_values(["__geno_rank__", "donor"])
            donors = sub["donor"].tolist()
            x = np.arange(len(donors))

            fig = plt.figure(figsize=FIGSIZE)
            ax = plt.gca()

            bottoms = np.zeros(len(donors))
            edgecolors = [EDGE_COLORS.get(g, "#444444") for g in sub["clinicogenotype"]]

            for b in buckets:
                vals = sub[b].values if b in sub.columns else np.zeros(len(donors))
                ax.bar(
                    x,
                    vals,
                    BAR_WIDTH,
                    bottom=bottoms,
                    label=b.replace("_", " ").title(),
                    color=SEG_COLORS.get(b, "#999999"),
                    edgecolor=edgecolors,
                    linewidth=2,
                )
                bottoms += vals

            ax.set_xticks(x)
            ax.set_xticklabels(donors, rotation=45, ha="right")
            ax.set_ylim(0, 100)
            ax.set_ylabel("Objects (%)")
            ax.set_title(f"{cell} — {obj} (per donor)", fontsize=TITLE_SIZE)
            ax.legend(
                title="Classification",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0.0,
                fontsize=10,
                title_fontsize=11,
            )
            ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

            cell_tag = cell.replace(" ", "_").replace("(", "").replace(")", "")
            obj_tag = obj.replace(" ", "_")
            fname = f"{cell_tag}_{obj_tag}_stacked_by_donor.png"
            fpath = os.path.join(out_dir, fname)
            plt.tight_layout()
            plt.savefig(fpath, bbox_inches="tight")
            plt.close(fig)
            saved.append(fpath)

    return saved


def make_metric_bars_by_donor(
    metric_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title_suffix: str,
    out_dir: str
) -> List[str]:
    """Non-stacked metric bars per donor (x-axis = donor, bar color = clinicogenotype)."""
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        "figure.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": AX_LABELSIZE,
        "xtick.labelsize": TICK_LABELSIZE,
        "ytick.labelsize": TICK_LABELSIZE,
    })

    object_types = ["LAMP2+", "Lipid", "Lipofuscin", "Lipidated Lipofuscin"]
    cell_types = ["Microglia (IBA1)", "Astrocytes (GFAP)", "Neurons (TUJ)", "Extracellular"]
    geno_rank = {g: i for i, g in enumerate(GENO_ORDER)}

    saved = []
    for cell in cell_types:
        for obj in object_types:
            sub = metric_df[(metric_df["cell_type"] == cell) & (metric_df["object_type"] == obj)]
            if sub.empty:
                continue

            sub = sub.copy()
            sub["__geno_rank__"] = sub["clinicogenotype"].map(geno_rank).fillna(len(GENO_ORDER))
            sub = sub.sort_values(["__geno_rank__", "donor"])
            donors = sub["donor"].tolist()
            x = np.arange(len(donors))
            vals = sub[metric_col].fillna(0).values
            edgecolors = [EDGE_COLORS.get(g, "#444444") for g in sub["clinicogenotype"]]
            facecolors = [FILL_COLORS.get(g, "#CCCCCC") for g in sub["clinicogenotype"]]

            fig = plt.figure(figsize=FIGSIZE)
            ax = plt.gca()
            ax.bar(
                x, vals, BAR_WIDTH,
                edgecolor=edgecolors,
                color=facecolors,
                linewidth=2,
            )

            ax.set_xticks(x)
            ax.set_xticklabels(donors, rotation=45, ha="right")
            ax.set_ylim(0, 100)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{cell} — {obj} — {title_suffix} (per donor)", fontsize=TITLE_SIZE)
            ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

            cell_tag = cell.replace(" ", "_").replace("(", "").replace(")", "")
            obj_tag = obj.replace(" ", "_")
            fname = f"{cell_tag}_{obj_tag}_{metric_col}_by_donor.png"
            fpath = os.path.join(out_dir, fname)
            plt.tight_layout()
            plt.savefig(fpath, bbox_inches="tight")
            plt.close(fig)
            saved.append(fpath)

    return saved


# -------------------------
# Main
# -------------------------

def run() -> None:
    if not os.path.isfile(CSV_PATH):
        print(f"[make_stacked_bars] CSV_PATH not found: {CSV_PATH}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_METRICS, exist_ok=True)

    print("[make_stacked_bars] Loading:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    print("[make_stacked_bars] Computing percentages…")
    plot_df = compute_percentages(df)

    # Save tidy inputs
    xlsx_path = os.path.join(OUTPUT_DIR, PLOT_INPUTS_XLSX)
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            plot_df.to_excel(writer, sheet_name="percentages", index=False)
            myelin_df, tg_rate_df = compute_metric_tables(plot_df)
            myelin_df.to_excel(writer, sheet_name="myelin_like_pct", index=False)
            tg_rate_df.to_excel(writer, sheet_name="tg_unsat_rate_pct", index=False)
        print(f"[make_stacked_bars] Wrote plot inputs: {xlsx_path} ({len(plot_df)} rows)")
    except Exception as e:
        print(f"[make_stacked_bars] Failed writing plot inputs Excel: {e}")
        # still compute metrics even if Excel fails
        myelin_df, tg_rate_df = compute_metric_tables(plot_df)

    print("[make_stacked_bars] Rendering stacked figures…")
    saved_stacked = make_stacked_plots(plot_df, OUTPUT_DIR)
    print("[make_stacked_bars] Saved", len(saved_stacked), "stacked plots in:", OUTPUT_DIR)

    print("[make_stacked_bars] Rendering metric figures…")
    # Myelin-like % bars
    saved_myelin = make_metric_bars(
        myelin_df,
        metric_col="myelin_like_pct",
        ylabel="Myelin-like objects (%)",
        title_suffix="Myelin-like %",
        out_dir=OUTPUT_DIR_METRICS,
    )
    # TG unsaturation rate bars
    saved_tg = make_metric_bars(
        tg_rate_df,
        metric_col="tg_unsat_rate_pct",
        ylabel="Unsaturated TG / Total TG (%)",
        title_suffix="TG Unsaturation Rate",
        out_dir=OUTPUT_DIR_METRICS,
    )
    print("[make_stacked_bars] Saved", len(saved_myelin) + len(saved_tg), "metric plots in:", OUTPUT_DIR_METRICS)


def run_by_donor() -> None:
    if not os.path.isfile(CSV_PATH):
        print(f"[make_stacked_bars/by_donor] CSV_PATH not found: {CSV_PATH}")
        return

    # Separate output dirs so you don't overwrite the pooled plots
    out_dir = OUTPUT_DIR + "_ByDonor"
    out_dir_metrics = OUTPUT_DIR_METRICS + "_ByDonor"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_metrics, exist_ok=True)

    print("[make_stacked_bars/by_donor] Loading:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    print("[make_stacked_bars/by_donor] Computing per-donor percentages…")
    plot_df = compute_percentages_by_donor(df)

    # Save tidy inputs
    xlsx_path = os.path.join(out_dir, "plot_inputs_stacked_bars_by_donor.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            plot_df.to_excel(writer, sheet_name="percentages_by_donor", index=False)
            myelin_df, tg_rate_df = compute_metric_tables(plot_df)
            myelin_df.to_excel(writer, sheet_name="myelin_like_pct_by_donor", index=False)
            tg_rate_df.to_excel(writer, sheet_name="tg_unsat_rate_pct_by_donor", index=False)
        print(f"[make_stacked_bars/by_donor] Wrote plot inputs: {xlsx_path} ({len(plot_df)} rows)")
    except Exception as e:
        print(f"[make_stacked_bars/by_donor] Failed writing plot inputs Excel: {e}")
        myelin_df, tg_rate_df = compute_metric_tables(plot_df)

    print("[make_stacked_bars/by_donor] Rendering stacked figures…")
    saved_stacked = make_stacked_plots_by_donor(plot_df, out_dir)
    print("[make_stacked_bars/by_donor] Saved", len(saved_stacked), "stacked plots in:", out_dir)

    print("[make_stacked_bars/by_donor] Rendering metric figures…")
    saved_myelin = make_metric_bars_by_donor(
        myelin_df,
        metric_col="myelin_like_pct",
        ylabel="Myelin-like objects (%)",
        title_suffix="Myelin-like %",
        out_dir=out_dir_metrics,
    )
    saved_tg = make_metric_bars_by_donor(
        tg_rate_df,
        metric_col="tg_unsat_rate_pct",
        ylabel="Unsaturated TG / Total TG (%)",
        title_suffix="TG Unsaturation Rate",
        out_dir=out_dir_metrics,
    )
    print("[make_stacked_bars/by_donor] Saved", len(saved_myelin) + len(saved_tg), "metric plots in:", out_dir_metrics)


if __name__ == "__main__":
    run()
    run_by_donor()