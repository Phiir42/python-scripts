#!/usr/bin/env python3
"""
Extract 2850/2930 ratios from Hyperspectral_Results_*.xlsx files and summarize.

Adds cortical layer parsing from file paths (L1..L6, WM) and produces:
- hyperspectral_ratios_droplets.csv    (all droplets, with layer info)
- hyperspectral_ratios_summary.xlsx:
    * Summary                                  (grouped stats)
    * Summary_By_Layer                         (grouped stats incl. layer)
    * Intracellular_Lipid_Points               (9 columns, individual ratios)
    * Intracellular_Lipofuscin_Points          (9 columns, individual ratios)
    * Intracellular_Lipidated_Lipofuscin_Points(9 columns, individual ratios)
"""

from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from typing import Dict


# ---------------------------------------------------------------------
# Hardcoded paths for Spyder
# ---------------------------------------------------------------------
INDIR = Path(r"D:/OneDrive - Stanford/Research Documents/AD Project/2025")
OUTDIR = Path(
    r"D:/OneDrive - Stanford/Research Documents/AD Project/2025/hyperspec_ratios"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
FILE_GLOB = "Hyperspectral_Results_*.xlsx"
SHEET_NAME = "Raw Data"
COL_WN_2850 = "Wavenumber 24"   # 2850 cm^-1
COL_WN_2930 = "Wavenumber 13"   # 2930 cm^-1
MIN_I2930 = 0.0                 # drop rows with I2930 <= this value

CONDITIONS = ["Control", "AD33", "AD44"]
CELL_TYPES = ["Microglia", "Astrocytes", "Neurons"]

# Robust category normalization (lowercased, spaces normalized)
CATEGORY_ALIASES = {
    "lipid": {"lipid", "lipids"},
    "lipofuscin": {"lipofuscin"},
    "lipidated_lipofuscin": {
        "lipidated lipofuscin",
        "lipidated_lipofuscin",
        "lipidatedlipofuscin",
        "lipid_lipofuscin",
    },
}

# Layer ordering & mapping
LAYER_ORDER = ["Layer I", "Layer II", "Layer III",
               "Layer IV", "Layer V", "Layer VI", "White Matter"]
LAYER_CODE_TO_NAME = {
    "L1": "Layer I",
    "L2": "Layer II",
    "L3": "Layer III",
    "L4": "Layer IV",
    "L5": "Layer V",
    "L6": "Layer VI",
    "WM": "White Matter",
}
LEGEND_LABELS = {
    "Control": "Non-dementia control",
    "AD33": r"AD $ \it{APOE} \ \varepsilon3/\varepsilon3 $",
    "AD44": r"AD $ \it{APOE} \ \varepsilon4/\varepsilon4 $",
}

COLORS: Dict[str, str] = {
    "Control": "#1f77b4",   # blue
    "AD33": "#c000c0",      # magenta
    "AD44": "#d62728",      # red
}

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


def _norm_text(s: pd.Series) -> pd.Series:
    """Normalize strings: lowercase, strip, collapse underscores to spaces."""
    return (
        s.astype(str)
        .str.strip()
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )


def _is_category(cat_series: pd.Series, target_key: str) -> pd.Series:
    """Return mask where category is in the alias set for target_key."""
    aliases = CATEGORY_ALIASES[target_key]
    return _norm_text(cat_series).isin(aliases)


def infer_layer_from_path(path: Path) -> tuple[str | None, str | None]:
    """
    Infer layer from full path. Looks for whole tokens L1..L6 or WM
    anywhere in the file name or directories.

    Returns:
        (layer_code, layer_name) or (None, None) if not found.
    """
    text = str(path).lower().replace("\\", "/")
    # Tokenize on non-alphanumerics to avoid 'l3' inside other words
    tokens = re.split(r"[^a-z0-9]+", text)
    # Look for l1..l6 or wm
    for t in tokens:
        if re.fullmatch(r"l[1-6]", t):
            code = t.upper()
            return code, LAYER_CODE_TO_NAME.get(code)
        if t == "wm":
            return "WM", LAYER_CODE_TO_NAME["WM"]
    return None, None


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
    layer_code, layer_name = infer_layer_from_path(xlsx_path)

    df = pd.DataFrame(
        {
            "file": xlsx_path.name,
            "category": raw.loc[mask, "Category"],
            "location": raw.loc[mask, "Location"],
            "I2850": i2850[mask],
            "I2930": i2930[mask],
            "condition": cond,
            "cell_type": ctype,
            "layer_code": layer_code,
            "layer": layer_name,
        }
    )
    df["ratio_2850_2930"] = df["I2850"] / df["I2930"]
    return df


def summarize_groups(droplets: pd.DataFrame) -> pd.DataFrame:
    """Group by condition/cell_type/category/location and compute summary stats."""
    gcols = ["condition", "cell_type", "category", "location"]

    def _agg(vals: pd.Series):
        vals = vals.dropna().to_numpy(dtype=float)
        n = vals.size
        mean = np.mean(vals) if n else np.nan
        median = np.median(vals) if n else np.nan
        std = np.std(vals, ddof=1) if n > 1 else np.nan
        sem = std / math.sqrt(n) if n > 1 else np.nan
        return pd.Series({"n": n, "mean": mean, "median": median, "std": std, "sem": sem})

    return (
        droplets.groupby(gcols)["ratio_2850_2930"]
        .apply(_agg)
        .reset_index()
    )


def summarize_groups_by_layer(droplets: pd.DataFrame) -> pd.DataFrame:
    """
    Group by condition/cell_type/category/location/layer and compute summary stats.
    Ensures layers are ordered I..VI, White Matter.
    """
    gcols = ["condition", "cell_type", "category", "location", "layer"]

    def _agg(vals: pd.Series):
        vals = vals.dropna().to_numpy(dtype=float)
        n = vals.size
        mean = np.mean(vals) if n else np.nan
        median = np.median(vals) if n else np.nan
        std = np.std(vals, ddof=1) if n > 1 else np.nan
        sem = std / math.sqrt(n) if n > 1 else np.nan
        return pd.Series({"n": n, "mean": mean, "median": median, "std": std, "sem": sem})

    out = (
        droplets.groupby(gcols)["ratio_2850_2930"]
        .apply(_agg)
        .reset_index()
    )
    # Order layers
    cat = pd.Categorical(out["layer"], categories=LAYER_ORDER, ordered=True)
    out = out.assign(layer=cat).sort_values(["condition", "cell_type", "category",
                                             "location", "layer"])
    return out


def make_intracellular_wide(droplets: pd.DataFrame, category_key: str) -> pd.DataFrame:
    """
    Build a 9-column wide table of individual ratios for *intracellular* droplets
    of a given category (category_key in CATEGORY_ALIASES):

      Microglia_Control, Microglia_AD33, Microglia_AD44,
      Astrocytes_Control, Astrocytes_AD33, Astrocytes_AD44,
      Neurons_Control, Neurons_AD33, Neurons_AD44

    Columns are padded with NaNs to equalize length.
    """
    cat_mask = _is_category(droplets["category"], category_key)
    loc_mask = _norm_text(droplets["location"]).eq("intracellular")

    df_sub = droplets.loc[
        cat_mask & loc_mask, ["condition", "cell_type", "ratio_2850_2930"]
    ]

    columns = []
    series_list = []

    for ctype in CELL_TYPES:
        for cond in CONDITIONS:
            m = (df_sub["cell_type"] == ctype) & (df_sub["condition"] == cond)
            s = df_sub.loc[m, "ratio_2850_2930"].reset_index(drop=True)
            series_list.append(s)
            columns.append(f"{ctype}_{cond}")

    max_len = max((len(s) for s in series_list), default=0)
    padded = [s.reindex(range(max_len)) for s in series_list]

    wide = pd.concat(padded, axis=1)
    wide.columns = columns
    return wide


def plot_intracellular_lipid_ratio_by_layer(
    droplets: pd.DataFrame,
    cell_type: str,
    outdir: Path,
    filename: str | None = None,
    colors: dict | None = None,
    legend_labels: dict | None = None,
    smooth_bands: bool = False,
    smooth_samples: int = 200,
    spline_k: int = 3,
    figsize: tuple[float, float] = (5.0, 5.0),
    dpi: int = 600,
    spine_w: float = 2.2,
    tick_w: float = 2.0,
    y_label: str = "Acyl-chain ratio (2850/2930)",
    transparent: bool = True,
    y_major_step: float | None = None,     # e.g., 0.1 for fixed spacing; None = auto
    set_ymin_to_zero: bool = False         # usually False for ratios
) -> Path:
    """
    Plot mean±SEM shaded error bands of intracellular *lipid* acyl-chain ratio (2850/2930)
    across cortical layers for the given cell_type ('Neurons', 'Astrocytes', 'Microglia').
    Saves a PNG and returns its path.
    """
    # Expect LAYER_ORDER to be defined elsewhere in your script:
    # LAYER_ORDER = ["Layer I","Layer II","Layer III","Layer IV","Layer V","Layer VI","White Matter"]

    if colors is None:
        colors = {"Control": "#1f77b4", "AD33": "#c000c0", "AD44": "#d62728"}
    if legend_labels is None:
        legend_labels = {
            "Control": "Non-dementia control",
            "AD33": r"AD $\it{APOE}\ \varepsilon3/\varepsilon3$",
            "AD44": r"AD $\it{APOE}\ \varepsilon4/\varepsilon4$",
        }

    # ---- filter: specific cell type, intracellular, category=lipid(s) ----
    cat = droplets["category"].astype(str).str.strip().str.lower()
    loc = droplets["location"].astype(str).str.strip().str.lower()

    is_lipid = cat.isin({"lipid", "lipids"})
    is_intra = loc.eq("intracellular")

    sub = droplets.loc[
        (droplets["cell_type"] == cell_type) & is_lipid & is_intra
    ].copy()

    sub = sub.dropna(subset=["layer"])
    if sub.empty:
        raise SystemExit(f"No intracellular lipid ratios found for {cell_type} with valid layer info.")

    # Order layers
    sub["layer"] = pd.Categorical(sub["layer"], categories=LAYER_ORDER, ordered=True)

    # X slots
    x = np.arange(len(LAYER_ORDER), dtype=float)

    # Optional smoother
    def _smooth_xy(xx, yy, samples=smooth_samples, k=spline_k):
        mask = np.isfinite(xx) & np.isfinite(yy)
        xv, yv = xx[mask], yy[mask]
        if len(xv) < (k + 1):
            x_new = np.linspace(xx.min(), xx.max(), samples)
            y_new = np.interp(x_new, xv, yv)
            return x_new, y_new
        from scipy.interpolate import make_interp_spline
        x_new = np.linspace(xv.min(), xv.max(), samples)
        y_new = make_interp_spline(xv, yv, k=k)(x_new)
        return x_new, y_new

    fig, ax = plt.subplots(figsize=figsize)

    # One line + band per condition
    for cond in ["Control", "AD33", "AD44"]:
        d = sub.loc[sub["condition"] == cond]
        if d.empty:
            continue

        grp = (
            d.groupby("layer")["ratio_2850_2930"]
             .agg(["mean", "count", "std"])
             .reindex(LAYER_ORDER)
        )
        mean = grp["mean"].to_numpy(dtype=float)
        sem  = (grp["std"] / np.sqrt(grp["count"].replace(0, np.nan))).to_numpy(dtype=float)
        lo, hi = mean - sem, mean + sem

        c = colors.get(cond, "black")
        if smooth_bands:
            xs, mean_s = _smooth_xy(x, mean)
            _,  lo_s   = _smooth_xy(x, lo)
            _,  hi_s   = _smooth_xy(x, hi)
            ax.plot(xs, mean_s, color=c, linewidth=2.2, label=legend_labels.get(cond, cond))
            ax.fill_between(xs, lo_s, hi_s, color=c, alpha=0.15, linewidth=0)
            ax.plot(x, mean, "o", color=c, ms=4, alpha=0.95)
        else:
            ax.plot(x, mean, color=c, linewidth=2.2, marker="o", ms=4, label=legend_labels.get(cond, cond))
            ax.fill_between(x, lo, hi, color=c, alpha=0.15, linewidth=0)

    # X axis
    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_ORDER, rotation=45, ha="right", fontweight="bold")

    # Y axis
    ax.set_ylabel(y_label, fontweight="bold", fontsize=14)
    if set_ymin_to_zero:
        ax.set_ylim(bottom=0)
    if y_major_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major_step))
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # Spines/ticks (Prism-like)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(spine_w)
    ax.spines["bottom"].set_linewidth(spine_w)

    ax.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False)
    ax.tick_params(axis="both", which="major", direction="out", length=8, width=tick_w, color="black", pad=8)

    # Draw data above axes
    ax.set_axisbelow(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_zorder(0)

    # Bounds to align first/last categories
    ax.set_xlim(-0.5, len(LAYER_ORDER) - 0.5)

    # Legend inside
    ax.legend(frameon=False, loc="upper right", fontsize=11)

    # Layout & save
    fig.subplots_adjust(left=0.17, bottom=0.28, right=0.97, top=0.97)
    outdir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"{cell_type}_Intracellular_Lipid_AcylRatio_byLayer.png".replace(" ", "_")
    fpath = outdir / filename
    fig.savefig(fpath, dpi=dpi, transparent=transparent)
    plt.close(fig)
    return fpath


def plot_lipid_ratio_by_layer_allcells(
    droplets: pd.DataFrame,
    outdir: Path,
    filename: str = "AllCells_Lipid_AcylRatio_byLayer.png",
    split_by_location: bool = False,   # False = combine intra+extra; True = separate
    figsize=(5.0, 5.0),
    dpi: int = 600,
    spine_w: float = 2.2,
    tick_w: float = 2.0,
    y_label: str = "Acyl-chain ratio (2850/2930)",
    y_major_step: float | None = None, # e.g., 0.1 to force spacing; None = auto
    transparent: bool = True
) -> Path:
    """
    Plot mean±SEM of lipid acyl-chain ratio (2850/2930) across layers for all cell types.
    If split_by_location=True, draws separate bands for Intracellular vs Extracellular
    (different linestyles) for each condition.
    """
    # normalize category/location and filter to *lipid* objects only
    cat = droplets["category"].astype(str).str.strip().str.lower()
    is_lipid = cat.isin({"lipid", "lipids"})
    df = droplets.loc[is_lipid].copy()

    # ensure layer order
    df = df.dropna(subset=["layer"])
    if df.empty:
        raise SystemExit("No lipid rows found with valid layer info.")
    df["layer"] = pd.Categorical(df["layer"], categories=LAYER_ORDER, ordered=True)

    x = np.arange(len(LAYER_ORDER), dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    if not split_by_location:
        # combine intra + extra; aggregate by condition × layer
        for cond in ["Control", "AD33", "AD44"]:
            d = df.loc[df["condition"] == cond]
            if d.empty:
                continue
            grp = (d.groupby("layer")["ratio_2850_2930"]
                     .agg(["mean", "count", "std"])
                     .reindex(LAYER_ORDER))
            mean = grp["mean"].to_numpy(dtype=float)
            sem  = (grp["std"] / np.sqrt(grp["count"].replace(0, np.nan))).to_numpy(dtype=float)
            lo, hi = mean - sem, mean + sem

            c = COLORS.get(cond, "black")
            ax.plot(x, mean, color=c, linewidth=2.2, marker="o", ms=4,
                    label=LEGEND_LABELS.get(cond, cond))
            ax.fill_between(x, lo, hi, color=c, alpha=0.15, linewidth=0)
    else:
        # split by location; aggregate by condition × location × layer
        # two linestyles: solid=Intracellular, dashed=Extracellular
        loc_norm = df["location"].astype(str).str.strip().str.capitalize()
        df = df.assign(_loc=loc_norm)  # "Intracellular"/"Extracellular"

        for cond in ["Control", "AD33", "AD44"]:
            for loc_name, ls in (("Intracellular", "-"), ("Extracellular", "--")):
                d = df.loc[(df["condition"] == cond) & (df["_loc"] == loc_name)]
                if d.empty:
                    continue
                grp = (d.groupby("layer")["ratio_2850_2930"]
                         .agg(["mean", "count", "std"])
                         .reindex(LAYER_ORDER))
                mean = grp["mean"].to_numpy(dtype=float)
                sem  = (grp["std"] / np.sqrt(grp["count"].replace(0, np.nan))).to_numpy(dtype=float)
                lo, hi = mean - sem, mean + sem

                label = f"{LEGEND_LABELS.get(cond, cond)} — {loc_name}"
                c = COLORS.get(cond, "black")
                ax.plot(x, mean, color=c, linewidth=2.2, marker="o", ms=3,
                        linestyle=ls, label=label)
                ax.fill_between(x, lo, hi, color=c, alpha=0.12, linewidth=0)

    # X axis
    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_ORDER, rotation=45, ha="right", fontweight="bold")

    # Y axis
    ax.set_ylabel(y_label, fontweight="bold", fontsize=14)
    if y_major_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major_step))
    for lab in ax.get_yticklabels():
        lab.set_fontweight("bold")

    # Spines/ticks
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(spine_w)
    ax.spines["bottom"].set_linewidth(spine_w)

    ax.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False)
    ax.tick_params(axis="both", which="major", direction="out",
                   length=8, width=tick_w, color="black", pad=8)

    # draw data above axes lines
    ax.set_axisbelow(False)
    ax.spines["left"].set_zorder(0)
    ax.spines["bottom"].set_zorder(0)

    # bounds so first/last categories align
    ax.set_xlim(-0.5, len(LAYER_ORDER) - 0.5)

    # Legend inside
    ax.legend(frameon=False, loc="upper right", fontsize=10)

    # Layout & save
    fig.subplots_adjust(left=0.17, bottom=0.28, right=0.97, top=0.97)
    outdir.mkdir(parents=True, exist_ok=True)
    fpath = outdir / filename
    fig.savefig(fpath, dpi=dpi, transparent=transparent)
    plt.close(fig)
    return fpath


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
    except Exception as exc:
        print(f"[WARN] Skipped {fp.name}: {exc}")

droplets = pd.concat(all_rows, ignore_index=True)

# Save droplets (full detail) as CSV
droplets_path = OUTDIR / "hyperspectral_ratios_droplets.csv"
droplets.to_csv(droplets_path, index=False)

for ct in ["Neurons", "Astrocytes", "Microglia"]:
    _ = plot_intracellular_lipid_ratio_by_layer(
        droplets=droplets,
        cell_type=ct,
        outdir=OUTDIR,
        smooth_bands=False,        # True if you want smoothed guides
        y_major_step=None,         # e.g., 0.1 to force specific ticks
        set_ymin_to_zero=False,    # typically False for ratios
        transparent=True
    )
print("[OK] Saved intracellular lipid ratio bands for Neurons, Astrocytes, Microglia.")

# Combine intra+extra (one band per condition):
plot_lipid_ratio_by_layer_allcells(
    droplets=droplets,
    outdir=OUTDIR,
    filename="AllCells_Lipid_AcylRatio_byLayer_COMBINED.png",
    split_by_location=False,
    y_major_step=None,      # e.g., 0.1 to force tick spacing
)

# Summaries
summary = summarize_groups(droplets)
summary_by_layer = summarize_groups_by_layer(droplets)

# Three 9-column “points” sheets (intracellular only)
wide_intra_lipid = make_intracellular_wide(droplets, "lipid")
wide_intra_lipofuscin = make_intracellular_wide(droplets, "lipofuscin")
wide_intra_lipidated = make_intracellular_wide(droplets, "lipidated_lipofuscin")

# Write everything into one Excel file
summary_path = OUTDIR / "hyperspectral_ratios_summary.xlsx"
with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    summary_by_layer.to_excel(writer, sheet_name="Summary_By_Layer", index=False)
    wide_intra_lipid.to_excel(writer, sheet_name="Intracellular_Lipid_Points", index=False)
    wide_intra_lipofuscin.to_excel(
        writer, sheet_name="Intracellular_Lipofuscin_Points", index=False
    )
    wide_intra_lipidated.to_excel(
        writer, sheet_name="Intracellular_Lipidated_Lipofuscin_Points", index=False
    )

print(f"[OK] Droplets CSV: {droplets_path}")
print(f"[OK] Summary XLSX: {summary_path}")
