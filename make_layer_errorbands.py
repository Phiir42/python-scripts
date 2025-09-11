#!/usr/bin/env python3
"""
Plot mean ± SEM as shaded error bands across cortical layers for
Lipids / Lipidated Lipofuscin / Lipofuscin in
Astrocytes, Microglia, and Neurons for Control, AD33, AD44.

- Reads a Prism-style Excel workbook where each sheet is named
  "{Condition} {CellType}" and contains a header row indicating the
  measurement within each layer column (e.g., "Lipids", etc.).
- Produces one PNG per cell type × object category.

Run with optional CLI args, or just execute in Spyder and it will use
the DEFAULT_XLSX and DEFAULT_OUTDIR paths defined below.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import make_interp_spline

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

LAYERS_ORDER: List[str] = [
    "Layer I", "Layer II", "Layer III", "Layer IV",
    "Layer V", "Layer VI", "White Matter",
]

OBJECTS: List[str] = ["Lipids", "Lipidated Lipofuscin", "Lipofuscin"]
CELL_TYPES: List[str] = ["Astrocytes", "Microglia", "Neurons"]
CONDITIONS = ["Control", "AD33", "AD44"]

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

YLABELS: Dict[str, str] = {
    "Lipids": "% lipid area/cell",
    "Lipidated Lipofuscin": "% lipidated lipofuscin area/cell",
    "Lipofuscin": "% lipofuscin area/cell",
}

# Per-category axis ranges and tick steps (adjust as needed)
YLIMS: Dict[str, Tuple[float, float]] = {
    "Lipids": (0.0, 5.0),
    "Lipidated Lipofuscin": (0.0, 7.0),
    "Lipofuscin": (0.0, 22.0),
}
YTICK_STEP: Dict[str, float] = {
    "Lipids": 1.0,
    "Lipidated Lipofuscin": 1.0,
    "Lipofuscin": 2.0,
}

# Aesthetics
FIGSIZE: Tuple[float, float] = (5.0, 5.0)  # inches
DPI: int = 600
SPINE_W: float = 2.4
TICK_W: float = 2.4

# Optional smoothing of the band edges/means (visual guide only)
SMOOTH_BANDS: bool = False
SMOOTH_SAMPLES: int = 200
SPLINE_K: int = 3

# Default paths used when no CLI args are provided
DEFAULT_XLSX: str = (
    r"D:/OneDrive - Stanford/Research Documents/AD Project/2025/"
    r"AD_Lipid_Statistics_CorticalLayers_prism.xlsx"
)
DEFAULT_OUTDIR: str = (
    r"D:/OneDrive - Stanford/Research Documents/AD Project/2025/plots_bands"
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _smooth_xy(
    x: np.ndarray,
    y: np.ndarray,
    samples: int = 200,
    k: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a smoothed (x, y) using a B-spline; falls back to np.interp."""
    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    if len(xv) < (k + 1):
        x_new = np.linspace(x.min(), x.max(), samples)
        y_new = np.interp(x_new, xv, yv)
        return x_new, y_new

    x_new = np.linspace(xv.min(), xv.max(), samples)
    spline = make_interp_spline(xv, yv, k=k)
    y_new = spline(x_new)
    return x_new, y_new


def tidy_sheet(xlsx_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Convert a sheet like 'Control Microglia' into long/tidy form with columns:
    file, layer, category, value, condition, cell_type.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # First row contains sublabels (e.g., "Lipids", "Lipidated Lipofuscin")
    sub = df.iloc[0]
    data = df.iloc[1:].copy()

    # Normalize file id column name to 'file'
    if "file_name" in data.columns:
        data = data.rename(columns={"file_name": "file"})
    elif "file" not in data.columns:
        data.insert(0, "file", np.arange(len(data), dtype=int))

    # Map (layer, object) -> actual column name
    col_map: Dict[Tuple[str, str], str] = {}
    current_layer = None
    for col in df.columns:
        if col in LAYERS_ORDER:
            current_layer = col
        sublabel = str(sub.get(col))
        if current_layer and sublabel in OBJECTS:
            col_map[(current_layer, sublabel)] = col

    # Build tidy rows
    rows = []
    for (layer, obj), colname in col_map.items():
        for _, r in data.iterrows():
            rows.append(
                {
                    "file": r["file"],
                    "layer": layer,
                    "category": obj,
                    "value": pd.to_numeric(r[colname], errors="coerce"),
                }
            )

    tidy = pd.DataFrame(rows)
    cond, ctype = sheet_name.split()
    tidy["condition"] = cond
    tidy["cell_type"] = ctype
    return tidy


def load_all(xlsx_path: str) -> pd.DataFrame:
    """Load all matching sheets and return a single tidy DataFrame."""
    xl = pd.ExcelFile(xlsx_path)
    valid = []
    for name in xl.sheet_names:
        parts = name.split()
        if len(parts) == 2 and parts[0] in CONDITIONS and parts[1] in CELL_TYPES:
            valid.append(name)

    if not valid:
        raise RuntimeError("No matching sheets found. Check sheet names.")

    frames = [tidy_sheet(xlsx_path, s) for s in valid]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_error_bands(
    df: pd.DataFrame,
    cell_type: str,
    category: str,
    outdir: str,
) -> str:
    """
    For a given cell type and object category, plot mean ± SEM across layers
    for each condition, using shaded bands. Returns the saved file path.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(LAYERS_ORDER), dtype=float)

    for cond in CONDITIONS:
        sub = df[
            (df["cell_type"] == cell_type)
            & (df["category"] == category)
            & (df["condition"] == cond)
        ]
        if sub.empty:
            continue

        grp = (
            sub.groupby("layer")["value"]
            .agg(["mean", "count", "std"])
            .reindex(LAYERS_ORDER)
        )
        mean = grp["mean"].to_numpy(dtype=float)
        sem = (grp["std"] / np.sqrt(grp["count"].replace(0, np.nan))).to_numpy(
            dtype=float
        )
        lo, hi = mean - sem, mean + sem
        color = COLORS.get(cond, "black")

        if SMOOTH_BANDS:
            xs, mean_s = _smooth_xy(x, mean, samples=SMOOTH_SAMPLES, k=SPLINE_K)
            _, lo_s = _smooth_xy(x, lo, samples=SMOOTH_SAMPLES, k=SPLINE_K)
            _, hi_s = _smooth_xy(x, hi, samples=SMOOTH_SAMPLES, k=SPLINE_K)
            ax.plot(xs, mean_s, color=color, linewidth=2.2, label=LEGEND_LABELS.get(cond, cond))
            ax.fill_between(xs, lo_s, hi_s, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, mean, "o", color=color, ms=4, alpha=0.95)
        else:
            ax.plot(x, mean, color=color, linewidth=2.2, marker="o", ms=4, label=LEGEND_LABELS.get(cond, cond))
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)

    # X axis
    ax.set_xticks(x)
    ax.set_xticklabels(LAYERS_ORDER, rotation=45, ha="right", fontweight="bold")

    # Y axis
    ax.set_ylabel(YLABELS.get(category, "Value"), fontweight="bold", fontsize=14)
    ymin, ymax = YLIMS.get(category, (0.0, None))
    ax.set_ylim(bottom=ymin, top=ymax)
    step = YTICK_STEP.get(category, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(step))

    # Spines and ticks
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(SPINE_W)
    ax.spines["bottom"].set_linewidth(SPINE_W)

    ax.tick_params(
        axis="both", which="both", bottom=True, top=False, left=True, right=False
    )
    ax.tick_params(
        axis="both", which="major", direction="out",
        length=10, width=TICK_W, color="black", pad=8,
    )
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # Legend inside the axes
    # ax.legend(frameon=False, loc="upper right", fontsize=11)

    # Layout
    fig.subplots_adjust(left=0.17, bottom=0.28, right=0.97, top=0.97)

    # Save
    os.makedirs(outdir, exist_ok=True)
    fname = f"{cell_type}_{category}_bands.png".replace(" ", "_")
    fpath = os.path.join(outdir, fname)
    fig.savefig(fpath, dpi=DPI, transparent=True)
    plt.close(fig)
    return fpath


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Make layer error-band plots.")
    parser.add_argument("--xlsx", default=DEFAULT_XLSX, help="Path to the workbook.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output folder.")
    args = parser.parse_args()

    df = load_all(args.xlsx)

    for ctype in CELL_TYPES:
        for obj in OBJECTS:
            plot_error_bands(df, ctype, obj, args.outdir)


if __name__ == "__main__":
    main()
