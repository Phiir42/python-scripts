"""
hyperspec_features.py
=====================
Feature engineering for the pipeline's **Peak Fits** sheets as written by
`lipid_analysis.hyperspec.process_hyperspectral_series`.

Input format (long table)
-------------------------
One row per (droplet, fitted-peak). Columns observed:
- 'Lipid ID', 'Category', 'Location', 'Cell Marker', 'LAMP2_Coloc',
- 'Peak' (1..7), 'Center_cm^-1', 'Amplitude', 'FitSuccess'

Goal
----
Return **one row per droplet** with CH-stretch amplitude features:
A2850, A2885, A2935, A2960, A3010, plus derived ratios:
R_pack = A2885/A2850, U = A3010/A2850, R_me = A2935/(A2850+A2885),
R_hi = (A2935+A2960)/(A2850+A2885).

How peaks are mapped
--------------------
We do *not* trust the 'Peak' index to be a semantic label. Instead we map by
nearest fitted center (cm^-1) to canonical CH-stretch targets:
  2850, 2885, 2935, 2960, 3010.
A fitted peak is accepted for a canonical target if its |Δcenter| <= 25 cm^-1.
If multiple fitted peaks tie, the nearest wins.

If no suitable peak is found for a target in a droplet, that target amplitude
is NaN for that droplet.

This module intentionally avoids any fingerprint-region logic.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

# Canonical target centers (cm^-1)
CANONICAL = {
    "A2850": 2850.0,
    "A2885": 2885.0,
    "A2935": 2935.0,
    "A2960": 2960.0,  # may appear near 2965–2968
    "A3010": 3010.0,  # may appear near 3015–3022
}

MAX_CENTER_DELTA = 25.0  # cm^-1


def _pivot_peak_fits_long(df_long: pd.DataFrame) -> pd.DataFrame:
    """Convert the Peak Fits long table into one row per droplet with Axxxx and Wxxxx columns."""
    # normalize column names (keep original case for outputs)
    cols = {c: c for c in df_long.columns}
    required = ["Lipid ID", "Center_cm^-1", "Amplitude"]
    for r in required:
        if r not in cols:
            raise ValueError(f"Peak Fits sheet missing required column: '{r}'")

    # detect width column name
    width_col = "Width_cm^-1" if "Width_cm^-1" in cols else ("Width" if "Width" in cols else None)

    # enforce numeric
    df = df_long.copy()
    df["Center_cm^-1"] = pd.to_numeric(df["Center_cm^-1"], errors="coerce")
    df["Amplitude"] = pd.to_numeric(df["Amplitude"], errors="coerce")
    if width_col is not None:
        df[width_col] = pd.to_numeric(df[width_col], errors="coerce")

    out_rows: List[Dict] = []
    for droplet, grp in df.groupby("Lipid ID", sort=False):
        row: Dict[str, float] = {"DropletID": droplet}
        centers = grp["Center_cm^-1"].to_numpy(dtype=float)
        amps    = grp["Amplitude"].to_numpy(dtype=float)
        widths  = grp[width_col].to_numpy(dtype=float) if width_col is not None else None

        for key, target in CANONICAL.items():
            # amplitude
            if centers.size == 0 or np.all(~np.isfinite(centers)):
                row[key] = np.nan
                row["W" + key[1:]] = np.nan  # e.g., W2850
                continue
            deltas = np.abs(centers - target)
            idx = int(np.nanargmin(deltas))
            if np.isfinite(deltas[idx]) and deltas[idx] <= MAX_CENTER_DELTA:
                row[key] = float(amps[idx]) if np.isfinite(amps[idx]) else np.nan
                if widths is not None and np.isfinite(widths[idx]):
                    row["W" + key[1:]] = float(widths[idx])
                else:
                    row["W" + key[1:]] = np.nan
            else:
                row[key] = np.nan
                row["W" + key[1:]] = np.nan

        out_rows.append(row)
    return pd.DataFrame(out_rows)


def compute_features_table(df_peak_fits: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts the **Peak Fits** long-format DataFrame and returns a feature table.
    If the input already looks like a pivoted table with A2850/A2885/... columns,
    it is passed through after ensuring required columns exist.
    """
    cols = set(df_peak_fits.columns.astype(str))
    looks_long = {"Lipid ID", "Center_cm^-1", "Amplitude"}.issubset(cols)
    if looks_long:
        wide = _pivot_peak_fits_long(df_peak_fits)
    else:
        # assume it's already pivoted like we want
        wide = df_peak_fits.copy()
        # ensure DropletID column exists for joins
        if "DropletID" not in wide.columns and "Lipid ID" in wide.columns:
            wide = wide.rename(columns={"Lipid ID": "DropletID"})

    # compute intensity proxies using widths: I_k = (A_k^2 / W_k^2)
    out = wide.copy()

    def _safe_sq_over_sq(a, w):
        a = float(a) if pd.notna(a) else np.nan
        w = float(w) if pd.notna(w) else np.nan
        if not np.isfinite(a) or not np.isfinite(w) or w <= 0:
            return np.nan
        return (a * a) / (w * w)

    # Pull amplitudes and widths (may be NaN if not found)
    A2850 = out.get("A2850"); W2850 = out.get("W2850")
    A2885 = out.get("A2885"); W2885 = out.get("W2885")
    A2935 = out.get("A2935"); W2935 = out.get("W2935")
    # Prefer 2960; fall back to 2910 if that's what the export used
    A2960 = out.get("A2960", out.get("A2910")); W2960 = out.get("W2960", out.get("W2910"))
    A3010 = out.get("A3010"); W3010 = out.get("W3010")  # broad band (x8)

    # Intensities
    out["I2850"] = [_safe_sq_over_sq(a, w) for a, w in zip(A2850, W2850)]
    out["I2885"] = [_safe_sq_over_sq(a, w) for a, w in zip(A2885, W2885)]
    out["I2935"] = [_safe_sq_over_sq(a, w) for a, w in zip(A2935, W2935)]
    out["I2960"] = [_safe_sq_over_sq(a, w) for a, w in zip(A2960, W2960)]
    out["I3010"] = [_safe_sq_over_sq(a, w) for a, w in zip(A3010, W3010)]

    # Derived intensity ratios (classifier features)
    I2850 = out["I2850"].astype(float)
    I2885 = out["I2885"].astype(float)
    I2935 = out["I2935"].astype(float)
    I2960 = out["I2960"].astype(float)
    I3010 = out["I3010"].astype(float)

    denom1I  = I2850.replace(0, np.nan)
    denom12I = (I2850 + I2885).replace(0, np.nan)
    denomBGI = (I2885 + I2935 + I2960).replace(0, np.nan)

    # Swap all to intensity-based definitions
    out["R_pack"] = (I2885 / denom1I).astype(float)  # packing/order
    out["U"]      = (I3010 / denom1I).astype(float)  # unsaturation
    out["R_hi"]   = ((I2935 + I2960) / denom12I).astype(float)
    out["Rbg_I"]  = (I2850 / denomBGI).astype(float)  # myelin guard (Option B)

    # optional fields that classifier expects; fill with NaN/False if absent
    if "SNR_fit" not in out.columns:
        out["SNR_fit"] = np.nan
    out["det_3010"] = out["I3010"].fillna(0) > 0

    return out
