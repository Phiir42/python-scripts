#!/usr/bin/env python3
"""
collect_myelin_totalfits_normalized.py

Walk the 2025 data tree, extract normalized total fits (y_fit) from each
'Hyperspectral_Myelin_AverageFits.xlsx' → 'Myelin_Average_Fits' sheet, and
write them to one workbook:

  Sheet 'Myelin_TotalFits':
    Row 1: ["x_cm1", x1, x2, ..., x32]
    Row 2+: [label, y1, y2, ..., y32]  (normalized y_fit, no rescaling)

  Sheet 'MeanSpectrum':
    Row 1: ["x_cm1", x1, x2, ..., x32]
    Row 2: ["mean",  mean_y1..mean_y32]  (mean across all included rows)

  Sheet 'Myelin_Classification':
    One row per frame with derived features (U, R_pack, R_hi, Rbg_I, det_3010)
    and classification outputs (class_label, class_score, rules_fired).

Requirements:
  pip install pandas openpyxl XlsxWriter
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd


# ── USER SETTINGS ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(r"D:\OneDrive - Stanford\Research Documents\AD Project\2025")
OUTPUT_FILE = ROOT_DIR / "Myelin_TotalFits_AllSessions.xlsx"
SHEET_FITS = "Myelin_TotalFits"
SHEET_MEAN = "MeanSpectrum"
SHEET_CLASS = "Myelin_Classification"

# If classify_rules.py is NOT importable via PYTHONPATH, optionally point to it:
CLASSIFY_RULES_PATH = Path(r"D:\OneDrive - Stanford\Research Documents\Python Scripts\lipid_analysis\classify_rules.py")

# Provided x-axis (cm^-1). Length must match y_fit length (expected 32).
X_CM1 = [
    2785.073459, 2792.871332, 2800.678952, 2808.496338, 2816.323508, 2824.16048,
    2832.007273, 2839.863905, 2847.730395, 2855.606762, 2863.493023, 2871.389198,
    2879.295305, 2887.211363, 2895.137391, 2903.073407, 2911.019432, 2918.975482,
    2926.941579, 2934.91774, 2942.903984, 2950.900331, 2958.906801, 2966.923411,
    2974.950182, 2982.987133, 2991.034282, 2999.091651, 3007.159257, 3015.237122,
    3023.325263, 3031.423701
]
# ──────────────────────────────────────────────────────────────────────────────


# ── Classifier hookup (prefer your module, fallback to a local copy) ─────────
def _import_classifier():
    import importlib.util, sys
    if CLASSIFY_RULES_PATH and CLASSIFY_RULES_PATH.exists():
        mod_name = "classify_rules_ext"  # avoid name clash with any installed module
        spec = importlib.util.spec_from_file_location(mod_name, str(CLASSIFY_RULES_PATH))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod  # register so __spec__ is set and reload won’t target 'classify_rules'
            spec.loader.exec_module(mod)  # type: ignore[arg-type]
            return mod.classify_table, mod.load_rules
    # fallback: normal import if on PYTHONPATH
    from classify_rules import classify_table, load_rules  # type: ignore
    return classify_table, load_rules


# ── Utilities ────────────────────────────────────────────────────────────────
def _parse_yfit(cell) -> np.ndarray:
    """Parse numpy-like y_fit string "[.. ..]" into float array."""
    if isinstance(cell, (list, tuple, np.ndarray)):
        try:
            return np.array(cell, dtype=float).ravel()
        except Exception:
            return np.array([], dtype=float)
    if not isinstance(cell, str):
        return np.array([], dtype=float)
    s = cell.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    parts = re.split(r"\s+", s.strip())
    vals: List[float] = []
    for p in parts:
        if not p:
            continue
        try:
            vals.append(float(p))
        except ValueError:
            try:
                vals.append(float(p.replace(',', '')))
            except Exception:
                return np.array([], dtype=float)
    return np.array(vals, dtype=float)


def _label_for_row(row: pd.Series, default: str) -> str:
    """Prefer 'Folder' then 'Series' for labeling."""
    folder = row.get("Folder")
    if isinstance(folder, str) and folder.strip():
        return folder.strip()
    series = row.get("Series")
    if isinstance(series, str) and series.strip():
        return series.strip()
    return default


def _height_proxy(A: Optional[float], w: Optional[float]) -> float:
    """
    Peak height proxy in the |sum|^2 model ~ (A^2)/(w^2) when w>0,
    else fallback to A^2; negatives are naturally handled by squaring.
    """
    try:
        Af = float(A)
    except Exception:
        Af = 0.0
    try:
        wf = float(w)
    except Exception:
        wf = 0.0
    if wf and abs(wf) > 0:
        return (Af * Af) / (wf * wf)
    return Af * Af


def _safe_div(num: float, den: float) -> float:
    EPS = 1e-12
    d = den if abs(den) > EPS else EPS
    return num / d


# ── Data collection ──────────────────────────────────────────────────────────
def collect_myelin_fits_and_features(root: Path, x_vec: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Returns:
      df_rows:     'Myelin_TotalFits' layout (row 1 = x; rows 2..N = y_fit)
      df_features: per-row features for classification
      warnings:    list of warn strings
    """
    warnings: List[str] = []
    x_arr = np.array(x_vec, dtype=float).ravel()
    nx = x_arr.size

    # y_fit table: first row is x_cm1
    out_rows: List[List[Optional[float]]] = [["x_cm1"] + list(x_arr)]

    # feature rows for classifier
    feat_records: List[Dict[str, Any]] = []

    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname != "Hyperspectral_Myelin_AverageFits.xlsx":
                continue
            fpath = Path(dirpath) / fname
            rel = str(fpath.relative_to(root))
            try:
                df = pd.read_excel(fpath, sheet_name="Myelin_Average_Fits")
            except Exception as e:
                warnings.append(f"[WARN] Could not open '{rel}': {e}")
                continue

            for idx, row in df.iterrows():
                # 1) y_fit collection (normalized)
                y_norm = _parse_yfit(row.get("y_fit"))
                if y_norm.size == 0:
                    warnings.append(f"[WARN] Empty/invalid y_fit in '{rel}' row {idx}")
                elif y_norm.size != nx:
                    warnings.append(f"[WARN] Length mismatch in '{rel}' row {idx}: len(y_fit)={y_norm.size} vs len(x)={nx}; skipped y_fit.")
                else:
                    label = _label_for_row(row, default=f"{rel}__row{idx}")
                    out_rows.append([label] + list(y_norm.astype(float)))

                # 2) Feature derivation from fitted peaks (A1..A8; w1..w8 if present)
                # Indices: 2850→1, 2885→2, 2935→3, 2960→7, 3010→8
                A1, A2, A3, A7, A8 = row.get("A1"), row.get("A2"), row.get("A3"), row.get("A7"), row.get("A8")
                w1, w2, w3, w7, w8 = row.get("w1"), row.get("w2"), row.get("w3"), row.get("w7"), row.get("w8")

                I2850 = _height_proxy(A1, w1)
                I2885 = _height_proxy(A2, w2)
                I2935 = _height_proxy(A3, w3)
                I2960 = _height_proxy(A7, w7)
                I3010 = _height_proxy(A8, w8)

                # Ratios (epsilon-safe)
                U      = _safe_div(I3010, I2850)
                R_pack = _safe_div(I2885, I2850)
                R_hi   = _safe_div(I2935 + I2960, I2850 + I2885)
                denom_bg = (I2885 + I2935 + I2960)
                Rbg_I  = _safe_div(I2850, denom_bg)

                # Simple detection flag for 3010
                det_3010 = 1 if I3010 > 1e-6 else 0

                feat_records.append({
                    "label": _label_for_row(row, default=f"{rel}__row{idx}"),
                    "file": rel,
                    "row_index": idx,
                    "U": U,
                    "R_pack": R_pack,
                    "R_hi": R_hi,
                    "Rbg_I": Rbg_I,
                    "det_3010": det_3010,
                    # store raw proxies (can help debugging)
                    "I2850": I2850, "I2885": I2885, "I2935": I2935, "I2960": I2960, "I3010": I3010,
                })

    # Build y_fit table
    max_len = max(len(r) for r in out_rows) if out_rows else 0
    padded = [r + [np.nan] * (max_len - len(r)) for r in out_rows]
    df_rows = pd.DataFrame(padded)

    # Feature table
    df_features = pd.DataFrame(feat_records)
    return df_rows, df_features, warnings


def write_output_with_classification(
    df_rows: pd.DataFrame,
    df_features: pd.DataFrame,
    out_path: Path
) -> None:
    """Write three sheets: fits, mean, classification."""
    # Compute mean spectrum (rows 2..N, cols starting at 2)
    if len(df_rows) > 1:
        numeric = df_rows.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce")
        mean_vals = numeric.mean(axis=0, skipna=True).to_list()
    else:
        mean_vals = []

    # Hook classifier
    classify_table, load_rules = _import_classifier()
    rules = load_rules(None)
    classified = classify_table(
        df_features[["U", "R_pack", "R_hi", "Rbg_I", "det_3010"]].copy(),
        rules=rules
    )
    # Stitch label/file/indices back on
    classified = pd.concat(
        [df_features[["label", "file", "row_index"]].reset_index(drop=True), classified.reset_index(drop=True)],
        axis=1
    )

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as xr:
        # Sheet 1: all fits (x row + rows of y_fit)
        df_rows.to_excel(xr, sheet_name=SHEET_FITS, header=False, index=False)

        # Sheet 2: mean spectrum (x row, then mean)
        mean_sheet = []
        mean_sheet.append(["x_cm1"] + list(X_CM1))
        if mean_vals:
            mean_sheet.append(["mean"] + mean_vals)
        pd.DataFrame(mean_sheet).to_excel(xr, sheet_name=SHEET_MEAN, header=False, index=False)

        # Sheet 3: classification table
        classified.to_excel(xr, sheet_name=SHEET_CLASS, index=False)


def main() -> None:
    print(f"[INFO] Scanning for Hyperspectral_Myelin_AverageFits.xlsx under:\n  {ROOT_DIR}")
    df_rows, df_features, warns = collect_myelin_fits_and_features(ROOT_DIR, X_CM1)

    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_output_with_classification(df_rows, df_features, out_path)

    print(f"[OK] Wrote total fits + mean + classification to: {out_path}")
    if warns:
        print("\n".join(warns))


if __name__ == "__main__":
    main()
