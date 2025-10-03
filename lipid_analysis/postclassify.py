"""
postclassify.py
---------------
Post-processing utilities to classify objects in Hyperspectral_Results_*.xlsx files
using CH-stretch rules (no fingerprint required). Integrates with the pipeline by
scanning a data directory and updating each results workbook with a new sheet
"Classification". Also writes per-file CSVs and a consolidated summary CSV.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

import pandas as pd

from .classify_rules import classify_table, load_rules
from .hyperspec_features import compute_features_table


def classify_hyperspectral_dir(
    directory: str,
    rules_json: Optional[str] = None,
    write_back: bool = True,
    consolidate: bool = True,
) -> str:
    """
    Scan `directory` for Hyperspectral_Results_*.xlsx, read the 'Peak Fits' sheet,
    compute features and apply rule-based classification. Writes a 'Classification'
    sheet to each workbook and saves per-file and consolidated CSV outputs.

    Returns the path to the consolidated CSV (if requested).
    """
    pattern = os.path.join(directory, "Hyperspectral_Results_*.xlsx")
    files = sorted(glob.glob(pattern))
    rules = load_rules(rules_json)

    consolidated_rows: List[pd.DataFrame] = []

    for f in files:
        try:
            xls = pd.ExcelFile(f, engine="openpyxl")
            sheet = (
                "Peak Fits"
                if "Peak Fits" in xls.sheet_names
                else xls.sheet_names[min(2, len(xls.sheet_names) - 1)]
            )
            df = pd.read_excel(xls, sheet_name=sheet)
        except Exception as e:
            print(f"[postclassify] Skipping {f}: failed to read sheet ({e})")
            continue

        feats = compute_features_table(df)
        out = classify_table(feats, rules)
        # Persist per-file CSV
        csv_out = os.path.splitext(f)[0] + "_classified.csv"
        out.to_csv(csv_out, index=False)

        if write_back:
            try:
                with pd.ExcelWriter(
                    f, mode="a", engine="openpyxl", if_sheet_exists="replace"
                ) as writer:
                    out.to_excel(writer, sheet_name="Classification", index=False)
            except TypeError:
                from openpyxl import load_workbook

                wb = load_workbook(f)
                if "Classification" in wb.sheetnames:
                    ws = wb["Classification"]
                    wb.remove(ws)
                    wb.save(f)
                with pd.ExcelWriter(f, mode="a", engine="openpyxl") as writer:
                    out.to_excel(writer, sheet_name="Classification", index=False)

        out2 = out.copy()
        out2.insert(0, "source_file", os.path.basename(f))
        out2.insert(1, "sheet_used", sheet)
        consolidated_rows.append(out2)

    consolidated_path = ""
    if consolidate and consolidated_rows:
        big = pd.concat(consolidated_rows, axis=0, ignore_index=True)
        consolidated_path = os.path.join(
            directory, "Hyperspectral_Classification_Summary.csv"
        )
        big.to_csv(consolidated_path, index=False)
        print(
            f"[postclassify] Wrote consolidated summary: {consolidated_path} ({len(big)} rows)"
        )
    else:
        print("[postclassify] No hyperspectral results found to classify.")

    return consolidated_path
