#!/usr/bin/env python3
"""
prep_prism_spyder.py — run inside Spyder (or any IDE)

Creates / replaces the following worksheets:

    1. Prism Prep                     (raw, zeros kept)
    2. Prism Averages                 (donor‑averaged, zeros kept)
    3. Prism Prep (no zeros)          (raw, zeros removed)
    4. Prism Averages (no zeros)      (donor‑averaged, zeros removed)
"""

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
INPUT_FILE  = r"C:\Users\clchr\OneDrive - Stanford\Research Documents\AD Project\2025\AD Lipid Statistics.xlsx"   # ← edit
OUTPUT_FILE = None        # leave None to get “…_prism.xlsx” in same folder
# ──────────────────────────────────────────────────────────────────────────────


import pathlib
import pandas as pd
from openpyxl import load_workbook


# ── CONSTANTS ────────────────────────────────────────────────────────────────
CELL_TYPES = ["Microglia", "Astrocytes", "Neurons"]
METRICS = [
    ("pure_lipid_percentage",       "Lipid Areas"),
    ("lipofuscin_percentage",       "Lipofuscin Areas"),
    ("lipid_lipofuscin_percentage", "Lipidated Lipofuscin Areas"),
]
CONDITIONS = ["Control", "AD33", "AD44"]
# ──────────────────────────────────────────────────────────────────────────────


# ╭────────────────────────── Helper utilities ───────────────────────────────╮
def _cond_donor_stub(file_name: str) -> str:
    """Return the 'condition‑donor' prefix (first two dash‑separated parts)."""
    parts = file_name.split("-", 2)
    return "-".join(parts[:2]) if len(parts) >= 2 else file_name
# ╰────────────────────────────────────────────────────────────────────────────╯


# ╭──────────────────────────── Data builders ────────────────────────────────╮
def build_prism_dataframe(xlsx: pathlib.Path, *, remove_zeros: bool = False) -> pd.DataFrame:
    """
    Stack cell‑level Series in metric→cell→condition order.
    If *remove_zeros* is True, 0 values are dropped (not just turned into NaN).
    """
    cols, max_len = [], 0

    for metric_key, _ in METRICS:
        for cell in CELL_TYPES:
            for cond in CONDITIONS:
                sheet = f"{cond} {cell}"
                s = pd.read_excel(xlsx, sheet_name=sheet, usecols=[metric_key])[metric_key]
                if remove_zeros:
                    s = s[s != 0].reset_index(drop=True)
                cols.append(s)
                max_len = max(max_len, len(s))

    # pad each column to the same length
    padded = [s.reindex(range(max_len)).reset_index(drop=True) for s in cols]
    return pd.concat(padded, axis=1)


def build_prism_average_dataframe(xlsx: pathlib.Path, *, remove_zeros: bool = False) -> pd.DataFrame:
    """
    Collapse each donor (condition‑donor stub) to a mean value per column.
    If *remove_zeros* is True, 0 values are discarded before the mean is taken.
    """
    series_list = []
    donors = set()

    for metric_key, _ in METRICS:
        for cell in CELL_TYPES:
            for cond in CONDITIONS:
                sheet = f"{cond} {cell}"
                df = pd.read_excel(
                    xlsx, sheet_name=sheet, usecols=["file_name", metric_key]
                ).copy()

                if remove_zeros:
                    df.loc[df[metric_key] == 0, metric_key] = pd.NA

                df["cond_donor"] = df["file_name"].map(_cond_donor_stub)
                s = df.groupby("cond_donor", dropna=False)[metric_key].mean()
                series_list.append(s)
                donors.update(s.index)

    donor_list = sorted(donors)
    padded = [s.reindex(donor_list) for s in series_list]
    wide = pd.concat(padded, axis=1)
    wide.insert(0, "cond_donor", donor_list)
    return wide
# ╰────────────────────────────────────────────────────────────────────────────╯


# ╭──────────────────────────── Excel helpers ────────────────────────────────╮
def _write_headers(ws, start_col: int):
    """Write the two‑row merged header block beginning at *start_col*."""
    col = start_col
    for _metric_key, metric_title in METRICS:
        for cell in CELL_TYPES:
            start, end = col, col + len(CONDITIONS) - 1
            ws.merge_cells(start_row=1, start_column=start,
                           end_row=1,   end_column=end)
            ws.cell(row=1, column=start).value = f"{cell} {metric_title}"
            col = end + 1

    col = start_col
    for _ in range(len(METRICS) * len(CELL_TYPES)):
        for cond in CONDITIONS:
            ws.cell(row=2, column=col, value=cond)
            col += 1


def _write_df(ws, df: pd.DataFrame, *, id_col: bool = False):
    """
    Dump DataFrame *df* to worksheet *ws* starting on row 3.
    If *id_col* is True, df.iloc[:,0] is written into col 1 and treated
    as an ID; otherwise df is written from col 1 onward.
    """
    offset = 1 if id_col else 0
    rows_iter = df.itertuples(index=False)
    for r, row in enumerate(rows_iter, start=3):
        if id_col:
            ws.cell(row=r, column=1, value=row[0])
        for c, val in enumerate(row[offset:], start=1+offset):
            ws.cell(row=r, column=c, value=val)
# ╰────────────────────────────────────────────────────────────────────────────╯


# ╭──────────────────────────── Main writer ──────────────────────────────────╮
def add_prism_sheets(infile: pathlib.Path, outfile: pathlib.Path) -> None:
    """Create/replace the four Prism sheets inside *outfile*."""
    wb = load_workbook(infile)

    # compute all four DataFrames only once
    df_raw          = build_prism_dataframe(infile, remove_zeros=False)
    df_raw_nz       = build_prism_dataframe(infile, remove_zeros=True)
    df_avg          = build_prism_average_dataframe(infile, remove_zeros=False)
    df_avg_nz       = build_prism_average_dataframe(infile, remove_zeros=True)

    sheet_specs = [
        ("Prism Prep",                    df_raw,     False),
        ("Prism Averages",                df_avg,     True ),
        ("Prism Prep (no zeros)",         df_raw_nz,  False),
        ("Prism Averages (no zeros)",     df_avg_nz,  True ),
    ]

    for name, df, has_id in sheet_specs:
        if name in wb.sheetnames:
            del wb[name]
        ws = wb.create_sheet(name)

        # First column header for averaged sheets
        if has_id:
            ws.merge_cells(start_row=1, start_column=1, end_row=2, end_column=1)
            ws.cell(row=1, column=1, value="Donor (cond‑stub)")

        _write_headers(ws, start_col=2 if has_id else 1)
        _write_df(ws, df, id_col=has_id)

    wb.save(outfile)
# ╰────────────────────────────────────────────────────────────────────────────╯


def main() -> None:
    in_path = pathlib.Path(INPUT_FILE).expanduser().resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path = (
        pathlib.Path(OUTPUT_FILE).expanduser().resolve()
        if OUTPUT_FILE
        else in_path.with_name(in_path.stem + "_prism.xlsx")
    )

    add_prism_sheets(in_path, out_path)
    print(f"✅  All four Prism sheets added.  Saved to: {out_path}")


if __name__ == "__main__":
    main()
