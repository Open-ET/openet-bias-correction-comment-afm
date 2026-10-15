"""Readers for the Volk et al. (2023) benchmark and the OpenET Phase II data.

Benchmark flux data: Zenodo 10.5281/zenodo.7636781, ``flux_ET_dataset.zip``
(md5 99f6668ca439e9bae48c4e7e08b5405d), CC BY 4.0.  Layout inside the zip:

    flux_ET_dataset/README.md
    flux_ET_dataset/station_metadata.xlsx
    flux_ET_dataset/variable_explanation.xlsx
    flux_ET_dataset/daily_data_files/<SITE>_daily_data.csv
    flux_ET_dataset/monthly_data_files/<SITE>_monthly_data.csv
    flux_ET_dataset/graphical_files/<SITE>_plots.html

OpenET model ET at the stations: Zenodo 10.5281/zenodo.10119477,
``OpenET_PhaseII_model_ET_dataset.zip`` (md5 26b2af882c21dc56c634ac9f77a2ba6e),
tab-delimited daily (overpass dates) and monthly files.

Column names in the flux-data-qaqc outputs are resolved case-insensitively
from candidate lists, so small naming differences between versions do not
break the analysis; unresolved columns raise a clear error.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ZENODO = {
    "flux": {"record": "7636781", "file": "flux_ET_dataset.zip",
             "md5": "99f6668ca439e9bae48c4e7e08b5405d"},
    "openet": {"record": "10119477", "file": "OpenET_PhaseII_model_ET_dataset.zip",
               "md5": "26b2af882c21dc56c634ac9f77a2ba6e"},
}

CANDIDATES = {
    "date": ["date", "Date", "DATE", "datetime"],
    "et": ["ET", "et", "ET_mm"],
    "et_corr": ["ET_corr", "et_corr", "ET_closed"],
    "et_fill": ["ET_fill", "et_fill", "ET_fill_val"],
    "et_gap": ["ET_gap", "et_gap"],
    "le": ["LE", "le"],
    "h": ["H", "h"],
    "rn": ["Rn", "RN", "NETRAD", "rn"],
    "g": ["G", "g"],
    "le_corr": ["LE_corr", "le_corr"],
    "h_corr": ["H_corr", "h_corr"],
    "ebr": ["ebr", "EBR"],
    "eto": ["gridMET_ETo", "ETo", "eto"],
    "t_avg": ["t_avg", "T_avg", "TA", "t_mean", "Tavg"],
    "vpd": ["vpd", "VPD"],
    "ws": ["ws", "WS", "wind"],
}


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def extract(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    return dest


def resolve(df: pd.DataFrame, key: str, required: bool = True) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in CANDIDATES[key]:
        if cand in df.columns:
            return cand
        if cand.lower() in cols:
            return cols[cand.lower()]
    if required:
        raise KeyError(f"none of {CANDIDATES[key]} found; columns are {list(df.columns)[:40]}...")
    return None


def find_root(base: Path) -> Path:
    """Directory that holds daily_data_files/ (handles nested zip layouts)."""
    for p in [base, *base.rglob("*")]:
        if p.is_dir() and (p / "daily_data_files").is_dir():
            return p
    raise FileNotFoundError(f"daily_data_files/ not found under {base}")


def load_daily(root: Path, site: str) -> pd.DataFrame:
    f = root / "daily_data_files" / f"{site}_daily_data.csv"
    df = pd.read_csv(f)
    dcol = resolve(df, "date", required=False)
    if dcol is None:
        dcol = df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol])
    return df.set_index(dcol).sort_index()


def load_monthly(root: Path, site: str) -> pd.DataFrame:
    f = root / "monthly_data_files" / f"{site}_monthly_data.csv"
    df = pd.read_csv(f)
    dcol = resolve(df, "date", required=False) or df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol])
    return df.set_index(dcol).sort_index()


def load_openet_phase2(folder: Path, which: str = "daily") -> pd.DataFrame:
    fn = folder / ("daily_data.dat" if which == "daily" else "monthly_data.dat")
    lines = fn.read_text().splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Site ID"))
    df = pd.read_csv(fn, sep="\t", skiprows=hdr)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df.rename(columns={"Site ID": "site", "DATE": "date"})


def standardize(daily: pd.DataFrame) -> pd.DataFrame:
    """Common column names (mm d-1 for ET, W m-2 for fluxes) for the analysis."""
    out = pd.DataFrame(index=daily.index)
    for key in ("et", "et_corr"):
        out[key] = pd.to_numeric(daily[resolve(daily, key)], errors="coerce")
    # energy terms are absent at the weighing-lysimeter stations (LYS_*)
    for key in ("le", "h", "rn", "g"):
        c = resolve(daily, key, required=False)
        out[key] = pd.to_numeric(daily[c], errors="coerce") if c is not None else np.nan
    for key in ("et_fill", "et_gap", "le_corr", "h_corr", "ebr", "eto", "t_avg", "vpd", "ws"):
        c = resolve(daily, key, required=False)
        if c is not None:
            out[key] = daily[c]
    if "et_gap" in out:
        out["et_gap"] = out["et_gap"].astype(str).str.lower().isin(["true", "1", "1.0"])
    else:
        out["et_gap"] = False
    out["valid_measured"] = out["et_corr"].notna() & ~out["et_gap"]
    return out


def period_month_days(start: str, end: str) -> pd.Series:
    """Days of each calendar month (index 1-12) in a study period given as 'YYYY-MM' bounds.

    Used to weight a monthly climatology to the months of a study period, e.g.
    GO26's 2020-01 to 2025-04 contains six Januaries but five Decembers.
    """
    days = pd.date_range(f"{start}-01", pd.Period(end, "M").end_time.normalize(), freq="D")
    return pd.Series(days.month).value_counts().reindex(range(1, 13), fill_value=0)


def month_weighted_mean(monthly_clim: pd.Series, weights: pd.Series) -> float:
    """Mean of a 12-value monthly climatology weighted by ``weights`` (NaN if a weighted month is missing)."""
    c = monthly_clim.reindex(range(1, 13))
    w = weights.reindex(range(1, 13)).fillna(0)
    if c[w > 0].isna().any():
        return np.nan
    return float((c.fillna(0) * w).sum() / w.sum())


def complete_months(daily: pd.DataFrame, max_gap_days: int = 5) -> pd.DataFrame:
    """Monthly sums of closed ET in months usable for evaluation (the V24 rule).

    A month qualifies when closed ET (measured or gap-filled) exists on every
    calendar day and at most ``max_gap_days`` of them are gap-filled.  Expects
    the output of :func:`standardize`.
    """
    g = daily.resample("MS").agg(et_corr=("et_corr", "sum"), n_gap=("et_gap", "sum"),
                                 n_et=("et_corr", "count"))
    g["n_days"] = g.index.days_in_month
    return g[(g["n_gap"] <= max_gap_days) & (g["n_et"] == g["n_days"])]
