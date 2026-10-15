"""Write a small synthetic stand-in for flux_ET_dataset/ (smoke tests only).

Column names follow flux-data-qaqc daily outputs (date, ET, ET_corr, ET_gap,
ET_fill, LE, H, Rn, G, LE_corr, H_corr, t_avg, vpd, ws).  Values are
synthetic and must never be used for analysis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write(root: Path, sites: list[str], start="2016-01-01", end="2020-12-31", seed=0) -> Path:
    rng = np.random.default_rng(seed)
    base = root / "flux_ET_dataset"
    (base / "daily_data_files").mkdir(parents=True, exist_ok=True)
    (base / "monthly_data_files").mkdir(parents=True, exist_ok=True)
    (base / "graphical_files").mkdir(parents=True, exist_ok=True)
    idx = pd.date_range(start, end, freq="D")
    doy = idx.dayofyear.to_numpy()
    for s in sites:
        season = np.clip(np.sin((doy - 80) / 365.25 * 2 * np.pi), 0, None)
        rn = 40 + 160 * season + rng.normal(0, 15, idx.size)
        g = 0.05 * rn
        le = (0.2 + 0.5 * season) * (rn - g) * 0.8 + rng.normal(0, 8, idx.size)
        h = (rn - g) * 0.8 - le
        f = (rn - g) / (le + h)
        le_corr = le * np.clip(f, 0.8, 1.4)
        et = le / 28.36
        et_corr = le_corr / 28.36
        gap = rng.random(idx.size) < 0.08
        et_fill = et_corr * (1 + rng.normal(0, 0.1, idx.size))
        et_corr_out = np.where(gap, et_fill, et_corr)
        df = pd.DataFrame({"date": idx, "ET": et, "ET_corr": et_corr_out, "ET_gap": gap,
                           "ET_fill": et_fill, "LE": le, "H": h, "Rn": rn, "G": g,
                           "LE_corr": le_corr, "H_corr": h * np.clip(f, 0.8, 1.4),
                           "t_avg": 12 + 12 * season, "vpd": 0.5 + 2 * season, "ws": 3 + rng.random(idx.size)})
        df.to_csv(base / "daily_data_files" / f"{s}_daily_data.csv", index=False)
        m = df.set_index("date")[["ET", "ET_corr"]].resample("MS").sum()
        m.to_csv(base / "monthly_data_files" / f"{s}_monthly_data.csv")
        (base / "graphical_files" / f"{s}_plots.html").write_text("<html><body>synthetic</body></html>")
    return base
