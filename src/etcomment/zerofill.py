"""Detect and simulate zero-filled observed ET.

A common way to create exact-zero 'observations' is to outer-join a model's
date range onto a flux record and fill the gaps with zero (``fillna(0)``), or
to clip -9999 fill values at zero.  The functions here (1) find runs of exact
zeros, (2) reproduce that join to create a zero-filled copy of a clean series,
and (3) give the expected bias shift, f x E, for any predictor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def zero_runs(values: pd.Series, tol: float = 1e-9, min_len: int = 1) -> pd.DataFrame:
    """Runs of consecutive |value| <= tol in a date-indexed series."""
    v = values.dropna()
    is0 = v.abs() <= tol
    rows = []
    start = prev = None
    for d, z in is0.items():
        contiguous = prev is not None and (d - prev) == pd.Timedelta(days=1)
        if z and (start is None or not contiguous):
            if start is not None:
                rows.append((start, prev))
            start = d
        elif not z and start is not None:
            rows.append((start, prev))
            start = None
        prev = d
    if start is not None:
        rows.append((start, prev))
    df = pd.DataFrame(rows, columns=["start", "end"])
    if df.empty:
        return df.assign(n_days=pd.Series(dtype=int))
    df["n_days"] = (df["end"] - df["start"]).dt.days + 1
    return df[df["n_days"] >= min_len].reset_index(drop=True)


def outer_join_zero_fill(obs: pd.Series, model_dates: pd.DatetimeIndex) -> pd.Series:
    """Reproduce 'reindex to the model's dates, then fillna(0)'."""
    return obs.reindex(model_dates).fillna(0.0)


def expected_bias_shift(missing: pd.Series, proxy_true: pd.Series) -> dict:
    """f, E and f x E for a record whose missing days would be set to zero.

    ``missing`` is a boolean series over the evaluation days; ``proxy_true`` is
    the best available estimate of true ET on those days (for the benchmark,
    the gap-filled value EToF x ETo).
    """
    m = missing.astype(bool)
    n = int(m.size)
    f = float(m.mean()) if n else np.nan
    e = float(proxy_true[m].mean()) if m.any() else 0.0
    return {"n_days": n, "n_missing": int(m.sum()), "f": f, "E": e, "delta_mbe": f * e}
