"""Tests for the digitization sensitivity of scripts/04 on synthetic series; no test reads data/ or results/."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "04_digitize_afm_figures.py"


@pytest.fixture(scope="module")
def s04():
    spec = importlib.util.spec_from_file_location("s04", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _series():
    """Six days: two exact zeros (0.0 and 0.02 mm/d, a zero only with the tolerance doubled), three observations,
    one day without an observation (a hidden dot) and one without the lines."""
    days = pd.date_range("2024-06-01", periods=7, freq="D")
    return pd.DataFrame({"obs": [0.0, 0.02, 2.0, 3.0, 1.0, np.nan, 1.5],
                         "openet": [2.0, 2.0, 2.5, 3.0, 2.0, 2.0, np.nan],
                         "xgb": [0.1, 0.1, 2.0, 3.5, 1.0, 1.0, np.nan]}, index=days)


def _pct(o, p, x):
    rmse = lambda a: np.sqrt(np.mean((np.asarray(a) - np.asarray(o)) ** 2))  # noqa: E731
    return 100 * (1 - rmse(x) / rmse(p))


def test_improvement_with_and_without_exact_zeros(s04):
    df = _series()
    r = s04.improvement(df, df["obs"], 0.015)
    assert (r["n_days"], r["n_nonzero_days"]) == (5, 4)  # the day without lines is left out
    assert r["improvement_pct_all"] == pytest.approx(_pct([0, 0.02, 2, 3, 1], [2, 2, 2.5, 3, 2], [0.1, 0.1, 2, 3.5, 1]))
    assert r["improvement_pct_nonzero"] == pytest.approx(_pct([0.02, 2, 3, 1], [2, 2.5, 3, 2], [0.1, 2, 3.5, 1]))
    assert s04.improvement(df, df["obs"], 0.03)["n_nonzero_days"] == 3  # 0.02 is a zero with the tolerance doubled


def test_sensitivity_adds_hidden_dots_and_doubles_the_zero_tolerance(s04):
    df = _series()
    with_hidden = df["obs"].copy()
    with_hidden.iloc[5] = 2.0  # a dot read under the lines
    s = s04.sensitivity(df, with_hidden, [("2024-06-06", 2.0, 0.4)])
    c = s["cases"]
    assert list(c) == ["digitized", "digitized_zero_tol_doubled", "hidden_added", "hidden_added_zero_tol_doubled"]
    assert c["hidden_added"]["n_days"] == 6 and c["hidden_added_zero_tol_doubled"]["n_nonzero_days"] == 4
    assert c["digitized"] == s04.improvement(df, df["obs"], s04.ZERO_TOL)
    assert c["hidden_added_zero_tol_doubled"]["improvement_pct_nonzero"] == pytest.approx(
        _pct([2, 3, 1, 2], [2.5, 3, 2, 2], [2, 3.5, 1, 1]))
    pcts = [v["improvement_pct_nonzero"] for v in c.values()]
    assert s["range_pct_nonzero"] == [min(pcts), max(pcts)]
    assert (s["zero_tol"], s["hidden_min_cover"]) == (s04.ZERO_TOL, s04.HIDDEN_MIN_COVER)
    assert s["hidden_added_days"] == [("2024-06-06", 2.0, 0.4)]
