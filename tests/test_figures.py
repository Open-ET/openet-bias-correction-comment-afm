"""Tests for the figure script (scripts/08) on synthetic tables; no test reads config/, data/ or results/."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "08_make_figures.py"


@pytest.fixture(scope="module")
def figs():
    spec = importlib.util.spec_from_file_location("figs08", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_table1_agreement_counts_and_tolerance_edge(figs):
    zf, va, gf, cl = figs.TABLE1_SERIES
    r = pd.DataFrame({figs.TABLE1_REF: [1.00, 1.00, 0.60, 0.77, 1.50],
                      zf: [1.00, 1.04, 0.72, 0.98, np.nan],  # differences 0, +0.04 (edge, inside), +0.12, +0.21
                      va: [1.30, 1.50, 1.80, 1.60, 1.58],
                      gf: [0.90, 1.00, 0.60, 0.77, 1.50],  # -0.10, then 0 at the other four sites
                      cl: [1.40, np.nan, 1.60, 1.77, 2.50]})  # +0.40, none, then +1.0 at three sites
    a = figs.table1_agreement(r)
    assert a.loc[zf, "n_sites"] == 4 and a.loc[zf, "n_within_tol"] == 2 and a.loc[zf, "n_within_0p01"] == 1
    assert a.loc[zf, "mean_diff"] == pytest.approx(0.37 / 4) and a.loc[zf, "max_diff"] == pytest.approx(0.21)
    assert a.loc[zf, "rmsd"] == pytest.approx(np.sqrt((0.04 ** 2 + 0.12 ** 2 + 0.21 ** 2) / 4))
    assert a.loc[va, "n_sites"] == 5 and a.loc[va, "n_within_tol"] == 0
    assert a.loc[va, "min_diff"] == pytest.approx(0.08) and a.loc[va, "max_diff"] == pytest.approx(1.20)
    assert a.loc[gf, "mean_diff"] == pytest.approx(-0.02) and a.loc[gf, "rmsd"] == pytest.approx(np.sqrt(0.01 / 5))
    assert a.loc[cl, "n_sites"] == 4 and a.loc[cl, "mean_diff"] == pytest.approx((0.40 + 3 * 1.0) / 4)
    assert figs.table1_agreement(r, tol=0.25).loc[zf, "n_within_tol"] == 4
