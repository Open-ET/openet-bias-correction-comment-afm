import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _script06():
    spec = importlib.util.spec_from_file_location(
        "shared", Path(__file__).resolve().parents[1] / "scripts" / "06_shared_site_comparison.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_record_coverage_ignores_empty_rows_before_and_after_the_record():
    """Shares of measured and gap-filled days count only the tower's record, not empty file rows."""
    idx = pd.date_range("2017-01-01", "2017-12-31", freq="D")
    et = pd.Series(np.nan, index=idx)
    et.loc["2017-06-29":"2017-11-30"] = 1.0  # tower starts in late June, file runs all year
    gap = pd.Series(False, index=idx)
    gap.loc["2017-07-01":"2017-08-31"] = True  # two months gap-filled inside the record
    d = pd.DataFrame({"et_corr": et, "et_gap": gap})
    d["valid_measured"] = d["et_corr"].notna() & ~d["et_gap"]
    c = _script06().record_coverage(d)
    n = (pd.Timestamp("2017-11-30") - pd.Timestamp("2017-06-29")).days + 1
    assert c["bm_days"] == n
    assert str(c["bm_first"]) == "2017-06-29" and str(c["bm_last"]) == "2017-11-30"
    assert np.isclose(c["bm_pct_gap_filled"], 100 * 62 / n)
    assert np.isclose(c["bm_pct_measured"] + c["bm_pct_gap_filled"], 100.0)
