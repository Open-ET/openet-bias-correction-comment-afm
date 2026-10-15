import numpy as np
import pandas as pd
import pytest

from etcomment import closure as cl
from etcomment import metrics as mt
from etcomment import zerofill as zf


def test_mse_decomposition_is_exact():
    rng = np.random.default_rng(1)
    o = rng.gamma(2.0, 1.0, 500)
    p = 0.8 * o + 0.4 + rng.normal(0, 0.5, 500)
    d = mt.mse_decomposition(p, o)
    assert np.isclose(d["mse"], d["bias2"] + d["variance_term"])
    assert np.isclose(d["variance_term"], d["spread_mismatch"] + d["correlation_deficit"])


def test_r2_cod_can_be_negative_and_pearson_cannot():
    o = np.array([1.0, 2.0, 3.0, 4.0])
    p = o + 5.0
    assert mt.r2_cod(p, o) < 0
    assert np.isclose(mt.r2_pearson(p, o), 1.0)


def test_mbe_invariant_to_daily_to_monthly_aggregation():
    idx = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    rng = np.random.default_rng(2)
    o = pd.Series(rng.gamma(2, 1, idx.size), idx)
    p = o + rng.normal(-0.2, 0.6, idx.size)
    daily = mt.mbe(p, o)
    monthly = mt.mbe(p.resample("MS").sum(), o.resample("MS").sum())
    days = o.resample("MS").size().mean()
    assert np.isclose(monthly, daily * days, rtol=1e-12)
    assert np.sign(monthly) == np.sign(daily)


def test_zero_fill_shifts_bias_by_f_times_e():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2021-01-01", periods=1000, freq="D")
    truth = pd.Series(rng.gamma(3, 1, idx.size), idx)
    model = truth + rng.normal(0.0, 0.5, idx.size)
    missing = pd.Series(rng.random(idx.size) < 0.25, idx)
    obs_zero = truth.where(~missing, 0.0)
    shift = mt.mbe(model, obs_zero) - mt.mbe(model, truth)
    exp = zf.expected_bias_shift(missing, truth)
    assert np.isclose(shift, exp["delta_mbe"])


def test_zero_runs_and_outer_join():
    obs = pd.Series([1.0, 2.0, 3.0], pd.to_datetime(["2024-06-01", "2024-06-02", "2024-06-10"]))
    full = pd.date_range("2024-05-28", "2024-06-12", freq="D")
    filled = zf.outer_join_zero_fill(obs, full)
    runs = zf.zero_runs(filled)
    assert list(runs["n_days"]) == [4, 7, 2]


def test_bowen_ratio_closure_equals_ebr_scaling_and_diverges():
    le, h, rn, g = 100.0, 50.0, 200.0, 20.0
    le_c, f = cl.bowen_ratio_closure(le, h, rn, g)
    assert np.isclose(le_c, le * (rn - g) / (le + h))
    assert np.isclose(le_c / (le_c + h * f), le / (le + h))  # Bowen ratio preserved
    # advection: H < 0 with LE + H -> 0 makes the factor explode
    le_c2, f2 = cl.bowen_ratio_closure(100.0, -98.0, 150.0, 10.0)
    assert f2 > 50
    flags = cl.br_diagnostics([100.0, 100.0], [-98.0, 50.0], [150.0, 200.0], [10.0, 20.0])
    assert bool(flags.loc[0, "flux_near_zero"]) and not bool(flags.loc[1, "any_problem"])


def test_ebr_windowed_factor_is_finite_where_daily_br_explodes():
    n = 60
    le = np.full(n, 100.0); h = np.full(n, 40.0); rn = np.full(n, 180.0); g = np.full(n, 10.0)
    h[30] = -99.0  # one advective day
    le_br, f_br = cl.bowen_ratio_closure(le, h, rn, g)
    le_ebr, f_ebr = cl.ebr_closure(le, h, rn, g, window=15)
    assert f_br[30] > 50
    assert 1.0 < f_ebr[30] < 1.5


def test_ameriflux_base_reader_and_reconstruction(tmp_path):
    import importlib.util
    from pathlib import Path
    from etcomment import ameriflux as af

    t = pd.date_range("2024-05-30", "2024-06-03 23:30", freq="30min")
    n = t.size
    le = np.where(t < pd.Timestamp("2024-06-01"), -9999, 100.0)  # tower starts 1 June
    df = pd.DataFrame({
        "TIMESTAMP_START": t.strftime("%Y%m%d%H%M"),
        "TIMESTAMP_END": (t + pd.Timedelta("30min")).strftime("%Y%m%d%H%M"),
        "LE": le, "H": np.where(le < 0, -9999, 50.0), "NETRAD": np.where(le < 0, -9999, 200.0),
        "G_1_1_1": np.where(le < 0, -9999, 10.0), "G_2_1_1": np.where(le < 0, -9999, 30.0),
    })
    f = tmp_path / "AMF_US-XXX_BASE_HH_1-5.csv"
    with open(f, "w") as fh:
        fh.write("# Site: US-XXX\n# Version: 1-5\n")
        df.to_csv(fh, index=False)
    hh = af.read_base(f)
    assert np.isclose(hh["G"].dropna().iloc[0], 20.0)  # replicate G sensors averaged
    d = af.daily(hh)
    assert list(d["LE_frac"].round(2)) == [0.0, 0.0, 1.0, 1.0, 1.0]
    spec = importlib.util.spec_from_file_location(
        "rec", Path(__file__).resolve().parents[1] / "scripts" / "10_reconstruct_go26_observed_et.py")
    rec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rec)
    out = rec.site_observed_et(tmp_path, "US-XXX", "2024-05", "2024-06")
    assert out["valid_LE_days"] == 3 and out["calendar_days"] == 61
    # zero-filled mean is the valid mean scaled by the retained fraction
    assert np.isclose(out["mean_et_unclosed_zero_filled"], out["mean_et_unclosed_valid"] * 3 / 61)
    # with whole days only, the half-hourly zero-filled sum gives the same mean
    assert np.isclose(out["mean_et_hh_sum_zero_filled_unclosed"], out["mean_et_unclosed_zero_filled"])


def test_zero_filled_daily_sum_counts_missing_as_zero():
    from etcomment import ameriflux as af

    t = pd.date_range("2024-06-01", "2024-06-03 23:30", freq="30min")
    le = pd.Series(100.0, index=t)
    le.loc["2024-06-02 12:00":"2024-06-02 23:30"] = np.nan  # half of day 2 missing
    le.loc["2024-06-03"] = np.nan                             # day 3 missing
    et = af.et_per_step(le, 2.45)
    daily = af.zero_filled_daily_sum(et)
    full = 100.0 * 86400 / 2.45e6
    assert np.allclose(daily.to_numpy(), [full, full / 2, 0.0])
    # the default pandas daily sum behaves the same way
    assert np.allclose(et.resample("D").sum().to_numpy(), daily.to_numpy())


def test_period_month_days_and_weighted_mean():
    from etcomment import benchmark as bm

    w = bm.period_month_days("2020-01", "2025-04")
    assert w[1] == 6 * 31 and w[12] == 5 * 31 and w[2] == 29 + 28 * 3 + 29 + 28
    clim = pd.Series(np.arange(1, 13, dtype=float), index=range(1, 13))
    flat = pd.Series(1.0, index=range(1, 13))
    assert np.isclose(bm.month_weighted_mean(flat, w), 1.0)
    # extra Januaries-Aprils pull the weighted mean below the plain monthly mean
    assert bm.month_weighted_mean(clim, w) < clim.mean()
    # a missing month that carries weight gives NaN; one without weight does not matter
    gap = clim.copy(); gap[7] = np.nan
    assert np.isnan(bm.month_weighted_mean(gap, w))
    assert np.isfinite(bm.month_weighted_mean(gap, bm.period_month_days("2020-01", "2020-03")))


def test_complete_months_uses_calendar_days():
    from etcomment import benchmark as bm

    idx = pd.date_range("2019-11-01", "2019-12-30", freq="D")  # record stops a day early
    d = pd.DataFrame({"et_corr": 1.0, "et_gap": False}, index=idx)
    d.iloc[:6, d.columns.get_loc("et_gap")] = True  # six gap-filled days in November
    m = bm.complete_months(d)
    assert len(m) == 0  # November fails the 5-gap rule, December is a day short
    d["et_gap"] = False
    assert list(bm.complete_months(d).index.month) == [11]
