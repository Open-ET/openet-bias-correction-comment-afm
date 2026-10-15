import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etcomment import openet_compare as oc


def _monthly(site: str, per_day: float = 2.0, start="2016-01", end="2025-12") -> pd.DataFrame:
    """Synthetic API monthly table: ET at ``per_day`` mm/d plus a calendar-month signal."""
    idx = pd.date_range(f"{start}-01", f"{end}-01", freq="MS")
    days = np.asarray(idx.days_in_month, float)
    et = (per_day + 0.1 * idx.month.to_numpy()) * days
    df = pd.DataFrame({"site": site, "date": idx, "et_ensemble": et, "et_sims": 1.2 * et,
                       "eto": 3.0 * days, "pr": 1.0 * days})
    return df


def _script():
    spec = importlib.util.spec_from_file_location(
        "s12", Path(__file__).resolve().parents[1] / "scripts" / "12_openet_api_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_season_names_follow_go26():
    assert list(oc.season([12, 1, 2, 3, 6, 9, 11])) == ["winter", "winter", "winter", "spring", "summer",
                                                         "fall", "fall"]


def test_fill_monthly_uses_same_calendar_month_mean_and_records_months():
    m = _monthly("X").set_index("date")["et_ensemble"]
    jan = m[m.index.month == 1]
    gappy = m.drop(pd.Timestamp("2020-01-01"))          # absent row
    gappy[pd.Timestamp("2021-01-01")] = np.nan          # NaN value
    gappy[pd.Timestamp("2018-01-01")] = 1000.0          # an outlying January
    filled, months = oc.fill_monthly(gappy, "2020-01", "2021-12")
    assert months == ["2020-01", "2021-01"]
    assert len(filled) == 24 and filled.notna().all()
    expected = jan.drop([pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")])
    expected[pd.Timestamp("2018-01-01")] = 1000.0
    assert np.isclose(filled[pd.Timestamp("2020-01-01")], expected.mean())
    assert np.isclose(filled[pd.Timestamp("2021-01-01")], expected.mean())
    # untouched months keep their values; the period mean is the total over calendar days
    assert filled[pd.Timestamp("2020-07-01")] == m[pd.Timestamp("2020-07-01")]
    assert np.isclose(oc.mean_daily(filled), filled.sum() / (366 + 365))
    # without data for a calendar month in any year the period mean is undefined
    no_feb = m[m.index.month != 2]
    f2, months2 = oc.fill_monthly(no_feb, "2020-01", "2020-03")
    assert months2 == ["2020-02"] and np.isnan(oc.mean_daily(f2))
    with pytest.raises(ValueError):
        oc.fill_monthly(pd.Series([1.0, 2.0], pd.to_datetime(["2020-01-01", "2020-01-15"])), "2020-01", "2020-01")


def test_zero_fill_prediction_weights_months_by_their_year():
    idx = pd.date_range("2023-01-01", "2024-12-01", freq="MS")
    days = np.asarray(idx.days_in_month, float)
    oe = pd.Series(2.0 * days, idx)                     # 2 mm/d throughout
    p = oc.zero_fill_prediction(oe, {2023: 0.5, 2024: 1.0})
    assert np.isclose(p, 2.0 * (0.5 * 365 + 1.0 * 366) / (365 + 366))
    assert np.isclose(oc.zero_fill_prediction(oe, {2023: 1.0, 2024: 1.0}), oc.mean_daily(oe))
    assert np.isnan(oc.zero_fill_prediction(oe, {2023: 0.5}))  # 2024 unknown


def test_go26_site_table_and_summary_recover_zero_fill_scaling():
    # T1 = c x OE x A at every site: the ratio is proportional to availability
    c = 0.8
    avail = {"US-AAA": 0.3, "US-BBB": 0.6, "US-CCC": 0.9, "US-DDD": 0.45}
    sites, monthly, by_year = [], [], []
    for i, (s, a) in enumerate(avail.items()):
        m = _monthly(s, per_day=1.5 + 0.3 * i)
        monthly.append(m)
        p = m[(m["date"] >= "2020-01-01") & (m["date"] <= "2021-12-01")]
        oe = p["et_ensemble"].sum() / np.asarray(p["date"].dt.days_in_month).sum()
        sites.append({"site": s, "koppen_go26": "Dfa", "period_start": "2020-01", "period_end": "2021-12",
                      "mean_obs_et_mm_d": c * oe * a, "map_mm": 500.0, "in_volk_benchmark": i == 0})
        by_year += [{"site": s, "year": y, "le_avail": a, "days_in_year": d, "days_in_period": d}
                    for y, d in ((2020, 366), (2021, 365))]
    monthly = pd.concat(monthly, ignore_index=True)
    monthly.loc[(monthly["site"] == "US-AAA") & (monthly["date"] == "2020-02-01"), "et_ensemble"] = np.nan
    ap = pd.DataFrame({"site": list(avail), "le_avail_period": list(avail.values())})
    t = oc.go26_site_table(pd.DataFrame(sites), monthly, ap, pd.DataFrame(by_year), models=["ensemble", "sims"])
    a = t.set_index("site")
    assert a.loc["US-AAA", "n_months_filled"] == 1 and a.loc["US-AAA", "months_filled"] == "2020-02"
    assert np.allclose(a["sims_mm_d"].drop("US-AAA"), 1.2 * a["oe_mm_d"].drop("US-AAA"))
    assert a.loc["US-AAA", "sims_n_months_filled"] == 0
    assert np.allclose(a["eto_annual_mm"], 3.0 * 365.25) and np.allclose(a["pr_annual_mm"], 365.25)
    assert np.allclose(a["oe_annual_mm"], a["oe_mm_d"] * 365.25)
    assert np.allclose(a["p_zf_mm_d"], a["oe_mm_d"] * a["le_avail_period"])
    assert np.allclose(a["t1_over_avail_over_oe"].drop("US-AAA"), c)
    assert np.allclose(a["p_zf_uniform_mm_d"], a["p_zf_mm_d"])  # whole years: both conventions agree
    s = oc.site_summary(t)
    assert s["all"]["n_sites"] == 4 and s["not_in_benchmark"]["n_sites"] == 3
    rv = s["not_in_benchmark"]["ratio_vs_availability"]
    assert np.isclose(rv["pearson_r"], 1.0) and np.isclose(rv["spearman_r"], 1.0) and np.isclose(rv["slope"], c)
    zf = s["not_in_benchmark"]["t1_vs_p_zf"]
    assert np.isclose(zf["slope_through_origin"], c) and zf["rmse_fit"] < 1e-9
    assert np.isclose(zf["mean_diff_t1_minus"], ((c - 1) * t["p_zf_mm_d"]).iloc[1:].mean())


def test_go26_site_table_without_availability_has_nan_predictions():
    m = _monthly("US-AAA")
    sites = pd.DataFrame([{"site": "US-AAA", "koppen_go26": "BSk", "period_start": "2024-01",
                           "period_end": "2025-02", "mean_obs_et_mm_d": 0.6, "map_mm": 200.0,
                           "in_volk_benchmark": False}])
    t = oc.go26_site_table(sites, m, None, None)
    assert t.loc[0, "n_days"] == 366 + 31 + 28
    assert np.isnan(t.loc[0, "le_avail_period"]) and np.isnan(t.loc[0, "p_zf_mm_d"])
    assert not t.loc[0, "irrigated_dry_climate"]


def test_period_availability_for_partial_years():
    # 2025 has 0.1 of the year's half-hours with LE; the period holds only Jan-Feb 2025
    idx = pd.date_range("2024-01-01", "2025-02-01", freq="MS")
    oe = pd.Series(np.asarray(idx.days_in_month, float), idx)  # 1 mm/d
    by = pd.DataFrame({"year": [2024, 2025], "le_avail": [1.0, 0.1], "days_in_year": [366, 365],
                       "days_in_period": [366, 59]}).set_index("year")
    uniform = oc.zero_fill_prediction(oe, by["le_avail"])
    within = oc.zero_fill_prediction(oe, oc.period_availability(by))
    assert np.isclose(uniform, (366 + 0.1 * 59) / 425)
    assert np.isclose(within, (366 + 0.1 * 365) / 425)   # the year's 36.5 days of records lie in Jan-Feb
    # scripts/11 writes the capped record days itself; they take precedence
    by["days_available"] = [366.0, 59.0]
    assert np.isclose(oc.period_availability(by)[2025], 1.0)
    by.loc[2025, "le_avail"] = 0.5   # more records than days in the period: capped at 1
    assert np.isclose(oc.period_availability(by.drop(columns="days_available"))[2025], 1.0)
    # without day counts the whole-year fraction is used
    assert np.allclose(oc.period_availability(by[["le_avail"]]), by["le_avail"])


def test_fit_through_origin_and_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    f = oc.fit_through_origin(2.5 * x, x)
    assert np.isclose(f["slope"], 2.5) and f["rmse"] < 1e-12
    c = oc.correlation(x, x ** 3)
    assert np.isclose(c["spearman_r"], 1.0) and c["pearson_r"] < 1.0
    assert np.isnan(oc.correlation([1.0, 2.0], [1.0, 2.0])["pearson_r"])


def _pairs() -> pd.DataFrame:
    """Two stations per Koppen group; OpenET is high in summer and low in winter."""
    rows = []
    shift = {"winter": -6.0, "spring": 0.0, "summer": 12.0, "fall": 3.0}
    for site, grp, extra, years in (("A1", "B", 0.0, 1), ("A2", "B", 6.0, 3), ("C1", "C", -3.0, 1),
                                    ("C2", "C", -3.0, 2)):
        for mon in pd.date_range("2016-01-01", periods=12 * years, freq="MS"):
            s = oc.season([mon.month])[0]
            obs = 50.0 + 10 * mon.month
            rows.append({"site": site, "month": mon, "koppen_group": grp, "season": s,
                         "n_days": mon.days_in_month, "obs": obs, "pred": obs + shift[s] + extra})
    return pd.DataFrame(rows)


def test_stratified_bias_pooled_statistics_without_station_weighted_mbes():
    p = _pairs()
    t = oc.stratified_bias(p, {"all": [], "season": ["season"], "season_x_group": ["season", "koppen_group"],
                               "site": ["site"]})
    t = t.set_index(["stratum_type", "stratum"])
    assert set(t.loc["season"].index) == set(oc.SEASONS)
    assert t.loc[("season", "summer"), "mbe"] == pytest.approx(12.0 + (0 + 6 * 3 - 3 - 3 * 2) / 7)
    pooled = (p["pred"] - p["obs"]).mean()
    assert t.loc[("all", "all"), "mbe"] == pytest.approx(pooled)
    assert t.loc[("all", "all"), "n_stations"] == 4 and t.loc[("all", "all"), "n"] == 12 * 7
    per_day = ((p["pred"] - p["obs"]) / p["n_days"]).mean()
    assert t.loc[("all", "all"), "mbe_per_day"] == pytest.approx(per_day)
    w = t.loc[("season_x_group", "winter, B")]
    assert w["n_stations"] == 2 and w["mbe"] == pytest.approx(-6.0 + 6.0 * 9 / 12)
    assert t.loc[("site", "C1"), "r2"] <= 1.0
    # station-weighted MBEs let single months be solved for (see the identifiability tests): not computed
    assert not [c for c in t.columns if "station_mean" in c]


def test_bias_stats_r2_and_percent():
    o = np.array([10.0, 20.0, 30.0, 40.0])
    s = oc.bias_stats(o + 5.0, o, n_days=[31, 30, 31, 30])
    assert s["mbe"] == 5.0 and np.isclose(s["mbe_pct"], 20.0)
    assert np.isclose(s["r2"], 1 - 4 * 25 / 500)
    assert np.isclose(s["mbe_per_day"], np.mean(5.0 / np.array([31, 30, 31, 30])))
    assert oc.bias_stats([np.nan], [1.0])["n"] == 0


def test_winter_check_fractions():
    idx = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    d = pd.DataFrame({"obs": 1.0, "openet": 0.5, "eto": 0.8, "pr": 0.0}, index=idx)
    d.loc["2020-01-01":"2020-01-10", "eto"] = 2.0          # 10 January days with obs < ETo
    d.loc["2020-02-01":"2020-02-04", "obs"] = np.nan       # no observation
    d.loc["2020-02-05", "obs"] = 0.0                       # near-zero observation
    d.loc["2020-11-01":"2020-11-15", "openet"] = np.nan    # OpenET missing
    w = oc.winter_check(d).set_index("months")
    assert w.loc["Jan", "n_days"] == 31 and np.isclose(w.loc["Jan", "frac_obs_gt_eto"], 21 / 31)
    assert w.loc["Feb", "n_days"] == 25 and w.loc["Feb", "n_obs_near_zero"] == 1
    assert np.isclose(w.loc["Feb", "frac_obs_gt_eto"], 24 / 25)
    assert np.isclose(w.loc["Nov", "frac_obs_gt_openet"], 1.0)
    assert np.isclose(w.loc["Jan-Feb", "mean_eto_mm_d"], (10 * 2.0 + 21 * 0.8 + 25 * 0.8) / 56)
    assert w.loc["Jan-Mar, Nov-Dec", "n_days"] == 31 + 25 + 31 + 30 + 31
    d.loc["2020-03-01":"2020-03-03", "obs"] = 2.5
    w = oc.winter_check(d).set_index("months")
    assert np.isclose(w.loc["Mar", "frac_obs_gt_1mm_d"], 3 / 31) and w.loc["Mar", "max_obs_mm_d"] == 2.5


def test_measured_climatology_and_distance():
    idx = pd.date_range("2001-01-01", "2002-12-31", freq="D")
    a = pd.DataFrame({"et_corr": 0.5, "et": 0.4, "eto": 1.0, "valid_measured": True}, index=idx)
    b = a.copy()
    b["et_corr"] = 1.5
    b.loc["2001-01-01":"2001-01-31", "valid_measured"] = False
    c = oc.measured_climatology({"S1": a, "S2": b}, (1, 2))
    n_a = int(np.isin(idx.month, (1, 2)).sum())
    assert c["n_stations"] == 2 and c["n_days"] == n_a + n_a - 31
    assert np.isclose(c["mean_closed_mm_d"], (0.5 * n_a + 1.5 * (n_a - 31)) / (2 * n_a - 31))
    assert c["station_mean_min"] == 0.5 and c["station_mean_max"] == 1.5
    assert np.isclose(c["frac_closed_gt_eto"], (n_a - 31) / (2 * n_a - 31))
    assert np.isclose(c["frac_closed_gt_1mm_d"], c["frac_closed_gt_eto"]) and c["max_closed_mm_d"] == 1.5
    assert oc.measured_climatology({}, (1,))["n_stations"] == 0
    # one degree of latitude is about 111.2 km
    assert np.isclose(oc.haversine_km(40.0, -88.0, 41.0, -88.0), 111.19, atol=0.05)


def test_monthly_and_daily_pairs_use_complete_measured_data():
    idx = pd.date_range("2015-12-01", "2016-03-31", freq="D")
    d = pd.DataFrame({"et_corr": 2.0, "et": 1.6, "et_gap": False}, index=idx)
    d.loc["2016-02-01":"2016-02-06", "et_gap"] = True     # six gap-filled days: February fails
    d["valid_measured"] = d["et_corr"].notna() & ~d["et_gap"]
    api = pd.DataFrame({"date": pd.date_range("2015-12-01", "2016-03-01", freq="MS"),
                        "et_ensemble": [70.0, 60.0, 50.0, np.nan], "eto": 1.0})
    mp = oc.monthly_pairs(d, api)
    assert list(mp["month"].dt.month) == [1, 3]           # 2015-12 before start, Feb has 6 gap days
    assert mp.loc[0, "et_corr"] == 62.0 and mp.loc[0, "et_ensemble"] == 60.0 and "eto" not in mp
    s = pd.Series(1.0, pd.date_range("2016-01-01", "2016-03-31", freq="D"))
    dp = oc.daily_pairs(d, s)
    assert len(dp) == 31 + 29 - 6 + 31 and (dp["openet"] == 1.0).all()


def test_load_availability_missing_and_partial(tmp_path):
    p, y, msgs = oc.load_availability(tmp_path / "a.csv", tmp_path / "b.csv")
    assert p is None and y is None and len(msgs) == 2 and "scripts/11" in msgs[0]
    pd.DataFrame({"site": ["US-AAA"], "le_avail_period": [0.4]}).to_csv(tmp_path / "a.csv", index=False)
    p, y, msgs = oc.load_availability(tmp_path / "a.csv", tmp_path / "b.csv")
    assert p is not None and y is None and len(msgs) == 1
    pd.DataFrame({"site": ["US-AAA"], "year": [2024], "avail": [0.4]}).to_csv(tmp_path / "b.csv", index=False)
    assert "lacks columns" in oc.load_availability(tmp_path / "a.csv", tmp_path / "b.csv")[2][0]
    pd.DataFrame({"site": ["US-AAA"], "year": [2024], "le_avail": [40.0]}).to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(ValueError):
        oc.load_availability(tmp_path / "a.csv", tmp_path / "b.csv")


def _go26_inputs(tmp_path: Path, with_availability: bool):
    sites = pd.DataFrame([
        {"site": s, "koppen_go26": "BSk", "period_start": "2023-01", "period_end": "2024-12",
         "mean_obs_et_mm_d": t1, "map_mm": 200.0, "in_volk_benchmark": False}
        for s, t1 in (("US-UTJ", 0.6), ("US-AAA", 1.2), ("US-BBB", 1.6))])
    monthly = pd.concat([_monthly(s, per_day=2.0) for s in sites["site"]], ignore_index=True)
    av, ay = tmp_path / "avail.csv", tmp_path / "avail_by_year.csv"
    if with_availability:
        a = {"US-UTJ": 0.3, "US-AAA": 0.6, "US-BBB": 0.8}
        pd.DataFrame({"site": list(a), "le_avail_period": list(a.values())}).to_csv(av, index=False)
        pd.DataFrame([{"site": s, "year": y, "le_avail": v, "days_in_year": n, "days_in_period": n}
                      for s, v in a.items() for y, n in ((2023, 365), (2024, 366))]).to_csv(ay, index=False)
    return sites, monthly, av, ay


@pytest.mark.parametrize("with_availability", [False, True])
def test_script_part_a_skips_availability_outputs_when_files_are_missing(tmp_path, capsys, with_availability):
    s12 = _script()
    sites, monthly, av, ay = _go26_inputs(tmp_path, with_availability)
    t, summary = s12.part_a(sites, monthly, av, ay, tmp_path, prov={"test": True})
    out = capsys.readouterr().out
    written = json.loads((tmp_path / "openet_api_go26_summary.json").read_text())
    assert (tmp_path / "openet_api_go26_sites.csv").exists()
    assert "https://openetdata.org" in written["attribution"]
    if with_availability:
        assert written["availability"]["available"] is True
        assert np.isfinite(t["p_zf_mm_d"]).all()
        assert np.isfinite(written["go26_towers"]["all"]["ratio_vs_availability"]["pearson_r"])
    else:
        assert "skipping" in out and "scripts/11" in out
        assert written["availability"]["available"] is False and written["availability"]["messages"]
        assert t["p_zf_mm_d"].isna().all() and t["le_avail_period"].isna().all()
        assert written["go26_towers"]["all"]["ratio_vs_availability"]["pearson_r"] is None
    flagged = {r["site"] for r in written["irrigated_dry_climate_sites"]}
    assert flagged == {"US-UTJ"}


def test_daily_sensitivity_in_site_table_and_summary():
    sites, monthly, by_year = [], [], []
    avail = {"US-AAA": 0.3, "US-BBB": 0.6, "US-CCC": 0.9, "US-DDD": 0.5}
    for i, (s, a) in enumerate(avail.items()):
        monthly.append(_monthly(s, per_day=1.5 + 0.3 * i))
        sites.append({"site": s, "koppen_go26": "Dfa", "period_start": "2020-01", "period_end": "2021-12",
                      "mean_obs_et_mm_d": 0.8 * a * (1.5 + 0.3 * i), "map_mm": 500.0, "in_volk_benchmark": False})
        by_year += [{"site": s, "year": 2020, "le_avail": a, "days_in_year": 366, "days_in_period": 366},
                    {"site": s, "year": 2021, "le_avail": a / 2, "days_in_year": 365, "days_in_period": 365}]
    monthly, by_year = pd.concat(monthly, ignore_index=True), pd.DataFrame(by_year)
    days = pd.date_range("2019-12-01", "2022-01-31", freq="D")
    daily = pd.concat([pd.DataFrame({"site": "US-AAA", "date": days, "et_ensemble": 2.0 + (days.year == 2021)}),
                       pd.DataFrame({"site": "US-BBB", "date": days[100:], "et_ensemble": 1.0})])  # BBB: partial
    t = oc.go26_site_table(pd.DataFrame(sites), monthly, None, by_year, daily=daily).set_index("site")
    n20, n21 = 366, 365
    assert np.isclose(t.loc["US-AAA", "oe_daily_mm_d"], (2.0 * n20 + 3.0 * n21) / (n20 + n21))
    aaa = t.loc["US-AAA"]
    assert np.isclose(aaa["ratio_t1_oe_daily"], aaa["t1_mm_d"] / aaa["oe_daily_mm_d"])
    assert np.isclose(t.loc["US-AAA", "p_zf_daily_mm_d"], (2.0 * 0.3 * n20 + 3.0 * 0.15 * n21) / (n20 + n21))
    assert t.loc[["US-BBB", "US-CCC", "US-DDD"], "oe_daily_mm_d"].isna().all()  # not covering the period / no data
    assert "oe_daily_mm_d" not in oc.go26_site_table(pd.DataFrame(sites), monthly, None, by_year)
    t = t.reset_index().assign(le_avail_period=list(avail.values()))
    s = oc.site_summary(t)["all"]
    d = s["daily_oe_where_available"]
    assert d["n_sites_daily"] == 1 and d["sites_daily"] == "US-AAA"
    alt = t.assign(oe_mm_d=t["oe_daily_mm_d"].fillna(t["oe_mm_d"]),
                   p_zf_mm_d=t["p_zf_daily_mm_d"].fillna(t["p_zf_mm_d"]))
    alt["ratio_t1_oe"] = alt["t1_mm_d"] / alt["oe_mm_d"]
    assert np.isclose(d["ratio_vs_availability"]["pearson_r"],
                      oc.correlation(alt["le_avail_period"], alt["ratio_t1_oe"])["pearson_r"])
    assert np.isclose(d["t1_vs_p_zf"]["slope_through_origin"],
                      oc.fit_through_origin(alt["t1_mm_d"], alt["p_zf_mm_d"])["slope"])
    assert not np.isclose(d["t1_vs_oe"]["pearson_r"], s["t1_vs_oe"]["pearson_r"])


def test_series_scale_and_monthly_agreement():
    idx = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    api = pd.Series(1.0 + np.arange(idx.size) / 30.0, idx)
    go = 0.8 * api.drop(idx[40:45])                        # GO26 line 20 % low, five days not plotted
    a = oc.series_agreement(go, api)
    assert a["n"] == idx.size - 5 and np.isclose(a["scale"], 0.8) and a["rmse_scaled"] < 1e-12
    assert np.isclose(a["pearson_r"], 1.0)                 # r cannot see the amplitude error
    m = oc.monthly_agreement(go, api)
    assert list(m["month"]) == ["2024-01", "2024-02", "2024-03"] and list(m["n"]) == [31, 24, 31]
    feb = api["2024-02"].drop(idx[40:45], errors="ignore")
    assert np.isclose(m.loc[1, "mean_b"], feb.mean()) and np.isclose(m.loc[1, "mean_diff"], -0.2 * feb.mean())
    empty = oc.series_agreement(go.iloc[:0], api)
    assert empty["n"] == 0 and np.isnan(empty["scale"])
    assert oc.monthly_agreement(go.iloc[:0], api).empty


def test_withheld_stations_protect_small_stations_and_class_remainders():
    rows = []
    for site, k, n in (("A", "BSk", 24), ("B", "BSk", 2), ("C", "BSk", 3), ("D", "Dfa", 24), ("E", "Dfa", 6),
                       ("F", "Cfa", 9), ("G", "Cfa", 2), ("H", "Cfa", 4), ("I", "Cfa", 8)):
        for mon in pd.date_range("2016-01-01", periods=n, freq="MS"):
            rows.append({"site": site, "koppen": k, "month": mon, "pred": 1.0, "obs": 2.0})
    p = pd.DataFrame(rows)
    # B and C (5 months) cannot stand alone in BSk, so A goes too; G and H (6 months) suffice in Cfa
    assert oc.withheld_stations(p) == ["A", "B", "C", "G", "H"]
    assert oc.withheld_stations(p, min_months=4) == ["B", "C", "G", "H"]  # G alone (2 months) needs H
    p.loc[(p["site"] == "E") & (p["month"] == "2016-01-01"), "pred"] = np.nan  # E: 5 months with a value
    assert oc.withheld_stations(p) == ["A", "B", "C", "D", "E", "G", "H"]
    t = oc.stratified_bias(p.assign(n_days=30), {"site": ["site"]}, time="month").set_index("stratum")
    assert t.loc["E", "first"] == pd.Timestamp("2016-02-01") and t.loc["E", "last"] == pd.Timestamp("2016-06-01")


def _digitized(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    rows, digi = [], tmp_path / "digi"
    digi.mkdir()
    for site, year, scale in (("US-UiC", 2020, 1.0), ("US-UTJ", 2024, 0.8)):
        idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        api = 1.0 + 3.0 * np.sin(np.pi * (idx.dayofyear.to_numpy() - 1) / 366) ** 2
        rows.append(pd.DataFrame({"site": site, "date": idx, "et_ensemble": api}))
        g = pd.DataFrame({"date": idx, "openet": scale * api, "openet_interp": False, "obs": 1.0})
        g.loc[100:104, "openet"] = np.nan                  # gaps in the plotted line
        g.loc[200:209, "openet_interp"] = True
        g.to_csv(digi / f"digitized_{site}.csv", index=False)
    return pd.concat(rows, ignore_index=True), digi


def test_script_part_b_reports_scale_and_monthly_differences(tmp_path, capsys):
    s12 = _script()
    daily, digi = _digitized(tmp_path)
    df, mo, ident = s12.part_b(daily, digi, tmp_path)
    assert "monthly means" in capsys.readouterr().out
    assert set(ident) == {"US-UiC", "US-UTJ"} and max(ident.values()) < 1 - oc.IDENT_TOL
    f = pd.read_csv(tmp_path / "openet_api_go26_figure_check.csv").set_index(["site", "days"])
    assert np.allclose(f.xs("all plotted days", level="days")["scale_go26_over_api"], [1.0, 0.8])
    assert (f["rmse_scaled_mm_d"] < 1e-4).all() and (f["pearson_r"] > 0.999).all()
    m = pd.read_csv(tmp_path / "openet_api_go26_figure_check_monthly.csv")
    assert list(m.columns) == ["site", "go26_figure", "month", "n", "mean_go26_openet_mm_d",
                               "mean_api_openet_mm_d", "diff_go26_minus_api_mm_d"]
    assert len(m) == 24 and m.groupby("site")["n"].sum().tolist() == [366 - 5, 366 - 5]
    utj = m[m["site"] == "US-UTJ"]
    assert np.allclose(utj["diff_go26_minus_api_mm_d"], -0.2 * utj["mean_api_openet_mm_d"], atol=1e-4)
    assert np.allclose(m.loc[m["site"] == "US-UiC", "diff_go26_minus_api_mm_d"], 0.0, atol=1e-4)
    # without the digitized series both files are written empty
    df2, mo2, _ = s12.part_b(daily, tmp_path / "none", tmp_path)
    assert df2.empty and mo2.empty


def _part_c_inputs():
    """Standardized daily benchmark frames, station metadata and API monthly totals (all models)."""
    frames, meta, api = {}, [], []
    for site, k, start, months in (("A", "BSk", "2016-01-01", 24), ("B", "BSk", "2016-01-01", 2),
                                   ("C", "BSk", "2016-05-01", 3), ("D", "Dfa", "2016-01-01", 24),
                                   ("E", "Dfa", "2017-01-01", 10)):
        idx = pd.date_range(start, pd.Timestamp(start) + pd.DateOffset(months=months) - pd.Timedelta(days=1))
        et = 1.0 + 3.0 * np.sin(np.pi * (idx.dayofyear.to_numpy() - 1) / 366) ** 2
        frames[site] = pd.DataFrame({"et_corr": et, "et": 0.8 * et, "et_gap": False, "valid_measured": True},
                                    index=idx)
        meta.append({"site": site, "koppen": k, "koppen_group": k[0], "technique": "Eddy covariance"})
        m = frames[site]["et_corr"].resample("MS").sum()
        api.append(pd.DataFrame({"site": site, "date": m.index, **{f"et_{x}": m.to_numpy() * (0.9 + 0.02 * i)
                                                                    for i, x in enumerate(oc.MODELS)}}))
    api = pd.concat(api, ignore_index=True)
    api.loc[(api["site"] == "D") & (api["date"] == "2016-06-01"), [f"et_{x}" for x in oc.MODELS]] = np.nan
    go = pd.DataFrame({"site": ["D"], "koppen_go26": ["Dfa"]})
    zones = pd.DataFrame({"zone": ["Overall (all sites)", "BSk", "Dfa"], "openet_bias": [0.6, 1.1, 0.5],
                          "source": ["GO26 Table 3"] * 3}).set_index("zone")
    return frames, pd.DataFrame(meta), api, go, zones


def test_script_part_c_keeps_station_months_out_of_results(tmp_path, capsys):
    s12 = _script()
    frames, meta, api, go, zones = _part_c_inputs()
    out, local = tmp_path / "results", tmp_path / "derived"
    out.mkdir()
    (out / "openet_api_benchmark_monthly.csv").write_text("site,month\n")  # output of earlier versions
    pairs, summ, no_openet, ident = s12.part_c(frames, meta, api, go, zones, out, local)
    printed = capsys.readouterr().out
    assert no_openet == ["D 2016-06"] and "D 2016-06" in printed
    assert "62 station-months with an OpenET ensemble value at 5 cropland stations" in printed
    assert sorted(p.name for p in out.iterdir()) == ["openet_api_benchmark_summary.csv"]
    assert set(ident) == set(oc.MODELS) and all(not v["stations_withheld_by_check"] for v in ident.values())
    sm = pd.read_csv(local / "benchmark_station_months.csv")
    assert len(sm) == 62 and sm["ensemble_minus_ec_mm"].notna().all()
    assert np.allclose(sm["ensemble_minus_ec_mm"], -0.1 * sm["ec_closed_et_mm"], atol=2e-3)
    s = pd.read_csv(out / "openet_api_benchmark_summary.csv")
    assert not [c for c in s.columns if "station_mean" in c]
    site = s[s["stratum_type"] == "site"].set_index(["model", "stratum"])
    withheld = site["openet_withheld"].groupby(level="stratum").all()
    assert withheld.to_dict() == {"A": True, "B": True, "C": True, "D": False, "E": False}
    assert site.loc[site["openet_withheld"], s12.OPENET_STATS].isna().all().all()
    assert site.loc[site["openet_withheld"], "n_months"].notna().all()
    assert not s.loc[s["stratum_type"] != "site", "openet_withheld"].any()
    e = s[(s["model"] == "ensemble") & (s["stratum_type"] == "all")].iloc[0]
    assert (e["first_month"], e["last_month"], e["n_months"]) == ("2016-01", "2017-12", 62)
    assert np.isclose(e["mbe_pct"], -10.0)


# ----------------------------------------------------------------------------- identifiability of OpenET values

def _station_months(spec, seed: int = 0) -> pd.DataFrame:
    """Benchmark-like pairs for (site, koppen, first month, number of months); OpenET 0.9 x EC plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for site, k, start, n in spec:
        for mon in pd.date_range(start, periods=n, freq="MS"):
            obs = 40.0 + 12 * mon.month + rng.normal(0, 5)
            rows.append({"site": site, "koppen": k, "month": mon, "season": oc.season([mon.month])[0],
                         "n_days": mon.days_in_month, "obs": obs, "pred": 0.9 * obs + rng.normal(0, 4)})
    return pd.DataFrame(rows)


# the withheld Cfa pool of the benchmark: US-OF1 Jun-Aug 2017, US-OF2 Jul 2017, US-OF4 Jul 2018 (+ US-OF6)
OF_LIKE = [("B1", "BSk", "2016-01-01", 24), ("BIG", "Cfa", "2016-01-01", 24), ("MID", "Cfa", "2018-01-01", 10),
           ("OF1", "Cfa", "2017-06-01", 3), ("OF2", "Cfa", "2017-07-01", 1), ("OF4", "Cfa", "2018-07-01", 1)]
STRATA3 = {"all": [], "koppen": ["koppen"], "site": ["site"]}


def _determined(p: pd.DataFrame, proj: pd.Series) -> list[str]:
    return sorted(f"{p.loc[j, 'site']} {p.loc[j, 'month']:%Y-%m}" for j in proj.index[proj > 1 - oc.IDENT_TOL])


def test_projection_diagonal_marks_unknowns_in_the_span():
    rows = pd.DataFrame([[1.0, 1.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 3.0, 3.0],
                         [0.0, 0.0, 0.0, 0.0]], columns=list("abcd"))
    assert np.allclose(oc.projection_diagonal(rows), [1.0, 1.0, 0.5, 0.5])  # a and b solvable; only c + d known
    assert (oc.projection_diagonal(rows.iloc[[4]]) == 0).all()


def test_identifiability_flags_months_solvable_from_the_published_mbes():
    # two withheld 30-day months (the OF1 and OF6 Junes) among 31-day months: the pooled MBEs in mm/month and
    # mm/d of the class minus its published stations give only their sum ...
    p = _station_months(OF_LIKE + [("OF6", "Cfa", "2018-06-01", 2)])
    assert oc.withheld_stations(p) == ["OF1", "OF2", "OF4", "OF6"]
    pub = oc.published_strata(p, STRATA3, oc.withheld_stations(p))
    assert ("site", "BIG") in pub and ("site", "OF1") not in pub and ("koppen", "Cfa") in pub
    idf = oc.identifiability(p, STRATA3, pub)
    assert _determined(p, idf["linear"]) == [] and _determined(p, idf["with_rmse"]) == []
    # ... and station-weighted MBEs (OF1 has 3 months, OF6 2) split it: why they are not published
    st = {}
    for name, cols in STRATA3.items():
        for key, g in (p.groupby(cols) if cols else [("all", p)]):
            if (name, oc._label(key)) in pub:
                w = pd.Series(0.0, p.index)
                w[g.index] = 1.0 / (g["site"].nunique() * g.groupby("site")["site"].transform("size"))
                st[(name, oc._label(key), "station_mean")] = w
                st[(name, oc._label(key), "station_mean_per_day")] = w / p["n_days"]
    rows = pd.concat([oc.stratum_functionals(p, STRATA3, pub, rmse=False), pd.DataFrame(st).T])
    assert _determined(p, oc.projection_diagonal(rows)) == ["OF1 2017-06", "OF6 2018-06"]
    # a single withheld 30-day month is fixed by the pooled MBEs alone; the check then withholds the
    # next-smallest published station of the class (MID) as well
    p = _station_months(OF_LIKE + [("OF6", "Cfa", "2018-07-01", 1)])
    idf = oc.identifiability(p, STRATA3, oc.published_strata(p, STRATA3, oc.withheld_stations(p)))
    assert _determined(p, idf["linear"]) == ["OF1 2017-06"]
    checked = oc.withheld_stations(p, strata=STRATA3)
    assert checked == ["MID", "OF1", "OF2", "OF4", "OF6"]
    assert oc.identifiability(p, STRATA3, oc.published_strata(p, STRATA3, checked))["with_rmse"].max() < 1 - 1e-6
    # when withholding stations cannot help, an error rather than a silent leak
    lonely = _station_months([("A", "BSk", "2016-01-01", 24), ("X", "Cfa", "2017-06-01", 3),
                              ("Y", "Cfa", "2017-07-01", 2), ("Z", "Cfa", "2018-07-01", 1)])
    with pytest.raises(ValueError, match="Cfa"):
        oc.withheld_stations(lonely, strata=STRATA3)


def _benchmark_inputs(spec, seed: int = 1):
    """Daily benchmark frames, station metadata and API monthly totals of all models."""
    rng = np.random.default_rng(seed)
    frames, meta, api = {}, [], []
    for site, k, start, months in spec:
        idx = pd.date_range(start, pd.Timestamp(start) + pd.DateOffset(months=months) - pd.Timedelta(days=1))
        et = 1.0 + 3.0 * np.sin(np.pi * (idx.dayofyear.to_numpy() - 1) / 366) ** 2 + rng.uniform(0, 0.5, idx.size)
        frames[site] = pd.DataFrame({"et_corr": et, "et": 0.8 * et, "et_gap": False, "valid_measured": True},
                                    index=idx)
        meta.append({"site": site, "koppen": k, "koppen_group": k[0], "technique": "Eddy covariance"})
        m = frames[site]["et_corr"].resample("MS").sum()
        api.append(pd.DataFrame({"site": site, "date": m.index,
                                 **{f"et_{x}": m.to_numpy() * (0.9 + 0.02 * i) + rng.normal(0, 3, m.size)
                                    for i, x in enumerate(oc.MODELS)}}))
    return frames, pd.DataFrame(meta), pd.concat(api, ignore_index=True)


def test_script_part_c_withholds_until_the_published_summary_determines_no_month(tmp_path, capsys):
    s12 = _script()
    frames, meta, api = _benchmark_inputs(OF_LIKE + [("OF6", "Cfa", "2018-07-01", 1)])
    go = pd.DataFrame({"site": ["BIG"], "koppen_go26": ["Cfa"]})
    zones = pd.DataFrame({"zone": ["Overall (all sites)", "Cfa"], "openet_bias": [0.6, 0.5],
                          "source": ["GO26 Table 3"] * 2}).set_index("zone")
    out, local = tmp_path / "results", tmp_path / "derived"
    out.mkdir()
    pairs, _, _, ident = s12.part_c(frames, meta, api, go, zones, out, local)
    assert "also withheld: MID" in capsys.readouterr().out
    assert all(v["stations_withheld_by_check"] == ["MID"] for v in ident.values())
    s = pd.read_csv(out / "openet_api_benchmark_summary.csv")
    assert not [c for c in s.columns if "station_mean" in c]
    # an outsider's view: the published rows of the file and the station-months of each stratum
    for model in oc.MODELS:
        pub = s[(s["model"] == model) & ~s["openet_withheld"]]
        assert pub.loc[pub["stratum_type"] == "site", "stratum"].tolist() == ["B1", "BIG"]
        q = pairs.rename(columns={f"et_{model}": "pred", "et_corr": "obs"})
        idf = oc.identifiability(q, s12.STRATA, set(zip(pub["stratum_type"], pub["stratum"])))
        assert len(idf) == len(pairs) and idf["with_rmse"].max() < 1 - 1e-6


def test_script_part_b_refuses_statistics_that_determine_a_daily_value(tmp_path):
    s12 = _script()
    daily, digi = _digitized(tmp_path)
    g = pd.read_csv(digi / "digitized_US-UTJ.csv", parse_dates=["date"])
    g["openet"] = 0.8 * daily.loc[daily["site"] == "US-UTJ", "et_ensemble"].to_numpy()
    g.loc[150, "openet"] = np.nan                         # one day of 2024 without GO26's line
    g.to_csv(digi / "digitized_US-UTJ.csv", index=False)
    t = pd.DataFrame({"site": ["US-UTJ"], "period_start": ["2024-01"], "period_end": ["2024-12"],
                      "oe_daily_mm_d": [2.0]})
    s12.part_b(daily, digi, tmp_path)                     # the figure check alone determines no day
    (tmp_path / "openet_api_go26_figure_check.csv").unlink()
    with pytest.raises(RuntimeError, match="2024-05-30"):  # the year sum of (a) minus the monthly sums would
        s12.part_b(daily, digi, tmp_path, t)
    assert not (tmp_path / "openet_api_go26_figure_check.csv").exists()


def test_script_part_e_keeps_api_values_out_of_results(tmp_path, monkeypatch):
    s12 = _script()
    from etcomment import maps

    monkeypatch.setattr(maps, "koppen_at", lambda lat, lon: ["Dfa"] * len(np.atleast_1d(lat)))
    idx = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    digi = tmp_path / "digi"
    digi.mkdir()
    g = pd.DataFrame({"date": idx, "obs": 1.0, "openet": 0.5})
    g.loc[g["date"] == "2020-11-02", "obs"] = np.nan       # GO26's OpenET line but no observation
    g.loc[g["date"].between("2020-03-24", "2020-03-26"), "openet"] = np.nan  # observations, no line
    g.to_csv(digi / "digitized_US-UiC.csv", index=False)
    daily = pd.DataFrame({"site": "US-UiC", "date": idx, "et_ensemble": np.linspace(0.2, 0.8, idx.size),
                          "eto": 1.0, "pr": 2.0})
    bidx = pd.date_range("2001-01-01", "2002-12-31", freq="D")
    frames = {"S1": pd.DataFrame({"et_corr": 0.6, "et": 0.5, "eto": 1.2, "valid_measured": True}, index=bidx)}
    meta = pd.DataFrame({"site": ["S1"], "koppen": ["Dfa"], "lat": [40.1], "lon": [-88.0]})
    go = pd.DataFrame({"site": ["US-UiC"], "latitude": [40.06], "longitude": [-88.2], "koppen_go26": ["Dfa"]})
    out, local = tmp_path / "results", tmp_path / "derived"
    out.mkdir()
    res = s12.part_e(frames, meta, daily, go, digi, out, local)
    r = pd.read_csv(out / "openet_api_uic_winter.csv")
    assert list(res.columns) == list(r.columns)
    assert not {"mean_openet_mm_d", "mean_pr_mm_d", "frac_obs_gt_openet"} & set(r.columns)
    u = r[r["group"] == "US-UiC"].set_index("months")
    assert u[["mean_eto_mm_d", "frac_obs_gt_eto"]].isna().all().all() and u["eto_source"].isna().all()
    assert np.isclose(u.loc["Jan-Feb", "mean_obs_mm_d"], 1.0) and u.loc["Nov", "n_days"] == 29
    assert np.isclose(r.loc[r["group"] != "US-UiC", "mean_eto_mm_d"], 1.2).all()  # benchmark ETo, not the API's
    # the full table (not deposited) shows why: with (b)'s monthly mean over the plotted days, the November
    # mean over the observation days gives the API value of 2 November
    full = pd.read_csv(local / s12.UIC_WINTER)
    fu = full[full["group"] == "US-UiC"].set_index("months")
    api = daily.set_index("date")["et_ensemble"]
    nov = oc.monthly_agreement(g.set_index("date")["openet"], api).set_index("month").loc["2020-11"]
    rec = nov["n"] * nov["mean_b"] - fu.loc["Nov", "n_days"] * fu.loc["Nov", "mean_openet_mm_d"]
    assert np.isclose(rec, api[pd.Timestamp("2020-11-02")], atol=5e-3)


def test_summary_json_rounds_openet_statistics(tmp_path):
    s12 = _script()
    s12.write_json(tmp_path / "s.json", {"provenance": {"v": 2.123456789},
                                         "go26_towers": {"r": 0.123456789, "pearson_p": 1.23456789e-7,
                                                         "n_sites": 22.0, "x": [1.000049, np.nan, -0.00001]}})
    j = json.loads((tmp_path / "s.json").read_text())
    assert j["provenance"]["v"] == 2.123456789
    assert j["go26_towers"] == {"r": 0.1235, "pearson_p": 1.235e-07, "n_sites": 22, "x": [1.0, None, 0.0]}


def test_fill_weights_reproduce_fill_monthly():
    m = _monthly("X").set_index("date")["et_ensemble"]
    gappy = m.drop(pd.to_datetime(["2020-01-01", "2021-07-01", "2016-07-01"]))
    gappy[pd.Timestamp("2022-07-01")] = np.nan
    have = gappy.dropna().sort_index()
    filled, months = oc.fill_monthly(gappy, "2020-01", "2022-12")
    w = oc.fill_weights(have.index, "2020-01", "2022-12")
    assert months == ["2020-01", "2021-07", "2022-07"]
    assert np.allclose(w.to_numpy() @ have.to_numpy(), filled.to_numpy())
    assert w.loc["2021-07-01"].gt(0).sum() == 7              # Julys with data: 10 years minus 2016, 2021, 2022
    assert w.loc["2020-02-01"].isna().all() == False         # noqa: E712
    assert oc.fill_weights(m[m.index.month != 2].index, "2020-01", "2020-03").loc["2020-02-01"].isna().all()
