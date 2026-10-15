"""Tests for the AmeriFlux BASE reconstruction (scripts/10) and the BASE downloader.

Everything runs on synthetic files in tmp_path; HTTP is mocked and real requests are blocked.  No test reads
config/, data/ or results/ (main() is always given --amf-public '' or a tmp folder).
"""
import hashlib
import importlib.util
import io
import itertools
import json
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import pytest
import requests

from etcomment import ameriflux as af
from etcomment import closure as cl

C = 0.0864 / 2.45  # mm/d per W m-2 of daily mean LE
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "10_reconstruct_go26_observed_et.py"


@pytest.fixture(scope="module")
def rec():
    spec = importlib.util.spec_from_file_location("rec10", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("real HTTP request attempted in a test")
    monkeypatch.setattr(requests.sessions.Session, "request", blocked)


def write_base(path: Path, t: pd.DatetimeIndex, cols: dict, version: str, site: str = "US-XXX") -> Path:
    step = t[1] - t[0]
    df = pd.DataFrame({"TIMESTAMP_START": t.strftime("%Y%m%d%H%M"),
                       "TIMESTAMP_END": (t + step).strftime("%Y%m%d%H%M")})
    for k, v in cols.items():
        df[k] = np.where(np.isnan(v), -9999.0, v)
    with open(path, "w") as fh:
        fh.write(f"# Site: {site}\n# Version: {version}\n")
        df.to_csv(fh, index=False)
    return path


def same(a, b) -> bool:
    """Equal, with NaN equal to NaN."""
    return a == b or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))


# -- file names, versions, reader -------------------------------------------------------------------
def test_base_names_versions_and_selection():
    assert af.parse_base_name("AMF_US-Ne1_BASE_HR_20-5.csv") == {"site": "US-Ne1", "resolution": "HR",
                                                                "version": "20-5"}
    assert af.parse_base_name(Path("x") / "AMF_US-xSL_BASE_HH_10-5.csv")["site"] == "US-xSL"
    assert af.parse_base_name("AMF_US-Ne1_FLUXNET_FULLSET_HH_2001-2020_3-5.csv") is None
    assert af.parse_base_name("AMF_US-Ne1_BASE-BADM_20-5.zip") is None
    assert af.normalize_version("Ver. 20-5") == "20-5"
    assert af.normalize_version(np.nan) == "" and af.normalize_version(None) == ""
    assert sorted(["10-5", "9-5", "2-5"], key=af.version_key) == ["2-5", "9-5", "10-5"]
    files = [Path(f"AMF_US-Ne1_BASE_HR_{v}.csv") for v in ("19-5", "20-5", "9-5")]
    files += [Path("AMF_US-Ne1_BASE_HH_20-5.csv"), Path("notes.csv")]
    sel = af.select_base_file(files, prefer="20-5")
    assert (sel["version"], sel["resolution"], sel["versions"]) == ("20-5", "HH", ["9-5", "19-5", "20-5"])
    assert af.select_base_file(files)["version"] == "20-5"                  # latest by default
    assert af.select_base_file(files, prefer="Ver. 19-5")["path"].name == "AMF_US-Ne1_BASE_HR_19-5.csv"
    assert af.select_base_file(files, prefer="21-5")["version"] == "20-5"   # cited version absent: latest
    assert af.select_base_file([Path("notes.csv")]) is None


def test_read_base_takes_turbulent_fluxes_from_one_position(tmp_path):
    t = pd.date_range("2024-06-01", periods=48, freq="30min")
    full = lambda v: np.full(t.size, float(v))  # noqa: E731
    f = write_base(tmp_path / "AMF_US-AAA_BASE_HH_2-5.csv", t, {
        "LE_1_1_1": full(100), "LE_1_2_1": full(np.nan),         # no LE at the second level
        "H_1_1_1": full(40), "H_1_2_1": full(80),                # H at two levels
        "NETRAD": full(np.nan), "NETRAD_1_1_1": full(200),       # empty unqualified column
        "G_PI_F_1_1_1": full(15), "LE_PI_F_1_1_1": full(999),    # gap-filled: only G is kept
        "TA_1_1_1": full(10), "TA_1_2_1": full(20),
        "VPD_PI": full(1.5), "VPD_PI_F": full(9.0),              # VPD as named at the GO26 sites
        "LE_SSITC_TEST_1_1_1": np.tile([0.0, 2.0], 24), "LE_SSITC_TEST_1_2_1": full(1)}, "2-5", "US-AAA")
    hh = af.read_base(f)
    assert hh["LE"].eq(100).all() and hh["NETRAD"].eq(200).all()
    assert hh["H"].eq(40).all()  # the LE position, not the mean of both levels
    assert "G" not in hh and hh["G_PI_F"].eq(15).all()
    assert hh["TA"].eq(15).all()  # other variables without an unqualified column: positions averaged
    assert hh["VPD"].eq(1.5).all() and hh.attrs["columns"]["VPD"] == ["VPD_PI"]
    assert hh["LE_SSITC_TEST"].tolist() == [0.0, 2.0] * 24  # the flag at the LE position only
    assert hh.attrs["columns"]["H"] == ["H_1_1_1"] and hh.attrs["columns"]["NETRAD"] == ["NETRAD_1_1_1"]
    assert hh.attrs["columns"]["LE_SSITC_TEST"] == ["LE_SSITC_TEST_1_1_1"]
    assert hh.attrs["replicates"] == {}
    assert af.base_header(f) == {"site": "US-AAA", "version": "2-5"}


def test_read_base_column_rules(tmp_path):
    t = pd.date_range("2024-06-01", periods=96, freq="30min")
    n = np.arange(t.size)
    some = lambda v, k: np.where(n < k, float(v), np.nan)  # noqa: E731  (first k records valid)
    f = write_base(tmp_path / "AMF_US-AAA_BASE_HH_1-5.csv", t, {
        "LE": some(80, 96), "LE_1_1_1": some(120, 96),                 # equal records: the unqualified one
        "NETRAD": some(150, 40), "NETRAD_1_1_1": some(200, 96),        # the one with more valid records
        "G": some(5, 20), "G_1_1_1": some(10, 96), "G_2_1_1": some(30, 96),  # unqualified G whenever it has data
        "LE_SSITC_TEST": some(1, 96), "LE_SSITC_TEST_1_1_1": some(2, 96)}, "1-5", "US-AAA")
    hh = af.read_base(f)
    used = hh.attrs["columns"]
    assert (used["LE"], used["NETRAD"], used["G"]) == (["LE"], ["NETRAD_1_1_1"], ["G"])
    assert hh["LE"].eq(80).all() and hh["G"].iloc[:20].eq(5).all() and hh["G"].iloc[20:].isna().all()
    assert used["LE_SSITC_TEST"] == ["LE_SSITC_TEST"] and hh["LE_SSITC_TEST"].eq(1).all()
    assert (af.position("LE_1_2_3"), af.position("VPD_PI_1_1_1"), af.position("LE")) == ((1, 2), (1, 1), ())
    assert af.best_position({(1, 1): 5, (): 5, (1, 2): 7}) == (1, 2)
    assert af.best_position({(1, 2): 5, (): 5, (1, 1): 5}) == ()

    def flag_used(cols: dict) -> list:
        return af.read_base(write_base(tmp_path / "AMF_US-FLG_BASE_HH_1-5.csv", t, cols, "1-5", "US-FLG")
                            ).attrs["columns"].get("LE_SSITC_TEST", [])

    # LE at one position with an unqualified flag (US-VT1: LE_1_1_1, LE_SSITC_TEST, empty LE_SSITC_TEST_1_1_1)
    assert flag_used({"LE_1_1_1": some(1, 96), "LE_SSITC_TEST": some(0, 96),
                      "LE_SSITC_TEST_1_1_1": some(0, 0)}) == ["LE_SSITC_TEST"]
    # LE at two levels: an unqualified flag cannot be assigned; a flag at another position is never used
    assert flag_used({"LE_1_1_1": some(1, 96), "LE_1_2_1": some(1, 50), "LE_SSITC_TEST": some(0, 96)}) == []
    assert flag_used({"LE": some(1, 96), "LE_SSITC_TEST_1_1_1": some(0, 96)}) == []


def test_read_base_keeps_net_radiometers_apart(tmp_path):
    """Two radiometers at one position (as at the Utah towers): never averaged; NETRAD is the one with the most
    records in the file and both are returned under their BASE names, while LE replicates are still averaged."""
    t = pd.date_range("2024-06-01", periods=96, freq="30min")
    n = np.arange(t.size)
    f = write_base(tmp_path / "AMF_US-RAD_BASE_HH_1-5.csv", t, {
        "LE_1_1_1": np.full(t.size, 100.0), "LE_1_1_2": np.where(n < 48, 50.0, np.nan),
        "NETRAD_1_1_1": np.where(n < 60, 300.0, np.nan), "NETRAD_1_1_2": np.where(n >= 40, 200.0, np.nan)},
        "1-5", "US-RAD")
    hh = af.read_base(f)
    assert hh.attrs["columns"]["NETRAD"] == ["NETRAD_1_1_1"] and hh["NETRAD"].iloc[:60].eq(300).all()
    assert hh["NETRAD"].iloc[60:].isna().all()                       # no switch to the other radiometer
    assert hh.attrs["replicates"] == {"NETRAD": ["NETRAD_1_1_1", "NETRAD_1_1_2"]}
    assert hh["NETRAD_1_1_2"].iloc[40:].eq(200).all() and hh["NETRAD_1_1_2"].iloc[:40].isna().all()
    assert hh["LE"].iloc[:48].eq(75).all() and hh["LE"].iloc[48:].eq(100).all()  # LE replicates averaged


# -- gap fill and closure factors ------------------------------------------------------------------
def test_mdv_daily_fills_missing_records_by_time_of_day():
    days = pd.date_range("2024-06-01", "2024-06-20")
    t = pd.date_range(days[0], periods=len(days) * 48, freq="30min")
    cycle = 100 + 200 * np.sin(np.pi * (np.arange(48) - 12) / 24).clip(0)  # diurnal cycle, the same every day
    x = pd.Series(np.tile(cycle, len(days)), index=t)
    night = (t.hour < 6) | (t.hour >= 18)
    gappy = x.mask(night & (t.day % 2 == 0))                                # night missing on even days
    gappy = gappy.mask((t.day == 10) & (t.hour == 3) & (t.minute == 0))    # and on 10 June at 03:00
    out = af.mdv_daily(gappy, days)
    assert np.allclose(out["mean"], cycle.mean())                           # the gaps are filled exactly
    assert np.isclose(gappy[t.day == 2].mean(), cycle[12:36].mean()) and cycle[12:36].mean() > cycle.mean() + 50
    assert out.loc["2024-06-02", "n_filled"] == 24 and out.loc["2024-06-02", "n_valid"] == 24
    assert out["n_unfilled"].eq(0).all()
    # a time of day without any value in the window stays missing; the day's mean is over the other records
    slot = gappy.mask((t.hour == 3) & (t.minute == 0))
    one = af.mdv_daily(slot, days)
    assert one["n_unfilled"].eq(1).all() and np.allclose(one["mean"], np.delete(cycle, 6).mean())
    # only records on the given days are used (here: nothing to fill 1 June from but 1-8 June)
    first = af.mdv_daily(gappy, days[:1])
    assert first.loc["2024-06-01", "n_filled"] == 0
    with pytest.raises(ValueError):
        af.mdv_daily(gappy, days[[0, 2]])
    assert af.mdv_daily(gappy, days[:0]).empty


def test_screened_bowen_ratio_factor():
    days = pd.date_range("2024-01-01", periods=30)
    flux = pd.Series(100.0, index=days)
    energy = pd.Series(120.0, index=days)
    flux.iloc[5], energy.iloc[6] = 0.5, -30.0       # factor 240 and -1.2: both fail
    flux.iloc[7], energy.iloc[7] = -50.0, -60.0     # factor 1.2 with both sums negative: fails
    flux.iloc[8] = 300.0                            # factor 0.4: fails
    ok = pd.Series(True, index=days)
    ok.iloc[9] = False                              # no energy-balance data that day
    out = cl.screened_br_factor(flux, energy, ok)
    assert out["passed"].sum() == 25 and not out["passed"].iloc[[5, 6, 7, 8, 9]].any()
    assert np.isclose(out["factor"].iloc[5], 240) and np.isnan(out["factor"].iloc[9])
    assert np.allclose(out["screened"], 1.2)        # failing days take the 15-day median of passing factors
    few = cl.screened_br_factor(flux, energy, ok & (days.day > 26))   # 4 passing days: < 5 in any window
    assert np.allclose(few["screened"], 1.2) and few["passed"].sum() == 4


def test_ebr_factor_follows_flux_data_qaqc():
    """IQR filter, 15-day median (>= 5 values) else 11-day median, (0.5, 2) test, then the day-of-year
    climatology centred on the next day, then the median."""
    days = pd.date_range("2021-01-01", "2022-12-31")
    energy = pd.Series(100.0, index=days)
    flux = pd.Series(80.0, index=days)              # EBR 0.8: factor 1.25
    flux.iloc[[10, 40, 41]] = [-300.0, 900.0, 5.0]  # outliers: removed by the IQR filter
    f = cl.ebr_factor(flux, energy)
    assert np.allclose(f, 1.25)
    # sparse data: one ratio every 4 days, so 15-day windows hold <= 4 values and the 11-day median is used
    sparse = pd.Series(np.nan, index=days)
    n4 = len(sparse.iloc[::4])
    sparse.iloc[::4] = np.where(np.arange(n4) % 2 == 0, 0.7, 0.9)
    s = cl.ebr_factor(sparse * 100, energy, sparse.notna())
    i = 100  # an 11-day window around a day holds 3 values (0.7, 0.9, 0.7 or 0.9, 0.7, 0.9)
    window = sparse.iloc[i - 5:i + 6].dropna()
    assert len(sparse.iloc[i - 7:i + 8].dropna()) < 5 and np.isclose(s.iloc[i], 1 / window.median())
    # a median whose factor is not strictly within (0.5, 2) is rejected; with nothing else, the median of the kept
    low = cl.ebr_factor(pd.Series(40.0, index=days[:30]), pd.Series(100.0, index=days[:30]))  # EBR 0.4
    assert np.allclose(low, 2.5)  # step 2 rejects 0.4 (factor 2.5 >= 2), the climatology is empty: median 0.4
    # a year without data is filled from the climatology of the other year, centred on the next day of year
    ramp = pd.Series(0.6 + 0.001 * days.dayofyear, index=days).where(days.year == 2021)
    g = cl.ebr_factor(ramp * 100, energy, ramp.notna())
    k = 180  # 2022 day of year 180: mean of the 2021 ratios on days 176-186 (centred on day 181)
    assert np.isclose(g[pd.Timestamp("2022-01-01") + pd.Timedelta(days=k - 1)], 1 / (0.6 + 0.001 * (k + 1)))
    assert cl.ebr_factor(flux, energy, pd.Series(False, index=days)).isna().all()


def test_closure_uses_matched_records_gap_fill_and_screens_singular_days(tmp_path, rec):
    """LE missing at night must bias neither the daily factor nor the gap-filled mean; a day with LE + H near 0 is
    screened and filled."""
    t = pd.date_range("2024-06-01", "2024-06-30 23:30", freq="30min")
    night = (t.hour < 6) | (t.hour >= 18)
    le, h = np.where(night, 10.0, 200.0), np.where(night, -10.0, 100.0)   # LE + H: 0 at night, 300 by day
    rn, g = np.where(night, -10.0, 400.0), np.where(night, -10.0, 40.0)   # Rn - G: 0 at night, 360 by day
    le[night & (t.day % 2 == 0)] = np.nan                                  # LE missing at night on even days
    sing = (t.day == 15) & ~night
    h[sing] = -le[sing] + 0.01                                             # 15 June: LE + H = 0.24 W m-2 in total
    write_base(tmp_path / "AMF_US-EBC_BASE_HH_1-5.csv", t, {"LE": le, "H": h, "NETRAD": rn, "G": g}, "1-5", "US-EBC")
    res, daily = rec.reconstruct_site(tmp_path, "US-EBC", "2024-06", "2024-06", "1-5")
    day = daily.set_index("date")
    normal = day.index != "2024-06-15"
    # per-variable daily means would give (360 / 2) / (200 + 45) = 0.73 on even days; matched records give 1.2
    assert np.allclose(day.loc[normal, "br_factor"], 1.2)
    assert day.loc["2024-06-15", "br_factor"] > 1000 and "BR_screen_fail" in day.loc["2024-06-15", "flags"]
    assert (res["valid_EB_days"], res["n_br_screen_fail"], res["n_br_factor_gt2"]) == (30, 1, 1)
    assert np.allclose(day["br_factor_screened"], 1.2) and np.allclose(day["ebr_factor"], 1.2)
    # the mean of valid records is 200 W m-2 on even days; the gap-filled mean is 105 every day
    assert np.allclose(day.loc[day.index.day % 2 == 0, "et_unclosed"], 200 * C)
    assert np.allclose(day["et_unclosed_mdv"], 105 * C)
    assert np.isclose(res["mean_et_unclosed_valid"], C * (105 + 200) / 2)
    assert np.isclose(res["mean_et_unclosed_mdv"], 105 * C)
    assert np.isclose(res["mdv_filled_share"], 15 * 24 / (30 * 48))
    assert np.isclose(res["mean_et_ebr_mdv"], 1.2 * 105 * C)
    assert np.isclose(res["mean_et_br_screened_mdv"], 1.2 * 105 * C)
    assert np.isclose(res["mean_et_br_screened"], 1.2 * res["mean_et_unclosed_valid"])
    assert np.allclose(day["et_ebr_mdv"], day["et_unclosed_mdv"] * 1.2)
    assert res["max_et_br_unscreened"] > 100 * res["mean_et_br_screened"]
    assert res["max_et_br_unscreened_date"] == "2024-06-15" and res["valid_EB_days_nov_mar"] == 0


def test_reconstruction_chooses_one_net_radiometer_per_period(tmp_path, rec):
    """Over the file NETRAD_1_1_1 has more records, in the study period NETRAD_1_1_2: the latter is used throughout
    and the closure with the other is reported as a sensitivity."""
    t = pd.date_range("2024-05-01", "2024-06-30 23:30", freq="30min")
    may = t.month == 5
    cols = {"LE": np.full(t.size, 100.0), "H": np.full(t.size, 50.0), "G": np.full(t.size, 10.0),
            "NETRAD_1_1_1": np.where(may | (t.day <= 5), 250.0, np.nan),  # all of May and 5 days of June
            "NETRAD_1_1_2": np.where(~may, 190.0, np.nan)}                  # June only
    write_base(tmp_path / "AMF_US-UTX_BASE_HH_1-5.csv", t, cols, "1-5", "US-UTX")
    res, daily = rec.reconstruct_site(tmp_path, "US-UTX", "2024-06", "2024-06", "1-5")
    assert (res["netrad_column"], res["netrad_column_alt"]) == ("NETRAD_1_1_2", "NETRAD_1_1_1")
    assert "NETRAD_1_1_2 used throughout" in res["notes"]
    assert np.allclose(daily["br_factor"], (190 - 10) / 150) and res["valid_EB_days"] == 30
    assert res["valid_EB_days_netrad_alt"] == 5
    assert np.isclose(res["mean_et_ebr_mdv_netrad_alt"], 100 * C * (250 - 10) / 150)
    single = rec.reconstruct_site(tmp_path, "US-UTX", "2024-05", "2024-05", "1-5")[0]  # May: only one reports
    assert (single["netrad_column"], single["netrad_column_alt"]) == ("NETRAD_1_1_1", "")


# -- reconstruction ---------------------------------------------------------------------------------
def _hourly_site(tmp_path):
    """US-YYY, hourly, January 2024: 20 full days, 6 h on the 21st, nothing after; data also outside."""
    t = pd.date_range("2023-12-31", "2024-02-01 23:00", freq="h")
    le = np.full(t.size, 100.0)
    le[(t >= "2024-01-21") & (t < "2024-02-01")] = np.nan
    le[(t >= "2024-01-21 10:00") & (t <= "2024-01-21 15:00")] = 100.0
    other = lambda v: np.where(np.isnan(le), np.nan, v)  # noqa: E731
    cols = {"LE": le, "H": other(50.0), "NETRAD": other(200.0), "G": other(20.0)}
    write_base(tmp_path / "AMF_US-YYY_BASE_HR_20-5.csv", t, cols, "20-5", "US-YYY")
    write_base(tmp_path / "AMF_US-YYY_BASE_HR_19-5.csv", t, {**cols, "LE": np.full(t.size, 50.0)}, "19-5", "US-YYY")
    return t, le


def test_reconstruction_versions_zero_fraction_and_daily_table(tmp_path, rec):
    t, le = _hourly_site(tmp_path)
    res, daily = rec.reconstruct_site(tmp_path, "US-YYY", "2024-01", "2024-01", go26_version="20-5")
    assert res["status"] == "ok" and res["base_files"] == "AMF_US-YYY_BASE_HR_20-5.csv"
    assert (res["base_resolution"], res["base_version"], res["version_matches_go26"]) == ("HR", "20-5", True)
    assert res["base_versions_available"] == "19-5;20-5" and res["base_last_record"] == "2024-02-01 23:00"
    assert res["calendar_days"] == 31 and res["file_days"] == 31
    assert res["valid_LE_days"] == 20 and res["valid_EB_days"] == 20
    assert res["period_days_before_first_LE"] == 0 and res["period_days_after_last_LE"] == 0
    assert np.isclose(res["f_days_missing"], 11 / 31)
    assert np.isclose(res["le_record_availability"], (20 * 24 + 6) / (31 * 24))
    assert np.isclose(res["mean_et_unclosed_valid"], 100 * C) and np.isclose(res["mean_et_unclosed_mdv"], 100 * C)
    hourly = 100 * 3600 / 2.45e6
    assert np.isclose(res["mean_et_hh_sum_zero_filled_unclosed"], (20 * 24 + 6) * hourly / 31)
    assert res["mean_et_hh_sum_zero_filled_file_days"] == res["mean_et_hh_sum_zero_filled_unclosed"]
    for k in ("mean_et_br_unscreened", "mean_et_ebr", "mean_et_ebr_mdv", "mean_et_br_screened",
              "mean_et_br_screened_mdv"):
        assert np.isclose(res[k], 100 * C * (200 - 20) / (100 + 50)), k
    out = rec.compare_with_go26(dict(res), 2.0)
    assert np.isclose(out["f_implied"], 1 - 2.0 / (100 * C)) and out["f_implied_unfilled"] == out["f_implied"]
    assert np.isclose(out["diff_mean_et_unclosed_valid"], 100 * C - 2.0)
    assert np.isclose(out["diff_mean_et_ebr_mdv"], 1.2 * 100 * C - 2.0)

    # the daily zero-filled sums are pandas' default daily sums of hourly ET
    s = pd.Series(le, index=t)[(t >= "2024-01-01") & (t < "2024-02-01")] * 3600 / 2.45e6
    assert np.allclose(daily["et_hh_sum_zero_filled"], s.resample("D").sum().to_numpy())
    day = daily.set_index("date")
    assert day.loc["2024-01-01", "flags"] == "" and np.isclose(day.loc["2024-01-01", "et_unclosed"], 100 * C)
    assert day.loc["2024-01-21", "flags"] == "LE_lt_50pct" and np.isnan(day.loc["2024-01-21", "et_unclosed"])
    assert np.isclose(day.loc["2024-01-21", "le_frac"], 0.25) and np.isnan(day.loc["2024-01-21", "et_ebr_mdv"])
    assert day.loc["2024-01-25", "flags"] == "no_LE" and day.loc["2024-01-25", "et_hh_sum_zero_filled"] == 0
    assert list(daily.columns) == rec.DAILY_COLUMNS

    # version choice: GO26's version when present, else the latest; the match is reported either way
    r19 = rec.site_observed_et(tmp_path, "US-YYY", "2024-01", "2024-01", go26_version="19-5")
    assert r19["base_version"] == "19-5" and np.isclose(r19["mean_et_unclosed_valid"], 50 * C)
    latest = rec.site_observed_et(tmp_path, "US-YYY", "2024-01", "2024-01", "19-5", prefer_version="")
    assert latest["base_version"] == "20-5" and latest["version_matches_go26"] is False
    unknown = rec.site_observed_et(tmp_path, "US-YYY", "2024-01", "2024-01")
    assert unknown["base_version"] == "20-5" and unknown["version_matches_go26"] is pd.NA
    assert rec.site_observed_et(tmp_path, "US-NOP", "2024-01", "2024-01")["status"] == "BASE file not found"
    assert rec.site_observed_et(tmp_path, "US-YYY", "2022-01", "2022-01", "20-5")["status"] == \
        "no records in the study period"


def test_file_ending_before_the_period_and_mean_over_file_days(tmp_path, rec):
    """A file whose last record is on 20 June: the zero-filled mean over the calendar days and over the days up
    to that record (what a daily series built from the file covers)."""
    t = pd.date_range("2024-06-01", "2024-06-20 12:00", freq="30min")
    write_base(tmp_path / "AMF_US-END_BASE_HH_1-5.csv", t, {"LE": np.full(t.size, 100.0)}, "1-5", "US-END")
    res, daily = rec.reconstruct_site(tmp_path, "US-END", "2024-06", "2024-06", "1-5")
    total = 100 * 1800 / 2.45e6 * t.size
    assert (res["calendar_days"], res["file_days"]) == (30, 20)
    assert np.isclose(res["mean_et_hh_sum_zero_filled_unclosed"], total / 30)
    assert np.isclose(res["mean_et_hh_sum_zero_filled_file_days"], total / 20)
    assert "after_file_end" in daily.set_index("date").loc["2024-06-21", "flags"]
    assert "after_file_end" not in daily.set_index("date").loc["2024-06-20", "flags"]


def test_installation_gap_and_soil_heat_flux_fallbacks(tmp_path, rec):
    t = pd.date_range("2023-05-01", "2023-06-30 23:30", freq="30min")
    on = t >= "2023-05-31"  # tower installed on 31 May
    le = np.where(on, 80.0, np.nan)
    write_base(tmp_path / "AMF_US-UTD_BASE_HH_2-5.csv", t,
               {"LE": le, "H": np.where(on, 40.0, np.nan), "NETRAD": np.where(on, 150.0, np.nan),
                "G_PI_F": np.full(t.size, 30.0)}, "2-5", "US-UTD")
    res, daily = rec.reconstruct_site(tmp_path, "US-UTD", "2023-05", "2023-06", "2-5")
    assert res["base_first_valid_LE"] == "2023-05-31" and res["period_days_before_first_LE"] == 30
    assert res["g_source"].startswith("PI gap-filled") and res["valid_EB_days"] == 31
    assert np.isclose(res["mean_et_br_unscreened"], 80 * C * (150 - 30) / 120)
    assert "before_first_LE" in daily.set_index("date").loc["2023-05-30", "flags"]
    assert "G_PI_F" in daily.set_index("date").loc["2023-06-15", "flags"]
    # without any G the closure assumes G = 0 and flags it
    write_base(tmp_path / "AMF_US-WT1_BASE_HH_4-5.csv", t,
               {"LE": le, "H": np.where(on, 40.0, np.nan), "NETRAD": np.where(on, 150.0, np.nan)}, "4-5", "US-WT1")
    res2, daily2 = rec.reconstruct_site(tmp_path, "US-WT1", "2023-06", "2023-06")
    assert res2["g_source"].startswith("assumed 0") and np.isclose(res2["mean_et_br_unscreened"], 80 * C * 150 / 120)
    assert daily2["flags"].str.contains("G_zero").all() and "G = 0 in the closure" in res2["notes"]
    # without NETRAD: no closure at all
    write_base(tmp_path / "AMF_US-NOR_BASE_HH_1-5.csv", t, {"LE": le, "H": np.where(on, 40.0, np.nan)}, "1-5",
               "US-NOR")
    res3, daily3 = rec.reconstruct_site(tmp_path, "US-NOR", "2023-06", "2023-06")
    assert res3["valid_EB_days"] == 0 and np.isnan(res3["mean_et_ebr_mdv"]) and daily3["et_ebr_mdv"].isna().all()
    assert "no NETRAD or G in BASE: no closure" in res3["notes"] and np.isclose(res3["mean_et_unclosed_mdv"], 80 * C)
    # a site note
    write_base(tmp_path / "AMF_US-RGB_BASE_HH_5-5.csv", t, {"LE": le}, "5-5", "US-RGB")
    assert "flooded rice" in rec.site_observed_et(tmp_path, "US-RGB", "2023-06", "2023-06")["notes"]


def test_winter_check_counts_bowen_ratio_failures(tmp_path, rec):
    t = pd.date_range("2019-12-01", "2020-02-29 23:30", freq="30min")
    day = t.floor("D")
    le, h, rn, g = (np.full(t.size, v) for v in (50.0, 20.0, 100.0, 10.0))
    h[(day >= "2020-01-10") & (day <= "2020-01-12")] = -40.0     # LE + H = 10: factor 9
    h[(day >= "2020-02-05") & (day <= "2020-02-06")] = -60.0     # LE + H = -10: factor -9
    rn[(day >= "2019-12-24") & (day <= "2019-12-25")] = np.nan   # no closure possible
    gone = day >= "2020-02-20"
    for a in (le, h, rn, g):
        a[gone] = np.nan
    write_base(tmp_path / "AMF_US-UiC_BASE_HH_2-5.csv", t, {"LE": le, "H": h, "NETRAD": rn, "G": g}, "2-5", "US-UiC")
    res, daily = rec.reconstruct_site(tmp_path, "US-UiC", "2019-12", "2020-02", "2-5")
    assert (res["n_br_factor_gt2"], res["n_br_factor_negative"], res["n_br_factor_lt05"]) == (3, 2, 0)
    assert (res["n_br_screen_fail"], res["valid_EB_days_nov_mar"], res["n_br_screen_fail_nov_mar"]) == (5, 79, 5)
    assert (res["max_et_br_unscreened_date"], res["min_et_br_unscreened_date"]) == ("2020-01-10", "2020-02-05")
    assert np.isclose(res["max_et_br_unscreened"], 50 * C * 9) and np.isclose(res["min_et_br_unscreened"], -50 * C * 9)
    assert "EB_missing" in daily.set_index("date").loc["2019-12-24", "flags"]

    rows = {r["window"]: r for r in rec.winter_rows("US-UiC", daily, rec.GO26_FIG8_UIC)}
    assert set(rows) == {"DJF", "2020-01", "2020-02", "2020-01/02"}
    djf = rows["DJF"]
    assert (djf["days"], djf["valid_LE_days"], djf["valid_EB_days"]) == (91, 81, 79)
    assert (djf["n_br_factor_gt2"], djf["n_br_factor_negative"], djf["n_br_screen_fail"]) == (3, 2, 5)
    assert np.isclose(djf["mean_et_unclosed_valid"], 50 * C) and np.isclose(djf["mean_et_unclosed_mdv"], 50 * C)
    assert np.isclose(djf["mean_et_unclosed_mdv_on_EB_days"], 50 * C)
    assert np.isclose(djf["mean_et_br_unscreened"], 50 * C * (74 * 90 / 70 + 3 * 9 - 2 * 9) / 79)
    assert np.isclose(djf["max_et_br_unscreened"], 50 * C * 9)
    # screened and EBR closure replace the failing and missing days by their neighbours' factor 90/70
    assert np.isclose(djf["mean_et_ebr_mdv"], 50 * C * 90 / 70) and np.isclose(djf["mean_et_br_screened_mdv"],
                                                                                50 * C * 90 / 70)
    assert (rows["2020-02"]["days"], rows["2020-02"]["valid_LE_days"]) == (29, 19)
    assert rows["2020-01"]["go26_fig8_obs_mean"] == 0.62 and rows["2020-01/02"]["go26_fig8_obs_max"] == 2.3
    assert "go26_fig8_obs_mean" not in rec.winter_rows("US-ARM", daily, rec.GO26_FIG8_UIC)[0]

    dig = tmp_path / "digitized_US-UiC.csv"
    pd.DataFrame({"date": pd.date_range("2020-01-01", "2020-03-01"), "obs": 1.0}).assign(
        obs=lambda x: np.where(x["date"].dt.month == 2, 2.0, 1.0)).to_csv(dig, index=False)
    ref = rec.go26_fig8_reference(dig)
    assert ref["2020-01"]["go26_fig8_obs_days"] == 31 and ref["2020-02"]["go26_fig8_obs_mean"] == 2.0
    assert ref["2020-01/02"]["go26_fig8_obs_max"] == 2.0


# -- quality flags (--qc-flags) -----------------------------------------------------------------------
def _flagged_site(base: Path, site: str, with_flag: bool = True) -> None:
    """Half-hourly 1-3 June 2024; LE = 300 W m-2 on the records flagged 2, else 100.

    1 June: LE in all 48 records, 10 flagged 2; 2 June: LE in 30 records, 10 of them flagged 2;
    3 June: LE in all records, flag missing.
    """
    t = pd.date_range("2024-06-01", "2024-06-03 23:30", freq="30min")
    i, day = np.tile(np.arange(48), 3), t.day.to_numpy()
    bad = ((day == 1) & (i >= 10) & (i < 20)) | ((day == 2) & (i < 10))
    le = np.where(bad, 300.0, 100.0)
    le[(day == 2) & (i >= 30)] = np.nan
    cols = {"LE": le, "H": np.full(t.size, 50.0), "NETRAD": np.full(t.size, 200.0), "G": np.full(t.size, 20.0)}
    if with_flag:
        cols["LE_SSITC_TEST"] = np.where(bad, 2.0, np.where((day == 1) | ((day == 2) & (i < 30)), 0.0, np.nan))
    write_base(base / f"AMF_{site}_BASE_HH_1-5.csv", t, cols, "1-5", site)


def test_quality_flag_variant_removes_flagged_records(tmp_path, rec):
    _flagged_site(tmp_path, "US-QCA")
    _flagged_site(tmp_path, "US-QCB", with_flag=False)
    res, daily = rec.reconstruct_site(tmp_path, "US-QCA", "2024-06", "2024-06", "1-5", qc_flags=True)
    plain = rec.site_observed_et(tmp_path, "US-QCB", "2024-06", "2024-06", "1-5")
    per_hh = 1800 / 2.45e6  # mm per half-hour per W m-2
    # default: every record (Table 1 as reproduced); the flag column changes nothing
    assert all(same(res[k], plain[k]) for k in rec.LE_STATS)
    assert res["valid_LE_days"] == 3 and res["calendar_days"] == 30
    assert np.isclose(res["mean_et_unclosed_valid"], C * (6800 / 48 + 5000 / 30 + 100) / 3)
    # gap fill: 2 June's 18 missing records take the 100 W m-2 of 1 and 3 June at the same times
    assert np.isclose(res["mean_et_unclosed_mdv"], C * (6800 / 48 + 6800 / 48 + 100) / 3)
    assert np.isclose(res["mean_et_hh_sum_zero_filled_unclosed"], (6800 + 5000 + 4800) * per_hh / 30)
    # QC: the 20 records flagged 2 are removed before the day filter, the gap fill, the closure and the sums
    assert (res["le_flag_column"], res["le_records_flag2"], res["valid_LE_days_qc"], res["valid_EB_days_qc"]) == (
        "LE_SSITC_TEST", 20, 2, 2)
    assert np.isclose(res["le_record_availability_qc"], (38 + 20 + 48) / (30 * 48))
    assert np.isclose(res["mean_et_unclosed_valid_qc"], 100 * C) and np.isclose(res["mean_et_unclosed_mdv_qc"], 100 * C)
    assert np.isclose(res["mean_et_br_unscreened_qc"], 100 * C * (200 - 20) / (100 + 50))  # matched records only
    assert np.isclose(res["mean_et_hh_sum_zero_filled_unclosed_qc"], (3800 + 2000 + 4800) * per_hh / 30)
    out = rec.compare_with_go26(dict(res), 0.5)
    assert np.isclose(out["f_implied_qc"], 1 - 0.5 / (100 * C)) and np.isclose(out["f_implied_unfilled_qc"],
                                                                              out["f_implied_qc"])
    assert np.isclose(out["diff_mean_et_hh_sum_zero_filled_unclosed_qc"], 10600 * per_hh / 30 - 0.5)
    assert list(daily.columns) == rec.DAILY_COLUMNS + [f"{k}_qc" for k in rec.QC_DAILY]
    day = daily.set_index("date")
    assert np.allclose(day.loc["2024-06-01":"2024-06-03", "le_frac_qc"], [38 / 48, 20 / 48, 1.0])
    assert np.isclose(day.loc["2024-06-02", "et_unclosed"], 5000 / 30 * C)  # 30 of 48 records: a valid day
    assert np.isnan(day.loc["2024-06-02", "et_unclosed_qc"])                  # 20 of 48 pass the flag: dropped
    # no flag at the LE position: the QC variant equals the default, and the row says so
    b = rec.site_observed_et(tmp_path, "US-QCB", "2024-06", "2024-06", "1-5", qc_flags=True)
    assert (b["le_flag_column"], b["le_records_flag2"]) == ("", 0) and "no LE_SSITC_TEST" in b["notes"]
    assert all(same(b[f"{k}_qc"], b[k]) for k in rec.LE_STATS)


def test_main_reports_both_variants_with_qc_flags(tmp_path, rec, capsys):
    cfg, base, out = tmp_path / "config", tmp_path / "base", tmp_path / "results"
    cfg.mkdir()
    base.mkdir()
    pd.DataFrame({"site": ["US-QCA", "US-QCB"], "koppen_go26": ["Dfa", "BSk"], "period_start": "2024-06",
                  "period_end": "2024-06", "mean_obs_et_mm_d": [0.5, 0.6], "base_doi": ["10.1/a", "10.1/b"],
                  "base_version_go26": "1-5"}).to_csv(cfg / "go26_sites.csv", index=False)
    _flagged_site(base, "US-QCA")
    _flagged_site(base, "US-QCB", with_flag=False)
    args = ["--config-dir", str(cfg), "--base-dir", str(base), "--out-dir", str(out), "--openet", "",
            "--amf-public", ""]
    rec.main(args)
    plain = pd.read_csv(out / "go26_observed_et_reconstruction.csv")
    assert not any(c.endswith("_qc") for c in plain.columns) and "le_flag_column" not in plain
    assert "mean_et_hh_sum_zero_filled (all calendar days" in (out / "go26_reconstructed_winter.csv").read_text()
    assert "version check not written" in capsys.readouterr().out
    assert not (out / "go26_base_version_check.csv").exists()
    rec.main(args + ["--qc-flags"])
    assert "With LE removed where LE_SSITC_TEST = 2" in capsys.readouterr().out
    s = pd.read_csv(out / "go26_observed_et_reconstruction.csv")
    assert [c for c in s.columns if c not in rec.QC_COLUMNS] == list(plain.columns)  # default columns unchanged
    s = s.set_index("site")
    assert (s.loc["US-QCA", "valid_LE_days"], s.loc["US-QCA", "valid_LE_days_qc"]) == (3, 2)
    assert s.loc["US-QCB", "valid_LE_days_qc"] == s.loc["US-QCB", "valid_LE_days"] == 3
    daily = (out / "go26_reconstructed_daily.csv").read_text()
    assert "LE_SSITC_TEST = 2" in daily and "et_unclosed_qc" in pd.read_csv(out / "go26_reconstructed_daily.csv",
                                                                           comment="#")


# -- version and figure checks -------------------------------------------------------------------------
def test_version_check_against_the_availability_summary(rec):
    """The cited file runs to December 2024; the summary's (earlier) version ended in February 2024, where the study
    period ends; in 2023 the cited file has 10 more days of LE than the summary's version."""
    t = pd.date_range("2023-01-01", "2024-12-31 23:30", freq="30min")
    le = pd.Series(100.0, index=t)
    le[(t >= "2023-03-01") & (t < "2023-03-11")] = np.nan           # 10 days missing in 2023 in both versions
    summary = pd.Series({2022: 0.0, 2023: (365 - 10) / 365 - 10 / 365, 2024: 60 / 366, 2025: 0.0})
    rows = rec.version_check("US-VVV", le, "2023-01", "2024-02", "3-5", 2025, summary, "2024-10-20")
    years = {r["year"]: r for r in rows}
    assert list(years) == ["2023", "2024", "period"]
    y23, y24, site = years["2023"], years["2024"], years["period"]
    assert y23["records_in_year"] == y23["records_in_period"] == 365 * 48 and y23["summary_records_exact"]
    assert np.isclose(y23["share_le_cited"], 355 / 365) and np.isclose(y23["share_diff_cited_minus_summary"], 10 / 365)
    assert y23["le_records_in_period_cited"] == 355 * 48 and y23["le_records_in_period_summary_max"] == 345 * 48
    assert (y24["records_in_year"], y24["records_in_period"]) == (366 * 48, 60 * 48)
    assert not y24["summary_records_exact"]
    assert y24["share_le_cited"] == 1.0 and y24["le_records_in_period_summary_max"] == 60 * 48
    assert site["le_records_in_period_cited"] == (355 + 60) * 48
    assert site["le_records_in_period_summary_max"] == (345 + 60) * 48
    assert site["cited_version_newer_than_summary"] is True and site["summary_describes"].startswith("an earlier")
    assert (site["summary_last_le_year"], site["summary_le_end_month_min"], site["summary_le_end_month_prefix"],
            site["summary_le_end_month"]) == (2024, "2024-02", "2024-02", "2024-02")
    assert site["period_ends_where_summary_le_ends"] is True
    assert (site["cited_last_le_month"], site["cited_le_months_after_period_end"]) == ("2024-12", 10)
    # the same LE in every year and a citation from the summary's year: the summary describes the cited data
    same_le = pd.Series({2022: 0.0, 2023: 355 / 365, 2024: 1.0, 2025: 0.0})
    s2 = rec.version_check("US-VVV", le, "2023-01", "2024-12", "3-5", 2024, same_le, "2024-10-20")[-1]
    assert s2["cited_version_newer_than_summary"] is False and "same share in every year" in s2["summary_describes"]
    assert s2["summary_le_end_month"] == "2024-12" and s2["period_ends_where_summary_le_ends"] is True
    # the summary's version holds more records than the cited file that year: the lower bound is used
    more = pd.Series({2023: 355 / 365, 2024: 0.95, 2025: 0.0})
    cut = le[le.index < "2024-11-01"]
    s3 = rec.version_check("US-VVV", cut, "2023-01", "2024-12", "3-5", 2026, more, "2025-10-20")[-1]
    assert (s3["summary_le_end_month_prefix"], s3["summary_le_end_month_min"], s3["summary_le_end_month"]) == (
        "", "2024-12", "2024-12")


def test_figure_check_compares_go26_plots_with_the_zero_filled_sums(tmp_path, rec):
    days = pd.date_range("2024-01-01", "2024-01-10")
    table = pd.DataFrame({"site": "US-FIG", "date": days, "le_frac": [1, 1, 0.3, 0, 0, 0, 1, 1, 1, 1.0],
                          "et_hh_sum_zero_filled": [1.0, 2.0, 0.4, 0, 0, 0, 1.5, 1.5, 2.5, 3.0]})
    obs = [1.01, 2.0, 0.405, 0.0, 0.001, 0.8, 1.55, np.nan, 2.0, 3.0]  # 6 Jan plotted where BASE has no LE
    dig = tmp_path / "digitized_US-FIG.csv"
    pd.DataFrame({"date": days, "obs": obs, "obs_is_zero": [abs(v) < 0.015 if np.isfinite(v) else False for v in obs]}
                 ).to_csv(dig, index=False)
    r = rec.figure_check("US-FIG", "Fig. X", table, dig)
    assert (r["days_compared"], r["first_day"], r["last_day"]) == (9, "2024-01-01", "2024-01-10")
    assert (r["n_within_0p02"], r["n_within_0p1"], r["n_diff_ge_0p5"]) == (6, 7, 2)
    assert np.isclose(r["max_abs_diff"], 0.8) and r["max_abs_diff_date"] == "2024-01-06"
    assert (r["days_with_le"], r["n_plotted_lt50pct_records"], r["n_plotted_no_le"]) == (6, 1, 3)
    assert np.isclose(r["mean_abs_diff_lt50pct_records"], 0.005)
    assert (r["n_nonzero_no_le"], r["nonzero_no_le_dates"]) == (1, "2024-01-06")
    assert (r["n_exact_zero"], r["n_exact_zero_no_le"], r["exact_zero_days_all_no_le"]) == (2, 2, True)
    assert rec.date_ranges(["2024-07-27", "2024-07-26", "2024-07-28", "2024-08-06"]) == (
        "2024-07-26..2024-07-28;2024-08-06")
    assert rec.date_ranges([]) == ""


# -- OpenET bias --------------------------------------------------------------------------------------
def _daily_table(site, start, end, et, valid_days, eb_days=None, closure=1.0):
    """Daily table with the first n days of each month valid (valid_days, by 'YYYY-MM' or 'default') and a
    closure factor on the first eb_days of them; closed = et x closure."""
    dates = pd.date_range(f"{start}-01", pd.Period(end, "M").end_time.normalize(), freq="D")
    dom = dates.day.to_numpy()
    month = dates.strftime("%Y-%m")
    nv = np.array([valid_days.get(m, valid_days["default"]) for m in month])
    ne = np.array([(eb_days or {}).get(m, n) for m, n in zip(month, nv)])
    ok, eb = dom <= nv, dom <= ne
    unclosed = np.where(ok, et, np.nan)
    closed = np.where(eb & ok, et * closure, np.nan)
    return pd.DataFrame({"site": site, "date": dates, "le_frac": ok.astype(float), "eb_frac": (eb & ok).astype(float),
                         "et_unclosed": unclosed, "et_unclosed_mdv": unclosed, "et_ebr_mdv": closed,
                         "et_br_screened_mdv": closed, "et_hh_sum_zero_filled": np.where(ok, et, 0.0), "flags": ""})


def test_openet_bias_shows_the_zero_fill_effect(tmp_path, rec):
    a = _daily_table("US-AAA", "2021-01", "2021-12", 2.0, {"default": 25, "2021-07": 10}, {"2021-03": 15}, 1.25)
    b = _daily_table("US-BBB", "2021-01", "2021-06", 1.0, {"default": 31})
    monthly = rec.monthly_observed(pd.concat([a, b], ignore_index=True), 20)
    get_m = lambda s, m: monthly[(monthly["site"] == s) & (monthly["month"] == m)].iloc[0]  # noqa: E731
    jul = get_m("US-AAA", "2021-07-01")
    assert np.isnan(jul["obs_unclosed_mdv"]) and jul["obs_zero_filled"] == 20.0 and jul["season"] == "JJA"
    jan = get_m("US-AAA", "2021-01-01")
    assert np.isclose(jan["obs_unclosed_mdv"], 62.0) and np.isclose(jan["obs_ebr_mdv"], 77.5) and jan["season"] == "DJF"
    mar = get_m("US-AAA", "2021-03-01")  # closure on 15 days only: not a common valid month
    assert mar["n_valid"] == 15 and np.isnan(mar["obs_unclosed_mdv"]) and mar["obs_zero_filled"] == 50.0

    months = pd.date_range("2021-01-01", "2022-01-01", freq="MS")
    openet = pd.concat([pd.DataFrame({"site": "US-AAA", "date": months, "et_ensemble": 2.0 * months.days_in_month}),
                        pd.DataFrame({"site": "US-BBB", "date": months[:6],
                                      "et_ensemble": [31.0, 28.0, 31.0, 30.0, 31.0, np.nan]})])
    t3 = tmp_path / "go26_table3.csv"
    pd.DataFrame({"zone": ["BSk", "Dfa", "Overall (all sites)", "Overall (excl. BWk)"],
                  "openet_bias": [1.103, 0.483, 0.596, 0.593]}).to_csv(t3, index=False)
    reported = rec.go26_reported_biases(t3)
    assert reported[("BSk", "all")] == 1.103 and reported[("all", "all")] == 0.596
    assert reported[("all", "JJA")] == 1.342 and ("Overall (excl. BWk)", "all") not in reported
    # every statistic (min_months=0: nothing withheld, not for publication)
    bias = rec.openet_bias(monthly, openet, {"US-AAA": "BSk", "US-BBB": "Dfa"}, reported, min_months=0)
    assert "openet_withheld" not in bias and bias["openet_mm_d"].notna().all()
    assert set(bias["month_set"]) == {"all_months", "common_valid_months"}
    assert set(bias.loc[bias["month_set"].eq("all_months"), "variant"]) == {"zero_filled_table1"}
    get = lambda z, s, ms, v: bias[(bias["koppen_go26"] == z) & (bias["season"] == s)  # noqa: E731
                                   & (bias["month_set"] == ms) & (bias["variant"] == v)].iloc[0]
    unc = get("BSk", "all", "common_valid_months", "unclosed_mdv")
    assert unc["n_months"] == 10 and np.isclose(unc["bias_mm_d"], 0.0) and np.isnan(unc["go26_reported_bias_mm_d"])
    ebr = get("BSk", "all", "common_valid_months", "ebr_mdv")
    assert ebr["n_months"] == 10 and np.isclose(ebr["bias_mm_d"], -0.5)
    # the four variants of a common_valid_months row are on identical site-months
    common = bias[bias["month_set"].eq("common_valid_months")]
    assert common.groupby(["koppen_go26", "season"])["n_months"].nunique().eq(1).all()
    assert common.groupby(["koppen_go26", "season"]).size().eq(4).all()
    dim = months[:12].days_in_month.to_numpy()
    valid = np.array([25] * 6 + [10] + [25] * 5)
    zf_all = get("BSk", "all", "all_months", "zero_filled_table1")
    assert zf_all["n_months"] == 12 and np.isclose(zf_all["bias_mm_d"], np.mean(2.0 * (dim - valid) / dim))
    assert zf_all["go26_reported_bias_mm_d"] == 1.103
    zf_common = get("BSk", "all", "common_valid_months", "zero_filled_table1")
    keep = ~np.isin(np.arange(12), [2, 6])
    assert np.isclose(zf_common["bias_mm_d"], np.mean((2.0 * (dim - valid) / dim)[keep]))
    assert zf_all["bias_mm_d"] > zf_common["bias_mm_d"] > unc["bias_mm_d"]
    assert get("Dfa", "all", "common_valid_months", "unclosed_mdv")["n_months"] == 5  # NaN OpenET dropped
    assert get("all", "all", "all_months", "zero_filled_table1")["n_sites"] == 2
    assert get("all", "DJF", "common_valid_months", "unclosed_mdv")["n_months"] == 3 + 2  # Jan, Feb, Dec + B
    assert get("all", "JJA", "all_months", "zero_filled_table1")["go26_reported_bias_mm_d"] == 1.342
    # as published: the 5 Dfa months are withheld, their observed mean kept
    held = rec.openet_bias(monthly, openet, {"US-AAA": "BSk", "US-BBB": "Dfa"}, reported)
    dfa = held[(held["koppen_go26"] == "Dfa") & (held["season"] == "all")]
    assert dfa["withheld_rule"].eq("n_months").all() and dfa[["openet_mm_d", "bias_mm_d"]].isna().all().all()
    assert np.allclose(dfa["observed_mm_d"], 1.0) and dfa["n_months"].eq(5).all()


def _determined(rows: list[np.ndarray]) -> list[int]:
    """Site-months (columns) whose OpenET value the rows' means fix: unit vectors in the rows' span."""
    if not rows:
        return []
    a = np.array(rows, float)
    r = np.linalg.matrix_rank(a)
    return [j for j in range(a.shape[1]) if np.linalg.matrix_rank(np.vstack([a, np.eye(a.shape[1])[j]])) == r]


def _small_combinations(rows: list[np.ndarray], min_months: int = 6) -> list[tuple[int, ...]]:
    """Sets of fewer than ``min_months`` site-months (columns) on which the rows reveal a combination (sum,
    difference): brute force over column subsets (independent of scripts/10's atom search)."""
    if not rows:
        return []
    a = np.array(rows, float)
    a = a[:, a.any(axis=0)]
    r = np.linalg.matrix_rank(a)
    found = []
    for k in range(1, min_months):
        for u in itertools.combinations(range(a.shape[1]), k):
            if any(set(f) <= set(u) for f in found):
                continue
            rest = [j for j in range(a.shape[1]) if j not in u]
            if np.linalg.matrix_rank(a[:, rest]) < r:
                found.append(u)
    return found


def test_openet_bias_withholds_small_strata_and_their_complements(rec):
    """One site, every month valid; no OpenET in Dec 2021-Feb 2022 and Dec 2022 (snow): DJF has 2 months."""
    daily = _daily_table("US-AAA", "2021-03", "2023-02", 2.0, {"default": 31})
    months = pd.date_range("2021-03-01", "2023-02-01", freq="MS")
    keep = ~months.isin(pd.to_datetime(["2021-12-01", "2022-01-01", "2022-02-01", "2022-12-01"]))
    openet = pd.DataFrame({"site": "US-AAA", "date": months[keep], "et_ensemble": 40.0 + np.arange(keep.sum())})
    bias = rec.openet_bias(rec.monthly_observed(daily, 20), openet, {"US-AAA": "BSk"})
    rule = bias.groupby("season")["withheld_rule"].agg(set).to_dict()
    # DJF (2 months) is withheld; then the zone total minus MAM, JJA and SON would give it: the smallest
    # other season (6 months, MAM first) is withheld too, leaving 8 months to subtraction
    assert rule == {"DJF": {"n_months"}, "MAM": {"margin"}, "JJA": {""}, "SON": {""}, "all": {""}}
    held = bias["openet_withheld"]
    assert bias.loc[held, ["openet_mm_d", "bias_mm_d"]].isna().all().all()
    assert np.allclose(bias["observed_mm_d"], 2.0) and bias["n_months"].tolist()[:4] == [2] * 4
    shown = bias[~held]
    assert np.allclose(shown["bias_mm_d"], shown["openet_mm_d"] - shown["observed_mm_d"])
    assert set(shown["n_months"]) == {6, 20}
    # without the margin rule, all - JJA - SON - MAM would give the two DJF months: n_months alone is not enough
    zone = bias[bias["koppen_go26"] == "BSk"].drop_duplicates("season").set_index("season")["n_months"]
    assert zone["all"] - zone[["MAM", "JJA", "SON"]].sum() == 2


def test_withhold_support_rule_closes_small_differences(rec):
    """Two variants whose Z1 rows differ by two months (x, y): the 'difference' rule withholds variant b's Z1 row,
    but it is the difference of b's published total and Z2 row (8 months, so the margin rule lets it pass), and
    variant a's Z1 row minus it reveals x + y.  No single month is determined (x and y are always together), so
    the rank rule passes too; the support rule withholds rows of the combination, the smallest first, until no
    combination of fewer than 6 site-months is left."""
    s1 = [f"z1-{i}" for i in range(10)]
    s2 = [f"z2-{i}" for i in range(10)]
    t = [f"t-{i}" for i in range(6)]
    rows = [("Z1", "a", s1), ("Z2", "a", s2), ("all", "a", s1 + s2),
            ("Z1", "b", s1[2:]), ("Z2", "b", s2 + t), ("all", "b", s1[2:] + s2 + t)]
    bias = pd.DataFrame({"koppen_go26": [r[0] for r in rows], "season": "all", "month_set": "m",
                         "variant": [r[1] for r in rows], "n_sites": 1, "n_months": [len(r[2]) for r in rows],
                         "openet_mm_d": 1.0, "observed_mm_d": 1.0, "bias_mm_d": 0.0})
    out = rec.withhold_openet(bias, [frozenset(r[2]) for r in rows], 6)
    assert out["withheld_rule"].tolist() == ["support", "support", "", "difference", "", ""]
    cols = s1 + s2 + t
    vec = lambda m: np.array([c in m for c in cols], float)  # noqa: E731
    published = [vec(r[2]) for r, w in zip(rows, out["openet_withheld"]) if not w]
    assert _determined(published) == [] and _small_combinations(published) == []
    without_support = [vec(r[2]) for r, rule in zip(rows, out["withheld_rule"]) if rule in ("", "support")]
    assert _determined(without_support) == [] and _small_combinations(without_support) == [(0, 1)]


def test_main_withholds_openet_values_that_could_be_solved_for(tmp_path, rec, capsys):
    """Synthetic hourly BASE files: US-AAA (BSk) Mar-Sep 2021, US-BBB (Dfa) 2021; LE on every day, except on only 10
    days in Sep 2021 at US-AAA and in Jul-Dec 2021 at US-BBB, so those months are not common valid months."""
    cfg, base, out = tmp_path / "config", tmp_path / "base", tmp_path / "results"
    cfg.mkdir()
    base.mkdir()
    periods = {"US-AAA": ("2021-03", "2021-09", "BSk", [9]),  # start, end, zone, months with LE on 10 days
               "US-BBB": ("2021-01", "2021-12", "Dfa", [7, 8, 9, 10, 11, 12])}
    pd.DataFrame({"site": list(periods), "koppen_go26": [p[2] for p in periods.values()],
                  "period_start": [p[0] for p in periods.values()], "period_end": [p[1] for p in periods.values()],
                  "mean_obs_et_mm_d": [1.0, 1.0], "base_doi": ["10.1/a", "10.1/b"],
                  "base_version_go26": "1-5"}).to_csv(cfg / "go26_sites.csv", index=False)
    parts = []
    for k, (site, (start, end, _, short)) in enumerate(periods.items()):
        t = pd.date_range(f"{start}-01", pd.Period(end, "M").end_time.floor("h"), freq="h")
        full = np.ones(t.size)
        le = np.where(np.isin(t.month, short) & (t.day > 10), np.nan, 100.0)
        write_base(base / f"AMF_{site}_BASE_HR_1-5.csv", t,
                   {"LE": le, "H": 50 * full, "NETRAD": 200 * full, "G": 20 * full}, "1-5", site)
        m = pd.date_range(f"{start}-01", f"{end}-01", freq="MS")
        parts.append(pd.DataFrame({"site": site, "date": m, "et_ensemble": 30.0 + 7 * k + 5.0 * np.arange(m.size)}))
    openet = pd.concat(parts, ignore_index=True)
    openet.to_csv(tmp_path / "openet_monthly.csv", index=False)
    rec.main(["--config-dir", str(cfg), "--base-dir", str(base), "--out-dir", str(out),
              "--openet", str(tmp_path / "openet_monthly.csv"), "--amf-public", ""])
    assert "OpenET values withheld in" in capsys.readouterr().out
    text = (out / "go26_reconstructed_openet_bias.csv").read_text()
    assert "fewer than 6 site-months" in text and "no single month's OpenET value" in text and "(support)" in text
    bias = pd.read_csv(out / "go26_reconstructed_openet_bias.csv", comment="#")
    bias["withheld_rule"] = bias["withheld_rule"].fillna("")
    held = bias["openet_withheld"].to_numpy(bool)
    get = lambda z, ms, v: bias[(bias["koppen_go26"] == z) & (bias["season"] == "all")  # noqa: E731
                                & (bias["month_set"] == ms) & (bias["variant"] == v)].iloc[0]
    # US-AAA: 7 months, 6 of them common valid months: one month apart -> the row with fewer rows is withheld
    assert get("BSk", "all_months", "zero_filled_table1")["withheld_rule"] == "difference"
    assert get("BSk", "common_valid_months", "ebr_mdv")["n_months"] == 6
    assert get("Dfa", "common_valid_months", "unclosed_mdv")["n_months"] == 6
    assert set(bias["withheld_rule"]) <= {"", "n_months", "difference", "margin", "rank", "support"}
    assert bias.loc[held, ["openet_mm_d", "bias_mm_d"]].isna().all().all()
    assert bias[["n_sites", "n_months", "observed_mm_d"]].notna().all().all()
    assert bias.loc[bias["month_set"].ne("all_months"), "go26_reported_bias_mm_d"].isna().all()
    shown = bias[~held]
    assert np.allclose(shown["bias_mm_d"], shown["openet_mm_d"] - shown["observed_mm_d"], atol=2e-4)

    # independent check: rebuild every row's site-months from the published daily table and OpenET months
    daily = pd.read_csv(out / "go26_reconstructed_daily.csv", comment="#", parse_dates=["date"])
    d = daily.assign(month=daily["date"].dt.to_period("M").dt.to_timestamp(),
                     common=daily[["et_unclosed_mdv", "et_ebr_mdv", "et_br_screened_mdv"]].notna().all(axis=1))
    sm = d.groupby(["site", "month"]).agg(n=("common", "sum")).reset_index()
    sm = sm.merge(openet.rename(columns={"date": "month"})[["site", "month"]], on=["site", "month"])
    zone, season = sm["site"].map({s: p[2] for s, p in periods.items()}), sm["month"].dt.month.map(rec.SEASON)
    rows = []
    for r in bias.itertuples():
        sel = np.ones(len(sm), bool)
        if r.koppen_go26 != "all":
            sel &= zone.eq(r.koppen_go26).to_numpy()
        if r.season != "all":
            sel &= season.eq(r.season).to_numpy()
        if r.month_set == "common_valid_months":
            sel &= (sm["n"] >= 20).to_numpy()
        assert sel.sum() == r.n_months
        rows.append(sel.astype(float))
    published = [x for x, h in zip(rows, held) if not h]
    assert len(sm) == 19 and _determined(published) == [] and _small_combinations(published) == []
    # all rows published would give single months (US-AAA's September: zone totals minus common months)
    sep = np.flatnonzero((sm["site"] == "US-AAA") & (sm["month"] == "2021-09-01"))[0]
    assert sep in _determined(rows)


# -- command line ---------------------------------------------------------------------------------------
def _config(tmp_path):
    cfg = tmp_path / "config"
    cfg.mkdir()
    pd.DataFrame({"site": ["US-UiC", "US-UTJ"], "koppen_go26": ["Dfa", "BSk"],
                  "period_start": ["2020-01", "2024-01"], "period_end": ["2020-02", "2024-02"],
                  "mean_obs_et_mm_d": [0.5, 0.59],
                  "base_doi": ["10.17190/AMF/1846665", "10.17190/AMF/2531150"]}).to_csv(cfg / "go26_sites.csv",
                                                                                         index=False)
    return cfg


def test_main_runs_without_and_with_base_files(tmp_path, rec, capsys):
    cfg, base, out = _config(tmp_path), tmp_path / "base", tmp_path / "results"
    base.mkdir()
    args = ["--config-dir", str(cfg), "--base-dir", str(base), "--out-dir", str(out), "--amf-public", "",
            "--openet", ""]
    rec.main(args)
    s = pd.read_csv(out / "go26_observed_et_reconstruction.csv")
    assert list(s["status"]) == ["BASE file not found"] * 2
    assert list(s["base_version_go26"]) == ["2-5", "3-5"]  # GO26's reference list when config has none
    assert not (out / "go26_reconstructed_daily.csv").exists()
    assert "no base_version_go26 column" in capsys.readouterr().out

    t = pd.date_range("2020-01-01", "2020-02-29 23:30", freq="30min")
    le = np.where(t.day <= 20, 60.0, np.nan)
    write_base(base / "AMF_US-UiC_BASE_HH_3-5.csv", t,
               {"LE": le, "H": le / 2, "NETRAD": le * 2, "G": le / 10}, "3-5", "US-UiC")
    openet = tmp_path / "openet_monthly.csv"
    pd.DataFrame({"site": "US-UiC", "date": ["2020-01-01", "2020-02-01"], "et_ensemble": [40.0, 50.0]}).to_csv(
        openet, index=False)
    rec.main(args[:-1] + [str(openet)])
    printed = capsys.readouterr().out
    assert "BASE versions differing from GO26's: US-UiC 3-5 (GO26 2-5)" in printed
    s = pd.read_csv(out / "go26_observed_et_reconstruction.csv").set_index("site")
    assert s.loc["US-UiC", "status"] == "ok" and not s.loc["US-UiC", "version_matches_go26"]
    assert s.loc["US-UiC", "valid_LE_days"] == 40
    assert np.isclose(s.loc["US-UiC", "f_days_missing"], 20 / 60, atol=1e-4)  # CSV keeps 4 decimals
    assert s.loc["US-UTJ", "status"] == "BASE file not found"
    text = (out / "go26_reconstructed_daily.csv").read_text()
    assert text.startswith("# ") and "CC BY 4.0" in text and "https://doi.org/10.17190/AMF/1846665" in text
    daily = pd.read_csv(out / "go26_reconstructed_daily.csv", comment="#")
    assert list(daily.columns) == rec.DAILY_COLUMNS and len(daily) == 60
    winter = pd.read_csv(out / "go26_reconstructed_winter.csv", comment="#")
    assert list(winter["window"]) == ["DJF", "2020-01", "2020-02", "2020-01/02"]
    bias = pd.read_csv(out / "go26_reconstructed_openet_bias.csv", comment="#")
    row = bias[(bias["koppen_go26"] == "all") & (bias["season"] == "all")
               & (bias["month_set"] == "all_months")].iloc[0]
    # two months of OpenET: withheld; the observed mean (zero-filled sums of 20 days) and the counts stay
    assert (row["n_months"], row["withheld_rule"], bool(row["openet_withheld"])) == (2, "n_months", True)
    assert np.isnan(row["openet_mm_d"]) and np.isnan(row["bias_mm_d"])
    assert np.isclose(row["observed_mm_d"], np.mean([20 * 60 * C / 31, 20 * 60 * C / 29]), atol=1e-4)
    assert bias["openet_mm_d"].isna().all()
    header = (out / "go26_reconstructed_openet_bias.csv").read_text()
    assert "no single month's OpenET value" in header
    assert not (out / "go26_reconstructed_figure_check.csv").exists()  # no digitized series in this folder


def test_main_writes_the_version_and_figure_checks(tmp_path, rec, monkeypatch, capsys):
    """With the (pinned) availability summary and GO26's digitized series in place, main writes both checks."""
    cfg, base, out, amf = _config(tmp_path), tmp_path / "base", tmp_path / "results", tmp_path / "amf"
    for p in (base, out, amf):
        p.mkdir()
    t = pd.date_range("2020-01-01", "2020-03-31 23:30", freq="30min")  # the file runs a month past the period
    le = np.where(t.day <= 20, 60.0, np.nan)
    write_base(base / "AMF_US-UiC_BASE_HH_2-5.csv", t, {"LE": le}, "2-5", "US-UiC")
    csv = ("create_date:2020-12-01\nSITE_ID,VARIABLE,BASENAME,GAP_FILLED,Y2019,Y2020\n"
           f"US-UiC,LE,LE,FALSE,0,{40 / 366}\n")
    path = amf / rec.amfp.availability_file(rec.amfp.PINNED_SHA)
    path.write_text(csv)
    monkeypatch.setattr(rec.amfp, "PINNED_MD5", hashlib.md5(csv.encode()).hexdigest())
    days = pd.date_range("2020-01-01", "2020-02-29")  # GO26's plot: the zero-filled sums, zero without LE
    pd.DataFrame({"date": days, "obs": 60 * C * (days.day <= 20), "obs_is_zero": days.day > 20}).to_csv(
        out / "digitized_US-UiC.csv", index=False)
    rec.main(["--config-dir", str(cfg), "--base-dir", str(base), "--out-dir", str(out), "--openet", "",
              "--amf-public", str(amf)])
    printed = capsys.readouterr().out
    assert "GO26's period ends in the month the summary's LE ends at 1 of 1 sites" in printed
    vc = pd.read_csv(out / "go26_base_version_check.csv", comment="#", dtype={"year": str})
    site = vc[vc["year"].eq("period")].iloc[0]
    assert (site["summary_le_end_month"], site["cited_last_le_month"], site["cited_le_months_after_period_end"]) == (
        "2020-02", "2020-03", 1)
    assert site["le_records_in_period_cited"] == 40 * 48 and bool(site["period_ends_where_summary_le_ends"])
    fc = pd.read_csv(out / "go26_reconstructed_figure_check.csv", comment="#").iloc[0]
    assert (fc["site"], fc["figure"], fc["days_compared"], fc["n_within_0p02"]) == ("US-UiC", "Fig. 8", 60, 60)
    assert (fc["n_exact_zero"], fc["n_exact_zero_no_le"], bool(fc["exact_zero_days_all_no_le"])) == (20, 20, True)


def test_download_flag_requires_policy_agreement(tmp_path, rec, monkeypatch):
    monkeypatch.setattr(af, "credentials", lambda *a: pytest.fail("credentials read without --agree-policy"))
    with pytest.raises(SystemExit):
        rec.main(["--download", "--config-dir", str(tmp_path), "--out-dir", str(tmp_path)])


# -- downloader (mocked HTTP) -----------------------------------------------------------------------------
USER, EMAIL = "tester-id-123", "tester@example.org"


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code, self._payload, self._content = status, payload, content

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload

    def iter_content(self, n):
        for i in range(0, len(self._content), n):
            yield self._content[i:i + n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Links only for the requested sites, as the data_download service; GETs of ``fail`` raise."""

    def __init__(self, files: dict[str, bytes], post_status=200, get_status=200, fail=()):
        self.files, self.post_status, self.get_status, self.fail = files, post_status, get_status, set(fail)
        self.posts, self.gets = [], []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "json": json, "headers": headers})
        links = [{"site_id": n[4:10], "url": f"https://example.org/BASE/{n}?={USER}"} for n in self.files
                 if n[4:10] in json["site_ids"]]
        return FakeResponse(self.post_status, {"data_urls": links})

    def get(self, url, stream=False, timeout=None):
        self.gets.append(url)
        name = Path(urlsplit(url).path).name
        if name in self.fail:
            raise requests.ConnectionError("connection reset")
        return FakeResponse(self.get_status, content=self.files[name])


def _zip_bytes(csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("AMF_US-ZZZ_BASE_HH_3-5.csv", csv_text)
        z.writestr("AMF_US-ZZZ_BIF_20260101.xlsx", b"not a spreadsheet")
        z.writestr("../evil.csv", "x")
        z.writestr("nested/AMF_US-ZZZ_BASE_HR_3-5.csv", "# Site: US-ZZZ\n")
    return buf.getvalue()


def _site_zip(site: str, base_csv: bool = True) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"AMF_{site}_BIF_20260101.xlsx", b"not a spreadsheet")
        if base_csv:
            z.writestr(f"AMF_{site}_BASE_HH_3-5.csv", f"# Site: {site}\nTIMESTAMP_START,LE\n202401010000,10\n")
    return buf.getvalue()


def test_download_base_request_extraction_and_provenance(tmp_path, capsys):
    csv_text = "# Site: US-ZZZ\n# Version: 3-5\nTIMESTAMP_START,TIMESTAMP_END,LE\n202401010000,202401010030,10\n"
    blob = _zip_bytes(csv_text)
    http = FakeSession({"AMF_US-ZZZ_BASE-BADM_3-5.zip": blob})
    out = tmp_path / "ameriflux_base"
    entries = af.download_base(["US-ZZZ"], out, USER, EMAIL, agree_policy=True, session=http)
    assert http.posts == [{"url": "https://amfcdn.lbl.gov/api/v1/data_download",
                           "headers": {"Content-Type": "application/json"},
                           "json": {"user_id": USER, "user_email": EMAIL, "data_product": "BASE-BADM",
                                    "data_policy": "CCBY4.0", "site_ids": ["US-ZZZ"],
                                    "intended_use": "Research - Remote sensing",
                                    "description": af.DEFAULT_DESCRIPTION, "is_test": ""}}]
    assert len(http.gets) == 1
    assert sorted(p.name for p in out.iterdir()) == ["AMF_US-ZZZ_BASE-BADM_3-5.zip", "AMF_US-ZZZ_BASE_HH_3-5.csv",
                                                     "AMF_US-ZZZ_BASE_HR_3-5.csv", "provenance.json"]
    assert not (tmp_path / "evil.csv").exists()
    assert (out / "AMF_US-ZZZ_BASE_HH_3-5.csv").read_text() == csv_text
    prov_text = (out / "provenance.json").read_text()
    prov = json.loads(prov_text)
    e = prov["files"][0]
    assert entries == prov["files"]
    assert (e["site"], e["zip"], e["version"]) == ("US-ZZZ", "AMF_US-ZZZ_BASE-BADM_3-5.zip", "3-5")
    assert e["url"] == "https://example.org/BASE/AMF_US-ZZZ_BASE-BADM_3-5.zip"
    assert e["md5"] == hashlib.md5(blob).hexdigest() and e["bytes"] == len(blob)
    assert datetime.fromisoformat(e["accessed_utc"]).utcoffset().total_seconds() == 0
    assert e["base_csv"][0] == {"file": "AMF_US-ZZZ_BASE_HH_3-5.csv", "resolution": "HH", "version": "3-5",
                                "md5": hashlib.md5(csv_text.encode()).hexdigest()}
    assert prov["data_policy"] == "CCBY4.0"
    printed = capsys.readouterr().out
    for secret in (USER, EMAIL):
        assert secret not in prov_text and secret not in printed

    assert "without its ' [amerifluxr download]' description tag" in prov["source"]

    # reruns use the cache: no request for a site that has a BASE file or whose zip is on disk
    assert af.download_base(["US-ZZZ"], out, USER, EMAIL, agree_policy=True, session=http) == []
    assert len(http.posts) == 1
    (out / "AMF_US-ZZZ_BASE_HH_3-5.csv").unlink()
    (out / "AMF_US-ZZZ_BASE_HR_3-5.csv").unlink()
    again = af.download_base(["US-ZZZ"], out, USER, EMAIL, agree_policy=True, session=http)
    assert (len(http.posts), len(http.gets)) == (1, 1) and again == prov["files"]  # same record, same access time
    assert (out / "AMF_US-ZZZ_BASE_HH_3-5.csv").read_text() == csv_text


def test_download_base_records_each_zip_before_a_later_failure(tmp_path):
    zips = {f"AMF_{s}_BASE-BADM_3-5.zip": _site_zip(s) for s in ("US-AAA", "US-BBB")}
    http = FakeSession(zips, fail={"AMF_US-BBB_BASE-BADM_3-5.zip"})
    out = tmp_path / "base"
    with pytest.raises(af.AmeriFluxError, match="failed") as err:
        af.download_base(["US-AAA", "US-BBB"], out, USER, EMAIL, agree_policy=True, session=http, log=lambda m: None)
    assert USER not in str(err.value) and EMAIL not in str(err.value)
    prov = json.loads((out / "provenance.json").read_text())
    assert [e["zip"] for e in prov["files"]] == ["AMF_US-AAA_BASE-BADM_3-5.zip"]  # written before the failure
    assert (out / "AMF_US-AAA_BASE_HH_3-5.csv").exists() and not list(out.glob("AMF_US-BBB*"))
    first = prov["files"][0]
    http.fail = set()
    af.download_base(["US-AAA", "US-BBB"], out, USER, EMAIL, agree_policy=True, session=http, log=lambda m: None)
    assert http.posts[-1]["json"]["site_ids"] == ["US-BBB"]  # only the failed site is requested again
    prov = json.loads((out / "provenance.json").read_text())
    assert [e["zip"] for e in prov["files"]] == [*zips] and prov["files"][0] == first


def test_download_base_does_not_request_a_cached_zip_again(tmp_path):
    http = FakeSession({"AMF_US-UTJ_BASE-BADM_3-5.zip": _site_zip("US-UTJ", base_csv=False)})
    printed = []
    for _ in range(3):  # a zip without a BASE CSV is reported, not requested again
        af.download_base(["US-UTJ"], tmp_path, USER, EMAIL, agree_policy=True, session=http, log=printed.append)
    assert (len(http.posts), len(http.gets)) == (1, 1) and "no BASE CSV inside" in " ".join(printed)
    assert json.loads((tmp_path / "provenance.json").read_text())["files"][0]["base_csv"] == []
    af.download_base(["US-UTJ"], tmp_path, USER, EMAIL, agree_policy=True, session=http, force=True,
                     log=printed.append)
    assert (len(http.posts), len(http.gets)) == (2, 2)


def test_download_base_keeps_the_request_that_fetched_a_zip(tmp_path):
    """A zip extracted again without a new request keeps the intended use and description sent with the request
    that fetched it (null when unknown); only zips fetched after this call's request get this call's values."""
    zips = {"AMF_US-AAA_BASE-BADM_3-5.zip": _site_zip("US-AAA")}
    http = FakeSession(zips)
    out, quiet = tmp_path / "base", (lambda m: None)
    first = af.download_base(["US-AAA"], out, USER, EMAIL, agree_policy=True, session=http, log=quiet,
                             intended_use="remote_sensing", description="first request")
    assert [(e["intended_use"], e["description"]) for e in first] == [("Research - Remote sensing", "first request")]
    (out / "AMF_US-AAA_BASE_HH_3-5.csv").unlink()
    again = af.download_base(["US-AAA"], out, USER, EMAIL, agree_policy=True, session=http, log=quiet,
                             intended_use="other_research", description="a later rerun, nothing requested")
    assert len(http.posts) == 1 and again == first == json.loads((out / "provenance.json").read_text())["files"]

    hand = tmp_path / "by_hand"  # a zip copied in by hand: nothing is known about its request
    hand.mkdir()
    (hand / "AMF_US-AAA_BASE-BADM_3-5.zip").write_bytes(zips["AMF_US-AAA_BASE-BADM_3-5.zip"])
    e = af.download_base(["US-AAA"], hand, USER, EMAIL, agree_policy=True, session=http, log=quiet,
                         description="a later rerun, nothing requested")[0]
    assert (e["url"], e["accessed_utc"], e["intended_use"], e["description"]) == (None, None, None, None)
    assert len(http.posts) == 1 and e["md5"] == first[0]["md5"]

    # a link returned by a new request whose zip is already on disk (a name not recognised as a cached zip)
    odd = "AMF_US-BBB_BASE-BADM_3-5-r1.zip"
    http = FakeSession({odd: _site_zip("US-BBB"), "AMF_US-CCC_BASE-BADM_3-5.zip": _site_zip("US-CCC")})
    d = tmp_path / "odd"
    old = af.download_base(["US-BBB"], d, USER, EMAIL, agree_policy=True, session=http, log=quiet,
                           description="first request")[0]
    (d / "AMF_US-BBB_BASE_HH_3-5.csv").unlink()
    got = {e["site"]: e for e in af.download_base(["US-BBB", "US-CCC"], d, USER, EMAIL, agree_policy=True,
                                                  session=http, log=quiet, intended_use="other_research",
                                                  description="second request")}
    assert [p["json"]["description"] for p in http.posts] == ["first request", "second request"]
    assert len(http.gets) == 2  # US-BBB's zip was not fetched again
    b, c = got["US-BBB"], got["US-CCC"]
    assert (b["accessed_utc"], b["intended_use"], b["description"]) == (
        old["accessed_utc"], "Research - Remote sensing", "first request")
    assert (c["intended_use"], c["description"]) == ("Research - Other", "second request") and c["accessed_utc"]


def test_download_base_refuses_without_policy_and_hides_credentials(tmp_path):
    http = FakeSession({"AMF_US-ZZZ_BASE-BADM_3-5.zip": _zip_bytes("x")})
    with pytest.raises(af.AmeriFluxError, match="data policy"):
        af.download_base(["US-ZZZ"], tmp_path, USER, EMAIL, session=http)
    assert http.posts == []
    with pytest.raises(ValueError):
        af.download_base(["not a site"], tmp_path, USER, EMAIL, agree_policy=True, session=http)
    with pytest.raises(ValueError):
        af.download_base(["US-ZZZ"], tmp_path, USER, EMAIL, agree_policy=True, intended_use="fun", session=http)
    for kw in ({"post_status": 500}, {"get_status": 403}):
        bad = FakeSession(http.files, **kw)
        with pytest.raises(af.AmeriFluxError) as err:
            af.download_base(["US-ZZZ"], tmp_path / "x", USER, EMAIL, agree_policy=True, session=bad)
        msg = str(err.value)
        assert USER not in msg and EMAIL not in msg and "?=" not in msg
    assert not any((tmp_path / "x").glob("*.zip*"))  # no partial download left behind


def test_safe_url_and_credentials(tmp_path, monkeypatch):
    assert af.safe_url(f"https://u:p@host.org:8443/a/f.zip?={USER}&k=v#x", (USER,)) == "https://host.org:8443/a/f.zip"
    assert af.safe_url("https://host.org/tester%40example.org/f.zip", (EMAIL,)) == "https://host.org/<redacted>/f.zip"
    monkeypatch.delenv("AMERIFLUX_USER_ID", raising=False)
    monkeypatch.delenv("AMERIFLUX_USER_EMAIL", raising=False)
    with pytest.raises(af.AmeriFluxError, match="AMERIFLUX_USER_ID and AMERIFLUX_USER_EMAIL"):
        af.credentials(tmp_path)
    (tmp_path / ".env").write_text(
        f"# AmeriFlux\nexport AMERIFLUX_USER_ID='{USER}'\nAMERIFLUX_USER_EMAIL=\"{EMAIL}\"\n")
    assert af.credentials(tmp_path) == (USER, EMAIL)
    monkeypatch.setenv("AMERIFLUX_USER_ID", "from-env")
    assert af.credentials(tmp_path) == ("from-env", EMAIL)  # environment first
    monkeypatch.setenv("AMERIFLUX_USER_EMAIL", "not-an-address")
    with pytest.raises(af.AmeriFluxError) as err:
        af.credentials(tmp_path)
    assert "not-an-address" not in str(err.value)


def test_credentials_env_comments_quotes_and_whitespace(tmp_path, monkeypatch):
    monkeypatch.delenv("AMERIFLUX_USER_ID", raising=False)
    monkeypatch.delenv("AMERIFLUX_USER_EMAIL", raising=False)
    env = tmp_path / ".env"
    # the README template filled in, inline comments kept
    env.write_text("AMERIFLUX_USER_ID=jdoe       # scripts/10 --download (optional)\n"
                   "AMERIFLUX_USER_EMAIL=jdoe@example.org    # scripts/10 --download (optional)\n")
    assert af.credentials(tmp_path) == ("jdoe", "jdoe@example.org")
    env.write_text("export AMERIFLUX_USER_ID='jd#oe'  # quoted: verbatim\nAMERIFLUX_USER_EMAIL=\"jdoe@example.org\"\n")
    assert af.credentials(tmp_path) == ("jd#oe", "jdoe@example.org")
    env.write_text("AMERIFLUX_USER_ID=  # fill in\nAMERIFLUX_USER_EMAIL=jdoe@example.org\n")
    with pytest.raises(af.AmeriFluxError, match="set AMERIFLUX_USER_ID"):
        af.credentials(tmp_path)
    for bad in ("'jane doe'", "jane doe", "'jane", "'jane' doe"):  # whitespace, unmatched quote, trailing text
        env.write_text(f"AMERIFLUX_USER_ID={bad}\nAMERIFLUX_USER_EMAIL=jdoe@example.org\n")
        with pytest.raises(af.AmeriFluxError, match="AMERIFLUX_USER_ID") as err:
            af.credentials(tmp_path)
        assert "jane" not in str(err.value) and "jdoe" not in str(err.value)
    env.write_text("AMERIFLUX_USER_ID=jdoe\nAMERIFLUX_USER_EMAIL=jdoe@example\n")
    with pytest.raises(af.AmeriFluxError, match="not an e-mail address") as err:
        af.credentials(tmp_path)
    assert "jdoe" not in str(err.value)
    monkeypatch.setenv("AMERIFLUX_USER_ID", "jane doe")
    with pytest.raises(af.AmeriFluxError, match=r"AMERIFLUX_USER_ID \(environment\) contains whitespace") as err:
        af.credentials(tmp_path)
    assert "jane" not in str(err.value)
