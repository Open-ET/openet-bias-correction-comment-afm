"""Tests for etcomment.amf_public and scripts/11 (synthetic data, mocked HTTP, no network)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests

from etcomment import ameriflux as af
from etcomment import amf_public as amf

SHA = "0123456789abcdef0123456789abcdef01234567"
CSV = (b"create_date:2025-10-20\nSITE_ID,VARIABLE,BASENAME,GAP_FILLED,Y2024,Y2025\n"
       b"US-AAA,LE,LE,FALSE,0.5,0.25\nUS-AAA,LE_PI_F,LE,TRUE,1,1\n")
PAGE = """<html><head><script>var t = "AmeriFlux BASE US-XXX Decoy, Ver. 99-9";</script></head><body>
<div class="clear panel" id="data-use-log-p"><div><p>AmeriFlux BASE US-XXX Someone, Ver. 7-7</p>
<div>Years Data Available: AmeriFlux BASE 1990 - 1991</div></div></div>
<table><tr><td>Years Data Collected:</td><td>
    2023 - Present </td></tr>
<tr><td>Years Data Available:</td><td><p>AmeriFlux BASE 2023 - 2025 &nbsp; <a href="#data-citation">Data
Citation</a></p><p>AmeriFlux FLUXNET 2023 - 2024</p></td></tr></table>
<div id="doilist"><ul><li class="major"><strong>AmeriFlux BASE:</strong>
<a href="https://doi.org/10.17190/AMF/1234567">https://doi.org/10.17190/AMF/1234567</a><br>
<strong>Citation:</strong> Jane Doe, John Roe (2026), AmeriFlux BASE US-XXX Kress, TX no-till cotton,
Ver. 4-5, AmeriFlux AMP, (Dataset). https://doi.org/10.17190/AMF/1234567</li>
<li class="major"><strong>Citation:</strong> Jane Doe (2026), AmeriFlux FLUXNET-1F US-XXX Kress, Ver. v1.3_r1
(2023-2024), AmeriFlux AMP</li></ul></div></body></html>"""


class _Resp:
    def __init__(self, content: bytes, url: str, status: int = 200):
        self.content, self.url, self.status_code = content, url, status

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class _Session:
    """Serves canned bodies by URL substring and records every request."""

    def __init__(self, routes: dict[str, bytes]):
        self.routes, self.calls = routes, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers))
        for key, body in self.routes.items():
            if key in url:
                return _Resp(body, url)
        return _Resp(b"", url, 404)


class _Offline:
    def get(self, url, headers=None, timeout=None):
        raise requests.ConnectionError("offline")


class _NoNetwork:
    def get(self, url, headers=None, timeout=None):
        raise AssertionError(f"network used: {url}")


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch):
    monkeypatch.setattr(amf, "MIN_INTERVAL_S", 0.0)


@pytest.fixture
def s11():
    spec = importlib.util.spec_from_file_location(
        "s11", Path(__file__).resolve().parents[1] / "scripts" / "11_ameriflux_availability.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _table() -> pd.DataFrame:
    return pd.DataFrame({
        "SITE_ID": ["US-AAA"] * 5 + ["US-BBB"],
        "VARIABLE": ["LE_1_1_1", "LE_1_2_1", "LE_PI_F_1_1_1", "LE_SSITC_TEST_1_1_1", "H_1_1_1", "LE"],
        "BASENAME": ["LE", "LE", "LE", "LE_SSITC_TEST", "H", "LE"],
        "GAP_FILLED": ["FALSE", "FALSE", "TRUE", "FALSE", "FALSE", "FALSE"],
        "Y2023": [0.0, 0.0, 1.0, 0.9, 0.9, 0.1],
        "Y2024": [0.5, 0.7, 1.0, 0.9, 0.9, 0.2],
        "Y2025": [0.8, 0.2, 1.0, 0.9, 0.9, 0.3],
    })


def test_le_availability_uses_one_position_like_read_base():
    t = _table()
    by = amf.le_availability(t, "US-AAA")
    assert list(by.index) == [2023, 2024, 2025]
    # LE_1_1_1 has more valid records (0.5 x 366 + 0.8 x 365 days) than LE_1_2_1 (0.7 x 366 + 0.2 x 365): its
    # values are used, not the per-year maximum over positions (0.7 in 2024); LE_PI_F and LE_SSITC_TEST ignored
    assert np.allclose(by.to_numpy(), [0.0, 0.5, 0.8])
    assert amf.le_variables(t, "US-AAA") == ["LE_1_1_1", "LE_1_2_1"]
    assert np.allclose(amf.le_availability(t, "US-BBB").to_numpy(), [0.1, 0.2, 0.3])
    assert amf.le_availability(t, "US-ZZZ").isna().all()
    tb = t.assign(GAP_FILLED=t["GAP_FILLED"] == "TRUE")  # boolean column, as read from the CSV
    assert amf.le_availability(tb, "US-AAA").equals(by)
    # replicates (third index) at the chosen position: per-year maximum
    rep = pd.concat([t, pd.DataFrame({"SITE_ID": ["US-AAA"], "VARIABLE": ["LE_1_1_2"], "BASENAME": ["LE"],
                                      "GAP_FILLED": ["FALSE"], "Y2023": [0.3], "Y2024": [0.4], "Y2025": [0.9]})])
    assert np.allclose(amf.le_availability(rep, "US-AAA").to_numpy(), [0.3, 0.5, 0.9])
    # equal valid records: the unqualified variable wins (as in read_base); names read_base skips are ignored
    tie = pd.DataFrame({"SITE_ID": "US-CCC", "VARIABLE": ["LE_1_1_1", "LE", "LE_PI_1_1_A"], "BASENAME": "LE",
                        "GAP_FILLED": "FALSE", "Y2023": [0.25, 0.75, 1.0], "Y2024": [0.5, 0.5, 1.0],
                        "Y2025": [0.75, 0.25, 1.0]})
    assert np.allclose(amf.le_availability(tie, "US-CCC").to_numpy(), [0.75, 0.5, 0.25])
    assert amf.le_variables(tie, "US-CCC") == ["LE", "LE_1_1_1"]


def test_le_availability_matches_what_read_base_reads(tmp_path):
    """A BASE file and its availability table: same position and per-year availability as read_base."""
    t = pd.date_range("2023-01-01", "2024-12-31 21:00", freq="3h")
    first = {y: np.flatnonzero(t.year == y) for y in (2023, 2024)}

    def valid(shares):  # the first share x records of each year are valid
        ok = np.zeros(t.size, bool)
        for y, share in shares.items():
            ok[first[y][:int(round(share * first[y].size))]] = True
        return np.where(ok, 100.0, np.nan)

    cols = {"LE_1_1_1": valid({2023: 0.75}), "LE_1_1_2": valid({2024: 0.5}),   # one position, two replicates
            "LE_1_2_1": valid({2023: 0.25, 2024: 0.75})}                       # another level
    df = pd.DataFrame({"TIMESTAMP_START": t.strftime("%Y%m%d%H%M"), **{k: np.nan_to_num(v, nan=-9999) for k, v in
                                                                         cols.items()}})
    path = tmp_path / "AMF_US-AAA_BASE_HH_1-5.csv"
    path.write_text("# Site: US-AAA\n# Version: 1-5\n" + df.to_csv(index=False))
    table = pd.DataFrame({"SITE_ID": "US-AAA", "VARIABLE": list(cols), "BASENAME": "LE", "GAP_FILLED": False,
                          **{f"Y{y}": [float(np.isfinite(v[first[y]]).mean()) for v in cols.values()]
                             for y in (2023, 2024)}})
    hh = af.read_base(path)
    assert hh.attrs["columns"]["LE"] == ["LE_1_1_1", "LE_1_1_2"]
    seen = hh["LE"].notna().groupby(hh.index.year).mean()
    by = amf.le_availability(table, "US-AAA")
    assert np.allclose(by.to_numpy(), seen.to_numpy()) and np.allclose(by.to_numpy(), [0.75, 0.5])
    assert table[["Y2023", "Y2024"]].max().tolist() == [0.75, 0.75]  # the old maximum over all LE overstated 2024


def test_parse_variable_availability_reads_create_date():
    t, create_date = amf.parse_variable_availability(CSV.decode())
    assert create_date == "2025-10-20" and t["GAP_FILLED"].tolist() == [False, True]
    assert amf.le_availability(t, "US-AAA").to_dict() == {2024: 0.5, 2025: 0.25}


def test_period_availability_full_years_and_leap_year():
    by = pd.Series({2020: 0.5, 2021: 1.0})
    p = amf.period_availability(by, "2020-01", "2021-12")
    assert p["calendar_days"] == 366 + 365 and p["n_full_years"] == 2 and p["n_years_capped"] == 0
    assert np.isclose(p["le_avail_period"], (0.5 * 366 + 365) / 731)
    assert np.isclose(p["le_avail_full_years"], p["le_avail_period"])
    t = amf.period_years(by, "2020-01", "2021-12")
    assert t["days_in_year"].tolist() == [366, 365] and np.isclose(t.loc[0, "days_available"], 183.0)


def test_period_availability_partial_final_year_and_clipping():
    # 2024 (leap) in full plus Jan-Feb 2025 (59 days): 0.1 x 365 = 36.5 days fit inside the period
    p = amf.period_availability(pd.Series({2024: 0.25, 2025: 0.1}), "2024-01", "2025-02")
    assert p["calendar_days"] == 366 + 59 and p["n_years_capped"] == 0
    assert np.isclose(p["le_avail_period"], (0.25 * 366 + 36.5) / 425)
    assert np.isclose(p["le_avail_full_years"], 0.25) and p["n_full_years"] == 1
    # 0.2 x 365 = 73 days exceed the 59 days of 2025 in the period: capped
    p = amf.period_availability(pd.Series({2024: 0.25, 2025: 0.2}), "2024-01", "2025-02")
    assert p["n_years_capped"] == 1 and np.isclose(p["le_avail_period"], (0.25 * 366 + 59) / 425)
    t = amf.period_years(pd.Series({2024: 0.25, 2025: 0.2}), "2024-01", "2025-02")
    assert t["capped"].tolist() == [False, True] and t["days_in_period"].tolist() == [366, 59]
    # a single month of a leap year (29 days); no full year
    p = amf.period_availability(pd.Series({2024: 0.05}), "2024-02", "2024-02")
    assert p["calendar_days"] == 29 and np.isclose(p["le_avail_period"], 0.05 * 366 / 29)
    assert np.isnan(p["le_avail_full_years"])
    # a year without an availability value makes the period unknown
    assert np.isnan(amf.period_availability(pd.Series({2024: 0.5}), "2024-01", "2025-01")["le_avail_period"])


def test_year_ranges():
    assert amf.year_ranges([2005, 2001, 2002, 2003, 2004, 2007, 2009, 2010]) == "2001-2005,2007,2009-2010"
    assert amf.year_ranges([]) == "" and amf.year_ranges([2024]) == "2024"


def test_parse_site_page_reads_version_and_years_and_skips_data_use_log():
    info = amf.parse_site_page(PAGE, "US-XXX")
    assert info["base_version"] == "4-5" and info["base_citation_year"] == 2026
    assert info["base_name"] == "Kress, TX no-till cotton" and info["base_doi"] == "10.17190/AMF/1234567"
    assert info["base_years_available"] == "2023-2025"
    assert (info["base_first_year"], info["base_last_year"]) == (2023, 2025)
    assert info["years_data_collected"] == "2023-Present"
    text = amf.page_text(PAGE)
    assert "Someone" not in text and "Decoy" not in text and "1990" not in text
    assert all(v is None for v in amf.parse_site_page("<html><body>No data</body></html>", "US-XXX").values())
    # another site id that starts with the same letters is not matched
    assert amf.parse_site_page(PAGE.replace("US-XXX", "US-XXX1"), "US-XXX")["base_version"] is None


def test_requests_are_spaced_and_identified(monkeypatch):
    slept = []
    monkeypatch.setattr(amf, "MIN_INTERVAL_S", 1.0)
    monkeypatch.setattr(amf, "_last_request", [0.0])
    monkeypatch.setattr(amf, "time", SimpleNamespace(monotonic=lambda: 100.0, sleep=slept.append))
    s = _Session({"example": b"{}"})
    amf._get("https://example.org/a", s)
    amf._get("https://example.org/b", s)
    assert slept == [1.0]
    assert all(h["User-Agent"] == amf.USER_AGENT and "@" not in h["User-Agent"] for _, h in s.calls)


def test_fetch_variable_availability_pins_caches_and_verifies(tmp_path, monkeypatch):
    monkeypatch.setattr(amf, "PINNED_SHA", "")
    monkeypatch.setattr(amf, "PINNED_MD5", "")
    commits = json.dumps([{"sha": SHA, "commit": {"committer": {"date": "2025-11-03T20:55:40Z"},
                                                  "message": "update\n\ndetails"}}]).encode()
    s = _Session({"api.github.com": commits, f"raw.githubusercontent.com/chuhousen/amerifluxr/{SHA}/": CSV})
    path, meta = amf.fetch_variable_availability(tmp_path, session=s)
    md5 = hashlib.md5(CSV).hexdigest()
    assert path.read_bytes() == CSV and path.name.endswith(f"{SHA[:10]}.csv")
    assert (meta["sha"], meta["md5"], meta["create_date"], meta["commit_message"]) == (
        SHA, md5, "2025-10-20", "update")
    prov = json.loads((tmp_path / "provenance.json").read_text())["files"][path.name]
    assert SHA in prov["url"] and prov["create_date"] == "2025-10-20" and prov["accessed_utc"]
    # once pinned, reruns use the cached file without network access and check its md5
    monkeypatch.setattr(amf, "PINNED_SHA", SHA)
    monkeypatch.setattr(amf, "PINNED_MD5", md5)
    path2, meta2 = amf.fetch_variable_availability(tmp_path, session=_Offline())
    assert path2 == path and meta2["md5"] == md5 and meta2["create_date"] == "2025-10-20"
    table, create_date = amf.read_variable_availability(path2)
    assert create_date == "2025-10-20" and amf.le_availability(table, "US-AAA").to_dict() == {2024: 0.5, 2025: 0.25}
    with pytest.warns(UserWarning):  # 'latest' cannot be resolved offline: fall back to the pinned file
        assert amf.fetch_variable_availability(tmp_path, sha="latest", session=_Offline())[0] == path
    path.write_bytes(CSV + b"US-BBB,LE,LE,FALSE,1,1\n")  # corrupted cache
    with pytest.raises(ValueError):
        amf.fetch_variable_availability(tmp_path, session=_Offline())
    path.unlink()  # a download that differs from the pinned file is rejected and not cached
    with pytest.raises(ValueError):
        amf.fetch_variable_availability(tmp_path, session=_Session({SHA: CSV + b"x"}))
    assert not path.exists()


def test_fetch_json_and_site_page_are_cached_with_access_time(tmp_path):
    body = json.dumps([{"SITE_ID": "US-XXX", "publish_years": [2023, 2024, 2025]}]).encode()
    s = _Session({"data_availability": body, "siteinfo/US-XXX": PAGE.encode()})
    data, meta = amf.fetch_json(amf.DATA_AVAILABILITY_URL, tmp_path, session=s)
    assert data[0]["publish_years"] == [2023, 2024, 2025] and meta["accessed_utc"]
    info = amf.site_page_info("US-XXX", tmp_path, session=s)
    assert info["base_version"] == "4-5" and info["page_url"] == "https://ameriflux.lbl.gov/sites/siteinfo/US-XXX"
    assert info["page_md5"] == hashlib.md5(PAGE.encode()).hexdigest() and info["page_accessed_utc"]
    n = len(s.calls)
    assert amf.fetch_json(amf.DATA_AVAILABILITY_URL, tmp_path, session=_Offline())[0] == data
    assert amf.site_page_info("US-XXX", tmp_path, session=_Offline())["base_version"] == "4-5"
    assert len(s.calls) == n
    assert set(amf.load_provenance(tmp_path)["files"]) == {
        "api_v1_data_availability_AmeriFlux_BASE-BADM_CCBY4.0.json", "siteinfo_US-XXX.html"}
    with pytest.warns(UserWarning):  # refresh with the network down falls back to the cache
        assert amf.fetch_json(amf.DATA_AVAILABILITY_URL, tmp_path, session=_Offline(), refresh=True)[0] == data
    # a cached copy without an access time does not date the snapshot: it is fetched again like a missing one
    names = amf.snapshot_files(["US-XXX"])
    assert amf.undated_files(tmp_path, names) == []
    (tmp_path / "provenance.json").unlink()
    assert amf.undated_files(tmp_path, names) == [names[0], names[2]]  # site_display was never cached
    assert amf.fetch_json(amf.DATA_AVAILABILITY_URL, tmp_path, session=s)[1]["accessed_utc"] and len(s.calls) == n + 1
    assert amf.undated_files(tmp_path, names) == ["siteinfo_US-XXX.html"]
    with pytest.warns(UserWarning, match="using the cached"):  # offline: the undated copy is used, still undated
        assert amf.site_page_info("US-XXX", tmp_path, session=_Offline())["page_accessed_utc"] is None


def test_script_builds_site_table_from_caches(tmp_path, monkeypatch, capsys, s11):
    monkeypatch.setattr(amf, "PINNED_SHA", SHA)
    monkeypatch.setattr(amf, "PINNED_MD5", hashlib.md5(CSV).hexdigest())
    pub = [{"SITE_ID": "US-AAA", "publish_years": [2023, 2024, 2025]}]
    s = _Session({SHA: CSV, "data_availability": json.dumps(pub).encode(),
                  "site_display": json.dumps([{"SITE_ID": "US-AAA", "TOWER_BEGAN": "2023"}]).encode(),
                  "siteinfo/US-AAA": PAGE.replace("US-XXX", "US-AAA").encode()})
    amf.fetch_variable_availability(tmp_path, session=s)  # fill the caches through the mocked network
    amf.fetch_json(amf.DATA_AVAILABILITY_URL, tmp_path, session=s)
    amf.fetch_json(amf.SITE_DISPLAY_URL, tmp_path, session=s)
    amf.site_page_info("US-AAA", tmp_path, session=s)

    def no_network(*args, **kwargs):
        raise AssertionError("network used")

    monkeypatch.setattr(amf, "_get", no_network)
    sites = pd.DataFrame({"site": ["US-AAA"], "koppen_go26": ["BSk"], "period_start": ["2024-01"],
                          "period_end": ["2025-02"], "mean_obs_et_mm_d": [0.59], "base_version_go26": ["3-5"]})
    out, by_year, prov = s11.build(sites, tmp_path)
    r = out.iloc[0]
    a = (0.5 * 366 + 59) / 425  # 0.25 x 365 = 91 days in 2025 capped at Jan-Feb
    assert np.isclose(r["le_avail_period"], a) and np.isclose(r["f_missing_period"], 1 - a)
    assert np.isclose(r["le_avail_full_years"], 0.5) and r["n_years_capped"] == 1
    assert np.isclose(r["implied_mean_valid_mm_d"], 0.59 / a)
    assert (r["base_version_current"], r["version_changed_since_go26"]) == ("4-5", True)
    assert (r["publish_years"], r["tower_began"], r["le_variables"]) == ("2023-2025", "2023", "LE")
    assert r["base_years_available"] == "2023-2025" and r["le_avail_source"].startswith(f"amerifluxr@{SHA[:10]}")
    assert by_year[["year", "days_in_period"]].values.tolist() == [[2024, 366], [2025, 59]]
    assert prov["variable_availability"]["sha"] == SHA and "US-AAA" in prov["site_pages"]
    s11.summary(out)
    assert "Below 60%" in capsys.readouterr().out


def test_script_keeps_the_snapshot_when_its_cache_is_missing(tmp_path, monkeypatch, capsys, s11):
    monkeypatch.setattr(amf, "PINNED_SHA", SHA)
    monkeypatch.setattr(amf, "PINNED_MD5", hashlib.md5(CSV).hexdigest())
    sites = tmp_path / "sites.csv"
    pd.DataFrame({"site": ["US-AAA"], "koppen_go26": ["BSk"], "period_start": ["2024-01"], "period_end": ["2024-12"],
                  "mean_obs_et_mm_d": [0.5], "base_version_go26": ["4-5"]}).to_csv(sites, index=False)
    cache, res = tmp_path / "cache", tmp_path / "results"
    res.mkdir()
    (res / "ameriflux_le_availability.csv").write_text("archived snapshot\n")
    args = ["--sites", str(sites), "--cache", str(cache), "--out", str(res)]
    monkeypatch.setattr(amf, "_session", [_NoNetwork()])
    with pytest.raises(SystemExit) as stop:  # the pages would describe the network today: stop, fetch nothing
        s11.main(args)
    msg = str(stop.value.code)
    assert "STOPPED" in msg and "--refresh" in msg and "siteinfo_US-AAA.html" in msg
    assert (res / "ameriflux_le_availability.csv").read_text() == "archived snapshot\n" and not cache.exists()

    s = _Session({SHA: CSV, "data_availability": json.dumps([{"SITE_ID": "US-AAA", "publish_years": [2024]}]).encode(),
                  "site_display": json.dumps([{"SITE_ID": "US-AAA", "TOWER_BEGAN": "2023"}]).encode(),
                  "siteinfo/US-AAA": PAGE.replace("US-XXX", "US-AAA").encode()})
    monkeypatch.setattr(amf, "_session", [s])
    s11.main(args + ["--refresh"])  # replaces the snapshot on purpose
    assert "Fetching all" in capsys.readouterr().out
    out = pd.read_csv(res / "ameriflux_le_availability.csv", dtype={"base_version_current": str})
    assert (out.loc[0, "site"], out.loc[0, "base_version_current"]) == ("US-AAA", "4-5")
    before = {p.name: p.read_bytes() for p in res.iterdir()}
    monkeypatch.setattr(amf, "_session", [_NoNetwork()])
    s11.main(args)  # cache in place: offline, byte-identical
    assert {p.name: p.read_bytes() for p in res.iterdir()} == before


def test_script_never_writes_a_provenance_without_access_times(tmp_path, monkeypatch, s11):
    monkeypatch.setattr(amf, "PINNED_SHA", SHA)
    monkeypatch.setattr(amf, "PINNED_MD5", hashlib.md5(CSV).hexdigest())
    sites = tmp_path / "sites.csv"
    pd.DataFrame({"site": ["US-AAA"], "koppen_go26": ["BSk"], "period_start": ["2024-01"], "period_end": ["2024-12"],
                  "mean_obs_et_mm_d": [0.5], "base_version_go26": ["4-5"]}).to_csv(sites, index=False)
    cache, res = tmp_path / "cache", tmp_path / "results"
    args = ["--sites", str(sites), "--cache", str(cache), "--out", str(res)]
    routes = {SHA: CSV, "data_availability": json.dumps([{"SITE_ID": "US-AAA", "publish_years": [2024]}]).encode(),
              "site_display": json.dumps([{"SITE_ID": "US-AAA", "TOWER_BEGAN": "2023"}]).encode(),
              "siteinfo/US-AAA": PAGE.replace("US-XXX", "US-AAA").encode()}
    monkeypatch.setattr(amf, "_session", [_Session(routes)])
    s11.main(args)  # the snapshot: every file fetched and dated
    snapshot = {p.name: p.read_bytes() for p in res.iterdir()}
    prov_file = cache / "provenance.json"
    complete = json.loads(prov_file.read_text())

    def kept() -> bool:
        return {p.name: p.read_bytes() for p in res.iterdir()} == snapshot

    # provenance.json lost: nothing dates the cached pages; stop before fetching or writing anything
    prov_file.unlink()
    monkeypatch.setattr(amf, "_session", [_NoNetwork()])
    with pytest.raises(SystemExit) as stop:
        s11.main(args)
    msg = str(stop.value.code)
    assert "STOPPED" in msg and "provenance.json is missing" in msg and "--refresh" in msg
    assert kept() and not prov_file.exists()
    # one page without an access time: the same, naming the page
    complete["files"]["siteinfo_US-AAA.html"].pop("accessed_utc")
    prov_file.write_text(json.dumps(complete))
    with pytest.raises(SystemExit) as stop:
        s11.main(args)
    assert "1 cached files have no access time in provenance.json (e.g. siteinfo_US-AAA.html)" in str(stop.value.code)
    assert kept()
    # --refresh offline: the cached copies cannot be dated, so the snapshot is kept rather than given null times
    prov_file.unlink()
    monkeypatch.setattr(amf, "_session", [_Offline()])
    with pytest.warns(UserWarning, match="using the cached"), pytest.raises(SystemExit) as stop:
        s11.main(args + ["--refresh"])
    assert "no access time for 4 cached AmeriFlux files" in str(stop.value.code)
    assert kept() and not prov_file.exists()
    # a new run elsewhere fetches the undated copies again and records when
    s = _Session(routes)
    monkeypatch.setattr(amf, "_session", [s])
    s11.main(["--sites", str(sites), "--cache", str(cache), "--out", str(tmp_path / "new")])
    assert len(s.calls) == 4
    prov = json.loads((tmp_path / "new" / "ameriflux_public_provenance.json").read_text())
    assert prov["site_pages"]["US-AAA"]["page_accessed_utc"] and all(
        prov[k]["accessed_utc"] for k in ("variable_availability", "data_availability", "site_display"))
    assert amf.undated_files(cache, [*amf.snapshot_files(["US-AAA"]), amf.availability_file(SHA)]) == []
