import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
import shapely
from pyproj import Geod, Transformer

from etcomment import openet_api as oa

TOWERS = [(-109.535, 37.2818), (-88.1961, 40.0628), (-121.499, 38.0992), (-96.1269, 45.6168), (-80.0, 25.5)]
KEY = "k3yABC123secret"  # fake


def _script02():
    spec = importlib.util.spec_from_file_location(
        "s02", Path(__file__).resolve().parents[1] / "scripts" / "02_download_openet_api.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Response:
    def __init__(self, status: int = 200, text: str = "", js=None):
        self.status_code, self.text, self._js = status, text, js

    def json(self):
        return self._js if self._js is not None else json.loads(self.text)

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(str(self.status_code))


@pytest.mark.parametrize("line,expected", [
    ("OPENET_API_KEY=abc123", "abc123"),
    ("OPENET_API_KEY=abc123          # scripts/02", "abc123"),       # README template with a value
    ("  export OPENET_API_KEY = abc123\t# note", "abc123"),
    ("OPENET_API_KEY=abc#123", "abc#123"),                            # '#' without whitespace is part of it
    ('OPENET_API_KEY="abc#123 x"', None),                             # quoted: verbatim, so whitespace -> error
    ('OPENET_API_KEY="abc #123"  # comment', None),
    ("OPENET_API_KEY='abc#123'   # comment", "abc#123"),
    ('OPENET_API_KEY="abc123" # comment', "abc123"),
])
def test_api_key_from_env_file(tmp_path, monkeypatch, line, expected):
    monkeypatch.delenv("OPENET_API_KEY", raising=False)
    (tmp_path / ".env").write_text(f"# OPENET_API_KEY=commented-out\nOTHER=1\n{line}\n")
    if expected is None:
        with pytest.raises(oa.OpenETError, match="whitespace") as e:
            oa.api_key(tmp_path)
        assert "abc" not in str(e.value) and "123" not in str(e.value)  # never echoes the value
    else:
        assert oa.api_key(tmp_path) == expected


@pytest.mark.parametrize("line,match", [
    ("OPENET_API_KEY=          # scripts/02", "set OPENET_API_KEY"),   # README template left empty
    ("OPENET_API_KEY=abc 123", "whitespace"),
    ('OPENET_API_KEY="abc123', "unterminated quote"),
])
def test_api_key_rejects_bad_values_without_echoing(tmp_path, monkeypatch, line, match):
    monkeypatch.delenv("OPENET_API_KEY", raising=False)
    (tmp_path / ".env").write_text(line + "\n")
    with pytest.raises(oa.OpenETError, match=match) as e:
        oa.api_key(tmp_path)
    assert "abc" not in str(e.value) and "123" not in str(e.value)


def test_api_key_environment_wins(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OPENET_API_KEY=fromfile\n")
    monkeypatch.setenv("OPENET_API_KEY", "  fromenv  ")
    assert oa.api_key(tmp_path) == "fromenv"
    monkeypatch.setenv("OPENET_API_KEY", "from env")
    with pytest.raises(oa.OpenETError, match="whitespace"):
        oa.api_key(tmp_path)
    monkeypatch.delenv("OPENET_API_KEY")
    assert oa.api_key(tmp_path) == "fromfile"
    with pytest.raises(oa.OpenETError, match="set OPENET_API_KEY"):
        oa.api_key(tmp_path / "missing")


@pytest.mark.parametrize("lon,lat", TOWERS)
def test_box_is_a_200_m_square_centred_on_the_tower(lon, lat):
    ring = oa.box(lon, lat)
    assert len(ring) == 5 and ring[0] == ring[-1]
    poly = shapely.Polygon(ring)
    assert poly.exterior.is_ccw  # GeoJSON (RFC 7946) exterior rings are counter-clockwise
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)  # equal-area
    x, y = tr.transform([p[0] for p in ring], [p[1] for p in ring])
    assert shapely.Polygon(list(zip(x, y))).area == pytest.approx(40_000, rel=2e-3)
    geod = Geod(ellps="WGS84")
    for a, b in zip(ring[:-1], ring[1:]):
        assert geod.inv(a[0], a[1], b[0], b[1])[2] == pytest.approx(200.0, abs=0.5)
    # sides run east-west and north-south; the centre is the tower
    assert ring[0][1] == pytest.approx(ring[1][1], abs=1e-7) and ring[0][0] == pytest.approx(ring[3][0], abs=1e-6)
    assert poly.centroid.x == pytest.approx(lon, abs=1e-6) and poly.centroid.y == pytest.approx(lat, abs=1e-6)


def test_box_size_parameter():
    ring = oa.box(-100.0, 40.0, size_m=1000.0)
    geod = Geod(ellps="WGS84")
    assert abs(geod.geometry_area_perimeter(shapely.Polygon(ring))[0]) == pytest.approx(1e6, rel=1e-3)


def test_feature_collection_properties_and_overlap_check():
    sites = pd.DataFrame({"site": ["US-Ne1", "US-Ne2"], "latitude": [41.16506, 41.16487],
                          "longitude": [-96.47664, -96.47010]})  # about 550 m apart
    fc = oa.feature_collection(sites)
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 2
    f = fc["features"][0]
    assert f["properties"] == {"site": "US-Ne1"} and f["geometry"]["type"] == "Polygon"
    assert f["geometry"]["coordinates"][0] == oa.box(-96.47664, 41.16506)
    close = pd.DataFrame({"site": ["A", "B"], "latitude": [40.0, 40.001], "longitude": [-100.0, -100.0]})  # 111 m
    with pytest.raises(ValueError, match="A and B overlap"):
        oa.feature_collection(close)
    assert len(oa.feature_collection(close, size_m=100.0)["features"]) == 2


def test_read_result_csv_parses_multipolygon_output():
    text = ("time,site,et,hectares,megaliters\n"
            "2020-02-01,US-UTJ,30.5,4.0,1.22\n"
            "2020-01-01,US-UTJ,20.0,4.0,0.8\n"
            "2020-01-01,US-ARM,10.0,4.0,0.4\n")
    df = oa.read_result_csv(text, "et")
    assert list(df.columns) == ["site", "date", "value"]
    assert list(df["site"]) == ["US-ARM", "US-UTJ", "US-UTJ"]
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-01-01", "2020-02-01"]
    assert list(df["value"]) == [10.0, 20.0, 30.5]
    eto = oa.read_result_csv("Time,site,ETo\n2021-03-01,US-UiC,55.1\n", "eto")
    assert eto.loc[0, "value"] == 55.1


@pytest.mark.parametrize("text", ["date,name,et\n2020-01-01,US-UTJ,1.0\n",   # no site attribute
                                  "time,site,value\n2020-01-01,US-UTJ,1.0\n",  # no column for the variable
                                  "site,et\nUS-UTJ,1.0\n"])                    # no time column
def test_read_result_csv_rejects_unexpected_columns(text):
    with pytest.raises(oa.OpenETError, match="unexpected CSV columns"):
        oa.read_result_csv(text, "et")


def test_wide_outer_joins_variables():
    d = pd.to_datetime(["2020-01-01", "2020-02-01"])
    et = pd.DataFrame({"site": ["B", "A", "A"], "date": [d[0], d[1], d[0]], "value": [3.0, 2.0, 1.0]})
    eto = pd.DataFrame({"site": ["A", "B"], "date": [d[0], d[1]], "value": [10.0, np.inf]})
    w = oa.wide({"et_ensemble": et, "eto": eto})
    assert list(w.columns) == ["site", "date", "et_ensemble", "eto"]
    assert list(zip(w["site"], w["date"].dt.month)) == [("A", 1), ("A", 2), ("B", 1), ("B", 2)]
    assert w["et_ensemble"].tolist()[:3] == [1.0, 2.0, 3.0] and np.isnan(w["et_ensemble"].iloc[3])
    assert w["eto"].iloc[0] == 10.0 and w["eto"].iloc[1:].isna().all()  # inf becomes NaN


def _geojson(tmp_path: Path, lats=(40.0, 40.01, 40.02)) -> Path:
    sites = pd.DataFrame({"site": ["A", "B", "C"][:len(lats)], "latitude": list(lats), "longitude": -100.0})
    gj = tmp_path / "sites_200m.geojson"
    gj.write_text(json.dumps(oa.feature_collection(sites)))
    return gj


def test_upload_and_cached_asset_are_tied_to_the_geojson_md5(tmp_path, monkeypatch):
    posts = []

    def fake_request(method, url, headers=None, timeout=None, **kw):
        posts.append(url)
        return _Response(js={"asset_id": "projects/openet/assets/users/acct_x/sites"})

    monkeypatch.setattr(oa.requests, "request", fake_request)
    cache = tmp_path / "cache"
    gj = _geojson(tmp_path)
    c = oa.Client(KEY, cache, min_interval_s=0)
    asset = c.upload(gj)
    md5 = oa.file_md5(gj)
    stored = json.loads((cache / ".asset.json").read_text())
    assert stored["id"] == asset and stored["md5"] == md5
    log = (cache / "manifest.json").read_text()
    assert json.loads(log)["calls"][-1]["geojson_md5"] == md5
    assert "acct_x" not in log and KEY not in log  # neither the asset id nor the key is logged
    assert c.cached_asset(gj) == asset
    gj.write_text(json.dumps(oa.feature_collection(pd.DataFrame(
        {"site": ["A"], "latitude": [41.0], "longitude": [-100.0]}))))  # other boxes, same file name
    assert c.cached_asset(gj) is None
    _geojson(tmp_path)
    assert c.cached_asset(gj) == asset
    old = dict(stored, utc=(datetime.now(timezone.utc) - timedelta(hours=71)).isoformat(timespec="seconds"))
    (cache / ".asset.json").write_text(json.dumps(old))
    assert c.cached_asset(gj) is None  # expired
    (cache / ".asset.json").write_text(json.dumps({k: v for k, v in stored.items() if k != "md5"}))
    assert c.cached_asset(gj) is None  # uploads without an md5 are not reused
    assert len(posts) == 1


def test_requests_log_the_geojson_md5(tmp_path, monkeypatch):
    calls = []

    def fake_request(method, url, headers=None, timeout=None, **kw):
        calls.append(url)
        if url.endswith("/account/upload"):
            return _Response(js={"asset_id": "projects/openet/assets/users/acct_x/sites"})
        if url.endswith("/multipolygon"):
            return _Response(js={"url": "https://example.org/x.csv"})
        if kw["json"]["variable"] == "pr":
            return _Response(404, text="no data")
        return _Response(text="time,et\n2020-01-01,1.5\n2020-01-02,1.7\n")

    def fake_get(url, timeout=None):
        calls.append(url)
        return _Response(text="time,site,et,hectares,megaliters\n2020-01-01,A,10.0,4,0.4\n2020-02-01,A,11.0,4,0.4\n")

    monkeypatch.setattr(oa.requests, "request", fake_request)
    monkeypatch.setattr(oa.requests, "get", fake_get)
    gj = _geojson(tmp_path)
    md5 = oa.file_md5(gj)
    c = oa.Client(KEY, tmp_path / "cache", min_interval_s=0)
    asset = c.upload(gj)
    df = c.multipolygon(asset, "2020-01-01", "2020-12-31", "monthly", "et", tag="m1")
    assert list(df["value"]) == [10.0, 11.0]
    n = len(calls)
    assert c.multipolygon(asset, "2020-01-01", "2020-12-31", "monthly", "et", tag="m1").equals(df)
    assert len(calls) == n  # cached: no request
    ring = oa.box(-100.0, 40.0)
    p = c.polygon("A", ring, "2020-01-01", "2020-01-02", "daily", "et", tag="p1", geojson_md5=md5)
    assert list(p["value"]) == [1.5, 1.7] and set(p["site"]) == {"A"}
    with pytest.raises(oa.OpenETError, match="HTTP 404"):
        c.polygon("A", ring, "2020-01-01", "2020-01-02", "daily", "pr", tag="p2", geojson_md5=md5)
    log = json.loads((tmp_path / "cache" / "manifest.json").read_text())["calls"]
    assert [e.get("geojson_md5") for e in log] == [md5] * 4  # upload, multipolygon, polygon, failed polygon
    assert log[-1]["status"] == 404 and log[-1]["tag"] == "p2" and log[-1]["site"] == "A"
    text = (tmp_path / "cache" / "manifest.json").read_text()
    assert KEY not in text and "acct_x" not in text
    # an asset from another session is logged without an md5 (unknown)
    c2 = oa.Client(KEY, tmp_path / "cache", min_interval_s=0)
    c2.multipolygon(asset, "2021-01-01", "2021-12-31", "monthly", "et", tag="m2")
    assert json.loads((tmp_path / "cache" / "manifest.json").read_text())["calls"][-1]["geojson_md5"] is None


def test_script02_keeps_the_cached_box_set(tmp_path, capsys):
    s02 = _script02()
    sites = pd.DataFrame({"site": ["A", "B", "C"], "latitude": [40.0, 40.01, 40.02], "longitude": -100.0,
                          "group": "go26"})
    gj = tmp_path / "sites_200m.geojson"
    fc, new = s02.box_set(sites, gj, cached=True)       # no box set on disk yet
    assert new and len(fc["features"]) == 3
    gj.write_text(json.dumps(fc))
    assert s02.box_set(sites, gj, cached=True) == (fc, True)
    # fewer boxes (e.g. no benchmark folder): the cached set is kept and nothing is rewritten
    kept, new = s02.box_set(sites.iloc[:2], gj, cached=True)
    assert not new and kept == fc and "keeping those" in capsys.readouterr().out
    assert s02.box_set(sites.iloc[:2], gj, cached=False)[1]  # nothing cached: the new boxes are written
    moved = sites.assign(latitude=[40.0, 40.011, 40.02])
    with pytest.raises(SystemExit, match=r"new or moved boxes \(1: B\)"):
        s02.box_set(moved, gj, cached=True)
    extra = pd.concat([sites, pd.DataFrame({"site": ["D"], "latitude": [40.03], "longitude": [-100.0]})])
    with pytest.raises(SystemExit, match="D"):
        s02.box_set(extra, gj, cached=True)


def test_script02_flags_extracts_made_with_other_boxes(tmp_path):
    s02 = _script02()
    man = tmp_path / "manifest.json"
    assert s02.stale_extracts(man, ["t1"], "x") == []
    calls = [{"endpoint": "/raster/timeseries/multipolygon", "status": 200, "tag": "t1"},            # no md5
             {"endpoint": "/raster/timeseries/multipolygon", "status": 200, "tag": "t2", "geojson_md5": "x"},
             {"endpoint": "/raster/timeseries/multipolygon", "status": 200, "tag": "t3", "geojson_md5": "y"},
             {"endpoint": "/raster/timeseries/polygon", "status": 404, "tag": "t4", "geojson_md5": "y"},
             {"endpoint": "/raster/timeseries/polygon", "status": 200, "tag": "t5", "geojson_md5": "y"},
             {"endpoint": "/raster/timeseries/polygon", "status": 200, "tag": "t5", "geojson_md5": "x"}]
    man.write_text(json.dumps({"calls": calls}))
    assert s02.stale_extracts(man, ["t1", "t2", "t3", "t4", "t5"], "x") == ["t3"]  # the latest entry counts
