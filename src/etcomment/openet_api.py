"""Client for the OpenET API (https://openet-api.org).

Golkar & de Oliveira (2026, Section 2.1) extracted daily ET for the ensemble
and the six models, plus gridMET reference ET and precipitation, "through the
OpenET data services" for a 200 m x 200 m polygon centred on each tower.  The
same boxes are built here (:func:`box`), uploaded once as a temporary GeoJSON
asset (:meth:`Client.upload`, valid for 72 h) and summarized for every polygon
in one request with the multipolygon endpoint (:meth:`Client.multipolygon`),
which returns a link to a CSV file that expires after five minutes.

Free-tier limits (October 2026): 100 requests per month, 20 per minute, 100
polygons per request and one-year windows for daily multipolygon requests.
Every result is cached on disk, so a rerun only spends requests on files that
are missing.  The md5 of the GeoJSON behind every request is kept in the
request log (``manifest.json``) and with the uploaded asset (``.asset.json``),
so an upload is reused only for the same boxes.

The API key is read from the ``OPENET_API_KEY`` environment variable or from a
``.env`` file in the repository root.  Never commit or deposit the key.
OpenET data are subject to the OpenET API Terms of Service: attribute
https://openetdata.org wherever values are shown or shared.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import shapely
from pyproj import Proj, Transformer

API = "https://openet-api.org"
MODELS = ["ensemble", "disalexi", "eemetric", "geesebal", "ptjpl", "sims", "ssebop"]


class OpenETError(RuntimeError):
    pass


def api_key(root: str | Path | None = None) -> str:
    """API key from ``OPENET_API_KEY`` (preferred) or a ``.env`` file in ``root``.

    In ``.env`` the line ``[export] OPENET_API_KEY=value`` sets it: a quoted value
    is taken verbatim between its quotes, an unquoted one ends where a ``#``
    comment starts (at the start of the value or after whitespace).  A key with
    whitespace is rejected; error messages never echo the value.
    """
    key = os.environ.get("OPENET_API_KEY", "").strip()
    env = Path(root) / ".env" if root is not None else None
    if not key and env is not None and env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip().removeprefix("export ").strip() == "OPENET_API_KEY":
                key = _env_value(value)
    if not key:
        raise OpenETError("set OPENET_API_KEY (environment variable or .env file) to use the OpenET API")
    if any(c.isspace() for c in key):
        raise OpenETError("OPENET_API_KEY contains whitespace (check for a stray space or a comment inside quotes)")
    return key


def _env_value(raw: str) -> str:
    """Value of a ``.env`` assignment: verbatim between matching quotes, else up to an inline comment."""
    v = raw.strip()
    if v[:1] in ("'", '"'):
        end = v.find(v[0], 1)
        if end < 0:
            raise OpenETError("unterminated quote in the .env value of OPENET_API_KEY")
        return v[1:end]
    return re.split(r"(?:^|\s)#", v, maxsplit=1)[0].strip()


def file_md5(path: str | Path) -> str:
    """md5 of a file (the GeoJSON of the boxes is identified by it)."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def box(lon: float, lat: float, size_m: float = 200.0) -> list[list[float]]:
    """Counter-clockwise ring of a ``size_m`` square centred on (lon, lat), sides N-S and E-W."""
    tr = Transformer.from_proj(Proj(proj="aeqd", lat_0=lat, lon_0=lon, datum="WGS84"), "EPSG:4326",
                               always_xy=True)
    h = size_m / 2
    lons, lats = tr.transform([-h, h, h, -h, -h], [-h, -h, h, h, -h])
    return [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lons, lats)]


def feature_collection(sites: pd.DataFrame, size_m: float = 200.0) -> dict:
    """GeoJSON boxes for ``sites`` (columns site, latitude, longitude); boxes must not overlap."""
    feats = [{"type": "Feature", "properties": {"site": str(r["site"])},
              "geometry": {"type": "Polygon", "coordinates": [box(r["longitude"], r["latitude"], size_m)]}}
             for _, r in sites.iterrows()]
    polys = [shapely.Polygon(f["geometry"]["coordinates"][0]) for f in feats]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].intersects(polys[j]):
                raise ValueError(f"boxes of {sites['site'].iloc[i]} and {sites['site'].iloc[j]} overlap")
    return {"type": "FeatureCollection", "features": feats}


def read_result_csv(text: str, variable: str, attribute: str = "site") -> pd.DataFrame:
    """Long table (site, date, value) from a multipolygon CSV.

    The CSV holds one row per polygon and time step (``time,site,et,hectares,
    megaliters`` for ET); the value column is named after the requested
    variable.  Time steps without data are absent, not zero.
    """
    df = pd.read_csv(io.StringIO(text))
    cols = {c.lower(): c for c in df.columns}
    tcol = next((cols[c] for c in ("time", "date", "datetime") if c in cols), None)
    vcol = cols.get(variable.lower())
    if tcol is None or vcol is None or attribute not in df.columns:
        raise OpenETError(f"unexpected CSV columns for {variable}: {list(df.columns)}")
    out = pd.DataFrame({"site": df[attribute].astype(str), "date": pd.to_datetime(df[tcol]),
                        "value": pd.to_numeric(df[vcol], errors="coerce")})
    return out.sort_values(["site", "date"]).reset_index(drop=True)


class Client:
    """Rate-limited, caching client.

    Every call is logged (without the key or asset id) to ``manifest.json``,
    with the md5 of the GeoJSON of the boxes (``geojson_md5``).
    """

    def __init__(self, key: str, cache: str | Path, min_interval_s: float = 4.0, timeout_s: float = 900.0):
        self._key = key
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self._last = 0.0
        self._asset_md5: dict[str, str] = {}  # asset id -> md5 of the GeoJSON it was uploaded from
        self.manifest_path = self.cache / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {"calls": []}

    # -- plumbing ------------------------------------------------------------------------------
    def _wait(self) -> None:
        dt = time.monotonic() - self._last
        if dt < self.min_interval_s:
            time.sleep(self.min_interval_s - dt)
        self._last = time.monotonic()

    def _log(self, entry: dict) -> None:
        entry["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.manifest["calls"].append(entry)
        self.manifest_path.write_text(json.dumps(self.manifest, indent=1))

    def _request(self, method: str, path: str, log: dict | None = None, **kw) -> requests.Response:
        """HTTP request; a failure is logged (with ``log``) and raised."""
        self._wait()
        r = requests.request(method, API + path, headers={"Authorization": self._key, "accept": "application/json"},
                             timeout=self.timeout_s, **kw)
        if r.status_code != 200:
            params = {k: v for k, v in (kw.get("json") or {}).items() if k not in ("asset_id", "geometry")}
            self._log({"endpoint": path, "status": r.status_code, "error": r.text[:500], "params": params,
                       **(log or {})})
            raise OpenETError(f"{path}: HTTP {r.status_code}: {r.text[:500]}")
        return r

    # -- endpoints -----------------------------------------------------------------------------
    def status(self) -> dict:
        return self._request("GET", "/account/status").json()

    def upload(self, geojson: str | Path) -> str:
        """Upload a GeoJSON file; returns the temporary asset id (valid for 72 h).

        The id embeds the account name, so it is kept in a local ``.asset.json``
        (with the file's md5, for reuse within 72 h) and not in the shareable
        request log.
        """
        p = Path(geojson)
        md5 = file_md5(p)
        with open(p, "rb") as fh:
            r = self._request("POST", "/account/upload", log={"file": p.name, "geojson_md5": md5},
                              files={"file": (p.name, fh, "application/geo+json")})
        body = r.json()
        asset = _find_asset_id(body)
        if asset is None:
            raise OpenETError("no asset id in the upload response")
        self._asset_md5[asset] = md5
        self._log({"endpoint": "/account/upload", "status": 200, "file": p.name, "geojson_md5": md5,
                   "expires": body.get("expires") if isinstance(body, dict) else None})
        (self.cache / ".asset.json").write_text(json.dumps(
            {"id": asset, "file": p.name, "md5": md5, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}))
        return asset

    def cached_asset(self, geojson: str | Path, max_age_h: float = 70.0) -> str | None:
        """Asset id of an earlier upload of a file with the same content (md5), if it has not expired."""
        f = self.cache / ".asset.json"
        if not f.exists():
            return None
        a = json.loads(f.read_text())
        md5 = file_md5(geojson)
        if a.get("md5") != md5:  # other boxes, or an upload made before md5s were kept
            return None
        age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(a["utc"])).total_seconds() / 3600
        if age_h >= max_age_h:
            return None
        self._asset_md5[a["id"]] = md5
        return a["id"]

    def multipolygon(self, asset_id: str, start: str, end: str, interval: str, variable: str,
                     model: str = "ensemble", reducer: str = "mean", version: float = 2.1,
                     reference_et: str = "gridmet", units: str = "mm", attribute: str = "site",
                     tag: str | None = None) -> pd.DataFrame:
        """Spatial mean of ``variable`` per polygon and time step; cached as ``<tag>.csv``.

        The request log records the md5 of the GeoJSON the asset was uploaded
        from (known when the asset came from :meth:`upload` or :meth:`cached_asset`).
        """
        tag = tag or f"{interval}_{variable}_{model}_{start}_{end}".lower()
        out = self.cache / f"{tag}.csv"
        if out.exists():
            return read_result_csv(out.read_text(), variable, attribute)
        md5 = self._asset_md5.get(asset_id)
        params = {"date_range": [start, end], "interval": interval, "asset_id": asset_id,
                  "attributes": [attribute], "reducer": reducer, "model": model, "variable": variable,
                  "reference_et": reference_et, "version": version, "units": units}
        r = self._request("POST", "/raster/timeseries/multipolygon", log={"tag": tag, "geojson_md5": md5},
                          json=params)
        body = r.json()
        url = _find_url(body)
        if url is None:
            raise OpenETError(f"no download link in response: {body}")
        csv = requests.get(url, timeout=self.timeout_s)
        csv.raise_for_status()
        df = read_result_csv(csv.text, variable, attribute)  # validates before caching
        out.write_text(csv.text)
        params.pop("asset_id")
        self._log({"endpoint": "/raster/timeseries/multipolygon", "status": 200, "tag": tag, "geojson_md5": md5,
                   "params": params, "rows": len(df), "sites": int(df["site"].nunique())})
        return df

    def polygon(self, site: str, ring: list[list[float]], start: str, end: str, interval: str, variable: str,
                model: str = "ensemble", reducer: str = "mean", version: float = 2.1,
                reference_et: str = "gridmet", units: str = "mm", tag: str | None = None,
                geojson_md5: str | None = None) -> pd.DataFrame:
        """Spatial mean of ``variable`` for one polygon (daily windows of up to 10 years); cached.

        ``geojson_md5`` (logged) identifies the box set the ring was taken from.
        """
        tag = tag or f"{interval}_{variable}_{model}_{site}_{start}_{end}".lower()
        out = self.cache / f"{tag}.csv"
        if not out.exists():
            params = {"date_range": [start, end], "interval": interval,
                      "geometry": [v for pt in ring[:-1] for v in pt], "model": model, "variable": variable,
                      "reference_et": reference_et, "reducer": reducer, "version": version, "units": units,
                      "file_format": "csv"}
            r = self._request("POST", "/raster/timeseries/polygon",
                              log={"tag": tag, "site": site, "geojson_md5": geojson_md5}, json=params)
            text = r.text
            if text.lstrip().startswith(("{", "[")):  # JSON body: a link or the records themselves
                body = r.json()
                url = _find_url(body)
                text = requests.get(url, timeout=self.timeout_s).text if url else pd.DataFrame(body).to_csv(
                    index=False)
            df = pd.read_csv(io.StringIO(text))
            df.insert(0, "site", site)
            read_result_csv(df.to_csv(index=False), variable)  # validates before caching
            df.to_csv(out, index=False)
            params.pop("geometry")
            self._log({"endpoint": "/raster/timeseries/polygon", "status": 200, "tag": tag, "site": site,
                       "geojson_md5": geojson_md5, "params": params, "rows": len(df)})
        return read_result_csv(out.read_text(), variable)


def _find_asset_id(body) -> str | None:
    if isinstance(body, str):
        return body if ("/" in body and " " not in body) else None
    if isinstance(body, dict):
        for k in ("asset_id", "assetId", "asset", "id", "path", "name"):
            v = body.get(k)
            if isinstance(v, str) and v:
                return v
        for v in body.values():
            a = _find_asset_id(v)
            if a:
                return a
    if isinstance(body, list):
        for v in body:
            a = _find_asset_id(v)
            if a:
                return a
    return None


def _find_url(body) -> str | None:
    if isinstance(body, str):
        return body if body.startswith("http") else None
    if isinstance(body, dict):
        for v in body.values():
            u = _find_url(v)
            if u:
                return u
    if isinstance(body, list):
        for v in body:
            u = _find_url(v)
            if u:
                return u
    return None


ETO_EE_MONTHLY = "projects/openet/assets/reference_et/conus/gridmet/monthly/v1"


def ee_monthly_eto(fc: dict, start_year: int, end_year: int, collection: str = ETO_EE_MONTHLY) -> pd.DataFrame:
    """Monthly bias-corrected gridMET ETo (Volk et al., 2026) for the boxes, read from Earth Engine.

    Area-weighted mean over each box at 30 m, as the API reduces it.  Needs the
    ``earthengine-api`` package and an authenticated account (project from the
    ``EE_PROJECT`` environment variable, else the default).  Used only to check
    that the API's ETo is this collection.
    """
    import ee

    ee.Initialize(project=os.environ.get("EE_PROJECT") or None)
    feats = ee.FeatureCollection([ee.Feature(ee.Geometry.Polygon(f["geometry"]["coordinates"]),
                                             {"site": f["properties"]["site"]}) for f in fc["features"]])

    def per_image(img):
        d = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        return img.reduceRegions(collection=feats, reducer=ee.Reducer.mean(), scale=30).map(
            lambda f: f.set("date", d))

    rows = []
    for y in range(start_year, end_year + 1):  # one call per year stays below EE's 5,000-element limit
        col = ee.ImageCollection(collection).filterDate(f"{y}-01-01", f"{y + 1}-01-01").select("eto")
        for f in col.map(per_image).flatten().getInfo()["features"]:
            p = f["properties"]
            rows.append({"site": p["site"], "date": pd.Timestamp(p["date"]), "value": p.get("mean")})
    return pd.DataFrame(rows).sort_values(["site", "date"]).reset_index(drop=True)


def compare_series(api: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Differences between two long (site, date, value) tables on their common rows."""
    m = api.merge(other, on=["site", "date"], suffixes=("_api", "_other")).dropna()
    d = m["value_api"] - m["value_other"]
    return {"n_pairs": int(len(m)), "n_sites": int(m["site"].nunique()), "mean_diff": float(d.mean()),
            "median_diff": float(d.median()), "p99_abs_diff": float(d.abs().quantile(0.99)),
            "max_abs_diff": float(d.abs().max()),
            "max_rel_diff": float((d.abs() / m["value_other"].abs()).replace(np.inf, np.nan).max())}


def wide(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join long (site, date, value) tables into one wide table keyed by site and date."""
    parts = [f.set_index(["site", "date"])["value"].rename(name) for name, f in frames.items()]
    out = pd.concat(parts, axis=1).sort_index().reset_index()
    for c in frames:
        out[c] = out[c].astype(float).replace([np.inf, -np.inf], np.nan)
    return out
