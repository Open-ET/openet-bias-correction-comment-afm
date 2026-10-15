"""Extract the current OpenET collection at the GO26 towers and the benchmark croplands.

Golkar & de Oliveira (2026; GO26, Section 2.1) used the OpenET API with a
200 m x 200 m polygon centred on each tower.  This script builds the same
boxes for their 22 towers and for the 60 cropland stations of the Volk et al.
(2023a) benchmark (75 boxes; seven towers are in both) and requests, with
OpenET collection v2.1 (which starts in 2016 in the API):

  monthly, 2016-2025, all 75 boxes (multipolygon requests on a temporary
      GeoJSON upload): ET of the ensemble and of each of the six models,
      reference ET (ETo) and precipitation
  daily, 2016-01 to 2025-07, one box per request: ensemble ET at US-UiC
      (GO26 Fig. 8), US-UTJ (Fig. S2) and the seven towers in the benchmark;
      ETo and precipitation at US-UiC and US-UTJ

Daily multipolygon requests for all boxes exceed the Earth Engine memory
limit of a free account, hence the single-box daily requests (the API allows
daily windows of up to 3,650 days for one polygon).  ETo is OpenET's
bias-corrected gridMET grass reference ET (Earth Engine collections
projects/openet/assets/reference_et/conus/gridmet/{daily,monthly}/v1;
Volk et al., 2026), which the API returns for reference_et = "gridmet";
precipitation is gridMET.  ``--verify-eto-ee`` checks the monthly ETo against
the Earth Engine collection directly (needs ``pip install .[ee]`` and an
authenticated Earth Engine account; no API requests).

A run from scratch uses 24 of the 100 monthly requests of a free account (one
upload, 10 monthly and 13 daily requests).  Responses are cached in
data/raw/openet_api/ with a request log (manifest.json: endpoint, parameters,
collection version, UTC time and, from October 2026 on, the md5 of the GeoJSON
of the boxes), so reruns only fetch missing files.  OpenET revises recent data
and updates collections, so a fresh extraction can differ slightly from the
cached one used for the published numbers (accessed 2026-10-15 UTC).

The cached extracts belong to the boxes in sites_200m.geojson.  When the inputs
give other boxes, that box set is kept if it contains every new box unchanged
(e.g. 74 boxes without the benchmark folder: the V24 list in config/ lacks
US-MC1), with a warning; new or moved boxes stop the run, as does a cached
extract whose logged GeoJSON md5 differs from the file.

Requires an OpenET API key (https://account.etdata.org) in the OPENET_API_KEY
environment variable or in a .env file in the repository root.  OpenET data
are subject to the OpenET API Terms of Service; the raw extracts stay in
data/raw/ and only derived statistics are written to results/.

Outputs
  data/raw/openet_api/sites.csv, sites_200m.geojson   the boxes
  data/raw/openet_api/<request>.csv                   one file per request (raw)
  data/raw/openet_api/openet_monthly.csv              site, date, et_<model>, eto, pr
  data/raw/openet_api/openet_daily.csv                site, date, et_ensemble, eto, pr
  results/openet_api_eto_ee_check.csv                 with --verify-eto-ee

Usage:
  python scripts/02_download_openet_api.py --dry-run   # list the requests not yet cached
  python scripts/02_download_openet_api.py
  python scripts/02_download_openet_api.py --verify-eto-ee
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from etcomment import openet_api as oa

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "openet_api"
VERSION = 2.1
# requests may span at most 10 years; v2.1 starts in 2016
MONTHLY = [("2016-01-01", "2019-12-31"), ("2020-01-01", "2025-12-31")]  # ensemble
MONTHLY_OTHER = ("2016-01-01", "2025-12-31")                             # models, ETo, precipitation
DAILY = ("2016-01-01", "2025-07-31")  # 3,500 days; covers every GO26 study period
SHARED = ["US-ARM", "US-Bi1", "US-Bi2", "US-Ne1", "US-Ne2", "US-Ne3", "US-xSL"]
DAILY_ET_SITES = ["US-UiC", "US-UTJ", *SHARED]
DAILY_MET_SITES = ["US-UiC", "US-UTJ"]


def site_table(flux_root: Path | None) -> pd.DataFrame:
    """GO26 towers (their coordinates) plus the benchmark cropland stations."""
    go = pd.read_csv(ROOT / "config" / "go26_sites.csv")[["site", "latitude", "longitude"]]
    go["group"] = "go26"
    meta = flux_root / "station_metadata.xlsx" if flux_root else None
    if meta is not None and meta.exists():
        m = pd.read_excel(meta, header=1).rename(columns={"Site ID": "site", "Latitude": "latitude",
                                                          "Longitude": "longitude"})
        crop = m[m["General classification"] == "Croplands"]
    else:  # V24 list in config/ (59 of the 60 cropland stations; US-MC1 is not in V24)
        m = pd.read_csv(ROOT / "config" / "volk2024_stations.csv", comment="#")
        crop = m[m["general_classification"] == "Croplands"]
    bm = crop[["site", "latitude", "longitude"]].copy()
    bm["group"] = "benchmark"
    go.loc[go["site"].isin(bm["site"]), "group"] = "go26+benchmark"
    return pd.concat([go, bm[~bm["site"].isin(go["site"])]], ignore_index=True)


def box_set(sites: pd.DataFrame, gj: Path, cached: bool) -> tuple[dict, bool]:
    """Boxes for this run and whether sites.csv and ``gj`` are (re)written.

    With cached extracts, the boxes in ``gj`` are those the extracts were made
    with: a different set from the inputs keeps them if it contains no new or
    moved box (and stops the run otherwise).
    """
    fc = oa.feature_collection(sites)
    if not cached or not gj.exists() or hashlib.md5(json.dumps(fc).encode()).hexdigest() == oa.file_md5(gj):
        return fc, True
    old = json.loads(gj.read_text())
    rings = {f["properties"]["site"]: f["geometry"]["coordinates"][0] for f in old["features"]}
    changed = sorted(f["properties"]["site"] for f in fc["features"]
                     if rings.get(f["properties"]["site"]) != f["geometry"]["coordinates"][0])
    if changed:
        raise SystemExit(f"new or moved boxes ({len(changed)}: {', '.join(changed[:10])}) relative to "
                         f"{gj.name} (md5 {oa.file_md5(gj)}), the boxes of the cached extracts: restore the "
                         f"inputs, or move the cached extracts and {gj.name} aside to extract again")
    print(f"warning: the inputs give {len(fc['features'])} boxes, but the cached extracts were made with the "
          f"{len(old['features'])} boxes in {gj.name}; keeping those (sites.csv and {gj.name} are not rewritten)")
    return old, False


def stale_extracts(manifest: Path, tags: list[str], md5: str) -> list[str]:
    """Cached extracts whose logged request names another GeoJSON (entries without an md5 are older)."""
    calls = json.loads(manifest.read_text())["calls"] if manifest.exists() else []
    made_with = {c["tag"]: c.get("geojson_md5") for c in calls if c.get("status") == 200 and "tag" in c}
    return [t for t in tags if made_with.get(t) not in (None, md5)]


def plan() -> list[dict]:
    """Every request, in order; ``site`` is None for multipolygon requests."""
    reqs = [dict(interval="monthly", variable="et", model="ensemble", start=a, end=b, site=None)
            for a, b in MONTHLY]
    reqs += [dict(interval="monthly", variable="et", model=m, start=MONTHLY_OTHER[0], end=MONTHLY_OTHER[1],
                  site=None) for m in oa.MODELS if m != "ensemble"]
    reqs += [dict(interval="monthly", variable=v, model="ensemble", start=MONTHLY_OTHER[0], end=MONTHLY_OTHER[1],
                  site=None) for v in ("eto", "pr")]
    reqs += [dict(interval="daily", variable="et", model="ensemble", start=DAILY[0], end=DAILY[1], site=s)
             for s in DAILY_ET_SITES]
    reqs += [dict(interval="daily", variable=v, model="ensemble", start=DAILY[0], end=DAILY[1], site=s)
             for s in DAILY_MET_SITES for v in ("eto", "pr")]
    for r in reqs:
        where = f"_{r['site']}" if r["site"] else ""
        r["tag"] = f"{r['interval']}_{r['variable']}_{r['model']}{where}_{r['start'][:4]}_{r['end'][:4]}"
    return reqs


def assemble(results: dict[str, pd.DataFrame]) -> None:
    for interval in ("daily", "monthly"):
        frames: dict[str, list[pd.DataFrame]] = {}
        for r in plan():
            if r["interval"] == interval and r["tag"] in results:
                name = f"et_{r['model']}" if r["variable"] == "et" else r["variable"]
                frames.setdefault(name, []).append(results[r["tag"]])
        if frames:
            w = oa.wide({k: pd.concat(v, ignore_index=True).drop_duplicates(["site", "date"])
                         for k, v in frames.items()})
            w.to_csv(OUT / f"openet_{interval}.csv", index=False, float_format="%.4f")
            print(f"wrote {OUT / f'openet_{interval}.csv'}: {len(w)} rows, {w['site'].nunique()} sites")


def verify_eto_ee(fc: dict) -> None:
    """Compare the API's monthly ETo with the bias-corrected gridMET collection on Earth Engine."""
    api = oa.read_result_csv((OUT / "monthly_eto_ensemble_2016_2025.csv").read_text(), "eto")
    ee_eto = oa.ee_monthly_eto(fc, 2016, 2025)
    stats = {"collection": oa.ETO_EE_MONTHLY, **oa.compare_series(api, ee_eto)}
    (ROOT / "results").mkdir(exist_ok=True)
    pd.DataFrame([stats]).to_csv(ROOT / "results" / "openet_api_eto_ee_check.csv", index=False)
    print("API monthly ETo minus Earth Engine collection (mm/month):",
          {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flux", default=str(ROOT / "data/raw/flux_ET_dataset"),
                    help="benchmark folder (for the cropland station coordinates)")
    ap.add_argument("--dry-run", action="store_true", help="list the requests that are not cached yet")
    ap.add_argument("--verify-eto-ee", action="store_true",
                    help="check the cached monthly ETo against Earth Engine (needs earthengine-api)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gj = OUT / "sites_200m.geojson"
    if args.verify_eto_ee:  # the boxes used for the cached extraction
        verify_eto_ee(json.loads(gj.read_text()))
        return
    try:
        from etcomment import benchmark as bm
        flux_root = bm.find_root(Path(args.flux))
    except FileNotFoundError:
        flux_root = None
    cached_tags = [r["tag"] for r in plan() if (OUT / f"{r['tag']}.csv").exists()]
    sites = site_table(flux_root)
    fc, new = box_set(sites, gj, bool(cached_tags))
    if new:
        sites.to_csv(OUT / "sites.csv", index=False)
        gj.write_text(json.dumps(fc))
    gj_md5 = oa.file_md5(gj)
    stale = stale_extracts(OUT / "manifest.json", cached_tags, gj_md5)
    if stale:
        raise SystemExit(f"cached extracts made with boxes other than {gj.name} (md5 {gj_md5}): "
                         f"{', '.join(stale)}; move them aside to extract again")
    rings = {f["properties"]["site"]: f["geometry"]["coordinates"][0] for f in fc["features"]}
    todo = [r for r in plan() if r["tag"] not in cached_tags]
    print(f"{len(rings)} boxes ({gj.name} md5 {gj_md5}); {len(plan())} requests in the plan, "
          f"{len(todo)} not cached")
    if args.dry_run:
        for r in todo:
            print("  ", r["tag"])
        return
    client = oa.Client(oa.api_key(ROOT), OUT) if todo else None  # a complete cache needs no key
    if client is not None:
        print("account:", client.status())
    asset = None
    results = {}
    for r in plan():
        cached = OUT / f"{r['tag']}.csv"
        common = dict(model=r["model"], version=VERSION, tag=r["tag"])
        if cached.exists():
            df = oa.read_result_csv(cached.read_text(), r["variable"])
        elif r["site"] is None:
            asset = asset or client.cached_asset(gj) or client.upload(gj)
            df = client.multipolygon(asset, r["start"], r["end"], r["interval"], r["variable"], **common)
        else:
            df = client.polygon(r["site"], rings[r["site"]], r["start"], r["end"], r["interval"], r["variable"],
                                geojson_md5=gj_md5, **common)
        results[r["tag"]] = df
        print(f"{r['tag']}: {len(df)} rows, {df['site'].nunique()} sites", flush=True)
    assemble(results)
    if client is not None:
        print("account:", client.status())


if __name__ == "__main__":
    main()
