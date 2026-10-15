"""LE availability at GO26's 22 towers from public AmeriFlux metadata (no login).

GO26's Table 1 site means are reproduced by summing the available half-hourly
LE with missing records counted as zero and dividing by calendar days. A site
mean is then A x (mean over the available records), where A is the share of
the study period's records with valid LE, so 1 - A of the period entered as
zero ET. This script estimates A for each GO26 period without BASE files:

  * per-year LE availability from amerifluxr's BASE variable-availability
    summary at a pinned commit (etcomment.amf_public.le_availability:
    non-gap-filled LE at the one position etcomment.ameriflux.read_base
    reads, the one with the most valid records; the per-year maximum over
    replicates at that position, exact at the 22 towers, which have one LE
    variable per position)
  * BASE publish years (AmeriFlux data_availability service), tower start year
    (site_display) and the current BASE version, its citation year and the
    years available (site pages), against the versions GO26 cite

A partial year is assumed to fall inside the period, an upper bound on A
(conservative for the zero-fill argument). The availability summary was made
on its create_date (2025-10-20 at the pinned commit), before the BASE versions
GO26 cite with a 2026 date were published, so at those sites A describes the
preceding version. implied_mean_valid_mm_d = GO26 mean / A is the unclosed
mean over valid records implied by zero-filling (approximate: missing
half-hours are not a random sample of the day).

Snapshot. The pinned summary can be downloaded again (commit SHA and md5); the
web-service responses and site pages cannot, because they describe the
network on the day they were fetched. They are documented by URL, access time
and md5 (data/raw/ameriflux_public/provenance.json, copied to
results/ameriflux_public_provenance.json), the values read from them
(base_version_current, base_years_available, publish_years, tower_began, ...)
are archived in results/ as a snapshot, and the cache itself is deposited with
the results. Site pages are cached without the site teams' e-mail addresses:
every address, including mailto: links, becomes '[email removed]' before a
fetched page is written (amf.redact_emails; a page cached earlier with
addresses is redacted in place on the next run). page_md5 in the provenance
is the md5 of the page as served; the cache's provenance.json gives it as
md5_original next to the md5 of the redacted copy, so redaction leaves the
results unchanged. Reruns use the cache; a rerun without it fetches the pages
again and so reflects the network on the rerun date. A cached file without an access
time in the cache's provenance.json (or with provenance.json missing) counts as
missing, since nothing then dates it. The script therefore stops without
fetching or writing anything when cached pages or responses are missing and
results/ameriflux_le_availability.csv already exists, unless --refresh is given
(or --out names another folder); otherwise it fetches them. It never writes a
provenance in which a page, response or summary it used has no access time:
when such a file cannot be fetched again, it stops and writes nothing.

Outputs
  results/ameriflux_le_availability.csv          one row per site
  results/ameriflux_le_availability_by_year.csv  site x year of each GO26 period
  results/ameriflux_public_provenance.json       URLs, access times, commit SHA, md5
  data/raw/ameriflux_public/                     cached downloads + provenance.json

Usage:
  python scripts/11_ameriflux_availability.py            # network only for files not yet cached
  python scripts/11_ameriflux_availability.py --refresh  # re-fetch the web services and site pages
  python scripts/11_ameriflux_availability.py --latest   # newest amerifluxr summary, not the pinned one
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from etcomment import amf_public as amf

ROOT = Path(__file__).resolve().parents[1]
# year of each BASE citation in GO26's reference list (= year the version they used was published)
GO26_VERSION_YEAR = {
    "US-ARM": 2026, "US-Bi1": 2026, "US-Bi2": 2026, "US-IAB": 2025, "US-MN1": 2025, "US-MN2": 2025,
    "US-MN3": 2025, "US-Mo1": 2026, "US-NSb": 2025, "US-Ne1": 2026, "US-Ne2": 2026, "US-Ne3": 2026,
    "US-RGA": 2025, "US-RGB": 2025, "US-UTD": 2026, "US-UTJ": 2026, "US-UTV": 2026, "US-UiC": 2025,
    "US-UiG": 2025, "US-VT1": 2025, "US-WT1": 2026, "US-xSL": 2025,
}
LOW = 0.60  # availability flagged in the summary


def _first_le_year(by_year: pd.Series) -> int | None:
    y = by_year[by_year > 0]
    return int(y.index.min()) if len(y) else None


def build(sites: pd.DataFrame, cache: Path, latest: bool = False,
          refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path, avail_meta = amf.fetch_variable_availability(cache, sha="latest" if latest else None)
    table, create_date = amf.read_variable_availability(path)
    table_year = int(create_date[:4]) if create_date else None
    pub, pub_meta = amf.fetch_json(amf.DATA_AVAILABILITY_URL, cache, refresh=refresh)
    disp, disp_meta = amf.fetch_json(amf.SITE_DISPLAY_URL, cache, refresh=refresh)
    pub = {d["SITE_ID"]: d.get("publish_years") or [] for d in pub}
    disp = {d["SITE_ID"]: d for d in disp}
    source = f"amerifluxr@{avail_meta['sha'][:10]} (create_date {create_date})"
    rows, years, pages = [], [], {}
    for _, s in sites.iterrows():
        site = s["site"]
        try:
            page = amf.site_page_info(site, cache, refresh=refresh)
        except Exception as err:  # keep going; the row shows what is missing
            warnings.warn(f"{site}: site page not available ({err})")
            page = {}
        pages[site] = {"file": f"siteinfo_{site}.html",
                       **{k: page.get(k) for k in ("page_url", "page_accessed_utc", "page_md5")}}
        by_year = amf.le_availability(table, site)
        per = amf.period_availability(by_year, s["period_start"], s["period_end"])
        t = amf.period_years(by_year, s["period_start"], s["period_end"])
        t.insert(0, "site", site)
        years.append(t)
        ver_go26, ver_now = s.get("base_version_go26"), page.get("base_version")
        go26_year = GO26_VERSION_YEAR.get(site)
        rows.append({
            "site": site, "koppen_go26": s["koppen_go26"], "period_start": s["period_start"],
            "period_end": s["period_end"], "calendar_days": per["calendar_days"],
            "go26_table1_mean": s["mean_obs_et_mm_d"],
            "base_version_go26": ver_go26, "go26_version_year": go26_year,
            "base_version_current": ver_now, "base_citation_year_current": page.get("base_citation_year"),
            "version_changed_since_go26": (None if ver_now is None or pd.isna(ver_go26)
                                           else bool(ver_now != ver_go26)),
            "table_predates_go26_version": (None if go26_year is None or table_year is None
                                            or go26_year == table_year else go26_year > table_year),
            "base_years_available": page.get("base_years_available"),
            "years_data_collected": page.get("years_data_collected"),
            "publish_years": amf.year_ranges(pub.get(site, [])),
            "tower_began": disp.get(site, {}).get("TOWER_BEGAN"),
            "le_variables": ";".join(amf.le_variables(table, site)),
            "first_le_year": _first_le_year(by_year),
            "le_avail_period": per["le_avail_period"], "le_avail_full_years": per["le_avail_full_years"],
            "n_full_years": per["n_full_years"], "n_years_capped": per["n_years_capped"],
            "valid_record_days": per["days_available"], "f_missing_period": 1 - per["le_avail_period"],
            "implied_mean_valid_mm_d": s["mean_obs_et_mm_d"] / per["le_avail_period"],
            "le_avail_source": source,
        })
    prov = {
        "generated_by": "scripts/11_ameriflux_availability.py",
        "user_agent": amf.USER_AGENT,
        "variable_availability": {**avail_meta, "latest_url": f"https://raw.githubusercontent.com/{amf.REPO}"
                                                              f"/master/{amf.AVAIL_PATH}"},
        "data_availability": pub_meta,
        "site_display": disp_meta,
        "site_pages": pages,
        "notes": ("Web services and site pages describe the network on the access date; the cached copies "
                  "in data/raw/ameriflux_public/ (deposited with the results) are the inputs of the results, "
                  "and a rerun without them reflects the network on the rerun date. Data Use Log not used."),
    }
    return pd.DataFrame(rows), pd.concat(years, ignore_index=True), prov


def summary(out: pd.DataFrame) -> None:
    show = out.assign(period=out["period_start"] + ".." + out["period_end"],
                      versions=out["base_version_go26"].astype(str) + " > " + out["base_version_current"].astype(str))
    cols = {"site": "site", "koppen_go26": "zone", "period": "period", "go26_table1_mean": "GO26",
            "versions": "BASE GO26 > now", "first_le_year": "LE from", "le_avail_period": "A period",
            "le_avail_full_years": "A full yrs", "f_missing_period": "1-A", "implied_mean_valid_mm_d": "GO26/A"}
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(show[list(cols)].rename(columns=cols).round(3).to_string(index=False))
    a = out.set_index("site")["le_avail_period"]
    print(f"\nLE availability over GO26's periods ({out['le_avail_source'].iloc[0]}): mean {a.mean():.3f}, "
          f"median {a.median():.3f}, min {a.min():.3f} ({a.idxmin()}), max {a.max():.3f} ({a.idxmax()})")
    low = a[a < LOW].sort_values()
    print(f"Below {LOW:.0%} at {len(low)} of {len(a)} sites: "
          + ", ".join(f"{k} {v:.3f}" for k, v in low.items()))
    ch = out[out["version_changed_since_go26"] == True]  # noqa: E712 (None = unknown)
    print("BASE version changed since GO26: "
          + (", ".join(f"{r.site} {r.base_version_go26} -> {r.base_version_current}" for r in ch.itertuples())
             or "none"))
    unknown = out.loc[out["base_version_current"].isna(), "site"].tolist()
    if unknown:
        print("Current version unknown (site page not parsed):", ", ".join(unknown))
    pre = out.loc[out["table_predates_go26_version"] == True, "site"].tolist()  # noqa: E712
    print(f"Availability summary made before the BASE version GO26 used was published (GO26 cite it with a "
          f"later year) at {len(pre)} sites: {', '.join(pre)}")
    same = out[(out["version_changed_since_go26"] == False)  # noqa: E712
               & out["base_citation_year_current"].notna()
               & (out["base_citation_year_current"] != out["go26_version_year"])]
    if len(same):
        print("Check: citation year on the site page differs from GO26's reference list at",
              ", ".join(same["site"]))


def snapshot_guard(cache: Path, res: Path, sites, refresh: bool, latest: bool = False) -> list[str]:
    """Files to fetch: pages and responses missing from ``cache``, and cached files without an access time in its
    provenance.json (amf.undated_files). Exits when fetching them would silently replace the snapshot in ``res``
    (no --refresh)."""
    snap = amf.snapshot_files(sites)
    absent = [n for n in snap if not (cache / n).exists()]
    undated = amf.undated_files(cache, snap + ([] if latest else [amf.availability_file(amf.PINNED_SHA)]))
    target = res / "ameriflux_le_availability.csv"
    if (absent or undated) and target.exists() and not refresh:
        bar = "!" * 100
        lines = [bar, f"STOPPED: the AmeriFlux snapshot cached in {cache} is incomplete:"]
        if absent:
            lines.append(f"  {len(absent)} pages or web-service responses are missing (e.g. {', '.join(absent[:3])})")
        if undated:
            lines.append(f"  {len(undated)} cached files have no access time in provenance.json (e.g. "
                         f"{', '.join(undated[:3])})" if (cache / "provenance.json").exists() else
                         f"  provenance.json is missing: nothing dates the {len(undated)} cached files")
        sys.exit("\n".join(lines + [
            f"{target} is a snapshot built from them; fetching them now would replace it, and the access times in",
            f"{res / 'ameriflux_public_provenance.json'}, with the network as it is today. Nothing was fetched or",
            "written. Copy data/raw/ameriflux_public/ (with its provenance.json) from the Zenodo deposit to reproduce",
            "the snapshot, rerun with --refresh to replace it on purpose, or write a new run elsewhere with --out.",
            bar]))
    return absent + undated


def undated(prov: dict) -> list[str]:
    """Files of :func:`build`'s provenance that were used without an access time (a site page that could not be
    read at all has no md5 and counts as missing, not undated)."""
    used = [prov["variable_availability"], prov["data_availability"], prov["site_display"]]
    return ([m.get("file") for m in used if not m.get("accessed_utc")]
            + [p["file"] for p in prov["site_pages"].values() if p.get("page_md5") and not p.get("page_accessed_utc")])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default=str(ROOT / "config" / "go26_sites.csv"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "raw" / "ameriflux_public"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch the web services and site pages (replaces the snapshot in --out)")
    ap.add_argument("--latest", action="store_true",
                    help="use the newest amerifluxr availability summary instead of the pinned commit")
    args = ap.parse_args(argv)
    sites = pd.read_csv(args.sites, dtype={"base_version_go26": str})
    cache, res = Path(args.cache), Path(args.out)
    fetch = snapshot_guard(cache, res, sites["site"], args.refresh, args.latest)
    if fetch or args.refresh:
        what = ("all AmeriFlux pages and web-service responses" if args.refresh else
                f"{len(fetch)} AmeriFlux files (missing, or cached without an access time)")
        print(f"Fetching {what} now: the results describe the network on this date, not the archived snapshot.")
    out, by_year, prov = build(sites, cache, latest=args.latest, refresh=args.refresh)
    if gaps := undated(prov):  # never write (or overwrite) a provenance without access times
        sys.exit(f"STOPPED: no access time for {len(gaps)} cached AmeriFlux files (e.g. {', '.join(gaps[:3])}): "
                 f"{cache / 'provenance.json'} lacks them and they could not be fetched again. "
                 f"Nothing was written to {res}.")
    res.mkdir(parents=True, exist_ok=True)
    out.to_csv(res / "ameriflux_le_availability.csv", index=False, float_format="%.4f")
    by_year.to_csv(res / "ameriflux_le_availability_by_year.csv", index=False, float_format="%.4f")
    (res / "ameriflux_public_provenance.json").write_text(json.dumps(prov, indent=1, sort_keys=True,
                                                                     default=_json_default))
    summary(out)
    print(f"\nwrote {res / 'ameriflux_le_availability.csv'}, ameriflux_le_availability_by_year.csv, "
          f"ameriflux_public_provenance.json")


def _json_default(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    raise TypeError(type(o))


if __name__ == "__main__":
    main()
