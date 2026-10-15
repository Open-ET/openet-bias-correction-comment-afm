"""AmeriFlux public metadata: BASE versions, publish years and LE availability (no login).

GO26's Table 1 site means are reproduced by summing the available half-hourly
LE of the AmeriFlux BASE files with missing records counted as zero and
dividing by calendar days.  Under that rule a site mean is A x (mean over the
available records), where A is the fraction of the study period's records with
a valid LE value, so 1 - A of the period entered as zero ET.  A is estimated
here from public summaries, without BASE files (which need an account):

* amerifluxr's BASE variable-availability summary (GitHub chuhousen/amerifluxr,
  data-summary/AMF_AA-Flx_BASE_VARIABLE-AVAILABILITY_LATEST.csv): for every
  site, variable and year the fraction of the calendar year's records (17,520
  or 17,568 half-hours; 8,760 or 8,784 hours) with a non-missing value.  The
  file is pinned to a git commit (PINNED_SHA) and checked against PINNED_MD5;
  its first line gives the date it was generated (create_date).  LE is taken
  at the one position etcomment.ameriflux.read_base reads (le_availability).
* AmeriFlux web services (amfcdn.lbl.gov/api/v1): data_availability (BASE
  publish years per site) and site_display (site metadata incl. TOWER_BEGAN).
* Site pages (ameriflux.lbl.gov/sites/siteinfo/<SITE>): current BASE version,
  its citation year and DOI, and "Years Data Available".  The per-user Data
  Use Log panel is skipped (it is filled in by JavaScript and never parsed).

The pinned summary can be downloaded again (commit SHA, md5).  The web-service
responses and site pages cannot: they change whenever a BASE version is
published.  They are documented by URL, access time and md5
(data/raw/ameriflux_public/provenance.json, copied to
results/ameriflux_public_provenance.json), the values read from them are
archived in results/ as a snapshot (scripts/11), and the cache itself is
deposited with the results.  A rerun without the cache fetches them again and
so reflects the pages on the rerun date.  A cached copy without an access time
in provenance.json does not document its date (:func:`undated_files`) and is
fetched again like a missing one.  Caveat: the availability summary describes
the BASE versions current at its create_date, which can be older than the
versions GO26 downloaded (03-2026).
"""
from __future__ import annotations

import calendar
import hashlib
import io
import json
import re
import time
import warnings
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from etcomment import ameriflux as af

REPO = "chuhousen/amerifluxr"
AVAIL_PATH = "data-summary/AMF_AA-Flx_BASE_VARIABLE-AVAILABILITY_LATEST.csv"
COMMITS_URL = f"https://api.github.com/repos/{REPO}/commits?path={AVAIL_PATH}&per_page=1"
RAW_URL = "https://raw.githubusercontent.com/" + REPO + "/{sha}/" + AVAIL_PATH
# newest commit touching the file when this analysis was run (create_date 2025-10-20)
PINNED_SHA = "7319e63255738281ccec2159d6dfba7111e4f9f3"
PINNED_MD5 = "484a80079679a0c75152b30351d98b26"
PINNED_COMMIT = {"commit_date": "2025-11-03T20:55:40Z", "commit_message": "202510 BASE summary update"}

DATA_AVAILABILITY_URL = "https://amfcdn.lbl.gov/api/v1/data_availability/AmeriFlux/BASE-BADM/CCBY4.0"
SITE_DISPLAY_URL = "https://amfcdn.lbl.gov/api/v1/site_display/AmeriFlux"
SITE_PAGE_URL = "https://ameriflux.lbl.gov/sites/siteinfo/{site}"
SITE_PAGE_FILE = "siteinfo_{site}.html"

USER_AGENT = ("etcomment/0.1 (research code: AmeriFlux LE availability for a comment on "
              "Golkar & de Oliveira 2026, Agric. For. Meteorol.; python-requests)")
MIN_INTERVAL_S = 1.0  # between requests
TIMEOUT_S = 60.0
_last_request = [0.0]
_session: list[requests.Session] = []


# -- network plumbing ------------------------------------------------------------------------------
def _get(url: str, session=None) -> requests.Response:
    """Polite GET: descriptive User-Agent and at least MIN_INTERVAL_S between requests."""
    if session is None:
        if not _session:
            _session.append(requests.Session())
        session = _session[0]
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    try:
        r = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)
    finally:
        _last_request[0] = time.monotonic()
    r.raise_for_status()
    return r


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def load_provenance(cache: str | Path) -> dict:
    p = Path(cache) / "provenance.json"
    return json.loads(p.read_text()) if p.exists() else {"files": {}}


def undated_files(cache: str | Path, names) -> list[str]:
    """Those of ``names`` cached in ``cache`` without an access time in its provenance.json (every cached one
    when provenance.json is missing): such a copy does not document when the network looked that way."""
    cache = Path(cache)
    files = load_provenance(cache)["files"]
    return [n for n in names if (cache / n).exists() and not files.get(n, {}).get("accessed_utc")]


def _record(cache: Path, name: str, meta: dict) -> None:
    prov = load_provenance(cache)
    prov["files"][name] = meta
    tmp = cache / "provenance.json.tmp"
    tmp.write_text(json.dumps(prov, indent=1, sort_keys=True))
    tmp.replace(cache / "provenance.json")


def _cached_fetch(url: str, path: Path, session=None, refresh: bool = False,
                  expected_md5: str | None = None, extra: dict | None = None) -> tuple[bytes, dict]:
    """Bytes of ``url`` cached at ``path`` (provenance in ``path.parent/provenance.json``).

    A cached file is used unless ``refresh`` or it has no access time in
    provenance.json (:func:`undated_files`; it is then fetched again); if a
    download fails, an existing cached copy is used instead (offline reruns).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh and not undated_files(path.parent, [path.name]):
        data = path.read_bytes()
    else:
        try:
            r = _get(url, session)
        except requests.RequestException as err:
            if not path.exists():
                raise
            warnings.warn(f"{url}: {err}; using the cached {path.name}")
            data = path.read_bytes()
        else:
            data = r.content
            if expected_md5 and md5_bytes(data) != expected_md5:
                raise ValueError(f"{url}: md5 {md5_bytes(data)} != expected {expected_md5}")
            part = path.with_name(path.name + ".part")  # no truncated cache files after an interruption
            part.write_bytes(data)
            part.replace(path)
            meta = {"url": url, "final_url": getattr(r, "url", url) or url, "file": path.name,
                    "md5": md5_bytes(data), "bytes": len(data), "accessed_utc": _utc_now(), **(extra or {})}
            _record(path.parent, path.name, meta)
            return data, meta
    got = md5_bytes(data)
    if expected_md5 and got != expected_md5:
        raise ValueError(f"cached {path} has md5 {got}, expected {expected_md5}; delete it to re-download")
    meta = load_provenance(path.parent)["files"].get(path.name, {"url": url, "file": path.name})
    return data, {**meta, "md5": got, "bytes": len(data)}


# -- amerifluxr BASE variable availability ---------------------------------------------------------
def resolve_latest_sha(session=None) -> dict:
    """Newest amerifluxr commit that touched the availability summary (GitHub API)."""
    body = _get(COMMITS_URL, session).json()
    if not body:
        raise ValueError(f"no commits returned by {COMMITS_URL}")
    c = body[0]
    return {"sha": c["sha"], "commit_date": c["commit"]["committer"]["date"],
            "commit_message": c["commit"]["message"].splitlines()[0]}


def availability_file(sha: str) -> str:
    """Cache file name of the availability summary at commit ``sha``."""
    return f"AMF_AA-Flx_BASE_VARIABLE-AVAILABILITY_{sha[:10]}.csv"


def fetch_variable_availability(cache_dir: str | Path, sha: str | None = None,
                                session=None) -> tuple[Path, dict]:
    """Cached copy of the BASE variable-availability summary at one amerifluxr commit.

    ``sha=None`` uses PINNED_SHA (or the newest commit if no SHA is pinned);
    ``sha="latest"`` resolves the newest commit that touched the file via the
    GitHub API.  The file is downloaded once from raw.githubusercontent.com at
    that commit into ``cache_dir`` as :func:`availability_file`, checked
    against PINNED_MD5 when the commit is the pinned one, and its provenance
    (sha, url, md5, create_date, access time) is written to
    ``cache_dir/provenance.json``.  Returns (path, provenance).
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    commit: dict = {}
    if sha is None:
        sha = PINNED_SHA or "latest"
    if sha == "latest":
        try:
            commit = resolve_latest_sha(session)
            sha = commit["sha"]
        except requests.RequestException as err:
            pinned = cache / availability_file(PINNED_SHA)
            if not (PINNED_SHA and pinned.exists()):
                raise
            warnings.warn(f"cannot resolve the latest commit ({err}); using the pinned {PINNED_SHA[:10]}")
            sha = PINNED_SHA
    if not commit and sha == PINNED_SHA:
        commit = dict(PINNED_COMMIT)
    path = cache / availability_file(sha)
    expected = PINNED_MD5 if (sha == PINNED_SHA and PINNED_MD5) else None
    data, meta = _cached_fetch(RAW_URL.format(sha=sha), path, session, expected_md5=expected,
                               extra={"sha": sha, "repo": REPO, "repo_path": AVAIL_PATH, **commit})
    if "create_date" not in meta:
        meta.update(sha=sha, create_date=_create_date(data.decode("utf-8", errors="replace")))
        if path.name in load_provenance(cache)["files"]:
            _record(cache, path.name, meta)
    return path, meta


def _create_date(text: str) -> str | None:
    first = text.split("\n", 1)[0].strip()
    return first.split(":", 1)[1].strip() if first.lower().startswith("create_date") else None


def parse_variable_availability(text: str) -> tuple[pd.DataFrame, str | None]:
    """(table, create_date) from the CSV text; GAP_FILLED becomes boolean."""
    create_date = _create_date(text)
    df = pd.read_csv(io.StringIO(text), skiprows=1 if create_date is not None else 0)
    df["GAP_FILLED"] = _as_bool(df["GAP_FILLED"])
    return df, create_date


def read_variable_availability(path: str | Path) -> tuple[pd.DataFrame, str | None]:
    return parse_variable_availability(Path(path).read_text(encoding="utf-8"))


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.upper().isin({"TRUE", "T", "1", "YES"})


def year_columns(table: pd.DataFrame) -> dict[int, str]:
    return {int(c[1:]): c for c in table.columns if re.fullmatch(r"Y\d{4}", str(c))}


def _le_rows(table: pd.DataFrame, site: str) -> pd.DataFrame:
    rows = table[(table["SITE_ID"] == site) & (table["BASENAME"] == "LE") & ~_as_bool(table["GAP_FILLED"])]
    return rows[rows["VARIABLE"].astype(str).str.match(af.VARS["LE"])]


def le_variables(table: pd.DataFrame, site: str) -> list[str]:
    """Non-gap-filled LE variables of ``site`` at all positions (BASENAME 'LE', GAP_FILLED False,
    named LE or LE_<H>_<V>_<R> as etcomment.ameriflux.read_base reads them)."""
    return sorted(_le_rows(table, site)["VARIABLE"].astype(str))


def le_availability(table: pd.DataFrame, site: str) -> pd.Series:
    """Per-year fraction of the calendar year's records with a valid (not gap-filled) LE.

    Follows etcomment.ameriflux.read_base, which reads LE from one position.
    The variables of :func:`le_variables` (gap-filled LE_PI_F/LE_F and flags
    such as LE_SSITC_TEST are excluded) are grouped by position: the
    unqualified name, or the horizontal and vertical indices of
    LE_<H>_<V>_<R>.  The position with the most valid records over all years
    of the table (sum of fraction x days in the year, proportional to records)
    is used, ties going to the unqualified name, then to the lowest position
    (ameriflux.best_position, as in read_base); other positions are ignored
    even in years when they report more (e.g. LE_1_2_1 at US-ARM in 2005).
    read_base averages the replicates (R) at that position, so a record is
    valid if any replicate is; the per-year maximum over the replicates is a
    lower bound of that union, exact when a single replicate reports.  All
    NaN if the site has no such variable.
    """
    years = year_columns(table)
    idx = pd.Index(sorted(years), name="year")
    rows = _le_rows(table, site)
    if rows.empty:
        return pd.Series(np.nan, index=idx, name="le_avail")
    vals = rows.set_index(rows["VARIABLE"].astype(str))[[years[y] for y in idx]]
    vals = vals.apply(pd.to_numeric, errors="coerce")
    groups: dict[tuple, list[str]] = {}
    for v in vals.index:
        groups.setdefault(af.position(v), []).append(v)
    per_position = {k: vals.loc[cols].max(axis=0) for k, cols in groups.items()}
    days = np.array([366 if calendar.isleap(y) else 365 for y in idx], dtype=float)
    valid = {k: float((s.fillna(0.0).to_numpy(dtype=float) * days).sum()) for k, s in per_position.items()}
    best = per_position[af.best_position(valid)]
    return pd.Series(best.to_numpy(dtype=float), index=idx, name="le_avail")


# -- study-period availability --------------------------------------------------------------------
def period_years(by_year: pd.Series, start: str, end: str) -> pd.DataFrame:
    """Per-year table for a study period ``start``-``end`` ('YYYY-MM', inclusive months).

    days_available = min(le_avail x days_in_year, days_in_period): the year's
    valid records, capped at the days of that year inside the period.  For a
    partial year this assumes the year's records fall inside the period, an
    upper bound on availability (conservative for the zero-fill argument).
    """
    s = pd.Period(start, "M").start_time
    e = pd.Period(end, "M").end_time.normalize()
    if e < s:
        raise ValueError(f"period end {end} before start {start}")
    rows = []
    for y in range(s.year, e.year + 1):
        y0, y1 = pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31)
        diy = (y1 - y0).days + 1
        dip = (min(e, y1) - max(s, y0)).days + 1
        frac = float(by_year.get(y, np.nan))
        ok = np.isfinite(frac)
        rows.append({"year": y, "le_avail": frac, "days_in_year": diy, "days_in_period": dip,
                     "days_available": min(frac * diy, dip) if ok else np.nan,
                     "capped": bool(ok and frac * diy > dip)})
    return pd.DataFrame(rows)


def period_availability(by_year: pd.Series, start: str, end: str) -> dict:
    """Fraction of the period's calendar days covered by valid LE records.

    le_avail_period = sum_y min(frac_y x days_in_year_y, days of y in the period)
    / calendar days (see :func:`period_years` for the partial-year bound);
    le_avail_full_years uses only the period's full calendar years (NaN if none).
    NaN if a year of the period has no availability value.
    """
    t = period_years(by_year, start, end)
    cal = int(t["days_in_period"].sum())
    known = bool(t["le_avail"].notna().all())
    days = float(t["days_available"].sum()) if known else np.nan
    full = t[t["days_in_period"] == t["days_in_year"]]
    full_avail = (float((full["le_avail"] * full["days_in_year"]).sum() / full["days_in_year"].sum())
                  if len(full) and full["le_avail"].notna().all() else np.nan)
    return {"calendar_days": cal, "days_available": days, "le_avail_period": days / cal if known else np.nan,
            "le_avail_full_years": full_avail, "n_full_years": int(len(full)),
            "n_years_capped": int(t["capped"].sum())}


def year_ranges(years) -> str:
    """Compact year list: [2001, 2002, 2003, 2005] -> '2001-2003,2005'."""
    ys = sorted({int(y) for y in years})
    if not ys:
        return ""
    parts, a, b = [], ys[0], ys[0]
    for y in ys[1:] + [None]:
        if y is not None and y == b + 1:
            b = y
            continue
        parts.append(f"{a}-{b}" if b > a else f"{a}")
        if y is not None:
            a = b = y
    return ",".join(parts)


# -- AmeriFlux web services and site pages --------------------------------------------------------
def json_cache_name(url: str) -> str:
    """Cache file name of a web-service response, e.g. api_v1_site_display_AmeriFlux.json."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", url.split("://", 1)[-1].split("/", 1)[-1]).strip("_") + ".json"


def snapshot_files(sites) -> list[str]:
    """Cached files that describe the network on their access date and cannot be fetched again as they
    were: the two web-service responses and the site pages of ``sites``."""
    return [json_cache_name(DATA_AVAILABILITY_URL), json_cache_name(SITE_DISPLAY_URL),
            *(SITE_PAGE_FILE.format(site=s) for s in sites)]


def fetch_json(url: str, cache: str | Path, session=None, refresh: bool = False,
               name: str | None = None) -> tuple[object, dict]:
    """JSON from a public AmeriFlux web service, cached in ``cache`` with its access time.

    Used for DATA_AVAILABILITY_URL (BASE publish years per site) and
    SITE_DISPLAY_URL (site metadata).  These services always describe the
    current state of the network and change over time; the cached copy (and
    the results built from it) is what the paper uses, so it is reused unless
    ``refresh``.  Returns (data, provenance).
    """
    data, meta = _cached_fetch(url, Path(cache) / (name or json_cache_name(url)), session, refresh)
    return json.loads(data.decode("utf-8")), meta


class _VisibleText(HTMLParser):
    """Text of an HTML page without scripts, styles and the ``div`` panels in ``skip_ids``."""

    def __init__(self, skip_ids=("data-use-log-p", "data-use-log")):
        super().__init__(convert_charrefs=True)
        self.skip_ids = set(skip_ids)
        self.parts: list[str] = []
        self._panel_depth = 0
        self._in_code = 0

    def handle_starttag(self, tag, attrs):
        if self._panel_depth:
            self._panel_depth += tag == "div"
            return
        if tag == "div" and dict(attrs).get("id") in self.skip_ids:
            self._panel_depth = 1
            return
        if tag in ("script", "style", "noscript", "template"):
            self._in_code += 1
        self.parts.append(" ")

    def handle_endtag(self, tag):
        if self._panel_depth:
            self._panel_depth -= tag == "div"
            return
        if tag in ("script", "style", "noscript", "template"):
            self._in_code = max(0, self._in_code - 1)
        self.parts.append(" ")

    def handle_data(self, data):
        if not self._panel_depth and not self._in_code:
            self.parts.append(data)


def page_text(html: str) -> str:
    p = _VisibleText()
    p.feed(html)
    p.close()
    return re.sub(r"\s+", " ", "".join(p.parts)).strip()


def parse_site_page(html: str, site: str) -> dict:
    """BASE version, citation year, DOI and data years from a site page's visible text.

    Matches e.g. "Citation: A. Suyker (2026), AmeriFlux BASE US-Ne1 Mead -
    irrigated continuous maize site, Ver. 20-5, AmeriFlux AMP, (Dataset).
    https://doi.org/10.17190/AMF/1246084" and "Years Data Available:
    AmeriFlux BASE 2001 - 2025".  Missing items are None.
    """
    text = page_text(html)
    s = re.escape(site)
    out = {"base_version": None, "base_name": None, "base_citation_year": None, "base_doi": None,
           "base_years_available": None, "base_first_year": None, "base_last_year": None,
           "years_data_collected": None}
    m = re.search(rf"(?:\((?P<year>\d{{4}})\),\s*)?AmeriFlux BASE {s}\b(?P<name>.{{0,200}}?),\s*"
                  rf"Ver\.\s*(?P<ver>\d+-\d+)(?P<tail>.{{0,120}})", text)
    if m:
        out["base_version"] = m["ver"]
        out["base_name"] = m["name"].strip()
        out["base_citation_year"] = int(m["year"]) if m["year"] else None
        d = re.search(r"https?://doi\.org/(10\.\d{4,9}/[^\s,;]+)", m["tail"])
        out["base_doi"] = d[1].rstrip(".") if d else None
    m = re.search(r"Years Data Available:\s*AmeriFlux BASE\s+(\d{4})(?:\s*-\s*(\d{4}|Present))?", text)
    if m:
        first = int(m[1])
        last = int(m[2]) if m[2] and m[2].isdigit() else (None if m[2] else first)
        out.update(base_first_year=first, base_last_year=last,
                   base_years_available=f"{first}-{m[2]}" if m[2] else f"{first}")
    m = re.search(r"Years Data Collected:\s*(\d{4})(?:\s*-\s*(\d{4}|Present))?", text)
    if m:
        out["years_data_collected"] = f"{m[1]}-{m[2]}" if m[2] else m[1]
    return out


def site_page_info(site: str, cache: str | Path, session=None, refresh: bool = False) -> dict:
    """Parsed site page (:func:`parse_site_page`) plus its URL, access time and md5.

    The page HTML is cached as ``cache/siteinfo_<SITE>.html`` and reused unless
    ``refresh``.  The page shows the current BASE version, which changes over
    time; the cached copy is what the paper uses.
    """
    url = SITE_PAGE_URL.format(site=site)
    data, meta = _cached_fetch(url, Path(cache) / SITE_PAGE_FILE.format(site=site), session, refresh)
    info = parse_site_page(data.decode("utf-8", errors="replace"), site)
    info.update(site=site, page_url=url, page_accessed_utc=meta.get("accessed_utc"), page_md5=meta["md5"])
    return info
