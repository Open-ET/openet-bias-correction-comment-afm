"""AmeriFlux BASE files: reader, version provenance and an optional downloader.

BASE files (AMF_<SITE>_BASE_HH_<ver>.csv for half-hourly records,
AMF_<SITE>_BASE_HR_<ver>.csv for hourly ones) carry two header lines starting
with '#' (site and version), then TIMESTAMP_START, TIMESTAMP_END and the
variables, with -9999 for missing values (Chu et al., 2023).  Variables may
carry position qualifiers _<H>_<V>_<R> (horizontal and vertical position,
replicate); a position is the unqualified name or one (H, V) pair.  Column
rules of :func:`read_base`:

* LE, H and NETRAD come from one position: the one with the most valid
  records in the file, ties going to the unqualified column, then to the
  lowest position.  H is taken at the position used for LE when it has data
  there, so that the Bowen ratio comes from one eddy-covariance system
  (US-ARM, for example, also reports LE and H at two other levels).
  Replicates at the chosen position are averaged.
* Other variables (G, TA, VPD, WS): the unqualified column when it has any
  data, else all qualified positions averaged (e.g. soil heat flux plates).
  VPD is read from VPD_PI (its name at the GO26 sites) or VPD.
* Gap-filled variables (_F, _PI_F) are not used, except the PI gap-filled
  soil heat flux, kept separately as ``G_PI_F`` because it is the only G at
  some sites (the Mead towers).
* LE_SSITC_TEST, the LE quality flag (steady-state and integral turbulence
  characteristics tests, Foken et al., 2004: 0 best, 1 moderate, 2 poor), is
  returned for the LE position when the file has it there (or an unqualified
  flag when all LE data are at one position, as at US-VT1); with several
  replicates the highest flag is kept.

Golkar & de Oliveira (2026, Section 2.1) write: "we removed days for which
more than 50% of the half hourly values were absent or gap filled, and
retained records passing the standard AmeriFlux quality flags".  The daily
aggregation keeps, for every variable, the fraction of valid sub-daily
records, so the day filter can be applied exactly (BASE LE holds no
gap-filled values) and compared with a zero-filled alternative; scripts/10
applies the quality flag as an option.

:func:`download_base` requests BASE-BADM zips from the AmeriFlux
data-download web service with the same fields as amerifluxr's
``amf_download_base`` (which appends " [amerifluxr download]" to the
description; here the description is sent as given): a POST with the user id
and e-mail, data product BASE-BADM, data policy CCBY4.0, site ids,
intended-use category and a description returns one link per site.
AmeriFlux records every request in its data-use log and reports it to the
site teams, so sites whose zip is already on disk are not requested again.
Credentials come only from the ``AMERIFLUX_USER_ID`` and
``AMERIFLUX_USER_EMAIL`` environment variables or a ``.env`` file; they are
never printed or stored, and the query string of the returned links, which
may identify the user, is dropped before anything is recorded.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import pandas as pd
import requests

VARS = {
    "LE": r"^LE(_\d+_\d+_\d+)?$",
    "H": r"^H(_\d+_\d+_\d+)?$",
    "NETRAD": r"^NETRAD(_\d+_\d+_\d+)?$",
    "G": r"^G(_\d+_\d+_\d+)?$",
    "TA": r"^TA(_\d+_\d+_\d+)?$",
    "VPD": r"^VPD(_PI)?(_\d+_\d+_\d+)?$",
    "WS": r"^WS(_\d+_\d+_\d+)?$",
}
G_PI_F = r"^G_PI_F(_\d+_\d+_\d+)?$"
LE_FLAG = r"^LE_SSITC_TEST(_\d+_\d+_\d+)?$"
QUALIFIER = re.compile(r"_(\d+)_(\d+)_(\d+)$")
ONE_POSITION = ("LE", "H", "NETRAD")

SITE_ID = re.compile(r"^[A-Z]{2}-[A-Za-z0-9]{3}$")
BASE_CSV = re.compile(r"^AMF_(?P<site>[A-Z]{2}-[A-Za-z0-9]{3})_BASE_(?P<resolution>HH|HR)_(?P<version>\d+-\d+)\.csv$")
BASE_ZIP = re.compile(r"^AMF_(?P<site>[A-Z]{2}-[A-Za-z0-9]{3})_BASE-BADM_(?P<version>\d+-\d+)\.zip$")

DATA_DOWNLOAD_URL = "https://amfcdn.lbl.gov/api/v1/data_download"
DATA_POLICY_URL = "https://ameriflux.lbl.gov/data/data-policy/"
INTENDED_USE = {  # amerifluxr::amf_download_base categories
    "synthesis": "Research - Multi-site synthesis",
    "remote_sensing": "Research - Remote sensing",
    "model": "Research - Land model/Earth system model",
    "other_research": "Research - Other",
    "education": "Education (Teacher or Student)",
    "other": "Other",
}
CCBY4_GUIDELINES = (
    "Data use guidelines for AmeriFlux CC-BY-4.0 Data Policy:\n"
    "(1) Data user is free to Share (copy and redistribute the material in any medium or format) "
    "and/or Adapt (remix, transform, and build upon the material) for any purpose.\n"
    "(2) Provide a citation to each site data product that includes the data-product DOI and/or "
    "recommended publication.\n"
    "(3) Acknowledge funding for supporting AmeriFlux data portal: U.S. Department of Energy Office "
    "of Science.")
DEFAULT_DESCRIPTION = ("Rebuilding the observed ET of Golkar and de Oliveira (2026, Agric. For. Meteorol. "
                       "391, 111524) from AmeriFlux BASE for a Comment on that paper")


class AmeriFluxError(RuntimeError):
    pass


# -- file names and versions -----------------------------------------------------------------------
def parse_base_name(name: str | Path) -> dict | None:
    """Site, resolution (HH half-hourly, HR hourly) and version of a BASE CSV file name."""
    m = BASE_CSV.match(Path(name).name)
    return m.groupdict() if m else None


def normalize_version(v) -> str:
    """'20-5' from '20-5', 'Ver. 20-5' or 'v20-5'; '' when there is none (also for NaN/None)."""
    m = re.search(r"\d+-\d+", str(v)) if v is not None else None
    return m.group(0) if m else ""


def version_key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v))


def select_base_file(paths, prefer: str = "") -> dict | None:
    """The BASE file to use among ``paths`` of one site.

    Version ``prefer`` (e.g. the one GO26 cite) when present, else the latest;
    a half-hourly file is preferred over an hourly one of the same version.
    Returns path, site, resolution, version and all versions found.
    """
    parsed = [(Path(p), m) for p in paths if (m := parse_base_name(p))]
    if not parsed:
        return None
    versions = sorted({m["version"] for _, m in parsed}, key=version_key)
    prefer = normalize_version(prefer)
    version = prefer if prefer in versions else versions[-1]
    path, meta = min((pm for pm in parsed if pm[1]["version"] == version),
                     key=lambda pm: (pm[1]["resolution"] != "HH", pm[0].name))
    return {"path": path, **meta, "versions": versions}


def base_header(path: str | Path) -> dict:
    """Metadata in the '#' lines that open a BASE file, e.g. {'site': 'US-Ne1', 'version': '20-5'}."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            key, sep, value = line[1:].partition(":")
            if sep:
                out[key.strip().lower()] = value.strip()
    return out


# -- reading and daily aggregation -------------------------------------------------------------------
def position(column: str) -> tuple[int, ...]:
    """(H, V) position of a column with an _H_V_R qualifier (LE_1_2_1 -> (1, 2)); () without one."""
    m = QUALIFIER.search(column)
    return (int(m[1]), int(m[2])) if m else ()


def best_position(valid: dict) -> tuple[int, ...]:
    """Position with the most valid records; ties go to the unqualified one (), then the lowest position."""
    return max(sorted(valid), key=lambda k: valid[k])


def _by_position(cols: list[str]) -> dict[tuple, list[str]]:
    groups: dict[tuple, list[str]] = {}
    for c in cols:
        groups.setdefault(position(c), []).append(c)
    return groups


def _choose_columns(df: pd.DataFrame, name: str, cols: list[str], prefer: tuple | None = None):
    """Columns behind variable ``name`` and the position they share (None when positions are mixed)."""
    groups = _by_position(cols)
    counts = {k: int(df[v].notna().any(axis=1).sum()) for k, v in groups.items()}
    if name in ONE_POSITION:
        if prefer in groups and counts[prefer] > 0:
            return groups[prefer], prefer
        key = best_position(counts)
        return groups[key], key
    if () in groups and counts[()] > 0:
        return groups[()], ()
    qualified = [c for k in sorted(groups) if k != () for c in groups[k]]
    return (qualified or groups[()]), None


def _flag_columns(df: pd.DataFrame, flags: list[str], le_cols: list[str], le_position) -> list[str]:
    """LE_SSITC_TEST columns for the LE in use: at its position, else an unqualified flag when all LE
    data are at that one position; [] when neither has data."""
    groups = _by_position(flags)
    has_data = {k: bool(df[v].notna().to_numpy().any()) for k, v in groups.items()}
    if has_data.get(le_position):
        return groups[le_position]
    le_positions = {position(c) for c in le_cols if df[c].notna().any()}
    if has_data.get(()) and le_positions == {le_position}:
        return groups[()]
    return []


def read_base(path: str | Path) -> pd.DataFrame:
    """Sub-daily LE, H, NETRAD, G, G_PI_F, TA, VPD, WS and LE_SSITC_TEST indexed by TIMESTAMP_START.

    Only variables present in the file are returned; the column rules are in
    the module docstring.  ``attrs["columns"]`` records the BASE columns
    behind each variable.
    """
    pats = {k: re.compile(v) for k, v in VARS.items()}
    gpf, flag = re.compile(G_PI_F), re.compile(LE_FLAG)

    def wanted(c: str) -> bool:
        return c == "TIMESTAMP_START" or any(p.match(c) for p in (gpf, flag, *pats.values()))

    df = pd.read_csv(path, comment="#", usecols=wanted, dtype={"TIMESTAMP_START": str},
                     na_values=[-9999, "-9999", -9999.0])
    t = pd.to_datetime(df.pop("TIMESTAMP_START").str.strip(), format="%Y%m%d%H%M")
    if len(df.columns):
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.mask(df <= -9999)
    df.index = t
    df = df[~df.index.duplicated()].sort_index()
    out = pd.DataFrame(index=df.index)
    used: dict[str, list[str]] = {}
    le_position = None
    for name, pat in pats.items():
        cols = [c for c in df.columns if pat.match(c)]
        if not cols:
            continue
        use, pos = _choose_columns(df, name, cols, prefer=le_position if name == "H" else None)
        if name == "LE":
            le_position = pos
        out[name] = df[use].mean(axis=1, skipna=True)
        used[name] = use
    gap_filled_g = [c for c in df.columns if gpf.match(c)]
    if gap_filled_g:
        out["G_PI_F"] = df[gap_filled_g].mean(axis=1, skipna=True)
        used["G_PI_F"] = gap_filled_g
    if "LE" in used:
        flags = _flag_columns(df, [c for c in df.columns if flag.match(c)],
                              [c for c in df.columns if pats["LE"].match(c)], le_position)
        if flags:
            out["LE_SSITC_TEST"] = df[flags].max(axis=1, skipna=True)
            used["LE_SSITC_TEST"] = flags
    out.index.name = "timestamp"
    out.attrs["columns"] = used
    return out


def time_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Most common interval between records (30 min in HH files, 1 h in HR files)."""
    return index.to_series().diff().mode().iloc[0]


def daily(hh: pd.DataFrame) -> pd.DataFrame:
    """Daily means plus the fraction of valid sub-daily records per variable."""
    per_day = int(pd.Timedelta("1D") / time_step(hh.index))
    g = hh.groupby(hh.index.floor("D"))
    means = g.mean()
    frac = g.count() / per_day
    frac.columns = [f"{c}_frac" for c in frac.columns]
    d = pd.concat([means, frac], axis=1)
    d.index.name = "date"
    return d


def zero_filled_daily_sum(hh_et: pd.Series) -> pd.Series:
    """Daily totals of a sub-daily series with missing values counted as zero.

    This is what ``Series.resample("D").sum()`` returns by default
    (``min_count=0``): missing records add nothing and a day without any data
    sums to 0.  Golkar & de Oliveira's (2026) Table 1 site means are
    reproduced by averaging such totals of half-hourly ET over calendar days.
    """
    return hh_et.resample("D").sum(min_count=0)


def et_per_step(le_wm2: pd.Series, lambda_mj_kg: float = 2.45, step: pd.Timedelta | None = None) -> pd.Series:
    """Convert LE (W m-2) to ET (mm per record) at ``step`` (default: the series' own time step)."""
    step = time_step(le_wm2.index) if step is None else pd.Timedelta(step)
    return le_wm2 * step.total_seconds() / (lambda_mj_kg * 1e6)


# -- download (AmeriFlux account needed) ---------------------------------------------------------------
def _env_value(raw: str, name: str) -> str:
    """Value of a .env assignment: quoted text verbatim, else the text before an inline ' #' comment."""
    raw = raw.strip()
    if raw[:1] in ("'", '"'):
        end = raw.find(raw[0], 1)
        if end < 0 or not re.fullmatch(r"\s*(#.*)?", raw[end + 1:]):
            raise AmeriFluxError(f"{name} in .env: unmatched quote or text after the closing quote")
        return raw[1:end]
    return "" if raw.startswith("#") else re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def credentials(root: str | Path | None = None) -> tuple[str, str]:
    """AmeriFlux user id and e-mail from the environment or a ``.env`` file in ``root``.

    ``.env`` lines are ``NAME=value`` (optionally after ``export``): a quoted
    value is taken verbatim, an unquoted one up to an inline comment (``#``
    after whitespace).  Values containing whitespace, and an e-mail without
    the form name@domain.tld, are rejected; error messages never repeat the
    values.
    """
    names = ("AMERIFLUX_USER_ID", "AMERIFLUX_USER_EMAIL")
    vals = {k: os.environ.get(k, "").strip() for k in names}
    source = {k: "environment" for k in names}
    env = Path(root) / ".env" if root is not None else None
    if not all(vals.values()) and env is not None and env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            name, sep, value = line.partition("=")
            name = name.strip()
            if sep and name in vals and not vals[name]:
                vals[name], source[name] = _env_value(value, name), ".env"
    missing = [k for k in names if not vals[k]]
    if missing:
        raise AmeriFluxError(f"set {' and '.join(missing)} (environment variables or a .env file in the "
                             "repository root) to download AmeriFlux BASE data")
    for k in names:
        if re.search(r"\s", vals[k]):
            raise AmeriFluxError(f"{k} ({source[k]}) contains whitespace; give a single word, quoted or "
                                 "unquoted, with any comment after ' #' or on its own line")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", vals["AMERIFLUX_USER_EMAIL"]):
        raise AmeriFluxError("AMERIFLUX_USER_EMAIL is not an e-mail address")
    return vals["AMERIFLUX_USER_ID"], vals["AMERIFLUX_USER_EMAIL"]


def download_request(user_id: str, user_email: str, site_ids, intended_use: str = "remote_sensing",
                     description: str = DEFAULT_DESCRIPTION) -> dict:
    """JSON body of the data_download request: the fields amerifluxr's amf_download_base sends.

    amerifluxr appends " [amerifluxr download]" to the description; here it is sent as given.
    """
    if intended_use not in INTENDED_USE:
        raise ValueError(f"intended_use must be one of {sorted(INTENDED_USE)}")
    return {"user_id": user_id, "user_email": user_email, "data_product": "BASE-BADM",
            "data_policy": "CCBY4.0", "site_ids": list(site_ids), "intended_use": INTENDED_USE[intended_use],
            "description": description, "is_test": ""}


def safe_url(url: str, secrets=()) -> str:
    """``url`` without user info, query string or fragment, and with any of ``secrets`` redacted."""
    p = urlsplit(url)
    host = (p.hostname or "") + (f":{p.port}" if p.port else "")
    out = urlunsplit((p.scheme, host, p.path, "", ""))
    for s in secrets:
        if s:
            for form in {s, quote(s, safe="")}:
                out = out.replace(form, "<redacted>")
    return out


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def extract_base_csvs(zip_path: str | Path, dest: str | Path) -> list[dict]:
    """Extract only the BASE CSV files of a BASE-BADM zip into ``dest`` (directory parts dropped)."""
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            meta = parse_base_name(name)
            if info.is_dir() or meta is None:
                continue
            target = Path(dest) / name
            with z.open(info) as src, open(target, "wb") as fh:
                shutil.copyfileobj(src, fh)
            out.append({"file": name, "resolution": meta["resolution"], "version": meta["version"],
                        "md5": _md5(target)})
    return sorted(out, key=lambda e: e["file"])


def _fetch(http, url: str, dest: Path, shown: str, timeout_s: float) -> None:
    """Stream ``url`` to ``dest`` via a temporary file; errors name only the cleaned link ``shown``."""
    tmp = dest.with_name(dest.name + ".part")
    try:
        try:
            with http.get(url, stream=True, timeout=timeout_s) as r:
                if r.status_code != 200:
                    raise AmeriFluxError(f"{shown}: HTTP {r.status_code}")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            fh.write(chunk)
        except requests.RequestException as e:
            raise AmeriFluxError(f"download of {shown} failed ({type(e).__name__})") from None
        if not zipfile.is_zipfile(tmp):
            raise AmeriFluxError(f"{shown} is not a zip file")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")  # never a truncated provenance file
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def download_base(site_ids, out_dir: str | Path, user_id: str, user_email: str, *, agree_policy: bool = False,
                  intended_use: str = "remote_sensing", description: str = DEFAULT_DESCRIPTION,
                  force: bool = False, session=None, timeout_s: float = 600.0, log=print) -> list[dict]:
    """Download BASE-BADM zips (CC-BY-4.0 policy) for ``site_ids`` and extract their BASE CSVs.

    Unless ``force``, sites that already have a BASE CSV in ``out_dir`` are
    skipped, and sites whose BASE-BADM zip is already there are extracted from
    it without a new request (AmeriFlux logs each request and reports it to
    the site teams); a cached zip without a BASE CSV is reported, not
    requested again.  Each zip is recorded in ``out_dir/provenance.json``
    (site, zip, version, link without query string, md5, size, UTC access
    time, the intended use and description sent with the request that
    fetched it, extracted CSVs and their md5) as soon as it is on disk, so a
    failed download keeps the records of the earlier ones.  A zip this call
    did not fetch keeps the access time, intended use and description of its
    earlier record (null without one), and its link too when no request is
    sent.  Returns the provenance entries of this call.
    """
    if not agree_policy:
        raise AmeriFluxError(f"read and accept the AmeriFlux CC-BY-4.0 data policy ({DATA_POLICY_URL}) "
                             "before downloading (agree_policy=True, --agree-policy)")
    if intended_use not in INTENDED_USE:
        raise ValueError(f"intended_use must be one of {sorted(INTENDED_USE)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sites = sorted(dict.fromkeys(str(s) for s in site_ids))
    bad = [s for s in sites if not SITE_ID.match(s)]
    if bad:
        raise ValueError(f"not AmeriFlux site ids: {bad}")
    prov_path = out_dir / "provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    prov.update(source=f"AmeriFlux BASE-BADM via {DATA_DOWNLOAD_URL} (request fields as "
                       "amerifluxr::amf_download_base, without its ' [amerifluxr download]' description tag)",
                data_policy="CCBY4.0", license="CC BY 4.0; cite each site's BASE data-product DOI")
    entries = {e["zip"]: e for e in prov.get("files", [])}
    new: list[dict] = []

    def record(dest: Path, url: str | None, accessed: str | None, request: dict) -> None:
        """Provenance of ``dest``; ``request`` holds the intended use and description sent with the request
        that fetched it (the earlier record for a zip this call did not fetch)."""
        if not zipfile.is_zipfile(dest):
            raise AmeriFluxError(f"{dest} is not a zip file; delete it to download it again")
        csvs = extract_base_csvs(dest, out_dir)
        if not csvs:
            log(f"{dest.name}: no BASE CSV inside")
        meta = BASE_ZIP.match(dest.name)
        entry = {"site": meta["site"] if meta else (csvs[0]["file"][4:10] if csvs else ""),
                 "zip": dest.name, "version": meta["version"] if meta else (csvs[0]["version"] if csvs else ""),
                 "url": url, "md5": _md5(dest), "bytes": dest.stat().st_size, "accessed_utc": accessed,
                 "intended_use": request.get("intended_use"), "description": request.get("description"),
                 "base_csv": csvs}
        entries[dest.name] = entry
        new.append(entry)
        prov["files"] = [entries[k] for k in sorted(entries)]
        _write_json(prov_path, prov)
        log(f"{dest.name}: {entry['bytes'] / 1e6:.1f} MB, md5 {entry['md5']}, "
            f"{', '.join(c['file'] for c in csvs) or 'no BASE CSV'}")

    todo = sites if force else [s for s in sites if not any(out_dir.glob(f"AMF_{s}_BASE_*.csv"))]
    if not force:  # zips already on disk: extract them instead of sending a new request
        cached = {s: sorted(p for p in out_dir.glob(f"AMF_{s}_BASE-BADM_*.zip") if BASE_ZIP.match(p.name))
                  for s in todo}
        for z in (z for s in todo for z in cached[s]):
            log(f"{z.name}: already downloaded, extracting (no new request)")
            old = entries.get(z.name, {})
            record(z, old.get("url"), old.get("accessed_utc"), old)
        todo = [s for s in todo if not cached[s]]
    if not todo:
        if not new:
            log("every requested site already has a BASE file; nothing to download")
        return new
    http = session if session is not None else requests
    secrets = (user_id, user_email)
    body = download_request(user_id, user_email, todo, intended_use, description)
    try:
        r = http.post(DATA_DOWNLOAD_URL, json=body, headers={"Content-Type": "application/json"}, timeout=120)
    except requests.RequestException as e:
        raise AmeriFluxError(f"data_download request failed ({type(e).__name__})") from None
    if r.status_code != 200:
        raise AmeriFluxError(f"data_download returned HTTP {r.status_code}")
    try:
        links = r.json().get("data_urls", [])
    except (ValueError, AttributeError):
        raise AmeriFluxError("data_download returned no list of data_urls") from None
    urls = [u.get("url") if isinstance(u, dict) else u for u in links]
    urls = [u for u in urls if isinstance(u, str) and u]
    if not urls:
        raise AmeriFluxError(f"no data returned for {', '.join(todo)} (CC-BY-4.0 BASE-BADM)")

    for url in urls:
        name = Path(urlsplit(url).path).name
        shown = safe_url(url, secrets)
        if not name.endswith(".zip"):
            log(f"skipped {shown}: not a zip file")
            continue
        dest = out_dir / name
        if dest.exists() and not force:  # not fetched after this request: keep the record of the one that was
            log(f"{name}: already downloaded, extracting")
            old = entries.get(name, {})
            record(dest, shown, old.get("accessed_utc"), old)
        else:
            _fetch(http, url, dest, shown, timeout_s)
            record(dest, shown, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   {"intended_use": body["intended_use"], "description": body["description"]})
    got = {e["site"] for e in new}
    if missing := [s for s in todo if s not in got]:
        log(f"no CC-BY-4.0 BASE-BADM data returned for: {', '.join(missing)}")
    return new
