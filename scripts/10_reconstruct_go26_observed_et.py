"""Rebuild GO26's observed ET from AmeriFlux BASE files and test how it was computed.

Golkar & de Oliveira (2026; GO26, Section 2.1) compute observed ET as
LE/lambda (lambda = 2.45 MJ/kg) from AmeriFlux BASE latent heat fluxes; from
these "we removed days for which more than 50% of the half hourly values were
absent or gap filled, and retained records passing the standard AmeriFlux
quality flags", and they close the energy balance with the Bowen-ratio method.
For each of their 22 towers and study periods (their Table 1,
config/go26_sites.csv) this script rebuilds, from one BASE file per site:

  * daily ET without closure on valid days (>= 50% of the day's LE records;
    BASE LE holds no gap-filled values), Bowen-ratio closed ET on valid
    energy-balance days (LE, H, NETRAD and G; G from the PI gap-filled series,
    or 0, where BASE has no measured G), and the daily sum of sub-daily ET
    with missing records and days counted as zero and no closure, which is
    what pandas' resample("D").sum() returns
  * site means of each over the study period, their differences from GO26's
    Table 1, the implied zero fraction f = 1 - Table 1 mean / valid-day mean,
    the share of days without valid LE and the share of sub-daily records
    with valid LE; the first and last valid LE in the file and the study-period
    days before the first one (installation)
  * GO26's quality-flag step, as an option (--qc-flags): LE is removed where
    its flag LE_SSITC_TEST (at the LE position, etcomment.ameriflux.read_base;
    0 best, 2 poor) is 2, before the day filter, the means and the zero-filled
    sums, and the results are reported next to the default in columns ending
    in _qc (summary and daily table).  le_flag_column names the flag used
    (empty where the file has none: then the _qc values equal the default;
    AmeriFlux's variable-availability summary lists the flag at 13 of the 22
    sites).  The default uses every record, as the co-author's reproduction
    of Table 1 did; the winter check and the OpenET bias use the default
  * a winter check: Dec-Feb means of unclosed and Bowen-ratio closed ET and
    the days with a Bowen-ratio factor < 0 or > 2; at US-UiC also January and
    February 2020, where GO26 Fig. 8 shows observed ET of 0.62 and 0.89 mm/d on
    average and up to 2.3 mm/d in central Illinois
  * with the OpenET monthly extract (scripts/02), the OpenET ensemble bias
    against monthly ET rebuilt from valid days (mean of valid days x days in
    month, months with >= 20 valid days; unclosed and Bowen-ratio closed) and
    against zero-filled monthly sums, by GO26 climate zone and season, which
    shows how much of GO26's positive OpenET bias the zero fill creates.
    OpenET values are not CC BY: the OpenET mean and bias of a row are left
    blank wherever they would let a single month's OpenET value be solved for
    (fewer than six site-months, or by subtraction; withhold_openet)

A co-author's run with BASE files downloaded on 7 April 2026 matched GO26's
Table 1 with the zero-filled sums of all records (no quality flags) within
0.02 mm/d at 18 of the 22 sites and within 0.03 mm/d at 21 (US-UTV: 0.12
mm/d; the Mead files were version 19-5, hourly, whereas GO26 used 20-5).  The
version of every file is recorded and compared with the version GO26 cite
(config column base_version_go26, else their reference list); rerun on those
versions before citing numbers.

Inputs
  data/raw/ameriflux_base/AMF_<SITE>_BASE_<HH|HR>_<ver>.csv  AmeriFlux BASE,
      CC BY 4.0, AmeriFlux account needed; cite each site's BASE DOI (in
      config/go26_sites.csv).  With several versions of a site the one GO26
      cite is used (--base-version go26, default), else the latest.
      --download --agree-policy fetches the current versions through the
      AmeriFlux data-download service with the credentials in
      AMERIFLUX_USER_ID and AMERIFLUX_USER_EMAIL (environment or .env);
      AmeriFlux logs each download and reports it to the site teams.
      Provenance goes to data/raw/ameriflux_base/provenance.json.
  data/raw/openet_api/openet_monthly.csv  OpenET monthly ET (optional)

Outputs (results/)
  go26_observed_et_reconstruction.csv  one row per site ('BASE file not found'
                                       without a BASE file; *_qc columns with
                                       --qc-flags)
  go26_reconstructed_daily.csv         daily series and flags (CC BY 4.0
                                       AmeriFlux BASE derivative; '#' header
                                       with the site DOIs)
  go26_reconstructed_winter.csv        Dec-Feb check; Jan-Feb 2020 at US-UiC
  go26_reconstructed_openet_bias.csv   OpenET bias by zone, season and variant
                                       (OpenET values withheld where single
                                       months could be solved for)
Read the last three with pandas.read_csv(path, comment="#").

Usage:
  python scripts/10_reconstruct_go26_observed_et.py [--qc-flags]
  python scripts/10_reconstruct_go26_observed_et.py --download --agree-policy [--sites US-UiC US-UTJ]
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from etcomment import ameriflux as af
from etcomment import closure as cl
from etcomment import openet_compare as oc

ROOT = Path(__file__).resolve().parents[1]
LAMBDA = 2.45  # MJ/kg (GO26 Eq. 1)
MJ_PER_WM2_DAY = 0.0864
SEASONS = ["DJF", "MAM", "JJA", "SON"]
SEASON = {m: SEASONS[(m % 12) // 3] for m in range(1, 13)}
# BASE versions in GO26's reference list; used only where config/go26_sites.csv has no base_version_go26
GO26_BASE_VERSIONS = {
    "US-ARM": "15-5", "US-Bi1": "16-5", "US-Bi2": "21-5", "US-IAB": "1-5", "US-MN1": "2-5", "US-MN2": "2-5",
    "US-MN3": "2-5", "US-Mo1": "5-5", "US-NSb": "1-5", "US-Ne1": "20-5", "US-Ne2": "20-5", "US-Ne3": "20-5",
    "US-RGA": "4-5", "US-RGB": "5-5", "US-UTD": "2-5", "US-UTJ": "3-5", "US-UTV": "4-5", "US-UiC": "2-5",
    "US-UiG": "1-5", "US-VT1": "4-5", "US-WT1": "4-5", "US-xSL": "10-5",
}
GO26_SEASON_BIAS = {"DJF": 0.081, "MAM": 0.412, "JJA": 1.342, "SON": 0.573}  # GO26 Table 4, OpenET, mm/d
# GO26 Fig. 8 (US-UiC, 2020) observed ET, rounded; replaced by the digitized series (scripts/04) when present
GO26_FIG8_UIC = {"2020-01": {"go26_fig8_obs_mean": 0.62}, "2020-02": {"go26_fig8_obs_mean": 0.89},
                 "2020-01/02": {"go26_fig8_obs_max": 2.3}}
UIC_WINDOWS = {"2020-01": [1], "2020-02": [2], "2020-01/02": [1, 2]}
QC_BAD = 2  # LE_SSITC_TEST value removed by --qc-flags

LE_STATS = ["valid_LE_days", "valid_EB_days", "retained_pct", "f_days_missing", "le_record_availability",
            "mean_et_unclosed_valid", "mean_et_unclosed_zero_filled", "mean_et_hh_sum_zero_filled_unclosed",
            "mean_et_br_valid", "mean_et_br_zero_filled", "n_br_factor_negative", "n_br_factor_gt2"]
DIFF_KEYS = ["mean_et_hh_sum_zero_filled_unclosed", "mean_et_unclosed_zero_filled", "mean_et_unclosed_valid",
             "mean_et_br_valid"]
QC_COLUMNS = (["le_flag_column", "le_records_flag2"] + [f"{k}_qc" for k in LE_STATS]
              + [f"diff_{k}_qc" for k in DIFF_KEYS] + ["f_implied_qc"])
SUMMARY_COLUMNS = [
    "site", "status", "koppen_go26", "period_start", "period_end", "base_doi",
    "base_files", "base_resolution", "base_version", "base_versions_available", "base_version_go26",
    "version_matches_go26", "le_column", "h_column", "netrad_column", "g_source",
    "base_first_valid_LE", "base_last_valid_LE", "calendar_days", "period_days_before_first_LE",
    "period_days_after_last_LE", *LE_STATS, "go26_table1_mean_obs_et",
    *(f"diff_{k}" for k in DIFF_KEYS), "f_implied", *QC_COLUMNS, "notes",
]
COUNT_COLUMNS = ["calendar_days", "period_days_before_first_LE", "period_days_after_last_LE", "valid_LE_days",
                 "valid_EB_days", "n_br_factor_negative", "n_br_factor_gt2", "go26_fig8_obs_days",
                 "le_records_flag2", "valid_LE_days_qc", "valid_EB_days_qc", "n_br_factor_negative_qc",
                 "n_br_factor_gt2_qc"]
DAILY_COLUMNS = ["site", "date", "le_frac", "et_unclosed", "et_br", "br_factor", "et_hh_sum_zero_filled", "flags"]
QC_DAILY = ["le_frac", "et_unclosed", "et_br", "et_hh_sum_zero_filled"]  # also written as *_qc with --qc-flags


def period_days(start: str, end: str) -> pd.DatetimeIndex:
    """Calendar days from the first day of month ``start`` to the last day of month ``end`` ('YYYY-MM')."""
    return pd.date_range(f"{start}-01", pd.Period(end, "M").end_time.normalize(), freq="D")


def join_flags(flags: dict[str, pd.Series]) -> np.ndarray:
    parts = [np.where(np.asarray(m, bool), name, "") for name, m in flags.items()]
    return np.array([";".join(p for p in row if p) for row in zip(*parts)], dtype=object)


def _le_variant(le: pd.Series, period: pd.DatetimeIndex, lam: float, h: pd.Series, rn: pd.Series,
                g: pd.Series, energy_ok: pd.Series) -> tuple[dict, dict]:
    """Period statistics (LE_STATS) and daily series of one sub-daily LE series.

    ``energy_ok`` marks the days with valid H, NETRAD and G; ``h``, ``rn`` and ``g`` are daily means.
    """
    step = af.time_step(le.index)
    d = af.daily(le.to_frame("LE")).reindex(period)
    le_frac = d["LE_frac"].fillna(0.0)
    le_ok = le_frac >= 0.5
    eb_ok = le_ok & energy_ok
    le_br, factor = cl.bowen_ratio_closure(d["LE"], h, rn, g)
    to_et = MJ_PER_WM2_DAY / lam
    et_unclosed = (d["LE"] * to_et).where(le_ok)
    et_br = pd.Series(le_br * to_et, index=period).where(eb_ok)
    factor = pd.Series(factor, index=period).where(eb_ok)
    # GO26 Table 1 as reproduced: sub-daily ET summed per calendar day with missing records
    # and days counted as zero, no closure, averaged over every calendar day of the period
    le_in = le[(le.index >= period[0]) & (le.index < period[-1] + pd.Timedelta("1D"))]
    et_zero = af.zero_filled_daily_sum(af.et_per_step(le_in, lam, step)).reindex(period, fill_value=0.0)
    n_days, n_le = len(period), int(le_ok.sum())
    stats = {"valid_LE_days": n_le, "valid_EB_days": int(eb_ok.sum()), "retained_pct": 100 * n_le / n_days,
             "f_days_missing": 1 - n_le / n_days,
             "le_record_availability": int(le_in.notna().sum()) / (n_days * int(pd.Timedelta("1D") / step)),
             "mean_et_unclosed_valid": float(et_unclosed.mean()),
             "mean_et_unclosed_zero_filled": float(et_unclosed.fillna(0.0).mean()),
             "mean_et_hh_sum_zero_filled_unclosed": float(et_zero.mean()),
             "mean_et_br_valid": float(et_br.mean()), "mean_et_br_zero_filled": float(et_br.fillna(0.0).mean()),
             "n_br_factor_negative": int((factor < 0).sum()), "n_br_factor_gt2": int((factor > 2).sum())}
    days = {"le_frac": le_frac, "le_ok": le_ok, "eb_ok": eb_ok, "et_unclosed": et_unclosed, "et_br": et_br,
            "br_factor": factor, "et_hh_sum_zero_filled": et_zero}
    return stats, days


def reconstruct_site(base_dir: Path, site: str, start: str, end: str, go26_version: str = "",
                     prefer_version: str | None = None, lam: float = LAMBDA,
                     qc_flags: bool = False) -> tuple[dict, pd.DataFrame | None]:
    """Summary row and daily table of one site over its study period (no table without a BASE file).

    ``prefer_version`` selects among several BASE versions (default: ``go26_version``; '' = latest).
    ``qc_flags`` adds the variant with LE removed where LE_SSITC_TEST is 2 (keys and columns *_qc).
    """
    go26_version = af.normalize_version(go26_version)
    prefer = go26_version if prefer_version is None else af.normalize_version(prefer_version)
    files = sorted(Path(base_dir).glob(f"AMF_{site}_BASE_*.csv"))
    sel = af.select_base_file(files, prefer=prefer)
    out: dict = {"site": site, "base_version_go26": go26_version}
    if sel is None:
        out["status"] = "BASE file not found" if not files else "BASE file name not recognised"
        return out, None
    hh = af.read_base(sel["path"])
    used = hh.attrs.get("columns", {})
    notes = []
    header = af.base_header(sel["path"])
    header_version = af.normalize_version(header.get("version"))
    if header_version and header_version != sel["version"]:
        notes.append(f"file header says version {header_version}")
    if header.get("site", site) != site:
        notes.append(f"file header says site {header['site']}")
    if len(hh) > 1 and af.time_step(hh.index) != pd.Timedelta("30min" if sel["resolution"] == "HH" else "1h"):
        notes.append(f"time step {af.time_step(hh.index)} does not match {sel['resolution']}")
    if len(sel["versions"]) > 1:
        notes.append(f"versions present {', '.join(sel['versions'])}, used {sel['version']}")
    out.update(base_files=sel["path"].name, base_resolution=sel["resolution"], base_version=sel["version"],
               base_versions_available=";".join(sel["versions"]),
               version_matches_go26=(sel["version"] == go26_version) if go26_version else pd.NA)
    if "LE" not in hh or not hh["LE"].notna().any():
        out.update(status="no valid LE in BASE file", notes="; ".join(notes))
        return out, None

    flag = hh.pop("LE_SSITC_TEST") if "LE_SSITC_TEST" in hh else None
    period = period_days(start, end)
    d = af.daily(hh).reindex(period)

    def frac(v: str) -> pd.Series:
        return d[f"{v}_frac"].fillna(0.0) if f"{v}_frac" in d else pd.Series(0.0, index=period)

    def mean(v: str) -> pd.Series:
        return d[v] if v in d else pd.Series(np.nan, index=period)

    if (frac("G") > 0).any():
        g, g_ok, g_source = mean("G"), frac("G") >= 0.5, "measured: " + ",".join(used["G"])
    elif (frac("G_PI_F") > 0).any():
        g, g_ok, g_source = mean("G_PI_F"), frac("G_PI_F") >= 0.5, "PI gap-filled: " + ",".join(used["G_PI_F"])
    else:
        g, g_ok, g_source = pd.Series(0.0, index=period), pd.Series(True, index=period), "assumed 0 (none in BASE)"
    eb_inputs = (period, lam, mean("H"), mean("NETRAD"), g, (frac("H") >= 0.5) & (frac("NETRAD") >= 0.5) & g_ok)
    stats, days = _le_variant(hh["LE"], *eb_inputs)

    valid = hh.index[hh["LE"].notna()]
    first, last = valid.min().normalize(), valid.max().normalize()
    out.update(
        status="ok", le_column=",".join(used.get("LE", [])), h_column=",".join(used.get("H", [])),
        netrad_column=",".join(used.get("NETRAD", [])), g_source=g_source,
        base_first_valid_LE=str(first.date()), base_last_valid_LE=str(last.date()),
        calendar_days=len(period), period_days_before_first_LE=int((period < first).sum()),
        period_days_after_last_LE=int((period > last).sum()), **stats)

    le_frac, le_ok, eb_ok, factor = days["le_frac"], days["le_ok"], days["eb_ok"], days["br_factor"]
    flags = {
        "no_LE": le_frac == 0,                                  # zero-filled sum is 0
        "LE_lt_50pct": (le_frac > 0) & (le_frac < 0.5),          # GO26's own rule drops the day
        "LE_partial": (le_frac >= 0.5) & (le_frac < 1),          # kept; zero-filled sum too low
        "before_first_LE": pd.Series(period < first, index=period),
        "after_last_LE": pd.Series(period > last, index=period),
        "EB_missing": le_ok & ~eb_ok,
        "G_PI_F": eb_ok & g_source.startswith("PI"),
        "G_zero": eb_ok & g_source.startswith("assumed"),
        "BR_factor_negative": factor < 0,
        "BR_factor_gt2": factor > 2,
    }
    values = ("le_frac", "et_unclosed", "et_br", "br_factor", "et_hh_sum_zero_filled")
    table = pd.DataFrame({"site": site, "date": period, **{k: days[k].to_numpy() for k in values},
                          "flags": join_flags(flags)})
    columns = DAILY_COLUMNS
    if qc_flags:  # GO26: "retained records passing the standard AmeriFlux quality flags"
        bad = flag.eq(QC_BAD) if flag is not None else pd.Series(False, index=hh.index)
        in_period = (hh.index >= period[0]) & (hh.index < period[-1] + pd.Timedelta("1D"))
        stats_qc, days_qc = _le_variant(hh["LE"].mask(bad), *eb_inputs)
        out.update(le_flag_column=",".join(used.get("LE_SSITC_TEST", [])),
                   le_records_flag2=int((bad & hh["LE"].notna() & in_period).sum()),
                   **{f"{k}_qc": v for k, v in stats_qc.items()})
        if flag is None:
            notes.append("no LE_SSITC_TEST at the LE position (_qc values = default)")
        for k in QC_DAILY:
            table[f"{k}_qc"] = days_qc[k].to_numpy()
        columns = DAILY_COLUMNS + [f"{k}_qc" for k in QC_DAILY]
    out["notes"] = "; ".join(notes)
    return out, table[columns]


def site_observed_et(base_dir: Path, site: str, start: str, end: str, go26_version: str = "",
                     prefer_version: str | None = None, qc_flags: bool = False) -> dict:
    """Summary row of :func:`reconstruct_site`."""
    return reconstruct_site(base_dir, site, start, end, go26_version, prefer_version, qc_flags=qc_flags)[0]


def _f_implied(go26_mean: float, valid_mean: float) -> float:
    return 1 - go26_mean / valid_mean if np.isfinite(valid_mean) and valid_mean != 0 else np.nan


def compare_with_go26(res: dict, go26_mean: float) -> dict:
    """Add GO26's Table 1 mean, the differences from it and the implied zero fraction (also for *_qc)."""
    res["go26_table1_mean_obs_et"] = go26_mean
    for sfx in ("", "_qc"):
        for k in DIFF_KEYS:
            if k + sfx in res:
                res[f"diff_{k}{sfx}"] = res[k + sfx] - go26_mean
    res["f_implied"] = _f_implied(go26_mean, res.get("mean_et_unclosed_valid", np.nan))
    if "mean_et_unclosed_valid_qc" in res:
        res["f_implied_qc"] = _f_implied(go26_mean, res["mean_et_unclosed_valid_qc"])
    return res


def winter_rows(site: str, table: pd.DataFrame, fig8: dict | None = None) -> list[dict]:
    """Dec-Feb statistics of a site's daily table (plus Jan and Feb 2020 at US-UiC)."""
    t = table.set_index("date")
    windows = {"DJF": t.index.month.isin([12, 1, 2])}
    if site == "US-UiC":
        windows |= {k: (t.index.year == 2020) & t.index.month.isin(m) for k, m in UIC_WINDOWS.items()}
    rows = []
    for label, mask in windows.items():
        s = t[mask]
        eb = s["et_br"].notna()
        rows.append({
            "site": site, "window": label, "days": len(s), "valid_LE_days": int(s["et_unclosed"].notna().sum()),
            "mean_et_unclosed_valid": s["et_unclosed"].mean(), "max_et_unclosed_valid": s["et_unclosed"].max(),
            "valid_EB_days": int(eb.sum()), "mean_et_unclosed_on_EB_days": s.loc[eb, "et_unclosed"].mean(),
            "mean_et_br_valid": s["et_br"].mean(), "max_et_br_valid": s["et_br"].max(),
            "n_br_factor_negative": int((s["br_factor"] < 0).sum()), "n_br_factor_gt2": int((s["br_factor"] > 2).sum()),
            "mean_et_hh_sum_zero_filled": s["et_hh_sum_zero_filled"].mean(),
            **((fig8 or {}).get(label, {}) if site == "US-UiC" else {})})
    return rows


def go26_fig8_reference(digitized: Path) -> dict:
    """GO26 Fig. 8 observed ET at US-UiC in Jan and Feb 2020 (digitized series if available)."""
    if not digitized.exists():
        return GO26_FIG8_UIC
    obs = pd.read_csv(digitized, parse_dates=["date"]).set_index("date")["obs"]
    out = {}
    for label, months in UIC_WINDOWS.items():
        s = obs[(obs.index.year == 2020) & obs.index.month.isin(months)]
        out[label] = {"go26_fig8_obs_days": int(s.count()), "go26_fig8_obs_mean": s.mean(),
                      "go26_fig8_obs_max": s.max()}
    return out


def monthly_observed(daily: pd.DataFrame, min_valid_days: int = 20) -> pd.DataFrame:
    """Monthly observed ET (mm/month) per site: valid-day means x days in month, and zero-filled sums."""
    d = daily.assign(month=daily["date"].dt.to_period("M").dt.to_timestamp())
    m = d.groupby(["site", "month"]).agg(
        days=("date", "size"), n_valid_LE=("et_unclosed", "count"), n_valid_EB=("et_br", "count"),
        mean_unclosed=("et_unclosed", "mean"), mean_br=("et_br", "mean"),
        obs_zero_filled=("et_hh_sum_zero_filled", "sum")).reset_index()
    dim = m["month"].dt.days_in_month
    m["obs_unclosed"] = (m["mean_unclosed"] * dim).where(m["n_valid_LE"] >= min_valid_days)
    m["obs_br"] = (m["mean_br"] * dim).where(m["n_valid_EB"] >= min_valid_days)
    m["obs_zero_filled"] = m["obs_zero_filled"].where(m["days"] == dim)
    m["season"] = m["month"].dt.month.map(SEASON)
    return m.drop(columns=["mean_unclosed", "mean_br"])


def openet_bias(monthly: pd.DataFrame, openet: pd.DataFrame, zones: dict, reported: dict | None = None,
                min_months: int = oc.MIN_STRATUM_MONTHS) -> pd.DataFrame:
    """OpenET ensemble minus observed monthly ET, by GO26 climate zone and season (mm/d).

    month_set 'valid_LE_months': months with enough valid LE days, on which all three
    observed variants are compared (the Bowen-ratio variant only where it has enough valid
    energy-balance days); 'all_months': every month of the study period with OpenET data,
    against the zero-filled sums only, as GO26's records would have been.  OpenET values
    are not CC BY: openet_mm_d and bias_mm_d are blanked where single months could be
    solved for (:func:`withhold_openet` with ``min_months``; 0 keeps every value, not for
    publication).
    """
    o = openet.rename(columns={"date": "month", "et_ensemble": "openet"})[["site", "month", "openet"]]
    o = o.assign(month=pd.to_datetime(o["month"]))
    pairs = monthly.merge(o, on=["site", "month"], how="inner").dropna(subset=["openet"])
    pairs["koppen_go26"] = pairs["site"].map(zones)
    sets = {"valid_LE_months": [("unclosed_valid_days", "obs_unclosed"), ("br_closed_valid_days", "obs_br"),
                                ("zero_filled_table1", "obs_zero_filled")],
            "all_months": [("zero_filled_table1", "obs_zero_filled")]}
    reported = reported or {}
    rows, members = [], []
    for zone in [*sorted(pairs["koppen_go26"].dropna().unique()), "all"]:
        for season in [*SEASONS, "all"]:
            sel = pd.Series(True, index=pairs.index)
            if zone != "all":
                sel &= pairs["koppen_go26"] == zone
            if season != "all":
                sel &= pairs["season"] == season
            for month_set, variants in sets.items():
                in_set = pairs["obs_unclosed"].notna() if month_set == "valid_LE_months" else True
                for variant, col in variants:
                    s = pairs[sel & in_set & pairs[col].notna()]
                    if s.empty:
                        continue
                    dim = s["month"].dt.days_in_month
                    p, ob = s["openet"] / dim, s[col] / dim
                    rows.append({"koppen_go26": zone, "season": season, "month_set": month_set, "variant": variant,
                                 "n_sites": int(s["site"].nunique()), "n_months": len(s),
                                 "openet_mm_d": p.mean(), "observed_mm_d": ob.mean(), "bias_mm_d": (p - ob).mean(),
                                 "go26_reported_bias_mm_d": reported.get((zone, season), np.nan)})
                    members.append(frozenset(s.index))
    out = pd.DataFrame(rows)
    return withhold_openet(out, members, min_months) if min_months and len(out) else out


def _determined_by(sets: list[frozenset], size: np.ndarray, shown: list[int]) -> int | None:
    """Smallest set in a combination of the ``shown`` sets that determines a single site-month, or None.

    Each set is one equation (the mean of its site-months' values).  A site-month is determined when its
    unit vector lies in the row space of the sets' indicator vectors (projection 1, exact up to rounding).
    """
    if not shown:
        return None
    cols = {m: j for j, m in enumerate(sorted(set().union(*(sets[i] for i in shown))))}
    a = np.zeros((len(shown), len(cols)))
    for r, i in enumerate(shown):
        a[r, [cols[m] for m in sets[i]]] = 1.0
    _, sv, vt = np.linalg.svd(a, full_matrices=False)
    rank = int((sv > sv.max() * max(a.shape) * np.finfo(float).eps).sum())
    hit = np.flatnonzero((vt[:rank] ** 2).sum(axis=0) > 1 - 1e-9)
    if not hit.size:
        return None
    e = np.zeros(len(cols))
    e[hit[0]] = 1.0
    c = np.linalg.lstsq(a.T, e, rcond=None)[0]  # least-norm combination of the sets giving that site-month
    return min((shown[r] for r in np.flatnonzero(np.abs(c) > 1e-9 * np.abs(c).max())), key=lambda i: size[i])


def withhold_openet(bias: pd.DataFrame, members: list, min_months: int = oc.MIN_STRATUM_MONTHS) -> pd.DataFrame:
    """Blank openet_mm_d and bias_mm_d of :func:`openet_bias` rows so that no single month's OpenET value can be
    solved for from the table and the public observed ET.

    ``members[i]`` holds the site-months (pair labels) of row i.  With the observed ET public
    (go26_reconstructed_daily.csv), a row is one linear equation on its site-months' OpenET values (their
    mean of OpenET / days in month); rows over the same site-months carry the same equation and are kept or
    withheld together.  Withheld, in this order (column withheld_rule):

    * n_months: rows with fewer than ``min_months`` site-months;
    * difference: rows of one zone and season whose site-months differ by 1 to ``min_months`` - 1 from those
      of a kept row (kept: the most rows, then the most site-months, then the first);
    * margin: parts of a published total (a zone over its seasons, all zones over the zones, for one
      month_set and variant) that would leave 1 to ``min_months`` - 1 site-months to subtraction, the
      smallest first, until those are 0 or at least ``min_months`` (as openet_compare.withheld_stations);
    * rank: while the published rows still determine one site-month (exact rank test), the smallest row of
      a combination that gives it.

    n_sites, n_months and observed_mm_d are kept.
    """
    out = bias.reset_index(drop=True)
    keys = [frozenset(m) for m in members]
    index = {s: i for i, s in enumerate(dict.fromkeys(keys))}
    sets, sid = list(index), np.array([index[k] for k in keys])
    size = np.array([len(s) for s in sets])
    rule: dict[int, str] = {int(i): "n_months" for i in np.flatnonzero(size < min_months)}

    for _, g in out.groupby(["koppen_go26", "season"], sort=False):
        pos = g.index.to_numpy()
        ids = [int(i) for i in dict.fromkeys(sid[pos]) if int(i) not in rule]
        n_rows = {i: int((sid[pos] == i).sum()) for i in ids}
        ok = [keep for k in range(len(ids), -1, -1) for keep in itertools.combinations(ids, k)
              if all(len(sets[a] ^ sets[b]) >= min_months for a, b in itertools.combinations(keep, 2))]
        best = max(ok, key=lambda keep: (sum(n_rows[i] for i in keep), sum(size[i] for i in keep)))
        rule.update({i: "difference" for i in ids if i not in best})

    at = {k: int(i) for k, i in zip(zip(out["koppen_go26"], out["season"], out["month_set"], out["variant"]), sid)}
    zones = [z for z in dict.fromkeys(out["koppen_go26"]) if z != "all"]
    seasons = [s for s in dict.fromkeys(out["season"]) if s != "all"]
    margins = ([((z, "all"), [(z, s) for s in seasons]) for z in [*zones, "all"]]
               + [(("all", s), [(z, s) for z in zones]) for s in [*seasons, "all"]])
    groups = list(dict.fromkeys(zip(out["month_set"], out["variant"])))

    def fix_margins() -> None:
        changed = True
        while changed:
            changed = False
            for g in groups:
                for parent, parts in margins:
                    p = at.get((*parent, *g))
                    if p is None or p in rule:
                        continue
                    kids = [at[(*c, *g)] for c in parts if (*c, *g) in at and at[(*c, *g)] not in rule]
                    hidden = len(sets[p] - set().union(*(sets[i] for i in kids)))
                    if 0 < hidden < min_months:  # kids is not empty: a published row has >= min_months
                        rule[min(kids, key=lambda i: size[i])] = "margin"
                        changed = True

    fix_margins()
    while (i := _determined_by(sets, size, [j for j in range(len(sets)) if j not in rule])) is not None:
        rule[i] = "rank"
        fix_margins()
    out["withheld_rule"] = [rule.get(int(i), "") for i in sid]
    out["openet_withheld"] = out["withheld_rule"] != ""
    out.loc[out["openet_withheld"], ["openet_mm_d", "bias_mm_d"]] = np.nan
    return out


def go26_reported_biases(table3: Path) -> dict:
    """GO26's OpenET biases: Table 3 by zone and overall, Table 4 by season (mm/d)."""
    rep = {("all", s): b for s, b in GO26_SEASON_BIAS.items()}
    if table3.exists():
        for zone, bias in pd.read_csv(table3)[["zone", "openet_bias"]].itertuples(index=False):
            if zone == "Overall (all sites)":
                rep[("all", "all")] = bias
            elif not str(zone).startswith("Overall"):
                rep[(zone, "all")] = bias
    return rep


def write_csv(df: pd.DataFrame, path: Path, header: list[str]) -> None:
    with open(path, "w", newline="") as fh:
        for line in header:
            fh.write(f"# {line}\n")
        df.to_csv(fh, index=False, float_format="%.4f")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-dir", default=str(ROOT / "data/raw/ameriflux_base"), help="folder with the BASE CSVs")
    ap.add_argument("--config-dir", default=str(ROOT / "config"), help="holds go26_sites.csv and go26_table3.csv")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--sites", nargs="+", help="subset of the GO26 sites (default: all 22)")
    ap.add_argument("--base-version", choices=["go26", "latest"], default="go26",
                    help="which BASE version to use where several are present")
    ap.add_argument("--openet", default=str(ROOT / "data/raw/openet_api/openet_monthly.csv"),
                    help="OpenET monthly CSV from scripts/02 for the bias table ('' to skip)")
    ap.add_argument("--min-valid-days", type=int, default=20, help="valid days for a month to count")
    ap.add_argument("--qc-flags", action="store_true",
                    help="also report the variant with LE removed where its quality flag LE_SSITC_TEST is 2 "
                         "(GO26's quality-flag step; columns *_qc)")
    ap.add_argument("--download", action="store_true",
                    help="download missing BASE files first (AmeriFlux account; needs --agree-policy)")
    ap.add_argument("--agree-policy", action="store_true",
                    help=f"confirm that you have read and accept the AmeriFlux CC-BY-4.0 data policy "
                         f"({af.DATA_POLICY_URL})")
    ap.add_argument("--intended-use", choices=sorted(af.INTENDED_USE), default="remote_sensing")
    ap.add_argument("--description", default=af.DEFAULT_DESCRIPTION,
                    help="intended use, recorded by AmeriFlux and sent to the site teams")
    ap.add_argument("--force-download", action="store_true",
                    help="request and download again even where BASE files or zips are present")
    args = ap.parse_args(argv)
    if args.download and not args.agree_policy:
        ap.error(f"--download needs --agree-policy (AmeriFlux CC-BY-4.0 data policy, {af.DATA_POLICY_URL})")

    base_dir, out_dir = Path(args.base_dir), Path(args.out_dir)
    sites = pd.read_csv(Path(args.config_dir) / "go26_sites.csv")
    if args.sites:
        unknown = sorted(set(args.sites) - set(sites["site"]))
        if unknown:
            ap.error(f"not GO26 sites: {unknown}")
        sites = sites[sites["site"].isin(args.sites)]
    config_versions = (sites.set_index("site")["base_version_go26"].map(af.normalize_version)
                       if "base_version_go26" in sites else pd.Series(dtype=str))
    if "base_version_go26" not in sites:
        print("config has no base_version_go26 column; using the versions in GO26's reference list")
    go26_versions = {s: config_versions.get(s) or GO26_BASE_VERSIONS.get(s, "") for s in sites["site"]}

    if args.download:
        print(af.CCBY4_GUIDELINES)
        user_id, user_email = af.credentials(ROOT)
        af.download_base(sites["site"], base_dir, user_id, user_email, agree_policy=True,
                         intended_use=args.intended_use, description=args.description, force=args.force_download)

    rows, tables = [], []
    for r in sites.itertuples(index=False):
        res, table = reconstruct_site(base_dir, r.site, r.period_start, r.period_end, go26_versions[r.site],
                                      prefer_version=None if args.base_version == "go26" else "",
                                      qc_flags=args.qc_flags)
        res.update(koppen_go26=r.koppen_go26, period_start=r.period_start, period_end=r.period_end,
                   base_doi=r.base_doi)
        rows.append(compare_with_go26(res, r.mean_obs_et_mm_d))
        if table is not None:
            tables.append(table)
    out = pd.DataFrame(rows)
    out = out[[c for c in SUMMARY_COLUMNS if c in out] + [c for c in out if c not in SUMMARY_COLUMNS]]
    for c in COUNT_COLUMNS:
        if c in out:
            out[c] = out[c].astype("Int64")
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "go26_observed_et_reconstruction.csv", index=False, float_format="%.4f")

    ok = out["status"].eq("ok")
    print(f"{ok.sum()} of {len(out)} sites have a BASE file in {base_dir}")
    for status, g in out[~ok].groupby("status"):
        print(f"  {status}: {', '.join(g['site'])}")
    if not ok.any():
        print("No BASE files: daily, winter and OpenET-bias outputs not written.")
        return
    show = ["site", "base_version", "base_version_go26", "valid_LE_days", "calendar_days", "le_record_availability",
            "mean_et_unclosed_valid", "mean_et_br_valid", "mean_et_hh_sum_zero_filled_unclosed",
            "go26_table1_mean_obs_et", "diff_mean_et_hh_sum_zero_filled_unclosed", "f_implied", "f_days_missing"]
    print("\n" + out.loc[ok, show].round(3).to_string(index=False) + "\n")
    d = out.loc[ok, "diff_mean_et_hh_sum_zero_filled_unclosed"].abs()
    print(f"Zero-filled sub-daily sums vs GO26 Table 1: within 0.02 mm/d at {(d <= 0.02).sum()} and within "
          f"0.03 mm/d at {(d <= 0.03).sum()} of {ok.sum()} sites")
    mismatch = [f"{r['site']} {r['base_version']} (GO26 {r['base_version_go26']})" for r in rows
                if r.get("status") == "ok" and r.get("version_matches_go26") is False]
    if mismatch:
        print("BASE versions differing from GO26's: " + ", ".join(mismatch))
    print("If go26_table1_mean_obs_et tracks the zero-filled variants rather than mean_et_*_valid, "
          "missing data were counted as zero ET.")
    if args.qc_flags:
        q = out.loc[ok, "diff_mean_et_hh_sum_zero_filled_unclosed_qc"].abs()
        print(f"\nWith LE removed where LE_SSITC_TEST = {QC_BAD} (--qc-flags; columns *_qc): zero-filled sums within "
              f"0.02 mm/d of Table 1 at {(q <= 0.02).sum()} and within 0.03 mm/d at {(q <= 0.03).sum()} of "
              f"{ok.sum()} sites")
        show_qc = ["site", "le_flag_column", "le_records_flag2", "valid_LE_days_qc", "le_record_availability_qc",
                   "mean_et_unclosed_valid_qc", "mean_et_br_valid_qc", "mean_et_hh_sum_zero_filled_unclosed_qc",
                   "diff_mean_et_hh_sum_zero_filled_unclosed_qc", "f_implied_qc"]
        print(out.loc[ok, show_qc].round(3).to_string(index=False))

    daily = pd.concat(tables, ignore_index=True)
    cite = [f"{r.site} BASE {r.base_version}: https://doi.org/{r.base_doi}" for r in out[ok].itertuples(index=False)]
    qc_note = ([f"*_qc (--qc-flags): the same with LE removed where its quality flag LE_SSITC_TEST = {QC_BAD}."]
               if args.qc_flags else [])
    write_csv(daily, out_dir / "go26_reconstructed_daily.csv", [
        "Daily ET rebuilt from AmeriFlux BASE data by scripts/10_reconstruct_go26_observed_et.py. AmeriFlux BASE",
        f"data are CC BY 4.0 (AmeriFlux CC-BY-4.0 data policy, {af.DATA_POLICY_URL}); this derivative is",
        "CC BY 4.0 too. Cite each site's BASE data product:", *cite,
        "le_frac: share of the day's sub-daily records with valid LE; et_unclosed: LE/2.45 MJ/kg (mm/d) on",
        "days with le_frac >= 0.5; et_br: daily Bowen-ratio closed ET on days with valid LE, H, NETRAD and G;",
        "br_factor: (NETRAD - G)/(LE + H) on those days; et_hh_sum_zero_filled: daily sum of sub-daily ET",
        "with missing records counted as zero (pandas resample('D').sum() default; reproduces GO26 Table 1).",
        "flags: no_LE, LE_lt_50pct, LE_partial, before_first_LE, after_last_LE, EB_missing, G_PI_F, G_zero,",
        "BR_factor_negative, BR_factor_gt2.", *qc_note])

    fig8 = go26_fig8_reference(out_dir / "digitized_US-UiC.csv")
    winter = pd.DataFrame([w for t in tables for w in winter_rows(t["site"].iloc[0], t, fig8)])
    if "go26_fig8_obs_days" in winter:
        winter["go26_fig8_obs_days"] = winter["go26_fig8_obs_days"].astype("Int64")
    write_csv(winter, out_dir / "go26_reconstructed_winter.csv", [
        "Winter (Dec-Feb) check of daily ET rebuilt from AmeriFlux BASE data (CC BY 4.0; site DOIs in",
        "go26_observed_et_reconstruction.csv), all records (no quality flags); go26_fig8_*: GO26 Fig. 8.",
        "Means in mm/d over valid days, except mean_et_hh_sum_zero_filled (all calendar days, missing data "
        "counted as zero)."])
    uic = winter[winter["site"].eq("US-UiC")]
    if len(uic):
        print("\nUS-UiC winter (mm/d):")
        print(uic.drop(columns="site").round(2).to_string(index=False))

    openet_path = Path(args.openet) if args.openet else None
    if openet_path is None or not openet_path.exists():
        print("No OpenET monthly file: bias table not written.")
        return
    openet = pd.read_csv(openet_path, parse_dates=["date"])
    zones = dict(zip(sites["site"], sites["koppen_go26"]))
    bias = openet_bias(monthly_observed(daily, args.min_valid_days), openet, zones,
                       go26_reported_biases(Path(args.config_dir) / "go26_table3.csv"))
    n = oc.MIN_STRATUM_MONTHS
    write_csv(bias, out_dir / "go26_reconstructed_openet_bias.csv", [
        "OpenET ensemble (OpenET, https://openetdata.org; OpenET API Terms of Service) minus observed ET",
        "rebuilt from AmeriFlux BASE data (CC BY 4.0; site DOIs in go26_observed_et_reconstruction.csv).",
        f"Monthly pairs in each GO26 study period; valid-day variants need >= {args.min_valid_days} valid days;",
        "all records (no quality flags). bias_mm_d = mean over site-months of (OpenET - observed)/days in month.",
        "go26_reported: Tables 3-4.",
        "OpenET values withheld (openet_withheld, withheld_rule): openet_mm_d and bias_mm_d are blank in rows with",
        f"fewer than {n} site-months (n_months), in rows of one zone and season whose site-months differ by 1-{n - 1}",
        "from those of a kept row (difference), in parts of a published zone or season total that would leave",
        f"1-{n - 1} site-months to subtraction (margin), and in rows that would still determine one site-month",
        "(rank), so no single month's OpenET value can be solved for from this table and the observed ET. n_sites,",
        "n_months and observed_mm_d are kept."])
    if len(bias):
        print(f"\nOpenET values withheld in {int(bias['openet_withheld'].sum())} of {len(bias)} rows (fewer than {n} "
              "site-months, or a single month solvable by subtraction; column withheld_rule)")
        tab = bias[bias["season"].eq("all")].pivot_table(index="koppen_go26", columns=["month_set", "variant"],
                                                         values="bias_mm_d")
        print("OpenET bias (mm/d) by GO26 zone, all seasons:")
        print(tab.round(2).to_string() if not tab.empty else "all withheld")


if __name__ == "__main__":
    main()
