"""Rebuild GO26's observed ET from AmeriFlux BASE files and test how it was computed.

Golkar & de Oliveira (2026; GO26, Section 2.1) compute observed ET as
LE/lambda (lambda = 2.45 MJ/kg) from AmeriFlux BASE latent heat fluxes; from
these "we removed days for which more than 50% of the half hourly values were
absent or gap filled, and retained records passing the standard AmeriFlux
quality flags", and they close the energy balance with the Bowen-ratio method.
For each of their 22 towers and study periods (their Table 1,
config/go26_sites.csv) this script rebuilds, from the BASE file of the version
GO26 cite (one file per site; all inputs offline):

  * GO26's Table 1 as reproduced: the daily sum of sub-daily ET with missing
    records and days counted as zero and no closure, which is what pandas'
    resample("D").sum() returns, averaged over the calendar days of the study
    period and, where a file ends before the period does, also over the days
    up to the file's last record (a daily series built from the file stops
    there)
  * daily ET on valid days (>= 50% of the day's records with LE; BASE LE holds
    no gap-filled values): the mean of the valid records (et_unclosed), and
    the mean with the missing records filled by the mean diurnal variation
    (et_unclosed_mdv: the mean of the same time of day over a centred 15-day
    window, etcomment.ameriflux.mdv_daily), which removes the bias of days
    whose LE is missing mostly at night
  * energy-balance closure on every valid day by flux-data-qaqc (Volk et
    al., 2021; version pinned in environment.yml), run on the daily means of
    LE, H, NETRAD and G over the sub-daily records in which all four are
    valid, on days where these cover >= 50% of the day (energy-balance days;
    etcomment.closure.matched_daily_means), so that all four terms describe
    the same hours; its daily correction factor multiplies the valid-day LE
    (etcomment.closure.fdq_factor).  EBR (et_ebr, et_ebr_mdv; the main
    closed variant): flux-data-qaqc's 'ebr' correction (the FLUXNET2015/
    ONEFlux energy-balance-ratio method with its limits); days without
    energy-balance terms take the factor its moving windows, adjacent years
    or day-of-year climatology give, and stay without one where none does.
    Bowen ratio as GO26 state it (et_br_fdq): their Section 2.1 names the
    Bowen-ratio method (Twine et al., 2000) and nothing more, and they
    compared it with EBR using flux-data-qaqc's routines, so it is
    flux-data-qaqc's 'br' correction: the valid-day LE times the day's factor
    (NETRAD - G)/(LE + H), unscreened and without gap filling, on
    energy-balance days; days on which flux-data-qaqc's closed LE is <= -100
    or >= 850 W m-2 (or the Bowen ratio is undefined) are left without a
    value; et_br_fdq_mdv: the same factor applied to the gap-filled
    valid-day LE.  Screened Bowen ratio
    (et_br_screened, et_br_screened_mdv; our screen of it): the day's factor
    (NETRAD - G)/(LE + H) where it is 0.5-2 with
    LE + H > 0 and NETRAD - G > 0; every other valid day takes the median of
    the passing factors in a centred 15-day window if the window holds at
    least 5 of them, otherwise the median of all passing factors of the
    period.
    Unscreened Bowen ratio (et_br_unscreened): the day's factor as is, a
    diagnostic only (it diverges as LE + H approaches zero), with its
    failures counted (factor < 0, 0-0.5, > 2, screen failures) and its most
    extreme days.  G is the PI gap-filled series where BASE has no measured
    G (the Mead towers) and 0 where it has none (US-IAB, US-MN1, US-MN2);
    US-WT1 has neither NETRAD nor G, so no closure.  Where two net
    radiometers at one position read differently and report at different
    times (US-UTD, US-UTJ, US-UTV), the one with the most valid records in
    the period is used throughout and the other gives the closed means in
    *_netrad_alt (sensitivity)
  * site means of every variant, their differences from GO26's Table 1, the
    implied zero fraction f = 1 - Table 1 mean / gap-filled valid-day mean
    (f_implied; f_implied_unfilled with the mean of valid records), the share
    of days without valid LE and of sub-daily records with LE, and the first
    and last valid LE and the last record of the file
  * GO26's quality-flag step, as an option (--qc-flags): LE is removed where
    its flag LE_SSITC_TEST (at the LE position, etcomment.ameriflux.read_base;
    0 best, 2 poor) is 2, before the day filter, the gap fill, the closure
    and the zero-filled sums, and the results are reported next to the
    default in columns ending in _qc.  le_flag_column names the flag used
    (empty where the file has none: then the _qc values equal the default).
    The default uses every record, which reproduces Table 1; the other
    outputs use the default
  * a winter check: Dec-Feb means of unclosed and closed ET and the Bowen-ratio
    failures; at US-UiC also January and February 2020, where GO26 Fig. 8
    shows observed ET of 0.62 and 0.89 mm/d on average and up to 2.3 mm/d
  * the BASE versions: per site and year of the study period, the share of
    records with LE in the cited file and in AmeriFlux's public
    variable-availability summary of 20 October 2025 (the copy scripts/11
    cached in data/raw/ameriflux_public/, pinned commit, read offline with
    etcomment.amf_public), which at the 11 sites whose cited version has a
    2026 date describes an earlier version; in-period LE records in each; and
    the month in which that earlier version's LE ends against GO26's period
    end (estimated: the month by which the cited file accumulates the
    summary's records of its last LE year, or the earliest month that can
    hold them where the cited file has fewer)
  * GO26's plotted daily observed ET (digitized by scripts/04 into
    results/digitized_US-UiC.csv, Fig. 8, 2020, and
    results/digitized_US-UTJ.csv, Fig. S2, 2024) against the BASE zero-filled
    daily sums, day by day
  * with the OpenET monthly extract (scripts/02), the OpenET ensemble bias by
    GO26 climate zone and season over two month sets: all site-months with
    OpenET against the zero-filled monthly sums (GO26's observed ET as
    reproduced; next to GO26's reported biases), and the months with >= 20
    days on which every valid-day variant exists (energy-balance days on
    which the Bowen ratio as GO26 state it has a value, so US-WT1 drops out),
    on which the zero-filled sums, the valid-day mean and the Bowen-ratio
    closed mean as GO26 state them, the gap-filled unclosed mean and its
    Bowen-ratio (as GO26 state it), EBR and screened Bowen-ratio closed means
    are compared on identical days and site-months.  OpenET values are not CC BY: the OpenET mean and
    bias of a row are left blank wherever they would let a single month's
    value, or any sum or difference over fewer than six site-months, be
    solved for (withhold_openet)
  * per tower, the OpenET ensemble and the observed ET averaged over the
    same days (Figure 2 of the comment; openet_site_means): every day of the
    study period against the zero-filled sums, and the days on which every
    valid-day variant exists against those variants; with the daily
    extract of scripts/02, the daily ensemble over the same days as a check
    (data/raw/openet_api/derived/, not deposited)

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
  data/raw/ameriflux_public/  amerifluxr availability summary (scripts/11; optional)
  data/raw/openet_api/openet_monthly.csv  OpenET monthly ET (optional)
  data/raw/openet_api/openet_daily_go26.csv  OpenET daily ET (optional)

Outputs (results/; read the CSV files with pandas.read_csv(path, comment="#"))
  go26_observed_et_reconstruction.csv  one row per site ('BASE file not found'
                                       without a BASE file; *_qc columns with
                                       --qc-flags; AmeriFlux BASE derivative)
  go26_reconstructed_daily.csv         daily series, closure factors and flags
                                       (AmeriFlux BASE derivative; '#' header
                                       with the definitions and site DOIs)
  go26_reconstructed_winter.csv        Dec-Feb check; Jan-Feb 2020 at US-UiC
  go26_base_version_check.csv          cited BASE file vs the October 2025
                                       availability summary, per site and year
  go26_reconstructed_figure_check.csv  GO26 Figs. 8 and S2 vs BASE, day by day
  go26_reconstructed_openet_bias.csv   OpenET bias by zone, season, month set
                                       and variant (OpenET values withheld
                                       where they could be solved for)
  go26_reconstructed_openet_sites.csv  OpenET and observed ET over the same
                                       days, per tower and day set (Figure 2)

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
from etcomment import amf_public as amfp
from etcomment import closure as cl
from etcomment import openet_compare as oc

ROOT = Path(__file__).resolve().parents[1]
LAMBDA = 2.45  # MJ/kg (GO26 Eq. 1)
MJ_PER_WM2_DAY = 0.0864
SEASONS = ["DJF", "MAM", "JJA", "SON"]
SEASON = {m: SEASONS[(m % 12) // 3] for m in range(1, 13)}
NOV_MAR = [11, 12, 1, 2, 3]
# BASE versions in GO26's reference list; used only where config/go26_sites.csv has no base_version_go26
GO26_BASE_VERSIONS = {
    "US-ARM": "15-5", "US-Bi1": "16-5", "US-Bi2": "21-5", "US-IAB": "1-5", "US-MN1": "2-5", "US-MN2": "2-5",
    "US-MN3": "2-5", "US-Mo1": "5-5", "US-NSb": "1-5", "US-Ne1": "20-5", "US-Ne2": "20-5", "US-Ne3": "20-5",
    "US-RGA": "4-5", "US-RGB": "5-5", "US-UTD": "2-5", "US-UTJ": "3-5", "US-UTV": "4-5", "US-UiC": "2-5",
    "US-UiG": "1-5", "US-VT1": "4-5", "US-WT1": "4-5", "US-xSL": "10-5",
}
# year of each BASE citation in GO26's reference list (the year the cited version was published; as scripts/11)
GO26_BASE_CITATION_YEAR = {
    "US-ARM": 2026, "US-Bi1": 2026, "US-Bi2": 2026, "US-IAB": 2025, "US-MN1": 2025, "US-MN2": 2025,
    "US-MN3": 2025, "US-Mo1": 2026, "US-NSb": 2025, "US-Ne1": 2026, "US-Ne2": 2026, "US-Ne3": 2026,
    "US-RGA": 2025, "US-RGB": 2025, "US-UTD": 2026, "US-UTJ": 2026, "US-UTV": 2026, "US-UiC": 2025,
    "US-UiG": 2025, "US-VT1": 2025, "US-WT1": 2026, "US-xSL": 2025,
}
GO26_SEASON_BIAS = {"DJF": 0.081, "MAM": 0.412, "JJA": 1.342, "SON": 0.573}  # GO26 Table 4, OpenET, mm/d
# GO26 Fig. 8 (US-UiC, 2020) observed ET, rounded; replaced by the digitized series (scripts/04) when present
GO26_FIG8_UIC = {"2020-01": {"go26_fig8_obs_mean": 0.62}, "2020-02": {"go26_fig8_obs_mean": 0.89},
                 "2020-01/02": {"go26_fig8_obs_max": 2.3}}
UIC_WINDOWS = {"2020-01": [1], "2020-02": [2], "2020-01/02": [1, 2]}
# GO26's daily panels of observed ET, digitized by scripts/04 (file in the results folder)
GO26_FIGURES = {"US-UiC": ("Fig. 8", "digitized_US-UiC.csv"), "US-UTJ": ("Fig. S2", "digitized_US-UTJ.csv")}
SITE_NOTES = {"US-RGB": "flooded rice: water heat storage is not measured, so the Bowen-ratio factor exceeds 2 "
                        "on many days"}
QC_BAD = 2  # LE_SSITC_TEST value removed by --qc-flags
MDV_WINDOW = 15  # days of the mean-diurnal-variation gap fill
# valid-day variants compared on identical days and site-months in the OpenET bias table (variant, daily column):
# GO26's processing as they state it (50% day filter, Bowen-ratio closure), then gap-filled, EBR, screened BR
VALID_VARIANTS = [("unclosed", "et_unclosed"), ("br_fdq", "et_br_fdq"), ("unclosed_mdv", "et_unclosed_mdv"),
                  ("br_fdq_mdv", "et_br_fdq_mdv"), ("ebr_mdv", "et_ebr_mdv"), ("br_screened_mdv", "et_br_screened_mdv")]

COUNT_STATS = ["valid_LE_days", "valid_EB_days", "mdv_unfilled_records", "n_br_factor_negative", "n_br_factor_lt05",
               "n_br_factor_gt2", "n_br_screen_fail", "n_br_fdq_left_out", "n_br_fdq_mdv_left_out",
               "valid_EB_days_nov_mar", "n_br_screen_fail_nov_mar"]
LE_STATS = ["valid_LE_days", "valid_EB_days", "retained_pct", "f_days_missing", "le_record_availability",
            "mdv_filled_share", "mdv_unfilled_records",
            "mean_et_unclosed_valid", "mean_et_unclosed_mdv", "mean_et_unclosed_zero_filled",
            "mean_et_hh_sum_zero_filled_unclosed", "mean_et_hh_sum_zero_filled_file_days",
            "mean_et_br_fdq", "mean_et_br_fdq_mdv", "mean_et_ebr", "mean_et_ebr_mdv", "mean_et_br_screened",
            "mean_et_br_screened_mdv", "mean_et_br_unscreened", "median_ebr_factor",
            "n_br_factor_negative", "n_br_factor_lt05", "n_br_factor_gt2", "n_br_screen_fail", "n_br_fdq_left_out",
            "n_br_fdq_mdv_left_out",
            "valid_EB_days_nov_mar", "n_br_screen_fail_nov_mar",
            "max_et_br_unscreened", "max_et_br_unscreened_date", "min_et_br_unscreened", "min_et_br_unscreened_date"]
DIFF_KEYS = ["mean_et_hh_sum_zero_filled_unclosed", "mean_et_hh_sum_zero_filled_file_days",
             "mean_et_unclosed_zero_filled", "mean_et_unclosed_valid", "mean_et_unclosed_mdv", "mean_et_br_fdq",
             "mean_et_br_fdq_mdv", "mean_et_ebr_mdv", "mean_et_br_screened_mdv"]
NETRAD_ALT_STATS = ["valid_EB_days", "mean_et_br_fdq", "mean_et_br_fdq_mdv", "mean_et_ebr", "mean_et_ebr_mdv",
                    "mean_et_br_screened_mdv", "n_br_screen_fail"]
QC_COLUMNS = (["le_flag_column", "le_records_flag2"] + [f"{k}_qc" for k in LE_STATS]
              + [f"diff_{k}_qc" for k in DIFF_KEYS] + ["f_implied_qc", "f_implied_unfilled_qc"])
SUMMARY_COLUMNS = [
    "site", "status", "koppen_go26", "period_start", "period_end", "base_doi",
    "base_files", "base_resolution", "base_version", "base_versions_available", "base_version_go26",
    "version_matches_go26", "le_column", "h_column", "netrad_column", "netrad_column_alt", "g_source",
    "base_first_valid_LE", "base_last_valid_LE", "base_last_record", "calendar_days", "file_days",
    "period_days_before_first_LE", "period_days_after_last_LE", *LE_STATS, "go26_table1_mean_obs_et",
    *(f"diff_{k}" for k in DIFF_KEYS), "f_implied", "f_implied_unfilled",
    *(f"{k}_netrad_alt" for k in NETRAD_ALT_STATS), *QC_COLUMNS, "notes",
]
COUNT_COLUMNS = (["calendar_days", "file_days", "period_days_before_first_LE", "period_days_after_last_LE",
                  "go26_fig8_obs_days", "le_records_flag2", *COUNT_STATS, *(f"{k}_qc" for k in COUNT_STATS)]
                 + [f"{k}_netrad_alt" for k in NETRAD_ALT_STATS if k in COUNT_STATS])
DAILY_COLUMNS = ["site", "date", "le_frac", "eb_frac", "et_unclosed", "et_unclosed_mdv", "br_factor",
                 "br_factor_screened", "ebr_factor", "et_br_fdq", "et_br_fdq_mdv", "et_ebr", "et_ebr_mdv",
                 "et_br_screened", "et_br_screened_mdv", "et_br_unscreened", "et_hh_sum_zero_filled", "flags"]
QC_DAILY = ["le_frac", "et_unclosed", "et_unclosed_mdv", "et_br_fdq", "et_br_fdq_mdv", "et_ebr_mdv",
            "et_br_screened_mdv", "et_hh_sum_zero_filled"]  # also written as *_qc with --qc-flags


def period_days(start: str, end: str) -> pd.DatetimeIndex:
    """Calendar days from the first day of month ``start`` to the last day of month ``end`` ('YYYY-MM')."""
    return pd.date_range(f"{start}-01", pd.Period(end, "M").end_time.normalize(), freq="D")


def join_flags(flags: dict[str, pd.Series]) -> np.ndarray:
    parts = [np.where(np.asarray(m, bool), name, "") for name, m in flags.items()]
    return np.array([";".join(p for p in row if p) for row in zip(*parts)], dtype=object)


def date_ranges(dates) -> str:
    """Consecutive days compacted: ['2024-07-26', '2024-07-27', '2024-08-06'] -> '2024-07-26..2024-07-27;2024-08-06'."""
    d = pd.DatetimeIndex(sorted(pd.to_datetime(list(dates))))
    if not len(d):
        return ""
    breaks = np.flatnonzero(np.diff(d.values.astype("datetime64[D]").astype(np.int64)) != 1)
    starts, ends = np.r_[0, breaks + 1], np.r_[breaks, len(d) - 1]
    return ";".join(f"{d[a]:%Y-%m-%d}" + (f"..{d[b]:%Y-%m-%d}" if b > a else "") for a, b in zip(starts, ends))


def _extreme(s: pd.Series, which: str) -> tuple[float, str]:
    s = s.dropna()
    if s.empty:
        return np.nan, ""
    day = s.idxmax() if which == "max" else s.idxmin()
    return float(s[day]), f"{day:%Y-%m-%d}"


def _le_variant(le: pd.Series, period: pd.DatetimeIndex, lam: float, eb: pd.DataFrame,
                file_end: pd.Timestamp) -> tuple[dict, dict]:
    """Period statistics (LE_STATS) and daily series of one sub-daily LE series of the study period.

    ``eb`` holds the sub-daily H, NETRAD and G of the same records (G = 0 where BASE has none);
    ``file_end`` is the file's last record.  See the module docstring for the variants.
    """
    step = af.time_step(le.index)
    per_day = int(pd.Timedelta("1D") / step)
    d = af.daily(le.to_frame("LE")).reindex(period)
    le_frac = d["LE_frac"].fillna(0.0)
    le_ok = le_frac >= 0.5
    to_et = MJ_PER_WM2_DAY / lam
    et_unclosed = (d["LE"] * to_et).where(le_ok)
    mdv = af.mdv_daily(le, period, MDV_WINDOW)
    et_mdv = (mdv["mean"] * to_et).where(le_ok)
    s = cl.matched_daily_means(pd.concat([le.rename("LE"), eb[["H", "NETRAD", "G"]]], axis=1), per_day)
    s = s.reindex(period)
    eb_frac = s["frac"].fillna(0.0)
    eb_ok = le_ok & (eb_frac >= 0.5)
    br = cl.screened_br_factor(s["flux"], s["energy"], eb_ok)
    factor, passed = br["factor"], br["passed"]
    f_scr = br["screened"].where(le_ok)
    eb_means = s[list(cl.FDQ_COLUMNS)].where(eb_ok, axis=0)  # flux-data-qaqc's inputs: energy-balance days only
    f_ebr = cl.fdq_factor(eb_means, "ebr").where(le_ok)
    f_fdq = cl.fdq_factor(eb_means, "br")  # Bowen ratio as GO26 state it: NaN where flux-data-qaqc leaves it out
    et_br = et_unclosed * factor  # unscreened: NaN off energy-balance days
    et_fdq = et_unclosed * f_fdq
    et_fdq_mdv = et_mdv * f_fdq
    et_zero = af.zero_filled_daily_sum(af.et_per_step(le, lam, step)).reindex(period, fill_value=0.0)
    in_file = period <= file_end.normalize()
    nov_mar = pd.Series(period.month.isin(NOV_MAR), index=period)
    fail = eb_ok & ~passed
    n_days, n_le = len(period), int(le_ok.sum())
    lo, hi = cl.BR_BOUNDS
    max_br, max_day = _extreme(et_br, "max")
    min_br, min_day = _extreme(et_br, "min")
    stats = {"valid_LE_days": n_le, "valid_EB_days": int(eb_ok.sum()), "retained_pct": 100 * n_le / n_days,
             "f_days_missing": 1 - n_le / n_days,
             "le_record_availability": int(le.notna().sum()) / (n_days * per_day),
             "mdv_filled_share": float(mdv.loc[le_ok, "n_filled"].sum() / (n_le * per_day)) if n_le else np.nan,
             "mdv_unfilled_records": int(mdv.loc[le_ok, "n_unfilled"].sum()),
             "mean_et_unclosed_valid": float(et_unclosed.mean()), "mean_et_unclosed_mdv": float(et_mdv.mean()),
             "mean_et_unclosed_zero_filled": float(et_unclosed.fillna(0.0).mean()),
             "mean_et_hh_sum_zero_filled_unclosed": float(et_zero.mean()),
             "mean_et_hh_sum_zero_filled_file_days": float(et_zero[in_file].mean()),
             "mean_et_br_fdq": float(et_fdq.mean()), "mean_et_br_fdq_mdv": float(et_fdq_mdv.mean()),
             "mean_et_ebr": float((et_unclosed * f_ebr).mean()), "mean_et_ebr_mdv": float((et_mdv * f_ebr).mean()),
             "mean_et_br_screened": float((et_unclosed * f_scr).mean()),
             "mean_et_br_screened_mdv": float((et_mdv * f_scr).mean()), "mean_et_br_unscreened": float(et_br.mean()),
             "median_ebr_factor": float(f_ebr.median()),
             "n_br_factor_negative": int((factor < 0).sum()),
             "n_br_factor_lt05": int(((factor >= 0) & (factor < lo)).sum()),
             "n_br_factor_gt2": int((factor > hi).sum()), "n_br_screen_fail": int(fail.sum()),
             "n_br_fdq_left_out": int((eb_ok & et_fdq.isna()).sum()),
             "n_br_fdq_mdv_left_out": int((eb_ok & et_fdq_mdv.isna()).sum()),
             "valid_EB_days_nov_mar": int((eb_ok & nov_mar).sum()),
             "n_br_screen_fail_nov_mar": int((fail & nov_mar).sum()),
             "max_et_br_unscreened": max_br, "max_et_br_unscreened_date": max_day,
             "min_et_br_unscreened": min_br, "min_et_br_unscreened_date": min_day}
    days = {"le_frac": le_frac, "le_ok": le_ok, "eb_frac": eb_frac, "eb_ok": eb_ok, "et_unclosed": et_unclosed,
            "et_unclosed_mdv": et_mdv, "mdv_unfilled": mdv["n_unfilled"], "br_factor": factor, "br_passed": passed,
            "br_factor_screened": f_scr, "ebr_factor": f_ebr, "et_br_fdq": et_fdq, "et_br_fdq_mdv": et_fdq_mdv,
            "et_ebr": et_unclosed * f_ebr,
            "et_ebr_mdv": et_mdv * f_ebr,
            "et_br_screened": et_unclosed * f_scr, "et_br_screened_mdv": et_mdv * f_scr, "et_br_unscreened": et_br,
            "et_hh_sum_zero_filled": et_zero}
    return stats, days


def reconstruct_site(base_dir: Path, site: str, start: str, end: str, go26_version: str = "",
                     prefer_version: str | None = None, lam: float = LAMBDA, qc_flags: bool = False,
                     extra: dict | None = None) -> tuple[dict, pd.DataFrame | None]:
    """Summary row and daily table of one site over its study period (no table without a BASE file).

    ``prefer_version`` selects among several BASE versions (default: ``go26_version``; '' = latest).
    ``qc_flags`` adds the variant with LE removed where LE_SSITC_TEST is 2 (keys and columns *_qc).
    ``extra``, if given, receives the sub-daily LE of the whole file ('le'; for :func:`version_check`).
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
    replicates = hh.attrs.get("replicates", {}).get("NETRAD", [])
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
    if extra is not None:
        extra["le"] = hh["LE"]

    period = period_days(start, end)
    last_record = hh.index.max()
    valid = hh.index[hh["LE"].notna()]
    first, last = valid.min().normalize(), valid.max().normalize()
    p = hh[(hh.index >= period[0]) & (hh.index < period[-1] + pd.Timedelta("1D"))].copy()
    if len(p) < 2:
        out.update(status="no records in the study period", notes="; ".join(notes))
        return out, None
    flag = p.pop("LE_SSITC_TEST") if "LE_SSITC_TEST" in p else None

    def has(v: str) -> bool:
        return v in p and bool(p[v].notna().any())

    if has("G"):
        g, g_source = p["G"], "measured: " + ",".join(used["G"])
    elif has("G_PI_F"):
        g, g_source = p["G_PI_F"], "PI gap-filled: " + ",".join(used["G_PI_F"])
    else:
        g, g_source = pd.Series(0.0, index=p.index), "assumed 0 (none in BASE)"
    n_rep = {c: int(p[c].notna().sum()) for c in replicates}
    order = sorted(replicates, key=lambda c: (-n_rep[c], c))
    if len(order) > 1 and n_rep[order[1]]:  # two radiometers: one throughout, the other as a sensitivity
        netrad_column, netrad_alt = order[0], order[1]
        rn = p[netrad_column]
        notes.append(f"{len(order)} net radiometers at one position ("
                     + ", ".join(f"{c} {n_rep[c]} records" for c in order)
                     + f" in the period): {netrad_column} used throughout, {netrad_alt} in *_netrad_alt")
    else:
        netrad_column, netrad_alt = (order[0] if order else ",".join(used.get("NETRAD", []))), ""
        rn = p[netrad_column] if order else (p["NETRAD"] if "NETRAD" in p else pd.Series(np.nan, index=p.index))
    eb = pd.DataFrame({"H": p["H"] if "H" in p else np.nan, "NETRAD": rn, "G": g}, index=p.index)
    if not rn.notna().any():
        notes.append("no NETRAD" + (" or G" if g_source.startswith("assumed") else "") + " in BASE: no closure")
    elif g_source.startswith("assumed"):
        notes.append("no soil heat flux in BASE: G = 0 in the closure")
    if site in SITE_NOTES:
        notes.append(SITE_NOTES[site])
    stats, days = _le_variant(p["LE"], period, lam, eb, last_record)
    file_days = int((period <= last_record.normalize()).sum())
    out.update(
        status="ok", le_column=",".join(used.get("LE", [])), h_column=",".join(used.get("H", [])),
        netrad_column=netrad_column, netrad_column_alt=netrad_alt, g_source=g_source,
        base_first_valid_LE=f"{first:%Y-%m-%d}", base_last_valid_LE=f"{last:%Y-%m-%d}",
        base_last_record=f"{last_record:%Y-%m-%d %H:%M}", calendar_days=len(period), file_days=file_days,
        period_days_before_first_LE=int((period < first).sum()), period_days_after_last_LE=int((period > last).sum()),
        **stats)
    if netrad_alt:
        alt, _ = _le_variant(p["LE"], period, lam, eb.assign(NETRAD=p[netrad_alt]), last_record)
        out.update({f"{k}_netrad_alt": alt[k] for k in NETRAD_ALT_STATS})

    le_frac, le_ok, eb_ok, factor = days["le_frac"], days["le_ok"], days["eb_ok"], days["br_factor"]
    lo, hi = cl.BR_BOUNDS
    flags = {
        "no_LE": le_frac == 0,                                  # zero-filled sum is 0
        "LE_lt_50pct": (le_frac > 0) & (le_frac < 0.5),          # GO26's own rule drops the day
        "LE_partial": (le_frac >= 0.5) & (le_frac < 1),          # kept; zero-filled sum too low
        "MDV_unfilled": le_ok & (days["mdv_unfilled"] > 0),     # a time of day without data in 15 days
        "before_first_LE": pd.Series(period < first, index=period),
        "after_last_LE": pd.Series(period > last, index=period),
        "after_file_end": pd.Series(period > last_record.normalize(), index=period),
        "EB_missing": le_ok & ~eb_ok,
        "G_PI_F": eb_ok & g_source.startswith("PI"),
        "G_zero": eb_ok & g_source.startswith("assumed"),
        "BR_factor_negative": factor < 0,
        "BR_factor_lt05": (factor >= 0) & (factor < lo),
        "BR_factor_gt2": factor > hi,
        "BR_screen_fail": eb_ok & ~days["br_passed"],
        "BR_fdq_left_out": eb_ok & days["et_br_fdq"].isna(),
        "BR_fdq_mdv_left_out": eb_ok & days["et_br_fdq_mdv"].isna(),
    }
    values = [c for c in DAILY_COLUMNS if c not in ("site", "date", "flags")]
    table = pd.DataFrame({"site": site, "date": period, **{k: days[k].to_numpy() for k in values},
                          "flags": join_flags(flags)})
    columns = DAILY_COLUMNS
    if qc_flags:  # GO26: "retained records passing the standard AmeriFlux quality flags"
        bad = flag.eq(QC_BAD) if flag is not None else pd.Series(False, index=p.index)
        stats_qc, days_qc = _le_variant(p["LE"].mask(bad), period, lam, eb, last_record)
        out.update(le_flag_column=",".join(used.get("LE_SSITC_TEST", [])),
                   le_records_flag2=int((bad & p["LE"].notna()).sum()),
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
    """Add GO26's Table 1 mean, the differences from it and the implied zero fractions (also for *_qc).

    f_implied uses the gap-filled valid-day mean (mean_et_unclosed_mdv), f_implied_unfilled the mean of the
    valid records (mean_et_unclosed_valid).
    """
    res["go26_table1_mean_obs_et"] = go26_mean
    for sfx in ("", "_qc"):
        for k in DIFF_KEYS:
            if k + sfx in res:
                res[f"diff_{k}{sfx}"] = res[k + sfx] - go26_mean
        if "mean_et_unclosed_mdv" + sfx in res:
            res["f_implied" + sfx] = _f_implied(go26_mean, res["mean_et_unclosed_mdv" + sfx])
            res["f_implied_unfilled" + sfx] = _f_implied(go26_mean, res["mean_et_unclosed_valid" + sfx])
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
        eb = s["et_unclosed"].notna() & (s["eb_frac"] >= 0.5)
        rows.append({
            "site": site, "window": label, "days": len(s), "valid_LE_days": int(s["et_unclosed"].notna().sum()),
            "mean_et_unclosed_valid": s["et_unclosed"].mean(), "mean_et_unclosed_mdv": s["et_unclosed_mdv"].mean(),
            "max_et_unclosed_mdv": s["et_unclosed_mdv"].max(), "valid_EB_days": int(eb.sum()),
            "mean_et_unclosed_mdv_on_EB_days": s.loc[eb, "et_unclosed_mdv"].mean(),
            "mean_et_br_fdq": s["et_br_fdq"].mean(), "mean_et_br_fdq_mdv": s["et_br_fdq_mdv"].mean(),
            "mean_et_ebr_mdv": s["et_ebr_mdv"].mean(), "max_et_ebr_mdv": s["et_ebr_mdv"].max(),
            "mean_et_br_screened_mdv": s["et_br_screened_mdv"].mean(),
            "mean_et_br_unscreened": s["et_br_unscreened"].mean(), "max_et_br_unscreened": s["et_br_unscreened"].max(),
            "min_et_br_unscreened": s["et_br_unscreened"].min(),
            "n_br_factor_negative": int((s["br_factor"] < 0).sum()),
            "n_br_factor_gt2": int((s["br_factor"] > cl.BR_BOUNDS[1]).sum()),
            "n_br_screen_fail": int(s["flags"].str.contains("BR_screen_fail").sum()),
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


def figure_check(site: str, figure: str, table: pd.DataFrame, digitized: Path) -> dict:
    """GO26's plotted daily observed ET (scripts/04 digitization: obs, obs_is_zero) against the BASE zero-filled
    daily sums of ``table`` (one site's daily table), on the plotted days of the study period.

    Differences are GO26 minus BASE (mm/d); le_frac is the share of the day's records with LE in the BASE file.
    """
    g = pd.read_csv(digitized, parse_dates=["date"]).set_index("date")
    x = g[["obs", "obs_is_zero"]].join(table.set_index("date")[["le_frac", "et_hh_sum_zero_filled"]], how="inner")
    x = x[x["obs"].notna()]
    diff = x["obs"] - x["et_hh_sum_zero_filled"]
    ad = diff.abs()
    zero = x["obs_is_zero"].astype(str).str.lower().isin({"true", "1", "1.0"})
    no_le, part = x["le_frac"] == 0, (x["le_frac"] > 0) & (x["le_frac"] < 0.5)
    has_le = ~no_le
    return {
        "site": site, "figure": figure, "first_day": f"{x.index.min():%Y-%m-%d}" if len(x) else "",
        "last_day": f"{x.index.max():%Y-%m-%d}" if len(x) else "", "days_compared": len(x),
        "mean_diff_go26_minus_base": diff.mean(), "mean_abs_diff": ad.mean(), "max_abs_diff": ad.max(),
        "max_abs_diff_date": f"{ad.idxmax():%Y-%m-%d}" if len(x) else "",
        "r": x["obs"].corr(x["et_hh_sum_zero_filled"]) if len(x) > 2 else np.nan,
        "n_within_0p02": int((ad <= 0.02).sum()), "n_within_0p1": int((ad <= 0.1).sum()),
        "n_diff_ge_0p5": int((ad >= 0.5).sum()),
        "days_with_le": int(has_le.sum()), "mean_abs_diff_days_with_le": ad[has_le].mean(),
        "max_abs_diff_days_with_le": ad[has_le].max(), "n_within_0p02_days_with_le": int((ad[has_le] <= 0.02).sum()),
        "n_plotted_lt50pct_records": int(part.sum()), "mean_abs_diff_lt50pct_records": ad[part].mean(),
        "n_plotted_no_le": int(no_le.sum()), "n_nonzero_no_le": int((~zero & no_le).sum()),
        "nonzero_no_le_dates": date_ranges(x.index[~zero & no_le]),
        "n_exact_zero": int(zero.sum()), "n_exact_zero_no_le": int((zero & no_le).sum()),
        "exact_zero_days_all_no_le": bool(zero.any() and (zero & no_le).sum() == zero.sum()),
    }


def _month_holding(count: float, year: int, per_day: int) -> int:
    """Earliest month of ``year`` by whose end ``count`` records fit (every record valid from 1 January)."""
    days = np.cumsum([pd.Period(f"{year}-{m:02d}", "M").days_in_month for m in range(1, 13)])
    return int(np.searchsorted(days * per_day, count - 1e-9) + 1) if count > 0 else 0


def version_check(site: str, le: pd.Series, start: str, end: str, cited_version: str, cited_year,
                  summary: pd.Series, summary_date: str | None) -> list[dict]:
    """The cited BASE file's LE against the availability summary (amf_public.le_availability: share of each
    calendar year's records with LE), per year of the study period plus a 'period' row with the site's flags.

    ``le`` is the sub-daily LE of the whole cited file.  In-period LE records of the summary's version are
    an upper bound (the year's records, capped at the year's records inside the period), exact for a year
    wholly inside the period.  The summary's last LE month is estimated in its last year with LE: the month
    by which the cited file accumulates the summary's records of that year (within one day of records),
    i.e. assuming the earlier version is the cited file cut short; where the cited file has fewer records
    that year, the earliest month that can hold them (summary_le_end_month_min, a lower bound).
    """
    step = af.time_step(le.index)
    per_day = int(pd.Timedelta("1D") / step)
    t0, t1 = pd.Timestamp(f"{start}-01"), pd.Period(end, "M").end_time.normalize() + pd.Timedelta("1D")
    ok = le.notna()
    years_n = {y: (pd.Timestamp(y + 1, 1, 1) - pd.Timestamp(y, 1, 1)).days * per_day for y in range(1900, 2101)}
    summary = summary.astype(float)
    rows = []
    for y in range(t0.year, (t1 - pd.Timedelta("1D")).year + 1):
        a, b = max(t0, pd.Timestamp(y, 1, 1)), min(t1, pd.Timestamp(y + 1, 1, 1))
        n_year, n_period = years_n[y], (b - a).days * per_day
        c_year = int(ok[le.index.year == y].sum())
        c_period = int(ok[(le.index >= a) & (le.index < b)].sum())
        share = float(summary.get(y, np.nan))
        rows.append({"site": site, "year": str(y), "period_start": start, "period_end": end,
                     "base_version_cited": cited_version, "records_in_year": n_year, "records_in_period": n_period,
                     "share_le_cited": c_year / n_year, "share_le_summary": share,
                     "share_diff_cited_minus_summary": c_year / n_year - share,
                     "share_le_cited_in_period": c_period / n_period, "le_records_in_period_cited": c_period,
                     "le_records_in_period_summary_max": (min(int(round(share * n_year)), n_period)
                                                          if np.isfinite(share) else pd.NA),
                     "summary_records_exact": n_period == n_year})
    t = pd.DataFrame(rows)
    years = sorted(set(summary.index) | set(le.index.year[ok]))
    diffs = [abs(int(ok[le.index.year == y].sum()) / years_n[y] - float(summary.get(y, 0.0) or 0.0))
             for y in years if y in summary.index and np.isfinite(summary.get(y, np.nan))]
    with_le = summary[summary > 0]
    site_row = {"site": site, "year": "period", "period_start": start, "period_end": end,
                "base_version_cited": cited_version, "cited_version_year": cited_year,
                "summary_create_date": summary_date,
                "records_in_period": int(t["records_in_period"].sum()),
                "le_records_in_period_cited": int(t["le_records_in_period_cited"].sum()),
                "le_records_in_period_summary_max": (int(t["le_records_in_period_summary_max"].sum())
                                                     if t["le_records_in_period_summary_max"].notna().all()
                                                     else pd.NA),
                "summary_records_exact": bool(t["summary_records_exact"].all()),
                "max_abs_share_diff_all_years": max(diffs) if diffs else np.nan}
    site_row["share_le_cited_in_period"] = site_row["le_records_in_period_cited"] / site_row["records_in_period"]
    summary_year = int(summary_date[:4]) if summary_date else None
    newer = (bool(cited_year > summary_year) if cited_year and summary_year and cited_year != summary_year
             else (False if site_row["max_abs_share_diff_all_years"] < 5e-4 else pd.NA))
    site_row["cited_version_newer_than_summary"] = newer
    site_row["summary_describes"] = (f"an earlier version than {cited_version}" if newer is True else
                                     f"the LE data of {cited_version} (same share in every year)"
                                     if site_row["max_abs_share_diff_all_years"] < 5e-4 else "")
    cited_last = le.index[ok].max()
    site_row["cited_last_le_month"] = f"{cited_last:%Y-%m}"
    site_row["cited_le_months_after_period_end"] = ((cited_last.to_period("M") - pd.Period(end, "M")).n
                                                    if cited_last.to_period("M") >= pd.Period(end, "M") else 0)
    if len(with_le):
        y = int(with_le.index.max())
        count = float(with_le[y]) * years_n[y]
        in_y = ok[le.index.year == y]
        cum = in_y.groupby(in_y.index.month).sum().reindex(range(1, 13), fill_value=0).cumsum()
        reached = cum.index[cum >= count - per_day]
        m_min = _month_holding(count, y, per_day)
        m_prefix = int(reached[0]) if len(reached) else 0
        site_row.update(summary_last_le_year=y, summary_le_share_last_year=float(with_le[y]),
                        summary_le_end_month_min=f"{y}-{m_min:02d}",
                        summary_le_end_month_prefix=f"{y}-{m_prefix:02d}" if m_prefix else "",
                        summary_le_end_month=f"{y}-{max(m_prefix, m_min):02d}")
        site_row["summary_le_after_last_year"] = bool((summary[summary.index > y] > 0).any())
        site_row["period_ends_where_summary_le_ends"] = (site_row["summary_le_end_month"] == end
                                                         and not site_row["summary_le_after_last_year"])
    return rows + [site_row]


def monthly_observed(daily: pd.DataFrame, min_valid_days: int = 20) -> pd.DataFrame:
    """Monthly observed ET (mm/month) per site: zero-filled sums of complete months (obs_zero_filled) and, for each
    valid-day variant (VALID_VARIANTS), its mean over the days on which every variant exists x days in month
    (obs_<variant>; months with >= ``min_valid_days`` such days, n_valid)."""
    d = daily.assign(month=daily["date"].dt.to_period("M").dt.to_timestamp())
    cols = [c for _, c in VALID_VARIANTS]
    common = d[cols].notna().all(axis=1)
    d = d.assign(common=common, **{f"_{v}": d[c].where(common) for v, c in VALID_VARIANTS})
    m = d.groupby(["site", "month"]).agg(
        days=("date", "size"), n_valid=("common", "sum"), obs_zero_filled=("et_hh_sum_zero_filled", "sum"),
        **{f"mean_{v}": (f"_{v}", "mean") for v, _ in VALID_VARIANTS}).reset_index()
    dim = m["month"].dt.days_in_month
    for v, _ in VALID_VARIANTS:
        m[f"obs_{v}"] = (m.pop(f"mean_{v}") * dim).where(m["n_valid"] >= min_valid_days)
    m["obs_zero_filled"] = m["obs_zero_filled"].where(m["days"] == dim)
    m["n_valid"] = m["n_valid"].astype(int)
    m["season"] = m["month"].dt.month.map(SEASON)
    return m


def openet_bias(monthly: pd.DataFrame, openet: pd.DataFrame, zones: dict, reported: dict | None = None,
                min_months: int = oc.MIN_STRATUM_MONTHS) -> pd.DataFrame:
    """OpenET ensemble minus observed monthly ET, by GO26 climate zone and season (mm/d).

    month_set 'all_months': every month of the study period with OpenET data, against the zero-filled sums
    (variant zero_filled_table1, GO26's observed ET as reproduced; go26_reported_bias_mm_d only here);
    'common_valid_months': the months in which every valid-day variant has a value (:func:`monthly_observed`),
    on which the zero-filled sums and the valid-day variants (VALID_VARIANTS: GO26's processing as they state it,
    unclosed and Bowen-ratio closed; then the gap-filled mean, unclosed and closed with that Bowen ratio, EBR or the
    screened Bowen ratio) are compared on identical site-months.  OpenET values are not CC BY: openet_mm_d and
    bias_mm_d are blanked where they could be solved for (:func:`withhold_openet` with ``min_months``; 0 keeps
    every value, not for publication).
    """
    o = openet.rename(columns={"date": "month", "et_ensemble": "openet"})[["site", "month", "openet"]]
    o = o.assign(month=pd.to_datetime(o["month"]))
    pairs = monthly.merge(o, on=["site", "month"], how="inner").dropna(subset=["openet"])
    pairs["koppen_go26"] = pairs["site"].map(zones)
    valid = [(v, f"obs_{v}") for v, _ in VALID_VARIANTS]
    sets = {"all_months": [("zero_filled_table1", "obs_zero_filled")],
            "common_valid_months": [("zero_filled_table1", "obs_zero_filled"), *valid]}
    in_common = pairs[[c for _, c in valid]].notna().all(axis=1)
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
                in_set = in_common if month_set == "common_valid_months" else True
                for variant, col in variants:
                    s = pairs[sel & in_set & pairs[col].notna()]
                    if s.empty:
                        continue
                    dim = s["month"].dt.days_in_month
                    p, ob = s["openet"] / dim, s[col] / dim
                    rows.append({"koppen_go26": zone, "season": season, "month_set": month_set, "variant": variant,
                                 "n_sites": int(s["site"].nunique()), "n_months": len(s),
                                 "openet_mm_d": p.mean(), "observed_mm_d": ob.mean(), "bias_mm_d": (p - ob).mean(),
                                 "go26_reported_bias_mm_d": (reported.get((zone, season), np.nan)
                                                             if month_set == "all_months" else np.nan)})
                    members.append(frozenset(s.index))
    out = pd.DataFrame(rows)
    return withhold_openet(out, members, min_months) if min_months and len(out) else out


def openet_site_means(daily: pd.DataFrame, openet: pd.DataFrame, periods: pd.DataFrame,
                      openet_daily: pd.DataFrame | None = None,
                      min_months: int = oc.MIN_STRATUM_MONTHS) -> pd.DataFrame:
    """Observed ET and the OpenET ensemble averaged over the same days at each tower (mm/d; Figure 2 of the comment).

    day_set 'calendar': every day of the study period, against the zero-filled daily sums (GO26's Table 1 as
    reproduced); OpenET is the period's monthly total over its days, months without OpenET filled with the calendar
    month's 2016-2025 mean (oe_mm_d of scripts/12).  day_set 'common': the days on which every valid-day variant
    exists (VALID_VARIANTS, as month_set common_valid_months of the bias table) in months with OpenET, against
    each variant's mean over these days; OpenET is the monthly total / days in month, weighted by these days in
    each month.  The valid-day variants share one day set: means over each variant's own days differ in a few
    months only, and their differences would reveal those months' OpenET.  OpenET is not CC BY: openet_mm_d of a
    'common' row is blanked (openet_withheld) where its days miss part of 1 to ``min_months`` - 1 months of the
    period, which the calendar mean would reveal; scripts/14 checks the table jointly with the other outputs.
    With ``openet_daily`` (daily ensemble), openet_daily_mm_d is its mean over the same days, a sensitivity check
    that is not for publication (single values could be solved for).
    """
    o = openet.assign(date=pd.to_datetime(openet["date"]))
    od = (openet_daily.assign(date=pd.to_datetime(openet_daily["date"])).set_index(["site", "date"])["et_ensemble"]
          if openet_daily is not None else None)
    rows = []
    for p in periods.itertuples(index=False):
        d = daily[daily["site"].eq(p.site)]
        m = o[o["site"].eq(p.site)].set_index("date")["et_ensemble"]
        if d.empty or m.isna().all():
            continue
        filled, _ = oc.fill_monthly(m, p.period_start, p.period_end)
        per_day = filled / filled.index.days_in_month  # OpenET of each month, mm/d
        month = d["date"].dt.to_period("M").dt.to_timestamp()
        common = d[[c for _, c in VALID_VARIANTS]].notna().all(axis=1) & month.isin(m.index[m.notna()])
        for day_set, sel in (("calendar", pd.Series(True, index=d.index)), ("common", common)):
            if not sel.any():
                continue
            s, mo = d[sel], month[sel]
            n_m = mo.value_counts().reindex(filled.index, fill_value=0)
            row = {"site": p.site, "koppen_go26": p.koppen_go26, "day_set": day_set, "n_days": len(s),
                   "n_months": int((n_m > 0).sum()),
                   "n_months_partial": int((n_m < filled.index.days_in_month).sum()),
                   "openet_mm_d": float(per_day.reindex(mo).mean()),
                   "obs_zero_filled_mm_d": s["et_hh_sum_zero_filled"].mean(),
                   **{f"obs_{v}_mm_d": s[c].mean() if day_set == "common" else np.nan for v, c in VALID_VARIANTS}}
            if od is not None:
                row["openet_daily_mm_d"] = od.reindex(list(zip(s["site"], s["date"]))).mean()
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["openet_withheld"] = out["day_set"].eq("common") & out["n_months_partial"].between(1, min_months - 1)
    out.loc[out["openet_withheld"], ["openet_mm_d", "openet_daily_mm_d"] if od is not None else "openet_mm_d"] = np.nan
    return out


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


def _small_support(sets: list[frozenset], order: np.ndarray, shown: list[int], min_months: int) -> int | None:
    """Set to withhold (lowest ``order``) from a combination of the ``shown`` sets that reveals a combination
    (a sum, a difference, ...) of fewer than ``min_months`` site-months' OpenET values, or None.

    Each shown set is one equation (the sum of its site-months' values), so whatever the sets reveal is constant
    on their atoms (the site-months that lie in the same shown sets).  A combination supported on the atoms U is
    revealed exactly when leaving out U's columns lowers the rank; every U with fewer than ``min_months``
    site-months is tried, so the test is exact (single site-months are the case of one atom of one site-month).
    """
    if not shown:
        return None
    pattern: dict = {}
    for r, i in enumerate(shown):
        for m in sets[i]:
            pattern.setdefault(m, []).append(r)
    atoms: dict = {}
    for m, p in pattern.items():
        atoms.setdefault(tuple(p), []).append(m)
    a = np.zeros((len(shown), len(atoms)))
    for j, p in enumerate(atoms):
        a[list(p), j] = 1.0
    n = np.array([len(v) for v in atoms.values()])
    rank = np.linalg.matrix_rank(a)
    small = [int(j) for j in np.flatnonzero(n < min_months)]
    for k in range(1, min_months):
        for u in itertools.combinations(small, k):
            if n[list(u)].sum() >= min_months:
                continue
            rest = [j for j in range(a.shape[1]) if j not in u]
            if rest and np.linalg.matrix_rank(a[:, rest]) == rank:
                continue
            if rest:  # combinations of the sets that vanish outside u
                _, sv, vt = np.linalg.svd(a[:, rest].T)
                null = vt[int((sv > sv.max() * max(a.shape) * np.finfo(float).eps).sum()):]
            else:
                null = np.eye(len(shown))
            y = next(c @ a for c in null if np.abs(c @ a[:, list(u)]).max() > 1e-9)
            c = np.linalg.lstsq(a.T, y, rcond=None)[0]  # least-norm combination of the sets giving it
            return min((shown[r] for r in np.flatnonzero(np.abs(c) > 1e-9 * np.abs(c).max())),
                       key=lambda i: order[i])
    return None


def withhold_openet(bias: pd.DataFrame, members: list, min_months: int = oc.MIN_STRATUM_MONTHS) -> pd.DataFrame:
    """Blank openet_mm_d and bias_mm_d of :func:`openet_bias` rows so that neither a single month's OpenET value nor
    any sum or difference over fewer than ``min_months`` site-months can be solved for from the table and the
    public observed ET.

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
      a combination that gives it;
    * support: while the published rows still reveal any combination (a sum or difference) of fewer than
      ``min_months`` site-months (exact test on the atoms of the published sets, :func:`_small_support`;
      e.g. a withheld row rebuilt from margins and subtracted from a published one), a row of a
      combination that gives it: rows of one zone and season before zone or season totals, then the
      smaller first.

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
    total = np.zeros(len(sets), bool)  # zone x season rows go before zone or season totals, then the smaller
    for i, z, s in zip(sid, out["koppen_go26"], out["season"]):
        total[i] |= z == "all" or s == "all"
    order = size + (size.max() + 1) * total
    while (i := _small_support(sets, order, [j for j in range(len(sets)) if j not in rule], min_months)) is not None:
        rule[i] = "support"
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


def load_availability_summary(cache: Path) -> tuple[pd.DataFrame | None, str | None, str]:
    """The pinned amerifluxr availability summary cached by scripts/11 (offline), its create_date and a note."""
    path = cache / amfp.availability_file(amfp.PINNED_SHA)
    if not path.exists():
        return None, None, f"{path} not found (run scripts/11): version check not written"
    if amfp.md5_bytes(path.read_bytes()) != amfp.PINNED_MD5:
        return None, None, f"{path} does not have the pinned md5 {amfp.PINNED_MD5}: version check not written"
    table, create_date = amfp.read_variable_availability(path)
    return table, create_date, f"availability summary {path.name} (create_date {create_date})"


def write_csv(df: pd.DataFrame, path: Path, header: list[str]) -> None:
    with open(path, "w", newline="") as fh:
        for line in header:
            fh.write(f"# {line}\n")
        df.to_csv(fh, index=False, float_format="%.4f")


def _true(s: pd.Series) -> pd.Series:
    """A True/False/NA column as booleans (NA: False)."""
    return s.map(lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else False).astype(bool)


def _within(d: pd.Series) -> str:
    return ", ".join(f"{(d <= t + 1e-9).sum()} within {t}" for t in (0.005, 0.01, 0.02, 0.03))


def print_summary(out: pd.DataFrame, ok: pd.Series, qc_flags: bool) -> None:
    show = ["site", "valid_LE_days", "le_record_availability", "mean_et_unclosed_valid", "mean_et_unclosed_mdv",
            "mean_et_br_fdq", "mean_et_br_fdq_mdv", "mean_et_ebr_mdv", "mean_et_br_screened_mdv", "mean_et_br_unscreened",
            "mean_et_hh_sum_zero_filled_unclosed", "go26_table1_mean_obs_et",
            "diff_mean_et_hh_sum_zero_filled_unclosed", "diff_mean_et_hh_sum_zero_filled_file_days", "f_implied",
            "f_implied_unfilled"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n" + out.loc[ok, show].round(3).to_string(index=False) + "\n")
    s = out[ok]
    cal = _within(s["diff_mean_et_hh_sum_zero_filled_unclosed"].abs())
    in_file = _within(s["diff_mean_et_hh_sum_zero_filled_file_days"].abs())
    print(f"Zero-filled sub-daily sums vs GO26 Table 1 ({ok.sum()} sites): calendar days {cal}; days up to the "
          f"file's last record {in_file} (mm/d)")
    eb = s["valid_EB_days"].sum()
    fail = s[["n_br_factor_negative", "n_br_factor_lt05", "n_br_factor_gt2", "n_br_screen_fail"]].sum()
    share = (s["n_br_screen_fail"] / s["valid_EB_days"]).where(s["valid_EB_days"] > 0)
    worst = s.loc[s["max_et_br_unscreened"].idxmax()] if s["max_et_br_unscreened"].notna().any() else None
    n_fail = int(fail["n_br_screen_fail"])
    winter = 100 * s["n_br_screen_fail_nov_mar"].sum() / max(n_fail, 1)
    print(f"Daily Bowen-ratio factor on {eb} energy-balance days at {int((s['valid_EB_days'] > 0).sum())} sites: "
          f"< 0 on {fail['n_br_factor_negative']}, 0-0.5 on {fail['n_br_factor_lt05']}, > 2 on "
          f"{fail['n_br_factor_gt2']}; fails the screen on {n_fail} ({100 * n_fail / max(eb, 1):.1f}%; by site "
          f"{100 * share.min():.1f}-{100 * share.max():.1f}%), {winter:.0f}% of them in November-March")
    if worst is not None:
        print(f"Largest unscreened Bowen-ratio closed ET: {worst['max_et_br_unscreened']:.1f} mm/d at "
              f"{worst['site']} on {worst['max_et_br_unscreened_date']}")
    lo, hi = cl.FDQ_LE_LIMITS
    print(f"Bowen ratio as GO26 state it (flux-data-qaqc {cl.fdq_version()} 'br'): "
          f"{int(s['n_br_fdq_left_out'].sum())} of {eb} "
          f"energy-balance days left out (closed LE <= {lo:g} or >= {hi:g} W/m2, or Bowen ratio undefined); "
          f"{int(s['n_br_fdq_mdv_left_out'].sum())} with the gap-filled LE")
    alt = s[s["netrad_column_alt"].fillna("").ne("")]
    for r in alt.itertuples(index=False):
        print(f"{r.site}: NETRAD {r.netrad_column} (EBR closed, gap-filled {r.mean_et_ebr_mdv:.3f} mm/d); with "
              f"{r.netrad_column_alt}: {r.mean_et_ebr_mdv_netrad_alt:.3f} mm/d")
    mismatch = [f"{r.site} {r.base_version} (GO26 {r.base_version_go26})" for r in s.itertuples(index=False)
                if pd.notna(r.version_matches_go26) and not bool(r.version_matches_go26)]
    if mismatch:
        print("BASE versions differing from GO26's: " + ", ".join(mismatch))
    print("If go26_table1_mean_obs_et tracks the zero-filled variants rather than the valid-day means, missing data "
          "were counted as zero ET.")
    if qc_flags:
        q = s["diff_mean_et_hh_sum_zero_filled_unclosed_qc"].abs()
        print(f"\nWith LE removed where LE_SSITC_TEST = {QC_BAD} (--qc-flags; columns *_qc): zero-filled sums vs "
              f"Table 1: {_within(q)} mm/d")
        show_qc = ["site", "le_flag_column", "le_records_flag2", "valid_LE_days_qc", "le_record_availability_qc",
                   "mean_et_unclosed_mdv_qc", "mean_et_ebr_mdv_qc", "mean_et_hh_sum_zero_filled_unclosed_qc",
                   "diff_mean_et_hh_sum_zero_filled_unclosed_qc", "f_implied_qc"]
        with pd.option_context("display.width", 250):
            print(s[show_qc].round(3).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-dir", default=str(ROOT / "data/raw/ameriflux_base"), help="folder with the BASE CSVs")
    ap.add_argument("--config-dir", default=str(ROOT / "config"), help="holds go26_sites.csv and go26_table3.csv")
    ap.add_argument("--out-dir", default=str(ROOT / "results"),
                    help="results folder (also read: digitized_US-UiC.csv and digitized_US-UTJ.csv of scripts/04)")
    ap.add_argument("--sites", nargs="+", help="subset of the GO26 sites (default: all 22)")
    ap.add_argument("--base-version", choices=["go26", "latest"], default="go26",
                    help="which BASE version to use where several are present")
    ap.add_argument("--openet", default=str(ROOT / "data/raw/openet_api/openet_monthly.csv"),
                    help="OpenET monthly CSV from scripts/02 for the bias table and site means ('' to skip)")
    ap.add_argument("--openet-daily", default=None,
                    help="OpenET daily CSV from scripts/02 for the site means' daily check (default: "
                         "openet_daily_go26.csv next to --openet; optional)")
    ap.add_argument("--amf-public", default=str(ROOT / "data/raw/ameriflux_public"),
                    help="cache of scripts/11 with the pinned availability summary ('' to skip the version check)")
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

    avail, avail_date, avail_note = (load_availability_summary(Path(args.amf_public)) if args.amf_public
                                     else (None, None, "no --amf-public: version check not written"))
    rows, tables, versions = [], [], []
    for r in sites.itertuples(index=False):
        extra: dict = {}
        res, table = reconstruct_site(base_dir, r.site, r.period_start, r.period_end, go26_versions[r.site],
                                      prefer_version=None if args.base_version == "go26" else "",
                                      qc_flags=args.qc_flags, extra=extra)
        res.update(koppen_go26=r.koppen_go26, period_start=r.period_start, period_end=r.period_end,
                   base_doi=r.base_doi)
        rows.append(compare_with_go26(res, r.mean_obs_et_mm_d))
        if table is not None:
            tables.append(table)
            if avail is not None:
                versions += version_check(r.site, extra["le"], r.period_start, r.period_end, res["base_version"],
                                          GO26_BASE_CITATION_YEAR.get(r.site), amfp.le_availability(avail, r.site),
                                          avail_date)
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
        print("No BASE files: daily, winter, version, figure and OpenET-bias outputs not written.")
        return
    print_summary(out, ok, args.qc_flags)

    daily = pd.concat(tables, ignore_index=True)
    cite = [f"{r.site} BASE {r.base_version}: https://doi.org/{r.base_doi}" for r in out[ok].itertuples(index=False)]
    qc_note = ([f"*_qc (--qc-flags): the same with LE removed where its quality flag LE_SSITC_TEST = {QC_BAD}."]
               if args.qc_flags else [])
    write_csv(daily, out_dir / "go26_reconstructed_daily.csv", [
        "Daily ET rebuilt from AmeriFlux BASE data by scripts/10_reconstruct_go26_observed_et.py. AmeriFlux BASE",
        f"data are CC BY 4.0 (AmeriFlux CC-BY-4.0 data policy, {af.DATA_POLICY_URL}); this derivative is",
        "CC BY 4.0 too. Cite each site's BASE data product:", *cite,
        "ET in mm/d (LE/2.45 MJ/kg). le_frac: share of the day's sub-daily records with LE (valid days: >= 0.5);",
        "eb_frac: share with LE, H, NETRAD and G all valid (energy-balance days: valid and eb_frac >= 0.5).",
        "et_unclosed: mean of the valid LE records on valid days; et_unclosed_mdv: the same with missing records",
        "filled by the mean diurnal variation (same time of day, centred 15-day window).",
        "br_factor: (NETRAD - G)/(LE + H) from those records on energy-balance days (unscreened); br_factor_screened:",
        "br_factor where 0.5-2 with LE + H > 0 and NETRAD - G > 0; every other valid day takes the median of such",
        "factors in a centred 15-day window if it holds at least 5 of them, else the median of all such factors of",
        "the study period. Closures by flux-data-qaqc (Volk et al., 2021), version "
        f"{cl.fdq_version()}, run on the daily means of LE, H, NETRAD and G over the records in which all four",
        "are valid, on energy-balance days. ebr_factor: its 'ebr' correction factor (ebc_cf: FLUXNET2015/ONEFlux",
        "energy balance ratio with flux-data-qaqc's limits) on every valid day it gives one for.",
        "et_br_fdq: Bowen-ratio closure as GO26 state it (Twine et al., 2000), flux-data-qaqc's 'br' correction:",
        "et_unclosed x br_factor, unscreened, on energy-balance days; blank where flux-data-qaqc's closed LE is",
        "<= -100 or >= 850 W/m2 or the Bowen ratio is undefined (flag BR_fdq_left_out); et_br_fdq_mdv: the same",
        "with et_unclosed_mdv (flag BR_fdq_mdv_left_out).",
        "et_ebr, et_br_screened: et_unclosed x factor; *_mdv: et_unclosed_mdv x factor (et_ebr_mdv: main closed ET);",
        "et_br_unscreened: et_unclosed x br_factor on every energy-balance day (diverges as LE + H -> 0; diagnostic).",
        "et_hh_sum_zero_filled: daily sum of sub-daily ET with missing records counted as zero (pandas",
        "resample('D').sum() default; reproduces GO26 Table 1). G: PI gap-filled where BASE has no measured G, 0",
        "where it has none (G_zero); NETRAD: one radiometer per site (go26_observed_et_reconstruction.csv).",
        "flags: no_LE, LE_lt_50pct, LE_partial, MDV_unfilled, before_first_LE, after_last_LE, after_file_end,",
        "EB_missing, G_PI_F, G_zero, BR_factor_negative, BR_factor_lt05, BR_factor_gt2, BR_screen_fail,",
        "BR_fdq_left_out, BR_fdq_mdv_left_out.", *qc_note])

    fig8 = go26_fig8_reference(out_dir / "digitized_US-UiC.csv")
    winter = pd.DataFrame([w for t in tables for w in winter_rows(t["site"].iloc[0], t, fig8)])
    if "go26_fig8_obs_days" in winter:
        winter["go26_fig8_obs_days"] = winter["go26_fig8_obs_days"].astype("Int64")
    write_csv(winter, out_dir / "go26_reconstructed_winter.csv", [
        "Winter (Dec-Feb) check of daily ET rebuilt from AmeriFlux BASE data (CC BY 4.0; site DOIs in",
        "go26_observed_et_reconstruction.csv), all records (no quality flags); go26_fig8_*: GO26 Fig. 8.",
        "Means in mm/d over valid days (definitions in go26_reconstructed_daily.csv), except",
        "mean_et_hh_sum_zero_filled (all calendar days, missing data counted as zero)."])
    uic = winter[winter["site"].eq("US-UiC")]
    if len(uic):
        print("\nUS-UiC winter (mm/d):")
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(uic.drop(columns="site").round(2).to_string(index=False))

    if versions:
        vc = pd.DataFrame(versions)
        for c in ("records_in_year", "records_in_period", "le_records_in_period_cited",
                  "le_records_in_period_summary_max", "cited_version_year", "cited_le_months_after_period_end",
                  "summary_last_le_year"):
            if c in vc:
                vc[c] = vc[c].astype("Int64")
        write_csv(vc, out_dir / "go26_base_version_check.csv", [
            "LE in the cited AmeriFlux BASE file (version GO26 cite; CC BY 4.0, site DOIs in",
            "go26_observed_et_reconstruction.csv) against AmeriFlux's BASE variable-availability summary of",
            f"{avail_date} (amerifluxr commit {amfp.PINNED_SHA[:7]}, cached by scripts/11): share of each calendar",
            "year's records with LE (at read_base's position). Rows per year of GO26's study period;",
            "le_records_in_period_summary_max: the summary's records capped at the year's records in the period",
            "(exact where summary_records_exact). Row 'period': the site. cited_version_year: year of GO26's",
            "citation; cited_version_newer_than_summary: that year is after the summary was made (False: same share",
            "in every year). summary_le_end_month: month in which the summary's version ends its LE, estimated in its",
            "last LE year: the month by which the cited file accumulates the summary's records (_prefix; assumes the",
            "earlier version is the cited file cut short), at least the earliest month that can hold them (_min);",
            "period_ends_where_summary_le_ends: it equals GO26's period end and the summary has no later LE."])
        site_rows = vc[vc["year"].eq("period")]
        flag = _true(site_rows["period_ends_where_summary_le_ends"])
        is_newer = _true(site_rows["cited_version_newer_than_summary"])
        print(f"\nBASE version check ({avail_note}): cited version newer than the summary at {int(is_newer.sum())} "
              f"sites ({', '.join(site_rows.loc[is_newer, 'site'])}); GO26's period ends in the month the "
              f"summary's LE ends at {int(flag.sum())} of {len(site_rows)} sites")
        cols = ["site", "base_version_cited", "cited_version_newer_than_summary", "summary_le_end_month_min",
                "summary_le_end_month_prefix", "period_end", "cited_last_le_month", "le_records_in_period_cited",
                "le_records_in_period_summary_max", "period_ends_where_summary_le_ends"]
        with pd.option_context("display.width", 250):
            print(site_rows[cols].to_string(index=False))
    else:
        print(f"\n{avail_note}")

    checks = [figure_check(site, label, daily[daily["site"].eq(site)], out_dir / name)
              for site, (label, name) in GO26_FIGURES.items()
              if (out_dir / name).exists() and daily["site"].eq(site).any()]
    if checks:
        fc = pd.DataFrame(checks)
        write_csv(fc, out_dir / "go26_reconstructed_figure_check.csv", [
            "GO26's plotted daily observed ET (Figs. 8 and S2, CC BY 4.0, digitized by scripts/04: obs, obs_is_zero)",
            "against the daily sums of AmeriFlux BASE LE with missing records counted as zero (et_hh_sum_zero_filled",
            "of go26_reconstructed_daily.csv; CC BY 4.0, site DOIs in go26_observed_et_reconstruction.csv) on the",
            "plotted days. Differences GO26 minus BASE in mm/d; *_days_with_le: days with any LE record;",
            "*_lt50pct_records: days with LE in fewer than half of the records (GO26's own rule drops them);",
            "nonzero_no_le_dates: days GO26 plot as non-zero where the BASE file has no LE; exact zero:",
            "|obs| < 0.015."])
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print("\nGO26 figures vs BASE zero-filled daily sums:")
            print(fc.drop(columns=["nonzero_no_le_dates"]).round(3).T.to_string())

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
        "Monthly pairs in each GO26 study period, all records (no quality flags); bias_mm_d = mean over",
        "site-months of (OpenET - observed)/days in month. month_set all_months: every site-month with OpenET,",
        "against the zero-filled monthly sums (zero_filled_table1, GO26's observed ET as reproduced);",
        "go26_reported_bias_mm_d: GO26 Tables 3-4, on these rows only. month_set common_valid_months: months with",
        f">= {args.min_valid_days} days on which every valid-day variant exists; zero_filled_table1, GO26's",
        "processing as they state it, unclosed (valid-day mean) and br_fdq (Bowen-ratio closed as flux-data-qaqc's",
        "'br' correction), then unclosed_mdv (gap-filled valid-day mean), br_fdq_mdv (gap-filled, Bowen-ratio closed",
        "as GO26 state it), ebr_mdv (EBR closed) and br_screened_mdv (screened Bowen-ratio closed), on identical days",
        "and site-months (definitions in go26_reconstructed_daily.csv).",
        "OpenET values withheld (openet_withheld, withheld_rule): openet_mm_d and bias_mm_d are blank in rows with",
        f"fewer than {n} site-months (n_months), in rows of one zone and season whose site-months differ by 1-{n - 1}",
        "from those of a kept row (difference), in parts of a published zone or season total that would leave",
        f"1-{n - 1} site-months to subtraction (margin), in rows that would still determine one site-month (rank),",
        f"and in rows that would still reveal a sum or difference over fewer than {n} site-months (support):",
        f"no single month's OpenET value and no combination of fewer than {n} site-months can be solved for from",
        "this table and the observed ET. n_sites, n_months and observed_mm_d are kept."])
    if len(bias):
        print(f"\nOpenET values withheld in {int(bias['openet_withheld'].sum())} of {len(bias)} rows (fewer than {n} "
              "site-months, or a month or a small combination solvable; column withheld_rule)")
        tab = bias[bias["season"].eq("all")].pivot_table(index="koppen_go26", columns=["month_set", "variant"],
                                                         values="bias_mm_d")
        print("OpenET bias (mm/d) by GO26 zone, all seasons:")
        with pd.option_context("display.width", 250):
            print(tab.round(2).to_string() if not tab.empty else "all withheld")

    daily_path = Path(args.openet_daily) if args.openet_daily else openet_path.parent / "openet_daily_go26.csv"
    openet_daily = pd.read_csv(daily_path, parse_dates=["date"]) if daily_path.exists() else None
    means = openet_site_means(daily, openet, sites, openet_daily)
    if means.empty:
        return
    write_csv(means.drop(columns="openet_daily_mm_d", errors="ignore"),
              out_dir / "go26_reconstructed_openet_sites.csv", [
        "OpenET ensemble (OpenET, https://openetdata.org; OpenET API Terms of Service) and observed ET rebuilt from",
        "AmeriFlux BASE data (CC BY 4.0; site DOIs in go26_observed_et_reconstruction.csv) averaged over the same",
        "days at each GO26 tower (mm/d), all records (no quality flags). day_set calendar: every day of the study",
        "period; obs_zero_filled_mm_d: the zero-filled daily sums (GO26's Table 1 as reproduced); openet_mm_d: the",
        "monthly totals over the period's days, months without OpenET filled with the calendar month's 2016-2025",
        "mean (oe_mm_d of openet_api_go26_sites.csv). day_set common: the days on which every valid-day variant",
        "exists (as month_set common_valid_months of go26_reconstructed_openet_bias.csv) in months with OpenET;",
        "obs_<variant>_mm_d: each variant's mean over these days (definitions in go26_reconstructed_daily.csv);",
        "openet_mm_d: the monthly total / days in month, weighted by these days in each month. n_months: months",
        "with at least one of the days; n_months_partial: months of the period that miss some of their days.",
        f"openet_withheld: openet_mm_d blank where 1-{n - 1} months miss some days (the calendar mean would reveal",
        "them); scripts/14 checks this table jointly with the other OpenET statistics."])
    print("\nOpenET minus observed ET over the same days, mean over towers (mm/d):")
    for day_set, cols in (("calendar", ["zero_filled"]),
                          ("common", ["zero_filled", "unclosed", "unclosed_mdv", "br_fdq_mdv", "ebr_mdv"])):
        s = means[means["day_set"].eq(day_set)]
        for v in cols:
            diff = (s["openet_mm_d"] - s[f"obs_{v}_mm_d"]).dropna()
            if len(diff):
                print(f"  {day_set:8s} {v:13s} {diff.mean():+.2f} (RMSD {np.sqrt((diff ** 2).mean()):.2f}; "
                      f"{len(diff)} towers)")
    if openet_daily is not None:  # the daily check stays with the raw extracts (OpenET Terms of Service)
        check = openet_path.parent / "derived" / "go26_openet_sites_daily_check.csv"
        check.parent.mkdir(parents=True, exist_ok=True)
        means.to_csv(check, index=False, float_format="%.4f")
        d = (means["openet_daily_mm_d"] - means["openet_mm_d"]).dropna()
        print(f"daily minus monthly OpenET over the same days: {d.min():+.3f} to {d.max():+.3f} mm/d, mean "
              f"{d.mean():+.3f} ({len(d)} tower-day sets; {check}, not for publication)")


if __name__ == "__main__":
    main()
