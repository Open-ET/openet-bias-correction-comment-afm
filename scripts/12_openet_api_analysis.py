"""Current OpenET (API collection v2.1) at GO26's 22 towers and the benchmark croplands.

Analysis-plan items 2, 3 and 6 without the AmeriFlux BASE files.  OpenET was
extracted for 200 m x 200 m boxes centred on each tower, as GO26 describe
(their Section 2.1), by scripts/02_download_openet_api.py.

(a) GO26 Table 1 means (T1) against OpenET over each study period.  If the
    observed ET counted missing half-hours and days as zero, T1/OpenET scales
    with the fraction of the period with LE in the AmeriFlux BASE files
    (scripts/11), and T1 follows the zero-fill prediction
    P_zf = sum_m OE_m a_y(m) / D (see etcomment.openet_compare).  OE comes from
    the API's monthly ensemble totals, which differ from the summed daily
    ensemble values GO26 used; the nine towers with daily extracts give a
    daily-based sensitivity check.
(b) GO26's plotted OpenET line (digitized Figs. 8 and S2) against the API
    daily ensemble, to check that the API returns the product GO26 used: r,
    mean difference, the least-squares scale through the origin (GO26 = b x
    API) and monthly mean differences (r alone says little on a seasonal
    series).
(c) Current OpenET against the benchmark's closed ET (Volk et al., 2023a) in
    calendar-complete months with <= 5 gap-filled days, 2016 onward, at the
    benchmark cropland stations (months without any OpenET value left out):
    by Koppen class and group, season, site and model (pooled), with GO26's zone (Table 3) and season (Table 4) biases alongside.
(d) Daily OpenET against measured closed and unclosed ET at the seven towers
    shared with GO26, 2016 onward.
(e) US-UiC winter: GO26's observed ET (Fig. 8, digitized) in Jan-Mar and
    Nov-Dec 2020 against OpenET, bias-corrected gridMET ETo and precipitation,
    and the benchmark's measured winter ET at stations of the same Koppen
    class and at the nearest stations.

OpenET data (OpenET, https://openetdata.org) are subject to the OpenET API
Terms of Service and are not CC BY.  results/ holds statistics over study
periods, seasons and strata, and monthly means of the API daily ensemble only
for GO26's two plotted series ((b): US-UiC 2020, US-UTJ 2024).  No single
OpenET value may be solvable from them, given everything public (benchmark ET,
digitized series, which months or days enter each statistic).  Before writing,
(b) and (d) check this for the API daily ensemble and (c) for the benchmark
station-months of every model (openet_compare.projection_diagonal); (c)
withholds further per-station rows until it holds
(openet_compare.withheld_stations) and does not publish station-weighted MBEs.
The station-month table of (c) and the US-UiC rows of (e) with OpenET, ETo and
precipitation (whose means over GO26's observation days, next to (b)'s means
over its plotted OpenET days, gave single daily values) are written next to
the raw extracts (data/raw/openet_api/derived/, not deposited).  Without the
availability files from scripts/11 the outputs that need them (P_zf and the
ratio statistics) are skipped with a message.  scripts/08_make_figures.py
plots (a) and (c) in Fig. 2b-d.

Outputs
  results/openet_api_go26_sites.csv, results/openet_api_go26_summary.json   (a; key numbers of b-e)
  results/openet_api_go26_figure_check.csv, ..._figure_check_monthly.csv    (b)
  results/openet_api_benchmark_summary.csv                                   (c)
  results/openet_api_shared_daily.csv                                        (d)
  results/openet_api_uic_winter.csv                                          (e; no OpenET values)
  data/raw/openet_api/derived/benchmark_station_months.csv                  (c; local, not deposited)
  data/raw/openet_api/derived/uic_winter_openet.csv                         (e; local, not deposited)

Usage:
  python scripts/12_openet_api_analysis.py [--api data/raw/openet_api] [--flux data/raw/flux_ET_dataset]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from etcomment import benchmark as bm
from etcomment import openet_compare as oc

ROOT = Path(__file__).resolve().parents[1]
SHARED = ["US-ARM", "US-Bi1", "US-Bi2", "US-Ne1", "US-Ne2", "US-Ne3", "US-xSL"]
START = "2016-01-01"  # OpenET collection v2.1 in the API starts in 2016
ATTRIBUTION = ("OpenET values (and statistics derived from them) come from OpenET, https://openetdata.org "
               "(OpenET API, collection v2.1, 200 m x 200 m boxes), subject to the OpenET API Terms of "
               "Service; they are not CC BY. The results files hold statistics over study periods, seasons and "
               "strata, and monthly means of the API daily ensemble only for GO26's plotted series at US-UiC (2020) "
               "and US-UTJ (2024); no single monthly or daily OpenET value can be solved for from them "
               "(openet_identifiability). Benchmark station-month values and the US-UiC winter comparison with "
               "OpenET, ETo and precipitation stay with the raw extracts (data/raw/openet_api/derived/, not "
               "deposited).")
STRATA = {"all": [], "koppen": ["koppen"], "koppen_group": ["koppen_group"], "season": ["season"],
          "season_x_group": ["season", "koppen_group"], "site": ["site"]}
# columns of the benchmark summary computed from OpenET values (blank where withheld)
OPENET_STATS = ["mean_openet_mm_month", "mbe_mm_month", "mbe_pct", "rmse_mm_month", "r2", "mbe_mm_d"]
STATION_MONTHS = "benchmark_station_months.csv"  # in --derived
UIC_WINTER = "uic_winter_openet.csv"  # in --derived: part (e) with the OpenET, ETo and precipitation columns
# part (e) columns from OpenET API values, kept out of results/ for the US-UiC rows
WINTER_OPENET = ["mean_openet_mm_d", "mean_eto_mm_d", "mean_pr_mm_d", "frac_obs_gt_eto", "frac_obs_gt_openet"]
JSON_DIGITS = 4  # floats derived from OpenET values carry the precision of the CSV files
JSON_UNROUNDED = ("provenance", "availability")  # summary sections not derived from OpenET values


def _jsonable(x, key: str = "", digits: int | None = None):
    """JSON-safe copy: NaN/NA -> null, numpy scalars -> Python, counts (keys n_*) -> int.

    With ``digits``, floats are rounded to that many decimals (p-values, keys
    ending in _p, to that many significant digits).
    """
    if isinstance(x, dict):
        return {str(k): _jsonable(v, str(k), digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v, "", digits) for v in x]
    if x is pd.NA or x is None:
        return None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        if not math.isfinite(v):
            return None
        if key.startswith("n_") and v.is_integer():
            return int(v)
        if digits is None:
            return v
        return float(f"{v:.{digits}g}") if key.endswith("_p") else round(v, digits) + 0.0
    return x


def write_json(path: Path, obj: dict) -> None:
    """Summary JSON; floats outside JSON_UNROUNDED sections rounded to JSON_DIGITS."""
    out = {str(k): _jsonable(v, str(k), None if k in JSON_UNROUNDED else JSON_DIGITS) for k, v in obj.items()}
    path.write_text(json.dumps(out, indent=1))


def assert_days_undetermined(rows: list[pd.Series], what: str) -> float:
    """Largest projection of a day's unit vector onto the published functionals; raises if a value is determined.

    ``rows``: one weight Series (index: dates) per published functional of the
    API daily values at one tower (see openet_compare.projection_diagonal).
    """
    if not rows:
        return 0.0
    proj = oc.projection_diagonal(pd.concat(rows, axis=1).fillna(0.0).T)
    det = proj.index[proj > 1 - oc.IDENT_TOL]
    if len(det):
        raise RuntimeError(f"{what}: the statistics to be written determine the API daily OpenET value on "
                           f"{', '.join(d.strftime('%Y-%m-%d') for d in det[:10])}; not writing them")
    return float(proj.max())


def period_year_rows(t: pd.DataFrame | None, site: str, api: pd.Series) -> list[pd.Series]:
    """Sums over each year of the GO26 period of the API daily values, where (a) gives a daily-based mean.

    A superset of what oe_daily_mm_d and p_zf_daily_mm_d give (both are
    weighted sums of these year sums), so checks that use it are conservative.
    """
    if t is None or "oe_daily_mm_d" not in t:
        return []
    r = t[(t["site"] == site) & t["oe_daily_mm_d"].notna()]
    if r.empty:
        return []
    d = oc.daily_period(api, str(r["period_start"].iloc[0]), str(r["period_end"].iloc[0])).index
    return [pd.Series(1.0, d[d.year == y]) for y in sorted(set(d.year))]


def rel(p: Path) -> str:
    """Path relative to the repository (file name only outside it), so no local paths are deposited."""
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name


def provenance(api: Path) -> dict:
    """Collection version, access dates and md5 of the API extracts and their boxes (no asset ids, no keys)."""
    out = {"openet_api_dir": rel(api)}
    man = api / "manifest.json"
    if man.exists():
        calls = [c for c in json.loads(man.read_text()).get("calls", []) if c.get("status") == 200
                 and "params" in c]
        if calls:
            out["collection_versions"] = sorted({c["params"].get("version") for c in calls})
            out["reference_et"] = sorted({c["params"].get("reference_et") for c in calls})
            out["accessed_utc"] = [min(c["utc"] for c in calls), max(c["utc"] for c in calls)]
    for f in ("openet_monthly.csv", "openet_daily.csv", "sites.csv", "sites_200m.geojson"):
        if (api / f).exists():
            out[f"md5_{f}"] = bm.md5(api / f)
    return out


def load_api(api: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = pd.read_csv(api / "openet_monthly.csv", parse_dates=["date"])
    daily = pd.read_csv(api / "openet_daily.csv", parse_dates=["date"])
    return monthly, daily


# ----------------------------------------------------------------------------- (a)

def period_month_rows(t: pd.DataFrame, monthly: pd.DataFrame, ay: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """Functionals of the API monthly values that the period means of (a) give, per series (et_<model>, eto, pr).

    Each mean is a weighted sum of the months with data (gaps filled with the
    calendar month's mean, openet_compare.fill_weights); p_zf_mm_d and
    p_zf_uniform_mm_d weight the ensemble months by their year's availability.
    Rows: ('go26_period', site, statistic); columns: (site, month).
    """
    out = {}
    for c in [f"et_{m}" for m in oc.MODELS] + ["eto", "pr"]:
        rows = {}
        for _, r in t.iterrows():
            m = monthly[monthly["site"] == r["site"]]
            if c not in m or m.empty:
                continue
            w = oc.fill_weights(m.loc[m[c].notna(), "date"], str(r["period_start"]), str(r["period_end"]))
            if w.isna().any().any():  # the mean is not defined, so not published
                continue
            days = oc.calendar_days(w.index)
            w.columns = pd.MultiIndex.from_product([[r["site"]], w.columns])
            rows[("go26_period", r["site"], "mean")] = w.sum() / days
            y = ay[ay["site"] == r["site"]].set_index("year") if ay is not None else pd.DataFrame()
            if c != "et_ensemble" or y.empty:
                continue
            for name, a in (("p_zf", oc.period_availability(y)), ("p_zf_uniform", y["le_avail"])):
                a = a.reindex(w.index.year).to_numpy(float)
                if np.isfinite(a).all():  # else the prediction is NaN, not published
                    rows[("go26_period", r["site"], name)] = w.mul(a, axis=0).sum() / days
        out[c] = pd.DataFrame(rows).T.fillna(0.0)
    return out


def part_a(sites: pd.DataFrame, monthly: pd.DataFrame, avail_csv: Path, avail_by_year_csv: Path,
           out: Path, prov: dict | None = None, daily: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """GO26 towers: OpenET period means, T1/OpenET, LE availability and the zero-fill prediction.

    With ``daily`` (API daily ensemble) the daily-based means are added where
    they cover the period (checked in (b) and (d)).  Before writing, checks that
    the period means of the monthly values determine no single month.
    """
    ap, ay, msgs = oc.load_availability(avail_csv, avail_by_year_csv)
    for m in msgs:
        print(f"skipping availability-dependent outputs (P_zf, ratio statistics): {m}")
    t = oc.go26_site_table(sites, monthly, ap, ay, daily=daily)
    s = oc.site_summary(t)
    ident = {}
    for c, f in period_month_rows(t, monthly, ay).items():
        proj = oc.projection_diagonal(f)
        det = proj.index[proj > 1 - oc.IDENT_TOL]
        if len(det):
            raise RuntimeError(f"(a) the period means would determine {c} at {list(det)[:5]}; not writing them")
        ident[c] = float(proj.max()) if len(proj) else 0.0
    t.to_csv(out / "openet_api_go26_sites.csv", index=False, float_format="%.4f")
    avail = {"available": ap is not None and ay is not None, "messages": msgs}
    for key, f in (("period_file", Path(avail_csv)), ("by_year_file", Path(avail_by_year_csv))):
        avail[key] = rel(f)
        if f.exists():
            avail[f"md5_{key}"] = bm.md5(f)
    if ay is not None and "days_in_period" in ay.columns:
        # P_zf must use the yearly availability that le_avail_period is built from
        d = {"period": [], "uniform": []}
        for _, r in t.iterrows():
            y = ay[ay["site"] == r["site"]].set_index("year")
            if len(y) and np.isfinite(r["le_avail_period"]):
                w = y["days_in_period"]
                d["period"].append(abs(float((oc.period_availability(y) * w).sum() / w.sum()) - r["le_avail_period"]))
                d["uniform"].append(abs(float((y["le_avail"] * w).sum() / w.sum()) - r["le_avail_period"]))
        if d["period"]:
            avail["max_abs_diff_le_avail_period_vs_yearly_used_for_p_zf"] = max(d["period"])
            avail["max_abs_diff_le_avail_period_vs_whole_year_fractions"] = max(d["uniform"])
    flagged = t[t["irrigated_dry_climate"]][["site", "koppen_go26", "period_start", "period_end", "oe_mm_d",
                                             "oe_annual_mm", "pr_annual_mm", "eto_annual_mm", "map_mm_go26",
                                             "t1_mm_d", "t1_annual_mm", "oe_annual_over_pr",
                                             "t1_annual_over_pr", "le_avail_period", "p_zf_mm_d"]]
    summary = {
        "attribution": ATTRIBUTION,
        "provenance": prov or {},
        "definitions": {
            "oe_mm_d": "OpenET ensemble total over the GO26 study period from the API's MONTHLY totals (months "
                       "without data filled with the site's 2016-2025 mean for the calendar month) divided by "
                       "calendar days. OpenET computes the monthly ensemble at a monthly time step, so it differs "
                       "from the sum of the daily ensemble values that GO26 used (see oe_daily_mm_d)",
            "oe_daily_mm_d": "mean of the API DAILY ensemble over the GO26 study period, at the towers whose daily "
                             "extracts cover it (sensitivity check)",
            "ratio_t1_oe": "GO26 Table 1 mean observed ET / oe_mm_d (ratio_t1_oe_daily: / oe_daily_mm_d)",
            "le_avail_period": "fraction of the period's half-hours with LE in the AmeriFlux BASE file "
                               "(scripts/11)",
            "p_zf_mm_d": "sum over months of OpenET ET x LE availability of that year within the period / "
                         "calendar days (expected Table 1 value if missing data count as zero and EC ET equals "
                         "OpenET); a partial year's records are taken to lie inside the period "
                         "(days_available / days_in_period, the convention of le_avail_period)",
            "p_zf_daily_mm_d": "as p_zf_mm_d from the daily ensemble: mean over the period's days of OpenET ET x "
                               "LE availability of the day's year",
            "p_zf_uniform_mm_d": "as p_zf_mm_d with the whole-year fraction le_avail applied to every month of "
                                 "the year (lower for periods that end inside a year)",
            "slope_through_origin": "least-squares b in T1 = b x predictor; rmse_fit is the RMSE of "
                                    "T1 - b x predictor, rmse_diff that of T1 - predictor",
            "daily_oe_where_available": "go26_towers statistics with oe_mm_d, ratio_t1_oe and p_zf_mm_d replaced "
                                        "by their daily-based values where those exist",
            "scale_go26_over_api": "go26_figure_check: least-squares b in (GO26's plotted OpenET) = b x (API daily "
                                   "ensemble); rmse_scaled_mm_d is the RMSE of GO26 - b x API",
            "openet_withheld": f"openet_api_benchmark_summary.csv: OpenET statistics left blank for strata of fewer "
                               f"than {oc.MIN_STRATUM_MONTHS} months, for stations with fewer than "
                               f"{oc.MIN_STRATUM_MONTHS} months, for the smallest other stations of their Koppen "
                               f"class until each class's withheld months are 0 or at least "
                               f"{oc.MIN_STRATUM_MONTHS}, and for any further station of a class in which the "
                               f"identifiability check still finds a determined station-month (none in this run "
                               f"unless listed in openet_identifiability). Station-weighted MBEs are not published: "
                               f"with the pooled MBEs they determined two station-months per model",
            "openet_identifiability": "check run by scripts/12 before writing (etcomment.openet_compare."
                                      "projection_diagonal). Taken as known: the benchmark ET, GO26's digitized "
                                      "series and which station-months or days enter each statistic. Every "
                                      "published statistic is then a linear functional of the unknown OpenET values "
                                      "(means, MBE and MBE % give sums; mbe_mm_d the sum of value/days; GO26-over-API "
                                      "scale with the RMSE the sum of GO26 x API) or, for RMSE, R2 and r, has a "
                                      "gradient at the true values. benchmark_summary (per model, over its "
                                      "station-months) and daily (per tower, over the API daily ensemble; the "
                                      "daily-based period means of openet_api_go26_sites.csv enter as sums over "
                                      "each year of the period, a superset): no unknown's unit vector lies in the "
                                      "span of these rows (projection < 1 - 1e-9; max_projection_* give the "
                                      "largest), so no single monthly or daily value can be solved for, nor reduced "
                                      "to finitely many candidates near the true values, from the results files. "
                                      "Rounding to 4 decimals adds uncertainty on top",
            "uic_winter": f"openet_api_uic_winter.csv: GO26's US-UiC observed ET (Fig. 8, digitized) by winter "
                          f"month of 2020 and the benchmark's measured winter ET. The comparison with the API's "
                          f"OpenET, ETo and precipitation on GO26's observation days is in "
                          f"data/raw/openet_api/derived/{UIC_WINTER} (not deposited)",
        },
        "availability": avail,
        "go26_towers": s,
        "irrigated_dry_climate_sites": flagged.to_dict("records"),
        "openet_identifiability": {"max_projection_go26_period_means": ident},
    }
    write_json(out / "openet_api_go26_summary.json", summary)
    print_a(t, s)
    print(f"  identifiability: the period means determine no single API monthly value (largest projection "
          f"{max(ident.values(), default=0.0):.3f})")
    return t, summary


def print_a(t: pd.DataFrame, s: dict) -> None:
    cols = ["site", "koppen_go26", "n_months_filled", "oe_mm_d", "oe_daily_mm_d", "t1_mm_d", "ratio_t1_oe",
            "le_avail_period", "p_zf_mm_d", "t1_over_avail_over_oe", "pr_annual_mm", "eto_annual_mm"]
    print("\n(a) GO26 Table 1 (T1) against OpenET v2.1 over each study period (mm/d; oe from the monthly "
          "ensemble, oe_daily from the daily ensemble)")
    print(t[[c for c in cols if c in t]].round(3).to_string(index=False))
    for k, v in s.items():
        ra, oe = v["ratio_vs_availability"], v["t1_vs_oe"]
        print(f"  {k} ({v['n_sites']} sites): T1/OE vs availability r = {ra['pearson_r']:.3f} "
              f"(Spearman {ra['spearman_r']:.2f}, p = {ra['pearson_p']:.1g}), slope {ra['slope']:.3f}; "
              f"T1 vs OE r = {oe['pearson_r']:.3f}, slope {oe['slope_through_origin']:.2f}, "
              f"RMSE {oe['rmse_fit']:.2f}")
        for name in ("p_zf", "p_zf_uniform"):
            zf = v[f"t1_vs_{name}"]
            print(f"      T1 vs {name}: r = {zf['pearson_r']:.3f}, slope {zf['slope_through_origin']:.3f}, "
                  f"fit RMSE {zf['rmse_fit']:.2f}; mean T1 - {name} {zf['mean_diff_t1_minus']:+.2f}, "
                  f"RMSE {zf['rmse_diff']:.2f}")
        if "daily_oe_where_available" in v:
            d = v["daily_oe_where_available"]
            print(f"      daily ensemble at the {d['n_sites_daily']} towers with daily extracts: T1/OE vs "
                  f"availability r = {d['ratio_vs_availability']['pearson_r']:.3f} "
                  f"(slope {d['ratio_vs_availability']['slope']:.3f}); T1 vs P_zf r = "
                  f"{d['t1_vs_p_zf']['pearson_r']:.3f} (slope {d['t1_vs_p_zf']['slope_through_origin']:.3f}); "
                  f"T1 vs OE r = {d['t1_vs_oe']['pearson_r']:.3f}")
    f = t[t["irrigated_dry_climate"]]
    print("  irrigated sites in dry climates (mm/yr): OpenET | gridMET P | GO26 T1 x 365.25 | GO26 MAP")
    for _, r in f.iterrows():
        print(f"    {r['site']}: {r['oe_annual_mm']:.0f} | {r['pr_annual_mm']:.0f} | {r['t1_annual_mm']:.0f} | "
              f"{r['map_mm_go26']:.0f}")


# ----------------------------------------------------------------------------- (b)

def part_b(daily: pd.DataFrame, digitized: Path, out: Path,
           t: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """GO26's plotted OpenET (digitized) against the API daily ensemble: agreement, scale, monthly means.

    Before writing, checks that these statistics, with the daily-based period
    means of (a) at the same tower (``t``), determine no API daily value.
    Returns the agreement table, the monthly means and the largest projection
    per tower.
    """
    rows, months, ident = [], [], {}
    for site, fig in (("US-UiC", "Fig. 8"), ("US-UTJ", "Fig. S2")):
        f = digitized / f"digitized_{site}.csv"
        if not f.exists():
            print(f"(b) {f} not found; skipping {site}")
            continue
        g = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        api = daily[daily["site"] == site].set_index("date")["et_ensemble"]
        funcs = period_year_rows(t, site, api)
        for label, keep in (("all plotted days", pd.Series(True, g.index)),
                            ("excluding gap-interpolated days", ~g["openet_interp"].astype(bool))):
            a = oc.series_agreement(g.loc[keep, "openet"], api)
            rows.append({"site": site, "go26_figure": fig, "days": label, "n": a["n"], "first": a["first"],
                         "last": a["last"], "mean_go26_openet_mm_d": a["mean_a"],
                         "mean_api_openet_mm_d": a["mean_b"], "mean_diff_go26_minus_api_mm_d": a["mean_diff"],
                         "rmse_mm_d": a["rmse"], "pearson_r": a["pearson_r"],
                         "scale_go26_over_api": a["scale"], "rmse_scaled_mm_d": a["rmse_scaled"]})
            j = pd.concat([g.loc[keep, "openet"].rename("a"), api.rename("b")], axis=1, join="inner").dropna()
            funcs += [pd.Series(1.0, j.index), j["a"], j["b"]]  # the gradients of mean, scale, RMSE and r
            if label == "all plotted days":  # the day sets of the monthly means
                funcs += [pd.Series(1.0, d.index) for _, d in j.groupby(j.index.to_period("M"))]
        months.append(oc.monthly_agreement(g["openet"], api).assign(site=site, go26_figure=fig))  # all plotted days
        ident[site] = assert_days_undetermined(funcs, f"(b) {site}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "openet_api_go26_figure_check.csv", index=False, float_format="%.4f")
    cols = ["site", "go26_figure", "month", "n", "mean_go26_openet_mm_d", "mean_api_openet_mm_d",
            "diff_go26_minus_api_mm_d"]
    mo = pd.concat(months, ignore_index=True) if months else pd.DataFrame(
        columns=["site", "go26_figure", "month", "n", "mean_a", "mean_b", "mean_diff"])
    mo = mo.rename(columns={"mean_a": "mean_go26_openet_mm_d", "mean_b": "mean_api_openet_mm_d",
                            "mean_diff": "diff_go26_minus_api_mm_d"})[cols]
    mo.to_csv(out / "openet_api_go26_figure_check_monthly.csv", index=False, float_format="%.4f")
    print("\n(b) GO26's plotted OpenET vs the API daily ensemble (scale: least-squares b in GO26 = b x API)")
    print(df.round(3).to_string(index=False))
    print("  monthly means over all plotted days (mm/d)")
    print(mo.round(3).to_string(index=False))
    if ident:
        print("  identifiability: no API daily value is determined (largest projection "
              + ", ".join(f"{k} {v:.3f}" for k, v in ident.items()) + ")")
    return df, mo, ident


# ----------------------------------------------------------------------------- benchmark inputs

def station_metadata(flux_root: Path) -> pd.DataFrame:
    from etcomment import maps as mp

    m = pd.read_excel(flux_root / "station_metadata.xlsx", header=1).rename(columns={
        "Site ID": "site", "General classification": "land_cover", "Latitude": "lat", "Longitude": "lon",
        "Measurement technique": "technique"})
    m = m[m["land_cover"] == "Croplands"].copy()
    m["site"] = m["site"].astype(str)
    m["koppen"] = mp.koppen_at(m["lat"], m["lon"])
    m["koppen_group"] = m["koppen"].str[0]
    return m.sort_values("site").reset_index(drop=True)


def load_frames(flux_root: Path, sites) -> dict[str, pd.DataFrame]:
    frames = {}
    for s in sorted(sites):
        if (flux_root / "daily_data_files" / f"{s}_daily_data.csv").exists():
            frames[s] = bm.standardize(bm.load_daily(flux_root, s))
    return frames


# ----------------------------------------------------------------------------- (c)

def go26_reference(model: str, stratum_type: str, stratum: str, zones: pd.DataFrame,
                   site_zone: dict[str, str]) -> tuple[float, str | None]:
    """GO26's OpenET bias (mm/d) for a stratum, where they report one (ensemble only)."""
    if model != "ensemble":
        return np.nan, None
    if stratum_type == "all":
        return float(zones.loc["Overall (all sites)", "openet_bias"]), "GO26 Table 3, all sites"
    if stratum_type == "koppen" and stratum in zones.index:
        return float(zones.loc[stratum, "openet_bias"]), zones.loc[stratum, "source"]
    if stratum_type == "season":
        return oc.GO26_SEASON_BIAS[stratum], "GO26 Table 4"
    if stratum_type == "site" and stratum in site_zone and site_zone[stratum] in zones.index:
        return float(zones.loc[site_zone[stratum], "openet_bias"]), f"GO26 Table 3, zone {site_zone[stratum]}"
    return np.nan, None


def check_benchmark(q: pd.DataFrame, st: pd.DataFrame, extra: pd.DataFrame | None = None) -> dict:
    """Identifiability of one model's station-months under the summary rows to be written.

    ``q``: pairs (site, month, pred, obs, n_days); ``st``: stratified summary
    with openet_withheld; ``extra``: other published functionals of the same
    monthly values, columns (site, month) (the period means of (a)).  Raises if
    a value is determined; returns the largest projections over the pairs.
    """
    pub = st[~st["openet_withheld"]]
    f = oc.stratum_functionals(q, STRATA, set(zip(pub["stratum_type"], pub["stratum"])))
    keys = pd.MultiIndex.from_frame(q.loc[f.columns, ["site", "month"]])
    f.columns = keys
    if extra is not None and len(extra):
        f = pd.concat([f, extra]).fillna(0.0)
    lin = oc.projection_diagonal(f[[k[-1] != "rmse" for k in f.index]])
    full = oc.projection_diagonal(f)
    det = full.index[full > 1 - oc.IDENT_TOL]
    if len(det):
        raise RuntimeError(f"(c) the summary rows would determine {', '.join(f'{s} {m:%Y-%m}' for s, m in det[:10])}"
                           f"; not writing them")
    return {"n_station_months": len(keys), "max_projection_linear": float(lin[keys].max()),
            "max_projection_with_rmse": float(full[keys].max())}


def part_c(frames: dict[str, pd.DataFrame], meta: pd.DataFrame, monthly: pd.DataFrame, go_sites: pd.DataFrame,
           zones: pd.DataFrame, out: Path, local: Path, period_rows: dict[str, pd.DataFrame] | None = None
           ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    """Monthly OpenET (ensemble and models) against closed benchmark ET, 2016 onward.

    Per model, per-station rows are withheld (openet_compare.withheld_stations
    with the identifiability check), and the rows to be written, together with
    the period means of (a) (``period_rows``, from :func:`period_month_rows`),
    are checked to determine no station-month before writing.  Returns the
    station-month pairs with an OpenET value, the stratified summary, the
    complete benchmark months without any OpenET value ('site YYYY-MM') and the
    check per model.  The pairs (OpenET values recoverable) go to ``local``,
    not to ``out``.
    """
    parts = []
    for _, m in meta.iterrows():
        s = m["site"]
        a = monthly[monthly["site"] == s]
        if s not in frames or a.empty:
            continue
        p = oc.monthly_pairs(frames[s], a, START)
        if p.empty:
            continue
        p.insert(0, "site", s)
        p["koppen"], p["koppen_group"] = m["koppen"], m["koppen_group"]
        p["season"] = oc.season(p["month"].dt.month)
        p["technique"] = m["technique"]
        parts.append(p)
    pairs = pd.concat(parts, ignore_index=True)
    none = pairs[[f"et_{m}" for m in oc.MODELS]].isna().all(axis=1)  # no OpenET value from any model
    no_openet = [f"{s} {d:%Y-%m}" for s, d in zip(pairs.loc[none, "site"], pairs.loc[none, "month"])]
    pairs = pairs[~none].reset_index(drop=True)
    rows, ident = [], {}
    for model in oc.MODELS:
        q = pairs.rename(columns={f"et_{model}": "pred", "et_corr": "obs"})
        st = oc.stratified_bias(q, STRATA, time="month")
        st.insert(0, "model", model)
        base, withheld = oc.withheld_stations(q), oc.withheld_stations(q, strata=STRATA)
        st["openet_withheld"] = (st["n"] < oc.MIN_STRATUM_MONTHS) | (
            (st["stratum_type"] == "site") & st["stratum"].isin(withheld))
        ident[model] = {**check_benchmark(q, st, (period_rows or {}).get(f"et_{model}")),
                        "stations_withheld_by_check": [s for s in withheld if s not in base]}
        if ident[model]["stations_withheld_by_check"]:
            print(f"(c) {model}: with the size rule alone the summary would determine station-months; also "
                  f"withheld: {', '.join(ident[model]['stations_withheld_by_check'])}")
        rows.append(st)
    summ = pd.concat(rows, ignore_index=True).rename(columns={
        "n": "n_months", "mean_obs": "mean_ec_mm_month", "mean_pred": "mean_openet_mm_month",
        "mbe": "mbe_mm_month", "mbe_per_day": "mbe_mm_d", "rmse": "rmse_mm_month",
        "first": "first_month", "last": "last_month"})
    small = summ[(summ["stratum_type"] != "site") & (summ["n_months"] < oc.MIN_STRATUM_MONTHS)]
    if len(small):  # withheld below, but without complementary withholding: check before depositing
        print(f"warning: strata with fewer than {oc.MIN_STRATUM_MONTHS} months: "
              f"{sorted(set(small['stratum_type'] + ' ' + small['stratum']))}")
    summ.loc[summ["openet_withheld"], OPENET_STATS] = np.nan
    for c in ("first_month", "last_month"):
        summ[c] = pd.to_datetime(summ[c]).dt.strftime("%Y-%m")
    site_zone = dict(zip(go_sites["site"], go_sites["koppen_go26"]))
    ref = [go26_reference(r.model, r.stratum_type, r.stratum, zones, site_zone) for r in summ.itertuples()]
    summ["go26_openet_bias_mm_d"] = [v for v, _ in ref]
    summ["go26_source"] = [src for _, src in ref]
    summ["go26_shared_site"] = (summ["stratum_type"] == "site") & summ["stratum"].isin(SHARED)
    new = ["first_month", "last_month", "openet_withheld"]
    summ = summ[[c for c in summ.columns if c not in new] + new]
    summ.to_csv(out / "openet_api_benchmark_summary.csv", index=False, float_format="%.4f")
    # station-month table: benchmark ET and OpenET minus benchmark ET, from which the OpenET monthly values
    # follow; kept with the raw extracts (not deposited)
    pm = pairs[["site", "month", "koppen", "koppen_group", "season", "technique", "n_days", "n_gap", "et_corr"]].copy()
    pm["month"] = pm["month"].dt.strftime("%Y-%m")
    pm["go26_shared_site"] = pm["site"].isin(SHARED)
    pm = pm.rename(columns={"et_corr": "ec_closed_et_mm"})
    for model in oc.MODELS:
        pm[f"{model}_minus_ec_mm"] = pairs[f"et_{model}"] - pairs["et_corr"]
    local.mkdir(parents=True, exist_ok=True)
    pm.to_csv(local / STATION_MONTHS, index=False, float_format="%.3f")
    old = out / "openet_api_benchmark_monthly.csv"
    if old.exists():  # written by earlier versions; it let the OpenET monthly values be rebuilt
        old.unlink()
        print(f"removed {rel(old)} (station-month values are now in {rel(local / STATION_MONTHS)})")
    print_c(summ, pairs, no_openet, ident)
    return pairs, summ, no_openet, ident


def print_c(summ: pd.DataFrame, pairs: pd.DataFrame, no_openet: list[str], ident: dict | None = None) -> None:
    e = summ[summ["model"] == "ensemble"]
    ens = pairs[pairs["et_ensemble"].notna()]
    print(f"\n(c) OpenET v2.1 vs benchmark closed ET, monthly, {START[:4]} onward: {len(ens)} station-months with "
          f"an OpenET ensemble value at {ens['site'].nunique()} cropland stations ({ens['month'].min():%Y-%m} to "
          f"{ens['month'].max():%Y-%m}); {len(no_openet)} complete benchmark months without any OpenET value "
          f"left out" + (f" ({', '.join(no_openet)})" if no_openet else ""))
    cols = ["model", "stratum_type", "stratum", "n_months", "n_stations", "mean_ec_mm_month", "mbe_mm_month",
            "mbe_pct", "mbe_mm_d", "rmse_mm_month", "r2", "go26_openet_bias_mm_d"]
    show = e[e["stratum_type"].isin(["all", "koppen_group", "koppen", "season"])
             | ((e["stratum_type"] == "site") & e["go26_shared_site"])]
    print(show[cols].round(3).to_string(index=False))
    m = summ[summ["stratum_type"] == "all"]
    print(m[cols].round(3).to_string(index=False))
    sg = e[e["stratum_type"] == "season_x_group"]
    print(sg[cols].round(3).to_string(index=False))
    w = summ[(summ["stratum_type"] == "site") & summ["openet_withheld"]].groupby("model", sort=False)["stratum"]
    print(f"  per-station OpenET statistics withheld in the summary file (see openet_compare.withheld_stations): "
          f"ensemble {', '.join(w.get_group('ensemble')) if 'ensemble' in w.groups else 'none'}; "
          f"stations per model: {({k: int(v) for k, v in w.size().items()})}")
    if ident:
        print("  identifiability: no station-month is determined by the summary rows and the period means of (a); "
              "largest projection, sums only / with RMSE: "
              + ", ".join(f"{k} {v['max_projection_linear']:.3f}/{v['max_projection_with_rmse']:.3f}"
                          for k, v in ident.items()))


# ----------------------------------------------------------------------------- (d)

def part_d(frames: dict[str, pd.DataFrame], daily: pd.DataFrame, go_sites: pd.DataFrame, zones: pd.DataFrame,
           shared_csv: Path, out: Path, t: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Daily OpenET against measured closed and unclosed ET at the seven shared towers.

    Before writing, checks that these statistics, with the daily-based period
    means of (a) at the same tower (``t``), determine no API daily value.
    Returns the table and the largest projection per tower.
    """
    parts = []
    for s in SHARED:
        od = daily[daily["site"] == s].set_index("date")["et_ensemble"]
        if s not in frames or od.empty:
            print(f"(d) no benchmark or API daily data for {s}")
            continue
        p = oc.daily_pairs(frames[s], od, START)
        p["site"] = s
        parts.append(p)
    pairs = pd.concat(parts)
    pairs["season"] = oc.season(pairs.index.month)
    zone = dict(zip(go_sites["site"], go_sites["koppen_go26"]))
    p2 = pd.read_csv(shared_csv).set_index("site") if shared_csv.exists() else None
    if p2 is None:
        print(f"(d) {shared_csv} not found (run scripts/06); Phase II context columns left empty")
    rows = []
    groups = [(s, g) for s, g in pairs.groupby("site", sort=False)] + [("All 7 towers", pairs)]
    funcs = {s: period_year_rows(t, s, daily[daily["site"] == s].set_index("date")["et_ensemble"])
             for s in pairs["site"].unique()}  # per tower; the all-tower rows are sums of these
    for s, g in groups:
        for season in ["all", *oc.SEASONS]:
            h = g if season == "all" else g[g["season"] == season]
            for ref, col in (("closed", "et_corr"), ("unclosed", "et")):
                b = oc.bias_stats(h["openet"], h[col])
                if s in funcs:  # the gradients of the mean and of the RMSE (and R2)
                    k = h[np.isfinite(h["openet"]) & np.isfinite(h[col])]
                    funcs[s] += [pd.Series(1.0, k.index), k["openet"] - k[col]]
                z = zone.get(s)
                r = {"site": s, "go26_zone": z, "season": season, "reference": ref, "n_days": b["n"],
                     "first": h.index.min().date().isoformat() if len(h) else None,
                     "last": h.index.max().date().isoformat() if len(h) else None,
                     "mean_ec_mm_d": b["mean_obs"], "mean_openet_mm_d": b["mean_pred"], "mbe_mm_d": b["mbe"],
                     "mbe_pct": b["mbe_pct"], "rmse_mm_d": b["rmse"], "r2": b["r2"],
                     "phase2_overpass_mbe_mm_d": np.nan, "phase2_n_overpass_days": np.nan,
                     "go26_zone_openet_bias_mm_d": np.nan,
                     "go26_season_openet_bias_mm_d": oc.GO26_SEASON_BIAS.get(season, np.nan)}
                if season == "all" and z in zones.index:
                    r["go26_zone_openet_bias_mm_d"] = float(zones.loc[z, "openet_bias"])
                if season == "all" and p2 is not None and s in p2.index:
                    r["phase2_overpass_mbe_mm_d"] = float(p2.loc[s, f"daily_mbe_{ref}"])
                    r["phase2_n_overpass_days"] = int(p2.loc[s, f"n_{ref}"])
                rows.append(r)
    ident = {s: assert_days_undetermined(f, f"(d) {s}") for s, f in funcs.items()}
    df = pd.DataFrame(rows)
    df["phase2_n_overpass_days"] = df["phase2_n_overpass_days"].astype("Int64")
    df.to_csv(out / "openet_api_shared_daily.csv", index=False, float_format="%.4f")
    print(f"\n(d) Daily OpenET v2.1 vs measured benchmark ET at the shared towers, {START[:4]} onward")
    cols = ["site", "season", "reference", "n_days", "mean_ec_mm_d", "mean_openet_mm_d", "mbe_mm_d", "rmse_mm_d",
            "r2", "phase2_overpass_mbe_mm_d", "go26_zone_openet_bias_mm_d", "go26_season_openet_bias_mm_d"]
    show = df[(df["season"] == "all") | (df["site"] == "All 7 towers")]
    print(show[cols].round(3).to_string(index=False))
    print(f"  identifiability: no API daily value is determined (largest projection "
          f"{max(ident.values(), default=0.0):.3f})")
    return df, ident


# ----------------------------------------------------------------------------- (e)

def part_e(frames: dict[str, pd.DataFrame], meta: pd.DataFrame, daily: pd.DataFrame, go_sites: pd.DataFrame,
           digitized: Path, out: Path, local: Path | None = None, n_nearest: int = 3) -> pd.DataFrame:
    """US-UiC winter 2020: GO26 observed ET vs OpenET, ETo and precipitation; benchmark winter ET.

    The full table goes to ``local`` (not deposited); results/ gets it without
    the OpenET, ETo and precipitation columns of the US-UiC rows (WINTER_OPENET):
    averaged over GO26's observation days, next to the monthly means of (b) over
    its plotted OpenET days, they gave single daily values (2020-11-02).
    Returns the results table.
    """
    from etcomment import maps as mp

    uic = go_sites.set_index("site").loc["US-UiC"]
    rows = []
    f = digitized / "digitized_US-UiC.csv"
    if f.exists():
        g = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        api = daily[daily["site"] == "US-UiC"].set_index("date")
        d = pd.DataFrame({"obs": g["obs"]}).join(api[["et_ensemble", "eto", "pr"]].rename(
            columns={"et_ensemble": "openet"}), how="left")
        w = oc.winter_check(d)
        w.insert(0, "source", "GO26 Fig. 8 observed ET (digitized), 2020, vs OpenET API")
        w["group"] = "US-UiC"
        w["eto_source"] = "OpenET API: bias-corrected gridMET ETo; precipitation gridMET"
        rows.append(w)
    else:
        print(f"(e) {f} not found; skipping GO26 rows")
    k_rubel = mp.koppen_at([uic["latitude"]], [uic["longitude"]])[0]
    meta = meta.assign(distance_km=oc.haversine_km(uic["latitude"], uic["longitude"], meta["lat"], meta["lon"]))
    winter = (1, 2, 3, 11, 12)
    has_winter = {s for s, d in frames.items() if (d["valid_measured"] & np.isin(d.index.month, winter)).any()}
    near = meta[meta["site"].isin(has_winter)].sort_values("distance_km").head(n_nearest)
    sets = [(f"benchmark croplands, Koppen {k_rubel} (class of US-UiC, Rubel et al. 2017 grid)", k_rubel,
             meta.loc[meta["koppen"] == k_rubel, "site"], np.nan)]
    if uic["koppen_go26"] != k_rubel:
        sets.append((f"benchmark croplands, Koppen {uic['koppen_go26']} (GO26's class for US-UiC)",
                     uic["koppen_go26"], meta.loc[meta["koppen"] == uic["koppen_go26"], "site"], np.nan))
    sets += [(f"benchmark station {r['site']}, nearest to US-UiC", r["koppen"], [r["site"]], r["distance_km"])
             for _, r in near.iterrows()]
    for label, k, sites, dist in sets:
        fr = {s: frames[s] for s in sites if s in frames}
        for months_label, months in oc.DEFAULT_WINTER_GROUPS.items():
            c = oc.measured_climatology(fr, months)
            if not c.get("n_days"):
                continue
            rows.append(pd.DataFrame([{
                "source": "Benchmark measured EBR-closed ET (Volk et al., 2023a), all years", "group": label,
                "koppen": k, "distance_km": dist, "months": months_label, "n_days": c["n_days"],
                "n_stations": c["n_stations"], "stations": c["stations"], "first_year": c["first_year"],
                "last_year": c["last_year"], "mean_obs_mm_d": c["mean_closed_mm_d"],
                "mean_unclosed_mm_d": c["mean_unclosed_mm_d"], "mean_eto_mm_d": c["mean_eto_mm_d"],
                "frac_obs_gt_eto": c["frac_closed_gt_eto"], "frac_obs_gt_1mm_d": c["frac_closed_gt_1mm_d"],
                "max_obs_mm_d": c["max_closed_mm_d"], "station_mean_min": c["station_mean_min"],
                "station_mean_median": c["station_mean_median"], "station_mean_max": c["station_mean_max"],
                "eto_source": "benchmark gridMET ETo (not bias-corrected)"}]))
    df = pd.concat(rows, ignore_index=True)
    lead = ["source", "group", "koppen", "distance_km", "months", "n_days", "n_stations", "mean_obs_mm_d",
            "mean_openet_mm_d", "mean_eto_mm_d", "mean_pr_mm_d", "frac_obs_gt_eto", "frac_obs_gt_openet",
            "frac_obs_gt_1mm_d", "max_obs_mm_d", "n_obs_near_zero", "mean_unclosed_mm_d", "station_mean_min",
            "station_mean_median", "station_mean_max", "first_year", "last_year", "stations", "eto_source"]
    df = df[[c for c in lead if c in df.columns]]
    for c in ("n_days", "n_stations", "n_obs_near_zero", "first_year", "last_year"):
        if c in df:
            df[c] = df[c].astype("Int64")
    if local is not None:
        local.mkdir(parents=True, exist_ok=True)
        df.to_csv(local / UIC_WINTER, index=False, float_format="%.4f")
    res = df.copy()
    is_uic = res["group"] == "US-UiC"
    res.loc[is_uic, [c for c in WINTER_OPENET + ["eto_source"] if c in res]] = np.nan
    res.loc[is_uic, "source"] = "GO26 Fig. 8 observed ET (digitized), 2020"
    res = res.drop(columns=[c for c in WINTER_OPENET if c in res and res[c].isna().all()])
    res.to_csv(out / "openet_api_uic_winter.csv", index=False, float_format="%.4f")
    print("\n(e) US-UiC winter: GO26 observed ET (Fig. 8, 2020) and benchmark measured ET (mm/d); the US-UiC "
          "OpenET, ETo and precipitation columns are not written to results/"
          + (f" (see {rel(local / UIC_WINTER)})" if local is not None else ""))
    cols = ["group", "months", "n_days", "n_stations", "mean_obs_mm_d", "mean_openet_mm_d", "mean_eto_mm_d",
            "mean_pr_mm_d", "frac_obs_gt_eto", "frac_obs_gt_openet", "frac_obs_gt_1mm_d", "max_obs_mm_d",
            "distance_km"]
    print(df[[c for c in cols if c in df.columns]].round(3).to_string(index=False))
    return res


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--api", default=str(ROOT / "data/raw/openet_api"), help="OpenET API extracts (scripts/02)")
    ap.add_argument("--flux", default=str(ROOT / "data/raw/flux_ET_dataset"), help="benchmark flux data")
    ap.add_argument("--avail", default=str(ROOT / "results/ameriflux_le_availability.csv"))
    ap.add_argument("--avail-by-year", default=str(ROOT / "results/ameriflux_le_availability_by_year.csv"))
    ap.add_argument("--digitized", default=str(ROOT / "results"), help="folder with digitized_US-*.csv")
    ap.add_argument("--shared", default=str(ROOT / "results/shared_site_comparison.csv"),
                    help="Phase II overpass-day biases at the shared towers (scripts/06)")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--derived", default=str(ROOT / "data/raw/openet_api/derived"),
                    help="folder for the tables of (c) and (e) that hold OpenET values (not deposited)")
    args = ap.parse_args()
    api, out = Path(args.api), Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    monthly, daily = load_api(api)
    go_sites = pd.read_csv(ROOT / "config" / "go26_sites.csv")
    zones = pd.read_csv(ROOT / "config" / "go26_table3.csv").set_index("zone")

    t, summary = part_a(go_sites, monthly, Path(args.avail), Path(args.avail_by_year), out, provenance(api), daily)
    ident = summary["openet_identifiability"]
    fc, fcm, ident["max_projection_daily"] = part_b(daily, Path(args.digitized), out, t)
    summary["go26_figure_check"] = fc.to_dict("records")
    summary["go26_figure_check_monthly"] = fcm.to_dict("records")
    try:
        flux_root = bm.find_root(Path(args.flux))
    except FileNotFoundError:
        flux_root = None
        print(f"benchmark not found under {args.flux} (run scripts/01); skipping (c)-(e)")
    if flux_root is not None:
        meta = station_metadata(flux_root)
        frames = load_frames(flux_root, meta["site"])
        period_rows = period_month_rows(t, monthly, oc.load_availability(Path(args.avail), Path(args.avail_by_year))[1])
        pairs, bench, no_openet, ident["benchmark_summary"] = part_c(frames, meta, monthly, go_sites, zones, out,
                                                                     Path(args.derived), period_rows)
        e = bench[bench["model"] == "ensemble"].set_index(["stratum_type", "stratum"])
        ens = pairs[pairs["et_ensemble"].notna()]
        keep = ["n_months", "n_stations", "mbe_mm_month", "mbe_pct", "mbe_mm_d", "rmse_mm_month", "r2",
                "go26_openet_bias_mm_d"]
        withheld = bench[(bench["stratum_type"] == "site") & bench["openet_withheld"]]
        summary["benchmark_monthly_ensemble"] = {
            "period": [f"{ens['month'].min():%Y-%m}", f"{ens['month'].max():%Y-%m}"],
            "n_station_months": int(len(ens)), "n_stations": int(ens["site"].nunique()),
            "complete_months_without_openet": no_openet,
            "stations_withheld": withheld.loc[withheld["model"] == "ensemble", "stratum"].tolist(),
            "all": e.loc[("all", "all"), keep].to_dict(),
            "season": {s: e.loc[("season", s), keep].to_dict() for s in oc.SEASONS if ("season", s) in e.index},
            "koppen_group": {k: e.loc[("koppen_group", k), keep].to_dict()
                             for k in sorted(pairs["koppen_group"].unique())},
        }
        summary["benchmark_monthly_models_all"] = (
            bench[bench["stratum_type"] == "all"].set_index("model")[keep[:-1]].to_dict("index"))
        d, ident_d = part_d(frames, daily, go_sites, zones, Path(args.shared), out, t)
        ident["max_projection_daily"].update(ident_d)
        summary["shared_daily_all_seasons"] = d[d["season"] == "all"].to_dict("records")
        w = part_e(frames, meta, daily, go_sites, Path(args.digitized), out, Path(args.derived))
        summary["uic_winter"] = w.to_dict("records")
    write_json(out / "openet_api_go26_summary.json", summary)
    print(f"\nwrote results/openet_api_*.csv and openet_api_go26_summary.json to {out}"
          + (f"; tables with OpenET values (not for deposit) to {rel(Path(args.derived))}"
             if flux_root is not None else ""))


if __name__ == "__main__":
    main()
