"""Current OpenET (API collection v2.1) at GO26's towers and the benchmark croplands.

Pure functions behind ``scripts/12_openet_api_analysis.py`` (the only file I/O
is :func:`load_availability`).

1. GO26 Table 1 versus OpenET.  If GO26's daily observed ET is the sum of the
   available half-hourly ET with missing records counted as zero (pandas
   ``resample("D").sum()``), their site mean over a period of D calendar days
   is about the mean unclosed ET on recorded half-hours times the fraction of
   the period with LE data.  With OpenET standing in for the true ET, the
   expected Table 1 value (the zero-fill prediction) is

       P_zf = sum_m OE_m * a_y(m) / D,

   where OE_m is the OpenET ET of month m (mm) and a_y the fraction of the
   period's half-hours in year y with LE in the AmeriFlux BASE file (a partial
   year's records are taken to lie inside the period, as in scripts/11; the
   whole-year fraction gives a lower variant).  T1/OE should then scale with
   LE availability; the slope estimates the ratio of unclosed EC ET to OpenET
   where data exist.
   OE_m is the API's MONTHLY ensemble total, which OpenET computes at a monthly
   time step; it is not the sum of the daily ensemble values that GO26 used.
   Over GO26's periods the two differ by up to about 6 % (US-xSL).  Where daily
   extracts exist (nine towers) the daily-based mean and P_zf are given as a
   sensitivity check (:func:`go26_site_table`, :func:`site_summary`).
2. Monthly and daily OpenET bias against the closed ET of the Volk et al.
   (2023a) benchmark, pooled over station-months, by Koppen class and group,
   season and site.  Seasons as GO26: winter DJF, spring MAM, summer JJA,
   fall SON.
3. GO26's US-UiC winter observations (Fig. 8, digitized) against OpenET,
   reference ET and precipitation, and the benchmark's measured winter ET.

Monthly OpenET totals are absent for months without data (snow or persistent
cloud; US-MN1-3 miss 12 of the 60 months of their GO26 period).  Period means
fill such months with the site's mean for the same calendar month over
2016-2025 and record which months were filled.

OpenET data (OpenET, https://openetdata.org) are subject to the OpenET API Terms
of Service and are not CC BY: attribute them wherever values are shown, and
publish only statistics from which no single OpenET value, and no sum or
difference over fewer than MIN_STRATUM_MONTHS site-months, can be solved for.
With everything public known (benchmark ET, digitized GO26 series, which months
or days enter each statistic), every published statistic is a linear functional
of the unknown OpenET values or, for RMSE, R2, r and slopes, has a gradient at
the true values.  :func:`projection_diagonal` tests single values: a value is
determined when its unit vector lies in the span of these rows.
:func:`small_supports` tests sums: it finds every combination of the rows that
is nonzero on fewer than MIN_STRATUM_MONTHS site-months.  :func:`withheld_stations`
uses the first test to choose the per-station rows of the benchmark summary
that must stay blank.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import null_space

from etcomment import benchmark as bm
from etcomment import metrics as mt

MODELS = ["ensemble", "disalexi", "eemetric", "geesebal", "ptjpl", "sims", "ssebop"]
SEASONS = ["winter", "spring", "summer", "fall"]
_SEASON_OF_MONTH = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                    6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"}
# GO26 Table 4: OpenET ensemble bias on their test set by season (mm/d)
GO26_SEASON_BIAS = {"winter": 0.081, "spring": 0.412, "summer": 1.342, "fall": 0.573}
# irrigated alfalfa and cotton towers in dry climates (Table 2 of the comment)
IRRIGATED_DRY = ["US-UTJ", "US-UTD", "US-UTV", "US-NSb"]
CLIMATOLOGY = ("2016-01", "2025-12")
DAYS_PER_YEAR = 365.25
AVAIL_PERIOD_COLS = ["site", "le_avail_period"]
AVAIL_YEAR_COLS = ["site", "year", "le_avail"]
# smallest number of site-months behind a published OpenET statistic, or behind any sum or difference that the
# published statistics give together (see withheld_stations and small_supports)
MIN_STRATUM_MONTHS = 6
# a value whose projection onto the published functionals exceeds 1 - IDENT_TOL can be solved for
IDENT_TOL = 1e-9


def season(months) -> np.ndarray:
    """GO26 season names for month numbers (winter = DJF, spring = MAM, summer = JJA, fall = SON)."""
    return np.array([_SEASON_OF_MONTH[int(m)] for m in np.atleast_1d(np.asarray(months))], dtype=object)


def month_starts(start: str, end: str) -> pd.DatetimeIndex:
    """Month starts from ``start`` to ``end`` ('YYYY-MM', inclusive)."""
    return pd.date_range(f"{start}-01", f"{end}-01", freq="MS")


def calendar_days(months) -> int:
    """Number of calendar days in the months of a date index."""
    return int(np.asarray(pd.DatetimeIndex(months).days_in_month).sum())


# ----------------------------------------------------------------------------- period means

def fill_monthly(monthly: pd.Series, start: str, end: str,
                 clim: tuple[str, str] = CLIMATOLOGY) -> tuple[pd.Series, list[str]]:
    """Monthly totals over ``start``..``end``; months without data take the same calendar month's mean.

    ``monthly`` is indexed by dates in each month (absent or NaN = no data).  The
    climatology uses every available month in ``clim``; a month stays NaN when
    that calendar month has no data in any year.  Returns the filled series and
    the filled months ('YYYY-MM').
    """
    idx = pd.DatetimeIndex(monthly.index).to_period("M").to_timestamp()
    if idx.has_duplicates:
        raise ValueError("duplicate months in the monthly series")
    s = pd.Series(np.asarray(monthly, float), index=idx).dropna()
    c = s[(s.index >= pd.Timestamp(f"{clim[0]}-01")) & (s.index <= pd.Timestamp(f"{clim[1]}-01"))]
    clim_mean = c.groupby(c.index.month).mean()
    months = month_starts(start, end)
    out = s.reindex(months)
    gap = out.isna().to_numpy()
    out.iloc[gap] = clim_mean.reindex(months.month[gap]).to_numpy()
    return out, [d.strftime("%Y-%m") for d in months[gap]]


def fill_weights(have, start: str, end: str, clim: tuple[str, str] = CLIMATOLOGY) -> pd.DataFrame:
    """Weights W with ``fill_monthly(x, start, end, clim)[0] == W @ x`` for the months with data ``have``.

    Rows: the months of ``start``..``end``; columns: ``have`` (month starts).  A
    month with data weighs itself, a filled month the same calendar month of
    the climatology years equally; a month that cannot be filled is a NaN row.
    """
    have = pd.DatetimeIndex(have).to_period("M").to_timestamp().sort_values()
    c = have[(have >= pd.Timestamp(f"{clim[0]}-01")) & (have <= pd.Timestamp(f"{clim[1]}-01"))]
    w = pd.DataFrame(0.0, index=month_starts(start, end), columns=have)
    for m in w.index:
        same = [m] if m in have else list(c[c.month == m.month])
        if same:
            w.loc[m, same] = 1.0 / len(same)
        else:
            w.loc[m] = np.nan
    return w


def mean_daily(filled: pd.Series) -> float:
    """Mean daily value (mm/d) of complete monthly totals: their sum over the calendar days."""
    if filled.empty or filled.isna().any():
        return np.nan
    return float(filled.sum() / calendar_days(filled.index))


def daily_period(daily: pd.Series, start: str, end: str) -> pd.Series:
    """Daily values on every calendar day of ``start``..``end`` ('YYYY-MM'); empty if any day lacks a value."""
    days = pd.date_range(f"{start}-01", pd.Timestamp(f"{end}-01") + pd.offsets.MonthEnd(0), freq="D")
    s = pd.Series(np.asarray(daily, float), index=pd.DatetimeIndex(daily.index)).reindex(days)
    return s if s.notna().all() else s.iloc[:0]


def zero_fill_prediction(filled: pd.Series, avail_by_year) -> float:
    """P_zf = sum_m OE_m * a_y(m) / D (mm/d); NaN if a year's availability is unknown."""
    a = pd.Series(avail_by_year, dtype=float)
    a.index = a.index.astype(int)
    idx = pd.DatetimeIndex(filled.index)
    w = a.reindex(idx.year).to_numpy()
    v = filled.to_numpy(float)
    if v.size == 0 or not (np.isfinite(w).all() and np.isfinite(v).all()):
        return np.nan
    return float((v * w).sum() / calendar_days(idx))


def period_availability(by_year: pd.DataFrame) -> pd.Series:
    """LE availability within the study period for each year (``by_year`` indexed by year).

    le_avail is the fraction of the whole year's half-hours with LE.  For a year
    the period covers only in part, scripts/11 takes the year's records to fall
    inside the period: days_available = min(le_avail x days_in_year,
    days_in_period).  This returns days_available / days_in_period (computed
    from le_avail when the column is absent), which matches le_avail_period
    and suits periods that end with the record; for whole years it is
    le_avail.  Without day counts the whole-year le_avail is returned.
    """
    if {"days_available", "days_in_period"} <= set(by_year.columns):
        return (by_year["days_available"] / by_year["days_in_period"]).clip(upper=1.0)
    if {"days_in_year", "days_in_period"} <= set(by_year.columns):
        return (by_year["le_avail"] * by_year["days_in_year"] / by_year["days_in_period"]).clip(upper=1.0)
    return by_year["le_avail"]


# ----------------------------------------------------------------------------- AmeriFlux LE availability

def load_availability(period_csv: str | Path, by_year_csv: str | Path
                      ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[str]]:
    """LE availability tables written by scripts/11 (either may be None) and messages about missing ones.

    ``period_csv`` needs columns site, le_avail_period; ``by_year_csv`` needs site,
    year, le_avail.  Availabilities are fractions (0-1); other values raise.
    """
    out: list[pd.DataFrame | None] = []
    msgs: list[str] = []
    for path, cols in ((Path(period_csv), AVAIL_PERIOD_COLS), (Path(by_year_csv), AVAIL_YEAR_COLS)):
        if not path.exists():
            msgs.append(f"{path.name} not found (run scripts/11_ameriflux_availability.py first)")
            out.append(None)
            continue
        df = pd.read_csv(path)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            msgs.append(f"{path.name} lacks columns {missing}")
            out.append(None)
            continue
        v = pd.to_numeric(df[cols[-1]], errors="coerce")
        if ((v < 0) | (v > 1 + 1e-9)).any():
            raise ValueError(f"{path}: {cols[-1]} must be a fraction between 0 and 1")
        df[cols[-1]] = v
        if "year" in cols:
            df["year"] = df["year"].astype(int)
        df["site"] = df["site"].astype(str)
        out.append(df)
    return out[0], out[1], msgs


# ----------------------------------------------------------------------------- GO26 towers

def go26_site_table(sites: pd.DataFrame, monthly: pd.DataFrame, avail_period: pd.DataFrame | None = None,
                    avail_by_year: pd.DataFrame | None = None, models: list[str] = MODELS,
                    clim: tuple[str, str] = CLIMATOLOGY, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """OpenET over each GO26 study period next to GO26's Table 1 mean.

    ``sites``: config/go26_sites.csv (site, koppen_go26, period_start, period_end,
    mean_obs_et_mm_d, map_mm, in_volk_benchmark).  ``monthly``: API monthly
    totals (site, date, et_<model>, eto, pr; mm).  OpenET means are totals over
    calendar days (gap months filled, :func:`fill_monthly`); ETo and
    precipitation are annual equivalents (x 365.25).  With ``daily`` (API daily
    ensemble: site, date, et_ensemble), towers whose daily values cover the
    whole period also get the daily-based mean (oe_daily_mm_d), T1 over it and
    the daily zero-fill prediction (p_zf_daily_mm_d = mean of OE_d * a_y(d)).
    """
    rows = []
    for _, g in sites.sort_values("site").iterrows():
        s, start, end = str(g["site"]), str(g["period_start"]), str(g["period_end"])
        m = monthly[monthly["site"] == s].set_index("date").sort_index()
        r = {"site": s, "koppen_go26": g.get("koppen_go26"), "period_start": start, "period_end": end,
             "in_benchmark": bool(g.get("in_volk_benchmark", False)),
             "irrigated_dry_climate": s in IRRIGATED_DRY}
        if m.empty:
            rows.append(r)
            continue
        filled, months = fill_monthly(m["et_ensemble"], start, end, clim)
        r.update({"n_days": calendar_days(filled.index), "n_months": int(filled.size),
                  "n_months_filled": len(months), "months_filled": ";".join(months),
                  "oe_mm_d": mean_daily(filled)})
        r["oe_annual_mm"] = r["oe_mm_d"] * DAYS_PER_YEAR
        for v in ("pr", "eto"):
            f, _ = fill_monthly(m[v], start, end, clim) if v in m else (pd.Series(dtype=float), [])
            r[f"{v}_annual_mm"] = mean_daily(f) * DAYS_PER_YEAR
        for mod in models:
            if mod == "ensemble" or f"et_{mod}" not in m:
                continue
            f, mm = fill_monthly(m[f"et_{mod}"], start, end, clim)
            r[f"{mod}_mm_d"] = mean_daily(f)
            r[f"{mod}_n_months_filled"] = len(mm)
        r["map_mm_go26"] = float(g["map_mm"]) if "map_mm" in g else np.nan
        r["t1_mm_d"] = float(g["mean_obs_et_mm_d"])
        r["t1_annual_mm"] = r["t1_mm_d"] * DAYS_PER_YEAR
        r["ratio_t1_oe"] = r["t1_mm_d"] / r["oe_mm_d"]
        r["oe_annual_over_pr"] = r["oe_annual_mm"] / r["pr_annual_mm"]
        r["t1_annual_over_pr"] = r["t1_annual_mm"] / r["pr_annual_mm"]
        a = np.nan
        if avail_period is not None:
            hit = avail_period.loc[avail_period["site"] == s, "le_avail_period"]
            a = float(hit.iloc[0]) if len(hit) else np.nan
        r["le_avail_period"] = a
        r["t1_over_avail"] = r["t1_mm_d"] / a if a and np.isfinite(a) else np.nan
        r["t1_over_avail_over_oe"] = r["t1_over_avail"] / r["oe_mm_d"]
        r["p_zf_mm_d"] = r["p_zf_uniform_mm_d"] = np.nan
        ay = avail_by_year[avail_by_year["site"] == s].set_index("year") if avail_by_year is not None else None
        if ay is not None and len(ay):
            r["p_zf_mm_d"] = zero_fill_prediction(filled, period_availability(ay))
            r["p_zf_uniform_mm_d"] = zero_fill_prediction(filled, ay["le_avail"])
        r["t1_minus_p_zf"] = r["t1_mm_d"] - r["p_zf_mm_d"]
        if daily is not None:  # sensitivity: the daily ensemble instead of the monthly totals
            r["oe_daily_mm_d"] = r["ratio_t1_oe_daily"] = r["p_zf_daily_mm_d"] = np.nan
            d = daily[daily["site"] == s]
            dd = daily_period(d.set_index("date")["et_ensemble"], start, end) if len(d) else d.iloc[:0]
            if len(dd):
                r["oe_daily_mm_d"] = float(dd.mean())
                r["ratio_t1_oe_daily"] = r["t1_mm_d"] / r["oe_daily_mm_d"]
                if ay is not None and len(ay):
                    w = period_availability(ay).reindex(dd.index.year).to_numpy(float)
                    r["p_zf_daily_mm_d"] = float(np.mean(dd.to_numpy() * w)) if np.isfinite(w).all() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def fit_through_origin(y, x) -> dict:
    """Least-squares slope of y = b x and the RMSE of its residuals."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0 or not np.any(x != 0):
        return {"n": int(x.size), "slope": np.nan, "rmse": np.nan}
    b = float(np.sum(x * y) / np.sum(x * x))
    return {"n": int(x.size), "slope": b, "rmse": float(np.sqrt(np.mean((y - b * x) ** 2)))}


def correlation(x, y) -> dict:
    """Pearson and Spearman correlation (with two-sided p-values) over finite pairs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(x.size), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan,
                "spearman_p": np.nan}
    p, s = stats.pearsonr(x, y), stats.spearmanr(x, y)
    return {"n": int(x.size), "pearson_r": float(p.statistic), "pearson_p": float(p.pvalue),
            "spearman_r": float(s.statistic), "spearman_p": float(s.pvalue)}


def _versus(t: pd.DataFrame, col: str) -> dict:
    """T1 against a predictor column: r, slope through the origin, fit RMSE and plain differences."""
    fit = fit_through_origin(t["t1_mm_d"], t[col])
    d = (t["t1_mm_d"] - t[col]).dropna()
    return {"n": fit["n"], "pearson_r": correlation(t[col], t["t1_mm_d"])["pearson_r"],
            "slope_through_origin": fit["slope"], "rmse_fit": fit["rmse"],
            "mean_diff_t1_minus": float(d.mean()) if len(d) else np.nan,
            "rmse_diff": float(np.sqrt(np.mean(d ** 2))) if len(d) else np.nan}


def _ratio_vs_availability(t: pd.DataFrame) -> dict:
    c = correlation(t["le_avail_period"], t["ratio_t1_oe"])
    return {**c, **{k: v for k, v in fit_through_origin(t["ratio_t1_oe"], t["le_avail_period"]).items()
                    if k != "n"}}


def site_summary(t: pd.DataFrame) -> dict:
    """Cross-site statistics of :func:`go26_site_table` for all towers and those outside the benchmark.

    With daily-based columns (``oe_daily_mm_d``), ``daily_oe_where_available``
    repeats the headline statistics with the daily-based OpenET mean and P_zf
    at the towers that have them (monthly-based values elsewhere).
    """
    out = {}
    for name, sub in (("all", t), ("not_in_benchmark", t[~t["in_benchmark"].astype(bool)])):
        out[name] = {
            "n_sites": int(len(sub)),
            "mean_t1_mm_d": float(sub["t1_mm_d"].mean()),
            "mean_oe_mm_d": float(sub["oe_mm_d"].mean()),
            "mean_ratio_t1_oe": float(sub["ratio_t1_oe"].mean()),
            "ratio_vs_availability": _ratio_vs_availability(sub),
            "t1_vs_oe": _versus(sub, "oe_mm_d"),
            "t1_vs_p_zf": _versus(sub, "p_zf_mm_d"),
            "t1_vs_p_zf_uniform": _versus(sub, "p_zf_uniform_mm_d"),
        }
        if "oe_daily_mm_d" in sub and sub["oe_daily_mm_d"].notna().any():
            d = sub.assign(oe_mm_d=sub["oe_daily_mm_d"].fillna(sub["oe_mm_d"]),
                           p_zf_mm_d=sub["p_zf_daily_mm_d"].fillna(sub["p_zf_mm_d"]))
            d["ratio_t1_oe"] = d["t1_mm_d"] / d["oe_mm_d"]
            out[name]["daily_oe_where_available"] = {
                "n_sites_daily": int(sub["oe_daily_mm_d"].notna().sum()),
                "sites_daily": ";".join(sub.loc[sub["oe_daily_mm_d"].notna(), "site"]),
                "ratio_vs_availability": _ratio_vs_availability(d),
                "t1_vs_oe": _versus(d, "oe_mm_d"),
                "t1_vs_p_zf": _versus(d, "p_zf_mm_d"),
            }
    return out


def numeric_leaves(x, path: tuple = ()):
    """(path, value) of every numeric leaf of nested dicts (booleans and strings skipped)."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from numeric_leaves(v, path + (str(k),))
    elif isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_)):
        yield path, float(x)


def summary_gradients(t: pd.DataFrame, columns, keep=None, rel_step: float = 1e-6) -> dict[tuple, pd.Series]:
    """Derivatives of the statistics of :func:`site_summary` with respect to per-site values of ``t``.

    ``columns``: the per-site values that vary (e.g. oe_mm_d, p_zf_mm_d); ratio_t1_oe follows oe_mm_d.
    ``keep``: the statistic paths (tuples of keys, as from :func:`numeric_leaves`) to return; default all.
    Central differences with a step of rel_step x max(1, |value|).  Returns {path: Series indexed by (site,
    column)} with the nonzero derivatives (rank statistics such as Spearman's have none).  Each published
    statistic thus enters the identifiability checks as sum_(site, column) derivative x the functional of the
    OpenET values that the column is.
    """
    base = t.set_index("site")

    def leaves(tt: pd.DataFrame) -> dict:
        tt = tt.copy()
        tt["ratio_t1_oe"] = tt["t1_mm_d"] / tt["oe_mm_d"]
        return {p: v for p, v in numeric_leaves(site_summary(tt.reset_index())) if keep is None or p in keep}

    paths = list(leaves(base))
    out: dict[tuple, dict] = {p: {} for p in paths}
    for s in base.index:
        for c in columns:
            if c not in base or not np.isfinite(base.loc[s, c]):
                continue
            h = rel_step * max(1.0, abs(float(base.loc[s, c])))
            up, dn = base.copy(), base.copy()
            up.loc[s, c] += h
            dn.loc[s, c] -= h
            lu, ld = leaves(up), leaves(dn)
            for p in paths:
                if p in lu and p in ld and np.isfinite(lu[p]) and np.isfinite(ld[p]) and lu[p] != ld[p]:
                    out[p][(s, c)] = (lu[p] - ld[p]) / (2 * h)
    return {p: pd.Series(g, dtype=float) for p, g in out.items() if g}


# ----------------------------------------------------------------------------- bias statistics

def bias_stats(pred, obs, n_days=None) -> dict:
    """n, means, MBE (and in % of the mean observation), RMSE and R2 = 1 - SSE/SST.

    With ``n_days`` (days per pair, e.g. for monthly totals) the MBE is also
    given per day, as the mean of (P - O) / n_days.
    """
    p, o = np.asarray(pred, float), np.asarray(obs, float)
    ok = np.isfinite(p) & np.isfinite(o)
    nd = None if n_days is None else np.asarray(n_days, float)[ok]
    p, o = p[ok], o[ok]
    out = {"n": int(p.size), "mean_obs": np.nan, "mean_pred": np.nan, "mbe": np.nan, "mbe_pct": np.nan,
           "rmse": np.nan, "r2": np.nan}
    if nd is not None:
        out["mbe_per_day"] = np.nan
    if p.size == 0:
        return out
    mbe = float(np.mean(p - o))
    out.update({"mean_obs": float(o.mean()), "mean_pred": float(p.mean()), "mbe": mbe,
                "mbe_pct": 100 * mbe / float(o.mean()) if o.mean() != 0 else np.nan,
                "rmse": float(np.sqrt(np.mean((p - o) ** 2))),
                "r2": mt.r2_cod(p, o) if p.size > 2 else np.nan})
    if nd is not None:
        out["mbe_per_day"] = float(np.mean((p - o) / nd))
    return out


def _label(key) -> str:
    key = key if isinstance(key, tuple) else (key,)
    return ", ".join(str(k) for k in key)


def _finite(pairs: pd.DataFrame, pred: str, obs: str) -> pd.DataFrame:
    return pairs[np.isfinite(pairs[pred].astype(float)) & np.isfinite(pairs[obs].astype(float))]


def _strata_groups(p: pd.DataFrame, strata: dict[str, list[str]]):
    """(stratum_type, stratum label, pairs) for every stratum, in the order of :func:`stratified_bias`."""
    for name, cols in strata.items():
        for key, g in (p.groupby(list(cols), sort=True) if cols else [("all", p)]):
            if len(g):
                yield name, _label(key), g


def stratified_bias(pairs: pd.DataFrame, strata: dict[str, list[str]], pred: str = "pred", obs: str = "obs",
                    site: str = "site", n_days: str | None = "n_days", time: str | None = None) -> pd.DataFrame:
    """Pooled bias statistics (:func:`bias_stats`) and the number of stations for every stratum.

    ``strata`` maps a stratum type to its grouping columns ([] = all pairs).
    With ``time``, the stratum's first and last values of that column are given
    too.  Station-weighted MBEs are deliberately not computed: next to the
    pooled MBEs in mm/month and mm/d they let single station-months be solved
    for (US-OF1 2017-06 and US-OF6 2018-06 in the benchmark, the only 30-day
    months among seven withheld Cfa months).
    """
    rows = []
    for name, lab, g in _strata_groups(_finite(pairs, pred, obs), strata):
        r = {"stratum_type": name, "stratum": lab, "n_stations": int(g[site].nunique()),
             **bias_stats(g[pred], g[obs], g[n_days] if n_days else None)}
        if time:
            r["first"], r["last"] = g[time].min(), g[time].max()
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- identifiability

def projection_diagonal(rows: pd.DataFrame) -> pd.Series:
    """Squared length of each unknown's unit vector projected onto the span of the functionals in ``rows``.

    ``rows`` holds one functional per row (the weights w of sum_j w_j x_j) and
    one column per unknown x_j.  An unknown whose value is 1 (within IDENT_TOL)
    is a linear combination of the functionals, so it can be solved for from
    them.  Nonlinear statistics (RMSE, R2, r) enter through their gradients at
    the true values; a value below 1 is then not even locally determined: the
    solutions near the true values form a continuum along which it changes
    (for a generic point, where the rank of the gradients is constant).
    """
    a = rows.to_numpy(float)
    a = a[np.abs(a).sum(axis=1) > 0]
    if not a.size:
        return pd.Series(0.0, index=rows.columns)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    _, sv, vt = np.linalg.svd(a, full_matrices=False)
    rank = int((sv > sv[0] * max(a.shape) * np.finfo(float).eps).sum())
    return pd.Series((vt[:rank] ** 2).sum(axis=0), index=rows.columns)


def _rank(a: np.ndarray, rtol: float) -> int:
    if a.size == 0:
        return 0
    s = np.linalg.svd(a, compute_uv=False)
    return int((s > s[0] * rtol).sum()) if s[0] > 0 else 0


def _parallel_classes(cols: np.ndarray, rtol: float) -> list[list[int]]:
    """Indices of parallel columns of ``cols`` (no zero columns)."""
    u = cols / np.linalg.norm(cols, axis=0)
    lead = np.argmax(np.abs(u) > 1e3 * rtol, axis=0)
    u = u * np.sign(u[lead, np.arange(u.shape[1])])
    digits = max(1, int(round(-np.log10(rtol))) - 1)
    keys: dict = {}
    for j in range(u.shape[1]):
        keys.setdefault(tuple(np.round(u[:, j], digits)), []).append(j)
    out = list(keys.values())
    for c in out:  # rounding may split parallel columns (less reduction, same result) but must not merge others
        if len(c) > 1 and np.abs(u[:, c] - u[:, [c[0]]]).max() > 10.0 ** (1 - digits):
            raise RuntimeError("small_supports: columns grouped as parallel are not")
    return out


def small_supports(rows: pd.DataFrame, group=None, min_groups: int = MIN_STRATUM_MONTHS, rtol: float = 1e-9,
                   max_subsets: int = 2_000_000) -> list[tuple]:
    """Sets of fewer than ``min_groups`` site-months on which a combination of the published functionals lives.

    ``rows`` holds one published linear functional per row (the weights w of
    sum_j w_j x_j) and one column per unknown OpenET value; ``group`` maps a
    column label to its site-month (default: the label itself).  A
    combination of the rows that is zero outside a set U of site-months
    reveals a sum (a difference, or another combination) of the values in U:
    rows of 6 site-months each can still give a single month by subtraction.
    Returns every minimal such U with fewer than ``min_groups`` site-months
    (sorted tuples of groups, smallest first); [] when there is none.

    Exact up to the relative tolerance ``rtol`` of the rank decisions.
    Columns that are parallel always enter a combination in a fixed ratio (a
    class); a class that spans ``min_groups`` or more site-months must get
    weight 0 in a small combination, so its columns restrict the combinations.
    The remaining columns are projected and classes re-formed until nothing
    changes, and the remaining small classes are enumerated: a set U lives
    under the rows exactly when dropping its columns lowers the rank.  Raises
    RuntimeError when more than ``max_subsets`` sets would have to be tried.
    """
    grp = group if callable(group) else (lambda k: k)
    rows = rows.fillna(0.0)  # a value outside a row's set has weight 0
    f = rows.loc[:, (rows.abs() > 0).any()]
    if f.empty:
        return []
    a = f.to_numpy(float)
    a = a / np.abs(a).max(axis=1, keepdims=True)
    groups = [grp(k) for k in f.columns]
    basis = np.eye(a.shape[0])  # combinations still allowed (columns: directions in the row space)
    alive = np.arange(a.shape[1])
    while alive.size:
        p = basis.T @ a[:, alive]
        norm = np.linalg.norm(p, axis=0)
        keep = norm > rtol * max(1.0, float(norm.max(initial=0.0)))
        alive, p = alive[keep], p[:, keep]
        if not alive.size:
            break
        big = [c for c in _parallel_classes(p, rtol) if len({groups[alive[j]] for j in c}) >= min_groups]
        if not big:
            break
        cols = np.concatenate([p[:, c] for c in big], axis=1)
        nb = null_space(cols.T, rcond=rtol)
        basis = basis @ nb
        drop = {j for c in big for j in c}
        alive = alive[[j for j in range(alive.size) if j not in drop]]
        if basis.shape[1] == 0:
            alive = alive[:0]
    if not alive.size:
        return []
    p = basis.T @ a[:, alive]
    d = _rank(p, rtol)
    if d == 0:
        return []
    cls = _parallel_classes(p, rtol)
    cg = [frozenset(groups[alive[j]] for j in c) for c in cls]
    order = [i for i in sorted(range(len(cls)), key=lambda i: (len(cg[i]), i)) if len(cg[i]) < min_groups]
    found: set[frozenset] = set()
    tried: set[frozenset] = set()
    for k in range(1, min_groups):
        for u in itertools.combinations(order, k):
            g = frozenset().union(*(cg[i] for i in u))  # every set of < min_groups site-months arises this way
            if len(g) >= min_groups or g in tried or any(x <= g for x in found):
                continue
            tried.add(g)
            if len(tried) > max_subsets:
                raise RuntimeError(f"small_supports: more than {max_subsets} sets to try ({len(cls)} classes left)")
            rest = [j for i in range(len(cls)) if not cg[i] <= g for j in cls[i]]  # drop every column inside g
            if _rank(p[:, rest], rtol) < d:
                found.add(g)
    minimal = [g for g in found if not any(x < g for x in found)]
    return sorted((tuple(sorted(g)) for g in minimal), key=lambda g: (len(g), g))


def published_strata(pairs: pd.DataFrame, strata: dict[str, list[str]], withheld=(), pred: str = "pred",
                     obs: str = "obs", site: str = "site", min_months: int = MIN_STRATUM_MONTHS) -> set[tuple[str, str]]:
    """(stratum_type, stratum) of the rows of :func:`stratified_bias` that are published.

    A row is published when it holds at least ``min_months`` pairs and is not
    the row of a ``withheld`` station (stratum grouped by ``site`` alone).
    """
    return {(name, lab) for name, lab, g in _strata_groups(_finite(pairs, pred, obs), strata)
            if len(g) >= min_months and not (list(strata[name]) == [site] and lab in set(withheld))}


def stratum_functionals(pairs: pd.DataFrame, strata: dict[str, list[str]], published, pred: str = "pred",
                        obs: str = "obs", n_days: str = "n_days", rmse: bool = True, per_day=None) -> pd.DataFrame:
    """Functionals of the predictions given by the published rows of :func:`stratified_bias`.

    With the observations and each stratum's pairs known (they follow from the
    public benchmark), a published row's mean prediction, MBE and MBE in %
    give the sum of its predictions, its MBE per day the sum of prediction /
    ``n_days``, and with ``rmse`` its RMSE and R2 the gradient of
    sum (P - O)^2 at the true values.  ``published``: (stratum_type, stratum)
    of the rows written; ``per_day``: those of them whose MBE per day is
    written (default: all).  Rows: (stratum_type, stratum, statistic);
    columns: the index of the pairs with finite values.
    """
    p = _finite(pairs, pred, obs)
    inv_d = 1.0 / p[n_days].to_numpy(float)
    err = p[pred].to_numpy(float) - p[obs].to_numpy(float)
    rows = {}
    for name, lab, g in _strata_groups(p, strata):
        if (name, lab) not in published:
            continue
        m = p.index.isin(g.index)
        rows[(name, lab, "sum")] = m.astype(float)
        if per_day is None or (name, lab) in per_day:
            rows[(name, lab, "sum_per_day")] = np.where(m, inv_d, 0.0)
        if rmse:
            rows[(name, lab, "rmse")] = np.where(m, err, 0.0)
    return pd.DataFrame(rows, index=p.index).T if rows else pd.DataFrame(columns=p.index, dtype=float)


def identifiability(pairs: pd.DataFrame, strata: dict[str, list[str]], published, pred: str = "pred",
                    obs: str = "obs", n_days: str = "n_days") -> pd.DataFrame:
    """Projection diagonal of every pair under the published rows: 'linear' (sums only) and 'with_rmse'.

    Index: the pairs with finite values.  A pair is determined (its value can be
    solved for) where a column exceeds 1 - IDENT_TOL; 'with_rmse' is never
    smaller than 'linear'.
    """
    f = stratum_functionals(pairs, strata, published, pred, obs, n_days, rmse=True)
    lin = f[[k[-1] != "rmse" for k in f.index]] if len(f) else f
    return pd.DataFrame({"linear": projection_diagonal(lin), "with_rmse": projection_diagonal(f)})


def withheld_stations(pairs: pd.DataFrame, pred: str = "pred", obs: str = "obs", site: str = "site",
                      group: str = "koppen", min_months: int = MIN_STRATUM_MONTHS,
                      strata: dict[str, list[str]] | None = None, n_days: str = "n_days") -> list[str]:
    """Stations whose own rows of the stratified summary are withheld so no monthly value can be solved for.

    Size rule: stations with fewer than ``min_months`` pairs, then the smallest
    other stations of the same ``group`` (Koppen class) until the withheld
    months of every group are 0 or at least ``min_months``.  The rule alone is
    not enough, because strata of different types combine: when a group's
    withheld months contain a single month of some length (e.g. one 30-day
    month among 31-day months), the pooled MBEs in mm/month and mm/d of the
    group, minus its published stations, give that month's value.  With
    ``strata`` (all rows of :func:`stratified_bias` with at least ``min_months``
    pairs are published), the smallest published station of a group is then
    withheld as long as :func:`identifiability` (with the RMSE) finds a
    determined pair in that group.  Raises ValueError when withholding stations
    cannot achieve that.
    """
    p = _finite(pairs, pred, obs)
    n = p.groupby(site).size()
    grp = p.groupby(site)[group].first()
    out = set(n[n < min_months].index)
    for k in sorted(grp.unique()):
        members = n[grp == k].sort_values(kind="stable")
        while 0 < members[members.index.isin(out)].sum() < min_months and not members.index.isin(out).all():
            out.add(members[~members.index.isin(out)].index[0])
    while strata is not None:
        pub = published_strata(p, strata, out, pred, obs, site, min_months)
        proj = identifiability(p, strata, pub, pred, obs, n_days)["with_rmse"]
        hit = sorted(set(p.loc[proj.index[proj > 1 - IDENT_TOL], group]))
        if not hit:
            break
        for k in hit:
            left = n[(grp == k) & ~n.index.isin(out)].sort_values(kind="stable")
            if left.empty:
                raise ValueError(f"{group} {k}: monthly values stay determined with all its stations withheld; "
                                 f"blank the statistics of the strata that contain them")
            out.add(left.index[0])
    return sorted(out)


def series_agreement(a: pd.Series, b: pd.Series) -> dict:
    """Agreement of two date-indexed series on common dates.

    n, means, mean(a - b), RMSE, Pearson r, and the least-squares scale through
    the origin (a = scale x b) with the RMSE of a - scale x b.  On seasonal
    series r is high whatever the amplitude; the scale shows amplitude errors.
    """
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if j.empty:
        return {"n": 0, "first": None, "last": None, "mean_a": np.nan, "mean_b": np.nan, "mean_diff": np.nan,
                "rmse": np.nan, "pearson_r": np.nan, "scale": np.nan, "rmse_scaled": np.nan}
    d = j["a"] - j["b"]
    f = fit_through_origin(j["a"], j["b"])
    return {"n": int(len(j)), "first": j.index.min().date().isoformat(), "last": j.index.max().date().isoformat(),
            "mean_a": float(j["a"].mean()), "mean_b": float(j["b"].mean()), "mean_diff": float(d.mean()),
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "pearson_r": float(np.corrcoef(j["a"], j["b"])[0, 1]) if len(j) > 2 else np.nan,
            "scale": f["slope"], "rmse_scaled": f["rmse"]}


def monthly_agreement(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Monthly means of two date-indexed series on their common dates: month, n, mean_a, mean_b, mean_diff (a - b)."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    g = j.groupby(j.index.to_period("M"))
    out = pd.DataFrame({"n": g.size(), "mean_a": g["a"].mean(), "mean_b": g["b"].mean()})
    out["mean_diff"] = out["mean_a"] - out["mean_b"]
    out.index = out.index.strftime("%Y-%m")
    return out.rename_axis("month").reset_index()


# ----------------------------------------------------------------------------- benchmark pairs

def monthly_pairs(daily: pd.DataFrame, api_monthly: pd.DataFrame, start: str = "2016-01-01",
                  max_gap_days: int = 5) -> pd.DataFrame:
    """Benchmark months usable for evaluation joined with API monthly totals.

    ``daily``: output of :func:`etcomment.benchmark.standardize`; months must be
    calendar-complete with at most ``max_gap_days`` gap-filled days (the V24
    rule).  ``api_monthly``: one station's API rows (date, et_<model>, ...).
    """
    g = bm.complete_months(daily.loc[pd.Timestamp(start):], max_gap_days)
    a = api_monthly.set_index("date") if "date" in api_monthly.columns else api_monthly
    a = a[[c for c in a.columns if c.startswith("et_")]].copy()
    a.index = pd.DatetimeIndex(a.index).to_period("M").to_timestamp()
    j = g.join(a, how="inner")
    j.index.name = "month"
    return j.reset_index()


def daily_pairs(daily: pd.DataFrame, api_daily: pd.Series, start: str = "2016-01-01") -> pd.DataFrame:
    """Measured (not gap-filled) benchmark days with closed and unclosed ET joined with daily OpenET."""
    d = daily.loc[pd.Timestamp(start):]
    d = d[d["valid_measured"]][["et_corr", "et"]]
    j = d.join(api_daily.rename("openet"), how="inner").dropna(subset=["openet"])
    j.index.name = "date"
    return j


# ----------------------------------------------------------------------------- winter checks

def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance (km, mean Earth radius 6371.0 km)."""
    p1, p2 = np.radians(np.asarray(lat1, float)), np.radians(np.asarray(lat2, float))
    dl = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


DEFAULT_WINTER_GROUPS = {"Jan": (1,), "Feb": (2,), "Mar": (3,), "Nov": (11,), "Dec": (12,),
                         "Jan-Feb": (1, 2), "Jan-Mar, Nov-Dec": (1, 2, 3, 11, 12)}


def winter_check(daily: pd.DataFrame, groups: dict[str, tuple[int, ...]] | None = None,
                 zero_tol: float = 0.015) -> pd.DataFrame:
    """Observed ET against OpenET, ETo and precipitation on days with an observation.

    ``daily`` is date-indexed with columns obs, openet, eto, pr (mm/d).  For each
    group of months: n, means, fraction of days with obs > ETo and obs > OpenET
    (over days where both exist), the fraction of days above 1 mm/d, the maximum
    and the number of near-zero observations.
    """
    groups = groups or DEFAULT_WINTER_GROUPS
    d = daily[daily["obs"].notna()]
    rows = []
    for label, months in groups.items():
        g = d[np.isin(d.index.month, months)]
        e, o = g.dropna(subset=["eto"]), g.dropna(subset=["openet"])
        rows.append({"months": label, "n_days": int(len(g)),
                     "mean_obs_mm_d": float(g["obs"].mean()) if len(g) else np.nan,
                     "mean_openet_mm_d": float(g["openet"].mean()) if len(g) else np.nan,
                     "mean_eto_mm_d": float(g["eto"].mean()) if len(g) else np.nan,
                     "mean_pr_mm_d": float(g["pr"].mean()) if len(g) else np.nan,
                     "frac_obs_gt_eto": float((e["obs"] > e["eto"]).mean()) if len(e) else np.nan,
                     "frac_obs_gt_openet": float((o["obs"] > o["openet"]).mean()) if len(o) else np.nan,
                     "frac_obs_gt_1mm_d": float((g["obs"] > 1.0).mean()) if len(g) else np.nan,
                     "max_obs_mm_d": float(g["obs"].max()) if len(g) else np.nan,
                     "n_obs_near_zero": int((g["obs"].abs() < zero_tol).sum())})
    return pd.DataFrame(rows)


def measured_climatology(frames: dict[str, pd.DataFrame], months: tuple[int, ...]) -> dict:
    """Measured closed ET in ``months`` pooled over stations (benchmark ``standardize`` frames).

    Returns station-days, the pooled mean of closed and unclosed ET, the range
    and median of station means, the station records' years, the fraction of
    days above 1 mm/d and the maximum, the benchmark's gridMET ETo and the
    fraction of days with closed ET > ETo.
    """
    parts, means = [], {}
    for s in sorted(frames):
        d = frames[s]
        d = d[d["valid_measured"] & np.isin(d.index.month, months)]
        if len(d):
            parts.append(d.assign(site=s))
            means[s] = float(d["et_corr"].mean())
    if not parts:
        return {"n_stations": 0, "n_days": 0}
    p = pd.concat(parts)
    e = p.dropna(subset=["eto"]) if "eto" in p else p.iloc[:0]
    sm = np.array(list(means.values()))
    return {"n_stations": len(means), "n_days": int(len(p)), "stations": ";".join(means),
            "first_year": int(p.index.year.min()), "last_year": int(p.index.year.max()),
            "mean_closed_mm_d": float(p["et_corr"].mean()), "mean_unclosed_mm_d": float(p["et"].mean()),
            "station_mean_min": float(sm.min()), "station_mean_median": float(np.median(sm)),
            "station_mean_max": float(sm.max()),
            "frac_closed_gt_1mm_d": float((p["et_corr"] > 1.0).mean()), "max_closed_mm_d": float(p["et_corr"].max()),
            "mean_eto_mm_d": float(pd.to_numeric(e["eto"], errors="coerce").mean()) if len(e) else np.nan,
            "frac_closed_gt_eto": float((e["et_corr"] > pd.to_numeric(e["eto"], errors="coerce")).mean())
            if len(e) else np.nan}
