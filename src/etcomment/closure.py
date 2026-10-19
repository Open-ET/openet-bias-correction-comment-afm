"""Energy-balance closure corrections and their failure modes.

Bowen-ratio (BR) closure (Twine et al., 2000) preserves beta = H/LE and forces
LE_c + H_c = Rn - G for the averaging period:

    LE_c = (Rn - G) / (1 + beta) = LE * (Rn - G) / (LE + H)

so at a daily step it is identical to multiplying LE by that day's inverse
closure ratio.  The FLUXNET2015/ONEFlux "energy balance ratio" (EBR) method
used by Volk et al. (2023a) applies the same ratio but computes it over a
moving window with outlier filtering, and fills the remaining gaps.

``br_diagnostics`` flags the days on which the BR factor is undefined,
negative or extreme, which is what an unscreened BR closure will pass on.

For daily data built from sub-daily records (AmeriFlux BASE, scripts/10):
``matched_daily_means`` forms LE + H and NETRAD - G from the records in which
all four are valid, ``fdq_bowen_ratio_le`` gives the daily Bowen-ratio closed
LE of flux-data-qaqc's 'br' correction (unscreened), ``screened_br_factor`` a
screened daily Bowen-ratio factor and ``ebr_factor`` the EBR factor of
flux-data-qaqc, the code behind the Volk et al. (2023a) benchmark and V24, for
every day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAMBDA_CONST = 2.45  # MJ kg-1, value used by Golkar & de Oliveira (2026)
BR_BOUNDS = (0.5, 2.0)  # daily Bowen-ratio factors kept by screened_br_factor
FDQ_LE_LIMITS = (-100.0, 850.0)  # W m-2: closed LE that flux-data-qaqc's 'br' correction leaves out (>= 0.4.0)


def wm2_to_mj(x):
    """Daily mean flux in W m-2 to MJ m-2 d-1."""
    return np.asarray(x, float) * 0.0864


def latent_heat_vaporization(t_air_c):
    """MJ kg-1 as a function of air temperature (Harrison, 1963; FAO-56)."""
    return 2.501 - 0.002361 * np.asarray(t_air_c, float)


def bowen_ratio_closure(le, h, rn, g):
    """Daily Bowen-ratio closed LE (same units as inputs)."""
    le, h, rn, g = (np.asarray(v, float) for v in (le, h, rn, g))
    flux = le + h
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = (rn - g) / flux
        return le * factor, factor


def ebr_closure(le, h, rn, g, window: int = 15, iqr_k: float = 1.5, min_valid: int = 5):
    """Simplified FLUXNET2015-style EBR closure on a daily series.

    The closure factor (Rn - G)/(LE + H) is computed per day, outliers beyond
    ``iqr_k`` x IQR are removed, and the factor applied to each day is the
    median of the valid factors in a centered ``window``-day window.  This is a
    transparent approximation of the ONEFlux/flux-data-qaqc routine, used only
    to show how windowing and filtering stabilize the daily BR factor.
    """
    le, h, rn, g = (pd.Series(np.asarray(v, float)) for v in (le, h, rn, g))
    flux = le + h
    energy = rn - g
    f = energy / flux
    valid = np.isfinite(f) & (flux.abs() > 1e-6) & (f > 0)
    q1, q3 = f[valid].quantile([0.25, 0.75])
    iqr = q3 - q1
    valid &= (f >= q1 - iqr_k * iqr) & (f <= q3 + iqr_k * iqr)
    fw = f.where(valid).rolling(window, center=True, min_periods=min_valid).median()
    return (le * fw).to_numpy(), fw.to_numpy()


def br_diagnostics(le, h, rn, g, extreme: float = 2.0) -> pd.DataFrame:
    """Flags for days on which the daily Bowen-ratio closure misbehaves.

    Columns
    -------
    h_negative        : H < 0 (advection or stable conditions)
    flux_near_zero    : |LE + H| < 5% of |Rn - G|  (factor diverges)
    factor_negative   : (Rn - G)/(LE + H) < 0      (closed LE changes sign)
    factor_extreme    : factor > ``extreme`` or < 1/``extreme``
    energy_negative   : Rn - G < 0
    """
    le, h, rn, g = (np.asarray(v, float) for v in (le, h, rn, g))
    flux = le + h
    energy = rn - g
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = energy / flux
    ok = np.isfinite(le) & np.isfinite(h) & np.isfinite(rn) & np.isfinite(g)
    out = pd.DataFrame({
        "h_negative": ok & (h < 0),
        "flux_near_zero": ok & (np.abs(flux) < 0.05 * np.abs(energy)),
        "factor_negative": ok & (factor < 0),
        "factor_extreme": ok & ((factor > extreme) | ((factor > 0) & (factor < 1 / extreme))),
        "energy_negative": ok & (energy < 0),
    })
    out["any_problem"] = out[["flux_near_zero", "factor_negative", "factor_extreme"]].any(axis=1)
    out["valid_inputs"] = ok
    return out


# -- daily closure factors from sub-daily records ----------------------------------------------------
def matched_daily_means(hh: pd.DataFrame, per_day: int) -> pd.DataFrame:
    """Daily means of LE + H (flux) and NETRAD - G (energy), W m-2, over the sub-daily records of ``hh``
    (columns LE, H, NETRAD, G) in which all four are valid, and the share of the day's ``per_day`` records
    they cover (frac).  Days without such a record are absent.

    Means of each variable over its own valid records would compare different parts of the day where data
    are missing by time of day (LE mostly at night at several GO26 towers, NETRAD at midday at the Mead
    towers) and bias the daily factor.
    """
    m = hh[["LE", "H", "NETRAD", "G"]].dropna()
    g = pd.DataFrame({"flux": m["LE"] + m["H"], "energy": m["NETRAD"] - m["G"]}).groupby(m.index.floor("D"))
    out = g.mean()
    out["frac"] = g.size() / per_day
    out.index.name = "date"
    return out


def fdq_bowen_ratio_le(le: pd.Series, flux: pd.Series, energy: pd.Series, limits=FDQ_LE_LIMITS) -> pd.Series:
    """Daily Bowen-ratio closed LE of flux-data-qaqc's 'br' correction (Volk et al., 2021; Twine et al., 2000).

    The day's available energy NETRAD - G (``energy``) is split with its measured Bowen ratio:
    LE_c = (NETRAD - G)/(1 + H/LE) = LE (NETRAD - G)/(LE + H), with ``flux`` = LE + H; the factor is not
    screened.  Days on which the Bowen ratio is undefined (LE = 0, or LE + H = 0) or LE_c <= limits[0] or
    >= limits[1] are left without a value, as in flux-data-qaqc since version 0.4.0 (6 October 2026; earlier
    versions kept every day).  All in W m-2.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        le_c = le * energy / flux
    return le_c.where(np.isfinite(le_c) & (le != 0) & (le_c > limits[0]) & (le_c < limits[1]))


def screened_br_factor(flux: pd.Series, energy: pd.Series, ok: pd.Series, bounds=BR_BOUNDS, window: int = 15,
                       min_days: int = 5) -> pd.DataFrame:
    """Daily Bowen-ratio closure factor (NETRAD - G)/(LE + H), screened, for every day of ``flux``.

    ``flux`` and ``energy`` are daily series on consecutive days and ``ok`` marks the days whose own factor
    may be used.  A day's factor passes where bounds[0] <= factor <= bounds[1] with LE + H > 0 and
    NETRAD - G > 0; every other day takes the median of the passing factors in a centred ``window``-day
    window (>= ``min_days`` of them), else the median of all passing factors.  Returns factor (the day's
    own, unscreened, on ``ok`` days), passed and screened (the factor applied).
    """
    ok = ok.astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (energy / flux).where(ok).replace([np.inf, -np.inf], np.nan)
    passed = ok & (flux > 0) & (energy > 0) & f.between(*bounds)
    fp = f.where(passed)
    screened = fp.fillna(fp.rolling(window, center=True, min_periods=min_days).median()).fillna(fp.median())
    return pd.DataFrame({"factor": f, "passed": passed, "screened": screened})


def ebr_factor(flux: pd.Series, energy: pd.Series, ok: pd.Series | None = None, iqr_k: float = 1.5,
               windows: tuple[int, int] = (15, 11), min_days: int = 5, bounds=BR_BOUNDS) -> pd.Series:
    """Energy-balance-ratio closure factor of flux-data-qaqc (Volk et al., 2023a; V24), for every day.

    ``flux`` = LE + H and ``energy`` = NETRAD - G are daily series on consecutive days; ``ok`` marks the
    days whose ratio may be used (default: all).

    1. EBR = flux / energy; values outside Q1 - k IQR .. Q3 + k IQR of the record (k = ``iqr_k``) are
       removed.
    2. Each day: the median EBR of the centred 15-day window if it holds >= ``min_days`` values, else of the
       centred 11-day window if it holds any; the result is rejected unless it is positive and 1/EBR lies
       strictly between ``bounds``.
    3. Remaining days: the "5-day climatology": the mean of step 2 by day of year, wrapped around the year
       end (days 362-366 before day 1, days 1-5 after day 366) and averaged over 11 days of year; as in
       flux-data-qaqc, the value for day k is the average centred on day k + 1 (truncated at the end).
    4. Remaining days: the median of the EBR values kept in step 1.

    Returns the factor 1/EBR applied to LE (and H); all NaN when no EBR passes step 1.  Run on the
    benchmark's daily flux and energy, this equals the benchmark's ebc_cf (flux-data-qaqc output) on
    99.5% of station-days (to 1e-6; every day at 98 of 157 stations, all days at US-Ne1-Ne3 and US-Bi1).
    Differences: flux-data-qaqc leaves the windows of a record's first 5-7 days empty (negative indices)
    and indexes the climatology by position, which misaligns it where a record lacks some days of year;
    here windows are truncated at both ends of the record and the climatology is indexed by day of year.
    A negative median, which flux-data-qaqc's test on |1/EBR| would keep, is rejected.
    """
    flux, energy = flux.astype(float), energy.astype(float)
    ok = pd.Series(True, index=flux.index) if ok is None else ok.astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (flux / energy).where(ok).replace([np.inf, -np.inf], np.nan)
    if not r.count():
        return pd.Series(np.nan, index=flux.index)
    q1, q3 = r.quantile([0.25, 0.75])
    r = r.where(r.between(q1 - iqr_k * (q3 - q1), q3 + iqr_k * (q3 - q1)))
    w1, w2 = windows
    n1 = r.notna().astype(float).rolling(w1, center=True, min_periods=1).sum()
    s = r.rolling(w1, center=True, min_periods=1).median().where(
        n1 >= min_days, r.rolling(w2, center=True, min_periods=1).median())
    with np.errstate(divide="ignore"):
        s = s.where((s > 0) & (1 / s > bounds[0]) & (1 / s < bounds[1]))
    doy = pd.Series(s.index.dayofyear, index=s.index)
    by = s.groupby(doy).mean().reindex(range(1, 367)).to_numpy()
    half = w2 // 2
    ext = pd.Series(np.r_[by[-half:], by, by[:half]], index=range(1 - half, 367 + half))
    roll = ext.rolling(2 * half + 1, center=True, min_periods=1).mean()
    clim = pd.Series(roll.reindex(range(2, 368)).to_numpy(), index=range(1, 367))
    return 1 / s.fillna(doy.map(clim)).fillna(r.median())
