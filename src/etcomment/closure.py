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
``matched_daily_means`` forms daily means of LE, H, NETRAD and G (and LE + H,
NETRAD - G) from the records in which all four are valid, ``fdq_factor`` runs
flux-data-qaqc (Volk et al., 2021; version pinned in environment.yml) on those
means and returns the daily closure factor of its 'br' (Bowen ratio, unscreened)
or 'ebr' (energy balance ratio) correction, and ``screened_br_factor`` gives a
screened daily Bowen-ratio factor (our screen, not flux-data-qaqc's).
"""
from __future__ import annotations

import configparser
import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

LAMBDA_CONST = 2.45  # MJ kg-1, value used by Golkar & de Oliveira (2026)
BR_BOUNDS = (0.5, 2.0)  # daily Bowen-ratio factors kept by screened_br_factor
FDQ_LE_LIMITS = (-100.0, 850.0)  # W m-2: corrected LE that flux-data-qaqc leaves out (>= 0.4.0; for reports only)
FDQ_COLUMNS = ("LE", "H", "NETRAD", "G")


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
    """Daily means of LE, H, NETRAD, G, LE + H (flux) and NETRAD - G (energy), W m-2, over the sub-daily
    records of ``hh`` (columns LE, H, NETRAD, G) in which all four are valid, and the share of the day's
    ``per_day`` records they cover (frac).  Days without such a record are absent.

    Means of each variable over its own valid records would compare different parts of the day where data
    are missing by time of day (LE mostly at night at several GO26 towers, NETRAD at midday at the Mead
    towers) and bias the daily factor.
    """
    m = hh[list(FDQ_COLUMNS)].dropna()
    g = m.assign(flux=m["LE"] + m["H"], energy=m["NETRAD"] - m["G"]).groupby(m.index.floor("D"))
    out = g.mean()
    out["frac"] = g.size() / per_day
    out.index.name = "date"
    return out


def fdq_version() -> str:
    """Installed flux-data-qaqc version (written into the outputs that use it)."""
    from importlib.metadata import version

    return version("fluxdataqaqc")


def fdq_factor(daily: pd.DataFrame, meth: str) -> pd.Series:
    """Daily closure factor of flux-data-qaqc's ``correct_data(meth)`` (Volk et al., 2021), for every day of ``daily``.

    ``daily`` holds daily means in W m-2 (columns LE, H, NETRAD, G) on consecutive days, NaN on days that may
    not be used.  flux-data-qaqc reads them through its own Data and QaQc classes (a config file and a CSV in a
    temporary folder; the station coordinates in the config are placeholders, which its daily corrections do
    not use) and corrects them without ET gap filling:

    * 'br' (Bowen ratio, Twine et al., 2000): LE_corr = (NETRAD - G)/(1 + H/LE), the factor not screened; the
      factor returned is LE_corr/LE = (NETRAD - G)/(LE + H), NaN where flux-data-qaqc leaves the day without a
      correction (Bowen ratio undefined, or LE_corr <= -100 or >= 850 W m-2).
    * 'ebr' (energy balance ratio, the FLUXNET2015/ONEFlux method with flux-data-qaqc's limits): its ebc_cf,
      which fills days without inputs from moving windows, adjacent years and a day-of-year climatology.

    All NaN where no day has all four inputs (flux-data-qaqc is then not run).  Where G is 0 on every day
    (none in BASE), NETRAD and G are both passed 1 W m-2 higher: flux-data-qaqc drops an all-zero column as
    missing, and both corrections use only NETRAD - G.
    """
    import fluxdataqaqc as fdq  # pinned in environment.yml

    if meth not in ("br", "ebr"):
        raise ValueError(f"meth must be 'br' or 'ebr', not {meth!r}")
    d = daily[list(FDQ_COLUMNS)].astype(float)
    if not d.notna().all(axis=1).any():
        return pd.Series(np.nan, index=daily.index)
    if (d["G"].dropna() == 0).all():  # G = 0 (none in BASE): flux-data-qaqc drops an all-zero column as missing,
        d = d.assign(NETRAD=d["NETRAD"] + 1.0, G=d["G"] + 1.0)  # so shift both; 'br' and 'ebr' use only NETRAD - G
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        d.rename_axis("date").to_csv(tmp / "daily.csv", date_format="%Y-%m-%d", na_rep="-9999")
        c = configparser.ConfigParser()
        c["METADATA"] = {"climate_file_path": str(tmp / "daily.csv"), "site_id": "site", "missing_data_value": "-9999",
                         "station_elevation": "0", "station_latitude": "40", "station_longitude": "-100"}
        c["DATA"] = {"datestring_col": "date"}
        for v, key in zip(FDQ_COLUMNS, ("latent_heat_flux", "sensible_heat_flux", "net_radiation", "ground_flux")):
            c["DATA"][f"{key}_col"], c["DATA"][f"{key}_units"] = v, "w/m2"
        with open(tmp / "config.ini", "w") as fh:
            c.write(fh)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):  # flux-data-qaqc reports its steps on stdout
            q = fdq.QaQc(fdq.Data(tmp / "config.ini"), drop_gaps=False)
            q.correct_data(meth=meth, et_gap_fill=False)
            out, col = q.df, q.variables  # flux-data-qaqc's names -> its columns (inputs may be renamed)
    need = "ebc_cf" if meth == "ebr" else "LE_corr"
    if col.get(need) not in out:
        raise RuntimeError(f"flux-data-qaqc made no '{meth}' correction:\n{log.getvalue()}")
    out.index = pd.DatetimeIndex(out.index)
    if meth == "ebr":
        f = out[col["ebc_cf"]]
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            f = (out[col["LE_corr"]] / out[col["LE"]]).replace([np.inf, -np.inf], np.nan)
    return f.reindex(pd.DatetimeIndex(daily.index)).set_axis(daily.index)


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

