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
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAMBDA_CONST = 2.45  # MJ kg-1, value used by Golkar & de Oliveira (2026)


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
