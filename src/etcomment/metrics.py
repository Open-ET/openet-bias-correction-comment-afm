"""Error metrics used in the comment.

Conventions follow both papers being compared:

* bias / MBE = mean(P - O)  (P = predicted/OpenET, O = observed)
* R2_cod = 1 - SSE/SST      (coefficient of determination; can be negative;
  this is the "R^2" used by Golkar and de Oliveira, e.g. BSk = -2.218)
* r2_pearson = corr(P, O)^2 (used by Volk et al. 2024)

``mse_decomposition`` implements Eq. (3) of Golkar and de Oliveira (2026):
MSE = Bias^2 + s_o^2 + s_p^2 - 2 r s_o s_p  (population moments), which is an
identity: MSE = Bias^2 + Var(P - O).
"""
from __future__ import annotations

import numpy as np


def _clean(p, o):
    p = np.asarray(p, float)
    o = np.asarray(o, float)
    ok = np.isfinite(p) & np.isfinite(o)
    return p[ok], o[ok]


def mbe(p, o) -> float:
    p, o = _clean(p, o)
    return float(np.mean(p - o))


def mae(p, o) -> float:
    p, o = _clean(p, o)
    return float(np.mean(np.abs(p - o)))


def rmse(p, o) -> float:
    p, o = _clean(p, o)
    return float(np.sqrt(np.mean((p - o) ** 2)))


def r2_cod(p, o) -> float:
    p, o = _clean(p, o)
    sse = np.sum((p - o) ** 2)
    sst = np.sum((o - o.mean()) ** 2)
    return float(1.0 - sse / sst) if sst > 0 else np.nan


def r2_pearson(p, o) -> float:
    p, o = _clean(p, o)
    if p.size < 3 or np.std(p) == 0 or np.std(o) == 0:
        return np.nan
    return float(np.corrcoef(p, o)[0, 1] ** 2)


def summary(p, o) -> dict:
    p_, o_ = _clean(p, o)
    return {
        "n": int(p_.size),
        "mean_obs": float(np.mean(o_)) if p_.size else np.nan,
        "mean_pred": float(np.mean(p_)) if p_.size else np.nan,
        "mbe": mbe(p, o),
        "mae": mae(p, o),
        "rmse": rmse(p, o),
        "r2_cod": r2_cod(p, o),
        "r2_pearson": r2_pearson(p, o),
    }


def mse_decomposition(p, o) -> dict:
    """Bias^2 and variance terms of the MSE (Golkar & de Oliveira Eq. 3)."""
    p, o = _clean(p, o)
    bias = float(np.mean(p - o))
    so = float(np.std(o))
    sp = float(np.std(p))
    rho = float(np.corrcoef(p, o)[0, 1]) if (so > 0 and sp > 0) else np.nan
    var_term = so**2 + sp**2 - 2 * rho * so * sp
    return {
        "mse": float(np.mean((p - o) ** 2)),
        "bias2": bias**2,
        "variance_term": var_term,
        "spread_mismatch": (so - sp) ** 2,
        "correlation_deficit": 2 * so * sp * (1 - rho),
        "bias": bias,
        "sd_obs": so,
        "sd_pred": sp,
        "rho": rho,
    }


def improvement_attribution(p_base, p_new, o) -> dict:
    """Share of the MSE reduction from Bias^2 vs the variance term.

    Mirrors the attribution in Golkar & de Oliveira (2026, Section 2.6):
    bias share = (Bias_base^2 - Bias_new^2) / (MSE_base - MSE_new).
    """
    b = mse_decomposition(p_base, o)
    n = mse_decomposition(p_new, o)
    d_mse = b["mse"] - n["mse"]
    d_b2 = b["bias2"] - n["bias2"]
    d_var = b["variance_term"] - n["variance_term"]
    return {
        "mse_base": b["mse"],
        "mse_new": n["mse"],
        "rmse_improvement_pct": 100 * (1 - np.sqrt(n["mse"]) / np.sqrt(b["mse"])),
        "bias_share_pct": 100 * d_b2 / d_mse if d_mse != 0 else np.nan,
        "variance_share_pct": 100 * d_var / d_mse if d_mse != 0 else np.nan,
    }


def zero_fill_bias_shift(f0: float, mean_true_on_filled: float) -> float:
    """Change in MBE when a fraction ``f0`` of observations is replaced by 0.

    If the true observed ET on the affected days averages ``mean_true_on_filled``,
    replacing those observations with zero lowers mean(O) by
    ``f0 * mean_true_on_filled`` and therefore raises MBE = mean(P - O) by the
    same amount, independent of the model being evaluated.
    """
    return float(f0 * mean_true_on_filled)
