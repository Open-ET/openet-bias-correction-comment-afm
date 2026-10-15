"""Train/test designs for site-based ET correction models.

Compares a record-level random split (what Golkar & de Oliveira's Table 5
labels "random split") with site-grouped designs (grouped hold-out,
leave-one-site-out, leave-one-cluster-out), optionally after injecting
zero-filled targets and median-imputed predictors on days without tower data.
The zone medians are what identifies a zero-filled day; real days that carry
the same values (e.g. missing meteorology imputed with the same medians) dilute
that signature, and :func:`matches_flagged` finds them.

Leave-one-cluster-out mirrors GO26's LOCO-CV (Section 3.4), which held out
colocated towers together (US-Ne1/Ne2/Ne3, US-MN1/MN2/MN3, US-Bi1/Bi2,
US-UiC/UiG): under leave-one-site-out a held-out tower's near-identical
neighbour stays in training.  Clusters come from :func:`distance_clusters`,
single linkage on great-circle distance, so a chain of neighbours forms one
cluster even when its end members are farther apart than the threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, train_test_split

from . import metrics as mt

EARTH_RADIUS_KM = 6371.0088  # IUGG mean radius


@dataclass
class Result:
    design: str
    n_test: int
    rmse_base: float
    rmse_model: float
    mbe_base: float
    mbe_model: float
    improvement_pct: float
    bias_share_pct: float
    n_flagged: int = 0
    flagged_mean_target: float = np.nan
    flagged_mean_base: float = np.nan
    flagged_mean_model: float = np.nan


def _score(design, y, base, pred, flag=None) -> Result:
    att = mt.improvement_attribution(base, pred, y)
    r = Result(design, int(len(y)), mt.rmse(base, y), mt.rmse(pred, y), mt.mbe(base, y),
               mt.mbe(pred, y), att["rmse_improvement_pct"], att["bias_share_pct"])
    if flag is not None and np.any(flag):
        r.n_flagged = int(np.sum(flag))
        r.flagged_mean_target = float(np.mean(y[flag]))
        r.flagged_mean_base = float(np.mean(base[flag]))
        r.flagged_mean_model = float(np.mean(pred[flag]))
    return r


def default_model(seed: int = 0):
    from xgboost import XGBRegressor

    return XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, random_state=seed, n_jobs=4)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km on a sphere of radius ``EARTH_RADIUS_KM`` (broadcasts)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(np.subtract(lon2, lon1))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def distance_clusters(sites, lat, lon, max_km: float = 10.0) -> pd.Series:
    """Cluster label per site: connected components of stations within ``max_km`` of each other.

    Two stations share a cluster when a chain of stations links them with every
    step <= ``max_km`` (single linkage).  A label is the sorted member IDs joined
    by "+" (a lone station keeps its ID), so labels do not depend on input order.
    """
    idx = pd.Index(sites, name="site")
    if idx.has_duplicates:
        raise ValueError("site IDs must be unique")
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError("every site needs a latitude and longitude")
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    _, comp = connected_components(d <= max_km, directed=False)
    label = pd.Series(idx.astype(str), index=comp).groupby(level=0).agg(lambda s: "+".join(sorted(s)))
    return pd.Series(label.loc[comp].to_numpy(), index=idx, name="cluster")


def _held_out_predict(X, y, groups, model_factory, seed) -> np.ndarray:
    """Predict every record with a model trained without the record's group."""
    pred = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = model_factory(seed).fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
    return pred


def run_designs(df: pd.DataFrame, features: list[str], target: str, baseline: str, group: str,
                model_factory: Callable = default_model, seed: int = 0,
                test_size: float = 0.2, flag: str | None = None, cluster: str | None = None,
                return_predictions: bool = False):
    """Score the baseline (e.g. OpenET ensemble) and a fitted model on each design.

    Designs: random record split, site-grouped hold-out and leave-one-site-out
    (sites = ``group``), plus leave-one-cluster-out when ``cluster`` names a
    column of cluster labels (each site in one cluster).  ``flag`` names a
    boolean column (e.g. zero-filled records); the mean target, baseline and
    prediction on flagged test records are reported alongside.  ``n_folds`` is
    the number of models fitted per design; ``n_groups`` and ``n_clusters``
    count the distinct sites and clusters in ``df``.  With
    ``return_predictions`` the held-out predictions of the leave-one-out
    designs (one column per design, index of ``df``) are returned as well.
    """
    if cluster is not None and (df[cluster].isna().any() or df.groupby(group)[cluster].nunique().max() > 1):
        raise ValueError(f"every {group} must belong to exactly one {cluster}")
    X, y, b, g = df[features], df[target].to_numpy(), df[baseline].to_numpy(), df[group].to_numpy()
    fl = df[flag].to_numpy(bool) if flag else np.zeros(len(df), bool)
    n_groups = int(pd.Series(g).nunique())
    out, loo = [], {}
    # 1) record-level random split
    tr, te = train_test_split(np.arange(len(df)), test_size=test_size, random_state=seed)
    m = model_factory(seed).fit(X.iloc[tr], y[tr])
    out.append(_score("random record split", y[te], b[te], m.predict(X.iloc[te]), fl[te]))
    # 2) grouped hold-out (whole sites)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(X, y, g))
    m = model_factory(seed).fit(X.iloc[tr], y[tr])
    out.append(_score("site-grouped hold-out", y[te], b[te], m.predict(X.iloc[te]), fl[te]))
    # 3) leave-one-site-out
    loo["leave-one-site-out"] = _held_out_predict(X, y, g, model_factory, seed)
    out.append(_score("leave-one-site-out", y, b, loo["leave-one-site-out"], fl))
    n_folds = [1, 1, n_groups]
    n_clusters = np.nan
    # 4) leave-one-cluster-out: colocated sites leave training together
    if cluster is not None:
        c = df[cluster].to_numpy()
        n_clusters = int(pd.Series(c).nunique())
        loo["leave-one-cluster-out"] = _held_out_predict(X, y, c, model_factory, seed)
        out.append(_score("leave-one-cluster-out", y, b, loo["leave-one-cluster-out"], fl))
        n_folds.append(n_clusters)
    res = pd.DataFrame([r.__dict__ for r in out])
    res["n_folds"] = n_folds
    res["n_groups"] = n_groups
    res["n_clusters"] = n_clusters
    return (res, pd.DataFrame(loo, index=df.index)) if return_predictions else res


def inject_zero_fill(df: pd.DataFrame, missing_col: str, target: str, met_cols: list[str],
                     group_col: str) -> pd.DataFrame:
    """Set targets to 0 and met features to their group-wide medians on missing days.

    Mirrors the data handling we infer for GO26: days without tower data get a
    zero target (fillna(0)) and climate-zone median predictors.  The medians are
    taken over the group's unflagged records, skipping NaN (a group without any
    value gives NaN).
    """
    d = df.copy()
    miss = d[missing_col].astype(bool)
    d.loc[miss, target] = 0.0
    for c in met_cols:
        med = d.loc[~miss].groupby(group_col)[c].median()
        d.loc[miss, c] = d.loc[miss, group_col].map(med).to_numpy()
    return d


def matches_flagged(df: pd.DataFrame, flag: str, cols: list[str]) -> pd.Series:
    """Unflagged records whose ``cols`` all equal those of some flagged record (NaN equals NaN).

    After :func:`inject_zero_fill` these are real days a model cannot tell from
    zero-filled days by their meteorology, e.g. days whose missing values were
    imputed with the same zone medians.  NaN counts as equal because a tree
    model routes all NaN the same way.
    """
    fl = df[flag].to_numpy(bool)
    v = df[cols].astype(float)
    hit = v.merge(v[fl].drop_duplicates(), how="left", on=cols, indicator=True)["_merge"].eq("both")
    return pd.Series(hit.to_numpy() & ~fl, index=df.index, name="matches_flagged")
