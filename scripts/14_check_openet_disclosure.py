"""Check jointly that the deposited OpenET statistics reveal no single site-month and no combination of fewer
than 6 site-months (the OpenET rule of this repository, README.md).

scripts/10 (withhold_openet) and scripts/12 (small_supports and projection checks) withhold OpenET values within
their own tables.  This script checks everything they publish together, with the observed data (public) treated
as known.  Run it before the deposit is built: scripts/13 stops unless the record written here passed and still
matches the files it read.

Unknowns, one family at a time: the monthly OpenET of the ensemble and of each of the six models, gridMET ETo
and precipitation (towers in one gridMET cell share one unknown) and the daily ensemble.  The unit of the rule is
the site-month (the daily values of a month belong to one).  Rows, one per published number, as weights on the
unknowns:
  bias10   results/go26_reconstructed_openet_bias.csv, published rows: mean over the row's site-months of OpenET /
           days in month; the site-months are rebuilt from public data only (go26_reconstructed_daily.csv,
           config/go26_sites.csv and the months without OpenET in openet_api_go26_sites.csv, months_filled) and
           must give the published n_months and observed_mm_d
  sites    results/openet_api_go26_sites.csv: study-period means (months without OpenET take the 2016-2025 mean
           of the calendar month, as scripts/12)
  json     results/openet_api_go26_summary.json, cross-tower statistics: the exactly linear ones, and gradients of
           the others at the true values (kind 'grad')
  bench_*  results/openet_api_benchmark_summary.csv: sums, sums per day (MBE per day), gradients of RMSE and R2
  fig, shared  results/openet_api_go26_figure_check.csv and openet_api_shared_daily.csv (daily ensemble)
Every linear row with a published value is applied to the raw extracts and must give that value (within 6e-5).
Per family, for the linear rows alone and with the gradients: the projection diagonal (no value determined;
etcomment.openet_compare.projection_diagonal) and the exact search of etcomment.disclosure (no combination of
fewer than 6 site-months).  Positive controls: rows known to leak are added back (the bias rows scripts/10
withholds by its 'support' rule, or all its withheld rows; the per-tower P_zf of the old scripts/12) and must
be found.

Inputs: data/raw/openet_api/openet_monthly.csv, openet_daily.csv and derived/benchmark_station_months.csv
(scripts/02 and 12; OpenET Terms of Service, not deposited), data/raw/flux_ET_dataset (scripts/01), config/ and
results/.  Writes data/raw/openet_api/derived/disclosure_check.json (md5 of every file read, the results per
family and control; no OpenET values).  About 10 minutes.

Usage: python scripts/14_check_openet_disclosure.py [--families et_ensemble daily ...] [--no-controls]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from etcomment import disclosure as dc
from etcomment import openet_compare as oc

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RAW = ROOT / "data/raw/openet_api"
RECORD = "data/raw/openet_api/derived/disclosure_check.json"  # read by scripts/13
MODELS = ["ensemble", "disalexi", "eemetric", "geesebal", "ptjpl", "sims", "ssebop"]
FAMILIES = [f"et_{m}" for m in MODELS] + ["eto", "pr", "daily"]
CLIM = (pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-01"))  # months of the climatology fill (scripts/12)
MIN_VALID_DAYS = 20  # scripts/10 --min-valid-days
TOL = 6e-5  # published values have 4 decimals
CODE = ["scripts/14_check_openet_disclosure.py", "src/etcomment/disclosure.py"]  # in the record: a change is stale


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"s{name[:2]}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


class Row:
    """One published number: weights on the unknowns, its kind ('lin' or 'grad') and its published value."""
    __slots__ = ("name", "kind", "w", "published")

    def __init__(self, name, kind, w: dict, published=np.nan):
        self.name, self.kind, self.w, self.published = name, kind, w, published

    def apply(self, values: dict) -> float:
        return float(sum(c * values[k] for k, c in self.w.items()))


class Inputs:
    """The files the rows are built from (every one recorded with its md5), the gridMET cells and the true values."""

    def __init__(self):
        self.files: dict[str, str] = {}
        self.problems: list[str] = []
        self.monthly = self.csv(RAW / "openet_monthly.csv", parse_dates=["date"])
        self.daily = self.csv(RAW / "openet_daily.csv", parse_dates=["date"])
        self.bsm = self.csv(RAW / "derived/benchmark_station_months.csv")
        self.go = self.csv(ROOT / "config/go26_sites.csv")
        self.t1 = dict(zip(self.go.site, self.go.mean_obs_et_mm_d))
        self.ay = self.csv(RES / "ameriflux_le_availability_by_year.csv")
        self.sites = self.csv(RES / "openet_api_go26_sites.csv")
        self.js = json.loads(self.text(RES / "openet_api_go26_summary.json"))
        self.bench = self.csv(RES / "openet_api_benchmark_summary.csv")
        self.shared = self.csv(RES / "openet_api_shared_daily.csv")
        self.fig = self.csv(RES / "openet_api_go26_figure_check.csv")
        self.bias = self.csv(RES / "go26_reconstructed_openet_bias.csv", comment="#")
        self.recon = self.csv(RES / "go26_reconstructed_daily.csv", comment="#", parse_dates=["date"])
        self.digitized = {s: self.csv(RES / f"digitized_{s}.csv", parse_dates=["date"]).set_index("date")
                          for s in self.fig["site"]}
        # Earth Engine ETo check (scripts/02 --verify-eto-ee, optional): one row of statistics of API minus Earth
        # Engine ETo (OpenET's public collection) over every site-month; its only linear statistic spans them all
        ee = RES / "openet_api_eto_ee_check.csv"
        if ee.exists() and not (len(t := self.csv(ee)) == 1 and t["n_pairs"].iloc[0] >= 100 * dc.MIN_SITE_MONTHS):
            self.problems.append("openet_api_eto_ee_check.csv is no longer one row over all site-months: add its "
                                 "statistics to this check")
        for f in CODE:
            self.files[f] = md5(ROOT / f)
        # gridMET cells: towers with an identical series (to 1e-4, the precision of the extract) share one unknown
        self.cell = {}
        for c in ["eto", "pr"] + [f"et_{m}" for m in MODELS]:
            p = self.monthly.pivot(index="date", columns="site", values=c)
            seen: dict = {}
            for s in p.columns:
                self.cell[(c, s)] = seen.setdefault(tuple(np.round(p[s].fillna(-9999.0).to_numpy(), 4)), s)
        self.truth = {}
        for c in ["eto", "pr"] + [f"et_{m}" for m in MODELS]:
            for s, d, v in zip(self.monthly.site, self.monthly.date, self.monthly[c]):
                if np.isfinite(v):
                    self.truth[(c, self.cell[(c, s)], d)] = v
        for s, d, v in zip(self.daily.site, self.daily.date, self.daily.et_ensemble):
            if np.isfinite(v):
                self.truth[("daily", s, d)] = v

    def _note(self, path: Path) -> None:
        self.files[path.relative_to(ROOT).as_posix()] = md5(path)

    def csv(self, path: Path, **kw) -> pd.DataFrame:
        self._note(path)
        return pd.read_csv(path, **kw)

    def text(self, path: Path) -> str:
        self._note(path)
        return path.read_text()

    def key(self, c, s, m):
        return (c, self.cell.get((c, s), s), pd.Timestamp(m))


# -- (A) bias table of scripts/10 ----------------------------------------------------------------------------------
def bias_rows(I: Inputs, s10) -> tuple[list[Row], list[Row]]:
    """Published and withheld rows of go26_reconstructed_openet_bias.csv, their site-months rebuilt from public data
    (n_months and observed_mm_d must match the table)."""
    variants = dict(s10.VALID_VARIANTS)
    d = I.recon[["site", "date", "et_hh_sum_zero_filled", *variants.values()]].copy()
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    common = d[list(variants.values())].notna().all(axis=1)
    g = d.assign(common=common).groupby(["site", "month"])
    m = pd.DataFrame({"days": g.size(), "n_common": g["common"].sum(),
                      "zf": g["et_hh_sum_zero_filled"].sum(min_count=1)})
    for v, c in variants.items():
        m[v] = d[c].where(common).groupby([d["site"], d["month"]]).mean()
    m = m.reset_index()
    m["dim"] = m["month"].dt.days_in_month
    nofill = {(s, pd.Timestamp(f"{x}-01")) for s, f in zip(I.sites["site"], I.sites["months_filled"].fillna(""))
              for x in str(f).split(";") if x}
    m["has_openet"] = [(s, mo) not in nofill for s, mo in zip(m["site"], m["month"])]
    zone = m["site"].map(dict(zip(I.go.site, I.go.koppen_go26)))
    season = m["month"].dt.month.map(s10.SEASON)
    published, withheld = [], []
    for r in I.bias.itertuples(index=False):
        sel = m["has_openet"] & (m["days"] == m["dim"])
        if r.koppen_go26 != "all":
            sel &= zone == r.koppen_go26
        if r.season != "all":
            sel &= season == r.season
        if r.month_set == "common_valid_months":
            sel &= m["n_common"] >= MIN_VALID_DAYS
        s = m[sel]
        obs = (s["zf"] / s["dim"] if r.variant == "zero_filled_table1" else s[r.variant]).mean()
        name = ("bias10", r.koppen_go26, r.season, r.month_set, r.variant)
        if len(s) != r.n_months or not abs(obs - r.observed_mm_d) <= TOL:
            I.problems.append(f"{name}: rebuilt {len(s)} site-months, observed {obs:.5f}; the table has "
                              f"{r.n_months}, {r.observed_mm_d}")
        w: dict = {}
        for site, mo, dim in zip(s["site"], s["month"], s["dim"]):
            k = I.key("et_ensemble", site, mo)
            w[k] = w.get(k, 0.0) + 1.0 / (len(s) * dim)
        withheld_row = bool(r.openet_withheld)
        row = Row(name + ((r.withheld_rule,) if withheld_row else ()), "lin", w,
                  np.nan if withheld_row else r.openet_mm_d)
        (withheld if withheld_row else published).append(row)
    return published, withheld


# -- (B) sites table of scripts/12 ---------------------------------------------------------------------------------
def fill_weights(I: Inputs, c: str, site: str, start: str, end: str) -> dict:
    """Weights on the monthly totals (mm) giving the period total, months without a value filled with the mean of
    the same calendar month over CLIM."""
    m = I.monthly[I.monthly.site == site].set_index("date")[c]
    have = set(m.index[m.notna()])
    clim = sorted(d for d in have if CLIM[0] <= d <= CLIM[1])
    w: dict = {}
    for d in pd.date_range(f"{start}-01", f"{end}-01", freq="MS"):
        same = [d] if d in have else [x for x in clim if x.month == d.month]
        for x in same:
            k = I.key(c, site, x)
            w[k] = w.get(k, 0.0) + 1.0 / len(same)
    return w


def period_days(start: str, end: str) -> int:
    return int(np.asarray(pd.date_range(f"{start}-01", f"{end}-01", freq="MS").days_in_month).sum())


def sites_rows(I: Inputs) -> list[Row]:
    out = []
    cols = {"oe_mm_d": ("et_ensemble", 1.0), "oe_annual_mm": ("et_ensemble", 365.25), "eto_annual_mm": ("eto", 365.25),
            "pr_annual_mm": ("pr", 365.25), **{f"{m}_mm_d": (f"et_{m}", 1.0) for m in MODELS[1:]}}
    for r in I.sites.itertuples(index=False):
        n = period_days(r.period_start, r.period_end)
        for col, (c, scale) in cols.items():
            v = getattr(r, col)
            if np.isfinite(v):
                w = fill_weights(I, c, r.site, r.period_start, r.period_end)
                out.append(Row(("sites", r.site, col), "lin", {k: x * scale / n for k, x in w.items()}, v))
    return out


# -- (C) cross-tower statistics of the summary JSON ----------------------------------------------------------------
def pzf_weights(I: Inputs, site: str, start: str, end: str) -> dict:
    """P_zf (scripts/12, no longer published): filled monthly totals x the year's LE availability / period days."""
    y = I.ay[I.ay.site == site].set_index("year")
    a = (y["days_available"] / y["days_in_period"]).clip(upper=1.0)
    n = period_days(start, end)
    m = I.monthly[I.monthly.site == site].set_index("date")["et_ensemble"]
    have = set(m.index[m.notna()])
    clim = sorted(d for d in have if CLIM[0] <= d <= CLIM[1])
    w: dict = {}
    for d in pd.date_range(f"{start}-01", f"{end}-01", freq="MS"):
        same = [d] if d in have else [x for x in clim if x.month == d.month]
        for x in same:
            k = I.key("et_ensemble", site, x)
            w[k] = w.get(k, 0.0) + float(a.loc[d.year]) / len(same) / n
    return w


def _stats_vs(t1, p) -> dict:
    t1, p = np.asarray(t1, float), np.asarray(p, float)
    b = (p * t1).sum() / (p * p).sum()
    return {"pearson_r": float(np.corrcoef(p, t1)[0, 1]), "slope_through_origin": b,
            "rmse_fit": float(np.sqrt(np.mean((t1 - b * p) ** 2))), "mean_diff_t1_minus": float(np.mean(t1 - p)),
            "rmse_diff": float(np.sqrt(np.mean((t1 - p) ** 2)))}


def _ratio_stats(ratio, avail) -> dict:
    ratio, avail = np.asarray(ratio, float), np.asarray(avail, float)
    b = (avail * ratio).sum() / (avail * avail).sum()
    return {"pearson_r": float(np.corrcoef(avail, ratio)[0, 1]), "slope": b,
            "rmse": float(np.sqrt(np.mean((ratio - b * avail) ** 2)))}


def json_rows(I: Inputs) -> list[Row]:
    """go26_towers of the summary JSON: exactly linear statistics, gradients of the others (central differences)."""
    t = I.sites.set_index("site")
    sites = list(t.index)
    oe_w = {s: {k: x / period_days(t.loc[s, "period_start"], t.loc[s, "period_end"]) for k, x in
                fill_weights(I, "et_ensemble", s, t.loc[s, "period_start"], t.loc[s, "period_end"]).items()}
            for s in sites}
    pz_w = {s: pzf_weights(I, s, t.loc[s, "period_start"], t.loc[s, "period_end"]) for s in sites}
    oe = {s: sum(c * I.truth[k] for k, c in w.items()) for s, w in oe_w.items()}
    pz = {s: sum(c * I.truth[k] for k, c in w.items()) for s, w in pz_w.items()}
    avail = t["le_avail_period"]

    def stats(sub, oe_, pz_):
        t1 = [I.t1[s] for s in sub]
        ratio = [I.t1[s] / oe_[s] for s in sub]
        st = {"mean_oe_mm_d": float(np.mean([oe_[s] for s in sub])), "mean_ratio_t1_oe": float(np.mean(ratio))}
        st.update({f"ratio_vs_availability.{k}": v for k, v in _ratio_stats(ratio, [avail[s] for s in sub]).items()})
        st.update({f"t1_vs_oe.{k}": v for k, v in _stats_vs(t1, [oe_[s] for s in sub]).items()})
        st.update({f"t1_vs_p_zf.{k}": v for k, v in _stats_vs(t1, [pz_[s] for s in sub]).items()})
        return st

    out = []
    linear = ("mean_oe_mm_d", "t1_vs_p_zf.mean_diff_t1_minus", "t1_vs_oe.mean_diff_t1_minus")
    for name, sub in {"all": sites, "not_in_benchmark": [s for s in sites if not bool(t.loc[s, "in_benchmark"])]}.items():
        base = stats(sub, oe, pz)
        pub = I.js["go26_towers"][name]
        for k, v in base.items():
            path = k.split(".")
            pv = pub[path[0]] if len(path) == 1 else pub[path[0]][path[1]]
            if abs(pv - v) > TOL:
                I.problems.append(f"json go26_towers.{name}.{k}: published {pv}, rebuilt {v:.6f}")
        for k in base:
            g: dict = {}
            for s in sub:
                for which, wts, val in (("oe", oe_w, oe), ("pz", pz_w, pz)):
                    h = 1e-6 * max(1.0, abs(val[s]))
                    up, dn = dict(val), dict(val)
                    up[s] += h
                    dn[s] -= h
                    pair = ((up, pz), (dn, pz)) if which == "oe" else ((oe, up), (oe, dn))
                    der = (stats(sub, *pair[0])[k] - stats(sub, *pair[1])[k]) / (2 * h)
                    if abs(der) > 1e-12:
                        for kk, c in wts[s].items():
                            g[kk] = g.get(kk, 0.0) + der * c
            if g:
                out.append(Row(("json", name, k), "lin" if k in linear else "grad", g))
        # slope_through_origin and rmse_diff of T1 against P_zf make sum(T1 x P_zf) exactly known
        out.append(Row(("json", name, "sum_t1_x_pzf"), "lin", {k: I.t1[s] * c for s in sub for k, c in pz_w[s].items()}))
    return out


# -- (D) benchmark summary of scripts/12 ---------------------------------------------------------------------------
STRATA = {"all": [], "koppen": ["koppen"], "koppen_group": ["koppen_group"], "season": ["season"],
          "season_x_group": ["season", "koppen_group"], "site": ["site"]}


def bench_rows(I: Inputs, model: str) -> list[Row]:
    b = I.bsm.copy()
    b["month"] = pd.to_datetime(b["month"])
    b["pred"] = b["ec_closed_et_mm"] + b[f"{model}_minus_ec_mm"]
    b = b[np.isfinite(b["pred"])]
    pub = I.bench[I.bench.model == model].set_index(["stratum_type", "stratum"])
    c = f"et_{model}"
    out = []
    for st, cols in STRATA.items():
        groups = [("all", b)] if not cols else [(", ".join(map(str, k)) if isinstance(k, tuple) else str(k), g)
                                                for k, g in b.groupby(cols, sort=True)]
        for lab, g in groups:
            if (st, lab) not in pub.index:
                I.problems.append(f"bench {model}: stratum {(st, lab)} not in openet_api_benchmark_summary.csv")
                continue
            r = pub.loc[(st, lab)]
            if int(r["n_months"]) != len(g) or abs(r["mean_ec_mm_month"] - g["ec_closed_et_mm"].mean()) > 6e-4:
                I.problems.append(f"bench {model} {(st, lab)}: {r['n_months']} months published, {len(g)} rebuilt")
            keys = [I.key(c, s, m) for s, m in zip(g["site"], g["month"])]
            if np.isfinite(r["mean_openet_mm_month"]) or np.isfinite(r["mbe_mm_month"]) or np.isfinite(r["mbe_pct"]):
                out.append(Row(("bench", model, st, lab, "sum"), "lin", {k: 1.0 / len(g) for k in keys},
                               r["mean_openet_mm_month"]))
            if np.isfinite(r["mbe_mm_d"]):
                obs = float(np.mean(g["ec_closed_et_mm"] / g["n_days"]))
                out.append(Row(("bench", model, st, lab, "sum_per_day"), "lin",
                               {k: 1.0 / (len(g) * nd) for k, nd in zip(keys, g["n_days"])}, r["mbe_mm_d"] + obs))
            if np.isfinite(r["rmse_mm_month"]) or np.isfinite(r["r2"]):
                out.append(Row(("bench", model, st, lab, "rmse"), "grad",
                               dict(zip(keys, (g["pred"] - g["ec_closed_et_mm"]).to_numpy()))))
    return out


# -- (E) daily ensemble: figure check and shared towers of scripts/12 ----------------------------------------------
def figure_rows(I: Inputs) -> list[Row]:
    out = []
    for r in I.fig.itertuples(index=False):
        api = I.daily[I.daily.site == r.site].set_index("date")["et_ensemble"]
        j = pd.concat([I.digitized[r.site]["openet"].rename("a"), api.rename("b")], axis=1, join="inner").dropna()
        if len(j) != r.n:
            I.problems.append(f"figure check {r.site}: {r.n} days published, {len(j)} rebuilt")
        keys = [("daily", r.site, d) for d in j.index]
        a, b = j["a"].to_numpy(), j["b"].to_numpy()
        out.append(Row(("fig", r.site, "sum"), "lin", {k: 1.0 / len(j) for k in keys}, r.mean_api_openet_mm_d))
        # scale (sum ab / sum b^2) and rmse together give sum ab exactly: linear
        out.append(Row(("fig", r.site, "sum_ab"), "lin", dict(zip(keys, a))))
        out.append(Row(("fig", r.site, "sum_b2_grad"), "grad", dict(zip(keys, 2 * b))))
        saa, sbb = ((a - a.mean()) ** 2).sum(), ((b - b.mean()) ** 2).sum()
        rr = ((a - a.mean()) * (b - b.mean())).sum() / np.sqrt(saa * sbb)
        out.append(Row(("fig", r.site, "r_grad"), "grad",
                       dict(zip(keys, (a - a.mean()) - rr * np.sqrt(saa / sbb) * (b - b.mean())))))
        rebuilt = {"mean_api_openet_mm_d": b.mean(), "scale_go26_over_api": (a * b).sum() / (b * b).sum(),
                   "pearson_r": rr, "rmse_mm_d": np.sqrt(np.mean((a - b) ** 2))}
        for k, v in rebuilt.items():
            if abs(getattr(r, k) - v) > TOL:
                I.problems.append(f"figure check {r.site} {k}: published {getattr(r, k)}, rebuilt {v:.6f}")
    return out


def shared_rows(I: Inputs, s12) -> list[Row]:
    from etcomment import benchmark as bm
    frames = s12.load_frames(bm.find_root(ROOT / "data/raw/flux_ET_dataset"), s12.SHARED)
    parts = []
    for s in s12.SHARED:
        od = I.daily[I.daily.site == s].set_index("date")["et_ensemble"]
        if s in frames and not od.empty:
            parts.append(oc.daily_pairs(frames[s], od, s12.START).assign(site=s))
    pairs = pd.concat(parts)
    pairs["season"] = oc.season(pairs.index.month)
    out = []
    for r in I.shared.itertuples(index=False):
        h = pairs if r.site == "All 7 towers" else pairs[pairs.site == r.site]
        h = h if r.season == "all" else h[h.season == r.season]
        col = "et_corr" if r.reference == "closed" else "et"
        k = h[np.isfinite(h["openet"]) & np.isfinite(h[col])]
        if len(k) != r.n_days:
            I.problems.append(f"shared {r.site} {r.season} {r.reference}: {r.n_days} days published, {len(k)} rebuilt")
        if not len(k):
            continue
        keys = [("daily", s, d) for s, d in zip(k["site"], k.index)]
        if np.isfinite(r.mean_openet_mm_d) or np.isfinite(r.mbe_mm_d) or np.isfinite(r.mbe_pct):
            out.append(Row(("shared", r.site, r.season, r.reference, "sum"), "lin", {x: 1.0 / len(k) for x in keys},
                           r.mean_openet_mm_d))
        if np.isfinite(r.rmse_mm_d) or np.isfinite(r.r2):
            out.append(Row(("shared", r.site, r.season, r.reference, "rmse"), "grad",
                           dict(zip(keys, (k["openet"] - k[col]).to_numpy()))))
    return out


# -- assembly and checks -------------------------------------------------------------------------------------------
def validate(I: Inputs, rows: list[Row]) -> int:
    """Apply every linear row with a published value to the true values; mismatches go to I.problems."""
    bad = 0
    for r in rows:
        if r.kind == "lin" and np.isfinite(r.published):
            v = r.apply(I.truth)
            if abs(v - r.published) > TOL:
                bad += 1
                I.problems.append(f"{r.name}: published {r.published}, rebuilt {v:.6f}")
    return bad


def matrix(rows: list[Row], family: str, kinds) -> tuple[np.ndarray, list]:
    """Rows of ``kinds`` restricted to the unknowns of ``family``, and those unknowns (sorted)."""
    sel = [{k: c for k, c in r.w.items() if k[0] == family} for r in rows if r.kind in kinds]
    sel = [w for w in sel if w]
    cols = sorted({k for w in sel for k in w}, key=str)
    idx = {k: j for j, k in enumerate(cols)}
    a = np.zeros((len(sel), len(cols)))
    for i, w in enumerate(sel):
        for k, c in w.items():
            a[i, idx[k]] += c
    return a, cols


def site_month(k) -> tuple:
    return (k[1], f"{pd.Timestamp(k[2]):%Y-%m}")


def check_family(rows: list[Row], family: str, kinds) -> dict:
    """Projection diagonal and exact small-support search of the rows of ``kinds`` on ``family``'s unknowns."""
    a, cols = matrix(rows, family, kinds)
    if not a.size:
        return {"rows": 0, "passed": True}
    t0 = time.time()
    proj = oc.projection_diagonal(pd.DataFrame(a, columns=pd.Index(cols, tupleize_cols=False))).to_numpy()
    groups = [site_month(k) for k in cols]
    res = {"rows": int(a.shape[0]), "unknowns": len(cols), "site_months": len(set(groups)),
           "max_projection": round(float(proj.max()), 4), "at": " ".join(site_month(cols[int(proj.argmax())])),
           "determined": [" ".join(site_month(cols[j])) for j in np.flatnonzero(proj > 1 - oc.IDENT_TOL)]}
    try:
        s = dc.small_support_search(a, groups)
        res.update(disjoint_bases=s["m"], sets_tried=s["tried"],
                   small_sets=[[" ".join(g) for g in c] for c in s["confirmed"]])
    except RuntimeError as err:
        res["error"] = str(err)
    res["seconds"] = round(time.time() - t0, 1)
    res["passed"] = not res["determined"] and not res.get("small_sets") and "error" not in res
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--families", nargs="+", choices=FAMILIES, help="default: all (needed for scripts/13)")
    ap.add_argument("--no-controls", action="store_true", help="skip the positive controls (not for scripts/13)")
    args = ap.parse_args(argv)
    families = args.families or FAMILIES
    s10, s12 = _load("10_reconstruct_go26_observed_et"), _load("12_openet_api_analysis")
    I = Inputs()
    published, withheld = bias_rows(I, s10)
    rows = published + sites_rows(I) + json_rows(I) + [r for m in MODELS for r in bench_rows(I, m)]
    if "daily" in families:
        rows += figure_rows(I) + shared_rows(I, s12)
    n_bad = validate(I, rows)
    print(f"{len(rows)} rows ({sum(r.kind == 'lin' for r in rows)} linear; {len(published)} of "
          f"{len(published) + len(withheld)} bias-table rows published); validation mismatches: {n_bad}; "
          f"other problems: {len(I.problems) - n_bad}")
    for p in I.problems[:20]:
        print("  PROBLEM", p)
    results: dict = {}
    for fam in families:
        results[fam] = {}
        for label, kinds in (("linear", ("lin",)), ("linear+gradients", ("lin", "grad"))):
            r = results[fam][label] = check_family(rows, fam, kinds)
            if r["rows"]:
                print(f"{fam} {label}: {r['rows']} rows, {r['site_months']} site-months; max projection "
                      f"{r['max_projection']} ({r['at']}), determined {len(r['determined'])}; disjoint bases "
                      f"{r.get('disjoint_bases')}, sets tried {r.get('sets_tried')}, small sets "
                      f"{len(r.get('small_sets', []))}{'; ' + r['error'] if 'error' in r else ''} ({r['seconds']} s)",
                      flush=True)
                for s in (r["determined"] + [", ".join(c) for c in r.get("small_sets", [])])[:10]:
                    print("    REVEALED", s)
    controls: dict = {}
    if not args.no_controls:  # rows known to leak, added back: the search must find them
        leak = [r for r in withheld if r.name[-1] == "support"] or withheld
        t = I.sites.set_index("site")
        pzf = [Row(("pzf", s), "lin", pzf_weights(I, s, t.loc[s, "period_start"], t.loc[s, "period_end"]))
               for s in t.index]
        for name, extra in (("bias rows withheld by scripts/10, added back", leak),
                            ("per-tower P_zf added", pzf)):
            r = check_family(rows + extra, "et_ensemble", ("lin",))
            controls[name] = {"rows_added": len(extra), "found": len(r.get("small_sets", [])) + len(r["determined"]),
                              "passed": bool(r.get("small_sets") or r["determined"])}
            print(f"control: {name} ({len(extra)} rows): {controls[name]['found']} sets found "
                  f"({'ok' if controls[name]['passed'] else 'NOT FOUND: the search is not working'})")
    passed = (not I.problems and all(r["passed"] for f in results.values() for r in f.values())
              and all(c["passed"] for c in controls.values()))
    complete = set(families) == set(FAMILIES) and bool(controls)
    record = {"passed": passed, "complete": complete, "checked_utc": datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"), "python": platform.python_version(), "min_site_months": dc.MIN_SITE_MONTHS,
        "problems": I.problems, "families": results, "controls": controls, "files": dict(sorted(I.files.items()))}
    (ROOT / RECORD).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / RECORD).write_text(json.dumps(record, indent=1) + "\n")
    print(f"\n{'PASSED' if passed else 'FAILED'}{'' if complete else ' (partial check: not valid for scripts/13)'}: "
          f"wrote {RECORD}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
