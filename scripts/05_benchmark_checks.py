"""Benchmark-wide checks on the Volk et al. (2023a) flux data.

1. Exact-zero and near-zero daily ET: how often measured (not gap-filled),
   closure-corrected daily ET is within 0.015 mm/d of zero at each of the
   161 stations, and the longest run of such days.  GO26's Fig. S2 has 206
   such days in 353 at US-UTJ, with a 156-day run.
2. Plain daily Bowen-ratio closure, LE (Rn - G)/(LE + H), versus the
   benchmark's EBR closure at every eddy covariance station: daily
   differences and how often the daily Bowen-ratio factor is negative,
   near-singular (|LE + H| < 5% of |Rn - G|) or outside 0.5-2.  Bowen-ratio
   energy-balance stations (closure forced by design) and the weighing
   lysimeters are excluded.
3. OpenET ensemble bias (Phase II data) at the cropland stations used by
   V24, on overpass days and in months with <= 5 gap-filled days.
4. The bias shift f x E that counting every non-measured day of each
   calendar-year record as zero would add (f = share of such days, E = the
   benchmark's gap-fill estimate of ET on them).

The seven sites shared with GO26 are treated in 06_shared_site_comparison.py.

Usage:
  python scripts/05_benchmark_checks.py --flux data/raw/flux_ET_dataset \
      --openet data/raw/OpenET_PhaseII_model_ET_dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from etcomment import benchmark as bm
from etcomment import closure as cl
from etcomment import maps as mp
from etcomment import metrics as mt

ROOT = Path(__file__).resolve().parents[1]
ZERO_TOL = 0.015  # mm/d, same tolerance as the digitization of GO26 Fig. S2


def station_metadata(root: Path) -> pd.DataFrame:
    m = pd.read_excel(root / "station_metadata.xlsx", header=1).rename(columns={
        "Site ID": "site", "General classification": "land_cover", "Latitude": "lat",
        "Longitude": "lon", "Measurement technique": "technique"})
    m["koppen"] = mp.koppen_at(m["lat"], m["lon"])
    m["koppen_group"] = m["koppen"].str[0]
    return m


def site_daily(root: Path, site: str) -> pd.DataFrame:
    d = bm.standardize(bm.load_daily(root, site))
    le_br, f_br = cl.bowen_ratio_closure(d["le"], d["h"], d["rn"], d["g"])
    # ET from BR-closed LE with the file's own LE -> ET factor (temperature-dependent lambda)
    d["et_br"] = le_br * (d["et_corr"] / d["le_corr"]) if "le_corr" in d else np.nan
    d["br_factor"] = f_br
    return d


def longest_run(mask: pd.Series) -> int:
    if not mask.any():
        return 0
    runs = (mask != mask.shift()).cumsum()
    return int(mask.groupby(runs).sum().max())


def zero_survey(root: Path, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, m in meta.iterrows():
        f = root / "daily_data_files" / f"{m['site']}_daily_data.csv"
        if not f.exists():
            continue
        d = bm.standardize(bm.load_daily(root, m["site"]))
        v = d["valid_measured"]
        e = d.loc[v, "et_corr"]
        near = (d["et_corr"].abs() < ZERO_TOL) & v
        rows.append({"site": m["site"], "land_cover": m["land_cover"], "koppen": m["koppen"],
                     "technique": m["technique"], "n_measured_days": int(v.sum()),
                     "n_exact_zero": int((e == 0).sum()), "n_near_zero": int(near.sum()),
                     "pct_near_zero": 100 * near.sum() / max(1, v.sum()),
                     "n_negative": int((e < 0).sum()), "longest_near_zero_run_days": longest_run(near)})
    return pd.DataFrame(rows)


def bowen_vs_ebr(root: Path, meta: pd.DataFrame, min_days: int = 30) -> pd.DataFrame:
    rows = []
    for _, m in meta[meta["technique"] == "Eddy covariance"].iterrows():
        f = root / "daily_data_files" / f"{m['site']}_daily_data.csv"
        if not f.exists():
            continue
        d = site_daily(root, m["site"])
        if d["h"].isna().all():
            continue
        fl = cl.br_diagnostics(d["le"], d["h"], d["rn"], d["g"])
        fl.index = d.index
        v = fl["valid_inputs"] & d["valid_measured"]
        if v.sum() < min_days:
            continue
        diff = (d["et_br"] - d["et_corr"])[v]
        fin = diff[np.isfinite(diff)]
        winter = np.isin(d.index.month, [11, 12, 1, 2, 3])
        prob = fl["any_problem"] & v
        n_prob = int(prob.sum())
        rows.append({
            "site": m["site"], "land_cover": m["land_cover"], "koppen": m["koppen"],
            "koppen_group": m["koppen_group"], "n_days": int(v.sum()),
            "median_diff_mm_d": float(fin.median()),
            "mean_diff_mm_d": float(fin.mean()),
            "mean_diff_mm_d_abs_lt5": float(fin[fin.abs() < 5].mean()),
            "n_abs_diff_ge5": int((diff.abs() >= 5).sum()),
            "p95_abs_diff_mm_d": float(fin.abs().quantile(0.95)),
            "pct_abs_diff_gt1": 100 * float((fin.abs() > 1).mean()),
            "max_abs_diff_mm_d": float(diff.abs().max()),
            "pct_h_negative": 100 * float(fl.loc[v, "h_negative"].mean()),
            "pct_factor_negative": 100 * float(fl.loc[v, "factor_negative"].mean()),
            "pct_flux_near_zero": 100 * float(fl.loc[v, "flux_near_zero"].mean()),
            "pct_factor_outside_0p5_2": 100 * float(fl.loc[v, "factor_extreme"].mean()),
            "pct_any_problem": 100 * float(prob[v].mean()),
            "n_problem_days": n_prob,
            # share of problem days in Nov-Mar (NaN without problem days) and, for reference,
            # the share of all valid days that fall in Nov-Mar
            "pct_problem_days_nov_mar": 100 * float(prob[v & winter].sum() / n_prob) if n_prob else np.nan,
            "pct_valid_days_nov_mar": 100 * float((v & winter).sum() / v.sum()),
        })
    return pd.DataFrame(rows)


def openet_bias(root: Path, oe_daily: pd.DataFrame, oe_monthly: pd.DataFrame, sites: list[str]) -> pd.DataFrame:
    rows = []
    for s in sites:
        d = site_daily(root, s)
        od = oe_daily[oe_daily["site"] == s].set_index("date")["Ensemble"]
        j = d.join(od.rename("openet"), how="inner")
        j = j[j["valid_measured"] & j["openet"].notna()]
        r = {"site": s, "n_daily": len(j)}
        for ref in ("et_corr", "et", "et_br"):
            ok = np.isfinite(j[ref])
            r[f"daily_mbe_vs_{ref}"] = mt.mbe(j.loc[ok, "openet"], j.loc[ok, ref]) if ok.any() else np.nan
            r[f"daily_rmse_vs_{ref}"] = mt.rmse(j.loc[ok, "openet"], j.loc[ok, ref]) if ok.any() else np.nan
        r["daily_r2cod_vs_et_corr"] = mt.r2_cod(j["openet"], j["et_corr"]) if len(j) > 2 else np.nan
        g = bm.complete_months(d)
        om = oe_monthly[oe_monthly["site"] == s].set_index("date")["Ensemble"]
        jm = g.join(om.rename("openet"), how="inner").dropna(subset=["openet"])
        r["n_monthly"] = len(jm)
        r["monthly_mbe_mm"] = mt.mbe(jm["openet"], jm["et_corr"]) if len(jm) else np.nan
        r["monthly_mbe_mm_d"] = float(((jm["openet"] - jm["et_corr"]) / jm["n_days"]).mean()) if len(jm) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def zero_fill_shift(root: Path, sites: list[str]) -> pd.DataFrame:
    """Bias shift if every non-measured day of whole calendar years were zero."""
    rows = []
    for s in sites:
        d = site_daily(root, s)
        full = pd.date_range(f"{d.index.min().year}-01-01", f"{d.index.max().year}-12-31", freq="D")
        x = d.reindex(full)
        missing = ~x["valid_measured"].fillna(False).astype(bool)
        proxy = pd.to_numeric(x["et_fill"] if "et_fill" in x else x["et_corr"], errors="coerce")
        known = proxy.notna()
        f = float(missing.mean())
        e = float(proxy[missing & known].mean()) if (missing & known).any() else np.nan
        rows.append({"site": s, "calendar_days": len(full), "non_measured_days": int(missing.sum()),
                     "f": f, "E_gapfill_mm_d": e, "delta_mbe_mm_d": f * e if np.isfinite(e) else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flux", default=str(ROOT / "data/raw/flux_ET_dataset"))
    ap.add_argument("--openet", default=str(ROOT / "data/raw/OpenET_PhaseII_model_ET_dataset"))
    args = ap.parse_args()
    root = bm.find_root(Path(args.flux))
    openet = Path(args.openet)
    meta = station_metadata(root)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    z = zero_survey(root, meta)
    z.to_csv(out / "benchmark_zero_survey.csv", index=False)
    br = bowen_vs_ebr(root, meta)
    br.to_csv(out / "benchmark_bowen_ratio_vs_ebr.csv", index=False)

    oe_d = bm.load_openet_phase2(openet, "daily")
    oe_m = bm.load_openet_phase2(openet, "monthly")
    v24 = pd.read_excel(openet / "Station_metadata.xlsx").rename(columns={"Unnamed: 0": "site"})
    crop = [s for s in v24.loc[v24["General classification"] == "Croplands", "site"]
            if (root / "daily_data_files" / f"{s}_daily_data.csv").exists()]
    openet_bias(root, oe_d, oe_m, crop).to_csv(out / "benchmark_openet_bias_cropland.csv", index=False)
    zf = zero_fill_shift(root, crop)
    zf.to_csv(out / "benchmark_zero_fill_shift_cropland.csv", index=False)

    # console summary
    c = z[z["land_cover"] == "Croplands"]
    print(f"zero survey: {len(z)} stations, {z['n_measured_days'].sum()} measured station-days, "
          f"{z['n_exact_zero'].sum()} exact zeros, {100 * z['n_near_zero'].sum() / z['n_measured_days'].sum():.2f}% "
          f"near zero, longest run {z['longest_near_zero_run_days'].max()} d")
    print(f"  croplands: {len(c)} stations, {c['n_measured_days'].sum()} station-days, {c['n_exact_zero'].sum()} exact zeros, "
          f"{100 * c['n_near_zero'].sum() / c['n_measured_days'].sum():.2f}% near zero, "
          f"longest run {c['longest_near_zero_run_days'].max()} d")
    bc = br[br["land_cover"] == "Croplands"]
    cols = ["n_days", "median_diff_mm_d", "mean_diff_mm_d", "mean_diff_mm_d_abs_lt5", "p95_abs_diff_mm_d",
            "pct_abs_diff_gt1", "pct_any_problem", "pct_h_negative", "pct_problem_days_nov_mar",
            "pct_valid_days_nov_mar"]
    print(f"Bowen ratio vs EBR, {len(bc)} cropland EC stations (medians by Koppen group):")
    print(bc.groupby("koppen_group")[cols].median().round(2).to_string())
    print(bc[cols].describe().round(2).to_string())
    print(f"  days with |BR - EBR| >= 5 mm/d: {bc['n_abs_diff_ge5'].sum()} of {bc['n_days'].sum()}")
    neg = (z.loc[z['land_cover'] == 'Croplands', 'n_negative'].sum(), c['n_measured_days'].sum())
    print(f"negative closed ET on {100 * z['n_negative'].sum() / z['n_measured_days'].sum():.2f}% of measured days "
          f"({100 * neg[0] / neg[1]:.2f}% at croplands)")
    print(f"zero-fill shift, {len(zf)} V24 cropland stations: median f {zf['f'].median():.2f}, "
          f"median delta MBE {zf['delta_mbe_mm_d'].median():.2f} mm/d "
          f"(IQR {zf['delta_mbe_mm_d'].quantile(.25):.2f}-{zf['delta_mbe_mm_d'].quantile(.75):.2f})")


if __name__ == "__main__":
    main()
