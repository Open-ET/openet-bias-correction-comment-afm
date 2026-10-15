"""How record-level splits and zero-filled targets inflate correction skill.

Builds a GO26-like correction dataset from public data only: OpenET Phase II
daily model ET at the benchmark cropland stations (overpass days) as inputs,
EBR-closed benchmark ET as the target, tower meteorology, day-of-year and
Koppen indicators as further features.  Two scenarios are scored under four
validation designs (random record split, site-grouped hold-out,
leave-one-site-out, leave-one-cluster-out):

  clean        targets are measured, non-gap-filled closed ET
  zero-filled  at a random 30% of sites the first quarter of the record is
               treated as "tower not yet recording": target = 0 and tower
               meteorology = climate-zone medians (GO26 Section 2.2)

Both scenarios are run with missing tower meteorology handled two ways, on the
same records with the same seeds and zero-filled sites:

  primary       every missing value takes the Koppen-zone median of all
                station-days before the experiment, as GO26 imputed missing
                predictors with climate-zone medians (Section 2.2; theirs from
                the training partition; 35% of their VPD values were missing).
                Some stations have no tower meteorology in the benchmark files:
                LYS_NE/NW/SE/SW none of rn, t_avg, vpd and ws; B_01, B_11 and
                ET_8 only rn; US-ARM, US-IB1, US-Ne1-3, S2 and MB_Pch no vpd.
                Their real days then carry zone medians as zero-filled days do
                (at the lysimeters the whole vector), which dilutes the
                zero-fill signature as in GO26's own data.  A zone with no value
                stays missing (t_avg, vpd, ws in BWk: B_01, B_11; vpd in BSh:
                MB_Pch), where GO26 used the global median.
  met_unfilled  sensitivity: missing values (+/-inf included) stay NaN, which
                XGBoost sends down a learned default branch, so only the
                zero-filled days carry zone medians: the undiluted case.

Leave-one-cluster-out mirrors GO26's LOCO-CV (Section 3.4), which held out
colocated towers together.  Clusters are connected components of the stations
in the experiment within --cluster-km (10 km) of each other: single linkage on
great-circle distance between the Station_metadata.xlsx coordinates.  For a
one-station cluster the two leave-one-out designs fit the same model, so their
pooled difference comes from the multi-station clusters (broken down in
leakage_experiment_clusters.csv).  The first three designs use the same splits
and seeds as before the cluster design was added.

Outputs (results/):
  leakage_experiment.csv           one row per scenario x seed x design (primary)
  leakage_experiment_summary.csv   mean, min and max over seeds (primary)
  leakage_experiment_clusters.csv  RMSE per multi-station cluster under LOSO and LOCO (primary)
  leakage_experiment_met_unfilled.csv, leakage_experiment_met_unfilled_summary.csv
                                   the first two tables for met_unfilled
  leakage_experiment_versions.json versions, seeds, model settings, input md5s, clusters,
                                   meteorology gaps and how each variant handles them (no run
                                   time, so that a rerun reproduces it; the time is printed)

Usage:
  python scripts/07_leakage_experiment.py --flux data/raw/flux_ET_dataset \
      --openet data/raw/OpenET_PhaseII_model_ET_dataset
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from etcomment import benchmark as bm
from etcomment import leakage as lk
from etcomment import maps as mp
from etcomment import metrics as mt

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["geeSEBAL", "PT-JPL", "SSEBop", "SIMS", "eeMETRIC", "DisALEXI", "Ensemble"]
MET = ["rn", "t_avg", "vpd", "ws"]
CLUSTER_KM = 10.0
FRAC_SITES, FRAC_RECORD = 0.3, 0.25
TEST_SIZE = 0.2
SUMMARY = ["improvement_pct", "bias_share_pct", "rmse_base", "rmse_model", "mbe_base", "mbe_model",
           "flagged_mean_target", "flagged_mean_base", "flagged_mean_model"]
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "openpyxl", "kgcpy"]
LOO = ["leave-one-site-out", "leave-one-cluster-out"]
VARIANTS = {"primary": "leakage_experiment", "met_unfilled": "leakage_experiment_met_unfilled"}  # output stems
TOWER_MET = {
    "primary": "missing values take the Koppen-zone median of all station-days before the experiment "
               "(GO26 Section 2.2); a zone with no value stays missing",
    "met_unfilled": "missing values (and +/-inf) stay NaN for XGBoost; only zero-filled records carry zone medians",
}


def cropland_stations(openet: Path) -> pd.DataFrame:
    meta = pd.read_excel(openet / "Station_metadata.xlsx").rename(
        columns={"Unnamed: 0": "site", "Latitude": "lat", "Longitude": "lon",
                 "General classification": "land_cover"})
    return meta[meta["land_cover"] == "Croplands"].copy()


def build(root: Path, openet: Path, cluster_km: float = CLUSTER_KM, fill_met: bool = True) -> pd.DataFrame:
    """Measured overpass days at the cropland stations that have at least 20 of them.

    ``fill_met`` (primary) gives missing tower meteorology the Koppen-zone median
    of all station-days; otherwise (met_unfilled) it stays NaN.  The records are
    the same either way.
    """
    oe = bm.load_openet_phase2(openet, "daily")
    crop = cropland_stations(openet)
    crop["koppen"] = mp.koppen_at(crop["lat"], crop["lon"])
    rows = []
    for _, m in crop.iterrows():
        f = root / "daily_data_files" / f"{m['site']}_daily_data.csv"
        if not f.exists():
            continue
        d = bm.standardize(bm.load_daily(root, m["site"]))
        o = oe[oe["site"] == m["site"]].set_index("date")[MODELS]
        j = o.join(d[[c for c in ["et_corr", "valid_measured", *MET] if c in d]], how="inner")
        j = j[j["valid_measured"]]
        if len(j) < 20:
            continue
        j["site"] = m["site"]
        j["koppen"] = m["koppen"][:3]
        rows.append(j.reset_index())
    df = pd.concat(rows, ignore_index=True)
    df["doy_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365.25)
    df = pd.concat([df, pd.get_dummies(df["koppen"], prefix="kg", dtype=float)], axis=1)
    for c in MET:
        if c in df:
            # a few raw values are +/-inf (e.g. vpd at Ellendale); treat them as missing
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if fill_met:
                df[c] = df[c].fillna(df.groupby("koppen")[c].transform("median"))
    df = df.dropna(subset=["Ensemble", "et_corr"])
    # clusters among the stations kept, so chains never run through an unused station
    st = crop.set_index("site").loc[sorted(df["site"].unique())]
    df["cluster"] = df["site"].map(lk.distance_clusters(st.index, st["lat"], st["lon"], cluster_km))
    return df


def mark_preinstall(df: pd.DataFrame, frac_sites=FRAC_SITES, frac_record=FRAC_RECORD, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    sites = df["site"].unique()
    chosen = rng.choice(sites, size=max(1, int(frac_sites * len(sites))), replace=False)
    miss = pd.Series(False, index=df.index)
    for s in chosen:
        idx = df.index[df["site"] == s]
        dates = df.loc[idx, "date"].sort_values()
        cut = dates.iloc[int(frac_record * len(dates))]
        miss.loc[idx[df.loc[idx, "date"] < cut]] = True
    return miss


def cluster_table(base: pd.DataFrame, stations: pd.DataFrame) -> list[dict]:
    """Members of each cluster, its largest single-linkage step and its distance to other clusters.

    The clusters are unchanged for any threshold from the largest step to (but
    excluding) the smallest distance between clusters.
    """
    st = stations.set_index("site").loc[sorted(base["site"].unique())]
    lab = base.groupby("site")["cluster"].first().loc[st.index].to_numpy()
    lat, lon = st["lat"].to_numpy(float), st["lon"].to_numpy(float)
    d = lk.haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    rows = []
    for c in sorted(set(lab)):
        inside = lab == c
        step = linkage(squareform(d[np.ix_(inside, inside)], checks=False), "single")[-1, 2] if inside.sum() > 1 else None
        other = d[np.ix_(inside, ~inside)]
        rows.append({"cluster": c, "sites": list(st.index[inside]), "n_records": int((base["cluster"] == c).sum()),
                     "largest_step_km": None if step is None else round(float(step), 3),
                     "nearest_other_cluster_km": round(float(other.min()), 3) if other.size else None})
    return rows


def per_cluster(df: pd.DataFrame, pred: pd.DataFrame, multi: list[str]) -> pd.DataFrame:
    """RMSE of OpenET and of the held-out model at each multi-station cluster (all its records)."""
    rows = []
    for c in multi:
        i = (df["cluster"] == c).to_numpy()
        y = df["et_corr"].to_numpy()[i]
        rows.append({"cluster": c, "n_sites": df.loc[i, "site"].nunique(), "n_records": int(i.sum()),
                     "rmse_base": mt.rmse(df["Ensemble"].to_numpy()[i], y),
                     "rmse_loso": mt.rmse(pred[LOO[0]].to_numpy()[i], y),
                     "rmse_loco": mt.rmse(pred[LOO[1]].to_numpy()[i], y)})
    return pd.DataFrame(rows)


def met_variants(bases: dict[str, pd.DataFrame], shared: dict[str, dict]) -> dict:
    """Tower-meteorology gaps in the experiment's records and how each variant handles them.

    ``bases`` maps each variant to its :func:`build` output; ``shared`` maps it to
    the unflagged records, per seed, whose meteorology equals a zero-filled record's.
    """
    raw = bases["met_unfilled"]
    met = [c for c in MET if c in raw]
    gap = raw[met].isna()
    none = gap.groupby(raw["site"]).all()
    out = {"note": "both variants use the same records, seeds and zero-filled sites (zero_fill.sites_by_seed)",
           "missing_station_days": {c: int(gap[c].sum()) for c in met},
           "stations_without": {c: sorted(none.index[none[c]]) for c in met},
           "variants": {}}
    for v, b in bases.items():
        left = b[met].isna().sum()
        files = [f"results/{VARIANTS[v]}{s}" for s in (".csv", "_summary.csv")]
        out["variants"][v] = {
            "tower_met": TOWER_MET[v],
            "outputs": files + (["results/leakage_experiment_clusters.csv"] if v == "primary" else []),
            "values_imputed": {c: int(gap[c].sum() - left[c]) for c in met},
            "values_left_missing": {c: int(left[c]) for c in met},
            "unflagged_records_sharing_zero_fill_met": shared.get(v, {}),
        }
    return out


def provenance(base: pd.DataFrame, stations: pd.DataFrame, root: Path, openet: Path, seeds: list[int],
               cluster_km: float, zero_filled_sites: dict[str, list[str]], tower_met: dict | None = None) -> dict:
    sites = sorted(base["site"].unique())
    members = cluster_table(base, stations)
    multi = [m for m in members if len(m["sites"]) > 1]
    gaps = [m["nearest_other_cluster_km"] for m in members if m["nearest_other_cluster_km"] is not None]
    files = {"OpenET_PhaseII_model_ET_dataset/daily_data.dat": openet / "daily_data.dat",
             "OpenET_PhaseII_model_ET_dataset/Station_metadata.xlsx": openet / "Station_metadata.xlsx"}
    files.update({f"flux_ET_dataset/daily_data_files/{s}_daily_data.csv":
                  root / "daily_data_files" / f"{s}_daily_data.csv" for s in sites})
    model = {k: str(v) if isinstance(v, float) and np.isnan(v) else v  # missing=nan
             for k, v in lk.default_model(0).get_params().items() if v is not None}
    model["random_state"] = "seed"
    # no run time: the file depends only on the inputs, settings and versions, so a rerun reproduces it
    # byte for byte (main() prints the time instead)
    return {
        "script": "scripts/07_leakage_experiment.py",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: metadata.version(p) for p in PACKAGES},
        "seeds": [int(s) for s in seeds],
        "seed_controls": "train_test_split and GroupShuffleSplit random_state, XGBoost random_state, "
                         "choice of zero-filled sites",
        "model": {"class": "xgboost.XGBRegressor", **model},
        "test_size": TEST_SIZE,
        "zero_fill": {"frac_sites": FRAC_SITES, "frac_record": FRAC_RECORD, "sites_by_seed": zero_filled_sites},
        "tower_meteorology": tower_met,
        "n_sites": len(sites),
        "n_records": int(len(base)),
        "clusters": {
            "rule": "connected components of stations within max_km of each other (single linkage, "
                    "great-circle distance), computed over the stations in the experiment",
            "max_km": float(cluster_km),
            "earth_radius_km": lk.EARTH_RADIUS_KM,
            "coordinates": "OpenET_PhaseII_model_ET_dataset/Station_metadata.xlsx (Latitude, Longitude)",
            "n_clusters": len(members),
            "n_multi_station": len(multi),
            "n_stations_in_multi_station": sum(len(m["sites"]) for m in multi),
            "same_clusters_for_max_km": [max((m["largest_step_km"] for m in multi), default=0.0),
                                         min(gaps, default=None)],
            "members": members,
        },
        "inputs": {
            "zenodo": {k: {"doi": f"10.5281/zenodo.{v['record']}", "file": v["file"], "md5": v["md5"]}
                       for k, v in bm.ZENODO.items()},
            "md5": {k: bm.md5(p) for k, p in files.items()},
        },
    }


def summarize(out: pd.DataFrame) -> pd.DataFrame:
    s = out.groupby(["scenario", "design"], sort=False)[SUMMARY].agg(["mean", "min", "max"])
    s.columns = [f"{c}_{stat}" for c, stat in s.columns]
    return s.reset_index()


def _fmt(summ: pd.DataFrame, cols: dict[str, str]) -> str:
    t = summ.set_index(["scenario", "design"])
    return pd.DataFrame({k: [f"{a:6.2f} ({b:.2f}..{c:.2f})" for a, b, c in
                             zip(t[f"{v}_mean"], t[f"{v}_min"], t[f"{v}_max"])]
                         for k, v in cols.items()}, index=t.index).to_string()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flux", default=str(ROOT / "data/raw/flux_ET_dataset"))
    ap.add_argument("--openet", default=str(ROOT / "data/raw/OpenET_PhaseII_model_ET_dataset"))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2],
                    help="random seeds (split, model and choice of zero-filled sites)")
    ap.add_argument("--cluster-km", type=float, default=CLUSTER_KM,
                    help="stations linked by steps of at most this distance are held out together")
    args = ap.parse_args()
    print(f"run started {datetime.now(timezone.utc).isoformat(timespec='seconds')} (UTC; not written to "
          f"leakage_experiment_versions.json, which a rerun reproduces byte for byte)", flush=True)
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    root, openet = bm.find_root(Path(args.flux)), Path(args.openet)
    stations = cropland_stations(openet)
    bases = {v: build(root, openet, args.cluster_km, fill_met=(v == "primary")) for v in VARIANTS}
    base = bases["primary"]
    keys = ["site", "date", "koppen", "cluster", "et_corr", *MODELS]
    if not bases["met_unfilled"][keys].equals(base[keys]):
        raise RuntimeError("the two meteorology variants must share their records")
    met = [c for c in MET if c in base]
    feats = MODELS + met + ["doy_sin", "doy_cos"] + [c for c in base if c.startswith("kg_")]
    multi = sorted(c for c, n in base.groupby("cluster")["site"].nunique().items() if n > 1)
    res, clus = {v: [] for v in VARIANTS}, {v: [] for v in VARIANTS}
    shared, zf_sites = {v: {} for v in VARIANTS}, {}
    for seed in args.seeds:
        miss = mark_preinstall(base, seed=seed)
        zf_sites[str(seed)] = sorted(base.loc[miss, "site"].unique())
        for v, b in bases.items():
            df = b.copy()
            df["tower_missing"] = miss
            zf = lk.inject_zero_fill(df, "tower_missing", "et_corr", met, "koppen")
            same = lk.matches_flagged(zf, "tower_missing", met)
            shared[v][str(seed)] = {"n_records": int(same.sum()), "sites": sorted(zf.loc[same, "site"].unique())}
            for name, d in (("clean", df), ("zero-filled + median-imputed", zf)):
                r, pred = lk.run_designs(d, feats, "et_corr", "Ensemble", "site", seed=seed, test_size=TEST_SIZE,
                                         flag="tower_missing", cluster="cluster", return_predictions=True)
                r.insert(0, "scenario", name)
                r.insert(1, "seed", seed)
                r["zero_filled_fraction"] = float(df["tower_missing"].mean())
                res[v].append(r)
                pc = per_cluster(d, pred, multi)
                pc.insert(0, "scenario", name)
                pc.insert(1, "seed", seed)
                clus[v].append(pc)
        print("seed", seed, "done", flush=True)
    summ = {}
    for v, stem in VARIANTS.items():
        out = pd.concat(res[v], ignore_index=True)
        out["n_sites"] = base["site"].nunique()
        out["n_records"] = len(base)
        out.to_csv(out_dir / f"{stem}.csv", index=False)
        summ[v] = summarize(out)
        summ[v].to_csv(out_dir / f"{stem}_summary.csv", index=False)
    clus = {v: pd.concat(c, ignore_index=True) for v, c in clus.items()}
    clus["primary"].to_csv(out_dir / "leakage_experiment_clusters.csv", index=False)
    prov = provenance(base, stations, root, openet, args.seeds, args.cluster_km, zf_sites,
                      met_variants(bases, shared))
    (out_dir / "leakage_experiment_versions.json").write_text(json.dumps(prov, indent=1, allow_nan=False) + "\n")

    c = prov["clusters"]
    print(f"{prov['n_sites']} stations, {prov['n_records']} records, {c['n_clusters']} clusters at "
          f"{c['max_km']:g} km ({c['n_multi_station']} with 2+ stations; same clusters for "
          f"{c['same_clusters_for_max_km'][0]:.2f}-{c['same_clusters_for_max_km'][1]:.2f} km):")
    for m in c["members"]:
        if len(m["sites"]) > 1:
            print(f"  {'/'.join(m['sites'])}: {m['n_records']} records, largest step {m['largest_step_km']:.2f} km")
    tm = prov["tower_meteorology"]
    print("\ntower meteorology missing on station-days:",
          ", ".join(f"{k} {n}" for k, n in tm["missing_station_days"].items()))
    for v, t in tm["variants"].items():
        n = "/".join(str(s["n_records"]) for s in t["unflagged_records_sharing_zero_fill_met"].values())
        print(f"  {v}: {n} unflagged records (seeds {args.seeds}) share a zero-filled record's meteorology")
    for v in VARIANTS:
        print(f"\n== {v}: {TOWER_MET[v]}\nmean (min..max) over seeds", args.seeds)
        print(_fmt(summ[v], {"RMSE gain %": "improvement_pct", "Bias2 share %": "bias_share_pct",
                             "RMSE model": "rmse_model", "MBE base": "mbe_base"}))
        print("\nflagged (zero-filled) records, mean ET mm/d")
        print(_fmt(summ[v], {"target": "flagged_mean_target", "OpenET": "flagged_mean_base",
                             "model": "flagged_mean_model"}))
    print("\nper-cluster RMSE (mm/d), mean over seeds (*_unfilled: met_unfilled)")
    pc = {v: t.groupby(["scenario", "cluster"], sort=False)[["n_records", "rmse_base", "rmse_loso", "rmse_loco"]]
          .mean() for v, t in clus.items()}
    print(pc["primary"].join(pc["met_unfilled"][["rmse_loso", "rmse_loco"]], rsuffix="_unfilled").round(3).to_string())


if __name__ == "__main__":
    main()
