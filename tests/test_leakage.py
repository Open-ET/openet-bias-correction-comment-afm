"""Validation designs of the leakage experiment (etcomment.leakage, scripts/07); synthetic data only."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from etcomment import leakage as lk

KM_PER_DEG = np.pi * lk.EARTH_RADIUS_KM / 180  # great-circle km per degree of latitude
DESIGNS = ["random record split", "site-grouped hold-out", "leave-one-site-out", "leave-one-cluster-out"]
FEATS = dict(features=["x", "s"], target="y", baseline="base", group="site")


def _script():
    spec = importlib.util.spec_from_file_location(
        "leakage_experiment", Path(__file__).resolve().parents[1] / "scripts" / "07_leakage_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _linear(seed):
    return LinearRegression()


def _synthetic(n_clusters=6, per_cluster=2, n=40, seed=0):
    """Colocated sites share a cluster offset in y and an almost identical site covariate ``s``."""
    rng = np.random.default_rng(seed)
    parts = []
    for k in range(n_clusters):
        for j in range(per_cluster):
            x = rng.uniform(0, 5, n)
            parts.append(pd.DataFrame({
                "site": f"K{k}S{j}", "cluster": f"K{k}", "x": x, "s": k + 0.001 * j, "base": x,
                "y": x + (-1) ** k * (k + 1) + rng.normal(0, 0.1, n), "flag": rng.random(n) < 0.1}))
    return pd.concat(parts, ignore_index=True)


def _inputs(tmp_path, mod, lat):
    """Synthetic flux_ET_dataset/ and OpenET Phase II files for scripts/07 (site -> latitude)."""
    import synthetic_benchmark as sb

    sites = list(lat)
    root = sb.write(tmp_path, sites, start="2018-01-01", end="2018-12-31")
    oe = tmp_path / "openet"
    oe.mkdir()
    pd.DataFrame({"General classification": "Croplands", "Latitude": list(lat.values()), "Longitude": -96.5},
                 index=sites).to_excel(oe / "Station_metadata.xlsx")
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", "2018-12-31", freq="8D").strftime("%Y-%m-%d")
    dat = pd.DataFrame([(s, d) for s in sites for d in dates], columns=["Site ID", "DATE"])
    for m in mod.MODELS:
        dat[m] = rng.uniform(0.5, 6.0, len(dat)).round(1)
    with open(oe / "daily_data.dat", "w") as f:
        f.write("\n    # synthetic stand-in for the OpenET Phase II daily file\n\n")
        dat[["Site ID", *mod.MODELS, "DATE"]].to_csv(f, sep="\t", index=False)
    return root, oe


def _edit_daily(root, site, fn):
    f = root / "daily_data_files" / f"{site}_daily_data.csv"
    fn(pd.read_csv(f)).to_csv(f, index=False)


def _met_gaps(tmp_path, monkeypatch, mod, bwk=True):
    """Stations as in the benchmark: one without any tower meteorology (LYS_*), one with +/-inf vpd
    (Ellendale) and, with ``bwk``, one with only rn in a zone of its own (B_01/B_11 in BWk)."""
    lat = {"S-A": 41.0, "S-B": 41.0 + 5 / KM_PER_DEG, "S-C": 42.0, "S-L": 42.5, **({"S-W": 43.5} if bwk else {})}
    monkeypatch.setattr(mod.mp, "koppen_at", lambda la, lo: ["BWk" if y > 43 else "Dfa" for y in la])
    root, oe = _inputs(tmp_path, mod, lat)
    _edit_daily(root, "S-L", lambda d: d.drop(columns=["Rn", "t_avg", "vpd", "ws"]))
    _edit_daily(root, "S-C", lambda d: d.assign(vpd=d["vpd"].where(~d["date"].str.startswith("2018-07"), np.inf)))
    if bwk:
        _edit_daily(root, "S-W", lambda d: d.drop(columns=["t_avg", "vpd", "ws"]))
    return root, oe


def _recording_factory():
    """Model factory whose models log the index of the rows they are fitted on and predict."""
    log = []

    class Recorder(LinearRegression):
        def fit(self, X, y):
            log.append({"train": X.index.to_numpy()})
            return super().fit(X, y)

        def predict(self, X):
            log[-1]["test"] = X.index.to_numpy()
            return super().predict(X)

    return (lambda seed: Recorder()), log


def test_haversine_km():
    assert np.isclose(lk.haversine_km(40.0, -100.0, 41.0, -100.0), KM_PER_DEG)
    assert lk.haversine_km(40.0, -100.0, 40.0, -100.0) == 0.0
    # a degree of longitude shrinks with cos(latitude)
    assert np.isclose(lk.haversine_km(60.0, 10.0, 60.0, 11.0), 0.5 * KM_PER_DEG, rtol=1e-4)


def test_two_stations_5_km_apart_and_one_50_km_away_give_two_clusters():
    sites = ["S-C", "S-A", "S-B"]
    lat = [41.0 + 50 / KM_PER_DEG, 41.0, 41.0 + 5 / KM_PER_DEG]
    lon = [-96.5, -96.5, -96.5]
    cl = lk.distance_clusters(sites, lat, lon, max_km=10)
    assert cl.nunique() == 2
    assert cl["S-A"] == cl["S-B"] == "S-A+S-B" and cl["S-C"] == "S-C"
    # labels do not depend on the order of the stations
    rev = lk.distance_clusters(sites[::-1], lat[::-1], lon[::-1], max_km=10)
    assert rev.sort_index().equals(cl.sort_index())


def test_clusters_chain_by_single_linkage():
    lat = 40.0 + np.array([0.0, 8.0, 16.0]) / KM_PER_DEG  # 8 km steps; the ends are 16 km apart
    lon = [-100.0] * 3
    assert set(lk.distance_clusters(["A", "B", "C"], lat, lon, max_km=10)) == {"A+B+C"}
    assert lk.distance_clusters(["A", "B", "C"], lat, lon, max_km=5).nunique() == 3
    with pytest.raises(ValueError):
        lk.distance_clusters(["A", "A"], [40.0, 40.1], [-100.0, -100.0])


def test_run_designs_appends_cluster_design_and_leaves_others_unchanged():
    df = _synthetic()
    res = lk.run_designs(df, **FEATS, model_factory=_linear, seed=1, flag="flag", cluster="cluster")
    old = lk.run_designs(df, **FEATS, model_factory=_linear, seed=1, flag="flag")
    assert list(res["design"]) == DESIGNS and list(old["design"]) == DESIGNS[:3]
    assert list(res["n_folds"]) == [1, 1, 12, 6] and list(old["n_folds"]) == [1, 1, 12]
    assert (res["n_groups"] == 12).all() and (res["n_clusters"] == 6).all() and old["n_clusters"].isna().all()
    pd.testing.assert_frame_equal(res.iloc[:3].drop(columns="n_clusters"), old.drop(columns="n_clusters"))
    loco = res.set_index("design").loc["leave-one-cluster-out"]
    assert loco["n_test"] == len(df) and loco["n_flagged"] == df["flag"].sum()
    assert np.isclose(loco["flagged_mean_target"], df.loc[df["flag"], "y"].mean())


def test_loco_never_puts_a_cluster_in_both_train_and_test():
    df = _synthetic()
    factory, log = _recording_factory()
    lk.run_designs(df, **FEATS, model_factory=factory, cluster="cluster")
    n_sites, n_clusters = df["site"].nunique(), df["cluster"].nunique()
    assert len(log) == 2 + n_sites + n_clusters  # random, hold-out, LOSO folds, LOCO folds
    loco = log[-n_clusters:]
    assert np.array_equal(np.sort(np.concatenate([f["test"] for f in loco])), df.index.to_numpy())
    for f in loco:
        train, test = set(df.loc[f["train"], "cluster"]), set(df.loc[f["test"], "cluster"])
        assert len(test) == 1 and not train & test
    # under leave-one-site-out the held-out site's colocated partner stays in training
    for f in log[2:2 + n_sites]:
        assert set(df.loc[f["train"], "cluster"]) & set(df.loc[f["test"], "cluster"])


def test_loco_matches_loso_for_single_site_clusters():
    df = _synthetic()
    df.loc[df["site"] == "K0S1", "cluster"] = "K0S1"  # K0S0 and K0S1 become one-site clusters
    _, pred = lk.run_designs(df, **FEATS, model_factory=_linear, cluster="cluster", return_predictions=True)
    assert list(pred.columns) == DESIGNS[2:] and pred.index.equals(df.index)
    single = df["site"].isin(["K0S0", "K0S1"]).to_numpy()
    assert np.array_equal(pred.iloc[single, 0], pred.iloc[single, 1])
    assert not np.allclose(pred.iloc[~single, 0], pred.iloc[~single, 1])


def test_loco_removes_the_colocated_site_leak():
    df = _synthetic()
    tree = lambda seed: DecisionTreeRegressor(min_samples_leaf=5, random_state=seed)  # noqa: E731
    res = lk.run_designs(df, **FEATS, model_factory=tree, cluster="cluster").set_index("design")
    # the partner site carries the cluster offset into leave-one-site-out training
    assert res.loc["leave-one-site-out", "rmse_model"] < 0.5
    assert res.loc["leave-one-cluster-out", "rmse_model"] > 10 * res.loc["leave-one-site-out", "rmse_model"]


@pytest.mark.parametrize("label", ["K5", None])
def test_run_designs_rejects_a_site_without_exactly_one_cluster(label):
    df = _synthetic()
    df.loc[df.index[0], "cluster"] = label
    with pytest.raises(ValueError):
        lk.run_designs(df, **FEATS, model_factory=_linear, cluster="cluster")


def test_inject_zero_fill():
    df = pd.DataFrame({
        "zone": ["A"] * 4 + ["B"] * 3,
        "miss": [True, False, False, False, True, False, False],
        "et": [3.0, 2.0, 4.0, 6.0, 5.0, 1.0, 2.0],
        "rn": [9.0, 1.0, 2.0, 10.0, 9.0, 4.0, 6.0],
        "ws": [np.nan, 1.0, 3.0, 5.0, 7.0, 2.0, 2.0],
    })
    out = lk.inject_zero_fill(df, "miss", "et", ["rn", "ws"], "zone")
    miss = df["miss"]
    assert (out.loc[miss, "et"] == 0.0).all()
    # missing days get the median of the zone's recorded days
    assert list(out.loc[miss, "rn"]) == [2.0, 5.0] and list(out.loc[miss, "ws"]) == [3.0, 2.0]
    pd.testing.assert_frame_equal(out.loc[~miss], df.loc[~miss])
    assert df.loc[0, "et"] == 3.0 and np.isnan(df.loc[0, "ws"])  # input untouched


def test_mark_preinstall_flags_the_first_quarter_at_30pct_of_sites():
    mod = _script()
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    df = pd.DataFrame({"site": np.repeat([f"S{i}" for i in range(10)], 8), "date": np.tile(dates, 10)})
    miss = mod.mark_preinstall(df, seed=0)
    per_site = miss.groupby(df["site"]).sum()
    assert (per_site > 0).sum() == 3 and set(per_site[per_site > 0]) == {2}
    assert df.loc[miss, "date"].max() == dates[1]
    assert miss.equals(mod.mark_preinstall(df, seed=0))


def test_build_adds_clusters_and_provenance_records_them(tmp_path, monkeypatch):
    mod = _script()
    monkeypatch.setattr(mod.mp, "koppen_at", lambda lat, lon: ["Dfa"] * len(lat))
    sites = ["S-A", "S-B", "S-C"]
    root, oe = _inputs(tmp_path, mod, dict(zip(sites, [41.0, 41.0 + 5 / KM_PER_DEG, 41.0 + 50 / KM_PER_DEG])))

    df = mod.build(root, oe)
    assert df.groupby("site")["cluster"].first().to_dict() == {"S-A": "S-A+S-B", "S-B": "S-A+S-B", "S-C": "S-C"}

    feats = mod.MODELS + mod.MET + ["doy_sin", "doy_cos"]
    _, pred = lk.run_designs(df, feats, "et_corr", "Ensemble", "site", model_factory=_linear,
                             cluster="cluster", return_predictions=True)
    pc = mod.per_cluster(df, pred, ["S-A+S-B"])
    assert pc.loc[0, "n_sites"] == 2 and pc.loc[0, "n_records"] == int(df["site"].isin(["S-A", "S-B"]).sum())

    prov = mod.provenance(df, mod.cropland_stations(oe), root, oe, [0, 1], 10.0, {"0": ["S-A"], "1": ["S-C"]})
    json.dumps(prov, allow_nan=False)
    assert {"numpy", "pandas", "scikit-learn", "xgboost"} <= set(prov["packages"]) and prov["python"]
    assert prov["seeds"] == [0, 1] and prov["n_sites"] == 3 and prov["n_records"] == len(df)
    c = prov["clusters"]
    assert c["n_clusters"] == 2 and c["n_multi_station"] == 1 and c["n_stations_in_multi_station"] == 2
    pair = next(m for m in c["members"] if m["cluster"] == "S-A+S-B")
    assert pair["sites"] == ["S-A", "S-B"] and np.isclose(pair["largest_step_km"], 5.0, atol=0.01)
    assert np.isclose(pair["nearest_other_cluster_km"], 45.0, atol=0.01)
    assert np.allclose(c["same_clusters_for_max_km"], [5.0, 45.0], atol=0.01)
    assert len(prov["inputs"]["md5"]) == 2 + len(sites)


def test_matches_flagged():
    df = pd.DataFrame({"f": [True, True, False, False, False, False],
                       "a": [1.0, np.nan, 1.0, np.nan, 1.0, np.nan],
                       "b": [3.0, 4.0, 3.0, 4.0, 4.0, np.nan]}, index=list("pqrstu"))
    m = lk.matches_flagged(df, "f", ["a", "b"])
    # whole vectors only, NaN equals NaN, flagged records never count
    assert m.index.equals(df.index) and list(m) == [False, False, True, True, False, False]
    assert not lk.matches_flagged(df.assign(f=False), "f", ["a", "b"]).any()


def test_primary_build_fills_missing_met_and_met_unfilled_leaves_it_missing(tmp_path, monkeypatch):
    mod = _script()
    root, oe = _met_gaps(tmp_path, monkeypatch, mod)
    filled, unfilled = mod.build(root, oe), mod.build(root, oe, fill_met=False)
    keys = ["site", "date", "koppen", "cluster", "et_corr", *mod.MODELS]
    pd.testing.assert_frame_equal(filled[keys], unfilled[keys])  # same records, so the same zero-filled days
    lys, bwk = filled["site"].eq("S-L"), filled["site"].eq("S-W")
    july = filled["site"].eq("S-C") & filled["date"].dt.month.eq(7)
    assert july.any() and unfilled.loc[july, "vpd"].isna().all()  # +/-inf count as missing
    # primary: a gap takes the zone median, unless the zone has no value (S-W, alone in BWk, has only rn)
    med = unfilled.loc[unfilled["koppen"].eq("Dfa"), mod.MET].median()
    assert not filled.loc[~bwk, mod.MET].isna().any().any()
    assert np.allclose(filled.loc[lys, mod.MET], med) and np.allclose(filled.loc[july, "vpd"], med["vpd"])
    assert filled.loc[bwk, ["t_avg", "vpd", "ws"]].isna().all().all() and filled.loc[bwk, "rn"].notna().all()
    # met_unfilled: gaps stay missing; recorded values are the same in both
    assert unfilled.loc[lys, mod.MET].isna().all().all()
    rec = unfilled[mod.MET].notna()
    assert filled[mod.MET][rec].equals(unfilled[mod.MET][rec])

    tm = mod.met_variants({"primary": filled, "met_unfilled": unfilled}, {})
    n_l, n_w, n_j = int(lys.sum()), int(bwk.sum()), int(july.sum())
    assert tm["stations_without"] == {"rn": ["S-L"], **dict.fromkeys(["t_avg", "vpd", "ws"], ["S-L", "S-W"])}
    assert tm["missing_station_days"] == {"rn": n_l, "t_avg": n_l + n_w, "vpd": n_l + n_w + n_j, "ws": n_l + n_w}
    p, u = tm["variants"]["primary"], tm["variants"]["met_unfilled"]
    assert p["values_imputed"] == {"rn": n_l, "t_avg": n_l, "vpd": n_l + n_j, "ws": n_l}
    assert p["values_left_missing"] == {"rn": 0, **dict.fromkeys(["t_avg", "vpd", "ws"], n_w)}
    assert u["values_imputed"] == dict.fromkeys(mod.MET, 0) and u["values_left_missing"] == tm["missing_station_days"]
    assert u["outputs"] == ["results/leakage_experiment_met_unfilled.csv",
                            "results/leakage_experiment_met_unfilled_summary.csv"]
    prov = mod.provenance(filled, mod.cropland_stations(oe), root, oe, [0], 10.0, {"0": ["S-L"]}, tm)
    json.dumps(prov, allow_nan=False)
    assert prov["tower_meteorology"] is tm


def test_zero_fill_meteorology_is_unique_only_without_met_imputation(tmp_path, monkeypatch):
    mod = _script()
    root, oe = _met_gaps(tmp_path, monkeypatch, mod, bwk=False)
    bases = {"primary": mod.build(root, oe), "met_unfilled": mod.build(root, oe, fill_met=False)}
    lys = bases["primary"]["site"].eq("S-L")
    chosen = set()
    for seed in range(4):
        miss = mod.mark_preinstall(bases["primary"], seed=seed)
        chosen |= set(bases["primary"].loc[miss, "site"])
        same = {v: lk.matches_flagged(lk.inject_zero_fill(b.assign(tower_missing=miss), "tower_missing", "et_corr",
                                                          mod.MET, "koppen"), "tower_missing", mod.MET)
                for v, b in bases.items()}
        assert not same["met_unfilled"].any()
        # with imputation every unflagged day of the station without meteorology looks zero-filled
        assert same["primary"].equals(lys & ~miss)
    assert "S-L" in chosen and len(chosen) > 1  # zero-filled with and without the station


def test_main_writes_both_variants_with_the_same_records_and_splits(tmp_path, monkeypatch):
    mod = _script()
    root, oe = _met_gaps(tmp_path, monkeypatch, mod, bwk=False)
    run = lk.run_designs
    tree = lambda seed: DecisionTreeRegressor(min_samples_leaf=5, random_state=seed)  # noqa: E731 (takes NaN)
    monkeypatch.setattr(lk, "run_designs", lambda *a, **k: run(*a, model_factory=tree, **k))
    monkeypatch.setattr(mod, "ROOT", tmp_path / "repo")  # no results/ yet
    monkeypatch.setattr("sys.argv", ["07", "--flux", str(root), "--openet", str(oe), "--seeds", "0", "1"])
    mod.main()
    out = tmp_path / "repo" / "results"
    p, u = (pd.read_csv(out / f"{stem}.csv") for stem in mod.VARIANTS.values())
    ps, us = (pd.read_csv(out / f"{stem}_summary.csv") for stem in mod.VARIANTS.values())
    assert list(p.columns) == list(u.columns) and list(ps.columns) == list(us.columns)
    assert len(p) == 2 * 2 * len(DESIGNS) and list(ps["design"]) == DESIGNS * 2
    same = ["scenario", "seed", "design", "n_test", "rmse_base", "mbe_base", "n_flagged", "flagged_mean_target",
            "flagged_mean_base", "zero_filled_fraction", "n_sites", "n_records"]
    pd.testing.assert_frame_equal(p[same], u[same])  # same records, splits and zero-filled days
    assert not p["rmse_model"].equals(u["rmse_model"])
    assert set(pd.read_csv(out / "leakage_experiment_clusters.csv")["cluster"]) == {"S-A+S-B"}
    prov = json.loads((out / "leakage_experiment_versions.json").read_text())
    assert "run_utc" not in prov  # no run time: a rerun reproduces the file byte for byte
    assert list(prov["zero_fill"]["sites_by_seed"]) == ["0", "1"]
    tm = prov["tower_meteorology"]["variants"]
    assert set(tm) == set(mod.VARIANTS) and tm["primary"]["outputs"][-1] == "results/leakage_experiment_clusters.csv"
    shared = {v: list(t["unflagged_records_sharing_zero_fill_met"].values()) for v, t in tm.items()}
    assert [s["n_records"] for s in shared["met_unfilled"]] == [0, 0]
    assert all(s["n_records"] > 0 and s["sites"] == ["S-L"] for s in shared["primary"])
