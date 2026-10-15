"""GO26 versus the Volk et al. (2023a) benchmark at the seven shared towers.

For US-ARM, US-Bi1, US-Bi2, US-Ne1, US-Ne2, US-Ne3 and US-xSL this script

1. compares GO26's mean observed ET (their Table 1, 2020-2025) with the
   benchmark's long-term mean daily ET at the same tower, weighted to the
   calendar months of GO26's study period (closed ET including the
   benchmark's gap-filled days; closed and unclosed ET on measured days
   only), and with the range of the benchmark's annual means;
2. scores the OpenET ensemble (Phase II data, Volk et al., 2023d) against
   EBR-closed, plain daily Bowen-ratio-closed and unclosed EC ET on
   overpass days, and against closed monthly ET in months with at most
   five gap-filled days (the V24 rule);
3. sets these against GO26's climate-zone results (their Table 3).

A tower's record (bm_first to bm_last, and the shares of its days that are
measured or gap-filled) runs from its first to its last day with closed ET:
the daily files start on 1 January, months before some towers began measuring.

Outputs
  results/shared_site_comparison.csv      one row per site
  results/shared_site_monthly_pairs.csv   monthly OpenET / EC pairs
  figures/fig5_shared_sites.{pdf,png}           (Fig. 5)
  figures/fig4_shared_site_daily.{pdf,png}      (Fig. 4)

Usage:
  python scripts/06_shared_site_comparison.py --flux data/raw/flux_ET_dataset \
      --openet data/raw/OpenET_PhaseII_model_ET_dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etcomment import benchmark as bm
from etcomment import closure as cl
from etcomment import metrics as mt
from etcomment import style as st

ROOT = Path(__file__).resolve().parents[1]
SHARED = ["US-ARM", "US-Bi1", "US-Bi2", "US-Ne1", "US-Ne2", "US-Ne3", "US-xSL"]
MIN_DAYS_COMPLETE_YEAR = 360


def site_daily(root: Path, site: str) -> pd.DataFrame:
    d = bm.standardize(bm.load_daily(root, site))
    le_br, f_br = cl.bowen_ratio_closure(d["le"], d["h"], d["rn"], d["g"])
    # convert BR-closed LE to ET with the file's own LE->ET factor (temperature-dependent lambda)
    d["et_br"] = le_br * (d["et_corr"] / d["le_corr"])
    d["br_factor"] = f_br
    return d


def record_coverage(d: pd.DataFrame) -> dict:
    """Record period and shares of measured and gap-filled days, from the first to the last day with closed ET."""
    rec = d.loc[d["et_corr"].first_valid_index():d["et_corr"].last_valid_index()]
    return {"bm_first": rec.index.min().date(), "bm_last": rec.index.max().date(), "bm_days": len(rec),
            "bm_measured_days": int(rec["valid_measured"].sum()),
            "bm_pct_measured": 100 * rec["valid_measured"].mean(),
            "bm_pct_gap_filled": 100 * (rec["et_gap"] & rec["et_corr"].notna()).mean()}


def mean_comparison(d: pd.DataFrame, w: pd.Series) -> dict:
    v = d["valid_measured"]
    m = d.index.month
    out = {
        "bm_mean_closed": bm.month_weighted_mean(d["et_corr"].groupby(m).mean(), w),
        "bm_mean_closed_measured": bm.month_weighted_mean(d.loc[v, "et_corr"].groupby(m[v]).mean(), w),
        "bm_mean_unclosed_measured": bm.month_weighted_mean(d.loc[v, "et"].groupby(m[v]).mean(), w),
    }
    annual = []
    for _, g in d.groupby(d.index.year):
        if g["et_corr"].notna().sum() >= MIN_DAYS_COMPLETE_YEAR:
            annual.append(bm.month_weighted_mean(g["et_corr"].groupby(g.index.month).mean(), w))
    annual = np.array([a for a in annual if np.isfinite(a)])
    out.update({"bm_n_complete_years": len(annual),
                "bm_annual_min": float(annual.min()) if len(annual) else np.nan,
                "bm_annual_max": float(annual.max()) if len(annual) else np.nan,
                "bm_annual_cv_pct": float(100 * annual.std(ddof=1) / annual.mean()) if len(annual) > 1 else np.nan})
    return out


def _scores(j: pd.DataFrame, ref: str, tag: str, out: dict) -> None:
    ok = np.isfinite(j[ref])
    out[f"n_{tag}"] = int(ok.sum())
    out[f"daily_mbe_{tag}"] = mt.mbe(j.loc[ok, "openet"], j.loc[ok, ref]) if ok.any() else np.nan
    out[f"daily_rmse_{tag}"] = mt.rmse(j.loc[ok, "openet"], j.loc[ok, ref]) if ok.any() else np.nan
    out[f"daily_r2_{tag}"] = mt.r2_cod(j.loc[ok, "openet"], j.loc[ok, ref]) if ok.sum() > 2 else np.nan


def daily_scores(d: pd.DataFrame, od: pd.Series) -> dict:
    """OpenET on overpass days that are measured days.

    ``closed``/``unclosed``: all such days.  ``ebr_common``/``br_common``: only the
    days on which plain daily Bowen-ratio closure can be computed, so the two
    closures are scored on identical days.  ``br_screened``: those days with the
    Bowen-ratio factor inside 0.5-2.
    """
    j = d.join(od.rename("openet"), how="inner")
    j = j[j["valid_measured"] & j["openet"].notna()]
    out = {"n_overpass_days": len(j)}
    _scores(j, "et_corr", "closed", out)
    _scores(j, "et", "unclosed", out)
    common = j[np.isfinite(j["et_br"])]
    _scores(common, "et_corr", "ebr_common", out)
    _scores(common, "et_br", "br_common", out)
    f = common["br_factor"]
    _scores(common[(f >= 0.5) & (f <= 2.0)], "et_br", "br_screened", out)
    return out


def monthly_pairs(d: pd.DataFrame, om: pd.Series, site: str) -> pd.DataFrame:
    g = bm.complete_months(d)  # calendar-complete months with <= 5 gap-filled days (V24 rule)
    j = g.join(om.rename("openet"), how="inner").dropna(subset=["openet"])
    j["site"] = site
    return j.reset_index().rename(columns={"index": "month"})


def monthly_scores(p: pd.DataFrame) -> dict:
    if len(p) == 0:
        return {"n_months": 0, "monthly_mbe_mm_month": np.nan, "monthly_mbe_mm_d": np.nan,
                "monthly_rmse_mm_month": np.nan, "monthly_r2": np.nan}
    return {"n_months": len(p),
            "monthly_mbe_mm_month": mt.mbe(p["openet"], p["et_corr"]),
            "monthly_mbe_mm_d": float(((p["openet"] - p["et_corr"]) / p["n_days"]).mean()),
            "monthly_rmse_mm_month": mt.rmse(p["openet"], p["et_corr"]),
            "monthly_r2": mt.r2_cod(p["openet"], p["et_corr"])}


def build(root: Path, openet: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    go = pd.read_csv(ROOT / "config" / "go26_sites.csv").set_index("site")
    zones = pd.read_csv(ROOT / "config" / "go26_table3.csv").set_index("zone")
    oe_d = bm.load_openet_phase2(openet, "daily")
    oe_m = bm.load_openet_phase2(openet, "monthly")
    rows, pairs = [], []
    for s in SHARED:
        d = site_daily(root, s)
        g = go.loc[s]
        w = bm.period_month_days(g["period_start"], g["period_end"])
        od = oe_d[oe_d["site"] == s].set_index("date")["Ensemble"]
        om = oe_m[oe_m["site"] == s].set_index("date")["Ensemble"]
        p = monthly_pairs(d, om, s)
        pairs.append(p)
        gs = np.isin(d.index.month, range(4, 11))
        ratio = (d["le"] + d["h"]) / (d["rn"] - d["g"])
        r = {"site": s, "go26_zone": g["koppen_go26"],
             "go26_period": f"{g['period_start']} to {g['period_end']}",
             **record_coverage(d),
             "bm_closure_ratio_apr_oct_median": float(ratio[gs & d["valid_measured"]].median()),
             "go26_mean_obs": float(g["mean_obs_et_mm_d"])}
        r.update(mean_comparison(d, w))
        r["go26_minus_bm_mean"] = r["go26_mean_obs"] - r["bm_mean_closed"]
        r["go26_over_bm_mean"] = r["go26_mean_obs"] / r["bm_mean_closed"]
        r["go26_below_bm_annual_min"] = bool(r["go26_mean_obs"] < r["bm_annual_min"])
        r.update(daily_scores(d, od))
        r.update(monthly_scores(p))
        z = zones.loc[g["koppen_go26"]]
        r["go26_zone_openet_bias"] = float(z["openet_bias"])
        r["go26_zone_openet_r2"] = float(z["openet_r2"])
        bm_bias = r["monthly_mbe_mm_d"] if np.isfinite(r["monthly_mbe_mm_d"]) else r["daily_mbe_closed"]
        r["bias_gap_go26_minus_bm"] = r["go26_zone_openet_bias"] - bm_bias
        rows.append(r)
    return pd.DataFrame(rows), pd.concat(pairs, ignore_index=True)


# ----------------------------------------------------------------------------- figures
# Figures are 6.5 in wide and inserted 1:1 in the manuscript (text >= 7 pt; see
# etcomment.style); axes are placed by their rectangles in inches.

def _inch_axes(fig, left: float, bottom: float, width: float, height: float):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([left / fw, bottom / fh, width / fw, height / fh])


def _site_axis(ax, sites, zones, brackets: bool = True):
    x = np.arange(len(sites))
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("US-", "") for s in sites])
    ax.set_xlim(-0.6, len(sites) - 0.4)
    if not brackets:
        return x
    # zone brackets under the tick labels, offset in points so they do not depend on the axes height
    fig = ax.get_figure()

    def below(pt: float):
        return ax.get_xaxis_transform() + matplotlib.transforms.ScaledTranslation(
            0, -pt / 72, fig.dpi_scale_trans)

    i = 0
    while i < len(sites):
        j = i
        while j + 1 < len(sites) and zones[j + 1] == zones[i]:
            j += 1
        ax.plot([i - 0.3, j + 0.3], [0, 0], transform=below(18), color=st.INK2, lw=0.6, clip_on=False,
                solid_capstyle="butt")
        ax.text((i + j) / 2, 0, zones[i], transform=below(20.5), ha="center", va="top", fontsize=7.5,
                color=st.INK2)
        i = j + 1
    return x


def fig5(s: pd.DataFrame, pairs: pd.DataFrame) -> None:
    """Fig. 5: mean ET, OpenET bias and monthly scatter at the seven shared towers.

    Panels a (top) and b (bottom) share the site axis on the left; the monthly
    scatter c is square on the right (one row of three panels is too cramped at
    6.5 in).
    """
    st.nature(8)
    W, H = st.FULL_WIDTH_IN, 3.95
    fig = plt.figure(figsize=(W, H))
    left, wab, hab = 0.68, 2.98, 1.42
    a = _inch_axes(fig, left, H - 0.27 - hab, wab, hab)
    b = _inch_axes(fig, left, 0.52, wab, hab)
    cl_ = left + wab + 0.7
    wc = W - 0.08 - cl_
    c = _inch_axes(fig, cl_, H - 0.27 - wc, wc, wc)
    sites, zones = list(s["site"]), list(s["go26_zone"])

    # a: mean observed ET (benchmark markers left of the tick, GO26 right)
    x = _site_axis(a, sites, zones, brackets=False)
    a.tick_params(labelbottom=False)
    xl, xr = x - 0.13, x + 0.13
    a.vlines(xl, s["bm_annual_min"], s["bm_annual_max"], color=st.INK, lw=1.0, zorder=2,
             label="Benchmark, range of annual means")
    a.scatter(xl, s["bm_mean_closed"], s=24, color=st.INK, zorder=3, label="Benchmark, closed ET")
    a.scatter(xl, s["bm_mean_unclosed_measured"], s=24, facecolor="white", edgecolor=st.INK, lw=0.8,
              zorder=3, label="Benchmark, unclosed ET")
    a.scatter(xr, s["go26_mean_obs"], s=32, marker="D", color=st.ORANGE, edgecolor="white", lw=0.5,
              zorder=4, label="GO26 (Table 1)")
    a.set_ylim(0, 4.85)
    a.set_yticks([0, 1, 2, 3, 4])
    a.set_ylabel("Mean observed ET\n(mm d$^{-1}$)")
    a.legend(loc="upper right", ncol=1, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2,
             handlelength=1.4)

    # b: OpenET ensemble bias (daily left of the tick, monthly right)
    x = _site_axis(b, sites, zones)
    b.axhline(0, color=st.AXIS, lw=0.6, zorder=1)
    i = 0
    first = True
    while i < len(sites):
        j = i
        while j + 1 < len(sites) and zones[j + 1] == zones[i]:
            j += 1
        val = s["go26_zone_openet_bias"].iloc[i]
        b.hlines(val, i - 0.35, j + 0.35, color=st.ORANGE, lw=2.0, zorder=2,
                 label="GO26, zone mean (Table 3)" if first else None)
        first = False
        i = j + 1
    b.scatter(x - 0.13, s["daily_mbe_closed"], s=24, color=st.INK, zorder=3,
              label="Benchmark, daily (overpass days)")
    mm = s["monthly_mbe_mm_d"]
    b.scatter((x + 0.13)[mm.notna().to_numpy()], mm[mm.notna()], s=24, marker="s", facecolor="white",
              edgecolor=st.INK, lw=0.8, zorder=3, label="Benchmark, monthly")
    for xi, (n, v) in zip(x, zip(s["n_overpass_days"], s["daily_mbe_closed"])):
        if n < 30:
            b.annotate(f"n = {n}", xy=(xi - 0.13, v), xytext=(0, -5), textcoords="offset points",
                       fontsize=7, color=st.INK2, ha="center", va="top")
    b.set_ylim(-1.9, 1.4)
    b.set_yticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0])
    b.set_ylabel("OpenET ensemble bias\n(mm d$^{-1}$)")
    b.legend(loc="lower left", ncol=1, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2,
             handlelength=1.4)

    # c: monthly scatter (axis limit from the data, so that every month counted in the text is drawn)
    lim = float(np.ceil(pairs[["et_corr", "openet"]].max().max() / 20) * 20)
    c.plot([0, lim], [0, lim], color=st.INK2, lw=0.75, ls="--", zorder=1)
    c.scatter(pairs["et_corr"], pairs["openet"], s=6, color=st.INK, lw=0, alpha=0.55, zorder=2)
    c.set_xlim(0, lim); c.set_ylim(0, lim)
    c.set_xticks(np.arange(0, lim + 1, 100)); c.set_yticks(np.arange(0, lim + 1, 100))
    c.set_xlabel("Closed EC ET (mm month$^{-1}$)")
    c.set_ylabel("OpenET ensemble (mm month$^{-1}$)")
    r2 = mt.r2_cod(pairs["openet"], pairs["et_corr"])
    mbe = mt.mbe(pairs["openet"], pairs["et_corr"])
    sign = "+" if mbe >= 0 else "\u2212"
    c.text(0.03, 0.98, f"Benchmark, {pairs['site'].nunique()} sites,\n{len(pairs)} months\nR$^2$ = {r2:.2f}\n"
           f"MBE {sign}{abs(mbe):.1f} mm month$^{{-1}}$",
           transform=c.transAxes, va="top", fontsize=7.5, color=st.INK, linespacing=1.2)
    c.text(0.98, 0.03, "GO26 OpenET baseline:\nmonthly R$^2$ = 0.14", transform=c.transAxes, ha="right",
           va="bottom", fontsize=7.5, color=st.ORANGE, linespacing=1.2)
    fig.align_ylabels([a, b])
    for ax, lab, dx in ((a, "a", 0.66), (b, "b", 0.66), (c, "c", 0.62)):
        pos = ax.get_position()
        fig.text(pos.x0 - dx / W, pos.y1 + 0.07 / H, lab, fontsize=10, fontweight="bold", va="bottom",
                 ha="left", color=st.INK)
    st.check_text_size(fig)
    out = ROOT / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig5_shared_sites.{ext}", dpi=600, metadata=st.PDF_METADATA if ext == "pdf" else None)
    plt.close(fig)


def fig4(root: Path, openet: Path, s: pd.DataFrame) -> None:
    """Fig. 4: daily benchmark ET at the seven shared towers, 2016-2020.

    Each panel's header sits above its axes so that it hides no data.
    """
    st.nature(8)
    oe_d = bm.load_openet_phase2(openet, "daily")
    n = len(SHARED)
    W = st.FULL_WIDTH_IN
    top, head, h, gap, bottom = 0.3, 0.17, 0.74, 0.04, 0.26
    H = top + n * (head + h) + (n - 1) * gap + bottom
    fig = plt.figure(figsize=(W, H))
    axes = []
    for k in range(n):
        y1 = H - top - k * (head + h + gap) - head
        axes.append(_inch_axes(fig, 0.42, y1 - h, W - 0.42 - 0.08, h))
    x0, x1 = pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31")
    series = {}
    for site in SHARED:
        d = bm.standardize(bm.load_daily(root, site)).loc[x0:x1]
        od = oe_d[(oe_d["site"] == site) & oe_d["date"].between(x0, x1)].set_index("date")["Ensemble"]
        series[site] = (d["et_corr"].where(d["valid_measured"]), d["et_corr"].where(d["et_gap"]), od)
    # one y scale for all panels that shows every value drawn (some measured days are below -1 mm/d)
    vals = pd.concat([v for t in series.values() for v in t])
    ylim = (min(-0.8, np.floor((vals.min() - 0.1) * 5) / 5), max(10.5, np.ceil((vals.max() + 0.1) * 2) / 2))
    rows = s.set_index("site")
    for ax, site in zip(axes, SHARED):
        meas, gap_, od = series[site]
        ax.plot(gap_.index, gap_, color=st.MUTED, lw=0.75, zorder=2, label="Benchmark, gap-filled (EToF × ETo)")
        ax.plot(meas.index, meas, color=st.INK, lw=0.75, zorder=3, label="Benchmark, measured EBR-closed ET")
        ax.scatter(od.index, od.values, s=8, color=st.BLUE, lw=0, zorder=4, label="OpenET ensemble, overpass days")
        ax.axvline(pd.Timestamp("2020-01-01"), color=st.ORANGE, lw=0.75, ls="--", zorder=1)
        row = rows.loc[site]
        # one decimal, so that 99.8% does not print as 100%
        ax.set_title(f"{site} ({row['go26_zone']}), {row['bm_first'].year}\u2013{row['bm_last'].year}: "
                     f"{row['bm_pct_measured']:.1f}% of record days measured",
                     loc="left", fontsize=7.5, pad=2.5, color=st.INK)
        ax.set_ylim(*ylim)
        ax.set_yticks([0, 5, 10])
        ax.set_xlim(x0, x1)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False)
    axes[0].annotate("GO26 study periods begin", xy=(pd.Timestamp("2020-01-01"), 1.0),
                     xycoords=("data", "axes fraction"), xytext=(-3, 2.5), textcoords="offset points",
                     ha="right", va="bottom", color=st.ORANGE, fontsize=7.5)
    axes[n // 2].set_ylabel("ET (mm d$^{-1}$)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    h_, l_ = axes[0].get_legend_handles_labels()
    order = [1, 0, 2]
    fig.legend([h_[i] for i in order], [l_[i] for i in order], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0), markerscale=1.5, handlelength=1.6, handletextpad=0.5,
               columnspacing=1.4, borderaxespad=0.3)
    st.check_text_size(fig)
    out = ROOT / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig4_shared_site_daily.{ext}", dpi=600,
                    metadata=st.PDF_METADATA if ext == "pdf" else None)
    plt.close(fig)


def main() -> None:
    st.rerun_with_fixed_hash_seed(__name__)  # byte-reproducible PDFs (etcomment.style)
    ap = argparse.ArgumentParser()
    ap.add_argument("--flux", default=str(ROOT / "data/raw/flux_ET_dataset"))
    ap.add_argument("--openet", default=str(ROOT / "data/raw/OpenET_PhaseII_model_ET_dataset"))
    args = ap.parse_args()
    root = bm.find_root(Path(args.flux))
    openet = Path(args.openet)
    s, pairs = build(root, openet)
    (ROOT / "results").mkdir(exist_ok=True)
    s.to_csv(ROOT / "results" / "shared_site_comparison.csv", index=False)
    pairs.to_csv(ROOT / "results" / "shared_site_monthly_pairs.csv", index=False)
    (ROOT / "figures").mkdir(exist_ok=True)
    fig4(root, openet, s)
    fig5(s, pairs)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(s.round(3).T.to_string())


if __name__ == "__main__":
    main()
