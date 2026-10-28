"""Figures 1 and 2 of the comment (6.5 in wide, inserted 1:1; see etcomment.style).

Fig. 1  US-UTJ daily ET re-plotted from the digitized GO26 Fig. S2.
Fig. 2  (a) Apparent OpenET bias created by zero-filled observations.
        (b) GO26 mean observed ET (annual equivalent) versus precipitation, and
            current OpenET over the same period at the four irrigated sites in
            dry climates.
        (c) GO26 Table 1 mean / OpenET mean versus LE availability (fraction of
            the period's half-hours with LE in the AmeriFlux BASE files): means
            that count missing data as zero scale with availability.
        (d) OpenET ensemble bias by season as GO26 report it (test set, their
            Table 4) and for current OpenET against the benchmark's closed ET;
            markers at the GO26 bars give current OpenET at GO26's towers against
            their observed ET rebuilt from the BASE files (scripts/10, all zones):
            as GO26's Table 1, missing data counted as zero (diamonds), and on
            valid days, gap-filled and closed with the energy-balance ratio (dots).
Table 1 figure (fig_table1_*; not yet numbered in the comment): GO26 Table 1 mean
        observed ET against the site means rebuilt from the AmeriFlux BASE files
        (scripts/10), one point per site, in a 2 x 2 grid:
        (a) missing values set to zero: daily sums of the available half-hours,
            missing half-hours and days counted as zero, over the calendar days
            of the period;
        (b) missing values set to NA: the mean of the valid half-hours on days
            with at least half of them (GO26's stated 50% filter), over those days;
        (c) gaps filled: as (b), with the missing half-hours of those days filled
            from the mean diurnal cycle of a centred 15-day window;
        (d) as (c), closed with the Bowen-ratio method as GO26 state it (Twine et
            al., 2000), flux-data-qaqc's 'br' correction (version in
            environment.yml): the day's factor, unscreened, on days with the
            energy-balance terms, leaving out days whose closed LE is <= -100 or
            >= 850 W m-2 (scripts/10);
            (a)-(c) have no closure.
OpenET figure (fig_openet_*; Figure 2 of the comment): the same EC site means against the current OpenET
        ensemble averaged over the same days (scripts/10, go26_reconstructed_openet_sites.csv), in the layout of
        the Table 1 figure: (a) the zero-filled means against OpenET over every day of the study period; (b)-(d)
        the valid-day, gap-filled and gap-filled Bowen-ratio closed means against OpenET over the days on which
        every variant has a value (one day set per tower, so 21 towers: US-WT1 has no net radiation).  Each panel
        gives the OpenET bias (mean of OpenET minus EC ET over the towers), the RMS difference and Pearson r.
        The OpenET attribution is in the figure caption of the comment, not on the figure.

Inputs: results/digitized_US-UTJ.csv, results/digitized_summary.json (scripts/04),
config/go26_sites.csv, results/ameriflux_le_availability.csv (scripts/11),
results/openet_api_go26_sites.csv and openet_api_benchmark_summary.csv (scripts/12;
OpenET API, collection 2.1) and, optionally, results/go26_reconstructed_openet_bias.csv
(scripts/10, which needs the AmeriFlux BASE files; without it Fig. 2d has no markers)
and results/go26_observed_et_reconstruction.csv (scripts/10; without it the Table 1
figure is not drawn).
OpenET values carry the attribution the OpenET API Terms of Service require
(OpenET, https://openetdata.org).
Outputs: figures/fig1_*.{pdf,png}, figures/fig2_*.{pdf,png}, figures/fig_table1_*.{pdf,png}.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

from etcomment import openet_compare as oc
from etcomment import style as st

logging.getLogger("fontTools").setLevel(logging.ERROR)  # font-subsetting chatter when writing PDFs
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
DAYS_PER_YEAR = 365.25
W = st.FULL_WIDTH_IN
# outputs of scripts/11 and scripts/12 that Fig. 2b-d plot
OPENET_RESULTS = ("ameriflux_le_availability.csv", "openet_api_go26_sites.csv",
                  "openet_api_benchmark_summary.csv")
CREDIT = "OpenET data: OpenET, https://openetdata.org"
# marker categories of the GO26 towers in Fig. 2b and c: face colour, legend label
CATS = {"flagged": (st.ORANGE, "Irrigated, dry climate"), "shared": (st.BLUE, "In Volk et al. benchmark"),
        "other": ("white", "Other GO26 site")}
# Fig. 2d markers: all-zone season rows (month_set, variant) of the OpenET bias table of scripts/10 (optional)
RECON_BIAS = "go26_reconstructed_openet_bias.csv"
RECON_SERIES = {"zero_filled": ("all_months", "zero_filled_table1"), "ebr": ("common_valid_months", "ebr_mdv")}
# their marker, size (pt^2), face, edge, edge width and legend label (under the title GO26_TOWERS_TITLE); a white
# ring of RING_PT outside the edge sets them off where they sit on a bar or on the zero line
GO26_TOWER_MARKERS = {"zero_filled": ("D", 20, "white", st.ORANGE, 1.0, "rebuilt obs. (zero-filled)"),
                      "ebr": ("o", 20, st.INK, st.INK, 0.0, "valid days, EBR-closed")}
GO26_TOWERS_TITLE = "GO26 towers, OpenET 2.1 vs"
RING_PT = 0.8
SEASON_CODE = {"winter": "DJF", "spring": "MAM", "summer": "JJA", "fall": "SON"}
# Table 1 figure: site means of scripts/10 (all without closure) against GO26's Table 1
RECON_SITES = "go26_observed_et_reconstruction.csv"
TABLE1_REF = "go26_table1_mean_obs_et"
TABLE1_SERIES = {  # column of scripts/10 (one panel each): marker face, panel title
    "mean_et_hh_sum_zero_filled_unclosed": (st.ORANGE, "Missing values set to zero"),
    "mean_et_unclosed_valid": (st.BLUE, "Missing values set to NA"),
    "mean_et_unclosed_mdv": (st.AQUA, "Missing values gap-filled"),
    "mean_et_br_fdq_mdv": (st.INK, "Gap-filled and closure-corrected"),
}
TABLE1_TOL = 0.04  # mm/d; Table 1 is rounded to 0.01 mm/d
# sites labelled in the zero-filled panel (outside TABLE1_TOL): label position (mm/d) and alignment
TABLE1_LABELS = {"US-UTV": (1.08, 0.80, "left"), "US-UTD": (0.12, 1.22, "left")}
# OpenET figure (Figure 2 of the comment): site means of scripts/10 over the same days as OpenET, one panel per
# (day_set, column) as the Table 1 figure: marker face, panel title
RECON_OPENET = "go26_reconstructed_openet_sites.csv"
OPENET_SERIES = {
    ("calendar", "obs_zero_filled_mm_d"): (st.ORANGE, "Missing values set to zero"),
    ("common", "obs_unclosed_mm_d"): (st.BLUE, "Missing values set to NA"),
    ("common", "obs_unclosed_mdv_mm_d"): (st.AQUA, "Missing values gap-filled"),
    ("common", "obs_br_fdq_mdv_mm_d"): (st.INK, "Gap-filled and closure-corrected"),
}


def zero_spans(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    z = df["obs_is_zero"].fillna(False).astype(bool)
    spans, start, prev = [], None, None
    for d, v in z.items():
        if v and start is None:
            start = d
        if not v and start is not None:
            spans.append((start, prev, (prev - start).days + 1)); start = None
        prev = d
    if start is not None:
        spans.append((start, prev, (prev - start).days + 1))
    return spans


def _axes(fig, left: float, bottom: float, width: float, height: float):
    """Axes placed by its rectangle in inches."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([left / fw, bottom / fh, width / fw, height / fh])


def _save(fig, name: str) -> None:
    st.check_text_size(fig)
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600, metadata=st.PDF_METADATA if ext == "pdf" else None)
    plt.close(fig)


def fig1() -> None:
    st.nature(8)
    df = pd.read_csv(ROOT / "results" / "digitized_US-UTJ.csv", parse_dates=["date"]).set_index("date")
    df = df.loc["2024-01-01":"2024-12-31"]
    fig = plt.figure(figsize=(W, 3.0))
    ax = _axes(fig, 0.42, 0.56, W - 0.42 - 0.1, 1.98)
    for a, b, n in zero_spans(df):
        ax.axvspan(a - pd.Timedelta(hours=12), b + pd.Timedelta(hours=12), color=st.ORANGE,
                   alpha=0.12, lw=0, zorder=0)
    ax.axhline(0, color=st.AXIS, lw=0.6, zorder=1)
    ax.plot(df.index, df["openet"], color=st.BLUE, lw=1.0, zorder=3, label="OpenET ensemble")
    ax.plot(df.index, df["xgb"], color=st.AQUA, lw=1.0, zorder=3, label="XGBoost (GO26)")
    nz = df[(~df["obs_is_zero"]) & df["obs"].notna()]
    z = df[df["obs_is_zero"]]
    ax.scatter(nz.index, nz["obs"], s=9, color=st.INK, lw=0, zorder=4, label="Observed ET, non-zero")
    ax.scatter(z.index, z["obs"], s=9, color=st.ORANGE, lw=0, zorder=5, label="Observed ET = 0.00")
    ax.set_xlim(pd.Timestamp("2023-12-28"), pd.Timestamp("2025-01-04"))
    ax.set_ylim(-0.6, 5.6)
    ax.set_ylabel("ET (mm d$^{-1}$)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    n_obs = int(df["obs"].notna().sum())
    n_zero = int(df["obs_is_zero"].sum())
    ax.annotate(f"US-UTJ, 2024: {n_zero} of {n_obs} plotted observations are exactly zero",
                xy=(0, 1), xycoords="axes fraction", xytext=(0, 19), textcoords="offset points",
                fontsize=8.5, fontweight="bold", va="bottom", ha="left", color=st.INK)
    ax.annotate("Re-plotted from the digitized Fig. S2 of Golkar and de Oliveira (2026); "
                "shading marks runs of exact-zero observed ET",
                xy=(0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                fontsize=7.5, va="bottom", ha="left", color=st.INK2)
    # annotate the runs >= 15 days
    for a, b, n in zero_spans(df):
        if n >= 15:
            mid = a + (b - a) / 2
            ax.text(mid, 5.22, f"{n} d", ha="center", va="center", fontsize=7.5, color=st.ORANGE)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, handlelength=1.8,
              columnspacing=1.8, markerscale=1.5, borderaxespad=0)
    _save(fig, "fig1_US-UTJ_zero_runs")


def _openet_results() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """GO26 towers (scripts/12) with LE availability (scripts/11), benchmark biases and their period."""
    missing = [f"results/{f}" for f in OPENET_RESULTS if not (RES / f).exists()]
    if missing:
        raise FileNotFoundError(f"Fig. 2 needs {', '.join(missing)}: run scripts/11_ameriflux_availability.py "
                                "and then scripts/12_openet_api_analysis.py first")
    t = pd.read_csv(RES / "openet_api_go26_sites.csv").drop(columns="le_avail_period", errors="ignore")
    av = pd.read_csv(RES / "ameriflux_le_availability.csv", usecols=["site", "le_avail_period"])
    t = t.merge(av, on="site", how="left")
    if t["le_avail_period"].isna().all():
        raise ValueError("results/ameriflux_le_availability.csv has no LE availability: rerun "
                         "scripts/11_ameriflux_availability.py and then scripts/12_openet_api_analysis.py")
    bench = pd.read_csv(RES / "openet_api_benchmark_summary.csv")
    a = bench[(bench["model"] == "ensemble") & (bench["stratum_type"] == "all")].iloc[0]
    return t, bench, f"{str(a['first_month'])[:4]}–{str(a['last_month'])[:4]}"


def _reconstructed_bias() -> pd.DataFrame | None:
    """OpenET bias by season at GO26's towers (scripts/10, all zones; mm/d), one column per RECON_SERIES.

    None where results/go26_reconstructed_openet_bias.csv is missing (scripts/10 needs the AmeriFlux BASE
    files) or holds none of these rows: Fig. 2d is then drawn without the markers.  Withheld values are NaN.
    """
    f = RES / RECON_BIAS
    if not f.exists():
        print(f"results/{RECON_BIAS} not found (scripts/10): Fig. 2d drawn without the GO26-tower markers")
        return None
    b = pd.read_csv(f, comment="#")
    need = {"koppen_go26", "season", "month_set", "variant", "bias_mm_d"}
    if not need <= set(b.columns):
        raise ValueError(f"results/{RECON_BIAS} lacks {sorted(need - set(b.columns))}: rerun scripts/10")
    bias = {(r.season, r.month_set, r.variant): r.bias_mm_d for r in b[b["koppen_go26"] == "all"].itertuples()}
    rec = pd.DataFrame({k: [bias.get((SEASON_CODE[s], *mv), np.nan) for s in oc.SEASONS]
                        for k, mv in RECON_SERIES.items()}, index=oc.SEASONS, dtype=float)
    if rec.isna().all(axis=None):
        print(f"results/{RECON_BIAS} has no all-zone season bias: Fig. 2d drawn without the GO26-tower markers")
        return None
    return rec


def _category(sites: pd.DataFrame) -> pd.Series:
    """Marker category of each GO26 tower (index: site), as in Table 2 and the benchmark station list."""
    c = np.where(sites["site"].isin(oc.IRRIGATED_DRY), "flagged",
                 np.where(sites["in_volk_benchmark"], "shared", "other"))
    return pd.Series(c, index=sites["site"].to_numpy())


def _label(ax, text: str, xy, dx: float, dy: float, ha: str = "left", va: str = "center",
           leader: bool = False) -> None:
    """Site label at an offset (points) from its marker, optionally with a leader line."""
    ax.annotate(text, xy=xy, xytext=(dx, dy), textcoords="offset points", fontsize=7.5, ha=ha, va=va,
                color=st.INK, arrowprops=dict(arrowstyle="-", lw=0.6, color=st.INK, shrinkA=0, shrinkB=3)
                if leader else None)


def _category_handles(ms: float = 5.2) -> list[Line2D]:
    return [Line2D([], [], marker="o", ls="", mfc=fc, mec=st.INK, mew=0.6, ms=ms, label=lab)
            for fc, lab in CATS.values()]


def _panel_zero_fill(ax, summ: dict, up) -> None:
    """(a) bias added by zero-filling a fraction f of observations whose true mean is E."""
    f = np.linspace(0, 0.6, 61)
    shades = {1.0: 0.45, 2.0: 0.72, 3.0: 1.0}
    ax.axhspan(0.364, 1.103, color="#e8e7e1", lw=0, zorder=0)
    ax.text(0.012, 1.13, "OpenET bias by climate zone\nreported by GO26", fontsize=7.5,
            color=st.INK2, va="bottom", ha="left", linespacing=1.15)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.9)
    for E, a in shades.items():
        ax.plot(f, f * E, color=st.BLUE, alpha=a, lw=1.2, zorder=2)
        # direct label along each line (rotation follows the line on the page)
        ax.text(0.24, 0.24 * E, f"E = {E:.0f} mm d$^{{-1}}$", fontsize=7.5, color=st.INK2,
                rotation=np.degrees(np.arctan(E)), transform_rotates_text=True,
                rotation_mode="anchor", ha="center", va="bottom", transform=ax.transData + up,
                zorder=3)
    x0 = summ["zero_fraction"]
    y0 = summ["openet_all"]["mbe"] - summ["openet_nonzero"]["mbe"]
    ax.scatter([x0], [y0], s=34, color=st.ORANGE, edgecolor=st.INK, lw=0.6, zorder=4)
    ax.annotate("US-UTJ (digitized):\nOpenET MBE with zeros\nminus MBE without zeros", xy=(x0, y0),
                xytext=(0.455, 0.16), fontsize=7.5, color=st.INK, ha="center", va="center",
                linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=st.INK, shrinkA=2, shrinkB=3.5))
    ax.set_xlabel("Fraction of observations set to zero, $f$")
    ax.set_ylabel("Bias added to every model (mm d$^{-1}$)")
    ax.set_title("Zero-filled days raise MBE = mean(P − O) by $f \\times E$", loc="left", pad=7,
                 color=st.INK)


def _panel_et_vs_p(ax, sites: pd.DataFrame, cat: pd.Series, t: pd.DataFrame, up) -> None:
    """(b) GO26 site-mean ET (annual equivalent) against precipitation, with OpenET at the flagged sites."""
    s = sites.set_index("site")
    p, et = s["map_mm"], s["mean_obs_et_mm_d"] * DAYS_PER_YEAR
    oe = t.set_index("site")["oe_mm_d"] * DAYS_PER_YEAR  # OpenET ensemble, GO26 study period
    fl = list(oc.IRRIGATED_DRY)
    ax.set_xlim(-60, 1300)  # room left of the flagged sites for their labels
    ax.set_ylim(0, 1150)
    ax.plot([0, 1150], [0, 1150], color=st.INK2, lw=0.75, ls="--", zorder=1)
    ax.text(1000, 1000, "ET = P", fontsize=7.5, color=st.INK2, ha="right", va="bottom", rotation=45,
            transform_rotates_text=True, rotation_mode="anchor", transform=ax.transData + up)
    ax.vlines(p[fl], et[fl], oe[fl], color=st.ORANGE, lw=0.8, zorder=2)
    for c in ("other", "shared", "flagged"):
        k = cat.index[cat == c]
        ax.scatter(p[k], et[k], s=26, facecolor=CATS[c][0], edgecolor=st.INK, lw=0.6, zorder=3)
    ax.scatter(p[fl], oe[fl], marker="D", s=20, facecolor="white", edgecolor=st.ORANGE, lw=1.0, zorder=4)
    for site in fl:  # flagged sites are labelled at their OpenET value, the top of each line
        _label(ax, site, (p[site], oe[site]), -5, 0, ha="right")
    for site, (dx, dy, ha, va) in {"US-Bi1": (5, 0, "left", "center"), "US-Bi2": (5, 0, "left", "center"),
                                   "US-xSL": (0, -6, "center", "top"),
                                   "US-WT1": (-2, 6, "center", "bottom")}.items():
        _label(ax, site, (p[site], et[site]), dx, dy, ha=ha, va=va)
    ax.set_xlabel("Mean annual precipitation, P (mm yr$^{-1}$; GO26 Table 1)")
    ax.set_ylabel("Mean ET × 365.25 (mm yr$^{-1}$)")
    k = oe[fl] / p[fl]
    ax.set_title(f"Irrigated, dry climate: GO26 ET ≈ P, OpenET {np.floor(k.min()):.0f}–{np.ceil(k.max()):.0f} × P",
                 loc="left", pad=7, color=st.INK)
    hs = _category_handles() + [Line2D([], [], marker="D", ls="", mfc="white", mec=st.ORANGE, mew=1.0, ms=4.6,
                                       label="OpenET collection 2.1,\nsame period")]
    ax.legend(handles=hs, loc="lower right", handlelength=0.8, handletextpad=0.4, labelspacing=0.35,
              borderpad=0.3, borderaxespad=0.3)


# site labels for panel c: offset (points), alignment and whether a leader line joins label and marker.
# The US-NSb and US-xSL markers overlap, so their leaders leave in different directions (up / right).
LABELS_C = {"US-UTV": (5, 0, "left", "center", False), "US-UTJ": (-30, 20, "right", "center", True),
            "US-NSb": (-10, 34, "right", "center", True), "US-UTD": (0, -6, "center", "top", False),
            "US-xSL": (30, -2, "left", "center", True), "US-WT1": (5, 0, "left", "center", False)}


def _panel_availability(ax, t: pd.DataFrame, cat: pd.Series) -> None:
    """(c) GO26 Table 1 mean / OpenET mean against LE availability, with the zero-fill fit through 0."""
    a = t.dropna(subset=["le_avail_period", "ratio_t1_oe"])
    c = cat.reindex(a["site"]).to_numpy()
    fit = oc.fit_through_origin(a["ratio_t1_oe"], a["le_avail_period"])
    r = oc.correlation(a["le_avail_period"], a["ratio_t1_oe"])["pearson_r"]
    ax.plot([0, 1], [0, fit["slope"]], color=st.INK2, lw=0.9, zorder=1)
    for k in ("other", "shared", "flagged"):
        d = a[c == k]
        ax.scatter(d["le_avail_period"], d["ratio_t1_oe"], s=26, facecolor=CATS[k][0], edgecolor=st.INK,
                   lw=0.6, zorder=3)
    xy = a.set_index("site")[["le_avail_period", "ratio_t1_oe"]]
    for s, (dx, dy, ha, va, leader) in LABELS_C.items():
        if s in xy.index:
            _label(ax, s, tuple(xy.loc[s]), dx, dy, ha=ha, va=va, leader=leader)
    ne = xy.loc[xy.index.intersection(["US-Ne1", "US-Ne2", "US-Ne3"])]
    if len(ne):  # one label left of the three Mead towers
        _label(ax, "US-Ne1–Ne3", (ne["le_avail_period"].min(), ne["ratio_t1_oe"].mean()), -6, 3, ha="right")
    ax.text(0.97, 0.04, f"r = {r:.2f}\nfit: GO26/OpenET = {fit['slope']:.2f} × availability",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=st.INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("LE availability (fraction of half-hours in period)")
    ax.set_ylabel("GO26 Table 1 mean / OpenET mean")
    ax.set_title("GO26/OpenET scales with LE availability", loc="left", pad=7, color=st.INK)
    hs = _category_handles() + [Line2D([], [], color=st.INK2, lw=0.9, label="Zero-fill expectation (fit)")]
    ax.legend(handles=hs, loc="upper left", handlelength=1.6, handletextpad=0.4, labelspacing=0.35,
              borderpad=0.3, borderaxespad=0.3)


def _panel_seasons(ax, bench: pd.DataFrame, period: str, rec: pd.DataFrame | None = None) -> None:
    """(d) OpenET bias by season: GO26 Table 4 against current OpenET vs the benchmark's closed ET and, at the
    GO26 bars, current OpenET at GO26's towers against their rebuilt observed ET (``rec``, _reconstructed_bias;
    GO26_TOWER_MARKERS, without value labels; left out when ``rec`` is None)."""
    e = bench[bench["model"] == "ensemble"].set_index(["stratum_type", "stratum"])

    def mbe(kind: str, key: str) -> float:
        return float(e.loc[(kind, key), "mbe_mm_d"]) if (kind, key) in e.index else np.nan

    x = np.arange(len(oc.SEASONS))
    w = 0.3
    xg, xb, xk = x - w - 0.02, x, x + w / 2 + 0.17  # GO26 bar (and GO26-tower markers), benchmark bar, Koppen groups
    go = np.array([oc.GO26_SEASON_BIAS[s] for s in oc.SEASONS])
    bmv = np.array([mbe("season", s) for s in oc.SEASONS])
    ax.axhline(0, color=st.INK, lw=0.6, zorder=2)
    ax.bar(xg, go, width=w, color=st.ORANGE, zorder=3)
    ax.bar(xb, bmv, width=w, color=st.BLUE, zorder=3)
    marks = {"B": "^", "C": "s", "D": "o"}
    lo = np.nanmin(bmv)
    for g, mk in marks.items():
        v = np.array([mbe("season_x_group", f"{s}, {g}") for s in oc.SEASONS])
        lo = min(lo, np.nanmin(v)) if np.isfinite(v).any() else lo
        ax.scatter(xk, v, marker=mk, s=18, facecolor="white", edgecolor=st.BLUE, lw=0.8, zorder=4)
    ylim = (min(-1.0, np.floor((lo - 0.2) * 10) / 10), 1.7)
    # GO26 value labels sit beyond the bar end or a marker on the bar, whichever reaches further
    hi, low = go.copy(), go.copy()
    hr = []  # legend handles of the GO26-tower markers
    if rec is not None:
        # room for their legend below the bars, and above for the raised GO26 labels under the two legends there
        ylim = (min(ylim[0], -1.35), 1.9)
        pt = ax.get_position().height * ax.figure.get_size_inches()[1] * 72 / (ylim[1] - ylim[0])  # pt per mm/d
        for key, (mk, s, fc, ec, lw, lab) in GO26_TOWER_MARKERS.items():
            v = rec[key].to_numpy(float)
            if not np.isfinite(v).any():
                continue
            ax.scatter(xg, v, marker=mk, s=s, facecolor=fc, edgecolor=ec, lw=lw, zorder=5,
                       path_effects=[pe.Stroke(linewidth=lw + 2 * RING_PT, foreground="white"), pe.Normal()])
            half = (np.sqrt(s) * (np.sqrt(0.5) if mk == "D" else 0.5) + lw / 2 + RING_PT) / pt  # half height, mm/d
            hi, low = np.fmax(hi, v + half), np.fmin(low, v - half)
            hr.append(Line2D([], [], marker=mk, ls="", mfc=fc, mec=ec, mew=lw, ms=np.sqrt(s), label=lab))
    for xi, v, end in [*zip(xg, go, np.where(go >= 0, hi, low)), *zip(xb, bmv, bmv)]:
        if np.isfinite(v):
            lab = "0.00" if abs(v) < 0.005 else f"{v:+.2f}".replace("-", "−")
            ax.text(xi, end + (0.04 if v >= 0 else -0.04), lab, ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7.5, color=st.INK)
    ax.set_xticks(x - 0.08, ["Winter\n(DJF)", "Spring\n(MAM)", "Summer\n(JJA)", "Fall\n(SON)"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.65, len(x) - 0.35)
    ax.set_ylabel("OpenET ensemble bias (mm d$^{-1}$)")
    ax.set_title("Seasonal OpenET bias: GO26 vs benchmark", loc="left", pad=7, color=st.INK)
    hb = [Patch(facecolor=st.ORANGE, label="GO26, test set\n(their Table 4)"),
          Patch(facecolor=st.BLUE, label=f"OpenET collection 2.1\nvs benchmark\nclosed ET\n({period})")]
    leg = ax.legend(handles=hb, loc="upper left", handlelength=1.4, handletextpad=0.5, labelspacing=0.5,
                    borderpad=0.3, borderaxespad=0.3)
    ax.add_artist(leg)
    if hr:
        ax.add_artist(ax.legend(handles=hr, loc="lower left", ncol=len(hr), title=GO26_TOWERS_TITLE,
                                alignment="left", handlelength=0.8, handletextpad=0.4, columnspacing=1.2,
                                borderpad=0.3, borderaxespad=0.3))
    hk = [Line2D([], [], marker=mk, ls="", mfc="white", mec=st.BLUE, mew=0.8, ms=4.5, label=lab)
          for mk, lab in zip(marks.values(), ("B (dry)", "C (temperate)", "D (continental)"))]
    ax.legend(handles=hk, loc="upper right", title="Benchmark by\nKöppen group", alignment="left",
              handlelength=1.0, handletextpad=0.3, labelspacing=0.25, borderpad=0.3, borderaxespad=0.3)


def fig2() -> None:
    st.nature(8)
    t, bench, period = _openet_results()
    rec = _reconstructed_bias()
    summ = json.load(open(RES / "digitized_summary.json"))["US-UTJ"]
    sites = pd.read_csv(ROOT / "config" / "go26_sites.csv")
    cat = _category(sites)
    fig = plt.figure(figsize=(W, 6.3))
    left, wl, right, wr, h = 0.50, 2.40, 3.64, W - 3.64 - 0.04, 2.35
    ax = _axes(fig, left, 3.66, wl, h)
    bx = _axes(fig, right, 3.66, wr, h)
    cx = _axes(fig, left, 0.50, wl, h)
    dx = _axes(fig, right, 0.50, wr, h)
    up = ScaledTranslation(0, 1.5 / 72, fig.dpi_scale_trans)
    _panel_zero_fill(ax, summ, up)
    _panel_et_vs_p(bx, sites, cat, t, up)
    _panel_availability(cx, t, cat)
    _panel_seasons(dx, bench, period, rec)
    fig.text((right + wr) / W, 0.03 / fig.get_size_inches()[1], CREDIT, ha="right", va="bottom", fontsize=7,
             color=st.INK2)
    # panel labels level with the titles
    for a_, lab_ in ((ax, "a"), (bx, "b"), (cx, "c"), (dx, "d")):
        a_.annotate(lab_, xy=(0, 1), xycoords="axes fraction", xytext=(-34, 7),
                    textcoords="offset points", fontsize=10, fontweight="bold", va="baseline",
                    ha="left", color=st.INK)
    _save(fig, "fig2_zero_fill_bias_and_site_ET")


def table1_agreement(r: pd.DataFrame, tol: float = TABLE1_TOL) -> pd.DataFrame:
    """Per TABLE1_SERIES column, over the sites with both values (differences rebuilt minus GO26's Table 1, mm/d):
    sites, mean and RMS difference, sites within ``tol`` (and within 0.01), and the smallest and largest difference."""
    rows = {}
    for col in TABLE1_SERIES:
        d = (r[col] - r[TABLE1_REF]).dropna()
        rows[col] = {"n_sites": len(d), "mean_diff": d.mean(), "rmsd": float(np.sqrt((d ** 2).mean())),
                     "n_within_tol": int((d.abs() <= tol + 1e-9).sum()),
                     "n_within_0p01": int((d.abs() <= 0.01 + 1e-9).sum()), "min_diff": d.min(), "max_diff": d.max()}
    return pd.DataFrame(rows).T


def _signed(v: float) -> str:
    return f"{v:+.2f}".replace("-", "−")


def fig_table1() -> None:
    f = RES / RECON_SITES
    if not f.exists():
        print(f"results/{RECON_SITES} not found (scripts/10): Table 1 figure not drawn")
        return
    st.nature(8)
    r = pd.read_csv(f)
    r = r[r["status"] == "ok"].set_index("site")
    agree = table1_agreement(r)
    x = r[TABLE1_REF]
    zf = next(iter(TABLE1_SERIES))
    lim = np.ceil(np.nanmax(r[list(TABLE1_SERIES)].to_numpy()) * 4 + 0.5) / 4  # shared by all panels
    ticks = np.arange(0, lim, 0.5)
    fig = plt.figure(figsize=(W, 6.25))
    side, gap_x, gap_y = 2.55, 0.42, 0.36
    left = (W - (2 * side + gap_x)) / 2 + 0.2  # the grid centred, with room for the y labels on the left
    up = ScaledTranslation(0, 1.5 / 72, fig.dpi_scale_trans)
    for i, (col, (fc, title)) in enumerate(TABLE1_SERIES.items()):
        row, c = divmod(i, 2)
        ax = _axes(fig, left + c * (side + gap_x), 0.5 + (1 - row) * (side + gap_y), side, side)
        ax.plot([0, lim], [0, lim], color=st.INK2, lw=0.75, ls="--", zorder=1)
        ax.text(lim - 0.3, lim - 0.3, "1:1", fontsize=7.5, color=st.INK2, ha="right", va="bottom", rotation=45,
                transform_rotates_text=True, rotation_mode="anchor", transform=ax.transData + up)
        ax.scatter(x, r[col], s=24, facecolor=fc, edgecolor="white", lw=0.6, zorder=3)
        if col == zf:  # the sites outside the band, with leaders
            d = r[col] - x
            for site, (tx, ty, ha) in TABLE1_LABELS.items():
                if site in d.index and abs(d[site]) > TABLE1_TOL + 1e-9:
                    ax.annotate(site, xy=(x[site], r.loc[site, col]), xytext=(tx, ty), fontsize=7.5, color=st.INK,
                                ha=ha, va="center",
                                arrowprops=dict(arrowstyle="-", lw=0.6, color=st.INK, shrinkA=1, shrinkB=3))
        a_ = agree.loc[col]
        ax.text(0.97, 0.03, f"Mean diff. {_signed(a_.mean_diff)} mm d$^{{-1}}$\nRMSD {a_.rmsd:.2f} mm d$^{{-1}}$\n"
                f"Within ±{TABLE1_TOL:.2f}: {a_.n_within_tol:.0f} of {a_.n_sites:.0f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=st.INK, linespacing=1.3)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks] if row else [])
        ax.set_yticks(ticks, [] if c else [f"{t:g}" for t in ticks])
        if not c:
            ax.set_ylabel("Reproduced EC ET (mm d$^{-1}$)")
        if row:
            ax.set_xlabel("GO26 EC ET, Table 1 (mm d$^{-1}$)")
        for text, dx, kw in (("abcd"[i], 0, dict(fontsize=10, fontweight="bold")), (title, 12, dict(fontsize=8))):
            ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(dx, 7), textcoords="offset points",
                        va="baseline", ha="left", color=st.INK, **kw)
    _save(fig, "fig_table1_reconstruction")


def openet_agreement(m: pd.DataFrame) -> pd.DataFrame:
    """Per OPENET_SERIES panel, over the towers with both values: towers, OpenET bias (mean of OpenET minus observed
    ET over the same days), RMS difference and Pearson r (mm/d)."""
    rows = {}
    for (day_set, col) in OPENET_SERIES:
        s = m[m["day_set"] == day_set].dropna(subset=["openet_mm_d", col])
        d = s["openet_mm_d"] - s[col]
        rows[(day_set, col)] = {"n_sites": len(s), "bias": d.mean(), "rmsd": float(np.sqrt((d ** 2).mean())),
                                "r": float(np.corrcoef(s["openet_mm_d"], s[col])[0, 1]) if len(s) > 2 else np.nan}
    return pd.DataFrame(rows).T


def fig_openet() -> None:
    f = RES / RECON_OPENET
    if not f.exists():
        print(f"results/{RECON_OPENET} not found (scripts/10): OpenET figure not drawn")
        return
    st.nature(8)
    m = pd.read_csv(f, comment="#")
    agree = openet_agreement(m)
    vals = np.concatenate([m.loc[m["day_set"] == ds, ["openet_mm_d", col]].to_numpy().ravel()
                           for ds, col in OPENET_SERIES])
    lim = np.ceil(np.nanmax(vals) * 4 + 0.5) / 4  # shared by all panels
    ticks = np.arange(0, lim, 0.5)
    fig = plt.figure(figsize=(W, 6.25))
    side, gap_x, gap_y, bottom = 2.55, 0.42, 0.36, 0.5  # the Table 1 figure's grid
    left = (W - (2 * side + gap_x)) / 2 + 0.2
    up = ScaledTranslation(0, 1.5 / 72, fig.dpi_scale_trans)
    for i, ((day_set, col), (fc, title)) in enumerate(OPENET_SERIES.items()):
        row, c = divmod(i, 2)
        s = m[m["day_set"] == day_set]
        ax = _axes(fig, left + c * (side + gap_x), bottom + (1 - row) * (side + gap_y), side, side)
        ax.plot([0, lim], [0, lim], color=st.INK2, lw=0.75, ls="--", zorder=1)
        # 1:1 label low on the line and the statistics top left: the towers lie at OpenET >= 1.5 mm/d
        ax.text(1.0, 1.0, "1:1", fontsize=7.5, color=st.INK2, ha="right", va="bottom", rotation=45,
                transform_rotates_text=True, rotation_mode="anchor", transform=ax.transData + up)
        ax.scatter(s["openet_mm_d"], s[col], s=24, facecolor=fc, edgecolor="white", lw=0.6, zorder=3)
        a_ = agree.loc[(day_set, col)]
        ax.text(0.03, 0.97, f"OpenET bias {_signed(a_.bias)} mm d$^{{-1}}$\nRMSD {a_.rmsd:.2f} mm d$^{{-1}}$\n"
                f"r = {a_.r:.2f}, {a_.n_sites:.0f} sites",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=st.INK, linespacing=1.3)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks] if row else [])
        ax.set_yticks(ticks, [] if c else [f"{t:g}" for t in ticks])
        if not c:
            ax.set_ylabel("Reproduced EC ET (mm d$^{-1}$)")
        if row:
            ax.set_xlabel("OpenET ET, same days (mm d$^{-1}$)")
        for text, dx, kw in (("abcd"[i], 0, dict(fontsize=10, fontweight="bold")), (title, 12, dict(fontsize=8))):
            ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(dx, 7), textcoords="offset points",
                        va="baseline", ha="left", color=st.INK, **kw)
    _save(fig, "fig_openet_reconstruction")  # OpenET attribution: in the caption of the comment


def main() -> None:
    st.rerun_with_fixed_hash_seed(__name__)  # byte-reproducible PDFs (etcomment.style)
    fig1()
    fig2()
    fig_table1()
    fig_openet()
    print("wrote figures/fig1_*, figures/fig2_*, figures/fig_table1_*, figures/fig_openet_*")


if __name__ == "__main__":
    main()
