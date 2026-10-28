"""Digitize the daily time-series panels of Golkar & de Oliveira (2026).

Inputs (CC BY 4.0; sources in data/README.md and README.md, Data streams):
  * Fig. S2 (US-UTJ, BSk) : supplementary file 1-s2.0-S0168192326005083-mmc2_lrg.jpg
  * Fig. 8  (US-UiC, Dfa) : embedded image 10 of the article PDF (pdfimages -png)

Outputs:
  results/digitized_<site>.csv           daily observed / OpenET / XGBoost values
  results/digitized_summary.json         statistics with and without exact-zero days, and
                                         ("sensitivity") the RMSE improvement with the dots
                                         hidden under the lines added and the zero
                                         tolerance doubled
  figures/qc/digitized_<site>_overlay.png digitized points drawn over the source

Usage:
  python scripts/04_digitize_afm_figures.py --figS2 <jpg> --fig8 <png>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etcomment import digitize as dg
from etcomment import metrics as mt

ROOT = Path(__file__).resolve().parents[1]
ZERO_TOL = 0.015  # mm/day (about 5 px in Fig. S2). Dots in the zero runs sit
                  # within +-0.012 mm/day of the zero line (+-3.4 px of
                  # digitizing noise); the closest genuine values are -0.029
                  # and +0.031 mm/day. Using 0.03 instead adds one zero day.
# Sensitivity of the RMSE improvement (summary key "sensitivity"; not in the digitized series): dots the two passes
# miss because a plotted line crosses them, read where at least this share of the dot's disk shows (at US-UTJ six of
# the 13 days of 2024 without a reading; each checked by eye: a dot crossed by a near-vertical XGBoost line)
HIDDEN_MIN_COVER = 0.3


def figS2_config(path: str) -> dict:
    """US-UTJ panel (BSk). Axis ticks located on the 7008 x 3171 px image."""
    rgb = dg.load_rgb(path)
    dark = rgb.max(axis=2) < 110
    # y ticks sit in a 32-px band left of the left spine (x ~ 388-393)
    yt = dg.find_tick_positions(dark, slice(355, 387), axis=0, min_count=20)
    # x ticks hang below the bottom spine (y ~ 2807-2812)
    xt = dg.find_tick_positions(dark, slice(2814, 2840), axis=1, min_count=15)
    assert len(yt) == 7, yt
    assert len(xt) == 8, xt
    y_vals = [5, 4, 3, 2, 1, 0, -1]
    x_dates = pd.to_datetime(
        ["2024-01-01", "2024-03-01", "2024-05-01", "2024-07-01",
         "2024-09-01", "2024-11-01", "2025-01-01", "2025-03-01"])
    cal = dg.AxisCalibration(dict(zip(xt, x_dates)), dict(zip(yt, y_vals)))
    # blank text boxes (statistics box, legend) and the title
    blank = [(0, 0, rgb.shape[1], 300), (480, 300, 1700, 760), (5590, 230, 6940, 760)]
    return dict(site="US-UTJ", climate="BSk", rgb=rgb, cal=cal, blank=blank,
                dates=pd.date_range("2024-01-01", "2025-02-28", freq="D"),
                dot_d=28.0, obs_mask=dg.dark_unsaturated(130, 45),
                row_limits=(int(cal.y_of(5.7)), int(cal.y_of(-1.85))),
                probe_offsets=(-2, -1, 0, 1, 2), max_gap_px=26,
                lines={"openet": dg.reddish(150, 50, 50), "xgb": dg.bluish(120, 80, 100)},
                printed={"openet_rmse": 1.877, "xgb_rmse": 0.885,
                         "improvement_pct": 52.8, "coverage_pct": 87.2},
                table1_mean_obs=0.59)


def fig8_config(path: str) -> dict:
    """US-UiC panel (Dfa, 2020). Axis calibrated from tick-label centers.

    The panel has no outward tick marks, so label centers are used.  Labels
    are centered on their ticks, so the error is ~1 px (0.01 mm/day; 0.2 day).
    """
    rgb = dg.load_rgb(path)
    txt = rgb[:, 55:92].max(axis=2) < 120
    rows = np.where(txt.sum(axis=1) > 0)[0]
    groups: list[list[int]] = []
    for r in rows:
        if groups and r - groups[-1][-1] <= 2:
            groups[-1].append(int(r))
        else:
            groups.append([int(r)])
    centers = [0.5 * (g[0] + g[-1]) for g in groups if 700 > g[0] > 60 or 760 > g[-1] > 700]
    # merge fragments of the same glyph (gap < 20 px)
    merged: list[float] = []
    for c in centers:
        if merged and c - merged[-1] < 20:
            merged[-1] = 0.5 * (merged[-1] + c)
        else:
            merged.append(c)
    merged = [c for c in merged if c < 760]
    assert len(merged) == 8, merged
    y_vals = [6, 5, 4, 3, 2, 1, 0, -1]
    lab = rgb[785:808, :].max(axis=2) < 120
    cols = np.where(lab.sum(axis=0) > 0)[0]
    g2: list[list[int]] = []
    for c in cols:
        if g2 and c - g2[-1][-1] <= 6:
            g2[-1].append(int(c))
        else:
            g2.append([int(c)])
    xc = [0.5 * (g[0] + g[-1]) for g in g2]
    assert len(xc) == 12, xc
    x_dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    cal = dg.AxisCalibration(dict(zip(xc, x_dates)), dict(zip(merged, y_vals)))
    blank = [(0, 0, rgb.shape[1], 62), (100, 62, 715, 135), (1160, 690, 2000, 730)]
    return dict(site="US-UiC", climate="Dfa", rgb=rgb, cal=cal, blank=blank,
                dates=pd.date_range("2020-01-01", "2020-12-31", freq="D"),
                dot_d=7.0, obs_mask=dg.dark_unsaturated(130, 40),
                row_limits=(int(cal.y_of(6.6)), int(cal.y_of(-1.45))),
                probe_offsets=(-1, 0, 1), max_gap_px=8,
                lines={"openet": dg.reddish(180, 60, 40), "xgb": dg.bluish(120, 50, 110)},
                printed={"openet_rmse": 1.002, "xgb_rmse": 0.745,
                         "improvement_pct": 25.6, "coverage_pct": 92.9},
                table1_mean_obs=None)


def run(cfg: dict) -> tuple[pd.DataFrame, dict]:
    rgb = cfg["rgb"].copy()
    for x0, y0, x1, y1 in cfg["blank"]:
        rgb[y0:y1, x0:x1] = 255
    cal: dg.AxisCalibration = cfg["cal"]
    obs = dg.extract_dots(rgb, cal, cfg["dates"], cfg["obs_mask"], cfg["dot_d"],
                          row_limits=cfg["row_limits"],
                          bridge_fns=list(cfg["lines"].values()),
                          max_gap_px=cfg["max_gap_px"],
                          probe_offsets=cfg["probe_offsets"])
    # second pass: dots crossed by both line series
    peaks = dg.disk_peaks(rgb, cal, cfg["obs_mask"], cfg["dot_d"], row_limits=cfg["row_limits"],
                          min_cover=0.6)
    obs, added = dg.fill_from_peaks(obs, peaks, max_offset=0.3)
    lw = {}
    out = {"obs": obs}
    for name, mfn in cfg["lines"].items():
        # line width from columns spread across the panel
        xs = [int(cal.x_of(d)) for d in cfg["dates"][::15]]
        lw[name] = dg.estimate_line_width(rgb, mfn, xs)
        out[name] = dg.extract_line(rgb, cal, cfg["dates"], mfn, lw[name],
                                    row_limits=cfg["row_limits"])
    df = pd.DataFrame(out)
    df.index.name = "date"
    # the lines are continuous polylines; fill short digitizing gaps (<= 3 days)
    # by linear interpolation in time and flag them
    for name in cfg["lines"]:
        df[f"{name}_interp"] = df[name].isna()
        df[name] = df[name].interpolate(method="time", limit=3, limit_area="inside")
        df[f"{name}_interp"] &= df[name].notna()
    df["obs_is_zero"] = df["obs"].abs() < ZERO_TOL
    # keep days where both OpenET and an observation exist
    both = df.dropna(subset=["obs", "openet", "xgb"])
    nz = both[~both["obs_is_zero"]]
    z = both[both["obs_is_zero"]]
    res = {
        "site": cfg["site"],
        "climate": cfg["climate"],
        "calibration": {
            "px_per_day": cal.px_per_day,
            "px_per_mm_day": cal.px_per_unit,
            "x_tick_max_residual_px": cal.x_resid_px,
            "y_tick_max_residual_px": cal.y_resid_px,
            "value_resolution_mm_day": 1.0 / cal.px_per_unit,
            "line_width_px": lw,
        },
        "second_pass_added_days": added,
        "n_days_with_obs": int(both.shape[0]),
        "n_exact_zero_days": int(z.shape[0]),
        "zero_fraction": float(z.shape[0] / both.shape[0]) if both.shape[0] else np.nan,
        "zero_runs": zero_runs(df),
        "n_negative_obs": int((both["obs"] < -ZERO_TOL).sum()),
        "min_obs": float(both["obs"].min()),
        "mean_obs_all": float(both["obs"].mean()),
        "mean_obs_nonzero": float(nz["obs"].mean()),
        "openet_mean_on_zero_days": float(z["openet"].mean()) if len(z) else np.nan,
        "xgb_mean_on_zero_days": float(z["xgb"].mean()) if len(z) else np.nan,
        "openet_all": mt.summary(both["openet"], both["obs"]),
        "openet_nonzero": mt.summary(nz["openet"], nz["obs"]),
        "xgb_all": mt.summary(both["xgb"], both["obs"]),
        "xgb_nonzero": mt.summary(nz["xgb"], nz["obs"]),
        "printed_on_figure": cfg["printed"],
        "afm_table1_mean_obs": cfg["table1_mean_obs"],
    }
    a = res["openet_all"]["rmse"]
    b = res["xgb_all"]["rmse"]
    res["improvement_pct_all"] = 100 * (1 - b / a)
    res["improvement_pct_nonzero"] = 100 * (1 - res["xgb_nonzero"]["rmse"] / res["openet_nonzero"]["rmse"])
    res["attribution_all"] = mt.improvement_attribution(both["openet"], both["xgb"], both["obs"])
    res["attribution_nonzero"] = mt.improvement_attribution(nz["openet"], nz["xgb"], nz["obs"])
    hidden = dg.disk_peaks(rgb, cal, cfg["obs_mask"], cfg["dot_d"], row_limits=cfg["row_limits"],
                           min_cover=HIDDEN_MIN_COVER)
    res["sensitivity"] = sensitivity(df, *dg.fill_from_peaks(df["obs"], hidden, max_offset=0.3))
    return df, res


def improvement(df: pd.DataFrame, obs: pd.Series, zero_tol: float) -> dict:
    """RMSE improvement of XGBoost over OpenET (%) against ``obs``: on every day with an observation and both lines,
    and without the days within ``zero_tol`` of zero (the exact zeros)."""
    both = df.assign(obs=obs).dropna(subset=["obs", "openet", "xgb"])
    nz = both[both["obs"].abs() >= zero_tol]

    def pct(s: pd.DataFrame) -> float:
        return 100 * (1 - mt.summary(s["xgb"], s["obs"])["rmse"] / mt.summary(s["openet"], s["obs"])["rmse"])

    return {"n_days": int(len(both)), "n_nonzero_days": int(len(nz)), "improvement_pct_all": pct(both),
            "improvement_pct_nonzero": pct(nz)}


def sensitivity(df: pd.DataFrame, with_hidden: pd.Series, hidden_added: list) -> dict:
    """The RMSE improvement with and without the exact zeros for the digitized observations and for those plus the
    dots hidden under the lines (``with_hidden``, ``hidden_added``: :func:`etcomment.digitize.fill_from_peaks` at
    HIDDEN_MIN_COVER), each with the zero tolerance ZERO_TOL and doubled; ``range_pct_nonzero``: the smallest and
    largest improvement without the exact zeros over the four cases."""
    cases = {f"{name}{suffix}": improvement(df, obs, tol)
             for name, obs in (("digitized", df["obs"]), ("hidden_added", with_hidden))
             for suffix, tol in (("", ZERO_TOL), ("_zero_tol_doubled", 2 * ZERO_TOL))}
    pcts = [c["improvement_pct_nonzero"] for c in cases.values()]
    return {"zero_tol": ZERO_TOL, "hidden_min_cover": HIDDEN_MIN_COVER, "hidden_added_days": hidden_added,
            "cases": cases, "range_pct_nonzero": [min(pcts), max(pcts)]}


def zero_runs(df: pd.DataFrame) -> list[dict]:
    z = df["obs_is_zero"].fillna(False).astype(bool)
    runs = []
    start = None
    prev = None
    for d, v in z.items():
        if v and start is None:
            start = d
        if not v and start is not None:
            runs.append({"start": str(start.date()), "end": str(prev.date()),
                         "n_days": int((prev - start).days + 1)})
            start = None
        prev = d
    if start is not None:
        runs.append({"start": str(start.date()), "end": str(prev.date()),
                     "n_days": int((prev - start).days + 1)})
    return runs


def overlay(cfg: dict, df: pd.DataFrame, out: Path) -> None:
    """QC image: the digitized values drawn over the source panel.

    The source fills the image at up to 3600 px wide (150 dpi, so it is legible
    at 100% zoom) and the legend sits in a strip below it, covering nothing.
    Observed values are hollow lime rings just wider than the source dots (the
    dot stays visible inside its ring); OpenET and XGBoost samples are short
    magenta and yellow ticks, one per day.
    """
    from matplotlib.lines import Line2D

    from etcomment import style as st
    st.nature(8)  # same fonts whatever ran before in this process
    cal = cfg["cal"]
    rgb = cfg["rgb"]
    h, w, _ = rgb.shape
    dpi = 150
    scale = min(1.0, 3600 / w)  # output px per source px
    pt = 72 / dpi  # points per output px
    img_w, img_h, strip = w * scale / dpi, h * scale / dpi, 0.42
    fig = plt.figure(figsize=(img_w, img_h + strip), dpi=dpi)
    ax = fig.add_axes([0, strip / (img_h + strip), 1, img_h / (img_h + strip)])
    ax.imshow(rgb.astype(np.uint8))
    ax.set_axis_off()
    xs = [cal.x_of(d) for d in df.index]
    edge = 0.8  # ring line width, pt
    ring = (cfg["dot_d"] * scale + 2 * edge / pt + 2) * pt
    tick = 0.8 * cal.px_per_day * scale * pt
    styles = (("obs", "obs: observed ET (digitized dots)",
               dict(marker="o", mfc="none", mec="lime", mew=edge, ms=ring)),
              ("openet", "openet: OpenET line (digitized)", dict(marker="_", color="magenta", mew=1.0, ms=tick)),
              ("xgb", "xgb: XGBoost line (digitized)", dict(marker="_", color="yellow", mew=1.0, ms=tick)))
    handles = []
    for col, label, kw in styles:
        ys = [cal.y_of(v) if np.isfinite(v) else np.nan for v in df[col]]
        ax.plot(xs, ys, ls="", alpha=0.9, **kw)
        big = dict(kw, ms=9 if col == "obs" else 12, mew=1.2 if col == "obs" else 2.0)
        handles.append(Line2D([], [], ls="", label=label, **big))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=True, facecolor="#4a4a48",
               edgecolor="none", framealpha=1.0, labelcolor="white", borderaxespad=0.3, columnspacing=2.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figS2", required=True)
    ap.add_argument("--fig8", required=True)
    args = ap.parse_args()
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    summary = {}
    for cfg in (figS2_config(args.figS2), fig8_config(args.fig8)):
        df, res = run(cfg)
        site = cfg["site"]
        df.to_csv(ROOT / "results" / f"digitized_{site}.csv", float_format="%.3f")
        overlay(cfg, df, ROOT / "figures" / "qc" / f"digitized_{site}_overlay.png")
        summary[site] = res
    with open(ROOT / "results" / "digitized_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
