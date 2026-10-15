"""Site-specific plots from the Volk et al. (2023) benchmark.

1. Copies the benchmark's interactive Bokeh pages
   (graphical_files/<SITE>_plots.html) for the selected sites.
2. Renders a full-page PNG of each page with headless Chromium (Playwright).
   The pages load BokehJS 2.1.1 from cdn.bokeh.org; offline, pass
   --bokehjs-dir with bokeh.min.js and bokeh-widgets.min.js from the
   bokeh==2.1.1 sdist (the scripts are then inlined into temporary copies).
3. Draws a static four-panel summary per site from the daily CSV and the
   OpenET Phase II data (works offline):
     a  daily ET: unclosed, EBR-closed, gap-filled, OpenET ensemble (overpass)
     b  daily closure ratio (LE+H)/(Rn-G)
     c  monthly closed ET vs OpenET ensemble
     d  LE+H vs Rn-G

Usage:
  python scripts/03_site_plots.py --flux data/raw/flux_ET_dataset \
      --openet data/raw/OpenET_PhaseII_model_ET_dataset [--sites US-Ne1 US-Ne2 ...] \
      [--bokehjs-dir path/to/bokeh-2.1.1/js] [--no-render]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etcomment import benchmark as bm
from etcomment import closure as cl
from etcomment import style as st

ROOT = Path(__file__).resolve().parents[1]
SHARED = ["US-ARM", "US-Bi1", "US-Bi2", "US-Ne1", "US-Ne2", "US-Ne3", "US-xSL"]
OUT = ROOT / "figures" / "site_plots"


def copy_and_render(root: Path, sites: list[str], render: bool = True,
                    bokehjs_dir: Path | None = None) -> None:
    """Copy the benchmark's Bokeh pages and save full-page PNG snapshots.

    The pages load BokehJS 2.1.1 from cdn.bokeh.org.  Without internet access,
    pass ``bokehjs_dir`` holding bokeh.min.js and bokeh-widgets.min.js from the
    bokeh==2.1.1 sdist on PyPI (bokeh/server/static/js/); a temporary copy of
    each page then points at those files.  The copies in html/ are unchanged.
    """
    (OUT / "html").mkdir(parents=True, exist_ok=True)
    (OUT / "png_bokeh").mkdir(parents=True, exist_ok=True)
    pages = []
    for s in sites:
        src = root / "graphical_files" / f"{s}_plots.html"
        if src.exists():
            dst = OUT / "html" / src.name
            shutil.copy2(src, dst)
            pages.append(dst)
        else:
            print("missing", src)
    if not render or not pages:
        return
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed; skipping PNG rendering")
        return
    targets = pages
    if bokehjs_dir is not None:
        tmp = OUT / "_offline"
        tmp.mkdir(exist_ok=True)
        import re
        js = {name: (Path(bokehjs_dir) / name).read_text(encoding="utf-8") for name in
              ("bokeh.min.js", "bokeh-widgets.min.js")}
        targets = []
        for f in pages:
            html = f.read_text(encoding="utf-8", errors="replace")
            # inline the two CDN scripts (file:// pages cannot load crossorigin scripts)
            for url, name in (("bokeh-2.1.1.min.js", "bokeh.min.js"),
                              ("bokeh-widgets-2.1.1.min.js", "bokeh-widgets.min.js")):
                pat = re.compile(r'<script[^>]*src="https://cdn\.bokeh\.org/bokeh/release/' + re.escape(url) + r'"[^>]*></script>')
                html = pat.sub(lambda m, n=name: "<script type=\"text/javascript\">" + js[n] + "</script>", html, count=1)
            t = tmp / f.name
            t.write_text(html, encoding="utf-8")
            targets.append(t)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1300, "height": 1000}, device_scale_factor=1.25)
        for f in targets:
            pg.goto(f.resolve().as_uri(), wait_until="load", timeout=180_000)
            pg.wait_for_timeout(6000)
            pg.screenshot(path=str(OUT / "png_bokeh" / f.name.replace(".html", ".png")), full_page=True)
            print("rendered", f.name)
        b.close()
    if bokehjs_dir is not None:
        shutil.rmtree(OUT / "_offline", ignore_errors=True)


def static_panel(root: Path, oe_d: pd.DataFrame, oe_m: pd.DataFrame, site: str) -> None:
    """Four-panel site summary, 6.5 in wide (text >= 7 pt; see etcomment.style)."""
    import matplotlib.dates as mdates
    st.nature(8)
    d = bm.standardize(bm.load_daily(root, site))
    od = oe_d[oe_d["site"] == site].set_index("date")["Ensemble"]
    om = oe_m[oe_m["site"] == site].set_index("date")["Ensemble"]
    W, H = st.FULL_WIDTH_IN, 4.5
    fig = plt.figure(figsize=(W, H))

    def axes_in(left, bottom, width, height):  # rectangle in inches
        return fig.add_axes([left / W, bottom / H, width / W, height / H])

    a = axes_in(0.5, 2.42, W - 0.5 - 0.1, 1.6)
    a.plot(d.index, d["et"], color="#b9b8b2", lw=0.75, label="Unclosed EC ET")
    meas = d["et_corr"].where(d["valid_measured"])
    gap = d["et_corr"].where(d["et_gap"])
    a.plot(d.index, meas, color=st.INK, lw=0.75, label="EBR-closed EC ET")
    a.plot(d.index, gap, color=st.AQUA, lw=0.75, label="Gap-filled (EToF × ETo)")
    a.scatter(od.index, od.values, s=7, color=st.BLUE, lw=0, zorder=4, label="OpenET ensemble (overpass)")
    a.set_ylabel("ET (mm d$^{-1}$)")
    # legend in a row above the axes so that it hides no data
    a.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.15, handlelength=1.6,
             handletextpad=0.5, columnspacing=1.4, markerscale=1.6)
    a.annotate(f"{site}: Volk et al. (2023) benchmark, daily record", xy=(0, 1), xycoords="axes fraction",
               xytext=(0, 20), textcoords="offset points", fontsize=8.5, va="bottom", ha="left", color=st.INK)
    a.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    a.xaxis.set_major_formatter(mdates.ConciseDateFormatter(a.xaxis.get_major_locator()))

    w, gap_in, bottom, height = (W - 0.5 - 0.1 - 2 * 0.62) / 3, 0.62, 0.56, 1.3
    b = axes_in(0.5, bottom, w, height)
    ratio = (d["le"] + d["h"]) / (d["rn"] - d["g"])
    b.scatter(ratio.index, ratio.clip(0, 2), s=2.5, color=st.INK2, lw=0)
    b.axhline(1, color=st.AXIS, lw=0.75)
    b.set_ylim(0, 2)
    b.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=5))
    b.xaxis.set_major_formatter(mdates.ConciseDateFormatter(b.xaxis.get_major_locator()))
    b.set_ylabel("(LE + H)/(R$_n$ − G)")
    # the median goes in the title: the scatter fills the panel
    b.set_title(f"Daily closure ratio (median {np.nanmedian(ratio.where((ratio > 0) & (ratio < 3))):.2f})",
                loc="left")
    c = axes_in(0.5 + w + gap_in, bottom, w, height)
    mon = bm.complete_months(d)["et_corr"]  # months with <= 5 gap-filled days (V24 rule)
    jm = pd.concat([mon.rename("obs"), om.rename("openet")], axis=1).dropna()
    c.scatter(jm["obs"], jm["openet"], s=9, color=st.BLUE, lw=0)
    lim = max(10, float(np.nanmax(jm.values))) * 1.05 if len(jm) else 10
    c.plot([0, lim], [0, lim], color=st.INK2, lw=0.75, ls="--")
    c.set_xlim(0, lim); c.set_ylim(0, lim)
    if len(jm) == 0:
        c.text(0.04, 0.96, "no paired monthly data", transform=c.transAxes, ha="left", va="top",
               fontsize=7.5, color=st.INK2)
    c.set_xlabel("Closed EC ET (mm month$^{-1}$)"); c.set_ylabel("OpenET (mm month$^{-1}$)")
    c.set_title("Monthly", loc="left")
    e = axes_in(0.5 + 2 * (w + gap_in), bottom, w, height)
    e.scatter(d["rn"] - d["g"], d["le"] + d["h"], s=3, color=st.INK2, lw=0, alpha=0.5)
    lo = np.nanpercentile(d["rn"] - d["g"], 1); hi = np.nanpercentile(d["rn"] - d["g"], 99)
    e.plot([lo, hi], [lo, hi], color=st.BLUE, lw=0.9)
    e.set_xlabel("R$_n$ − G (W m$^{-2}$)"); e.set_ylabel("LE + H (W m$^{-2}$)")
    e.set_title("Energy balance", loc="left")
    st.check_text_size(fig)
    (OUT / "static").mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "static" / f"{site}_summary.{ext}", dpi=600,
                    metadata=st.PDF_METADATA if ext == "pdf" else None)
    plt.close(fig)


def main() -> None:
    st.rerun_with_fixed_hash_seed(__name__)  # byte-reproducible PDFs (etcomment.style)
    ap = argparse.ArgumentParser()
    ap.add_argument("--flux", default=str(ROOT / "data/raw/flux_ET_dataset"))
    ap.add_argument("--openet", default=str(ROOT / "data/raw/OpenET_PhaseII_model_ET_dataset"))
    ap.add_argument("--sites", nargs="*", default=SHARED)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--bokehjs-dir", default=None,
                    help="folder with bokeh.min.js and bokeh-widgets.min.js (v2.1.1) for offline rendering")
    args = ap.parse_args()
    root = bm.find_root(Path(args.flux))
    copy_and_render(root, args.sites, render=not args.no_render,
                    bokehjs_dir=Path(args.bokehjs_dir) if args.bokehjs_dir else None)
    oe_d = bm.load_openet_phase2(Path(args.openet), "daily")
    oe_m = bm.load_openet_phase2(Path(args.openet), "monthly")
    for s in args.sites:
        static_panel(root, oe_d, oe_m, s)
        print("plotted", s)


if __name__ == "__main__":
    main()
