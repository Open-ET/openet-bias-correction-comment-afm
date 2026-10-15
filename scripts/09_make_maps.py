"""Site map of the conterminous United States (CONUS), Fig. 3 of the comment.

The 22 Golkar & de Oliveira (2026) sites and the 59 cropland stations of the
Volk et al. (2023, 2024) benchmark over every Koppen-Geiger class present in
CONUS (Rubel et al., 2017). Only CONUS is drawn: neighboring countries, oceans
and lakes that are not wholly inside CONUS are left blank, and the map has no
scale bar, graticule or latitude/longitude labels.

Output: figures/fig3_site_map.pdf / .png, 6.5 in (165.1 mm) wide to be inserted
1:1 (all text >= 7 pt; see etcomment.style), 600 dpi.

Usage:
  python scripts/09_make_maps.py            # uses config/volk2024_stations.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from etcomment import maps as mp
from etcomment import style as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

FLAGGED = ["US-UTJ", "US-UTD", "US-UTV", "US-NSb"]
LABELS = {
    # label: (anchor site, dx_km, dy_km, ha); at 6.5 in wide 1 in = 757 km and a
    # 7.5-pt label is 270-490 km long, so labels stay inside the map extent
    "US-UTJ": ("US-UTJ", 230, -250, "left"),
    "US-UTD": ("US-UTD", 300, 60, "left"),
    "US-UTV": ("US-UTV", -150, 250, "center"),
    "US-NSb": ("US-NSb", -150, -200, "right"),
    "US-Bi1, Bi2": ("US-Bi1", -170, 290, "center"),
    "US-Ne1\u2013Ne3": ("US-Ne2", 140, 300, "left"),
    "US-ARM": ("US-ARM", 150, -230, "left"),
    "US-xSL": ("US-xSL", 120, 260, "left"),
    "US-UiC, UiG": ("US-UiC", 160, -250, "left"),
}
# site categories, in legend order
CATS = [
    ("flagged", "GO26 site with implausible observed ET (Table 2)", st.ORANGE),
    ("shared", "GO26 site also in the Volk et al. benchmark", st.BLUE),
    ("other", "Other GO26 site", "#ffffff"),
]


def load_sites(stations_xlsx: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    go = pd.read_csv(ROOT / "config" / "go26_sites.csv")
    go["cat"] = np.where(go["site"].isin(FLAGGED), "flagged",
                         np.where(go["in_volk_benchmark"], "shared", "other"))
    if str(stations_xlsx).endswith(".csv"):
        meta = pd.read_csv(stations_xlsx, comment="#")
        v = meta[meta["general_classification"] == "Croplands"].copy()
    else:
        meta = pd.read_excel(stations_xlsx).rename(columns={"Unnamed: 0": "site"})
        v = meta[meta["General classification"] == "Croplands"].copy()
        v = v.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
    for df in (go, v):
        df["x"], df["y"] = mp.to_albers(df["longitude"].values, df["latitude"].values)
    go["kg_rubel"] = mp.koppen_at(go["latitude"], go["longitude"])
    return go, v


def kg_colors(raster: mp.KoppenRaster, lighten: float, keep: list[str] | None,
               other_gray: str = "#eeeeec", outside: str = "#ffffff", water: str = "#ffffff",
               palette: dict | None = None, min_frac: float = 0.003) -> tuple[np.ndarray, list[str]]:
    """RGBA image for the CONUS part of the raster; returns (image, classes for the legend).

    Classes covering less than ``min_frac`` of CONUS are drawn but not listed
    (``min_frac=0`` lists every class present).
    """
    h, w = raster.codes.shape
    img = np.ones((h, w, 4))
    land_outside = (~raster.us_mask) & (~raster.water)
    img[land_outside, :3] = np.array(matplotlib.colors.to_rgb(outside))
    img[raster.water, :3] = np.array(matplotlib.colors.to_rgb(water))
    present = []
    inside = raster.us_mask & ~raster.water
    total = inside.sum()
    for i, name in enumerate(raster.classes):
        sel = inside & (raster.codes == i)
        if not sel.any() or name == "Ocean":
            continue
        frac = sel.sum() / total
        if keep is not None and name not in keep:
            img[sel, :3] = np.array(matplotlib.colors.to_rgb(other_gray))
            continue
        if palette is not None and name in palette:
            img[sel, :3] = np.array(matplotlib.colors.to_rgb(palette[name]))
        else:
            img[sel, :3] = mp.lighten(mp.KG_RGB.get(name, (200, 200, 200)), lighten)
        if frac >= min_frac:
            present.append(name)
    order = ["Af", "Am", "Aw", "BWh", "BWk", "BSh", "BSk", "Csa", "Csb", "Csc", "Cfa", "Cfb", "Cfc",
             "Cwa", "Cwb", "Dsa", "Dsb", "Dsc", "Dwa", "Dwb", "Dfa", "Dfb", "Dfc", "ET", "EF"]
    present = [c for c in order if c in present]
    return img, present


def conus_outline(states):
    """CONUS polygon in Albers, with the narrow gaps between neighboring states closed."""
    return states.to_crs(mp.ALBERS).geometry.union_all().buffer(1500).buffer(-1500)


def lakes_inside(lakes, nat, min_frac=0.9):
    """Lakes lying (almost) wholly inside CONUS; border lakes such as the Great Lakes are left out."""
    from shapely.geometry import MultiPolygon, Polygon
    polys = [nat] if nat.geom_type == "Polygon" else list(nat.geoms)
    filled = MultiPolygon([Polygon(p.exterior) for p in polys]).buffer(5000)
    lk = lakes.to_crs(mp.ALBERS)
    frac = lk.geometry.intersection(filled).area / lk.geometry.area
    return lk[frac >= min_frac]


def draw_base(ax, raster, img, states, lakes=None, lw_state=0.25, lw_nat=0.6,
              state_col="#8c8c88", lake_fill="#ffffff", lake_edge="#9a9a96"):
    x0, x1, y0, y1 = raster.extent
    ax.imshow(img, extent=(x0, x1, y0, y1), origin="upper", interpolation="nearest", zorder=0)
    nat = conus_outline(states)
    if lakes is not None:
        lakes_inside(lakes, nat).plot(ax=ax, facecolor=lake_fill, edgecolor=lake_edge,
                                      linewidth=0.3, zorder=1.8)
    states.to_crs(mp.ALBERS).boundary.plot(ax=ax, color=state_col, linewidth=lw_state, zorder=2)
    import geopandas as gpd
    gpd.GeoSeries([nat], crs=mp.ALBERS).boundary.plot(ax=ax, color="#3d3d3a", linewidth=lw_nat, zorder=3)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("")
    ax.set_ylabel("")


def draw_sites(ax, go, v, ms_v=13, ms_go=34, edge="#0b0b0b"):
    ax.scatter(v["x"], v["y"], s=ms_v, marker="^", facecolor="none", edgecolor=st.INK2,
               linewidth=0.6, zorder=4)
    for cat, _, col in CATS[::-1]:
        d = go[go["cat"] == cat]
        ax.scatter(d["x"], d["y"], s=ms_go, marker="o", facecolor=col, edgecolor=edge,
                   linewidth=0.7, zorder=6)


def label_sites(ax, go, fontsize, halo=True):
    pos = go.set_index("site")
    for text, (anchor, dx, dy, ha) in LABELS.items():
        x, y = pos.loc[anchor, ["x", "y"]]
        tx, ty = x + dx * 1e3, y + dy * 1e3
        ax.annotate(text, xy=(x, y), xytext=(tx, ty), ha=ha, va="center", fontsize=fontsize,
                    color=st.INK, zorder=8,
                    arrowprops=dict(arrowstyle="-", color=st.INK, lw=0.6, shrinkA=0, shrinkB=3.2),
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")] if halo else None)


def map_all_classes(go, v, raster, states):
    """Fig. 3: GO26 sites and benchmark cropland stations over every Koppen-Geiger class in CONUS.

    The map fills the 6.5-in width; the site legend sits in the blank area
    south-west of CONUS (Pacific/Mexico, outside the mapped data) and the class
    legend runs below the map.
    """
    st.nature(8)
    img, present = kg_colors(raster, lighten=0.45, keep=None, min_frac=0.0)
    x0, x1, y0, y1 = raster.extent
    width = st.FULL_WIDTH_IN
    map_h = width * (y1 - y0) / (x1 - x0)
    legend_h = 0.70
    fig = plt.figure(figsize=(width, map_h + legend_h))
    ax = fig.add_axes([0.0, legend_h / (map_h + legend_h), 1.0, map_h / (map_h + legend_h)])
    draw_base(ax, raster, img, states, lakes=LAKES, lw_state=0.25, lw_nat=0.6,
              state_col="#9b9b97", lake_fill="#ffffff", lake_edge="#9b9b97")
    draw_sites(ax, go, v)
    label_sites(ax, go, fontsize=7.5)
    ax.set_axis_off()
    # the tundra code ET would read as evapotranspiration in this paper
    handles = [Patch(facecolor=mp.lighten(mp.KG_RGB[c], 0.45), edgecolor="#6d6d69", linewidth=0.3,
                     label="ET (tundra)" if c == "ET" else c) for c in present]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, 0.0), ncol=8,
               handlelength=1.2, handleheight=0.9, columnspacing=1.0, handletextpad=0.5,
               frameon=False, borderaxespad=0.1,
               title="K\u00f6ppen\u2013Geiger class (Rubel et al., 2017)", alignment="left")
    site_legend(ax)
    return fig


def site_legend(ax):
    hs = [Line2D([], [], marker="o", ls="", mfc=col, mec=st.INK, mew=0.7, ms=5.8, label=lab)
          for _, lab, col in CATS]
    hs.append(Line2D([], [], marker="^", ls="", mfc="none", mec=st.INK2, mew=0.6, ms=5.2,
                     label="Volk et al. (2024) cropland station"))
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=1,
              handletextpad=0.4, labelspacing=0.3, frameon=False, borderaxespad=0.3)


LAKES = None


def main() -> None:
    st.rerun_with_fixed_hash_seed(__name__)  # byte-reproducible PDFs (etcomment.style)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", default=str(ROOT / "config" / "volk2024_stations.csv"),
                    help="config/volk2024_stations.csv (default) or the Station_metadata.xlsx")
    ap.add_argument("--res", type=float, default=2500.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    go, v = load_sites(Path(args.stations))
    global LAKES
    states = mp.load_conus_states()
    LAKES = mp.load_gshhs(level=2, res="i", min_area_km2=1500)
    raster = mp.koppen_raster_albers(extent=(-2.62e6, 2.30e6, 0.22e6, 3.22e6), res=args.res, states=states)
    go[["site", "koppen_go26", "kg_rubel", "cat"]].to_csv(ROOT / "results" / "site_koppen_check.csv", index=False)
    for name, fn in (("fig3_site_map", map_all_classes),):
        fig = fn(go, v, raster, states)
        st.check_text_size(fig)
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=600, metadata=st.PDF_METADATA if ext == "pdf" else None)
        plt.close(fig)
        print("wrote", OUT / f"{name}.pdf")


if __name__ == "__main__":
    main()
