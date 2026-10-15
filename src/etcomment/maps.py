"""Base layers for the CONUS site maps (Fig. 3).

Everything here uses data that install from PyPI, so the maps build without
network access to map servers:

* State boundaries: U.S. Census Bureau county polygons shipped with
  ``basemap-data`` (dissolved to states), NAD83 (EPSG:4269).
* Koppen-Geiger climate: the 100-arc-second (~2.8 km) grid shipped with
  ``kgcpy`` (Rubel et al., 2017, Meteorol. Z. 26, 115-125; 1986-2010), the same
  classification cited by Volk et al. (2024).

Projection: USA Contiguous Albers Equal Area Conic (EPSG:5070; NAD83;
standard parallels 29.5 N and 45.5 N; central meridian 96 W; latitude of
origin 23 N).
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from importlib import resources

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from PIL import Image
from pyproj import Transformer

ALBERS = "EPSG:5070"
NAD83 = "EPSG:4269"
WGS84 = "EPSG:4326"

NON_CONUS = {"AK", "HI", "PR", "VI", "GU", "AS", "MP", "UM"}

# Standard Koppen-Geiger colors (Beck et al., 2018, Sci. Data 5, 180214).
KG_RGB = {
    "Af": (0, 0, 255), "Am": (0, 120, 255), "Aw": (70, 170, 250), "As": (70, 170, 250),
    "BWh": (255, 0, 0), "BWk": (255, 150, 150), "BSh": (245, 165, 0), "BSk": (255, 220, 100),
    "Csa": (255, 255, 0), "Csb": (200, 200, 0), "Csc": (150, 150, 0),
    "Cwa": (150, 255, 150), "Cwb": (100, 200, 100), "Cwc": (50, 150, 50),
    "Cfa": (200, 255, 80), "Cfb": (100, 255, 80), "Cfc": (50, 200, 0),
    "Dsa": (255, 0, 255), "Dsb": (200, 0, 200), "Dsc": (150, 50, 150), "Dsd": (150, 100, 150),
    "Dwa": (170, 175, 255), "Dwb": (90, 120, 220), "Dwc": (75, 80, 180), "Dwd": (50, 0, 135),
    "Dfa": (0, 255, 255), "Dfb": (55, 200, 255), "Dfc": (0, 125, 125), "Dfd": (0, 70, 95),
    "ET": (178, 178, 178), "EF": (102, 102, 102),
}
KG_NAMES = {
    "BWh": "Hot desert", "BWk": "Cold desert", "BSh": "Hot semi-arid", "BSk": "Cold semi-arid",
    "Csa": "Hot-summer Mediterranean", "Csb": "Warm-summer Mediterranean",
    "Cfa": "Humid subtropical", "Cfb": "Temperate oceanic",
    "Dsa": "Hot-summer continental, dry summer", "Dsb": "Warm-summer continental, dry summer",
    "Dsc": "Subarctic, dry summer",
    "Dwa": "Hot-summer continental, dry winter", "Dwb": "Warm-summer continental, dry winter",
    "Dfa": "Hot-summer humid continental", "Dfb": "Warm-summer humid continental",
    "Dfc": "Subarctic", "ET": "Tundra", "Cwa": "Monsoon-influenced humid subtropical",
    "Cwb": "Subtropical highland", "Aw": "Tropical savanna", "Am": "Tropical monsoon",
    "Af": "Tropical rainforest", "Cfc": "Subpolar oceanic", "Csc": "Cold-summer Mediterranean",
    "EF": "Ice cap",
}


def _basemap_data_dir() -> str:
    import mpl_toolkits.basemap_data as bmd  # type: ignore

    return list(bmd.__path__)[0]


def load_conus_states() -> gpd.GeoDataFrame:
    """CONUS state polygons (NAD83) dissolved from the Census county layer."""
    shp = os.path.join(_basemap_data_dir(), "UScounties.shp")
    counties = gpd.read_file(shp)
    counties = counties[~counties["STATE"].isin(NON_CONUS)]
    states = counties.dissolve(by="STATE", as_index=False)[["STATE", "geometry"]]
    # county edges do not meet exactly; close the slivers they leave (in meters)
    states = states.set_crs(NAD83, allow_override=True).to_crs(ALBERS)
    states["geometry"] = states.geometry.buffer(300).buffer(-300)
    return states.to_crs(NAD83)


def load_gshhs(level: int = 2, res: str = "i", bbox=(-130, 20, -60, 55),
               min_area_km2: float = 0.0) -> gpd.GeoDataFrame:
    """GSHHS polygons from basemap-data: level 1 land, 2 lakes, 3 islands in lakes."""
    from shapely.geometry import Polygon

    d = _basemap_data_dir()
    meta = open(os.path.join(d, f"gshhsmeta_{res}.dat")).read().split("\n")
    polys, areas = [], []
    with open(os.path.join(d, f"gshhs_{res}.dat"), "rb") as f:
        for row in meta:
            p = row.split()
            if len(p) < 7 or int(p[0]) != level:
                continue
            area, npts, south, north = float(p[1]), int(p[2]), float(p[3]), float(p[4])
            if area < min_area_km2 or north < bbox[1] or south > bbox[3]:
                continue
            f.seek(int(p[5]))
            xy = np.frombuffer(f.read(int(p[6])), dtype="<f4").astype(float).reshape(npts, 2)
            if xy[:, 0].max() < bbox[0] or xy[:, 0].min() > bbox[2] or npts < 4:
                continue
            polys.append(Polygon(xy).buffer(0))
            areas.append(area)
    return gpd.GeoDataFrame({"area_km2": areas}, geometry=polys, crs=WGS84)


def _kgcpy_grid() -> tuple[np.ndarray, dict[int, str]]:
    with resources.files("kgcpy").joinpath("kmz_int_reshape.png").open("rb") as fp:
        Image.MAX_IMAGE_PIXELS = None
        arr = np.array(Image.open(io.BytesIO(fp.read())))
    with resources.files("kgcpy").joinpath("kg_zoneNum.csv").open("rb") as fp:
        lut = pd.read_csv(io.BytesIO(fp.read()))
    names = dict(zip(lut["zoneNum"].astype(int), lut["kg_zone"].astype(str)))
    return arr, names


def koppen_at(lat, lon) -> list[str]:
    """Koppen-Geiger class at points (same lookup as kgcpy.lookupCZ)."""
    arr, names = _kgcpy_grid()
    ny, nx = arr.shape
    lat = np.atleast_1d(np.asarray(lat, float))
    lon = np.atleast_1d(np.asarray(lon, float))
    ix = np.clip(np.round((lon + 180) * nx / 360 - 0.5).astype(int), 0, nx - 1)
    iy = np.clip(np.round(-(lat - 90) * ny / 180 - 0.5).astype(int), 0, ny - 1)
    return [names[int(v)] for v in arr[iy, ix]]


@dataclass
class KoppenRaster:
    codes: np.ndarray        # class names as integer index into ``classes``
    classes: list[str]
    us_mask: np.ndarray      # True inside CONUS
    water: np.ndarray        # True for ocean / large lakes
    extent: tuple[float, float, float, float]  # x0, x1, y0, y1 (EPSG:5070 m)


def koppen_raster_albers(extent=(-2.42e6, 2.30e6, 0.22e6, 3.22e6), res=2500.0,
                         states: gpd.GeoDataFrame | None = None) -> KoppenRaster:
    """Resample the kgcpy grid onto an Albers grid (nearest neighbor)."""
    arr, names = _kgcpy_grid()
    ny, nx = arr.shape
    x0, x1, y0, y1 = extent
    xs = np.arange(x0 + res / 2, x1, res)
    ys = np.arange(y1 - res / 2, y0, -res)
    X, Y = np.meshgrid(xs, ys)
    tr = Transformer.from_crs(ALBERS, WGS84, always_xy=True)
    lon, lat = tr.transform(X, Y)
    ix = np.clip(np.round((lon + 180) * nx / 360 - 0.5).astype(int), 0, nx - 1)
    iy = np.clip(np.round(-(lat - 90) * ny / 180 - 0.5).astype(int), 0, ny - 1)
    raw = arr[iy, ix]
    classes = [names[k] for k in sorted(names)]
    code_index = {k: i for i, k in enumerate(sorted(names))}
    codes = np.vectorize(code_index.get)(raw)
    water = raw == [k for k, v in names.items() if v == "Ocean"][0]
    if states is None:
        states = load_conus_states()
    us = states.to_crs(ALBERS).geometry.union_all().simplify(500)
    us_mask = shapely.contains_xy(us, X, Y)
    return KoppenRaster(codes, classes, us_mask, water, (x0, x1, y0, y1))


def lighten(rgb, frac: float) -> tuple[float, float, float]:
    """Blend an 0-255 RGB triple with white; returns 0-1 floats."""
    r, g, b = (np.asarray(rgb, float) / 255.0)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def to_albers(lon, lat):
    tr = Transformer.from_crs(WGS84, ALBERS, always_xy=True)
    return tr.transform(np.asarray(lon, float), np.asarray(lat, float))
