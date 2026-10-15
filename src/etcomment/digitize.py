"""Digitize daily time-series panels from published figures.

The Golkar and de Oliveira (2026) supplement and main text include daily
time-series panels (Fig. S2, US-UTJ; Fig. 8, US-UiC) that show observed
(closure-corrected) eddy-covariance ET as dots and the OpenET ensemble and the
XGBoost prediction as lines.  The article is CC BY 4.0, so the figures can be
re-analyzed and redistributed with attribution.

This module recovers, for every calendar day on the panel:

* the observed ET value (center of the dark dot drawn at that day, if any),
* the value of a colored line series (OpenET, XGBoost) at that day.

The approach is deliberately simple and transparent:

1. Axis calibration is linear and is fitted to tick-mark pixel positions
   (``AxisCalibration``).
2. Observed dots are found with a "dark and unsaturated" color mask.  For each
   day the single pixel column through that day's x-position is scanned; a
   vertical run of dark pixels at least ``min_dot_frac`` of the dot diameter
   long is accepted as that day's dot (neighboring days' dots are ~1 day
   apart and cut that column only in a short chord, so they are rejected).
3. Line series are found with a color mask.  The value at a day is the center
   of the colored run in that day's column, corrected at local peaks/troughs
   for the line half-width.

Every digitized value carries a pixel-level uncertainty of roughly half a dot
or half a line width; ``digitize_panel`` reports it in data units.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from PIL import Image

Mask = Callable[[np.ndarray], np.ndarray]


# ----------------------------------------------------------------------------
# Color masks
# ----------------------------------------------------------------------------
def dark_unsaturated(max_value: int = 130, max_chroma: int = 45) -> Mask:
    """Mask for gray/black markers drawn with alpha over pastel backgrounds."""

    def _m(rgb: np.ndarray) -> np.ndarray:
        mx = rgb.max(axis=-1)
        mn = rgb.min(axis=-1)
        return (mx < max_value) & ((mx - mn) < max_chroma)

    return _m


def reddish(min_r: int = 150, min_rg: int = 50, min_rb: int = 50) -> Mask:
    def _m(rgb: np.ndarray) -> np.ndarray:
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        return (r > min_r) & ((r - g) > min_rg) & ((r - b) > min_rb)

    return _m


def bluish(min_b: int = 120, min_br: int = 80, max_r: int = 100) -> Mask:
    def _m(rgb: np.ndarray) -> np.ndarray:
        r, b = rgb[..., 0], rgb[..., 2]
        return (b > min_b) & ((b - r) > min_br) & (r < max_r)

    return _m


# ----------------------------------------------------------------------------
# Axis calibration
# ----------------------------------------------------------------------------
@dataclass
class AxisCalibration:
    """Linear pixel <-> data mapping fitted to tick marks.

    ``x_ticks`` maps pixel column -> pandas Timestamp; ``y_ticks`` maps pixel
    row -> data value.  A least-squares line is fitted to each set, and the
    residuals are kept so the calibration error can be reported.
    """

    x_ticks: dict[float, pd.Timestamp]
    y_ticks: dict[float, float]
    origin: pd.Timestamp = field(init=False)
    x_coef: tuple[float, float] = field(init=False)
    y_coef: tuple[float, float] = field(init=False)
    x_resid_px: float = field(init=False)
    y_resid_px: float = field(init=False)

    def __post_init__(self) -> None:
        self.origin = min(self.x_ticks.values())
        px = np.array(list(self.x_ticks.keys()), float)
        days = np.array([(t - self.origin).days for t in self.x_ticks.values()], float)
        bx, ax = np.polyfit(days, px, 1)  # px = ax + bx*days
        self.x_coef = (ax, bx)
        self.x_resid_px = float(np.max(np.abs(px - (ax + bx * days))))
        py = np.array(list(self.y_ticks.keys()), float)
        vy = np.array(list(self.y_ticks.values()), float)
        by, ay = np.polyfit(vy, py, 1)  # py = ay + by*value
        self.y_coef = (ay, by)
        self.y_resid_px = float(np.max(np.abs(py - (ay + by * vy))))

    # data -> pixel
    def x_of(self, date: pd.Timestamp) -> float:
        ax, bx = self.x_coef
        return ax + bx * (date - self.origin).days

    def y_of(self, value: float) -> float:
        ay, by = self.y_coef
        return ay + by * value

    # pixel -> data
    def value_of(self, row: float) -> float:
        ay, by = self.y_coef
        return (row - ay) / by

    @property
    def px_per_day(self) -> float:
        return self.x_coef[1]

    @property
    def px_per_unit(self) -> float:
        return abs(self.y_coef[1])


def find_tick_positions(mask: np.ndarray, band: slice, axis: int, min_count: int) -> list[float]:
    """Return centers of tick marks found in a thin band next to a spine.

    ``axis=0`` scans rows (y ticks, band is a column slice); ``axis=1`` scans
    columns (x ticks, band is a row slice).
    """
    if axis == 0:
        prof = mask[:, band].sum(axis=1)
    else:
        prof = mask[band, :].sum(axis=0)
    idx = np.where(prof >= min_count)[0]
    groups: list[list[int]] = []
    for i in idx:
        if groups and i - groups[-1][-1] <= 1:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return [0.5 * (g[0] + g[-1]) for g in groups]


# ----------------------------------------------------------------------------
# Run helpers
# ----------------------------------------------------------------------------
def _runs(col: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index pairs of True runs in a 1-D boolean array."""
    if not col.any():
        return []
    d = np.diff(np.concatenate([[0], col.astype(int), [0]]))
    starts = np.where(d == 1)[0]
    stops = np.where(d == -1)[0]
    return list(zip(starts.tolist(), stops.tolist()))


# ----------------------------------------------------------------------------
# Series extraction
# ----------------------------------------------------------------------------
def _merge_runs(runs: list[tuple[int, int]], bridge: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Merge dark runs separated by short gaps that are fully 'bridge' pixels.

    Lines drawn on top of a dot cut its dark run in two; when the cut is made
    of line-colored pixels and is no longer than ``max_gap`` the two pieces
    are re-joined.
    """
    if not runs:
        return []
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        gap = s - pe
        if 0 < gap <= max_gap and bridge[pe:s].all():
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def extract_dots(
    rgb: np.ndarray,
    cal: AxisCalibration,
    dates: Sequence[pd.Timestamp],
    mask_fn: Mask,
    dot_diameter_px: float,
    min_dot_frac: float = 0.6,
    max_dot_frac: float = 1.5,
    row_limits: tuple[int, int] | None = None,
    bridge_fns: Sequence[Mask] = (),
    max_gap_px: int = 0,
    probe_offsets: Sequence[int] = (-2, -1, 0, 1, 2),
) -> pd.Series:
    """Value of the dot centered on each date's column (NaN when no dot).

    For each date, columns at ``x_d + probe_offsets`` are scanned.  In each
    column, dark runs (optionally re-joined across line-colored gaps) whose
    length lies in [min_dot_frac, max_dot_frac] x diameter are dot
    candidates; the candidate closest to one diameter is kept, and the
    median center over the probed columns is returned.
    """
    mask = mask_fn(rgb)
    bridge = np.zeros_like(mask)
    for bf in bridge_fns:
        bridge |= bf(rgb)
    if row_limits is not None:
        lo, hi = row_limits
        mask[:lo, :] = False
        mask[hi:, :] = False
    solid = mask | bridge  # a dot partly covered by a line still counts
    rad = 0.5 * dot_diameter_px
    ri = int(np.ceil(rad))
    yy, xx = np.mgrid[-ri:ri + 1, -ri:ri + 1]
    disk = (yy**2 + xx**2) <= rad**2

    def coverage(x: int, y: int) -> float:
        y0, y1, x0_, x1 = y - ri, y + ri + 1, x - ri, x + ri + 1
        if y0 < 0 or x0_ < 0 or y1 > solid.shape[0] or x1 > solid.shape[1]:
            return 0.0
        return float(solid[y0:y1, x0_:x1][disk].mean())

    out = {}
    lo_len = min_dot_frac * dot_diameter_px
    hi_len = max_dot_frac * dot_diameter_px
    for d in dates:
        x0 = int(round(cal.x_of(d)))
        centers = []
        for off in probe_offsets:
            x = x0 + off
            if not (0 <= x < mask.shape[1]):
                continue
            runs = _runs(mask[:, x])
            if max_gap_px > 0:
                runs = _merge_runs(runs, bridge[:, x], max_gap_px)
            cands = [(s, e) for s, e in runs if lo_len <= (e - s) <= hi_len]
            long_runs = [(s, e) for s, e in runs if hi_len < (e - s) <= 3 * dot_diameter_px]
            if cands:
                s, e = min(cands, key=lambda r: abs((r[1] - r[0]) - dot_diameter_px))
                centers.append(0.5 * (s + e - 1))
            elif long_runs:
                # two dots (this day's and a neighbor's) merged in this
                # column: take the row where a full disk centered on this
                # column is best covered by marker pixels
                s, e = long_runs[0]
                rows = range(s + ri // 2, e - ri // 2)
                scores = [(coverage(x, y), y) for y in rows]
                best_score, best_row = max(scores)
                if best_score >= 0.85:
                    centers.append(float(best_row))
        # require a majority of probed columns to agree (rejects neighbor chords)
        if len(centers) >= max(1, (len(probe_offsets) + 1) // 2):
            out[d] = cal.value_of(float(np.median(centers)))
        else:
            out[d] = np.nan
    return pd.Series(out, name="value")


def extract_line(
    rgb: np.ndarray,
    cal: AxisCalibration,
    dates: Sequence[pd.Timestamp],
    mask_fn: Mask,
    line_width_px: float,
    row_limits: tuple[int, int] | None = None,
) -> pd.Series:
    """Value of a polyline (vertex per day) at each date.

    The center of the colored run at the day's column is used; at local
    extrema the outer edge minus half the line width is used instead, because
    the union of two thick segments meeting at a peak is centered below it.
    """
    mask = mask_fn(rgb)
    if row_limits is not None:
        lo, hi = row_limits
        mask[:lo, :] = False
        mask[hi:, :] = False
    half_day = 0.5 * cal.px_per_day
    vals = {}
    prev_center = None
    for d in dates:
        x = int(round(cal.x_of(d)))
        runs = _runs(mask[:, x])
        if not runs:
            vals[d] = np.nan
            continue
        # choose the run closest to the previous center (lines may cross other
        # colored artists); otherwise the longest run
        if prev_center is not None:
            s, e = min(runs, key=lambda r: abs(0.5 * (r[0] + r[1]) - prev_center))
        else:
            s, e = max(runs, key=lambda r: r[1] - r[0])
        center = 0.5 * (s + e - 1)
        # neighbors at segment midpoints
        mids = []
        for off in (-half_day, half_day):
            xm = int(round(x + off))
            if 0 <= xm < mask.shape[1]:
                rr = _runs(mask[:, xm])
                if rr:
                    sm, em = min(rr, key=lambda r: abs(0.5 * (r[0] + r[1]) - center))
                    mids.append(0.5 * (sm + em - 1))
        row = center
        if len(mids) == 2:
            if all(m > center for m in mids):  # peak (rows increase downward)
                row = s + 0.5 * line_width_px
            elif all(m < center for m in mids):  # trough
                row = (e - 1) - 0.5 * line_width_px
        vals[d] = cal.value_of(row)
        prev_center = center
    return pd.Series(vals, name="value")


def estimate_line_width(rgb: np.ndarray, mask_fn: Mask, x_cols: Sequence[int]) -> float:
    """Median vertical thickness of a line over nearly flat stretches."""
    mask = mask_fn(rgb)
    widths = []
    for x in x_cols:
        for s, e in _runs(mask[:, x]):
            widths.append(e - s)
    if not widths:
        raise ValueError("no line pixels found in the sampled columns")
    return float(np.percentile(widths, 25))


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(int)


def disk_peaks(
    rgb: np.ndarray,
    cal: AxisCalibration,
    mask_fn: Mask,
    dot_diameter_px: float,
    row_limits: tuple[int, int] | None = None,
    min_cover: float = 0.6,
) -> pd.DataFrame:
    """2-D marker detection: local maxima of disk coverage of the marker mask.

    Used as a second pass for dots that the column scan misses because two
    line series are drawn across them.  Returns one row per peak with the
    fractional-day offset of the peak from the nearest calendar day.
    """
    from scipy.ndimage import maximum_filter
    from scipy.signal import fftconvolve

    mask = mask_fn(rgb).astype(float)
    if row_limits is not None:
        lo, hi = row_limits
        mask[:lo, :] = 0
        mask[hi:, :] = 0
    r = int(round(0.5 * dot_diameter_px))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = ((yy**2 + xx**2) <= r * r).astype(float)
    disk /= disk.sum()
    cov = fftconvolve(mask, disk, mode="same")
    size = (max(3, r + 1), max(3, int(0.6 * cal.px_per_day)))
    peaks = (cov == maximum_filter(cov, size=size)) & (cov >= min_cover)
    ys, xs = np.where(peaks)
    rows = []
    for y, x in zip(ys, xs):
        days = (x - cal.x_coef[0]) / cal.x_coef[1]
        rows.append({
            "date": cal.origin + pd.Timedelta(days=int(round(days))),
            "day_offset": float(days - round(days)),
            "value": cal.value_of(float(y)),
            "cover": float(cov[y, x]),
        })
    return pd.DataFrame(rows)


def fill_from_peaks(obs: pd.Series, peaks: pd.DataFrame, max_offset: float = 0.3) -> tuple[pd.Series, list]:
    """Fill NaN days in ``obs`` with the best-covered 2-D peak on that day."""
    filled = obs.copy()
    added = []
    if peaks.empty:
        return filled, added
    p = peaks[peaks["day_offset"].abs() <= max_offset]
    for d, grp in p.groupby("date"):
        if d in filled.index and not np.isfinite(filled.loc[d]):
            best = grp.sort_values("cover", ascending=False).iloc[0]
            filled.loc[d] = best["value"]
            added.append((str(pd.Timestamp(d).date()), round(float(best["value"]), 3), round(float(best["cover"]), 2)))
    return filled, added
