"""Figure style (Nature-like matplotlib settings) used by every figure.

Legibility rule. The manuscript is US Letter with 1-in margins, so the text
width is 6.5 in. Every figure is drawn at that final size (``FULL_WIDTH_IN``,
165.1 mm; heights as needed) and inserted 1:1, so the font sizes set here are
the printed sizes. All text is at least 7 pt: 8 pt base (axis labels, titles),
7.5 pt tick labels and legends, annotations >= 7 pt and bold lowercase panel
labels (a, b, c) at 10 pt. Data lines are >= 0.75 pt and axes 0.6 pt.
``check_text_size`` enforces the 7-pt floor before a figure is saved; legends
sit outside the data or in empty regions. Sans-serif (Helvetica/Arial), RGB,
no unnecessary boxes or grid lines, color-blind-safe palette, 600-dpi PNG plus
vector PDF.

Colors follow the validated categorical palette of the dataviz reference
(first three slots validate all-pairs for color-vision deficiency):
blue #2a78d6, orange #eb6834, aqua #1baf7a; ink #0b0b0b / #52514e; grid
#e1e0d9.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.text import Text

MM = 1 / 25.4  # inches per mm
FULL_WIDTH_IN = 6.5  # manuscript text width (US Letter, 1-in margins)
FULL_WIDTH_MM = 165.1
MIN_TEXT_PT = 7.0

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _sans() -> list[str]:
    available = {f.name for f in font_manager.fontManager.ttflist}
    prefs = ["Univers", "Helvetica", "Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
    return [p for p in prefs if p in available] or ["DejaVu Sans"]


def nature(fontsize: float = 8.0) -> None:
    """Nature-style defaults for figures printed at their drawn size.

    ``fontsize`` is the base size (axis labels, titles); tick labels and
    legends are 0.5 pt smaller.
    """
    small = fontsize - 0.5
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": _sans(),
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": small,
        "ytick.labelsize": small,
        "legend.fontsize": small,
        "legend.title_fontsize": small,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlepad": 4.0,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "legend.frameon": False,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.02, size: float = 10) -> None:
    """Bold lowercase Nature panel label in axes coordinates."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=size, fontweight="bold",
            va="bottom", ha="right", color=INK)


def check_text_size(fig, min_pt: float = MIN_TEXT_PT) -> float:
    """Return the smallest font size (pt) of the visible text in ``fig``.

    Figures are drawn at their printed size, so this is the printed size.
    Raises ValueError listing the offending strings if any visible, non-empty
    text is smaller than ``min_pt``.  Tick labels are filled in first.
    """
    fig.draw_without_rendering()
    sizes, small = [], []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        size = float(t.get_fontsize())
        sizes.append(size)
        if size < min_pt - 1e-9:
            small.append(f"{t.get_text()!r} ({size:g} pt)")
    if small:
        raise ValueError(f"text smaller than {min_pt:g} pt: " + "; ".join(small[:12]))
    return min(sizes) if sizes else float("nan")
