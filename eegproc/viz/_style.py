"""Shared plotting style: palette, colour maps and axis helpers.

Colours follow one rule per job:

* identity (a few series)   -> categorical slots, in fixed order
* magnitude (power, score)  -> one-hue blue ramp, light -> dark
* polarity (ERD %, µV, ICA) -> blue <-> red with a neutral grey midpoint
* state (quality grades)    -> reserved status colours, always with a label
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIVERGING = ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#f0efec", "#f3a5a4", "#e66767", "#c93b3a",
             "#8c2323"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
GRADE_STATUS = {"good": STATUS["good"], "fair": STATUS["warning"], "poor": STATUS["serious"],
                "bad": STATUS["critical"]}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TRACE = "#3a3a38"

CMAP_SEQ = LinearSegmentedColormap.from_list("eegproc_seq", SEQ_BLUE)
# Same ramp but starting at the surface colour: for heatmaps where 0 means "nothing".
CMAP_SEQ0 = LinearSegmentedColormap.from_list("eegproc_seq0", [SURFACE] + SEQ_BLUE)
CMAP_DIV = LinearSegmentedColormap.from_list("eegproc_div", DIVERGING)

RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False, "legend.fontsize": 8,
    "font.family": "sans-serif", "lines.linewidth": 1.5, "lines.solid_capstyle": "round",
    "figure.dpi": 100, "savefig.dpi": 130, "savefig.bbox": "tight",
}


def style():
    """Context manager applying the eegproc style: ``with style(): ...``."""
    return mpl.rc_context(RC)


def grid_y(ax) -> None:
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def grid_x(ax) -> None:
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)


def finish(fig, show: bool = False, path=None):
    if path is not None:
        fig.savefig(path)
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return fig
