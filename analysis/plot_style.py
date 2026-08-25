"""Shared plotting palette/style for analysis/ scripts (light-mode, static PNGs).

Palette values are the validated default from the dataviz skill
(references/palette.md) -- categorical order is a CVD-safety mechanism,
not cosmetic, so channel/label -> color assignments below are fixed, not
re-cycled per plot.
"""
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
DIVERGING_BLUE_RED = ("#2a78d6", "#f0efec", "#e34948")  # good(neg->pos) ... midpoint ... bad, or vice versa per use
STATUS_GOOD, STATUS_WARNING, STATUS_SERIOUS, STATUS_CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"

CHANNEL_COLOR = {f"ch{i}": CATEGORICAL[i - 1] for i in range(1, 9)}  # fixed identity, all 8 slots
LABEL_COLOR = {0: CATEGORICAL[0], 1: CATEGORICAL[7], 2: CATEGORICAL[3], 3: CATEGORICAL[2]}  # baseline/neg/neutral/pos
LABEL_NAME = {0: "baseline/rest", 1: "negative", 2: "neutral", 3: "positive"}


def style_axes(ax):
    """Recessive grid/axes, no top/right spines -- applied to every chart."""
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def new_figure(figsize):
    fig = plt.figure(figsize=figsize, facecolor=SURFACE)
    return fig
