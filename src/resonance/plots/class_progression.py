"""Per-reaction-family progression through training stages (three panels).

(a) training reactions per family, (b) student minus teacher accuracy per
family and stage, (c) accuracy trajectories per family with the pooled overall.

Input (``class_progression.json``)::

    {"stages": ["base", "S1", "S2", "S3"],
     "teacher": "Kimi K3",                       # label only
     "families": [
        {"name": "acylation", "short": "Acyl.", "n_train": 7700, "n_test": 969,
         "correct": {"teacher": 812, "base": 26, "S1": 426, "S2": 531, "S3": 563}},
        ...],
     "overall": {"n": 4819, "correct": {"base": ..., "S1": ..., ...}}}

``correct`` values are counts out of ``n_test``; every stage plus ``teacher``
is required for each family. Families plot in the given order; colours come
from ``style.FAMILY_COLORS`` by name, or a family's ``"color"``.

Optional layout keys: ``figsize`` ([w, h] in inches), ``legend_ncol`` (panel C legend
columns, default 1) and ``count_size`` (panel A count-label font size).
"""
import math
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.lines import Line2D

from . import style
from .schema import DataError, require, require_list


def validate(data):
    require(data, ["stages", "families", "overall"], "class_progression")
    stages = require_list(data, "stages", "class_progression")
    for i, fam in enumerate(require_list(data, "families", "class_progression")):
        where = f"families[{i}]"
        require(fam, ["name", "n_train", "n_test", "correct"], where)
        require(fam["correct"], ["teacher", *stages], f"{where}.correct")
        if any(not 0 <= fam["correct"][s] <= fam["n_test"] for s in ["teacher", *stages]):
            raise DataError(f"{where}: correct counts must lie in [0, n_test]")
    require(data["overall"], ["n", "correct"], "overall")
    require(data["overall"]["correct"], stages, "overall.correct")


def _color(fam):
    return fam.get("color") or style.FAMILY_COLORS.get(fam["name"], style.MUTED)


def plot(data):
    LBL, TCK = style.FS_LABEL + 2, style.FS_TICK + 2   # axis labels / tick labels, in black
    stages, fams = data["stages"], data["families"]
    short = [f.get("short", f["name"]) for f in fams]
    acc = np.array([[100 * f["correct"][s] / f["n_test"] for s in stages] for f in fams])
    teacher = np.array([100 * f["correct"]["teacher"] / f["n_test"] for f in fams])
    delta = acc - teacher[:, None]
    overall = [100 * data["overall"]["correct"][s] / data["overall"]["n"] for s in stages]

    figsize = data.get("figsize", (style.TEXTWIDTH, style.TEXTWIDTH / ((1 + math.sqrt(5)) / 2)))
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.5], width_ratios=[1, 1.05],
                          hspace=0.03, wspace=0.04)
    ax_a = fig.add_subplot(gs[0, 0])
    right = gs[0, 1].subgridspec(1, 2, width_ratios=[1, 0.035], wspace=0.03)
    ax_b, cax = fig.add_subplot(right[0, 0]), fig.add_subplot(right[0, 1])
    bottom = gs[1, :].subgridspec(1, 2, width_ratios=[1.65, 1], wspace=0.02)
    ax_c, ax_leg = fig.add_subplot(bottom[0, 0]), fig.add_subplot(bottom[0, 1])
    ax_leg.set_axis_off()

    # (a) training counts.
    y = np.arange(len(fams))
    counts = [f["n_train"] for f in fams]
    ax_a.bar(y, counts, width=0.72, color=[_color(f) for f in fams], zorder=3)
    ax_a.set_xticks(y, short, rotation=50, ha="right", fontsize=style.FS_LEG)
    ax_a.set_ylim(0, max(counts) * 1.2)
    ax_a.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: "0" if v == 0 else f"{v / 1000:g}k"))
    ax_a.set_ylabel("Training reactions", fontsize=LBL, color="black")
    ax_a.yaxis.label.set_y(0.4)   # sits lower so the panel letter clears it
    ax_a.set_xlim(-0.6, len(fams) - 0.4)
    ax_a.tick_params(axis="x", length=0)
    style.open_axes(ax_a)
    for xi, count in zip(y, counts, strict=True):
        ax_a.text(xi, count + max(counts) * 0.015, f"{count / 1000:.1f}k", ha="center",
                  va="bottom", fontsize=data.get("count_size", style.FS_ANNOT + 1.5))

    # (b) student minus teacher; green = student ahead.
    bound = max(1, int(np.ceil(np.abs(delta).max())))
    norm = mpl.colors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    im = ax_b.imshow(delta, cmap=style.TAN_GREEN, norm=norm, aspect="auto",
                     interpolation="nearest")
    ax_b.set_xticks(np.arange(len(stages)), stages)
    ax_b.set_yticks(y, short, fontsize=style.FS_LEG)
    ax_b.tick_params(axis="y", length=0)
    ax_b.set_xticks(np.arange(-0.5, len(stages), 1), minor=True)
    ax_b.set_yticks(np.arange(-0.5, len(fams), 1), minor=True)
    ax_b.grid(which="minor", color="white", linewidth=0.6)
    ax_b.tick_params(which="minor", length=0)
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, cax=cax, ticks=[-bound, 0, bound])
    cbar.ax.tick_params(labelsize=style.FS_LEG, length=2)
    cbar.set_label(f"{data.get('student', 'Student')} − {data.get('teacher', 'Teacher')} (pp)",
                   fontsize=TCK, color="black")
    # matplotlib stub declares the colorbar outline as Spines, not Spine.
    cbar.outline.set_visible(False)  # ty: ignore[call-non-callable]

    # (c) trajectories; overall drawn thick and muted, behind.
    x = np.arange(len(stages))
    ax_c.plot(x, overall, "-", color=style.MUTED, lw=2.6, solid_capstyle="round", zorder=2)
    for fam, row in zip(fams, acc, strict=True):
        ax_c.plot(x, row, "-", color=_color(fam), lw=1.5, marker="o", ms=3.5,
                  markeredgecolor="white", markeredgewidth=0.7, zorder=4)
    ax_c.set_xticks(x, stages)
    ax_c.set_xlim(-0.12, len(stages) - 0.88)
    top = max(20, math.ceil(acc.max() / 20) * 20)
    ax_c.set_ylim(0, top)
    ax_c.set_yticks(np.arange(0, top + 1, 20))
    ax_c.set_ylabel("Top-1 accuracy (%)", fontsize=LBL, color="black")
    # The rotated y-label is taller than the panel; leaving it out of the layout lets the
    # axes fill the row instead of shrinking to contain the label.
    ax_c.yaxis.label.set_in_layout(False)
    style.open_axes(ax_c)
    handles = [Line2D([0], [0], color=style.MUTED, lw=2.6)] + [
        Line2D([0], [0], color=_color(f), lw=1.5, marker="o", ms=3.5,
               markeredgecolor="white", markeredgewidth=0.7) for f in fams]
    names = ["Overall"] + [f.get("legend", f.get("short", f["name"])) for f in fams]
    ax_leg.legend(handles, names, loc="center left", fontsize=10, handlelength=1.4,
                  ncol=data.get("legend_ncol", 1), columnspacing=0.8,
                  borderaxespad=0, handletextpad=0.4, labelspacing=0.3)
    for ax in (ax_a, ax_b, ax_c, cax):
        ax.tick_params(axis="both", which="major", labelsize=TCK, colors="black")
    ax_c.tick_params(axis="x", labelsize=LBL)   # stage names under panel C, at axis-label size
    # Panel letters: reserve title space during layout, then freeze the layout and place
    # each letter at the panel's left-most extent (tick labels and y-label included),
    # just above whichever is higher, the plot area or the y-label.
    # Panel c reserves no title row, so its axes extend to the top of the row.
    for ax in (ax_a, ax_b):
        ax.set_title(" ", fontsize=style.FS_LABEL + 3, pad=3)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    renderer = cast(Any, fig.canvas).get_renderer()   # Agg canvas (FigureCanvasBase lacks it)
    to_fig = fig.transFigure.inverted()
    for letter, ax in zip(("A.", "B.", "C."), (ax_a, ax_b, ax_c), strict=True):
        ybox, frame = ax.yaxis.get_tightbbox(renderer), ax.get_window_extent(renderer)
        assert ybox is not None
        x, y = to_fig.transform((min(ybox.x0, frame.x0), max(ybox.y1, frame.y1) + 3 * fig.dpi / 72))
        cap = 1 - (style.FS_LABEL + 3) / 72 / fig.get_figheight()   # keep the letter inside the figure
        fig.text(x, min(y, cap), letter, ha="left", va="bottom", fontsize=style.FS_LABEL + 3, fontweight="bold")
    return fig
