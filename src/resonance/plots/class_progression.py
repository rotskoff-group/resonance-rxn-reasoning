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
"""
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
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
    stages, fams = data["stages"], data["families"]
    short = [f.get("short", f["name"]) for f in fams]
    acc = np.array([[100 * f["correct"][s] / f["n_test"] for s in stages] for f in fams])
    teacher = np.array([100 * f["correct"]["teacher"] / f["n_test"] for f in fams])
    delta = acc - teacher[:, None]
    overall = [100 * data["overall"]["correct"][s] / data["overall"]["n"] for s in stages]

    fig = plt.figure(figsize=(style.TEXTWIDTH, style.TEXTWIDTH / ((1 + math.sqrt(5)) / 2)),
                     layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.45], width_ratios=[1, 1.05],
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
    ax_a.set_ylabel("Training reactions", fontsize=style.FS_LABEL)
    ax_a.set_xlim(-0.6, len(fams) - 0.4)
    ax_a.tick_params(axis="x", length=0)
    style.open_axes(ax_a)
    for xi, count in zip(y, counts):
        ax_a.text(xi, count + max(counts) * 0.015, f"{count / 1000:.1f}k", ha="center",
                  va="bottom", fontsize=style.FS_ANNOT)

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
    cbar.set_label(f"Acc(Student) − Acc({data.get('teacher', 'Teacher')}) (pp)",
                   fontsize=style.FS_LEG)
    cbar.outline.set_visible(False)

    # (c) trajectories; overall drawn thick and muted, behind.
    x = np.arange(len(stages))
    ax_c.plot(x, overall, "-", color=style.MUTED, lw=2.6, solid_capstyle="round", zorder=2)
    for fam, row in zip(fams, acc):
        ax_c.plot(x, row, "-", color=_color(fam), lw=1.5, marker="o", ms=3.5,
                  markeredgecolor="white", markeredgewidth=0.7, zorder=4)
    ax_c.set_xticks(x, stages)
    ax_c.set_xlim(-0.12, len(stages) - 0.88)
    top = max(20, math.ceil(acc.max() / 20) * 20)
    ax_c.set_ylim(0, top)
    ax_c.set_yticks(np.arange(0, top + 1, 20))
    ax_c.set_ylabel("Top-1 accuracy (%)", fontsize=style.FS_LABEL)
    ax_c.set_xlabel("Resonance Training Stages", fontsize=style.FS_LABEL)
    style.open_axes(ax_c)
    handles = [Line2D([0], [0], color=style.MUTED, lw=2.6)] + [
        Line2D([0], [0], color=_color(f), lw=1.5, marker="o", ms=3.5,
               markeredgecolor="white", markeredgewidth=0.7) for f in fams]
    names = ["Overall"] + [f.get("legend", f.get("short", f["name"])) for f in fams]
    ax_leg.legend(handles, names, loc="center left", fontsize=10, handlelength=1.4,
                  borderaxespad=0, handletextpad=0.4, labelspacing=0.3)
    for letter, ax in zip("abc", [ax_a, ax_b, ax_c]):
        ax.set_title(f"({letter})", loc="left", fontsize=style.FS_LABEL,
                     fontweight="bold", pad=3)
    return fig
