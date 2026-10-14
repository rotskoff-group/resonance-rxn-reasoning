"""Accuracy change per condition as labelled dots, one panel per stage.

Input (``feedback_delta.json``; same ``points`` as intervention_scatter, tokens unused)::

    {"stages": ["S1", "S2", "S3"],
     "conditions": [{"id": "C4a_mild", "label": "Mild doubt", "color": "#00538a"}, ...],
     "points": [{"stage": "S1", "condition": "C4a_mild", "delta_accuracy_pp": -0.6,
                 "delta_se_pp": 0.4}, ...],                 # delta_se_pp optional
     "bound": 20,                                            # optional symmetric x-limit
     "error_z": 1.96,                                        # optional: bars = z * SE (default 1)
     "marker_scale": 0.7,                                    # optional dot size factor
     "xlabel": "$\\\\Delta$Acc (pp)"}                          # optional

Conditions are rows in the given order; each dot is annotated with its value.
"""
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from . import style
from .schema import DataError, require, require_list


def validate(data):
    stages = require_list(data, "stages", "feedback_delta")
    conditions = require_list(data, "conditions", "feedback_delta")
    for i, c in enumerate(conditions):
        require(c, ["id", "label"], f"conditions[{i}]")
    ids = {c["id"] for c in conditions}
    for i, p in enumerate(require_list(data, "points", "feedback_delta")):
        require(p, ["stage", "condition", "delta_accuracy_pp"], f"points[{i}]")
        if p["stage"] not in stages or p["condition"] not in ids:
            raise DataError(f"points[{i}]: unknown stage or condition")


def plot(data):
    stages, conditions = data["stages"], data["conditions"]
    points = {(p["stage"], p["condition"]): p for p in data["points"]}
    z = data.get("error_z", 1)
    extent = max(abs(p["delta_accuracy_pp"]) + z * p.get("delta_se_pp", 0) for p in points.values())
    bound = data.get("bound") or max(1, math.ceil(extent * 1.15))
    fig, axes = plt.subplots(1, len(stages), sharex=True, sharey=True, layout="constrained",
                             figsize=data.get("figsize", (style.TEXTWIDTH, 0.36 * style.TEXTWIDTH)))
    axes = [axes] if len(stages) == 1 else list(axes)
    for ax, stage in zip(axes, stages, strict=True):
        ax.axvline(0, color=style.ZERO_LINE, lw=1.0, ls="--", zorder=1)
        for k, cond in enumerate(conditions):
            p = points.get((stage, cond["id"]))
            if p is None:
                continue
            color = style.series_style(k, cond)[0]
            value = p["delta_accuracy_pp"]
            if "delta_se_pp" in p:
                ax.errorbar(value, k, xerr=z * p["delta_se_pp"], fmt="none", ecolor=color,
                            elinewidth=1.6, capsize=4, capthick=1.4, zorder=2)
            ax.plot(value, k, "o", color=color, ms=style.MARKER_SIZE * data.get("marker_scale", 1),
                    markeredgecolor="white",
                    markeredgewidth=0.7, zorder=3)
            ax.annotate(f"{value:+.1f}", (value, k), xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom", fontsize=style.FS_ANNOT, color=style.BLACK)
        ax.set_xlim(-bound, bound)
        ax.set_xticks([-bound, 0, bound])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:+g}"))
        ax.set_yticks(range(len(conditions)), [c["label"] for c in conditions])
        ax.set_ylim(len(conditions) - 0.5, -0.8)
        ax.set_title(stage, fontsize=style.FS_TITLE, pad=3)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[len(axes) // 2].set_xlabel(data.get("xlabel", r"$\Delta$Acc (pp)"), fontsize=style.FS_LABEL)
    return fig
