"""Plurality-vote accuracy as a function of the number of voted draws, by class.

Input (``plurality.json``)::

    {"ks": [1, 3, 5, 7, 9],
     "stages": ["S1", "S2", "S3"],
     "series": [{"stage": "S1", "label": "acylation", "n": 1107,
                 "accuracy_pct": [39.5, 47.1, 53.9, 57.0, 59.0],
                 "color": "#00538a"}, ...],             # color optional
     "legend_n": true}                                   # optional: show (n=...) in legend

``accuracy_pct[i]`` is plurality@``ks[i]`` over the class's ``n`` reactions,
voting over the first k independent draws. A series label keeps its colour
across stages.
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import style
from .schema import DataError, require, require_list


def validate(data):
    ks = require_list(data, "ks", "plurality")
    stages = require_list(data, "stages", "plurality")
    for i, s in enumerate(require_list(data, "series", "plurality")):
        require(s, ["stage", "label", "n", "accuracy_pct"], f"series[{i}]")
        if s["stage"] not in stages:
            raise DataError(f"series[{i}]: unknown stage {s['stage']!r}")
        if len(s["accuracy_pct"]) != len(ks):
            raise DataError(f"series[{i}]: accuracy_pct needs one value per k ({len(ks)})")


def plot(data):
    ks, stages = data["ks"], data["stages"]
    labels = list(dict.fromkeys(s["label"] for s in data["series"]))
    colors = {}
    for i, label in enumerate(labels):
        first = next(s for s in data["series"] if s["label"] == label)
        colors[label] = first.get("color") or style.CATEGORICAL[i % len(style.CATEGORICAL)]

    fig, axes = plt.subplots(1, len(stages), figsize=data.get("figsize", (style.TEXTWIDTH, 0.47 * style.TEXTWIDTH)),
                             sharey=True, layout="constrained")
    axes = [axes] if len(stages) == 1 else list(axes)
    top = max(max(s["accuracy_pct"]) for s in data["series"])
    for stage, ax in zip(stages, axes):
        for s in (s for s in data["series"] if s["stage"] == stage):
            ax.plot(ks, s["accuracy_pct"], "-", color=colors[s["label"]], lw=1.5 * style.MARKER_SIZE / 5,
                    marker="o", ms=style.MARKER_SIZE * 0.7, markeredgecolor="white", markeredgewidth=0.7)
        ax.set_title(stage, fontsize=style.FS_TITLE, pad=4)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_ylim(*data.get("ylim", (0, max(20, -(-top // 20) * 20))))
        if "yticks" in data:
            ax.set_yticks(data["yticks"])
        style.open_axes(ax)
    axes[0].set_ylabel(data.get("ylabel", "Plurality@k accuracy (%)"),
                       fontsize=style.FS_LABEL + data.get("ylabel_size_offset", 0))
    axes[len(axes) // 2].set_xlabel(data.get("xlabel", "Draws voted (k)"), fontsize=style.FS_LABEL)
    n_by_label = {}
    for s in data["series"]:
        n_by_label.setdefault(s["label"], s["n"])
    handles = [Line2D([0], [0], color=colors[l], lw=1.5 * style.MARKER_SIZE / 5, marker="o",
                      ms=style.MARKER_SIZE * 0.7,
                      markeredgecolor="white", markeredgewidth=0.7) for l in labels]
    names = [f"{l} (n={n_by_label[l]})" if data.get("legend_n", True) else l for l in labels]
    fig.legend(handles, names,
               loc="outside lower center", ncol=3, fontsize=style.FS_LEG,
               handlelength=1.6, columnspacing=1.2)
    return fig
