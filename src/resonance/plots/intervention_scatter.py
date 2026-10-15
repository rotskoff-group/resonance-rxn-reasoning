"""Accuracy change vs reasoning length for prompt interventions, one panel per stage.

Used for the reasoning-length prompts (commit / consider / think more).

Input (``length_prompts.json``)::

    {"stages": ["S1", "S2", "S3"],
     "baseline_label": "Neutral",              # optional; titled "S1 · Neutral: 30.9%"
     "conditions": [{"id": "B2_commit_first", "label": "Commit",
                     "color": "#10735d", "marker": "o"}, ...],   # color/marker optional
     "points": [{"stage": "S1", "condition": "B2_commit_first",
                 "delta_accuracy_pp": 0.26, "mean_reasoning_tokens": 486.9,
                 "delta_se_pp": 0.31, "tokens_se": 4.2,           # optional
                 "baseline_accuracy_pct": 32.06}, ...]}           # optional

``delta_accuracy_pp`` is intervention minus baseline accuracy in percentage
points. When ``delta_se_pp`` / ``tokens_se`` are present they are drawn as
±1 SE bars (e.g. from repeated samples per prompt).
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from . import style
from .schema import DataError, require, require_list

POINT_KEYS = ["stage", "condition", "delta_accuracy_pp", "mean_reasoning_tokens"]


def validate(data):
    stages = require_list(data, "stages", "intervention_scatter")
    conditions = require_list(data, "conditions", "intervention_scatter")
    for i, cond in enumerate(conditions):
        require(cond, ["id", "label"], f"conditions[{i}]")
    ids = {c["id"] for c in conditions}
    seen = set()
    for i, point in enumerate(require_list(data, "points", "intervention_scatter")):
        require(point, POINT_KEYS, f"points[{i}]")
        key = (point["stage"], point["condition"])
        if point["stage"] not in stages or point["condition"] not in ids:
            raise DataError(f"points[{i}]: unknown stage or condition {key}")
        if key in seen:
            raise DataError(f"points[{i}]: duplicate {key}")
        seen.add(key)


def _span(values, pad):
    lo, hi = min(values), max(values)
    margin = (hi - lo) * pad or 1
    return lo - margin, hi + margin


def plot(data):
    stages, conditions, points = data["stages"], data["conditions"], data["points"]
    styles = {c["id"]: style.series_style(i, c) for i, c in enumerate(conditions)}
    xs = [0] + [p["delta_accuracy_pp"] + s * p.get("delta_se_pp", 0)
                for p in points for s in (-1, 1)]
    ys = [p["mean_reasoning_tokens"] + s * p.get("tokens_se", 0)
          for p in points for s in (-1, 1)]

    fig, axes = plt.subplots(1, len(stages), figsize=data.get("figsize", (style.TEXTWIDTH, 0.46 * style.TEXTWIDTH)),
                             sharex=True, sharey=True, layout="constrained")
    axes = [axes] if len(stages) == 1 else list(axes)
    for stage, ax in zip(stages, axes, strict=True):
        ax.axvline(0, color=style.ZERO_LINE, lw=1, ls=(0, (4, 3)), zorder=0)
        ax.grid(axis="y", color=style.GRID, lw=0.6)
        stage_points = {p["condition"]: p for p in points if p["stage"] == stage}
        for cond in conditions:
            p = stage_points.get(cond["id"])
            if p is None:
                continue
            color, marker = styles[cond["id"]]
            if "delta_se_pp" in p or "tokens_se" in p:
                ax.errorbar(p["delta_accuracy_pp"], p["mean_reasoning_tokens"],
                            xerr=p.get("delta_se_pp"), yerr=p.get("tokens_se"),
                            fmt="none", ecolor=color, elinewidth=0.9, capsize=2, zorder=2)
            ax.scatter(p["delta_accuracy_pp"], p["mean_reasoning_tokens"], s=style.MARKER_SIZE ** 2 * 1.7,
                       color=color, marker=marker, edgecolors="white", linewidths=0.6,
                       zorder=3)
        title = stage
        baselines = {p.get("baseline_accuracy_pct") for p in stage_points.values()} - {None}
        if data.get("baseline_label") and len(baselines) == 1:
            title = f"{stage} · {data['baseline_label']}: {baselines.pop():.1f}%"
        ax.set_title(title, fontsize=style.FS_TITLE, pad=6)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _: "0" if abs(v) < 1e-10 else f"{v:+g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        style.open_axes(ax)
    axes[0].set_xlim(*_span(xs, 0.16))
    axes[0].set_ylim(*_span(ys, 0.18))
    axes[0].set_ylabel(data.get("ylabel", "Reasoning tokens (avg.)"), fontsize=style.FS_LABEL)
    axes[len(axes) // 2].set_xlabel(data.get("xlabel", r"$\Delta$ Acc vs baseline (pp)"),
                                    fontsize=style.FS_LABEL)
    handles = [Line2D([], [], marker=styles[c["id"]][1], color=styles[c["id"]][0],
                      linestyle="none", markersize=style.MARKER_SIZE + 1, label=c["label"])
               for c in conditions]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(conditions),
               fontsize=style.FS_LEG + 0.5, columnspacing=1.4, handletextpad=0.4)
    return fig
