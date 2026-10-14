"""Top-1 accuracy of each training stage next to reference (frontier) models.

Input (``staged_accuracy.json``)::

    {"ylabel": "Top-1\\naccuracy (%)",            # optional
     "groups": [
        {"name": "Qwen3-8B", "kind": "stages",
         "bars": [{"label": "base", "correct": 59, "n": 5000}, ...]},
        {"name": "Frontier", "kind": "references",
         "bars": [{"label": "Opus\\n5", "correct": 3934, "n": 5000}, ...]}]}

Bars show correct/n with 95% Wilson intervals. ``kind`` picks the colour ramp;
any bar may set ``"color"``. A dashed divider separates consecutive groups.
"""
import math

import matplotlib.pyplot as plt
import numpy as np

from . import style
from .schema import DataError, require, require_list
from .stats import wilson

FS_BIG_LABEL = 15
FS_BIG_TICK = 13


def validate(data):
    for g, group in enumerate(require_list(data, "groups", "staged_accuracy")):
        require(group, ["kind", "bars"], f"groups[{g}]")
        if group["kind"] not in ("stages", "references"):
            raise DataError(f"groups[{g}].kind must be 'stages' or 'references'")
        for b, bar in enumerate(require_list(group, "bars", f"groups[{g}]")):
            require(bar, ["label", "correct", "n"], f"groups[{g}].bars[{b}]")
            if not 0 <= bar["correct"] <= bar["n"]:
                raise DataError(f"groups[{g}].bars[{b}]: need 0 <= correct <= n")


def plot(data):
    labels, values, lows, highs, colors, dividers = [], [], [], [], [], []
    for group in data["groups"]:
        ramp = style.STAGE_COLORS if group["kind"] == "stages" else style.REFERENCE_COLORS
        if labels:
            dividers.append(len(labels) - 0.5)
        for i, bar in enumerate(group["bars"]):
            value = 100 * bar["correct"] / bar["n"]
            lo, hi = wilson(bar["correct"], bar["n"])
            labels.append(bar["label"])
            values.append(value)
            lows.append(value - lo)
            highs.append(hi - value)
            colors.append(bar.get("color") or ramp[i % len(ramp)])

    fig, ax = plt.subplots(figsize=(style.TEXTWIDTH, style.TEXTWIDTH / (1 + math.sqrt(5))),
                           layout="constrained")
    x = np.arange(len(values))
    ax.bar(x, values, width=0.65, color=colors, zorder=3, yerr=[lows, highs], capsize=2.5,
           error_kw=dict(lw=0.9, capthick=0.9, ecolor="#333333", zorder=4))
    for d in dividers:
        ax.axvline(d, color="#a6a6a6", linestyle=(0, (4, 3)), lw=1)
    ax.set_xticks(x, labels, fontsize=FS_BIG_TICK)
    ax.set_xlim(-0.6, len(values) - 0.4)
    top = max(20, math.ceil(max(v + h for v, h in zip(values, highs, strict=True)) / 20) * 20)
    ax.set_ylim(0, top)
    ax.set_yticks(np.arange(0, top + 1, 20))
    ax.tick_params(axis="y", labelsize=FS_BIG_TICK)
    ax.set_ylabel(data.get("ylabel", "Top-1\naccuracy (%)"), fontsize=FS_BIG_LABEL)
    style.open_axes(ax)
    return fig
