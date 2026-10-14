"""Fig 7: one group per reaction class (the 11 classes and abbreviations of Fig 3); within each
group one bar per model: Qwen3.8-27B stages, then Kimi K3 and Opus 5.

Ported from run_dir('paper_figures')/scripts/fig7_grouped_bars.py (default variant). Reads
class11_counts.json (`resonance.paper.classes`).

    python -m resonance.paper.fig7 --data <dir> --out <figures>     # writes Fig_7.{pdf,png}
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from resonance.plots import style

# Stages: one blue gradient, light (base) to dark (S3); skip S2 is neutral grey (an ablation, not a stage).
# References keep their Fig 2 colours.
MODELS = [("27b_base", "Base", "#9ecae1"), ("27b_s1", "S1", "#4a98c9"),
          ("27b_s2", "S2", "#1f5fa8"), ("27b_s3", "S3", "#0b2e63"),
          ("27b_skip_s2", "skip S2", "#8c8c8c"),
          ("kimi", "Kimi K3", "#9c5a8e"), ("opus", "Opus 5", "#df7954")]
N_TRAINED = 5   # a gap separates the trained models from the references in each group
FS_TICK, FS_LABEL, FS_LEG = 13, 14, 13


def main():
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    style.use("default")
    models, counts, stem = MODELS, "class11_counts.json", "Fig_7"
    classes = json.loads((args.data / counts).read_text())["classes"]

    # Groups fill 70% of each class slot (30% whitespace between classes); drawn at the
    # 7.5 in width of Figs 4-6 so the bars stay thick.
    group, width = 0.7, 7.5
    fig, ax = plt.subplots(figsize=(width, 3.9))
    fig.subplots_adjust(left=0.08, right=0.997, top=0.87, bottom=0.21)
    gap = 0.35                                   # in bar widths
    slots = len(models) + gap
    w = group / slots
    offsets = np.array([k + (gap if k >= N_TRAINED else 0) for k in range(len(models))])
    offsets = (offsets - (slots - 1) / 2) * w
    x = np.arange(len(classes))
    for (key, _, color), off in zip(models, offsets, strict=True):
        v = [100 * c["correct"][key] / c.get("n_rows", {}).get(key, c["n_test"]) for c in classes]
        ax.bar(x + off, v, w * 0.97, color=color, edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_xticks(x, [c["short"] for c in classes], fontsize=FS_TICK, color="black",
                  rotation=35, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.55, len(classes) - 0.45)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25))
    ax.tick_params(axis="y", labelsize=FS_TICK, colors="black")
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=FS_LABEL, color="black")
    ax.yaxis.grid(True, color=style.GRID, lw=0.6, zorder=0)
    style.open_axes(ax)
    handles = [Patch(color=color, label=label) for _, label, color in models]
    fig.legend(handles=handles, loc="upper center", ncol=len(models), fontsize=FS_LEG, frameon=False,
               handlelength=1.1, columnspacing=0.9, handletextpad=0.35, bbox_to_anchor=(0.55, 1.0))
    style.save(fig, args.out, stem)


if __name__ == "__main__":
    main()
