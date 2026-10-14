"""Fig 3: Qwen3-8B by reaction class. (A) student minus teacher (Kimi K3) per class and stage;
(B) accuracy per class across stages, with the pooled overall.

Ported from run_dir('paper_figures')/scripts/fig3_classes.py. Reads class11_counts.json
(`resonance.paper.classes`); all 11 classes of the appendix table "Reaction classes in USPTO-50k".

    python -m resonance.paper.fig3 --data <dir> --out <figures>     # writes Fig_3.{pdf,png}
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from resonance.plots import style

MIN_N = 0          # every class is plotted
STAGES = [("8b_base", "base"), ("8b_s1", "S1"), ("8b_s2", "S2"), ("8b_s3", "S3")]
TEACHER = "kimi"
FS_TICK, FS_LABEL, FS_LEG = 10, 11, 9.5


def main():
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    style.use("default")
    d = json.loads((args.data / "class11_counts.json").read_text())
    classes = [c for c in d["classes"] if c["n_test"] >= MIN_N]
    acc = np.array([[100 * c["correct"][m] / c["n_test"] for m, _ in STAGES] for c in classes])
    teacher = np.array([100 * c["correct"][TEACHER] / c["n_test"] for c in classes])
    delta = acc - teacher[:, None]
    overall = [100 * d["overall"]["correct"][m] / d["overall"]["n"] for m, _ in STAGES]
    stage_labels = [s for _, s in STAGES]

    fig = plt.figure(figsize=(6.5, 2.7))
    # columns: heatmap, colourbar, spacer (colourbar labels + panel B y-label), trajectories, legend
    gs = fig.add_gridspec(1, 5, width_ratios=[0.9, 0.04, 0.78, 1.6, 0.66], wspace=0.05,
                          left=0.11, right=0.995, top=0.91, bottom=0.12)
    ax_a, cax, ax_b, ax_leg = (fig.add_subplot(gs[0, i]) for i in (0, 1, 3, 4))
    ax_leg.set_axis_off()

    # (A) student minus teacher; green = student ahead.
    bound = max(1, int(np.ceil(np.abs(delta).max())))   # symmetric, unclipped
    norm = mpl.colors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    im = ax_a.imshow(delta, cmap=style.TAN_GREEN, norm=norm, aspect="auto",
                     interpolation="nearest")
    y = np.arange(len(classes))
    ax_a.set_xticks(np.arange(len(STAGES)), stage_labels)
    ax_a.set_yticks(y, [c["short"] for c in classes])
    ax_a.set_xticks(np.arange(-0.5, len(STAGES), 1), minor=True)
    ax_a.set_yticks(np.arange(-0.5, len(classes), 1), minor=True)
    ax_a.grid(which="minor", color="white", linewidth=0.8)
    ax_a.tick_params(which="both", length=0, labelsize=FS_TICK, colors="black")
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, cax=cax, ticks=[-bound, 0, bound])
    cbar.ax.set_yticklabels([f"−{bound}", "0", f"+{bound}"])
    cbar.ax.tick_params(labelsize=FS_TICK - 1, length=2, colors="black")
    cbar.set_label("Resonance-8B − Kimi K3 (pp)", fontsize=FS_LABEL, color="black", labelpad=1)
    # matplotlib stub declares the colorbar outline as Spines, not Spine.
    cbar.outline.set_visible(False)  # ty: ignore[call-non-callable]

    # (B) trajectories; overall drawn black and dashed, behind.
    x = np.arange(len(STAGES))
    ax_b.plot(x, overall, color=style.BLACK, lw=2.0, ls=(0, (4, 2)), zorder=2)
    for c, row in zip(classes, acc, strict=True):
        ax_b.plot(x, row, "-", color=c["color"], lw=1.4, marker="o", ms=3.2,
                  markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    ax_b.set_xticks(x, stage_labels)
    ax_b.set_xlim(-0.12, len(STAGES) - 0.88)
    ax_b.set_ylim(0, 60)
    ax_b.set_yticks(range(0, 61, 20))
    ax_b.set_ylabel("Top-1 accuracy (%)", fontsize=FS_LABEL, color="black", labelpad=2)
    ax_b.tick_params(labelsize=FS_TICK, colors="black")
    style.open_axes(ax_b)
    handles = [Line2D([0], [0], color=style.BLACK, lw=2.0, ls=(0, (4, 2)))] + [
        Line2D([0], [0], color=c["color"], lw=1.4, marker="o", ms=3.2,
               markeredgecolor="white", markeredgewidth=0.6) for c in classes]
    ax_leg.legend(handles, ["Overall"] + [c["short"] for c in classes], loc="center left",
                  fontsize=FS_LEG, frameon=False, handlelength=1.5, handletextpad=0.4,
                  labelspacing=0.3, borderaxespad=0)

    for letter, ax in (("A.", ax_a), ("B.", ax_b)):
        ax.set_title(letter, loc="left", fontsize=FS_LABEL + 1, fontweight="bold", pad=4,
                     x=-0.55 if ax is ax_a else -0.3)
    style.save(fig, args.out, "Fig_3")


if __name__ == "__main__":
    main()
