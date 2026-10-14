"""Appendix backtracking figures from Fig_A_backtrack.json (`resonance.paper.backtrack`).

Ported from run_dir('paper_figures')/scripts/render_appendix_backtrack.py.

Fig_A1: backtracking in the base model, the SFT corpus and the default S1-S3 samples.
Fig_A2: accuracy change vs backtracking rate for the Fig 4 (length) and Fig 5 (feedback) prompts.

    python -m resonance.paper.fig_appendix --data <dir>/Fig_A_backtrack.json --out <figures>
"""
import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from resonance.plots import style

STAGES = ["S1", "S2", "S3"]
STAGE_MARKERS = {"S1": "o", "S2": "s", "S3": "^"}
BASELINE_COLOR = "#8a8981"
PANELS = [  # (accuracy figure, baseline condition, baseline label, SE multiplier)
    ("Fig_4", "C5_fresh", "Default (resample)", 1.0),
    ("Fig_5", "C1_review_full", "Neutral review", 1.96),
]


def fig_a1(d, out):
    rows = [("Qwen3-8B base", d["reference"]["Qwen3-8B base"], style.STAGE_COLORS[0]),
            ("SFT corpus", d["reference"]["SFT corpus (Kimi K3)"], style.DARKTAN)]
    rows += [(s, d["stages"][s]["C5_fresh"], style.STAGE_COLORS[i + 1])
             for i, s in enumerate(STAGES)]
    labels = [r[0] for r in rows][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(style.TEXTWIDTH, 2.0), sharey=True)
    for ax, key, xlabel, fmt in [
            (axes[0], "any_backtrack_pct", "Traces with any backtracking (%)", "{:.1f}%"),
            (axes[1], "mean_events_per_1k", "Backtracking events per 1k tokens", "{:.2f}")]:
        vals = [r[1][key] for r in rows][::-1]
        cols = [r[2] for r in rows][::-1]
        ax.barh(range(len(rows)), vals, color=cols, height=0.6)
        for y, v in enumerate(vals):
            ax.text(v, y, "  " + fmt.format(v), va="center", fontsize=style.FS_ANNOT,
                    color=style.BLACK)
        ax.set_xlim(0, max(vals) * 1.18)
        ax.set_xlabel(xlabel)
        ax.set_yticks(range(len(rows)), labels)
        style.open_axes(ax)
    axes[0].axhline(2.5, color=style.GRID, lw=0.8)
    axes[1].axhline(2.5, color=style.GRID, lw=0.8)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.22, top=0.96, wspace=0.12)
    return style.save(fig, out, "Fig_A1")


def fig_a2(d, out):
    fig, axes = plt.subplots(1, 2, figsize=(style.TEXTWIDTH, 3.3))
    for ax, (fig_id, base_id, base_label, z) in zip(axes, PANELS, strict=True):
        acc = d["accuracy"][fig_id]
        conds = [(base_id, base_label, BASELINE_COLOR)] + \
                [(c["id"], c["label"], c["color"]) for c in acc["conditions"]]
        pts = {(p["stage"], p["condition"]): p for p in acc["points"]}
        ax.axhline(0, color=style.ZERO_LINE, lw=0.7, zorder=0)
        for cid, _, color in conds:
            xs = [d["stages"][s][cid]["mean_events_per_1k"] for s in STAGES]
            ys = [pts[(s, cid)]["delta_accuracy_pp"] if (s, cid) in pts else 0.0 for s in STAGES]
            es = [z * pts[(s, cid)]["delta_se_pp"] if (s, cid) in pts else 0.0 for s in STAGES]
            ax.plot(xs, ys, color=color, lw=0.8, alpha=0.5, zorder=1)
            for s, x, y, e in zip(STAGES, xs, ys, es, strict=True):
                if e:
                    ax.errorbar(x, y, yerr=e, color=color, lw=0.8, capsize=0, zorder=2)
                ax.plot(x, y, STAGE_MARKERS[s], color=color, ms=style.MARKER_SIZE,
                        mec="white", mew=0.6, zorder=3)
        ax.set_xlabel("Backtracking events per 1k tokens")
        style.open_axes(ax)
        ax.legend(handles=[Line2D([], [], color=c, marker="o", lw=0, ms=style.MARKER_SIZE - 1,
                                  label=lab) for _, lab, c in conds],
                  loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
                  fontsize=style.FS_LEG - 0.5, handletextpad=0.1, columnspacing=0.8)
    axes[0].set_ylabel("Change in accuracy (pp)")
    axes[0].set_title("(a) Reasoning-length prompts", fontsize=style.FS_TITLE, loc="left")
    axes[1].set_title("(b) Feedback prompts", fontsize=style.FS_TITLE, loc="left")
    stage_handles = [Line2D([], [], color=style.BLACK, marker=m, lw=0, ms=style.MARKER_SIZE - 1,
                            label=s) for s, m in STAGE_MARKERS.items()]
    fig.legend(handles=stage_handles, loc="upper right", ncol=3, fontsize=style.FS_LEG,
               handletextpad=0.1, columnspacing=0.8)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.34, top=0.88, wspace=0.28)
    return style.save(fig, out, "Fig_A2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    style.use("default")
    d = json.load(open(args.data))
    for p in fig_a1(d, args.out) + fig_a2(d, args.out):
        print(p)


if __name__ == "__main__":
    main()
