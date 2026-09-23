"""
Plotting utilities for benchmark accuracy comparison and GRPO training monitoring.

Benchmark comparison plots (model_results: Dict[str, dict] from evaluate()):
    plot_model_accuracy          — overall accuracy bar chart
    plot_accuracy_by_question_type — grouped bar per question type
    plot_accuracy_and_delta      — side-by-side accuracy + Δ vs baseline
    plot_delta_vs_baseline       — Δ bar chart per question type
    plot_accuracy_heatmap        — heatmap of per-type accuracy
    plot_lollipop_delta          — horizontal lollipop Δ chart

Training monitoring plots (metrics: List[dict] from GRPO trainer logs,
                           critic_rows: List[dict] from critic_logs/*.jsonl):
    plot_training_metrics        — 2×2: reward, frac_zero_std, grad_norm, loss/lr
    plot_completion_stats        — completion length and clipped ratio over steps
    plot_product_tiers           — tier rates, Tanimoto distribution, training trend
    plot_principle_scores        — mean±std and trajectory per constitutional principle
    plot_judge_flags             — judge information-leak flag rates
    plot_critic_product_correlation — critic score vs product accuracy (3-panel)
"""

import json
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker


def plot_model_accuracy(
    model_results: Dict[str, dict],
    title: str = "Overall Accuracy by Model",
    output_path: Optional[str] = None,
    colors: Optional[List[str]] = None,
) -> None:
    """Bar chart of overall accuracy for each model.

    Args:
        model_results: Mapping of model name → dict returned by evaluate().
        title: Plot title.
        output_path: If given, save the figure to this path instead of showing.
        colors: Bar colors in model order. If omitted, matplotlib defaults are used.
    """
    models = list(model_results.keys())
    accuracies = [model_results[m]["accuracy"] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(max(4, len(models) * 1.2), 5))
    bars = ax.bar(models, accuracies, color=colors) if colors else ax.bar(models, accuracies)
    ax.set_ylabel("Accuracy (%)")
    # ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.spines[["top", "right"]].set_visible(False)

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_accuracy_and_delta(
    model_results: Dict[str, dict],
    baseline_label: str,
    title_accuracy: str = "Overall Accuracy",
    title_delta: str = "Δ Accuracy vs Baseline",
    output_path: Optional[str] = None,
    colors: Optional[List[str]] = None,
    baseline_color: str = "#aaaaaa",
) -> None:
    """Side-by-side overall accuracy (left) and Δ vs baseline (right) in one figure.

    A shared color map is built once and applied to both panels: the baseline bar
    in the left panel uses *baseline_color*, and each non-baseline model gets the
    same color in both panels so the two charts are immediately cross-readable.

    Args:
        model_results: Mapping of model label → dict returned by evaluate().
        baseline_label: Key in model_results used as the zero-reference.
        title_accuracy: Title for the overall-accuracy subplot.
        title_delta: Title for the delta subplot.
        output_path: If given, save to this path instead of showing.
        colors: Colors for non-baseline models in order. Defaults to green palette.
        baseline_color: Color for the baseline bar in the accuracy panel.
    """
    import matplotlib.ticker as mtick
    import numpy as np

    if colors is None:
        colors = [
            "#1b6e3a",  # dark green
            "#3aab68",  # mid green
            "#86d4a4",  # light green
        ]

    all_labels = list(model_results.keys())
    compare_labels = [lbl for lbl in all_labels if lbl != baseline_label]

    # Shared color map — used identically in both panels
    color_map = {baseline_label: baseline_color}
    for i, label in enumerate(compare_labels):
        color_map[label] = colors[i % len(colors)]

    baseline = model_results[baseline_label]
    question_types = sorted(
        {qt for r in model_results.values() for qt in r["per_question_type"]}
    )

    n_models = len(all_labels)
    n_compare = len(compare_labels)
    n_types = len(question_types)

    fig_width = 2 + n_models * 1.2 + max(n_types * 1.5, 6)
    fig, (ax_acc, ax_delta) = plt.subplots(
        1, 2,
        figsize=(fig_width, 5),
        gridspec_kw={"width_ratios": [n_models, max(n_types * n_compare, 4)]},
    )

    # --- Left panel: overall accuracy ---
    accuracies = [model_results[m]["accuracy"] * 100 for m in all_labels]
    bars = ax_acc.bar(all_labels, accuracies, color=[color_map[lbl] for lbl in all_labels])
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title(title_accuracy)
    ax_acc.set_ylim(0, 100)
    ax_acc.set_xticks(range(n_models))
    ax_acc.set_xticklabels(all_labels, rotation=15, ha="right")
    ax_acc.spines[["top", "right"]].set_visible(False)
    for bar, acc in zip(bars, accuracies):
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # --- Right panel: delta vs baseline ---
    if not question_types:
        ax_delta.text(0.5, 0.5, "No per-question-type data", ha="center", va="center")
    else:
        x = np.arange(n_types)
        width = 0.8 / n_compare
        qtype_labels = [qt.replace("_", "\n") for qt in question_types]

        for i, label in enumerate(compare_labels):
            r = model_results[label]
            deltas = [
                (
                    r["per_question_type"].get(qt, {}).get("accuracy", 0.0)
                    - baseline["per_question_type"].get(qt, {}).get("accuracy", 0.0)
                ) * 100
                for qt in question_types
            ]
            ax_delta.bar(
                x + (i - n_compare / 2 + 0.5) * width,
                deltas,
                width=width,
                label=label,
                color=color_map[label],
            )

        ax_delta.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax_delta.set_xticks(x)
        ax_delta.set_xticklabels(qtype_labels, fontsize=11)
        ax_delta.set_ylabel("Δ Accuracy vs baseline (pp)")
        ax_delta.set_title(title_delta)
        ax_delta.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax_delta.legend(loc="upper right", fontsize=8)
        ax_delta.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_accuracy_by_question_type(
    model_results: Dict[str, dict],
    title: str = "Accuracy by Question Type",
    output_path: Optional[str] = None,
) -> None:
    """Grouped bar chart of per-question-type accuracy for each model.

    Args:
        model_results: Mapping of model name → dict returned by evaluate().
        title: Plot title.
        output_path: If given, save the figure to this path instead of showing.
    """
    import numpy as np

    models = list(model_results.keys())
    question_types = sorted(
        {qt for m in models for qt in model_results[m]["per_question_type"]}
    )

    if not question_types:
        print("No per-question-type data available.")
        return

    n_types = len(question_types)
    n_models = len(models)
    x = np.arange(n_types)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(6, n_types * 1.5), 5))

    for i, model in enumerate(models):
        per_type = model_results[model]["per_question_type"]
        accuracies = [
            per_type.get(qt, {}).get("accuracy", 0) * 100 for qt in question_types
        ]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, accuracies, width, label=model)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(question_types, rotation=20, ha="right")
    ax.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_delta_vs_baseline(
    model_results: Dict[str, dict],
    baseline_label: str,
    title: str = "Δ Accuracy vs Baseline",
    output_path: Optional[str] = None,
    colors: Optional[List[str]] = None,
) -> None:
    """Grouped bar chart of per-question-type accuracy delta vs a baseline model.

    Args:
        model_results: Mapping of model name → dict returned by evaluate().
        baseline_label: Key in model_results to use as the zero-reference baseline.
        title: Plot title.
        output_path: If given, save the figure to this path instead of showing.
        colors: List of colors for the non-baseline models, in order. Defaults
                to a green palette distinguishable from the blue CPT-run scheme.
    """
    import matplotlib.ticker as mtick
    import numpy as np

    # Default: green shades — clearly distinct from the blue CPT palette
    if colors is None:
        colors = [
            "#1b6e3a",  # dark green
            "#3aab68",  # mid green
            "#86d4a4",  # light green
        ]

    baseline = model_results[baseline_label]
    compare_labels = [lbl for lbl in model_results if lbl != baseline_label]

    question_types = sorted(
        {qt for r in model_results.values() for qt in r["per_question_type"]}
    )

    if not question_types:
        print("No per-question-type data available.")
        return

    n_types = len(question_types)
    n_compare = len(compare_labels)
    x = np.arange(n_types)
    width = 0.8 / n_compare

    qtype_labels = [qt.replace("_", "\n") for qt in question_types]

    fig, ax = plt.subplots(figsize=(max(10, n_types * 1.5), 5))

    for i, label in enumerate(compare_labels):
        r = model_results[label]
        deltas = [
            (
                r["per_question_type"].get(qt, {}).get("accuracy", 0.0)
                - baseline["per_question_type"].get(qt, {}).get("accuracy", 0.0)
            ) * 100
            for qt in question_types
        ]
        color = colors[i % len(colors)]
        ax.bar(
            x + (i - n_compare / 2 + 0.5) * width,
            deltas,
            width=width,
            label=label,
            color=color,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(qtype_labels, fontsize=11)
    ax.set_ylabel("\u0394 Accuracy vs baseline (pp)")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc="upper right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# New visualisations
# ---------------------------------------------------------------------------


def plot_accuracy_heatmap(
    model_results: Dict[str, dict],
    title: str = "Accuracy heatmap",
    output_path: Optional[str] = None,
    cmap: str = "Blues",
    annot_fontsize: int = 9,
) -> None:
    """Heatmap of per-question-type accuracy — models as rows, question types as columns.

    Cells are annotated with the raw accuracy percentage.  Colour encodes absolute
    accuracy, making cross-model and cross-question-type comparisons scannable at
    a glance without bar-chart clutter.

    Args:
        model_results: Mapping of model label → dict returned by evaluate().
        title: Figure title.
        output_path: If given, save the figure to this path instead of showing.
        cmap: Matplotlib colourmap name.  "Blues" works well for a single range;
              swap in "RdYlGn" if you prefer a diverging range around 50 %.
        annot_fontsize: Font size for the in-cell accuracy labels.
    """
    import numpy as np

    models = list(model_results.keys())
    question_types = sorted(
        {qt for r in model_results.values() for qt in r["per_question_type"]}
    )
    if not question_types:
        print("No per-question-type data available.")
        return

    data = np.array(
        [
            [
                model_results[m]["per_question_type"].get(qt, {}).get("accuracy", 0.0) * 100
                for qt in question_types
            ]
            for m in models
        ]
    )

    cell_w = max(1.2, len(question_types) * 0.9)
    cell_h = max(0.6, len(models) * 0.6)
    fig, ax = plt.subplots(figsize=(cell_w + 1.5, cell_h + 1.0))

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    ax.set_xticks(range(len(question_types)))
    ax.set_xticklabels(question_types, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.tick_params(length=0)

    # Annotate each cell with its value; switch text colour for readability
    thresh = (data.max() + data.min()) / 2
    for i in range(len(models)):
        for j in range(len(question_types)):
            colour = "white" if data[i, j] > thresh else "black"
            ax.text(
                j, i, f"{data[i, j]:.1f}%",
                ha="center", va="center",
                fontsize=annot_fontsize, color=colour,
            )

    fig.colorbar(im, ax=ax, label="Accuracy (%)", pad=0.02)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_lollipop_delta(
    model_results: Dict[str, dict],
    baseline_label: str,
    title: str = "Δ Accuracy vs baseline",
    output_path: Optional[str] = None,
    colors: Optional[List[str]] = None,
    pos_color: str = "#1b6e3a",
    neg_color: str = "#c0392b",
) -> None:
    """Horizontal lollipop chart of per-question-type accuracy delta vs a baseline.

    Each non-baseline model gets its own row per question type, drawn as a stem
    (line from zero) plus a dot at the delta value.  Positive deltas are coloured
    *pos_color*, negative ones *neg_color*, so improvement vs regression is
    immediately obvious without a legend entry.

    When there are multiple comparison models the rows are grouped by question type
    with a small gap between groups, and a legend distinguishes models by colour.

    Args:
        model_results: Mapping of model label → dict returned by evaluate().
        baseline_label: Key in model_results used as the zero-reference.
        title: Plot title.
        output_path: If given, save the figure to this path instead of showing.
        colors: Per-model dot/stem colors for multi-model comparisons.  Overrides
                the pos/neg colour logic when supplied.
        pos_color: Dot/stem colour for positive deltas (single comparison model).
        neg_color: Dot/stem colour for negative deltas (single comparison model).
    """
    import numpy as np

    baseline = model_results[baseline_label]
    compare_labels = [lbl for lbl in model_results if lbl != baseline_label]
    question_types = sorted(
        {qt for r in model_results.values() for qt in r["per_question_type"]}
    )
    if not question_types:
        print("No per-question-type data available.")
        return

    n_q = len(question_types)
    n_c = len(compare_labels)

    # Build y positions: group by question type, one row per comparison model
    group_gap = 0.4
    bar_h = 0.6
    y_ticks_mid: List[float] = []
    all_y, all_delta, all_color = [], [], []

    for qi, qt in enumerate(question_types):
        group_start = qi * (n_c * bar_h + group_gap)
        group_positions = [group_start + ci * bar_h for ci in range(n_c)]
        y_ticks_mid.append(float(np.mean(group_positions)))

        for ci, label in enumerate(compare_labels):
            r = model_results[label]
            delta = (
                r["per_question_type"].get(qt, {}).get("accuracy", 0.0)
                - baseline["per_question_type"].get(qt, {}).get("accuracy", 0.0)
            ) * 100
            all_y.append(group_positions[ci])
            all_delta.append(delta)
            if colors:
                all_color.append(colors[ci % len(colors)])
            else:
                all_color.append(pos_color if delta >= 0 else neg_color)

    fig_h = max(4, n_q * (n_c * bar_h + group_gap) * 0.45)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    for y, delta, color in zip(all_y, all_delta, all_color):
        ax.plot([0, delta], [y, y], color=color, linewidth=1.8, solid_capstyle="round")
        ax.scatter([delta], [y], color=color, s=55, zorder=3)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_ticks_mid)
    ax.set_yticklabels(question_types, fontsize=10)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{'+' if v > 0 else ''}{v:.0f} pp"
        )
    )
    ax.set_xlabel("Δ Accuracy vs baseline (percentage points)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="0.90", zorder=0)

    # Legend when there are multiple comparison models
    if n_c > 1:
        from matplotlib.lines import Line2D
        legend_colors = colors if colors else [pos_color] * n_c
        handles = [
            Line2D(
                [0], [0],
                marker="o", color=legend_colors[ci % len(legend_colors)],
                markersize=7, linewidth=1.8, label=label,
            )
            for ci, label in enumerate(compare_labels)
        ]
        ax.legend(handles=handles, fontsize=8, loc="lower right")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Additional benchmark evaluation plots
# ---------------------------------------------------------------------------


def plot_precision_recall_f1(
    model_results: Dict[str, dict],
    title: str = "Precision / Recall / F1 by Question Type",
    output_path: Optional[str] = None,
) -> None:
    """Grouped bar chart of Precision, Recall, and F1 per question type for each model.

    Only question types that have F1 data (requires sklearn during evaluate()) are shown.

    Args:
        model_results: Mapping of model name → dict returned by evaluate().
        title: Figure suptitle.
        output_path: If given, save instead of showing.
    """
    import numpy as np

    models = list(model_results.keys())
    question_types = sorted(
        {
            qt
            for m in models
            for qt, stats in model_results[m]["per_question_type"].items()
            if "f1" in stats
        }
    )
    if not question_types:
        print("No F1 data available — run evaluate() with sklearn installed.")
        return

    metrics = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]
    n_types = len(question_types)
    n_models = len(models)

    fig, axes = plt.subplots(
        1, n_types, figsize=(max(8, n_types * 3), 4.5), sharey=True, squeeze=False
    )
    axes_row = axes[0]

    for col, qt in enumerate(question_types):
        ax = axes_row[col]
        x = np.arange(len(metrics))
        width = 0.8 / n_models
        for i, model in enumerate(models):
            per_type = model_results[model]["per_question_type"].get(qt, {})
            vals = [per_type.get(m, 0.0) for m in metrics]
            ax.bar(
                x + (i - n_models / 2 + 0.5) * width,
                vals,
                width,
                label=model if col == 0 else "",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_title(qt.replace("_", "\n"), fontsize=9)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)

    axes_row[0].set_ylabel("Score")
    if n_models > 1:
        handles, labels = axes_row[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=8)

    fig.suptitle(title)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_answer_rate(
    model_results: Dict[str, dict],
    title: str = "Answer Rate by Question Type",
    output_path: Optional[str] = None,
) -> None:
    """Grouped bar chart of the fraction of questions answered per question type.

    A low answer rate means the model isn't producing parseable output — useful
    for catching format regressions early.

    Args:
        model_results: Mapping of model name → dict returned by evaluate().
        title: Plot title.
        output_path: If given, save instead of showing.
    """
    import numpy as np

    models = list(model_results.keys())
    question_types = sorted(
        {qt for m in models for qt in model_results[m]["per_question_type"]}
    )
    if not question_types:
        print("No per-question-type data available.")
        return

    n_types = len(question_types)
    n_models = len(models)
    x = np.arange(n_types)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(6, n_types * 1.5), 5))

    for i, model in enumerate(models):
        per_type = model_results[model]["per_question_type"]
        rates = [
            100
            * per_type.get(qt, {}).get("answered", 0)
            / max(per_type.get(qt, {}).get("total", 1), 1)
            for qt in question_types
        ]
        ax.bar(x + (i - n_models / 2 + 0.5) * width, rates, width, label=model)

    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Answer rate (%)")
    ax.set_title(title)
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(question_types, rotation=20, ha="right")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training monitoring plots  (consume GRPOTrainer metric-dict lists)
# ---------------------------------------------------------------------------


def _roll(arr: list, w: int) -> list:
    import numpy as np
    if len(arr) < w:
        return arr
    return list(np.convolve(arr, np.ones(w) / w, mode="valid"))


def _save_or_show(fig, output_path: Optional[str]) -> None:
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_training_metrics(
    metrics: List[dict],
    title: str = "Training Metrics",
    output_path: Optional[str] = None,
    rolling_window: int = 10,
    reward_max: Optional[float] = None,
) -> None:
    """2×2 dashboard of core GRPO training metrics over logged steps.

    Panels: total reward | primary vs secondary reward | frac_reward_zero_std | step time.

    Args:
        metrics: List of metric dicts as emitted by GRPOTrainer at each logging step.
        title: Figure suptitle.
        output_path: If given, save instead of showing.
        rolling_window: Moving-average window applied as an overlay on raw traces.
        reward_max: If given, draw a horizontal reference line at this reward ceiling.
    """
    steps = list(range(len(metrics)))
    reward = [m.get("reward", 0.0) for m in metrics]
    primary = [m.get("rewards/_logging_primary_reward/mean", 0.0) for m in metrics]
    secondary = [m.get("rewards/_secondary_reward_fn/mean", 0.0) for m in metrics]
    zero_std = [m.get("frac_reward_zero_std", 0.0) for m in metrics]
    step_time = [m.get("step_time", 0.0) for m in metrics]

    w = rolling_window
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title, fontsize=13)

    ax = axes[0, 0]
    ax.plot(steps, reward, alpha=0.35, color="steelblue", linewidth=0.8)
    if len(reward) >= w:
        ax.plot(steps[w - 1:], _roll(reward, w), color="steelblue", linewidth=2,
                label=f"{w}-step MA")
    if reward_max is not None:
        ax.axhline(reward_max, color="gray", linestyle="--", linewidth=0.8,
                   label=f"max={reward_max}")
    ax.set_title("Total reward")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0, 1]
    ax.plot(steps, primary, alpha=0.35, color="tomato", linewidth=0.8)
    if len(primary) >= w:
        ax.plot(steps[w - 1:], _roll(primary, w), color="tomato", linewidth=2, label="primary")
    ax.plot(steps, secondary, alpha=0.35, color="seagreen", linewidth=0.8)
    if len(secondary) >= w:
        ax.plot(steps[w - 1:], _roll(secondary, w), color="seagreen", linewidth=2, label="secondary")
    ax.set_title("Primary vs secondary reward")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 0]
    ax.plot(steps, zero_std, alpha=0.35, color="orange", linewidth=0.8)
    if len(zero_std) >= w:
        ax.plot(steps[w - 1:], _roll(zero_std, w), color="orange", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("frac_reward_zero_std (dead gradient groups)")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    ax.plot(steps, step_time, alpha=0.45, color="purple", linewidth=0.8)
    if len(step_time) >= w:
        ax.plot(steps[w - 1:], _roll(step_time, w), color="purple", linewidth=2)
    ax.set_title("Step time (seconds)")
    ax.set_ylabel("Seconds")
    ax.spines[["top", "right"]].set_visible(False)

    for ax in axes.flat:
        ax.set_xlabel("Training step")

    _save_or_show(fig, output_path)


def plot_completion_stats(
    metrics: List[dict],
    title: str = "Completion Lengths",
    output_path: Optional[str] = None,
    rolling_window: int = 10,
) -> None:
    """Mean/min/max completion token length and clipped ratio over training steps.

    Args:
        metrics: List of metric dicts from GRPOTrainer.
        title: Left-panel title.
        output_path: If given, save instead of showing.
        rolling_window: Moving-average window.
    """
    steps = list(range(len(metrics)))
    mean_len = [m.get("completions/mean_length", 0.0) for m in metrics]
    min_len = [m.get("completions/min_length", 0.0) for m in metrics]
    max_len = [m.get("completions/max_length", 0.0) for m in metrics]
    clipped = [m.get("completions/clipped_ratio", 0.0) for m in metrics]

    w = rolling_window
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.fill_between(steps, min_len, max_len, alpha=0.15, color="steelblue",
                     label="min–max range")
    ax1.plot(steps, mean_len, alpha=0.35, color="steelblue", linewidth=0.8)
    if len(mean_len) >= w:
        ax1.plot(steps[w - 1:], _roll(mean_len, w), color="steelblue", linewidth=2,
                 label=f"mean ({w}-step MA)")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Tokens")
    ax1.set_title(title)
    ax1.legend(fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(steps, clipped, alpha=0.35, color="tomato", linewidth=0.8)
    if len(clipped) >= w:
        ax2.plot(steps[w - 1:], _roll(clipped, w), color="tomato", linewidth=2)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Clipped ratio")
    ax2.set_title("Completion clipped ratio")
    ax2.set_ylim(0, 1)
    ax2.spines[["top", "right"]].set_visible(False)

    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# Critic diagnostic plots  (consume critic_logs/ JSONL directories)
# ---------------------------------------------------------------------------


def _load_critic_rows(critic_log_dir: str) -> List[dict]:
    """Load all JSONL rows from a critic log directory (recursive glob)."""
    import glob
    rows: List[dict] = []
    for path in sorted(glob.glob(f"{critic_log_dir}/**/*.jsonl", recursive=True)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    return rows


_PRODUCT_RE = re.compile(r"<product>(.*?)</product>", re.DOTALL | re.IGNORECASE)


def plot_product_tiers(
    critic_log_dir: str,
    title: str = "Product Tiered Reward",
    output_path: Optional[str] = None,
    rolling_window: int = 10,
) -> None:
    """Three-panel product reward diagnostic: tier rates, Tanimoto histogram, training trend.

    Reads `policy_trace` and `true_product_smiles` from critic log JSONL files,
    deduplicates by (step, completion_idx), and computes per-completion tier
    membership.  Requires rdkit.

    Args:
        critic_log_dir: Path to the critic_logs/ directory.
        title: Figure suptitle.
        output_path: If given, save instead of showing.
        rolling_window: Moving-average window for the training trend panel.
    """
    import numpy as np

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdMolDescriptors
    except ImportError:
        print("rdkit is required for plot_product_tiers")
        return

    all_rows = _load_critic_rows(critic_log_dir)
    seen: set = set()
    rows = []
    for r in all_rows:
        key = (r.get("step"), r.get("completion_idx"))
        if key not in seen:
            seen.add(key)
            rows.append(r)

    if not rows:
        print(f"No critic rows found in {critic_log_dir}")
        return

    has_tag, valid_smiles_flag, formula_match, exact_match = [], [], [], []
    tanimoto_scores: list = []
    step_data: dict = {}

    for row in rows:
        completion = row.get("policy_trace", "")
        gt = str(row.get("true_product_smiles", ""))
        step = row.get("step", 0)
        bucket = step_data.setdefault(step, {"n": 0, "tag": 0, "valid": 0})
        bucket["n"] += 1

        m = _PRODUCT_RE.search(completion)
        if not m:
            has_tag.append(0)
            valid_smiles_flag.append(0)
            formula_match.append(0)
            exact_match.append(0)
            continue
        has_tag.append(1)
        bucket["tag"] += 1

        pred_mol = Chem.MolFromSmiles(m.group(1).strip())
        gt_mol = Chem.MolFromSmiles(gt)
        if pred_mol is None or gt_mol is None:
            valid_smiles_flag.append(0)
            formula_match.append(0)
            exact_match.append(0)
            continue
        valid_smiles_flag.append(1)
        bucket["valid"] += 1

        fp_p = rdMolDescriptors.GetMorganFingerprintAsBitVect(pred_mol, 2, 2048)
        fp_g = rdMolDescriptors.GetMorganFingerprintAsBitVect(gt_mol, 2, 2048)
        tanimoto_scores.append(DataStructs.TanimotoSimilarity(fp_p, fp_g))
        fm = rdMolDescriptors.CalcMolFormula(pred_mol) == rdMolDescriptors.CalcMolFormula(gt_mol)
        formula_match.append(int(fm))
        exact_match.append(int(Chem.MolToSmiles(pred_mol) == Chem.MolToSmiles(gt_mol)))

    n = len(rows) or 1
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(title)

    labels = ["Has tag", "Valid SMILES", "Formula match", "Exact match"]
    values = [100 * sum(x) / n for x in [has_tag, valid_smiles_flag, formula_match, exact_match]]
    axes[0].bar(labels, values, color=["steelblue", "seagreen", "orange", "tomato"])
    axes[0].set_ylabel("% of completions")
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(values):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    axes[0].set_title("Overall tier rates")
    axes[0].spines[["top", "right"]].set_visible(False)

    if tanimoto_scores:
        axes[1].hist(tanimoto_scores, bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
        axes[1].axvline(np.mean(tanimoto_scores), color="tomato", linestyle="--",
                        label=f"mean={np.mean(tanimoto_scores):.3f}")
        axes[1].set_xlabel("Tanimoto similarity")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Tanimoto distribution (valid SMILES only)")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No valid SMILES yet", ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].set_title("Tanimoto distribution")
    axes[1].spines[["top", "right"]].set_visible(False)

    w = rolling_window
    if step_data:
        sorted_steps = sorted(step_data)
        tag_pct = [100 * step_data[s]["tag"] / max(step_data[s]["n"], 1) for s in sorted_steps]
        valid_pct = [100 * step_data[s]["valid"] / max(step_data[s]["n"], 1) for s in sorted_steps]
        axes[2].plot(sorted_steps, tag_pct, color="steelblue", alpha=0.3, linewidth=0.8)
        axes[2].plot(sorted_steps, valid_pct, color="seagreen", alpha=0.3, linewidth=0.8)
        if len(sorted_steps) >= w:
            axes[2].plot(sorted_steps[w - 1:], _roll(tag_pct, w),
                         color="steelblue", linewidth=2, label=f"has tag ({w}-step MA)")
            axes[2].plot(sorted_steps[w - 1:], _roll(valid_pct, w),
                         color="seagreen", linewidth=2, label=f"valid SMILES ({w}-step MA)")
        axes[2].set_xlabel("Training step")
        axes[2].set_ylabel("%")
        axes[2].set_ylim(0, 100)
        axes[2].set_title("Tag & valid SMILES rate over training")
        axes[2].legend(fontsize=8)
    axes[2].spines[["top", "right"]].set_visible(False)

    _save_or_show(fig, output_path)


def plot_principle_scores(
    critic_log_dir: str,
    principles: Optional[List[str]] = None,
    title: str = "Constitutional Reward by Principle",
    output_path: Optional[str] = None,
    rolling_window: int = 5,
) -> None:
    """Mean ± std bar chart and score trajectories per constitutional principle.

    Args:
        critic_log_dir: Path to the critic_logs/ directory.
        principles: Principle IDs to include. If None, auto-detected from logs.
        title: Figure suptitle.
        output_path: If given, save instead of showing.
        rolling_window: Moving-average window for the trajectory panel.
    """
    import numpy as np
    from collections import defaultdict

    rows = _load_critic_rows(critic_log_dir)
    if not rows:
        print(f"No critic rows found in {critic_log_dir}")
        return

    principle_scores: dict = defaultdict(list)
    principle_steps: dict = defaultdict(list)
    for row in rows:
        pid = row.get("principle_id", "unknown")
        score = row.get("score")
        step = row.get("step", 0)
        if score is not None:
            principle_scores[pid].append(float(score))
            principle_steps[pid].append(step)

    if principles is None:
        principles = sorted(principle_scores.keys())
    if not principles:
        print("No principle data found.")
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    means = [float(np.mean(principle_scores.get(p, [0]))) for p in principles]
    stds = [float(np.std(principle_scores.get(p, [0]))) for p in principles]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    axes[0].bar(principles, means, yerr=stds, capsize=4,
                color=colors[:len(principles)], alpha=0.8)
    axes[0].set_ylabel("Mean score (0 / 0.5 / 1.0)")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Mean ± std per principle")
    axes[0].spines[["top", "right"]].set_visible(False)

    w = rolling_window
    for i, pid in enumerate(principles):
        if pid not in principle_steps:
            continue
        pairs = sorted(zip(principle_steps[pid], principle_scores[pid]))
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        c = colors[i % len(colors)]
        axes[1].plot(xs, ys, alpha=0.2, linewidth=0.8, color=c)
        if len(ys) >= w:
            axes[1].plot(xs[w - 1:], _roll(ys, w), linewidth=2, color=c, label=pid)

    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"Score trajectory ({w}-step MA)")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].spines[["top", "right"]].set_visible(False)

    _save_or_show(fig, output_path)


def plot_critic_product_correlation(
    critic_log_dir: str,
    principles: Optional[List[str]] = None,
    title: str = "Critic Score vs Product Prediction Accuracy",
    output_path: Optional[str] = None,
) -> None:
    """Three-panel correlation: constitutional critic score vs product accuracy tier.

    Panels:
      1. Mean critic score per product accuracy tier (no_tag → exact match)
      2. Tanimoto similarity vs mean critic score scatter + Pearson r
      3. Per-principle mean score: exact-match completions vs not

    Requires rdkit; scipy is used for Pearson r if available.

    Args:
        critic_log_dir: Path to the critic_logs/ directory.
        principles: Principle IDs. If None, auto-detected from logs.
        title: Figure suptitle.
        output_path: If given, save instead of showing.
    """
    import numpy as np
    from collections import defaultdict

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdMolDescriptors
    except ImportError:
        print("rdkit is required for plot_critic_product_correlation")
        return

    all_rows = _load_critic_rows(critic_log_dir)
    if not all_rows:
        print(f"No critic rows found in {critic_log_dir}")
        return

    if principles is None:
        principles = sorted({r["principle_id"] for r in all_rows if r.get("principle_id") is not None})

    # Build product tier per unique (step, completion_idx)
    comp_product: dict = {}
    for r in all_rows:
        key = (r.get("step"), r.get("completion_idx"))
        if key in comp_product:
            continue
        completion = r.get("policy_trace", "")
        gt = str(r.get("true_product_smiles", ""))
        rec: dict = {"tanimoto": None, "tier": "no_tag"}
        m = _PRODUCT_RE.search(completion)
        if m:
            pred_mol = Chem.MolFromSmiles(m.group(1).strip())
            gt_mol = Chem.MolFromSmiles(gt)
            if pred_mol is not None and gt_mol is not None:
                fp_p = rdMolDescriptors.GetMorganFingerprintAsBitVect(pred_mol, 2, 2048)
                fp_g = rdMolDescriptors.GetMorganFingerprintAsBitVect(gt_mol, 2, 2048)
                rec["tanimoto"] = DataStructs.TanimotoSimilarity(fp_p, fp_g)
                fm = rdMolDescriptors.CalcMolFormula(pred_mol) == rdMolDescriptors.CalcMolFormula(gt_mol)
                em = Chem.MolToSmiles(pred_mol) == Chem.MolToSmiles(gt_mol)
                rec["tier"] = "exact" if em else ("formula" if fm else "valid_wrong")
            else:
                rec["tier"] = "invalid_smiles"
        comp_product[key] = rec

    # Mean constitutional score per completion
    comp_const: dict = defaultdict(list)
    for r in all_rows:
        key = (r.get("step"), r.get("completion_idx"))
        s = r.get("score")
        if s is not None:
            comp_const[key].append(float(s))

    joint = [
        {**prod, "mean_const": float(np.mean(comp_const[key]))}
        for key, prod in comp_product.items()
        if comp_const.get(key)
    ]

    tier_order = ["no_tag", "invalid_smiles", "valid_wrong", "formula", "exact"]
    tier_labels = ["No tag", "Invalid\nSMILES", "Valid\n(wrong)", "Formula\nmatch", "Exact\nmatch"]
    tier_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#17becf", "#2ca02c"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=13)

    ax = axes[0]
    tier_means, tier_stds, tier_ns = [], [], []
    for t in tier_order:
        subset = [j["mean_const"] for j in joint if j["tier"] == t]
        tier_means.append(float(np.mean(subset)) if subset else 0.0)
        tier_stds.append(float(np.std(subset)) if subset else 0.0)
        tier_ns.append(len(subset))
    ax.bar(range(len(tier_order)), tier_means, yerr=tier_stds, capsize=4,
           color=tier_colors, alpha=0.85)
    ax.set_xticks(range(len(tier_order)))
    ax.set_xticklabels(tier_labels, fontsize=8)
    ax.set_ylabel("Mean constitutional score")
    ax.set_ylim(0, 1)
    ax.set_title("Mean critic score by product tier")
    ax.spines[["top", "right"]].set_visible(False)
    for i, (mv, sv, nv) in enumerate(zip(tier_means, tier_stds, tier_ns)):
        ax.text(i, mv + sv + 0.02, f"n={nv}", ha="center", fontsize=7)

    ax = axes[1]
    valid_joint = [j for j in joint if j["tanimoto"] is not None]
    if valid_joint:
        xs = [j["tanimoto"] for j in valid_joint]
        ys = [j["mean_const"] for j in valid_joint]
        tier_c = dict(zip(tier_order, tier_colors))
        ax.scatter(xs, ys, c=[tier_c[str(j["tier"])] for j in valid_joint],
                   alpha=0.5, s=20, linewidths=0)
        try:
            from scipy.stats import pearsonr
            r_val, p_val = pearsonr(xs, ys)
            z = np.polyfit(xs, ys, 1)
            ax.plot([0, 1], list(np.polyval(z, [0, 1])), color="black",
                    linewidth=1.5, linestyle="--")
            ax.set_title(f"Tanimoto vs critic score\n(r={r_val:.3f}, p={p_val:.3g}, n={len(xs)})")
        except ImportError:
            ax.set_title(f"Tanimoto vs critic score (n={len(xs)})")
        ax.set_xlabel("Tanimoto similarity (predicted vs true product)")
        ax.set_ylabel("Mean constitutional score")
    else:
        ax.text(0.5, 0.5, "No valid SMILES", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Tanimoto vs critic score")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    exact_set = {k for k, v in comp_product.items() if v["tier"] == "exact"}
    pid_exact: dict = defaultdict(list)
    pid_wrong: dict = defaultdict(list)
    for r in all_rows:
        key = (r.get("step"), r.get("completion_idx"))
        pid = r.get("principle_id")
        s = r.get("score")
        if pid not in principles or s is None:
            continue
        (pid_exact if key in exact_set else pid_wrong)[pid].append(float(s))

    x = np.arange(len(principles))
    bw = 0.35
    m_exact = [float(np.mean(pid_exact[p])) if pid_exact[p] else 0.0 for p in principles]
    m_wrong = [float(np.mean(pid_wrong[p])) if pid_wrong[p] else 0.0 for p in principles]
    ax.bar(x - bw / 2, m_exact, bw, label="Exact match", color="#2ca02c", alpha=0.85)
    ax.bar(x + bw / 2, m_wrong, bw, label="Not exact", color="#d62728", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(principles, fontsize=9)
    ax.set_ylabel("Mean constitutional score")
    ax.set_ylim(0, 1)
    ax.set_title("Critic score by principle:\nexact match vs not")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    _save_or_show(fig, output_path)


def plot_judge_flags(
    critic_log_dir: str,
    flag_patterns: Optional[Dict[str, str]] = None,
    title: str = "Judge Information-Leak Flags",
    output_path: Optional[str] = None,
) -> None:
    """Horizontal bar chart of how often the judge cites privileged information.

    Scans `critic_response` fields for patterns that indicate the judge is
    referencing SMARTS, atom maps, reaction names, or mechanism descriptions
    that the student model never had access to.

    Args:
        critic_log_dir: Path to the critic_logs/ directory.
        flag_patterns: Dict of {flag_name: regex_pattern}. If None, uses defaults.
        title: Figure title.
        output_path: If given, save instead of showing.
    """
    if flag_patterns is None:
        flag_patterns = {
            "cited_smarts":    r"\bsmarts\b",
            "cited_atom_maps": r"\batom[\s\-]?map(?:\s+number)?\b",
            "cited_rxn_name":  r"\breaction[\s\-]?name\b|\bnamed[\s\-]?reaction\b",
            "cited_mech_desc": r"\bmechanism[\s\-]?description\b",
        }

    compiled = {name: re.compile(pat, re.IGNORECASE) for name, pat in flag_patterns.items()}
    rows = _load_critic_rows(critic_log_dir)
    if not rows:
        print(f"No critic rows found in {critic_log_dir}")
        return

    total = len(rows)
    flag_counts = {name: 0 for name in compiled}
    any_leak = 0

    for row in rows:
        response = row.get("critic_response", "")
        fired = False
        for name, pat in compiled.items():
            if pat.search(response):
                flag_counts[name] += 1
                fired = True
        if fired:
            any_leak += 1

    all_flags = {**flag_counts, "any_info_leak": any_leak}
    labels = list(all_flags.keys())
    pcts = [100 * all_flags[k] / (total or 1) for k in labels]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.65)))
    colors = ["#d62728" if k == "any_info_leak" else "steelblue" for k in labels]
    bars = ax.barh(labels, pcts, color=colors)
    ax.set_xlabel("% of judge responses")
    ax.set_xlim(0, 110)
    ax.set_title(f"{title}\n(n={total} critic rows)")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, pct in zip(bars, pcts):
        ax.text(pct + 1, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=9)

    _save_or_show(fig, output_path)