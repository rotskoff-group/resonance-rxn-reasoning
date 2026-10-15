"""
Plotting utilities for benchmark accuracy comparison.

Benchmark comparison plots (model_results: Dict[str, dict] from evaluate()):
    plot_model_accuracy          — overall accuracy bar chart
    plot_accuracy_by_question_type — grouped bar per question type
    plot_accuracy_and_delta      — side-by-side accuracy + Δ vs baseline
    plot_delta_vs_baseline       — Δ bar chart per question type
    plot_accuracy_heatmap        — heatmap of per-type accuracy
    plot_lollipop_delta          — horizontal lollipop Δ chart
"""

import matplotlib.pyplot as plt
import matplotlib.ticker


def plot_model_accuracy(
    model_results: dict[str, dict],
    title: str = "Overall Accuracy by Model",
    output_path: str | None = None,
    colors: list[str] | None = None,
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

    for bar, acc in zip(bars, accuracies, strict=True):
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
    model_results: dict[str, dict],
    baseline_label: str,
    title_accuracy: str = "Overall Accuracy",
    title_delta: str = "Δ Accuracy vs Baseline",
    output_path: str | None = None,
    colors: list[str] | None = None,
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
    for bar, acc in zip(bars, accuracies, strict=True):
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
    model_results: dict[str, dict],
    title: str = "Accuracy by Question Type",
    output_path: str | None = None,
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
    model_results: dict[str, dict],
    baseline_label: str,
    title: str = "Δ Accuracy vs Baseline",
    output_path: str | None = None,
    colors: list[str] | None = None,
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
    model_results: dict[str, dict],
    title: str = "Accuracy heatmap",
    output_path: str | None = None,
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
    model_results: dict[str, dict],
    baseline_label: str,
    title: str = "Δ Accuracy vs baseline",
    output_path: str | None = None,
    colors: list[str] | None = None,
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
    y_ticks_mid: list[float] = []
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

    for y, delta, color in zip(all_y, all_delta, all_color, strict=True):
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
    model_results: dict[str, dict],
    title: str = "Precision / Recall / F1 by Question Type",
    output_path: str | None = None,
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
    model_results: dict[str, dict],
    title: str = "Answer Rate by Question Type",
    output_path: str | None = None,
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
