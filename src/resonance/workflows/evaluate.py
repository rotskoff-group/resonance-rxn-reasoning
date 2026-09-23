"""
Evaluation script for benchmark output.

Reads a JSONL file produced by benchmark.py, extracts the model's answer
from the \\boxed{} tag in each response, and compares it against the
ground-truth `answer` field.

Run with Hydra (mirrors benchmark.py):
  python -m resonance.workflows.evaluate results_path=/path/to/results.jsonl
"""

import json
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from resonance.evaluation import plots as P
from resonance.evaluation.extractors import TASK_PARSERS

try:
    from sklearn.metrics import precision_recall_fscore_support

    _sklearn_available = True
except ImportError:
    _sklearn_available = False


def extract_boxed(text: str) -> Optional[str]:
    """Extract the content of the last \\boxed{} in text, handling nested braces."""
    results = []
    i = 0
    marker = r"\boxed{"
    while True:
        pos = text.find(marker, i)
        if pos == -1:
            break
        start = pos + len(marker)
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start : j - 1])
        i = pos + 1
    if not results:
        return None
    return results[-1].strip()


def answers_match(
    predicted: str, ground_truth: str, answer_type: Optional[str]
) -> bool:
    """Return True if predicted matches ground_truth.

    For numeric answer types a float comparison with a small relative tolerance
    is used. All other types fall back to case-insensitive string equality.
    """
    if answer_type in ("numeric", "float", "int", "integer", "number"):
        try:
            pred_val = float(predicted.replace(",", ""))
            true_val = float(str(ground_truth).replace(",", ""))
            if true_val == 0:
                return abs(pred_val) < 1e-6
            return abs(pred_val - true_val) / abs(true_val) < 0.01
        except ValueError:
            return predicted.lower() == str(ground_truth).lower()
    return predicted.lower() == str(ground_truth).strip().lower()


def evaluate(results_path: str, question_type: Optional[str] = None) -> dict:
    """Evaluate a benchmark results JSONL file.

    Args:
        results_path: Path to the JSONL file from benchmark.py.
        question_type: If set, only evaluate records with this question_type.

    Returns:
        Dict with keys: total, answered, correct, skipped, accuracy,
        and per_question_type breakdown.
    """
    records = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if question_type:
        records = [r for r in records if r.get("question_type") == question_type]

    total = len(records)
    answered = 0
    correct = 0
    skipped = 0  # records without a ground-truth answer field
    per_type: dict[str, dict] = {}

    # compute F1 score using predicted and true labels for each question
    _NO_ANSWER = "__no_answer__"
    pairs_per_type: dict[str, list[tuple]] = {}

    for record in records:
        gt = record.get("answer")
        if gt is None:
            skipped += 1
            continue

        qt = record.get("question_type", "unknown")
        at = record.get("answer_type")

        if qt in TASK_PARSERS:
            extract_fn, match_fn = TASK_PARSERS[qt]
            predicted = extract_fn(record.get("response", ""))
        else:
            extract_fn, match_fn = extract_boxed, answers_match
            predicted = extract_fn(record.get("response", ""))

        if qt not in per_type:
            per_type[qt] = {"total": 0, "answered": 0, "correct": 0}
            pairs_per_type[qt] = []
        per_type[qt]["total"] += 1

        gt_label = str(gt).strip().lower()
        if predicted is None:
            pred_label = _NO_ANSWER
        else:
            answered += 1
            per_type[qt]["answered"] += 1
            pred_label = str(predicted).strip().lower()
            if match_fn(predicted, gt, at):
                correct += 1
                per_type[qt]["correct"] += 1

        pairs_per_type[qt].append((pred_label, gt_label))

    evaluable = total - skipped
    accuracy = correct / evaluable if evaluable > 0 else 0.0

    for qt, stats in per_type.items():
        ev = stats["total"]
        stats["accuracy"] = stats["correct"] / ev if ev > 0 else 0.0

        if _sklearn_available and qt in pairs_per_type:
            preds = [p for p, _ in pairs_per_type[qt]]
            trues = [t for _, t in pairs_per_type[qt]]
            unique_true = set(trues)
            unique_pred = set(preds)

            yn_set = {"yes", "no"}
            if (
                unique_true - {_NO_ANSWER} <= yn_set
                and unique_pred - {_NO_ANSWER} <= yn_set
            ):
                filtered = [
                    (t, p)
                    for t, p in zip(trues, preds)
                    if t != _NO_ANSWER and p != _NO_ANSWER
                ]
                if filtered:
                    f_trues, f_preds = zip(*filtered)
                else:
                    f_trues, f_preds = [], []
                avg = "binary"
                pos_label = "yes"
                p, r, f, _ = precision_recall_fscore_support(
                    list(f_trues),
                    list(f_preds),
                    average=avg,
                    pos_label=pos_label,
                    zero_division=0,
                    labels=["yes", "no"],
                )
            else:
                # Macro F1 for multi-class (integer counts, strings, etc.)
                avg = "macro"
                labels = sorted(unique_true)
                p, r, f, _ = precision_recall_fscore_support(
                    trues, preds, average=avg, zero_division=0, labels=labels
                )
            stats["precision"] = float(p)
            stats["recall"] = float(r)
            stats["f1"] = float(f)

    # Weighted by sample count to taken into account class imbalance when averaging across question types
    overall_precision = overall_recall = overall_f1 = None
    if _sklearn_available:
        qt_stats = [
            (s["total"], s.get("precision"), s.get("recall"), s.get("f1"))
            for s in per_type.values()
            if s.get("f1") is not None
        ]
        if qt_stats:
            total_weight = sum(n for n, *_ in qt_stats)
            overall_precision = sum(n * p for n, p, r, f in qt_stats) / total_weight
            overall_recall = sum(n * r for n, p, r, f in qt_stats) / total_weight
            overall_f1 = sum(n * f for n, p, r, f in qt_stats) / total_weight

    return {
        "total": total,
        "skipped": skipped,
        "answered": answered,
        "correct": correct,
        "accuracy": accuracy,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "per_question_type": per_type,
    }


def print_results(results: dict, results_path: str) -> None:
    print(f"\nResults: {results_path}")
    print(f"  Total records : {results['total']}")
    print(f"  Skipped (no GT): {results['skipped']}")
    print(f"  Answered       : {results['answered']}")
    print(f"  Correct        : {results['correct']}")
    print(f"  Accuracy       : {results['accuracy']:.1%}")
    if results.get("f1") is not None:
        print(f"  Precision      : {results['precision']:.3f}")
        print(f"  Recall         : {results['recall']:.3f}")
        print(f"  F1 (wtd avg)   : {results['f1']:.3f}")

    if results["per_question_type"]:
        print("\n  Per question type:")
        for qt, stats in sorted(results["per_question_type"].items()):
            base = (
                f"    {qt:<30} {stats['correct']:>4}/{stats['total']:<4}"
                f"  acc={stats['accuracy']:.1%}"
            )
            if "f1" in stats:
                base += f"  P={stats['precision']:.2f}  R={stats['recall']:.2f}  F1={stats['f1']:.2f}"
            print(base)


def _run(
    results_path: str, question_type: Optional[str], plot_cfg: Optional[dict]
) -> None:
    if not Path(results_path).exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    results = evaluate(results_path, question_type=question_type)
    print_results(results, results_path)

    if not plot_cfg:
        return

    output_dir = plot_cfg.get("output_dir")
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _out(name: str) -> Optional[str]:
        return str(Path(output_dir) / name) if output_dir else None

    primary_label = Path(results_path).stem
    model_results = {primary_label: results}

    # Merge optional additional result files for comparison plots
    for entry in plot_cfg.get("additional_results") or []:
        if isinstance(entry, dict):
            label = entry.get("label", Path(entry["path"]).stem)
            path = entry["path"]
        else:
            label = Path(str(entry)).stem
            path = str(entry)
        if Path(path).exists():
            model_results[label] = evaluate(path, question_type=question_type)
        else:
            print(f"Warning: additional_results path not found: {path}")

    # --- Single-model plots ---
    if plot_cfg.get("overall_accuracy", True):
        P.plot_model_accuracy(model_results, output_path=_out("overall_accuracy.png"))

    if plot_cfg.get("accuracy_by_question_type", True):
        P.plot_accuracy_by_question_type(
            model_results, output_path=_out("accuracy_by_question_type.png")
        )

    if plot_cfg.get("accuracy_heatmap"):
        P.plot_accuracy_heatmap(model_results, output_path=_out("accuracy_heatmap.png"))

    if plot_cfg.get("precision_recall_f1"):
        P.plot_precision_recall_f1(
            model_results, output_path=_out("precision_recall_f1.png")
        )

    if plot_cfg.get("answer_rate"):
        P.plot_answer_rate(model_results, output_path=_out("answer_rate.png"))

    # --- Multi-model delta/comparison plots ---
    baseline_label = plot_cfg.get("baseline_label") or primary_label

    if plot_cfg.get("lollipop_delta") and len(model_results) > 1:
        P.plot_lollipop_delta(
            model_results,
            baseline_label=baseline_label,
            output_path=_out("lollipop_delta.png"),
        )

    if plot_cfg.get("delta_vs_baseline") and len(model_results) > 1:
        P.plot_delta_vs_baseline(
            model_results,
            baseline_label=baseline_label,
            output_path=_out("delta_vs_baseline.png"),
        )

    if plot_cfg.get("accuracy_and_delta") and len(model_results) > 1:
        P.plot_accuracy_and_delta(
            model_results,
            baseline_label=baseline_label,
            output_path=_out("accuracy_and_delta.png"),
        )


def _hydra_main() -> None:
    @hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
    def _inner(cfg: DictConfig) -> None:
        raw = (
            OmegaConf.to_container(cfg.plot, resolve=True) if cfg.get("plot") else None
        )
        plot_cfg: Optional[dict] = raw if isinstance(raw, dict) else None
        _run(cfg.results_path, cfg.get("question_type"), plot_cfg)

    _inner()


def main():
    _hydra_main()


if __name__ == "__main__":
    main()
