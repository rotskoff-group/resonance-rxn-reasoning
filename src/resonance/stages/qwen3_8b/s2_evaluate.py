"""Score completed S2 USPTO results with the repository's standard evaluator.

Ported from run_dir('s2_8b_bench')/scripts/evaluate.py (run by hand after the benchmark).

    python -m resonance.stages.qwen3_8b.s2_evaluate [--root DIR]
"""
import argparse
import hashlib
import json
from pathlib import Path

from resonance.evaluation import plots
from resonance.paths import run_dir
from resonance.stages.qwen3_8b._common import S2_MODEL
from resonance.workflows.evaluate import evaluate, print_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(run_dir('s2_8b_bench')),
                    help='Benchmark run directory holding outputs/results.jsonl.')
    ROOT = Path(ap.parse_args().root)
    source = ROOT / 'outputs/results.jsonl'
    metrics = evaluate(str(source))
    assert metrics['total'] == 5000 and metrics['skipped'] == 0, metrics
    metrics['results_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    metrics['checkpoint'] = str(S2_MODEL)
    (ROOT / 'outputs/evaluation.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print_results(metrics, str(source))
    out = ROOT / 'outputs/plots'
    out.mkdir(exist_ok=True)
    results = {'Kimi no-ID 8B S2 (2229)': metrics}
    plots.plot_model_accuracy(results, output_path=str(out / 'overall_accuracy.png'))
    plots.plot_accuracy_by_question_type(results, output_path=str(out / 'accuracy_by_question_type.png'))


if __name__ == '__main__':
    main()
