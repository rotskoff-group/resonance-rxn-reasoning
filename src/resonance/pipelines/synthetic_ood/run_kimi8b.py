"""Benchmark one Kimi no-ID 8B stage on the forward synthetic alpha=0.5 split, then score it.

Waits for a free NVLink pair (per-GPU locks shared with the other local evals), runs the
resonance benchmark workflow per split with this model's USPTO benchmark settings, and
scores with score_opus.py. A split with a complete results.jsonl is skipped; a partial one
is moved aside, because the workflow opens results.jsonl with "w".

Ported from run_dir('synthetic_ood')/scripts/kimi8b_synthetic.py. The configs
(configs/paper/data/kimi8b_noid_*) keep every setting that ran; the paths they held are
supplied here as overrides from resonance.paths (MODELS below).

CLI: python -m resonance.pipelines.synthetic_ood.run_kimi8b {s1,s2,s3} [--dry-run]
"""
import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from resonance.paths import run_dir

ROOT = run_dir('synthetic_ood')
LOCKS = run_dir('pipeline_27b') / 'outputs'   # per-GPU locks shared with the other local evals
CONFIGS = Path(__file__).resolve().parents[2] / 'configs/paper/data'
PY = sys.executable
SUITES = ['synthetic_alpha_0.5']
MODELS = {
    's1': run_dir('traces') / 'outputs/model_0a_no_reaction_id',
    's2': run_dir('s2_8b') / 'outputs/model/checkpoint-2229',
    's3': run_dir('s3_8b') / 'outputs/benchmark_model',
}


def free_memory():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    return {int(r.split(',')[0]): int(r.split(',')[1]) for r in rows}


def acquire_pair():
    while True:
        for pair in [(0, 1), (2, 3), (4, 5), (6, 7)]:
            if not all(free_memory().get(g, 0) >= 85000 for g in pair):
                continue
            locks = contextlib.ExitStack()
            try:
                for g in pair:
                    lock = locks.enter_context((LOCKS / f'gpu-{g}.lock').open('a'))
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                locks.close()
                continue
            if all(free_memory().get(g, 0) >= 85000 for g in pair):
                return pair, locks
            locks.close()
        print('Waiting for a free, unlocked H100 NVLink pair', flush=True)
        time.sleep(60)


def line_count(path):
    return sum(1 for _ in path.open()) if path.exists() else 0


def benchmark_cmd(stage, suite, out):
    return [str(PY), '-m', 'resonance.workflows.benchmark',
            '--config-path', str(CONFIGS),
            '--config-name', f'kimi8b_noid_{stage}_{suite}',
            f'data_path={ROOT / "data" / suite / "test.json"}',
            f'inference.model_name={MODELS[stage]}',
            f'output_path={out / "results.jsonl"}',
            f'hydra.run.dir={out}/hydra']


def score_cmd(stage, suite, out):
    return [str(PY), '-m', 'resonance.pipelines.synthetic_ood.score_opus', '--suite', suite,
            '--out-dir', str(out), '--model', f'kimi8b_noid_{stage}', '--root', str(ROOT)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['s1', 's2', 's3'])
    ap.add_argument('--dry-run', action='store_true', help='print the commands, run nothing')
    args = ap.parse_args()
    stage = args.stage
    if args.dry_run:
        for suite in SUITES:
            out = ROOT / 'outputs/kimi8b_noid' / stage / suite
            print(' '.join(benchmark_cmd(stage, suite, out)))
            print(' '.join(score_cmd(stage, suite, out)))
        return
    pair, locks = acquire_pair()
    env = {**os.environ, 'WANDB_MODE': 'disabled',
           'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'TOKENIZERS_PARALLELISM': 'false',
           'CUDA_VISIBLE_DEVICES': ','.join(map(str, pair))}
    print(f'{stage}: GPUs {pair}', flush=True)
    with locks:
        for suite in SUITES:
            out = ROOT / 'outputs/kimi8b_noid' / stage / suite
            out.mkdir(parents=True, exist_ok=True)
            results = out / 'results.jsonl'
            rows = len(json.loads((ROOT / 'data' / suite / 'test.json').read_text()))
            if line_count(results) != rows:
                if results.exists():
                    results.rename(out / f'results.partial.{int(time.time())}.jsonl')
                subprocess.run(benchmark_cmd(stage, suite, out), env=env, check=True)
            subprocess.run(score_cmd(stage, suite, out), env=env, check=True)
            assert (out / 'report.json').exists(), f'{stage} {suite}: incomplete coverage'


if __name__ == '__main__':
    main()
