"""Run one deterministic quarter of the 27B base evaluation on one GPU pair.

Ported from run_dir('pipeline_27b')/scripts/evaluate_base_shard.py.
CLI (inside a 2-GPU Slurm allocation): python -m resonance.stages.qwen3_8_27b.evaluate_base_shard --shard i
"""
import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import time

from resonance.paths import run_dir
from resonance.stages.common import checkpoint
from resonance.stages.qwen3_8_27b.config import PROMPTS, STAGE_DIR, traces_path
from resonance.stages.qwen3_8_27b.evaluate_local import BENCHMARK_BIN, ROOT, benchmark_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, choices=range(4), required=True)
    args = ap.parse_args()

    from resonance.pipelines.traces.validate_no_id import validate
    identity = validate(traces_path())
    run_id = 'no-id-' + identity['source_sha256'][:16]
    root = ROOT/'outputs'/run_id/'base'
    manifest = json.loads((root/'DOWNLOAD_COMPLETE.json').read_text())
    assert manifest['source_sha256'] == identity['source_sha256']
    model = checkpoint(root/'model')
    env = benchmark_env()
    allocated = os.environ.get('SLURM_JOB_GPUS')
    assert allocated, 'This launcher must run inside a Slurm GPU allocation'
    pair = allocated.split(',')
    assert len(pair) == 2 and len(set(pair)) == 2, f'Expected two allocated GPUs, got {allocated}'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert visible == allocated, f'Slurm GPU environment mismatch: {visible=} {allocated=}'
    print(f'Using Slurm-assigned physical GPUs {allocated} for shard {args.shard}', flush=True)

    with contextlib.ExitStack() as locks:
        for gpu in pair:
            lock = locks.enter_context((ROOT/f'outputs/gpu-{gpu}.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX)
        while True:
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                '--format=csv,noheader,nounits'], text=True).splitlines()
            free = {r.split(',')[0].strip(): int(r.split(',')[1]) for r in rows}
            if all(free.get(gpu, 0) >= 85000 for gpu in pair):
                break
            print(f'Waiting for Slurm GPUs {allocated} with >=85000 MiB free each', flush=True)
            time.sleep(30)
        env['CUDA_VISIBLE_DEVICES'] = allocated
        out = root/'eval-shards-4'/f'shard-{args.shard}'
        subprocess.run([str(BENCHMARK_BIN/'python'), str(STAGE_DIR/'generate.py'),
            '--stage', 'eval', '--model', str(model),
            '--data', str(run_dir('uspto')/'test.json'),
            '--prompts', str(PROMPTS), '--out', str(out), '--tp', '2',
            '--shard', str(args.shard), '--shards', '4'], env=env, check=True)


if __name__ == '__main__':
    main()
