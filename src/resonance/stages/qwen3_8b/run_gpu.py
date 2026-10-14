"""Run the no-ID S1 benchmark or four independent rollout workers safely.

Ported from run_dir('s1_8b')/scripts/run_gpu.py.

    python -m resonance.stages.qwen3_8b.run_gpu {benchmark,rollouts}

Takes the shared local GPU lock (RESONANCE_GPU_LOCK_DIR), waits for free cards, then runs
the benchmark workflow on one NVLink pair, or four TP=1 `sample_rollouts` workers and merges
their shards into rollouts.jsonl + COMPLETE.json.
"""
import argparse
import fcntl
import json
import os
import subprocess
import time

from resonance.paths import run_dir
from resonance.stages.common import checkpoint, write_json
from resonance.stages.qwen3_8b._common import LOCK_DIR, PY, REPO, S1_MODEL, TRAIN_JSON, workflow_cmd

ROOT = run_dir('s1_8b')
MODEL = S1_MODEL

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['benchmark', 'rollouts'])
    args = ap.parse_args()
    checkpoint(MODEL)
    # Launch after S1 training (afterok) so training and the final save succeeded.
    # save_model writes consolidated weights, but no root trainer_state.json.
    out = ROOT/'outputs'/args.action
    out.mkdir(parents=True, exist_ok=True)
    (ROOT/'logs').mkdir(parents=True, exist_ok=True)
    with (LOCK_DIR/'local-evaluation.lock').open('a') as lock:
        print('Waiting for shared local GPU lock', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        while True:
            rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                           '--format=csv,noheader,nounits'],
                                          text=True).splitlines()
            free = {int(row.split(',')[0]) for row in rows if int(row.split(',')[1]) >= 88000}
            if args.action == 'benchmark':
                pairs = [p for p in [(0,1),(2,3),(4,5),(6,7)] if set(p) <= free]
                devices = list(pairs[0]) if pairs else []
            else:
                devices = sorted(free)[:4] if len(free) >= 4 else []
            if devices:
                break
            print('Waiting for free local GPUs', flush=True)
            time.sleep(60)
        env = {**os.environ, 'PYTHONPATH':str(REPO/'src'), 'VLLM_WORKER_MULTIPROC_METHOD':'spawn',
               'TOKENIZERS_PARALLELISM':'false'}
        print(f'{args.action}: model={MODEL}, GPUs={devices}', flush=True)
        if args.action == 'benchmark':
            subprocess.run([PY,*workflow_cmd('benchmark','s1_benchmark',f'hydra.run.dir={out}/hydra')],env={**env,'CUDA_VISIBLE_DEVICES':','.join(map(str,devices))},check=True,cwd=str(REPO))
        else:
            children = []
            try:
                for shard, gpu in enumerate(devices):
                    logfile = (ROOT/f'logs/rollout_worker_{shard}.log').open('a')
                    try:
                        child = subprocess.Popen([PY,'-m','resonance.stages.qwen3_8b.sample_rollouts',
                                                '--model',str(MODEL),
                                                '--shard',str(shard),'--nshards','4',
                                                '--out',str(out/f'rollouts.shard{shard}-of-4.jsonl')],
                                                env={**env,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=logfile,stderr=subprocess.STDOUT)
                    finally:
                        logfile.close()
                    children.append(child)
                while any(p.poll() is None for p in children):
                    if any(p.poll() not in (None,0) for p in children):
                        raise RuntimeError('Rollout worker failed; see worker logs')
                    time.sleep(5)
                assert all(p.returncode == 0 for p in children)
            finally:
                for p in children:
                    if p.poll() is None:
                        p.terminate()
                for p in children:
                    try:
                        p.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
            expected = len(json.loads(TRAIN_JSON.read_text()))
            count = 0
            merged = out/'rollouts.jsonl'
            with merged.with_suffix('.jsonl.tmp').open('w') as dest:
                for shard in range(4):
                    with (out/f'rollouts.shard{shard}-of-4.jsonl').open() as src:
                        for line in src:
                            row = json.loads(line)
                            assert row['row_idx'] == count and len(row['responses']) == 8
                            dest.write(line)
                            count += 1
            assert count == expected, (count, expected)
            merged.with_suffix('.jsonl.tmp').replace(merged)
            write_json(out/'COMPLETE.json',{'model':str(MODEL),'rows':count,'samples_per_row':8,'temperature':0.9,'max_tokens':1024})

if __name__ == '__main__':
    main()
