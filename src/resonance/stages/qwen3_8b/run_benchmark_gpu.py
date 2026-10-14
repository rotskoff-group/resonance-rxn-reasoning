"""Acquire the existing local benchmark lock and an idle NVLink GPU pair.

Ported from run_dir('s3_8b')/scripts/run_benchmark_gpu.py; runs benchmark_replay.

    python -m resonance.stages.qwen3_8b.run_benchmark_gpu
"""
import fcntl
import os
import subprocess
import time

from resonance.stages.qwen3_8b._common import LOCK_DIR, PY, REPO

if __name__ == '__main__':
    with (LOCK_DIR/'local-evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        while True:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
            free={int(r.split(',')[0]) for r in rows if int(r.split(',')[1])>=88000}
            pairs=[p for p in [(0,1),(2,3),(4,5),(6,7)] if set(p)<=free]
            if pairs:
                break
            print('Waiting for an idle NVLink pair',flush=True)
            time.sleep(30)
        pair=pairs[0]
        print(f'Benchmark GPUs: {pair}',flush=True)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,pair)),'VLLM_WORKER_MULTIPROC_METHOD':'spawn',
             'TOKENIZERS_PARALLELISM':'false','PYTHONPATH':str(REPO/'src')}
        subprocess.run([PY,'-m','resonance.stages.qwen3_8b.benchmark_replay'],env=env,check=True)
