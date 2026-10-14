"""Serialize our local evaluations and wait for genuinely free H100s.

Ported from run_dir('pipeline_27b')/scripts/evaluate_local.py.
CLI: python -m resonance.stages.qwen3_8_27b.evaluate_local --stage {base,s1,s2,s3}
     [--gpu-pair 6,7] [--verify-first]   (RESONANCE_BENCH_BIN: vLLM environment's bin/)
"""
import argparse
import contextlib
import fcntl
import json
import os
import shutil
import subprocess
import time

from resonance.paths import run_dir
from resonance.stages.common import checkpoint, write_json
from resonance.stages.qwen3_8_27b.config import PROMPTS, SRC, STAGE_DIR, benchmark_bin, pipeline_root, traces_path

ROOT = pipeline_root()
BENCHMARK_BIN = benchmark_bin()


def benchmark_env():
    # An absolute Python path does not activate the environment for JIT subprocesses.
    env = {**os.environ, 'PATH': str(BENCHMARK_BIN) + os.pathsep + os.environ.get('PATH', ''),
           'FLASHINFER_WORKSPACE_BASE': str(ROOT/'outputs/build-cache/benchmark-v2'),
           'MAX_JOBS': '4',
           'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'TOKENIZERS_PARALLELISM': 'false',
           'PYTHONPATH': str(SRC)}
    for tool in ('ninja', 'cc', 'nvcc'):
        executable = shutil.which(tool, path=env['PATH'])
        if executable is None:
            raise RuntimeError(f'Benchmark build tool missing from PATH: {tool}')
        print(f'Benchmark build tool: {tool}={executable}', flush=True)
        subprocess.run([executable, '--version'], env=env, check=True)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['base','s1','s2','s3'], required=True)
    ap.add_argument('--gpu-pair', choices=['0,1', '2,3', '4,5', '6,7'],
                    help='Use a specified free NVLink pair with per-GPU locks.')
    ap.add_argument('--verify-first', action='store_true',
                    help='Require a 32-reaction inference check before full evaluation.')
    args = ap.parse_args()
    env = benchmark_env()
    from resonance.pipelines.traces.validate_no_id import validate
    identity = validate(traces_path())
    run_id = 'no-id-' + identity['source_sha256'][:16]
    root = ROOT/'outputs'/run_id/args.stage
    manifest = json.loads((root/'DOWNLOAD_COMPLETE.json').read_text())
    assert manifest['source_sha256'] == identity['source_sha256']
    model = checkpoint(root/'model')
    with contextlib.ExitStack() as locks:
        lock_paths = ([ROOT/f'outputs/gpu-{gpu}.lock' for gpu in args.gpu_pair.split(',')]
                      if args.gpu_pair else [ROOT/'outputs/local-evaluation.lock'])
        for path in lock_paths:
            lock = locks.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX)
        # Slurm on this machine does not enforce GPU allocation. Do not collide
        # with the all-GPU 8B SFT (including resubmissions) during model loading.
        while True:
            queue = subprocess.check_output(['squeue','-h','-u',os.environ['USER'],
                                             '-o','%i|%j|%T'], text=True).splitlines()
            busy = [r for r in queue if len(r.split('|')) == 3 and
                    r.split('|')[1].startswith(os.environ.get('RESONANCE_WAIT_JOB_PREFIX', 'q8b_s1_sft')) and
                    r.split('|')[2] in ('RUNNING','CONFIGURING','COMPLETING')]
            if busy:
                print(f'Waiting for local 8B SFT: {busy}', flush=True)
                time.sleep(60)
                continue
            rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                '--format=csv,noheader,nounits'], text=True).splitlines()
            free = [r.split(',')[0].strip() for r in rows if int(r.split(',')[1]) >= 85000]
            if args.gpu_pair:
                pair = args.gpu_pair.split(',')
                free = pair if set(pair) <= set(free) else []
            if len(free) >= 2:
                break
            print('Waiting for two local GPUs with >=85000 MiB free', flush=True)
            time.sleep(60)
        env['CUDA_VISIBLE_DEVICES'] = ','.join(free[:2])
        command = [str(BENCHMARK_BIN/'python'), str(STAGE_DIR/'generate.py'),
            '--stage','eval','--model',str(model),'--data',str(run_dir('uspto')/'test.json'),
            '--prompts',str(PROMPTS),'--tp','2']
        if args.verify_first:
            smoke = root/'eval-build-verification-v2'
            subprocess.run(command + ['--out',str(smoke),'--limit','32'], env=env, check=True)
            report = json.loads((smoke/'COMPLETE.json').read_text())
            assert report['rows'] == 32 and report['completions'] == 32
            print('BUILD AND INFERENCE VERIFIED: 32 reactions completed; starting full benchmark', flush=True)
        subprocess.run(command + ['--out',str(root/'eval')], env=env, check=True)
        results = {}
        for stage in ('base','s1','s2','s3'):
            report = ROOT/'outputs'/run_id/stage/'eval/COMPLETE.json'
            if report.exists():
                results[stage] = json.loads(report.read_text())
        identities = [r['identity'] for r in results.values()]
        for key in ('data_sha256','prompts_sha256','temperature','max_tokens','max_model_len',
                    'seed','top_p','top_k','thinking','n','limit'):
            assert len({r[key] for r in identities}) == 1, f'Evaluation mismatch: {key}'
        write_json(ROOT/'outputs'/run_id/'comparison.json', results)
        lines = ['| Stage | Exact accuracy | Valid SMILES | Truncated | Mean output tokens |',
                 '|---|---:|---:|---:|---:|']
        for stage, r in results.items():
            lines.append(f"| {stage.upper()} | {r['accuracy']:.2%} | {r['valid_smiles_fraction']:.2%} | "
                         f"{r['truncation_fraction']:.2%} | {r['mean_output_tokens']:.0f} |")
        (ROOT/'outputs'/run_id/'comparison.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
