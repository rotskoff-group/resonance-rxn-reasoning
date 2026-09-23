"""Validate prepared S2 data and reserve eight free GPUs before full SFT.

Ported from run_dir('s2_8b')/scripts/train.py.

    python -m resonance.stages.qwen3_8b.s2_train

This is the 8-GPU version that finished the published S2 (resume from checkpoint-500 to
step 2229). The run started on 4 GPUs (effective batch 32) and switched to 8 GPUs (effective
batch 64) at the resume; the 4-GPU version of this script was not preserved. The
`effective_batch: 32` written to TRAIN_COMPLETE.json below is the original script's constant
and is wrong for steps after 500 -- see README.md.
"""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from datasets import load_from_disk

from resonance.paths import run_dir
from resonance.stages.common import checkpoint, write_json
from resonance.stages.qwen3_8b._common import LOCK_DIR, PY, REPO, S1_MODEL, workflow_cmd

ROOT = run_dir('s2_8b')
BIN = Path(PY).parent


def main():
    manifest = json.loads((ROOT/'outputs/DATASET_COMPLETE.json').read_text())
    model = S1_MODEL
    assert manifest['model'] == str(model)
    checkpoint(model)
    ds = load_from_disk(ROOT/'outputs/sft_dataset')
    assert len(ds) == manifest['accepted'] > 0
    assert ds._fingerprint == manifest['dataset_fingerprint']
    assert max(ds['num_tokens']) <= manifest['max_length'] == 1024
    digest = hashlib.sha256()
    with Path(manifest['source']).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == manifest['source_sha256']
    output = ROOT/'outputs/model'
    resume = ROOT/'outputs/model/checkpoint-500'
    if output.exists() and any(output.iterdir()):
        checkpoint(resume)
        state = json.loads((resume/'trainer_state.json').read_text())
        assert state['global_step'] == 500
        print(f'Resuming training from verified {resume}', flush=True)
    with contextlib.ExitStack() as stack:
        # Keep existing dynamic-pair benchmark launchers from racing training startup.
        lock = stack.enter_context((LOCK_DIR/'local-evaluation.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Reserve all eight GPUs for the distributed SFT job.
        for gpu in range(8):
            lock = stack.enter_context((LOCK_DIR/f'gpu-{gpu}.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX)
        while True:
            rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                           '--format=csv,noheader,nounits'], text=True).splitlines()
            free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in rows}
            if all(free.get(gpu, 0) >= 78000 for gpu in range(8)):
                break
            print('Waiting for GPUs 0–7 with >=78000 MiB free each', flush=True)
            time.sleep(60)
        available = shutil.disk_usage(ROOT).free
        assert available >= 300*1024**3, f'Refusing training: only {available/1024**3:.1f} GiB free; require 300'
        env = {**os.environ, 'PATH': str(BIN)+os.pathsep+os.environ.get('PATH',''),
               'PYTHONPATH':str(REPO/'src'), 'PYTHONNOUSERSITE':'1',
               'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7', 'TOKENIZERS_PARALLELISM':'false'}
        print(f'Training on {len(ds)} examples; GPUs 0–7; free disk {available/1024**3:.1f} GiB', flush=True)
        subprocess.run([PY,'-m','accelerate.commands.launch','--multi_gpu',
                        '--num_processes','8','--mixed_precision','bf16','--main_process_port','29517',
                        *workflow_cmd('sft','s2_sft',f'hydra.run.dir={ROOT}/outputs/hydra')],
                       env=env, cwd=str(REPO), check=True)
        checkpoint(output)
        write_json(ROOT/'outputs/TRAIN_COMPLETE.json',
                   {'model':str(output),'init_model':str(model),'examples':len(ds),
                    'epochs':3,'effective_batch':32,'learning_rate':1e-5,
                    'source_sha256':manifest['source_sha256']})


if __name__ == '__main__':
    main()
