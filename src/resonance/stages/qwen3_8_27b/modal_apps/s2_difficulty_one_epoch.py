"""One epoch of difficulty-balanced RS-SFT from S1, with easy replay before DPO.

Ported from run_dir('pipeline_27b')/modal/s2_difficulty_one_epoch.py. Upload the V5
dataset first (volume_transfer.py upload-s2-v5).
Launch: python -m modal run --detach src/resonance/stages/qwen3_8_27b/modal_apps/s2_difficulty_one_epoch.py
"""
import json
import pathlib

import modal

from resonance.stages.qwen3_8_27b.config import (
    APP_S2,
    CONTAINER_ENV,
    PIPELINE_VOLUME,
    S1_VOLUME,
    pipeline_root,
    traces_path,
    with_code,
)

APP_NAME = APP_S2
ROOT = pipeline_root()
VOLUME = PIPELINE_VOLUME
N_NODES = 1
GPUS_PER_NODE = 8
WORLD_SIZE = N_NODES * GPUS_PER_NODE
VARIANT = 's2-v5-difficulty-mix-1ep'
app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME)
s1_volume = modal.Volume.from_name(S1_VOLUME)
mounts: dict[str | pathlib.PurePosixPath, modal.Volume | modal.CloudBucketMount] = (
    {'/pipeline': data_volume, '/s1': s1_volume})
train_image = (modal.Image.from_registry(
        'nvidia/cuda:12.6.0-devel-ubuntu22.04', add_python='3.11')
    .entrypoint([]).apt_install('build-essential', 'git')
    .uv_pip_install('torch==2.10.0', extra_index_url='https://download.pytorch.org/whl/cu126')
    .uv_pip_install('transformers==5.17.0', 'accelerate', 'datasets',
        'deepspeed==0.19.0', 'flash-linear-attention', 'ninja', 'packaging',
        'wheel', 'setuptools', 'psutil')
    .run_commands('MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=9.0 pip install causal-conv1d --no-build-isolation')
    .uv_pip_install('tilelang==0.1.14')
    .env({'HF_HOME': '/pipeline/hf', 'TOKENIZERS_PARALLELISM': 'false',
          'OMP_NUM_THREADS': '4', 'PYTORCH_ALLOC_CONF': 'expandable_segments:True'})
    .uv_pip_install('trl==1.3.0', 'peft==0.19.1', 'rdkit==2025.9.6')
    .env(CONTAINER_ENV))
train_image = with_code(train_image)


def paths(run_id):
    assert run_id.startswith('no-id-') and len(run_id) == 22
    run = pathlib.Path('/pipeline/runs') / run_id
    return run / VARIANT, run / 'S2_V5_difficulty_mix_dataset', pathlib.Path('/s1/runs') / run_id / 'model'


def subprocess_run(cmd, timeout, env=None):
    import os
    import signal
    import subprocess
    p = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        rc = p.wait(timeout=timeout)
        if rc:
            raise subprocess.CalledProcessError(rc, cmd)
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()


@app.function(image=train_image, volumes=mounts, cpu=2, memory=8192, timeout=900, retries=0)
def validate_data(run_id: str, manifest_sha256: str):
    import hashlib

    from datasets import load_from_disk
    data_volume.reload()
    _, data_root, _ = paths(run_id)
    raw = (data_root/'manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest_sha256
    manifest = json.loads(raw)
    assert manifest['unique_reactions'] == 40000
    for name, expected in manifest['dataset_files'].items():
        digest = hashlib.sha256()
        with (data_root/name).open('rb') as handle:
            for chunk in iter(lambda: handle.read(8*1024*1024), b''):
                digest.update(chunk)
        assert digest.hexdigest() == expected, name
    ds = load_from_disk(str(data_root/'dataset'))
    assert len(ds) == manifest['examples']
    assert 'labels' in ds.column_names
    return {'validated_examples':len(ds), 'manifest_sha256':manifest_sha256}


@app.function(image=train_image, volumes=mounts, gpu='H200:8', cpu=32,
              memory=256*1024, ephemeral_disk=2*1024*1024,
              timeout=24*3600, retries=0)
def train(run_id: str, segment: int = 0):
    import os
    import time

    from resonance.stages.common import checkpoint, write_json

    node_rank = 0
    root, data_root, source = paths(run_id)
    assert segment < 2
    data_volume.reload()
    s1_volume.reload()
    checkpoint(source)
    assert (data_root/'dataset').is_dir()
    manifest = json.loads((data_root/'manifest.json').read_text())
    assert manifest['unique_reactions'] == 40000
    assert manifest['name'] == 'S2_V5_difficulty_mix_dataset'
    root.mkdir(parents=True, exist_ok=True)
    journal = root/f'train_execution_{segment}.json'
    if (root/'TRAIN_COMPLETE.json').exists():
        return {'status': 'completed'}
    if node_rank == 0:
        assert not journal.exists(), 'This paid segment already ran; inspect before resubmitting'

    started = time.monotonic()
    status = 'failed'
    try:
        subprocess_run([
            'python', '-m', 'torch.distributed.run', '--standalone',
            f'--nproc_per_node={GPUS_PER_NODE}', '/pipeline_code/train_stage.py'],
            timeout=23*3600,
            env={**os.environ, 'STAGE_ROOT': str(root),
                 'STAGE_DATA_ROOT': str(data_root), 'STAGE_MODEL': str(source),
                 'STAGE': 's2', 'TRAIN_MODE': 'lora', 'S2_NUM_TRAIN_EPOCHS': '1',
                 'EXPECTED_WORLD_SIZE': str(WORLD_SIZE),
                 'GRADIENT_ACCUMULATION_STEPS': '4',
                 'PIPELINE_VOLUME': VOLUME, 'TRAIN_SOFT_SECONDS': str(20*3600)})
        assert (root/'TRAIN_COMPLETE.json').exists() or (root/'PAUSED.json').exists()
        status = 'completed' if (root/'TRAIN_COMPLETE.json').exists() else 'paused'
    finally:
        if node_rank == 0:
            write_json(journal, {'status': status, 'wall_seconds': time.monotonic()-started,
                'nodes': N_NODES, 'gpus_per_node': GPUS_PER_NODE, 'gpus': WORLD_SIZE,
                'gpu_type': 'H200', 'automatic_retries': 0})
            data_volume.commit()
    return {'status': status}


@app.function(image=train_image, volumes=mounts, cpu=16, memory=256*1024,
              timeout=3*3600, retries=0)
def merge(run_id: str):
    import subprocess

    from resonance.stages.common import checkpoint, write_json

    data_volume.reload()
    s1_volume.reload()
    root, data_root, source = paths(run_id)
    assert (root/'TRAIN_COMPLETE.json').is_file()
    if not (root/'MERGE_COMPLETE.json').exists():
        subprocess.run(['python', '/pipeline_code/merge.py', str(root), str(source)], check=True)
    checkpoint(root/'model')
    report = {
        'run_id': run_id, 'stage': 's2', 'variant': VARIANT,
        'source': str(source), 'data_root': str(data_root),
        'training': json.loads((root/'TRAIN_COMPLETE.json').read_text()),
        'training_config': json.loads((root/'training_config.json').read_text()),
        'model_path': str(root/'model')}
    write_json(root/'COMPLETE.json', report)
    data_volume.commit()
    return report


@app.local_entrypoint()
def main():
    from resonance.pipelines.traces.validate_no_id import validate

    source = traces_path()
    manifest = validate(source)
    run_id = 'no-id-' + manifest['source_sha256'][:16]
    target = ROOT/'outputs'/run_id/VARIANT
    target.mkdir(parents=True, exist_ok=True)
    import hashlib
    data_manifest = ROOT/'outputs'/run_id/'S2_V5_difficulty_mix_dataset/manifest.json'
    print(json.dumps(validate_data.remote(run_id, hashlib.sha256(data_manifest.read_bytes()).hexdigest())), flush=True)
    for segment in range(2):
        result = train.remote(run_id, segment)
        if result['status'] == 'completed':
            break
    else:
        raise RuntimeError('S2 LoRA exhausted two resumable 20-hour segments')
    report = merge.remote(run_id)
    (target/'COMPLETE.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
