"""S3/DPO from the completed one-epoch difficulty-mix S2 checkpoint.

Ported from run_dir('pipeline_27b')/modal/s3_from_s2_v5.py. The paper's S3 rollouts
started here on H200 and finished via resume_h100_rollouts.py --target s2_dpo; pairs,
training and merge ran through `train_after_rollouts`.
Launch: python -m modal run --detach \
        src/resonance/stages/qwen3_8_27b/modal_apps/s3_from_s2_v5.py[::train_after_rollouts]
"""
import json
import pathlib

import modal

from resonance.paths import run_dir
from resonance.stages.qwen3_8_27b.config import (
    APP_S3,
    CONTAINER_ENV,
    PIPELINE_VOLUME,
    PROMPTS,
    RUN_ID,
    S1_VOLUME,
    pipeline_root,
    traces_path,
    with_code,
)

ROOT = pipeline_root()
S1_ROOT = run_dir('s1_27b')
VOLUME = PIPELINE_VOLUME
VARIANT = 's3-from-s2-v5-difficulty-mix-1ep'
PARENT_VARIANT = 's2-v5-difficulty-mix-1ep'
ROLLOUT_SHARDS = 4  # Four TP=2 workers use eight H200s in total.
app = modal.App(APP_S3)
data_volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
s1_volume = modal.Volume.from_name(S1_VOLUME, create_if_missing=True)
mounts: dict[str | pathlib.PurePosixPath, modal.Volume | modal.CloudBucketMount] = (
    {'/pipeline': data_volume, '/s1': s1_volume})

# Same package versions as S1; add dependencies before Modal's live source mounts.
train_image = (modal.Image.from_registry('nvidia/cuda:12.6.0-devel-ubuntu22.04', add_python='3.11')
    .entrypoint([]).apt_install('build-essential', 'git')
    .uv_pip_install('torch==2.10.0', extra_index_url='https://download.pytorch.org/whl/cu126')
    .uv_pip_install('transformers==5.17.0', 'accelerate', 'datasets', 'deepspeed==0.19.0',
        'flash-linear-attention', 'ninja', 'packaging', 'wheel', 'setuptools', 'psutil')
    .run_commands('MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=9.0 pip install causal-conv1d --no-build-isolation')
    .uv_pip_install('tilelang==0.1.14')
    .env({'HF_HOME': '/pipeline/hf', 'TOKENIZERS_PARALLELISM': 'false',
          'OMP_NUM_THREADS': '4', 'PYTORCH_ALLOC_CONF': 'expandable_segments:True'})
    .uv_pip_install('trl==1.3.0', 'peft==0.19.1', 'rdkit==2025.9.6')
    .env(CONTAINER_ENV))
train_image = with_code(train_image)
# Keep vLLM's supported Transformers 4.x environment separate from the training image.
infer_image = (modal.Image.from_registry('nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04', add_python='3.11')
    # Triton needs a host compiler; FlashInfer also JIT-compiles CUDA kernels.
    .entrypoint([]).apt_install('build-essential')
    .run_commands("printf 'int main(void) { return 0; }\\n' | cc -x c - -o /tmp/compiler-smoke"
                  " && /tmp/compiler-smoke && rm /tmp/compiler-smoke")
    .run_commands("printf '#include <cuda_runtime.h>\\n__global__ void smoke(float *x) { x[threadIdx.x] = 0.0f; }\\n'"
                  " > /tmp/compiler-smoke.cu && /usr/local/cuda/bin/nvcc -std=c++20 -arch=sm_90a"
                  " -static-global-template-stub=false -c /tmp/compiler-smoke.cu -o /tmp/compiler-smoke.o"
                  " && rm /tmp/compiler-smoke.cu /tmp/compiler-smoke.o")
    .uv_pip_install('vllm==0.18.1', 'transformers==4.57.6', 'rdkit==2025.9.6')
    .env({'CUDA_HOME': '/usr/local/cuda', 'TOKENIZERS_PARALLELISM': 'false',
          'VLLM_WORKER_MULTIPROC_METHOD': 'spawn'})
    .env(CONTAINER_ENV))
infer_image = with_code(infer_image)


def paths(run_id, stage):
    assert run_id.startswith('no-id-') and len(run_id) == 22 and all(c in '0123456789abcdef' for c in run_id[6:])
    assert stage == 's3'
    root = pathlib.Path('/pipeline/runs') / run_id / VARIANT
    source = (pathlib.Path('/s1/runs') / run_id / 'model' if stage == 's2'
              else pathlib.Path('/pipeline/runs') / run_id / PARENT_VARIANT / 'model')
    return root, source


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


@app.function(image=train_image, volumes=mounts, cpu=4, memory=16384, timeout=1800, retries=0)
def initialize(run_id: str, stage: str, train_raw: bytes, prompts_raw: bytes, source_sha: str):
    import hashlib

    from resonance.stages.common import checkpoint, write_json
    s1_volume.reload()
    data_volume.reload()
    root, source = paths(run_id, stage)
    parent = json.loads((source.parent/'COMPLETE.json').read_text())
    if stage == 's2':
        assert parent['data']['source_sha256'] == source_sha
    else:
        assert parent['run_id'] == run_id and parent['variant'] == PARENT_VARIANT
        assert parent['training']['global_step'] == 3086
        assert parent['training']['epochs'] == 1.0
        assert parent['model_path'] == str(source)
        s1_parent = json.loads((pathlib.Path('/s1/runs')/run_id/'COMPLETE.json').read_text())
        assert s1_parent['data']['source_sha256'] == source_sha
    checkpoint(source)
    root.mkdir(parents=True, exist_ok=True)
    train_data = json.loads(train_raw)
    identity = {'run_id': run_id, 'stage': stage, 'source': str(source),
        'source_sha256': source_sha, 'parent_variant': PARENT_VARIANT,
        'parent_complete_sha256': hashlib.sha256((source.parent/'COMPLETE.json').read_bytes()).hexdigest(),
        'train_sha256': hashlib.sha256(train_raw).hexdigest(),
        'train_rows': len(train_data), 'rollout_shards': ROLLOUT_SHARDS,
        'prompts_sha256': hashlib.sha256(prompts_raw).hexdigest()}
    if (root/'input.json').exists():
        assert json.loads((root/'input.json').read_text()) == identity
    write_json(root/'input.json', identity)
    (root/'train.json').write_bytes(train_raw)
    (root/'prompts.json').write_bytes(prompts_raw)
    data_volume.commit()
    return identity


@app.function(image=infer_image, volumes=mounts, gpu='H200:2', cpu=16,
              memory=96*1024, timeout=12*3600, retries=0, max_containers=ROLLOUT_SHARDS)
def rollout(run_id: str, stage: str, shard: int):
    import os
    import time

    from resonance.stages.common import write_json
    s1_volume.reload()
    data_volume.reload()
    root, source = paths(run_id, stage)
    out = root / f'rollouts/{shard}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        return json.loads((out/'COMPLETE.json').read_text())
    # A failed invocation requires deliberate resubmission; no repeated paid retry loop.
    started = time.monotonic()
    status = 'failed'
    try:
        subprocess_run(['python', '/pipeline_code/generate.py', '--model', str(source),
            '--data', str(root/'train.json'), '--prompts', str(root/'prompts.json'),
            '--out', str(out), '--stage', stage, '--shard', str(shard), '--shards', str(ROLLOUT_SHARDS), '--tp', '2'],
            timeout=11*3600+45*60, env={**os.environ, 'PIPELINE_VOLUME': VOLUME})
        assert (out/'COMPLETE.json').exists()
        status = 'completed'
    finally:
        write_json(out/'execution.json', {'status': status, 'wall_seconds': time.monotonic()-started,
            'gpus': 2, 'automatic_retries': 0})
        data_volume.commit()
    return json.loads((out/'COMPLETE.json').read_text())


@app.function(image=train_image, volumes=mounts, cpu=8, memory=64*1024, timeout=3600, retries=0)
def prepare(run_id: str, stage: str):
    data_volume.reload()
    s1_volume.reload()
    root, source = paths(run_id, stage)
    subprocess_run(['python', '/pipeline_code/prepare_stage.py', '--root', str(root),
                    '--model', str(source), '--stage', stage,
                    '--shards', str(ROLLOUT_SHARDS)], timeout=3300)
    data_volume.commit()
    return json.loads((root/'data_stats.json').read_text())


@app.function(image=train_image, volumes=mounts, gpu='H200:8', cpu=32,
              memory=256*1024, ephemeral_disk=2*1024*1024,
              timeout=24*3600, retries=0, max_containers=1)
def train(run_id: str, stage: str, segment: int = 0):
    import os
    import time

    from resonance.stages.common import write_json
    data_volume.reload()
    s1_volume.reload()
    root, source = paths(run_id, stage)
    assert segment < (2 if stage == 's2' else 1)
    stats = json.loads((root/'data_stats.json').read_text())
    assert stats['examples'] > 0
    print(f"DPO_PAIRS_READY: {stats['examples']} pairs from {stats['source_rows']} reactions; "
          "starting 8-H200 training", flush=True)
    journal = root/f'train_execution_{segment}.json'
    if (root/'TRAIN_COMPLETE.json').exists():
        return {'status': 'completed'}
    assert not journal.exists(), 'This paid segment already ran; inspect it before any resubmission'
    started = time.monotonic()
    status = 'failed'
    try:
        subprocess_run(['python', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
            '/pipeline_code/train_stage.py'], timeout=23*3600,
            env={**os.environ, 'STAGE_ROOT': str(root), 'STAGE_MODEL': str(source),
                 'STAGE': stage, 'TRAIN_MODE': 'lora', 'S3_NUM_TRAIN_EPOCHS': '1', 'PIPELINE_VOLUME': VOLUME,
                 'TRAIN_SOFT_SECONDS': str(20*3600)})
        assert (root/'TRAIN_COMPLETE.json').exists() or (root/'PAUSED.json').exists()
        status = 'completed' if (root/'TRAIN_COMPLETE.json').exists() else 'paused'
    finally:
        write_json(journal, {'status': status,
            'wall_seconds': time.monotonic()-started, 'gpus': 8, 'automatic_retries': 0})
        data_volume.commit()
    return {'status': status}


@app.function(image=train_image, volumes=mounts, cpu=16, memory=256*1024,
              timeout=3*3600, retries=0)
def finish(run_id: str, stage: str):
    from resonance.stages.common import checkpoint, sha256, write_json
    data_volume.reload()
    s1_volume.reload()
    root, source = paths(run_id, stage)
    assert (root/'TRAIN_COMPLETE.json').exists()
    if stage == 's3' and not (root/'MERGE_COMPLETE.json').exists():
        subprocess_run(['python', '/pipeline_code/merge.py', str(root), str(source)], timeout=2*3600)
    checkpoint(root/'model')
    report = {**json.loads((root/'input.json').read_text()),
              'training': json.loads((root/'TRAIN_COMPLETE.json').read_text()),
              'data': json.loads((root/'data_stats.json').read_text()),
              'model_path': str(root/'model'), 'files': {}}
    for p in sorted((root/'model').iterdir()):
        if p.is_file():
            report['files'][p.name] = {'bytes': p.stat().st_size, 'sha256': sha256(p)}
    write_json(root/'COMPLETE.json', report)
    data_volume.commit()
    return report


@app.function(image=train_image, volumes=mounts, cpu=4, memory=16384, timeout=1800, retries=0)
def preflight():
    import subprocess
    subprocess.run(['python', '/pipeline_code/preflight.py'], check=True)
    return 'training imports and DPO preprocessing passed'


@app.function(image=infer_image, cpu=4, memory=16384, timeout=1200, retries=0)
def inference_preflight():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams  # noqa: F401  (preflight: the image must provide them)
    from vllm.transformers_utils.config import get_config
    model = 'Qwen/Qwen3.8-27B'
    revision = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
    config = get_config(model, trust_remote_code=False, revision=revision)
    assert config.model_type == 'qwen3_5'
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    prompt = tok.apply_chat_template([{'role':'user','content':'Predict the product.'}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True, preserve_thinking=True)
    assert '<think>' in prompt
    assert len(tok.encode(prompt, add_special_tokens=False)) > 0
    return {'model_type': config.model_type, 'architectures': config.architectures,
            'tokenizer': type(tok).__name__, 'prompt_suffix': prompt[-80:]}


@app.local_entrypoint()
def main(stage: str = 's3', check_only: bool = False):
    assert stage == 's3'
    if check_only:
        print(preflight.remote())
        print(inference_preflight.remote())
        return
    from resonance.pipelines.traces.validate_no_id import validate
    source = traces_path()
    manifest = validate(source)
    run_id = 'no-id-' + manifest['source_sha256'][:16]
    target = ROOT/'outputs'/run_id/VARIANT
    target.mkdir(parents=True, exist_ok=True)
    print(preflight.remote(), flush=True)
    identity = initialize.remote(run_id, stage, (run_dir('uspto')/'train.json').read_bytes(),
                                 PROMPTS.read_bytes(), manifest['source_sha256'])
    (target/'input.json').write_text(json.dumps(identity, indent=2))
    results = list(rollout.starmap([(run_id, stage, s) for s in range(ROLLOUT_SHARDS)], return_exceptions=True))
    failures = [str(r) for r in results if isinstance(r, Exception)]
    assert not failures, failures
    assert len(results) == ROLLOUT_SHARDS
    stats = prepare.remote(run_id, stage)
    (target/'data_stats.json').write_text(json.dumps(stats, indent=2))
    assert stats['examples'] > 0
    print(f"DPO_PAIRS_READY: {stats['examples']} pairs from {stats['source_rows']} reactions. "
          "Mining complete; training has not started yet.", flush=True)
    print(json.dumps(stats), flush=True)
    for segment in range(2 if stage == 's2' else 1):
        trained = train.remote(run_id, stage, segment)
        if trained['status'] == 'completed':
            break
    else:
        raise RuntimeError(f'{stage} reached its compute allowance with a resumable checkpoint; '
                           'downstream stages are stopped')
    result = finish.remote(run_id, stage)
    (target/'COMPLETE.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


@app.local_entrypoint()
def train_after_rollouts():
    """Mine completed rollouts, train one DPO epoch, and merge the adapter."""
    target = ROOT/'outputs'/RUN_ID/VARIANT
    target.mkdir(parents=True, exist_ok=True)
    def status(value, **extra):
        (target/'one_epoch_training_status.json').write_text(
            json.dumps({'status': value, 'num_train_epochs': 1, **extra}, indent=2)+'\n')
    try:
        status('preparing_pairs')
        stats = prepare.remote(RUN_ID, 's3')
        assert stats['examples'] > 0
        (target/'s3_one_epoch_data_stats.json').write_text(json.dumps(stats, indent=2)+'\n')
        status('training', pairs=stats['examples'])
        result = train.remote(RUN_ID, 's3', 0)
        assert result['status'] == 'completed', 'Training paused; inspect checkpoint before resuming'
        status('merging')
        report = finish.remote(RUN_ID, 's3')
        (target/'s3_one_epoch_COMPLETE.json').write_text(json.dumps(report, indent=2)+'\n')
        status('complete')
        print(json.dumps(report), flush=True)
    except Exception as exc:
        status('failed', error=str(exc))
        raise
