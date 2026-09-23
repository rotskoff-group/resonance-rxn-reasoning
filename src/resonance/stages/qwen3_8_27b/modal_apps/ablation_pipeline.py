"""Concurrent eight-H200 stage-elimination arms with isolated artifacts.

Ported from run_dir('ablation_27b')/modal/pipeline.py. Fig 7 uses the skip_s2 arm
(rollouts resumed on H100, then `train_after_rollouts`, then a local benchmark).
Launch: python -m modal run --detach src/resonance/stages/qwen3_8_27b/modal_apps/ablation_pipeline.py
        --arm skip_s2 --s2-policy difficulty_mix [--resume-failed]   (or ::train_after_rollouts)
"""
import json
import pathlib
import modal

from resonance.paths import run_dir
from resonance.stages.qwen3_8_27b.config import (
    ABLATION_VOLUME, APP_ABLATION, APP_EVAL, APP_PIPELINE, APP_RESUME, APP_S1, APP_S2, APP_S3,
    CONTAINER_ENV, PIPELINE_VOLUME, PROMPTS, RUN_ID, S1_VOLUME, pipeline_root, traces_path, with_code)

ROOT = run_dir('ablation_27b')
from resonance.stages.qwen3_8_27b.config import BASE_MODEL as BASE_ID, BASE_REVISION, SOURCE_SHA
VOLUME = ABLATION_VOLUME
app = modal.App(APP_ABLATION)
data_volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
s1_volume = modal.Volume.from_name(S1_VOLUME)
mounts = {'/pipeline': data_volume, '/s1': s1_volume}

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


def paths(arm, stage):
    assert arm in ('skip_s1', 'skip_s2')
    assert stage in ('s2', 's3') and not (arm == 'skip_s2' and stage == 's2')
    root = pathlib.Path('/pipeline/runs') / RUN_ID / arm / stage
    if arm == 'skip_s2':
        source = pathlib.Path('/s1/runs') / RUN_ID / 'model'
    elif stage == 's3':
        source = root.parent/'s2/model'
    else:
        source = pathlib.Path('/pipeline/base/model')
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


@app.function(image=train_image, volumes=mounts, cpu=4, memory=16384,
              timeout=3600, retries=0)
def initialize(arm: str, stage: str, train_raw: bytes, test_raw: bytes,
               prompts_raw: bytes, s2_policy: str):
    import hashlib
    import sys
    from huggingface_hub import snapshot_download
    from resonance.stages.common import checkpoint, write_json
    data_volume.reload()
    s1_volume.reload()
    root, source = paths(arm, stage)
    assert s2_policy in ('difficulty_mix', 'all_correct')
    if arm == 'skip_s1' and stage == 's2':
        marker = source.parent/'COMPLETE.json'
        expected = {'model_id': BASE_ID, 'revision': BASE_REVISION}
        if marker.exists():
            assert json.loads(marker.read_text()) == expected
        else:
            # Read the existing S1 download cache, but copy into the isolated ablation volume.
            cache = pathlib.Path('/s1/hf/hub')
            cached = cache/('models--'+BASE_ID.replace('/', '--'))/'snapshots'/BASE_REVISION
            if cached.exists():
                import shutil
                checkpoint(cached)
                source.mkdir(parents=True, exist_ok=True)
                for p in cached.iterdir():
                    if p.is_file() and p.suffix in ('.json', '.safetensors', '.jinja', '.txt', '.model'):
                        shutil.copyfile(p, source/p.name)
            else:
                snapshot_download(BASE_ID, revision=BASE_REVISION, local_dir=source,
                    allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model'],
                    max_workers=4)
            checkpoint(source)
            write_json(marker, expected)
    elif arm == 'skip_s2':
        parent = json.loads((source.parent/'COMPLETE.json').read_text())
        assert parent['global_step'] == 3750 and parent['epochs'] == 3
        assert parent['data']['source_sha256'] == SOURCE_SHA
    else:
        parent = json.loads((source.parent/'COMPLETE.json').read_text())
        assert parent['arm'] == arm and parent['stage'] == 's2'
    checkpoint(source)
    assert len(json.loads(train_raw)) == 40000 and len(json.loads(test_raw)) == 5000
    root.mkdir(parents=True, exist_ok=True)
    identity = {'arm': arm, 'stage': stage, 'source': str(source), 'run_id': RUN_ID,
        'train_sha256': hashlib.sha256(train_raw).hexdigest(), 'train_rows': 40000,
        'test_sha256': hashlib.sha256(test_raw).hexdigest(),
        'prompts_sha256': hashlib.sha256(prompts_raw).hexdigest(),
        's2_policy': s2_policy, 'base_revision': BASE_REVISION, 'source_corpus_sha256': SOURCE_SHA,
        'gpus_per_arm': 8, 'rollout_shards': 4, 'rollout_tp': 2,
        's2_epochs': 1, 'train_mode': 'lora'}
    if (root/'input.json').exists():
        assert json.loads((root/'input.json').read_text()) == identity
    write_json(root/'input.json', identity)
    (root/'train.json').write_bytes(train_raw)
    (root/'test.json').write_bytes(test_raw)
    (root/'prompts.json').write_bytes(prompts_raw)
    # Validate the exact DPO implementation before starting any full-size paid GPUs.
    preflight = pathlib.Path('/pipeline')/f'preflight-{arm}.json'
    if not preflight.exists():
        subprocess_run(['python', '/pipeline_code/preflight.py'], timeout=600)
        write_json(preflight, {'passed': True, 'arm': arm})
    data_volume.commit()
    return identity
# Keep vLLM's supported Transformers 4.x environment separate from the training image.
infer_image = (modal.Image.from_registry('nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04', add_python='3.11')
    # Triton needs a host compiler; FlashInfer also JIT-compiles CUDA kernels.
    .entrypoint([]).apt_install('build-essential')
    .run_commands("printf 'int main(void) { return 0; }\\n' | cc -x c - -o /tmp/compiler-smoke && /tmp/compiler-smoke && rm /tmp/compiler-smoke")
    .run_commands("printf '#include <cuda_runtime.h>\\n__global__ void smoke(float *x) { x[threadIdx.x] = 0.0f; }\\n' > /tmp/compiler-smoke.cu && /usr/local/cuda/bin/nvcc -std=c++20 -arch=sm_90a -static-global-template-stub=false -c /tmp/compiler-smoke.cu -o /tmp/compiler-smoke.o && rm /tmp/compiler-smoke.cu /tmp/compiler-smoke.o")
    .uv_pip_install('vllm==0.18.1', 'transformers==4.57.6', 'rdkit==2025.9.6')
    .env({'CUDA_HOME': '/usr/local/cuda', 'TOKENIZERS_PARALLELISM': 'false',
          'VLLM_WORKER_MULTIPROC_METHOD': 'spawn'})
    .env(CONTAINER_ENV))
infer_image = with_code(infer_image)


@app.function(image=infer_image, volumes=mounts, gpu='H200:2', cpu=16,
              memory=96*1024, timeout=12*3600, retries=0, max_containers=4)
def rollout(arm: str, stage: str, shard: int, evaluation: bool = False,
            resume_failed: bool = False):
    import os
    import sys
    import time
    from resonance.stages.common import checkpoint, write_json
    data_volume.reload()
    s1_volume.reload()
    root, source = paths(arm, stage)
    assert 0 <= shard < 4
    if evaluation:
        assert (root/'COMPLETE.json').exists()
        source = root/'model'
    checkpoint(source)
    out = root / ('eval' if evaluation else 'rollouts') / str(shard)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        return json.loads((out/'COMPLETE.json').read_text())
    journal = out/'execution.json'
    if journal.exists():
        assert resume_failed, 'Inspect the previous paid attempt before retrying'
        previous = json.loads(journal.read_text())
        assert previous['status'] == 'failed', previous
        # Explicit user-requested retry after the workspace billing interruption.
        # generate.py validates identity and resumes only missing reaction rows.
        journal.rename(out/f'execution.before-resume-{time.time_ns()}.json')
        data_volume.commit()
    started, status = time.monotonic(), 'failed'
    try:
        subprocess_run(['python', '/pipeline_code/generate.py', '--model', str(source),
            '--data', str(root/('test.json' if evaluation else 'train.json')),
            '--prompts', str(root/'prompts.json'), '--out', str(out),
            '--stage', 'eval' if evaluation else stage, '--shard', str(shard),
            '--shards', '4', '--tp', '2'], timeout=11*3600+45*60,
            env={**os.environ, 'PIPELINE_VOLUME': VOLUME})
        assert (out/'COMPLETE.json').exists()
        status = 'completed'
    finally:
        write_json(out/'execution.json', {'status': status,
            'wall_seconds': time.monotonic()-started, 'gpus': 2, 'automatic_retries': 0})
        data_volume.commit()
    return json.loads((out/'COMPLETE.json').read_text())


@app.function(image=train_image, volumes=mounts, cpu=8, memory=64*1024,
              timeout=3600, retries=0)
def prepare(arm: str, stage: str):
    data_volume.reload()
    s1_volume.reload()
    root, source = paths(arm, stage)
    for shard in range(4):
        done = json.loads((root/f'rollouts/{shard}/COMPLETE.json').read_text())
        assert done['rows'] == 10000 and done['identity']['stage'] == stage
    if not (root/'data_stats.json').exists():
        subprocess_run(['python', '/pipeline_code/ablation/prepare.py', '--root', str(root),
                        '--model', str(source), '--stage', stage], timeout=3300)
        data_volume.commit()
    return json.loads((root/'data_stats.json').read_text())


@app.function(image=train_image, volumes=mounts, gpu='H200:8', cpu=32,
              memory=256*1024, ephemeral_disk=2*1024*1024,
              timeout=24*3600, retries=0, max_containers=1)
def train(arm: str, stage: str, segment: int = 0):
    import os
    import sys
    import time
    from resonance.stages.common import write_json
    data_volume.reload()
    s1_volume.reload()
    root, source = paths(arm, stage)
    assert 0 <= segment < (2 if stage == 's2' else 1)
    if (root/'TRAIN_COMPLETE.json').exists():
        return {'status': 'completed'}
    journal = root/f'train_execution_{segment}.json'
    assert not journal.exists(), 'Inspect the previous paid attempt before retrying'
    started, status = time.monotonic(), 'failed'
    try:
        subprocess_run(['python', '-m', 'torch.distributed.run', '--standalone',
            '--nproc_per_node=8', '/pipeline_code/train_stage.py'], timeout=23*3600,
            env={**os.environ, 'STAGE_ROOT': str(root), 'STAGE_MODEL': str(source),
                 'STAGE': stage, 'TRAIN_MODE': 'lora', 'S2_NUM_TRAIN_EPOCHS': '1',
                 **({'S3_NUM_TRAIN_EPOCHS': '1'} if arm == 'skip_s2' and stage == 's3' else {}),
                 'EXPECTED_WORLD_SIZE': '8', 'GRADIENT_ACCUMULATION_STEPS': '4',
                 'PIPELINE_VOLUME': VOLUME, 'TRAIN_SOFT_SECONDS': str(20*3600)})
        assert (root/'TRAIN_COMPLETE.json').exists() or (root/'PAUSED.json').exists()
        status = 'completed' if (root/'TRAIN_COMPLETE.json').exists() else 'paused'
    finally:
        write_json(journal, {'status': status, 'gpus': 8,
                            'wall_seconds': time.monotonic()-started, 'automatic_retries': 0})
        data_volume.commit()
    return {'status': status}


@app.function(image=train_image, volumes=mounts, cpu=16, memory=256*1024,
              timeout=3*3600, retries=0)
def finish(arm: str, stage: str):
    import sys
    from resonance.stages.common import checkpoint, sha256, write_json
    data_volume.reload()
    s1_volume.reload()
    root, source = paths(arm, stage)
    assert (root/'TRAIN_COMPLETE.json').exists()
    if not (root/'MERGE_COMPLETE.json').exists():
        subprocess_run(['python', '/pipeline_code/merge.py', str(root), str(source)], timeout=2*3600)
    checkpoint(root/'model')
    report = {**json.loads((root/'input.json').read_text()),
              'training': json.loads((root/'TRAIN_COMPLETE.json').read_text()),
              'training_config': json.loads((root/'training_config.json').read_text()),
              'data': json.loads((root/'data_stats.json').read_text()),
              'model_path': str(root/'model'),
              'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                        for p in sorted((root/'model').iterdir()) if p.is_file()}}
    write_json(root/'COMPLETE.json', report)
    data_volume.commit()
    return report


@app.function(image=train_image, volumes=mounts, cpu=2, memory=8192,
              timeout=1200, retries=0)
def summarize_eval(arm: str, stage: str):
    import sys
    from resonance.stages.common import iter_jsonl, write_json
    data_volume.reload()
    root, _ = paths(arm, stage)
    seen, correct, valid, truncated, tokens = set(), 0, 0, 0, 0
    for shard in range(4):
        out = root/f'eval/{shard}'
        identity = json.loads((out/'COMPLETE.json').read_text())['identity']
        assert identity['stage'] == 'eval' and identity['shards'] == 4 and identity['shard'] == shard
        assert identity['model'] == str(root/'model')
        assert identity['data_sha256'] == json.loads((root/'input.json').read_text())['test_sha256']
        for row in iter_jsonl(out/'rollouts.jsonl'):
            assert row['row_idx'] not in seen and row['row_idx'] % 4 == shard
            seen.add(row['row_idx'])
            assert len(row['completions']) == 1
            c = row['completions'][0]
            correct += c['correct']
            valid += c['prediction'] is not None
            truncated += c['finish_reason'] == 'length'
            tokens += len(c['token_ids'])
    assert seen == set(range(5000))
    report = {'arm': arm, 'stage': stage, 'rows': 5000, 'correct': correct,
              'accuracy': correct/5000, 'valid_smiles_fraction': valid/5000,
              'truncation_fraction': truncated/5000, 'mean_output_tokens': tokens/5000}
    write_json(root/'eval/COMPLETE.json', report)
    data_volume.commit()
    return report


@app.local_entrypoint()
def main(arm: str = 'skip_s1', s2_policy: str = 'difficulty_mix', initialize_only: bool = False,
         resume_failed: bool = False):
    assert arm in ('skip_s1', 'skip_s2')
    target = ROOT/'outputs'/arm
    target.mkdir(parents=True, exist_ok=True)
    train_raw = (run_dir('uspto')/'train.json').read_bytes()
    test_raw = (run_dir('uspto')/'test.json').read_bytes()
    prompts_raw = PROMPTS.read_bytes()
    stages = ['s2', 's3'] if arm == 'skip_s1' else ['s3']
    try:
        for stage in stages:
            identity = initialize.remote(arm, stage, train_raw, test_raw, prompts_raw, s2_policy)
            (target/f'{stage}_input.json').write_text(json.dumps(identity, indent=2)+'\n')
            if initialize_only:
                print(json.dumps(identity), flush=True)
                return
            (target/'status.json').write_text(json.dumps({'status': 'rollouts', 'stage': stage}))
            reports = list(rollout.starmap([(arm, stage, s, False, resume_failed) for s in range(4)],
                                         return_exceptions=True))
            failures = [str(r) for r in reports if isinstance(r, Exception)]
            assert not failures, failures
            assert len(reports) == 4
            stats = prepare.remote(arm, stage)
            (target/f'{stage}_data_stats.json').write_text(json.dumps(stats, indent=2)+'\n')
            (target/'status.json').write_text(json.dumps({'status': 'training', 'stage': stage}))
            for segment in range(2 if stage == 's2' else 1):
                result = train.remote(arm, stage, segment)
                if result['status'] == 'completed':
                    break
            else:
                raise RuntimeError('Training reached its bounded allowance; inspect resumable checkpoint')
            report = finish.remote(arm, stage)
            (target/f'{stage}_COMPLETE.json').write_text(json.dumps(report, indent=2)+'\n')
        (target/'status.json').write_text(json.dumps({'status': 'evaluation', 'stage': 's3'}))
        reports = list(rollout.starmap([(arm, 's3', s, True) for s in range(4)]))
        assert len(reports) == 4
        report = summarize_eval.remote(arm, 's3')
        (target/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
        (target/'status.json').write_text(json.dumps({'status': 'complete'}))
        print(json.dumps(report), flush=True)
    except Exception as exc:
        (target/'status.json').write_text(json.dumps({'status': 'failed', 'error': str(exc)}))
        raise


@app.local_entrypoint()
def train_after_rollouts():
    """Mine completed rollouts, train one DPO epoch, and merge the adapter."""
    target = ROOT/'outputs'/'skip_s2'
    target.mkdir(parents=True, exist_ok=True)
    def status(value, **extra):
        (target/'one_epoch_training_status.json').write_text(
            json.dumps({'status': value, 'num_train_epochs': 1, **extra}, indent=2)+'\n')
    try:
        status('preparing_pairs')
        stats = prepare.remote('skip_s2', 's3')
        assert stats['examples'] > 0
        (target/'s3_one_epoch_data_stats.json').write_text(json.dumps(stats, indent=2)+'\n')
        status('training', pairs=stats['examples'])
        result = train.remote('skip_s2', 's3', 0)
        assert result['status'] == 'completed', 'Training paused; inspect checkpoint before resuming'
        status('merging')
        report = finish.remote('skip_s2', 's3')
        (target/'s3_one_epoch_COMPLETE.json').write_text(json.dumps(report, indent=2)+'\n')
        status('complete')
        print(json.dumps(report), flush=True)
    except Exception as exc:
        status('failed', error=str(exc))
        raise
