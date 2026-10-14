"""Resume only the three authorized 27B rollout streams on H100s.

Ported from run_dir('pipeline_27b')/modal/resume_h100_rollouts.py.
Launch: python -m modal run --detach \
        src/resonance/stages/qwen3_8_27b/modal_apps/resume_h100_rollouts.py --target {s2_dpo,skip_s2}
"""
import json
from datetime import UTC
from pathlib import Path

import modal

from resonance.paths import run_dir
from resonance.stages.qwen3_8_27b.config import (
    ABLATION_VOLUME,
    APP_RESUME,
    CONTAINER_ENV,
    PIPELINE_VOLUME,
    RUN_ID,
    S1_VOLUME,
    pipeline_root,
    with_code,
)

ROOT = pipeline_root()
RUN = RUN_ID
MAIN_VOLUME = PIPELINE_VOLUME
app = modal.App(APP_RESUME)
main_volume = modal.Volume.from_name(MAIN_VOLUME, create_if_missing=False)
ablation_volume = modal.Volume.from_name(ABLATION_VOLUME, create_if_missing=False)
s1_volume = modal.Volume.from_name(S1_VOLUME, create_if_missing=False)
image = (modal.Image.from_registry('nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04', add_python='3.11')
    .entrypoint([]).apt_install('build-essential')
    .uv_pip_install('vllm==0.18.1', 'transformers==4.57.6', 'rdkit==2025.9.6')
    .env({'CUDA_HOME': '/usr/local/cuda', 'TOKENIZERS_PARALLELISM': 'false',
          'VLLM_WORKER_MULTIPROC_METHOD': 'spawn'})
    .env(CONTAINER_ENV))
image = with_code(image)


def specification(target):
    assert target in ('skip_s1', 'skip_s2', 's2_dpo')
    if target == 's2_dpo':
        return ('s3', MAIN_VOLUME, f'/pipeline/runs/{RUN}/s3-from-s2-v5-difficulty-mix-1ep',
                f'/pipeline/runs/{RUN}/s2-v5-difficulty-mix-1ep/model')
    stage = 's2' if target == 'skip_s1' else 's3'
    source = '/pipeline/base/model' if target == 'skip_s1' else f'/s1/runs/{RUN}/model'
    return stage, ABLATION_VOLUME, f'/pipeline/runs/{RUN}/{target}/{stage}', source


def execute(target, shard):
    import os
    import signal
    import subprocess
    import time
    from datetime import datetime

    from resonance.stages.common import checkpoint, sha256, write_json
    assert 0 <= shard < 4
    stage, volume_name, root_path, source = specification(target)
    volume = modal.Volume.from_name(volume_name)
    volume.reload()
    if target == 'skip_s2':
        s1_volume.reload()
    checkpoint(source)
    root = Path(root_path)
    identity = json.loads((root/'input.json').read_text())
    assert identity['train_rows'] == 40000 and identity['stage'] == stage
    assert identity['source'] == source
    assert sha256(root/'train.json') == identity['train_sha256']
    assert sha256(root/'prompts.json') == identity['prompts_sha256']
    out = root/f'rollouts/{shard}'
    if (out/'COMPLETE.json').exists():
        return json.loads((out/'COMPLETE.json').read_text())
    # Preserve failure records; generation validates identity and repairs only a partial final line.
    attempt = out/f'h100-resume-{time.time_ns()}.json'
    journal = {'status': 'running', 'target': target, 'shard': shard, 'gpu_requested': 'H100:2',
               'started_at': datetime.now(UTC).isoformat(), 'automatic_retries': 0}
    write_json(attempt, journal)
    volume.commit()
    started = time.monotonic()
    try:
        cmd = ['python', '/pipeline_code/generate.py', '--model', source,
            '--data', str(root/'train.json'), '--prompts', str(root/'prompts.json'),
            '--out', str(out), '--stage', stage, '--shard', str(shard), '--shards', '4', '--tp', '2']
        proc = subprocess.Popen(cmd, env={**os.environ, 'PIPELINE_VOLUME': volume_name}, start_new_session=True)
        try:
            rc = proc.wait(timeout=11*3600+30*60)
            assert rc == 0, f'Generation failed with exit code {rc}'
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        result = json.loads((out/'COMPLETE.json').read_text())
        assert result['rows'] == 10000
        journal['status'] = 'complete'
        return result
    except Exception as exc:
        journal.update(status='failed', error=str(exc))
        raise
    finally:
        journal.update(wall_seconds=time.monotonic()-started,
                       ended_at=datetime.now(UTC).isoformat())
        write_json(attempt, journal)
        volume.commit()


@app.function(image=image, volumes={'/pipeline': ablation_volume, '/s1': s1_volume},
              gpu='H100:2', cpu=16, memory=96*1024, timeout=12*3600, retries=0, max_containers=4)
def ablation_rollout(target: str, shard: int):
    assert target in ('skip_s1', 'skip_s2')
    return execute(target, shard)


@app.function(image=image, volumes={'/pipeline': main_volume},
              gpu='H100:2', cpu=16, memory=96*1024, timeout=12*3600, retries=0, max_containers=4)
def dpo_rollout(shard: int):
    return execute('s2_dpo', shard)


@app.local_entrypoint()
def main(target: str):
    import fcntl
    from datetime import datetime
    stage, _, _, _ = specification(target)
    local = (ROOT/'outputs'/RUN/'s3-from-s2-v5-difficulty-mix-1ep' if target == 's2_dpo'
             else run_dir('ablation_27b')/'outputs'/target)
    local.mkdir(parents=True, exist_ok=True)
    status_path = local/'h100_rollout_status.json'
    with (local/'h100_rollout.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {'status': 'running', 'target': target, 'stage': stage,
            'started_at': datetime.now(UTC).isoformat(), 'gpu': 'H100:2', 'workers': 4,
            'training_started': False}
        status_path.write_text(json.dumps(status, indent=2))
        try:
            results = list(dpo_rollout.map(range(4), return_exceptions=True) if target == 's2_dpo'
                           else ablation_rollout.starmap([(target, s) for s in range(4)], return_exceptions=True))
            failures = [str(r) for r in results if isinstance(r, Exception)]
            assert not failures, failures
            assert len(results) == 4 and sum(r['rows'] for r in results) == 40000
            (local/'h100_rollout_reports.json').write_text(json.dumps(results, indent=2))
            status.update(status='rollouts_complete', rows=40000,
                          completions=sum(r['completions'] for r in results))
        except Exception as exc:
            status.update(status='failed', error=str(exc))
            raise
        finally:
            status['updated_at'] = datetime.now(UTC).isoformat()
            status_path.write_text(json.dumps(status, indent=2))
            print(json.dumps(status), flush=True)
