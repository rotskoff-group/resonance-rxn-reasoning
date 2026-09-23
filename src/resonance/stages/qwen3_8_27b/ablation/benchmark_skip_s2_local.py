"""Download the completed skip-S2 model and run the matched local test benchmark.

Ported from run_dir('ablation_27b')/scripts/benchmark_skip_s2_local.py.
CLI: python -m resonance.stages.qwen3_8_27b.ablation.benchmark_skip_s2_local
"""
import contextlib
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import time
from collections import defaultdict

from resonance.stages.common import checkpoint, iter_jsonl, sha256, write_json

from resonance.paths import run_dir
from resonance.stages.qwen3_8_27b import evaluate_local, local_run
from resonance.stages.qwen3_8_27b.config import ABLATION_VOLUME, PROMPTS, RUN_ID, SRC, STAGE_DIR

ROOT = run_dir('ablation_27b')
REFERENCE = run_dir('pipeline_27b')
OUT = ROOT/'outputs/skip_s2/local_eval'
MODEL = ROOT/'outputs/skip_s2/model-local'
VOLUME = ABLATION_VOLUME
REMOTE = f'/runs/{RUN_ID}/skip_s2/s3'


def status(value, **extra):
    write_json(OUT/'status.json', {'status': value, 'job': os.environ.get('SLURM_JOB_ID'), **extra})


def download():
    import modal
    multiprocessing.cpu_count = lambda: 2
    manifest = json.loads((ROOT/'outputs/skip_s2/s3_one_epoch_COMPLETE.json').read_text())
    assert manifest['arm'] == 'skip_s2' and manifest['stage'] == 's3'
    assert manifest['training']['global_step'] == 429
    assert manifest['training']['epochs'] == 1.0
    volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
    remote_manifest = json.loads(b''.join(volume.read_file(REMOTE+'/COMPLETE.json')))
    assert remote_manifest['files'] == manifest['files'], 'Remote model identity changed'
    assert remote_manifest['training'] == manifest['training']
    MODEL.mkdir(parents=True, exist_ok=True)
    needed = sum(info['bytes'] for name, info in manifest['files'].items()
                 if not (MODEL/name).exists())
    assert shutil.disk_usage(MODEL).free > needed + 10*1024**3, 'Insufficient disk space'
    for name, info in manifest['files'].items():
        assert Path(name).name == name
        dest = MODEL/name
        if not dest.exists():
            print(f'Downloading {name}: {info["bytes"]:,} bytes', flush=True)
            temporary = dest.with_suffix(dest.suffix+'.partial')
            with temporary.open('wb') as stream:
                for chunk in volume.read_file(REMOTE+'/model/'+name):
                    stream.write(chunk)
            assert temporary.stat().st_size == info['bytes'], name
            assert sha256(temporary) == info['sha256'], name
            temporary.replace(dest)
        else:
            assert dest.stat().st_size == info['bytes'] and sha256(dest) == info['sha256'], name
    checkpoint(MODEL)
    write_json(MODEL/'DOWNLOAD_COMPLETE.json', {
        'volume': VOLUME, 'remote': REMOTE+'/model', 'verification': 'all file SHA256 hashes',
        'training': manifest['training'], 'files': manifest['files']})
    return manifest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'run.lock').open('a') as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT/'report.json').exists():
            print((OUT/'report.json').read_text())
            return
        status('downloading')
        manifest = download()
        data = run_dir('uspto')/'test.json'
        prompts = PROMPTS
        assert sha256(data) == manifest['test_sha256']
        assert sha256(prompts) == manifest['prompts_sha256']
        # Reuse the local runner's process-group cleanup and progress watchdog.
        helpers = {'benchmark_env': evaluate_local.benchmark_env,
                   'BENCHMARK_BIN': evaluate_local.BENCHMARK_BIN,
                   'run_generation': local_run.run_generation}
        env = helpers['benchmark_env']()
        env['PYTHONPATH'] = str(SRC)
        env['PYTHONUNBUFFERED'] = '1'
        env.pop('PIPELINE_VOLUME', None)
        status('waiting_for_gpus')
        def free_memory():
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                '--format=csv,noheader,nounits'], text=True).splitlines()
            return {int(r.split(',')[0]): int(r.split(',')[1]) for r in rows}
        while True:
            free = free_memory()
            selected = None
            for pair in ((0, 1), (2, 3), (4, 5), (6, 7)):
                if not all(free.get(g, 0) >= 85000 for g in pair):
                    continue
                locks = contextlib.ExitStack()
                try:
                    for g in pair:
                        lock = locks.enter_context((REFERENCE/f'outputs/gpu-{g}.lock').open('a'))
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    locks.close()
                    continue
                current = free_memory()
                if not all(current.get(g, 0) >= 85000 for g in pair):
                    locks.close()
                    continue
                selected = pair
                break
            if selected is not None:
                break
            print('Waiting for a free, unlocked local H100 pair', flush=True)
            time.sleep(30)
        with locks:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, selected))
            status('benchmarking', devices=selected, model=str(MODEL), rows=5000)
            helpers['run_generation']([
                str(helpers['BENCHMARK_BIN']/'python'), str(STAGE_DIR/'generate.py'),
                '--model', str(MODEL), '--data', str(data), '--prompts', str(prompts),
                '--out', str(OUT/'generation'), '--stage', 'eval', '--tp', '2'], env, OUT)
        summary = json.loads((OUT/'generation/COMPLETE.json').read_text())
        assert summary['rows'] == 5000
        labels = json.loads((run_dir('uspto_classes')/'outputs/uspto_labels_test.json').read_text())['labels']
        classes = defaultdict(lambda: {'rows': 0, 'correct': 0, 'valid': 0, 'truncated': 0, 'tokens': 0})
        seen = set()
        for row in iter_jsonl(OUT/'generation/rollouts.jsonl'):
            idx = row['row_idx']
            assert idx not in seen and len(row['completions']) == 1
            seen.add(idx)
            c = row['completions'][0]
            group = classes[labels[str(idx)]['level1']]
            group['rows'] += 1
            group['correct'] += int(c['correct'])
            group['valid'] += int(c['prediction'] is not None)
            group['truncated'] += int(c['finish_reason'] == 'length')
            group['tokens'] += len(c['token_ids'])
        assert seen == set(range(5000))
        for group in classes.values():
            group.update(accuracy=group['correct']/group['rows'],
                         valid_fraction=group['valid']/group['rows'],
                         mean_output_tokens=group['tokens']/group['rows'])
        report = {**summary, 'arm': 'skip_s2', 'training': manifest['training'], 'classes': dict(classes)}
        write_json(OUT/'report.json', report)
        write_json(ROOT/'outputs/skip_s2/evaluation.json', report)
        status('complete', accuracy=report['accuracy'])
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=str(exc))
        raise
