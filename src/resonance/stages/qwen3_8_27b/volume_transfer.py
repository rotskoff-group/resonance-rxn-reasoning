"""Copy 27B artifacts between the Modal volumes and the local run directory.

The published runs did these copies by hand with `modal volume get/put`; this module
records them as commands. Local layout matches pipeline_root()/outputs/<RUN_ID>/...

CLI: python -m resonance.stages.qwen3_8_27b.volume_transfer STEP [--dry-run]
    get-s2-rollouts   S1 train rollouts (s2/rollouts/{0,1}) used to build the S2 datasets
    put-s2-v5         upload S2_V5_difficulty_mix_dataset for s2_difficulty_one_epoch.py
    get-s2-eval       S2-V5 test rollouts (s2-v5-difficulty-mix-1ep/eval/shards/{0..3})
    get-s3-eval       S3 test rollouts (s3-from-s2-v5-difficulty-mix-1ep/eval/shards/{0..3})
"""
import argparse
import subprocess
import sys

from resonance.stages.qwen3_8_27b.config import PIPELINE_VOLUME, RUN_ID, pipeline_root

ROLLOUT_FILES = ('rollouts.jsonl', 'COMPLETE.json', 'identity.json', 'progress.json', 'execution.json')
EVAL_FILES = ('rollouts.jsonl', 'COMPLETE.json')


def plan(step):
    """(direction, remote path on the volume, local path) triples for one step."""
    local = pipeline_root() / 'outputs' / RUN_ID
    remote = f'/runs/{RUN_ID}'
    if step == 'get-s2-rollouts':
        return [('get', f'{remote}/s2/rollouts/{s}/{name}', local / f's2/rollouts/{s}/{name}')
                for s in range(2) for name in ROLLOUT_FILES]
    if step == 'put-s2-v5':
        name = 'S2_V5_difficulty_mix_dataset'
        return [('put', f'{remote}/{name}', local / name)]
    variant = {'get-s2-eval': 's2-v5-difficulty-mix-1ep',
               'get-s3-eval': 's3-from-s2-v5-difficulty-mix-1ep'}[step]
    return [('get', f'{remote}/{variant}/eval/shards/{s}/{name}',
             local / f'{variant}/eval/shards/{s}/{name}')
            for s in range(4) for name in EVAL_FILES]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['get-s2-rollouts', 'put-s2-v5', 'get-s2-eval', 'get-s3-eval'])
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    for direction, remote, local in plan(args.step):
        if direction == 'get':
            local.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, '-m', 'modal', 'volume', 'get', '--force', PIPELINE_VOLUME, remote, str(local)]
        else:
            assert (local / 'manifest.json').is_file(), f'Build {local} first (build_s2_difficulty_mix)'
            cmd = [sys.executable, '-m', 'modal', 'volume', 'put', PIPELINE_VOLUME, str(local), remote]
        print(' '.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
