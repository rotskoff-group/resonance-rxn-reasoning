"""Download committed stage artifacts without holding local GPUs.

Ported from run_dir('pipeline_27b')/scripts/download_checkpoint.py.
CLI: python -m resonance.stages.qwen3_8_27b.download_checkpoint --stage {s1,s2,s3}
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import modal
from resonance.stages.common import checkpoint, sha256, write_json

from resonance.stages.qwen3_8_27b.config import PIPELINE_VOLUME, S1_VOLUME, pipeline_root, traces_path

ROOT = pipeline_root()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['s1','s2','s3'], required=True)
    args = ap.parse_args()
    from resonance.pipelines.traces.validate_no_id import validate
    identity = validate(traces_path())
    run_id = 'no-id-' + identity['source_sha256'][:16]
    vol_name = (S1_VOLUME if args.stage=='s1' else PIPELINE_VOLUME)
    remote = f'/runs/{run_id}' + ('' if args.stage=='s1' else f'/{args.stage}')
    volume = modal.Volume.from_name(vol_name)
    manifest = json.loads(b''.join(volume.read_file(remote+'/COMPLETE.json')))
    source_sha = manifest['data']['source_sha256'] if args.stage=='s1' else manifest['source_sha256']
    assert source_sha == identity['source_sha256']
    target = ROOT/'outputs'/run_id/args.stage
    model = target/'model'
    model.mkdir(parents=True, exist_ok=True)
    if (target/'DOWNLOAD_COMPLETE.json').exists():
        checkpoint(model)
        return
    # Size checks at each file avoid filling the shared filesystem during another job's save.
    entries = list(volume.iterdir(remote+'/model', recursive=False))
    for entry in entries:
        name = Path(entry.path).name
        if name.startswith('.'):
            continue
        dest = model/name
        expected_hash = manifest.get('files', {}).get(name, {}).get('sha256')
        if dest.exists() and dest.stat().st_size == entry.size and (not expected_hash or sha256(dest)==expected_hash):
            continue
        assert shutil.disk_usage(ROOT).free > entry.size + 100*2**30, 'Less than 100 GiB safety margin on local disk'
        partial = dest.with_name(dest.name+'.partial')
        subprocess.run([sys.executable, '-m', 'modal', 'volume', 'get', '--force', vol_name,
                        remote+'/model/'+name, str(partial)], check=True)
        assert partial.stat().st_size == entry.size
        if expected_hash:
            assert sha256(partial) == expected_hash, f'Checksum mismatch: {name}'
        os.replace(partial, dest)
    checkpoint(model)
    write_json(target/'DOWNLOAD_COMPLETE.json', {'stage': args.stage, 'run_id': run_id,
        'volume': vol_name, 'remote': remote, 'source_sha256': source_sha, 'model': str(model),
        'verification': 'sha256' if manifest.get('files') else 'file sizes and complete shard index'})


if __name__ == '__main__':
    main()
