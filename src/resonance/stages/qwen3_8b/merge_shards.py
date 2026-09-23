"""Validate and atomically merge the eight contiguous self-play shards.

Ported from run_dir('s3_8b')/scripts/merge_shards.py.

    python -m resonance.stages.qwen3_8b.merge_shards [--shard-dir DIR] [--out FILE]
"""
import argparse
import json
import os
from pathlib import Path

from resonance.paths import run_dir
from resonance.stages.qwen3_8b._common import TRAIN_JSON

if __name__ == '__main__':
    _ap = argparse.ArgumentParser()
    _ap.add_argument('--shard-dir', default=str(run_dir('s3_8b') / 'outputs'))
    _ap.add_argument('--out', default=None, help='Default: <shard-dir>/completions.jsonl')
    _args = _ap.parse_args()
    OUT = Path(_args.shard_dir)
    source = json.loads(TRAIN_JSON.read_text())
    nshards = 8
    target = Path(_args.out) if _args.out else OUT / 'completions.jsonl'
    assert not target.exists(), target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.jsonl.tmp')
    count = 0
    with tmp.open('w') as dst:
        for shard in range(nshards):
            path = OUT / f'completions.shard{shard}-of-{nshards}.jsonl'
            assert path.is_file() and path.stat().st_size > 0, path
            with path.open() as src:
                for line in src:
                    row = json.loads(line)
                    assert row['row_idx'] == count, (path, row['row_idx'], count)
                    assert row['reaction_id'] == source[count]['metadata']['reaction_id']
                    assert len(row['completions']) == 16
                    assert row['prompt_messages']
                    dst.write(line)
                    count += 1
    assert count == len(source) == 40000, count
    os.replace(tmp, target)
    print(f'Verified and merged {count} reactions / {count * 16} completions -> {target}')
