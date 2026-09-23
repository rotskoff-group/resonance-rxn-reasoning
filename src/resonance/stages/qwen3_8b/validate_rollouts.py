"""Require the complete train corpus before mining any self-play pairs.

Ported from run_dir('s3_8b')/scripts/validate_rollouts.py.

    python -m resonance.stages.qwen3_8b.validate_rollouts [completions.jsonl]
"""
import json
import sys
from pathlib import Path

from resonance.paths import run_dir
from resonance.stages.qwen3_8b._common import TRAIN_JSON

if __name__ == '__main__':
    source = json.loads(TRAIN_JSON.read_text())
    completions = Path(sys.argv[1]) if len(sys.argv) > 1 else run_dir("s3_8b") / "outputs/completions.jsonl"
    count = 0
    with completions.open() as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            assert i < len(source)
            assert row["reaction_id"] == source[i]["metadata"]["reaction_id"]
            assert len(row["completions"]) == 16
            assert row["prompt_messages"]
            count += 1
    assert count == len(source) == 40000, count
    print(f"Verified {count} reactions and {count * 16} completions")
