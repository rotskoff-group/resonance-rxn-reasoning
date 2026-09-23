"""Validate and merge four completed deterministic base-evaluation shards.

Ported from run_dir('pipeline_27b')/scripts/merge_base_eval_shards.py.
CLI: python -m resonance.stages.qwen3_8_27b.merge_base_eval_shards [--root PIPELINE_DIR]
"""
import json
import os
import shutil
from pathlib import Path

from resonance.stages.common import iter_jsonl, write_json
from resonance.stages.qwen3_8_27b.config import RUN_ID, pipeline_root

ROOT = pipeline_root()


def main():
    run_id = RUN_ID
    base = ROOT/'outputs'/run_id/'base'
    shard_root = base/'eval-shards-4'
    rows = []
    reports = []
    for shard in range(4):
        directory = shard_root/f'shard-{shard}'
        report = json.loads((directory/'COMPLETE.json').read_text())
        assert report['rows'] == 1250 and report['completions'] == 1250
        assert report['identity']['shard'] == shard and report['identity']['shards'] == 4
        shard_rows = list(iter_jsonl(directory/'rollouts.jsonl'))
        assert len(shard_rows) == 1250
        assert all(row['row_idx'] % 4 == shard for row in shard_rows)
        rows.extend(shard_rows)
        reports.append(report)
    rows.sort(key=lambda row: row['row_idx'])
    assert [row['row_idx'] for row in rows] == list(range(5000))
    identities = [r['identity'] for r in reports]
    for key in ('model','data_sha256','prompts_sha256','stage','n','temperature',
                'max_tokens','max_model_len','seed','limit','top_p','top_k','thinking'):
        assert len({i[key] for i in identities}) == 1, key

    target = base/'eval'
    assert not target.exists(), f'Refusing to replace existing {target}'
    tmp = base/'eval.merge-tmp'
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    with (tmp/'rollouts.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row)+'\n')
        f.flush()
        os.fsync(f.fileno())
    identity = dict(identities[0], shard=0, shards=1)
    write_json(tmp/'identity.json', identity)
    total = len(rows)
    correct = sum(c['correct'] for row in rows for c in row['completions'])
    valid = sum(c['prediction'] is not None for row in rows for c in row['completions'])
    truncated = sum(c['finish_reason'] == 'length' for row in rows for c in row['completions'])
    tokens = sum(len(c['token_ids']) for row in rows for c in row['completions'])
    report = {'identity': identity, 'rows': total, 'completions': total,
              'accuracy': correct/total, 'valid_smiles_fraction': valid/total,
              'truncation_fraction': truncated/total, 'mean_output_tokens': tokens/total,
              'execution_shards': 4}
    write_json(tmp/'progress.json', {'new_rows': total, 'remaining': 0, 'seconds': 0,
                                     'eta_hours': 0, 'merged_from_shards': 4})
    write_json(tmp/'COMPLETE.json', report)
    os.replace(tmp, target)

    results = {}
    for stage in ('base','s1','s2','s3'):
        path = ROOT/'outputs'/run_id/stage/'eval/COMPLETE.json'
        if path.exists():
            results[stage] = json.loads(path.read_text())
    for key in ('data_sha256','prompts_sha256','temperature','max_tokens','max_model_len',
                'seed','top_p','top_k','thinking','n','limit'):
        assert len({r['identity'][key] for r in results.values()}) == 1, key
    write_json(ROOT/'outputs'/run_id/'comparison.json', results)
    lines = ['| Stage | Exact accuracy | Valid SMILES | Truncated | Mean output tokens |',
             '|---|---:|---:|---:|---:|']
    for stage, result in results.items():
        lines.append(f"| {stage.upper()} | {result['accuracy']:.2%} | "
                     f"{result['valid_smiles_fraction']:.2%} | "
                     f"{result['truncation_fraction']:.2%} | "
                     f"{result['mean_output_tokens']:.0f} |")
    (ROOT/'outputs'/run_id/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT,
                    help='directory holding outputs/<RUN_ID>/base/eval-shards-4')
    ROOT = ap.parse_args().root
    main()
