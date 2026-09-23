"""Build S2 token sequences or S3 tokenized preferences without retemplating thoughts.

Ported from run_dir('pipeline_27b')/scripts/prepare_stage.py.
CLI: python prepare_stage.py --root STAGE_ROOT --model SOURCE_MODEL --stage {s2,s3} [--shards n]
"""
import argparse
import json
import random
from pathlib import Path
from resonance.stages.common import iter_jsonl, similarity, write_json


def select(rows, stage, eos):
    result = []
    stats = {'reactions': 0, 'truncated_excluded': 0, 'overlength_excluded': 0,
             'no_usable_pair': 0, 'examples': 0}
    for row in rows:
        stats['reactions'] += 1
        rng = random.Random(42+row['row_idx'])
        usable = []
        for c in row['completions']:
            if c['finish_reason'] != 'stop':
                stats['truncated_excluded'] += 1
                continue
            ids = list(c['token_ids'])
            if not ids or ids[-1] != eos:
                ids.append(eos)
            length = len(row['prompt_ids']) + len(ids)
            if length > (2048 if stage == 's2' else 4096):
                stats['overlength_excluded'] += 1
                continue
            usable.append({**c, 'ids': ids})
        correct = [c for c in usable if c['correct']]
        if stage == 's2':
            result.extend({'input_ids': row['prompt_ids'] + c['ids']} for c in correct)
        else:
            wrong = [c for c in usable if not c['correct']]
            if not correct or not wrong:
                stats['no_usable_pair'] += 1
                continue
            chosen = rng.choice(correct)
            rejected = max(wrong, key=lambda c: similarity(c['prediction'], row['ground_truth']))
            result.append({'prompt_ids': row['prompt_ids'], 'chosen_ids': chosen['ids'],
                           'rejected_ids': rejected['ids']})
    stats['examples'] = len(result)
    return result, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--stage', choices=['s2', 's3'], required=True)
    ap.add_argument('--shards', type=int, default=2)
    args = ap.parse_args()
    from datasets import Dataset
    from transformers import AutoTokenizer
    root = Path(args.root)
    assert args.shards > 0
    reports = [json.loads((root / f'rollouts/{s}/COMPLETE.json').read_text()) for s in range(args.shards)]
    expected = json.loads((root / 'input.json').read_text())['train_rows']
    def rows():
        seen = set()
        for shard in range(args.shards):
            for row in iter_jsonl(root/f'rollouts/{shard}/rollouts.jsonl'):
                assert row['row_idx'] not in seen
                seen.add(row['row_idx'])
                yield row
        assert seen == set(range(expected)), 'Missing or extra training reactions'
    for report in reports:
        assert report['identity']['model'] == args.model
        assert report['identity']['stage'] == args.stage
        assert report['identity']['data_sha256'] == json.loads((root/'input.json').read_text())['train_sha256']
    tok = AutoTokenizer.from_pretrained(args.model)
    data, stats = select(rows(), args.stage, tok.eos_token_id)
    assert data, 'No usable training examples'
    Dataset.from_list(data).save_to_disk(str(root / 'dataset'))
    stats.update({'stage': args.stage, 'model': args.model, 'source_rows': expected,
                  'loss': 'all non-padding tokens' if args.stage=='s2' else 'DPO sigmoid + 0.2 chosen NLL'})
    write_json(root / 'data_stats.json', stats)
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
