"""Stage data with validated rollout identity and an explicit S2 selection policy.

Ported from run_dir('ablation_27b')/scripts/prepare.py. `select` is the pipeline's
prepare_stage.select (byte-identical to the ablation's reference_selection.select).
CLI: python prepare.py --root ARM_STAGE_ROOT --model SOURCE_MODEL --stage {s2,s3}
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from resonance.stages.common import iter_jsonl, write_json
from resonance.stages.qwen3_8_27b.prepare_stage import select


def select_s2(rows, tok, policy):
    assert policy in ('difficulty_mix', 'all_correct')
    data, rejected, histogram = [], Counter(), Counter()
    for row in rows:
        assert len(row['completions']) == 8
        prompt, usable = list(row['prompt_ids']), []
        for ci, completion in enumerate(row['completions']):
            ids = list(completion['token_ids'])
            if not ids or ids[-1] != tok.eos_token_id:
                ids.append(tok.eos_token_id)
            reason = ('incorrect' if not completion['correct'] else
                      'unfinished' if completion['finish_reason'] != 'stop' else
                      'overlength' if len(prompt) + len(ids) > 2048 else None)
            if reason:
                rejected[reason] += 1
            else:
                usable.append((ci, ids))
        k = len(usable)
        histogram[k] += 1
        if not usable and policy == 'difficulty_mix':
            ids = tok.encode(f"</think>\n\n<answer>\n{row['ground_truth']}\n</answer>",
                             add_special_tokens=False)
            if not ids or ids[-1] != tok.eos_token_id:
                ids.append(tok.eos_token_id)
            usable = [(None, ids)]
        for ci, ids in usable:
            assert len(prompt) + len(ids) <= 2048
            data.append({'input_ids': prompt + ids, 'labels': [-100]*len(prompt) + ids,
                         'row_idx': row['row_idx'], 'completion_idx': ci,
                         'usable_correct_count': k,
                         'source_type': 'gold_answer_only' if ci is None else 'student_correct'})
    data.sort(key=lambda r: (r['row_idx'], r['completion_idx'] or 0))
    if policy == 'difficulty_mix':
        hard, gold, easy = [], [], defaultdict(list)
        for i, row in enumerate(data):
            k = row['usable_correct_count']
            if k == 0:
                gold.append(i)
            elif k == 8:
                easy[row['row_idx']].append(i)
            else:
                hard.append(i)
        rng = random.Random(42)
        replay, remaining = [], []
        easy_ids = sorted(easy)
        for idx in easy_ids:
            rng.shuffle(easy[idx])
            replay.append(easy[idx][0])
        # Preserve the reference policy, capping the requested replay at available data.
        replay_target = min(8*len(easy), max(len(easy), round(len(hard)/2)))
        rng.shuffle(easy_ids)
        for slot in range(1, 8):
            remaining.extend(easy[idx][slot] for idx in easy_ids)
        replay.extend(remaining[:replay_target-len(replay)])
        data = [data[i] for i in sorted(hard + gold + replay)]
    groups = Counter('gold_fallback' if r['usable_correct_count'] == 0 else
                     'easy_replay' if r['usable_correct_count'] == 8 else 'mixed_success'
                     for r in data)
    return data, {'examples': len(data), 'unique_reactions': len({r['row_idx'] for r in data}),
                  'excluded_completions': dict(rejected), 'usable_correct_histogram': dict(histogram),
                  'groups': dict(groups), 'policy': policy, 'loss': 'assistant tokens only',
                  'max_context': 2048, 'selection_seed': 42}


def main():
    from datasets import Dataset, load_from_disk
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--stage', choices=['s2', 's3'], required=True)
    args = ap.parse_args()
    root = args.root
    identity = json.loads((root/'input.json').read_text())
    assert not (root/'dataset').exists(), 'Inspect existing dataset before rebuilding'
    expected = identity['train_rows']
    def rows():
        seen = set()
        for shard in range(4):
            folder = root/f'rollouts/{shard}'
            report = json.loads((folder/'COMPLETE.json').read_text())
            ri = report['identity']
            assert ri['model'] == args.model and ri['stage'] == args.stage
            assert ri['data_sha256'] == identity['train_sha256']
            assert ri['prompts_sha256'] == identity['prompts_sha256']
            assert ri['shard'] == shard and ri['shards'] == 4 and ri['limit'] == 0
            assert ri['n'] == (8 if args.stage == 's2' else 16)
            count = 0
            for row in iter_jsonl(folder/'rollouts.jsonl'):
                idx = row['row_idx']
                assert idx not in seen and idx % 4 == shard and 0 <= idx < expected
                assert len(row['completions']) == ri['n']
                seen.add(idx)
                count += 1
                yield row
            assert count == report['rows']
        assert seen == set(range(expected)), 'Incomplete rollout coverage'
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.stage == 's2':
        data, stats = select_s2(rows(), tok, identity['s2_policy'])
        if identity['s2_policy'] == 'difficulty_mix':
            assert stats['unique_reactions'] == expected
    else:
        data, stats = select(rows(), 's3', tok.eos_token_id)
    assert data, 'No usable training examples; downstream training stopped'
    ds = Dataset.from_list(data)
    ds.save_to_disk(str(root/'dataset'))
    assert len(load_from_disk(str(root/'dataset'))) == len(data)
    stats.update({'stage': args.stage, 'source': args.model, 'input': identity,
                  'dataset_fingerprint': ds._fingerprint})
    write_json(root/'data_stats.json', stats)
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
