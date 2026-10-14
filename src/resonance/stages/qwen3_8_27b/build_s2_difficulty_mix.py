"""Keep all mixed-success correct traces, with a 2:1 ratio to easy replay.

Ported from run_dir('pipeline_27b')/scripts/build_s2_difficulty_mix.py.
CLI: python -m resonance.stages.qwen3_8_27b.build_s2_difficulty_mix [--run-root RUN]
     [--source RUN/S2_V4_full_all_correct_dataset] [--out RUN/S2_V5_difficulty_mix_dataset]
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, load_from_disk

from resonance.stages.common import sha256
from resonance.stages.qwen3_8_27b.config import RUN_ID, pipeline_root

RUN = pipeline_root()/'outputs'/RUN_ID
SOURCE = RUN/'S2_V4_full_all_correct_dataset'
OUT = RUN/'S2_V5_difficulty_mix_dataset'


def main():
    assert not OUT.exists(), f'Refusing to overwrite {OUT}'
    data = load_from_disk(str(SOURCE/'dataset'))
    assert isinstance(data, Dataset)
    metadata = data.select_columns(['row_idx', 's1_pass_at_8', 'source_type', 'reaction_class',
                                    'prompt_tokens', 'assistant_tokens'])
    hard, gold, easy = [], [], defaultdict(list)
    for i, row in enumerate(metadata):
        k = row['s1_pass_at_8']
        if k == 0:
            assert row['source_type'] == 'gold_answer_only'
            gold.append(i)
        elif k == 8:
            easy[row['row_idx']].append(i)
        else:
            assert 1 <= k <= 7 and row['source_type'] == 's1_all_correct'
            hard.append(i)
    rng = random.Random(42)
    replay = []
    remaining = []
    # Every easy reaction is retained once; extra traces are distributed by reaction.
    easy_ids = sorted(easy)
    for idx in easy_ids:
        rng.shuffle(easy[idx])
        replay.append(easy[idx][0])
    replay_target = max(len(easy), round(len(hard)/2))
    rng.shuffle(easy_ids)
    for slot in range(1,8):
        remaining.extend(easy[idx][slot] for idx in easy_ids)
    replay.extend(remaining[:replay_target-len(replay)])
    selected = sorted(hard + gold + replay)
    assert len(set(selected)) == len(selected)
    ds = data.select(selected)
    stats = Counter()
    tokens = Counter()
    classes = Counter()
    ids = set()
    for row in ds.select_columns(metadata.column_names):
        k = row['s1_pass_at_8']
        group = 'gold_fallback' if k == 0 else 'easy_replay' if k == 8 else 'mixed_success'
        stats[group] += 1
        tokens['prompt'] += row['prompt_tokens']
        tokens['assistant'] += row['assistant_tokens']
        classes[row['reaction_class']] += 1
        ids.add(row['row_idx'])
    assert ids == set(range(40000))
    assert stats['mixed_success'] == len(hard) and stats['easy_replay'] == replay_target
    OUT.mkdir()
    ds.save_to_disk(str(OUT/'dataset'))
    wanted = set(selected)
    with (SOURCE/'provenance.jsonl').open() as source, (OUT/'provenance.jsonl').open('w') as target:
        n = 0
        for i,line in enumerate(source):
            if i in wanted:
                r = json.loads(line)
                assert r['dataset_row'] == i
                r['source_dataset_row'] = i
                r['dataset_row'] = n
                target.write(json.dumps(r)+'\n')
                n += 1
    assert n == len(ds)
    manifest = {'name':OUT.name,'examples':len(ds),'unique_reactions':len(ids),
                'policy':{'difficulty':'number of usable exact-correct S1 completions out of 8 on TRAIN only',
                          'mixed_success':'all correct completions from 1–7/8 reactions',
                          'easy_replay':'one random completion per 8/8 reaction, plus a second for randomly chosen '
                                        'reactions to approach 2:1 mixed/easy ratio',
                          'zero_success':'one existing gold-answer-only fallback per 0/8 reaction',
                          'selection_seed':42,'loss':'assistant tokens only','max_context':2048,
                          'epochs':1,'initialization':'S1','class_selection':'none; all 40000 train reactions retained',
                          'rationale':'Practical starting mix, not a validated optimal ratio. Emphasize unreliable '
                                      'successes while retaining all reaction coverage before DPO.'},
                'groups':dict(stats),'class_counts':dict(sorted(classes.items())),
                'tokens':{**tokens,'total':sum(tokens.values())},
                'source_manifest':str(SOURCE/'manifest.json'),
                'source_manifest_sha256':sha256(SOURCE/'manifest.json'),
                'source_fingerprint':data._fingerprint}
    manifest['dataset_files'] = {str(p.relative_to(OUT)):sha256(p)
                                 for p in sorted((OUT/'dataset').rglob('*')) if p.is_file()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert len(load_from_disk(str(OUT/'dataset'))) == len(ds)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root', type=Path, default=RUN)
    ap.add_argument('--source', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=None)
    cli = ap.parse_args()
    SOURCE = cli.source or cli.run_root/'S2_V4_full_all_correct_dataset'
    OUT = cli.out or cli.run_root/'S2_V5_difficulty_mix_dataset'
    main()
