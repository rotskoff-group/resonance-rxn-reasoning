"""All usable correct S1 rollouts across the full USPTO train split.

Ported from run_dir('pipeline_27b')/scripts/build_s2_full.py. `sha256` now comes from
resonance.stages.common (same digest as build_s2_all_correct.sha256, which it imported).
CLI: python -m resonance.stages.qwen3_8_27b.build_s2_full --run-root RUN [--out DIR]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from resonance.stages.common import sha256


def main():
    from datasets import Dataset, load_from_disk
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=None,
                        help='default: RUN/S2_V4_full_all_correct_dataset')
    args = parser.parse_args()
    root = args.run_root.resolve()
    out = args.out.resolve() if args.out else root / 'S2_V4_full_all_correct_dataset'
    assert not out.exists(), f'Refusing to overwrite {out}'
    prior = json.loads((root / 'S2_V2_dataset/manifest.json').read_text())
    inputs = prior['inputs']
    for key in ('source', 'labels'):
        assert sha256(Path(inputs[key])) == inputs[key + '_sha256']
    source = json.loads(Path(inputs['source']).read_text())
    labels = json.loads(Path(inputs['labels']).read_text())['labels']
    assert len(source) == len(labels) == 40000
    tok = AutoTokenizer.from_pretrained(str(root / 's1/model'), local_files_only=True)
    eos = tok.eos_token_id
    assert eos is not None
    rows, provenance, files = [], [], []
    seen = set()
    rejected = Counter()
    coverage = Counter()
    for shard in (0, 1):
        path = root / f's2/rollouts/{shard}/rollouts.jsonl'
        identity = (json.loads((path.parent / 'identity.json').read_text())
                    if (path.parent / 'identity.json').exists()
                    else json.loads((path.parent / 'COMPLETE.json').read_text())['identity'])
        assert identity['data_sha256'] == inputs['source_sha256']
        files.append({'path': str(path), 'sha256': sha256(path)})
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                r = json.loads(line)
                idx = r['row_idx']
                assert idx not in seen and idx % 2 == shard and 0 <= idx < len(source)
                seen.add(idx)
                assert len(r['completions']) == 8
                prompt = list(r['prompt_ids'])
                usable = []
                for ci, c in enumerate(r['completions']):
                    ids = list(c['token_ids'])
                    if not ids or ids[-1] != eos:
                        ids.append(eos)
                    reason = ('incorrect' if not c['correct'] else 'unfinished' if c['finish_reason'] != 'stop'
                              else 'overlength' if len(prompt) + len(ids) > 2048 else None)
                    if reason:
                        rejected[reason] += 1
                    else:
                        usable.append((ci, ids))
                coverage[len(usable)] += 1
                source_type = 's1_all_correct' if usable else 'gold_answer_only'
                if not usable:
                    answer = f"</think>\n\n<answer>\n{r['ground_truth']}\n</answer>"
                    ids = tok.encode(answer, add_special_tokens=False)
                    if not ids or ids[-1] != eos:
                        ids.append(eos)
                    usable = [(None, ids)]
                for ci, ids in usable:
                    assert len(prompt) + len(ids) <= 2048
                    rows.append({'input_ids': prompt + ids, 'labels': [-100] * len(prompt) + ids,
                                 'row_idx': idx, 'reaction_id': source[idx]['metadata']['reaction_id'],
                                 'reaction_class': labels[str(idx)]['level1'], 'target_group': 'full',
                                 'source_type': source_type, 's1_pass_at_8': 0 if ci is None else len(usable),
                                 'prompt_tokens': len(prompt), 'assistant_tokens': len(ids)})
                    provenance.append({'row_idx': idx, 'completion_idx': ci, 'source_type': source_type,
                                       'rollout_file': str(path), 'rollout_line': line_number})
    assert seen == set(range(40000))
    order = sorted(range(len(rows)), key=lambda i: (rows[i]['row_idx'], provenance[i]['completion_idx'] or 0))
    rows = [rows[i] for i in order]
    provenance = [provenance[i] for i in order]
    tokens = {k: sum(r[k + '_tokens'] for r in rows) for k in ('prompt', 'assistant')}
    manifest = {'name': out.name, 'examples': len(rows), 'unique_reactions': len(seen),
                'policy': {'selection': 'all 40000 USPTO training reactions; no class upsampling',
                           'correct_rollout_choice':
                               'all exact-correct, normally finished samples fitting 2048 total tokens',
                           'zero_success_fallback': 'gold product, answer-only; same policy as S2 V2',
                           'loss': 'assistant tokens only',
                           'duplicate_completion_policy': 'retain each sampled completion',
                           'reaction_weighting': 'max(1, usable correct count)'},
                'source_types': dict(Counter(r['source_type'] for r in rows)),
                'class_counts': dict(sorted(Counter(r['reaction_class'] for r in rows).items())),
                'usable_correct_count_histogram': dict(sorted(coverage.items())),
                'excluded_completions': dict(rejected),
                'tokens': {**tokens, 'total': sum(tokens.values()), 'one_epoch_total': sum(tokens.values())},
                'inputs': {'original_inputs': inputs, 'rollouts': files}}
    assert len(rows) == sum(max(1, k) * v for k, v in coverage.items())
    out.mkdir()
    v2 = load_from_disk(str(root / 'S2_V2_dataset/dataset'))
    assert isinstance(v2, Dataset)
    features = v2.features
    Dataset.from_list(rows, features=features).save_to_disk(str(out / 'dataset'))
    with (out / 'provenance.jsonl').open('w') as handle:
        for i, p in enumerate(provenance):
            handle.write(json.dumps({'dataset_row': i, **p}) + '\n')
    assert len(load_from_disk(str(out / 'dataset'))) == len(rows)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
