"""Validate the complete no-ID rollout corpus and retain usable exact-correct traces.

Ported from run_dir('s2_8b')/scripts/filter_rollouts.py.

    python -m resonance.stages.qwen3_8b.filter_rollouts [--out-dir DIR]

Writes <out-dir>/sft_dataset, filter_stats.json and DATASET_COMPLETE.json
(default out-dir: run_dir('s2_8b')/outputs).
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
from rdkit import RDLogger
from resonance.evaluation.extractors import extract_forward_rxn, match_forward_rxn
from resonance.paths import run_dir
from resonance.stages.qwen3_8b import sample_rollouts
from resonance.stages.qwen3_8b._common import S1_MODEL, TRAIN_JSON

ROOT = run_dir('s2_8b')
SAMPLE = run_dir('s1_8b')
MODEL = S1_MODEL
MAX_LENGTH = 1024  # Prior 8B S2 recipe; reject rather than truncate longer traces.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(ROOT/'outputs'))
    args = ap.parse_args()
    RDLogger.DisableLog('rdApp.*')
    complete = json.loads((SAMPLE/'outputs/rollouts/COMPLETE.json').read_text())
    assert complete['model'] == str(MODEL)
    assert complete['rows'] == 40000 and complete['samples_per_row'] == 8
    source = SAMPLE/'outputs/rollouts/rollouts.jsonl'
    out = Path(args.out_dir)
    assert not (out/'sft_dataset').exists(), 'Dataset already exists; inspect before rebuilding'
    train = json.loads(TRAIN_JSON.read_text())
    assert len(train) == 40000
    # The exact rollout prompt constants (sample_rollouts imports vLLM only inside main).
    prompts = {'SYSTEM_PROMPT': sample_rollouts.SYSTEM_PROMPT,
               'USER_TEMPLATE': sample_rollouts.USER_TEMPLATE}
    tok = AutoTokenizer.from_pretrained(MODEL)
    stats = Counter()
    texts, row_ids, sample_ids, lengths = [], [], [], []
    covered = set()
    digest = hashlib.sha256()
    with source.open('rb') as f:
        for idx, line in enumerate(f):
            digest.update(line)
            row = json.loads(line)
            assert row['row_idx'] == idx
            expected = train[idx]
            smiles = expected['original_smiles']
            assert row['reaction_id'] == expected['metadata']['reaction_id']
            assert row['starting_material'] == smiles['Starting Material']
            assert row['reagent_list'] == (smiles.get('Reagents') or '')
            assert row['gold_product'] == smiles['End Product']
            assert len(row['responses']) == len(row['finish_reasons']) == 8
            messages = [
                {'role': 'system', 'content': prompts['SYSTEM_PROMPT']},
                {'role': 'user', 'content': prompts['USER_TEMPLATE'].format(
                    starting_material=row['starting_material'],
                    reagent_list=row['reagent_list'] or 'None specified')},
            ]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for sample, (response, finish) in enumerate(zip(row['responses'], row['finish_reasons'])):
                stats['responses'] += 1
                if finish != 'stop':
                    stats['rejected_unfinished'] += 1
                    continue
                predicted = extract_forward_rxn(response)
                if predicted is None or not match_forward_rxn(predicted, row['gold_product'], 'smiles'):
                    stats['rejected_incorrect_or_invalid'] += 1
                    continue
                stats['finished_correct'] += 1
                # Re-rendering an assistant turn can strip/rewrite reasoning tags.
                # Keep exactly the prompt used by llm.chat and the generated text.
                text = prompt + response + tok.eos_token + '\n'
                length = len(tok.encode(text, add_special_tokens=False))
                if length > MAX_LENGTH:
                    stats['rejected_overlength_correct'] += 1
                    continue
                texts.append(text)
                row_ids.append(idx)
                sample_ids.append(sample)
                lengths.append(length)
                covered.add(idx)
                stats['accepted'] += 1
            stats['reactions'] += 1
            if stats['reactions'] % 1000 == 0:
                print(json.dumps(dict(stats)), flush=True)
    assert stats['reactions'] == 40000 and stats['responses'] == 320000
    assert texts, 'No usable training examples'
    assert stats['responses'] == sum(stats[k] for k in (
        'rejected_unfinished', 'rejected_incorrect_or_invalid', 'rejected_overlength_correct', 'accepted'))
    ds = Dataset.from_dict({'text': texts, 'row_idx': row_ids, 'sample_idx': sample_ids,
                            'num_tokens': lengths})
    ds.save_to_disk(out/'sft_dataset')
    loaded = load_from_disk(out/'sft_dataset')
    assert len(loaded) == stats['accepted'] and max(loaded['num_tokens']) <= MAX_LENGTH
    report = {**dict(stats), 'reactions_covered': len(covered),
              'source': str(source), 'source_sha256': digest.hexdigest(), 'model': str(MODEL),
              'max_length': MAX_LENGTH, 'max_accepted_tokens': max(lengths),
              'dataset_fingerprint': loaded._fingerprint,
              'prompt_sha256': hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()}
    (out/'filter_stats.json').write_text(json.dumps(report, indent=2)+'\n')
    (out/'DATASET_COMPLETE.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
