"""Check that a checkpoint reproduces its recorded USPTO-50k test generations.

Re-runs the first `--rows` recorded prompts with the recorded sampling settings and
compares the new generations with the saved ones: Top-1 accuracy on both, how often
the extracted answer is the same, and how often the whole response is byte-identical.
vLLM batching makes outputs close but not always identical, so the check is on accuracy
and answer agreement; identical text is reported for information.

Two recorded formats:
    benchmark   8B `results.jsonl` (idx, system_prompt, task_prompt, response, answer),
                replayed with `llm.chat`.
    rollouts    27B `rollouts.jsonl` (row_idx, prompt_ids, completions, ground_truth),
                replayed from the saved token ids with seed 42 + row_idx.

    python -m resonance.release.replay --model <dir> --recorded <results.jsonl> \
        --format benchmark --seed fixed --out replay.json
"""
import argparse
import ast
import glob
import json
from pathlib import Path
from typing import Any

from resonance.evaluation.extractors import extract_forward_rxn, match_forward_rxn


def load(pattern, fmt, rows):
    # A glob, so sharded 27B evaluations (eval/shards/*/rollouts.jsonl) load as one set.
    paths = sorted(glob.glob(pattern))
    assert paths, f'no recorded generations match {pattern}'
    records = [json.loads(line) for path in paths for line in open(path)]
    key = 'idx' if fmt == 'benchmark' else 'row_idx'
    records.sort(key=lambda r: int(r[key]))
    records = records[:rows]
    for r in records:
        if fmt == 'benchmark':
            r['_row'] = int(r['idx'])
            r['_gold'] = r['answer']
            r['_saved'] = r['response']
        else:
            r['_row'] = int(r['row_idx'])
            r['_gold'] = r['ground_truth']
            completions = r['completions']
            if isinstance(completions, str):
                completions = ast.literal_eval(completions)
            r['_saved'] = completions[0]['text']
            ids = r['prompt_ids']
            r['_prompt_ids'] = ast.literal_eval(ids) if isinstance(ids, str) else ids
    return records


def correct(text, gold):
    answer = extract_forward_rxn(text or '')
    return answer, bool(answer) and match_forward_rxn(answer, gold, 'smiles')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--recorded', required=True, help='recorded generations (a glob for shards)')
    ap.add_argument('--format', choices=['benchmark', 'rollouts'], required=True)
    ap.add_argument('--seed', choices=['fixed', 'row', 'none'], required=True,
                    help='fixed: 42 on every request; row: 42 + row index; none: unseeded')
    ap.add_argument('--rows', type=int, default=200)
    ap.add_argument('--max-model-len', type=int, default=16384)
    ap.add_argument('--max-tokens', type=int, default=16384)
    ap.add_argument('--tp', type=int, default=2)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    records = load(args.recorded, args.format, args.rows)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.85, seed=42,
              limit_mm_per_prompt={'image': 0, 'video': 0})

    def params(row):
        seed = {'fixed': 42, 'row': 42 + row, 'none': None}[args.seed]
        return SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.max_tokens, seed=seed)

    if args.format == 'benchmark':
        messages: list[Any] = [[{'role': 'system', 'content': r['system_prompt']},
                     {'role': 'user', 'content': r['task_prompt']}] for r in records]
        new = []
        for start in range(0, len(records), 64):   # the published runs used batches of 64
            chunk = records[start:start + 64]
            outputs = llm.chat(messages[start:start + 64], [params(r['_row']) for r in chunk], use_tqdm=False)
            new += [o.outputs[0].text for o in outputs]
    else:
        outputs = llm.generate([{'prompt_token_ids': r['_prompt_ids']} for r in records],
                               [params(r['_row']) for r in records], use_tqdm=False)
        new = [o.outputs[0].text for o in outputs]

    rows, same_answer, same_text, saved_ok, new_ok = [], 0, 0, 0, 0
    for r, text in zip(records, new, strict=True):
        saved_answer, s_ok = correct(r['_saved'], r['_gold'])
        new_answer, n_ok = correct(text, r['_gold'])
        saved_ok += s_ok
        new_ok += n_ok
        agree = (saved_answer == new_answer) or (s_ok and n_ok)
        same_answer += agree
        same_text += text == r['_saved']
        rows.append({'row': r['_row'], 'saved_correct': s_ok, 'new_correct': n_ok,
                     'saved_answer': saved_answer, 'new_answer': new_answer})
    n = len(records)
    report = {'model': args.model, 'recorded': args.recorded, 'rows': n, 'seed': args.seed,
              'saved_accuracy': saved_ok / n, 'replay_accuracy': new_ok / n,
              'answer_agreement': same_answer / n, 'identical_text': same_text / n, 'per_row': rows}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != 'per_row'}, indent=1))


if __name__ == '__main__':
    main()
