"""Resumable vLLM generation used identically on Modal and local evaluation.

Ported from run_dir('pipeline_27b')/scripts/generate.py.
CLI: python generate.py --model M --data TRAIN_OR_TEST_JSON --prompts PROMPTS_JSON --out DIR
     --stage {s2,s3,eval} [--shard i --shards n --tp 2 --limit k]
"""
import argparse
import json
import os
import time
from pathlib import Path

from resonance.stages.common import checkpoint, iter_jsonl, prediction, records, sha256, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage', required=True, choices=['s2', 's3', 'eval'])
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--tp', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    checkpoint(args.model)
    settings = {'s2': (8, .9, 1024, 2048), 's3': (16, .7, 8192, 10240),
                'eval': (1, .6, 16384, 20480)}
    n, temperature, max_tokens, context = settings[args.stage]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    identity = {'model': args.model, 'data_sha256': sha256(args.data),
                'prompts_sha256': sha256(args.prompts), 'stage': args.stage,
                'shard': args.shard, 'shards': args.shards, 'n': n,
                'temperature': temperature, 'max_tokens': max_tokens,
                'max_model_len': context, 'seed': 42, 'limit': args.limit,
                'top_p': .95, 'top_k': 20, 'thinking': True}
    if (out / 'identity.json').exists():
        assert json.loads((out / 'identity.json').read_text()) == identity, 'Resume identity mismatch'
    write_json(out / 'identity.json', identity)
    dataset = records(args.data, json.loads(Path(args.prompts).read_text()))
    if args.limit:
        dataset = dataset[:args.limit]
    dataset = [r for r in dataset if r['row_idx'] % args.shards == args.shard]
    done = set()
    for row in iter_jsonl(out/'rollouts.jsonl', repair=True):
        assert row['row_idx'] not in done, 'Duplicate rollout rows'
        done.add(row['row_idx'])
        assert len(row['completions']) == n
    assert done <= {r['row_idx'] for r in dataset}
    pending = [r for r in dataset if r['row_idx'] not in done]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if pending:
        llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype='bfloat16',
                   max_model_len=context, gpu_memory_utilization=.85, seed=42,
                   max_num_seqs=64, enable_prefix_caching=True,
                   limit_mm_per_prompt={'image': 0, 'video': 0})
        started = time.monotonic()
        with (out / 'rollouts.jsonl').open('a') as f:
            for start in range(0, len(pending), 32):
                chunk = pending[start:start+32]
                formatted = [tokenizer.apply_chat_template(r['messages'], tokenize=False,
                    add_generation_prompt=True, enable_thinking=True, preserve_thinking=True)
                    for r in chunk]
                for prompt in formatted:
                    assert '<think>' in prompt
                    assert len(tokenizer.encode(prompt, add_special_tokens=False)) + max_tokens <= context
                # Seed per row makes restart/sharding independent of chunk boundaries.
                params = [SamplingParams(n=n, temperature=temperature, top_p=.95, top_k=20,
                          max_tokens=max_tokens, seed=42+r['row_idx']) for r in chunk]
                generated = llm.generate(formatted, params, use_tqdm=False)
                for row, prompt, result in zip(chunk, formatted, generated, strict=True):
                    completions = []
                    for gen in result.outputs:
                        ids = list(gen.token_ids)
                        pred = prediction(gen.text)
                        completions.append({'text': gen.text, 'token_ids': ids,
                            'finish_reason': gen.finish_reason, 'prediction': pred,
                            'correct': pred is not None and pred == row['ground_truth']})
                    assert len(completions) == n
                    assert result.prompt_token_ids is not None
                    record = {**row, 'prompt': prompt,
                              'prompt_ids': list(result.prompt_token_ids), 'completions': completions}
                    f.write(json.dumps(record) + '\n')
                f.flush()
                os.fsync(f.fileno())
                count = min(start+32, len(pending))
                elapsed = time.monotonic()-started
                write_json(out / 'progress.json', {'new_rows': count, 'remaining': len(pending)-count,
                    'seconds': elapsed, 'eta_hours': (len(pending)-count)*elapsed/count/3600})
                print(json.dumps({'completed': len(done)+count, 'total': len(dataset),
                                  'elapsed_seconds': elapsed}), flush=True)
                if os.environ.get('PIPELINE_VOLUME'):
                    import modal
                    modal.Volume.from_name(os.environ['PIPELINE_VOLUME']).commit()
    ids = set()
    total = correct = valid = truncated = tokens = 0
    for row in iter_jsonl(out/'rollouts.jsonl'):
        assert row['row_idx'] not in ids
        ids.add(row['row_idx'])
        for c in row['completions']:
            total += 1
            correct += c['correct']
            valid += c['prediction'] is not None
            truncated += c['finish_reason']=='length'
            tokens += len(c['token_ids'])
    assert ids == {r['row_idx'] for r in dataset}
    assert total == len(dataset)*n
    report = {'identity': identity, 'rows': len(ids), 'completions': total,
              'accuracy': correct/total, 'valid_smiles_fraction': valid/total,
              'truncation_fraction': truncated/total, 'mean_output_tokens': tokens/total}
    write_json(out / 'COMPLETE.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
