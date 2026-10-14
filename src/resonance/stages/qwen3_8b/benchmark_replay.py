"""Benchmark the merged DPO checkpoint against the exact saved S1 test prompts.

Ported from run_dir('s3_8b')/scripts/benchmark_dpo.py. It does not go through
workflows/benchmark: it replays the system/task prompts saved in the S1 benchmark results
(the "none" system prompt, see README.md) with vLLM `llm.chat`, in batches of 64.

    python -m resonance.stages.qwen3_8b.benchmark_replay   # normally via run_benchmark_gpu
"""
import json
import os
from typing import Any

from resonance.paths import run_dir
from resonance.stages.common import checkpoint, sha256, write_json

ROOT=run_dir('s3_8b')
SOURCE=run_dir('s1_8b')/'outputs/benchmark/results.jsonl'
OUT=ROOT/'outputs/benchmark'


def main():
    from vllm import LLM, SamplingParams
    model=checkpoint(ROOT/'outputs/benchmark_model')
    assert (model/'MERGE_COMPLETE.json').exists()
    source=[json.loads(line) for line in SOURCE.open()]
    source.sort(key=lambda r:r['idx'])
    assert [r['idx'] for r in source]==list(range(5000))
    OUT.mkdir(parents=True,exist_ok=True)
    sampling: dict[str, Any]=dict(temperature=.6,top_p=.95,top_k=20,seed=42,max_tokens=16384,n=1)
    identity={'model':str(model),'source_prompts':str(SOURCE),'source_sha256':sha256(SOURCE),
              'merge':json.loads((model/'MERGE_COMPLETE.json').read_text()),'sampling':sampling,
              'max_model_len':16384,'tensor_parallel_size':2,'rows':5000}
    if (OUT/'identity.json').exists():
        assert json.loads((OUT/'identity.json').read_text())==identity
    write_json(OUT/'identity.json',identity)
    results=OUT/'results.jsonl'
    done=set()
    if results.exists():
        with results.open() as handle:
            for line in handle:
                assert line.endswith('\n'), 'Inspect interrupted final record before resuming'
                r=json.loads(line)
                assert r['idx'] not in done
                done.add(r['idx'])
    assert done <= set(range(5000))
    pending=[r for r in source if r['idx'] not in done]
    if pending:
        llm=LLM(model=str(model),tensor_parallel_size=2,dtype='bfloat16',max_model_len=16384)
        params=SamplingParams(**sampling)
        with results.open('a') as handle:
            for start in range(0,len(pending),64):
                chunk=pending[start:start+64]
                messages: list[Any]=[[{'role':'system','content':r['system_prompt']},
                                      {'role':'user','content':r['task_prompt']}]
                         for r in chunk]
                outputs=llm.chat(messages,sampling_params=params)
                assert len(outputs)==len(chunk)
                for row,result in zip(chunk,outputs, strict=True):
                    c=result.outputs[0]
                    handle.write(json.dumps({**row,'response':c.text,'finish_reason':c.finish_reason,'output_tokens':len(c.token_ids)})+'\n')
                    done.add(row['idx'])
                handle.flush()
                os.fsync(handle.fileno())
                write_json(OUT/'progress.json',{'completed':len(done),'total':5000})
                print(f'Completed {len(done)}/5000',flush=True)
    assert done==set(range(5000))
    write_json(OUT/'COMPLETE.json',{'rows':5000,'results_sha256':sha256(results),'identity':identity})


if __name__ == '__main__':
    main()
