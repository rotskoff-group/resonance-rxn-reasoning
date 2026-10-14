"""Score complete 8B DPO results and compare matched S1 predictions by class.

Ported from run_dir('s3_8b')/scripts/evaluate_benchmark.py.

    python -m resonance.stages.qwen3_8b.evaluate_benchmark [--bench-dir DIR]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from resonance.evaluation.extractors import extract_forward_rxn, match_forward_rxn
from resonance.paths import run_dir
from resonance.stages.qwen3_8b._common import USPTO_TEST_LABELS
from resonance.workflows.evaluate import evaluate

ROOT=run_dir('s3_8b')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bench-dir',default=str(ROOT/'outputs/benchmark'))
    out=Path(ap.parse_args().bench_dir)
    assert (out/'COMPLETE.json').exists()
    rows=[json.loads(line) for line in (out/'results.jsonl').open()]
    assert len(rows)==5000 and {r['idx'] for r in rows}==set(range(5000))
    s1_results=run_dir('s1_8b')/'outputs/benchmark/results.jsonl'
    source={r['idx']:r for r in (json.loads(line) for line in s1_results.open())}
    labels=json.loads(USPTO_TEST_LABELS.read_text())['labels']
    counts=defaultdict(Counter)
    transitions=Counter()
    def correct(r):
        p=extract_forward_rxn(r['response'])
        return p is not None and match_forward_rxn(p,r['answer'],r.get('answer_type'))
    for r in rows:
        old=source[r['idx']]
        for key in ('system_prompt','task_prompt','answer','answer_type','question_type'):
            assert r.get(key)==old.get(key), (r['idx'],key)
        assert r['question_type']=='forward_rxn'
        a,b=correct(old),correct(r)
        cls=labels[str(r['idx'])]['level1']
        c=counts[cls]
        c['rows']+=1
        c['s1_correct']+=a
        c['s3_correct']+=b
        c['length_limited']+=r['finish_reason']=='length'
        transitions['retained' if a and b else 'lost' if a else 'gained' if b else 'both_wrong']+=1
    report=evaluate(str(out/'results.jsonl'))
    assert report['total']==5000 and report['skipped']==0
    assert report['correct']==sum(c['s3_correct'] for c in counts.values())
    report['classes']=[{'class':cls,**dict(c),'s1_accuracy':c['s1_correct']/c['rows'],
                        's3_accuracy':c['s3_correct']/c['rows'],'delta_pp':100*(c['s3_correct']-c['s1_correct'])/c['rows']}
                       for cls,c in sorted(counts.items())]
    report['s1_accuracy']=sum(c['s1_correct'] for c in counts.values())/5000
    report['s1_to_s3_transitions']=dict(transitions)
    (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
