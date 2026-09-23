"""Compare the existing S2 and S3 benchmark outputs, retaining sampling caveats.

Ported from run_dir('s3_8b')/scripts/compare_s2_s3.py (run by hand).

    python -m resonance.stages.qwen3_8b.compare_s2_s3 [--out FILE]
"""
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path
from resonance.evaluation.extractors import extract_forward_rxn,match_forward_rxn
from resonance.paths import run_dir
from resonance.stages.qwen3_8b._common import USPTO_TEST_LABELS
ROOT=run_dir('s3_8b')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default=str(ROOT/'outputs/benchmark/s2_s3_comparison.json'))
    out_path=Path(ap.parse_args().out)
    def read(p):
        rows=[json.loads(line) for line in p.open()]
        assert len(rows)==5000 and {r['idx'] for r in rows}==set(range(5000))
        return {r['idx']:r for r in rows}
    def correct(r):
        pred=extract_forward_rxn(r['response'])
        return pred is not None and match_forward_rxn(pred,r['answer'],r.get('answer_type'))
    s2=read(run_dir('s2_8b_bench')/'outputs/results.jsonl')
    s3=read(ROOT/'outputs/benchmark/results.jsonl')
    labels=json.loads(USPTO_TEST_LABELS.read_text())['labels']
    counts=defaultdict(Counter);transitions=Counter()
    for idx,a in s2.items():
        b=s3[idx]
        for key in ('system_prompt','task_prompt','answer','answer_type'):
            assert a.get(key)==b.get(key),(idx,key)
        ac,bc=correct(a),correct(b)
        c=counts[labels[str(idx)]['level1']]
        c['rows']+=1;c['s2_correct']+=ac;c['s3_correct']+=bc
        transitions['retained' if ac and bc else 'lost' if ac else 'gained' if bc else 'both_wrong']+=1
    ca=sum(c['s2_correct'] for c in counts.values());cb=sum(c['s3_correct'] for c in counts.values())
    assert ca==1946 and cb==2140
    report={'rows':5000,'s2_correct':ca,'s3_correct':cb,'s2_accuracy':ca/5000,'s3_accuracy':cb/5000,
            'delta_pp':100*(cb-ca)/5000,'transitions':dict(transitions),
            's2_success_retention':transitions['retained']/ca,
            'comparison_note':'All saved prompts/reference answers match; nominal temperature/top-p/top-k/context match. S2 did not specify a sampling seed; S3 specified 42 and generated in batches of 64. This is an observed single-run difference, not a seed-matched causal estimate.',
            'classes':[{'class':cls,**dict(c),'s2_accuracy':c['s2_correct']/c['rows'],'s3_accuracy':c['s3_correct']/c['rows'],'delta_pp':100*(c['s3_correct']-c['s2_correct'])/c['rows']} for cls,c in sorted(counts.items())]}
    out_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='classes'},indent=2))


if __name__=='__main__':main()
