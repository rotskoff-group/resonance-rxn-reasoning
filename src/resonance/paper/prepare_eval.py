"""Build aligned prompt-only inputs from each lineage's own benchmark artifacts.

Ported from run_dir('metareasoning')/no_id/scripts/prepare_eval.py. Writes
<out>/data/eval_greedy.json (turn-1 records per stage) and <out>/data/input_audit.json.

    python -m resonance.paper.prepare_eval [--config metareasoning_eval.json] [--out <dir>]
"""
import argparse,hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
from resonance.paper import config as paper_config
if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config',default='metareasoning_eval.json',help='path, or a name in configs/paper/figures')
    p.add_argument('--out',type=Path,help="default: the config's run_dir")
    a=p.parse_args()
    c=paper_config.load(a.config); run=a.out or Path(c['run_dir']); (run/'data').mkdir(parents=True,exist_ok=True)
    test=json.load(open(c['test_json'])); labels=json.load(open(c['labels_test']))['labels']
    benches={}
    audit={'name':c['name'],'models':{},'n':len(test)}
    for tag,spec in c['models'].items():
        rows=list(map(json.loads,open(spec['benchmark']))); bench={r['idx']:r for r in rows}
        assert len(bench)==len(rows)==len(test) and set(bench)==set(range(len(test)))
        tok=AutoTokenizer.from_pretrained(spec['path'],local_files_only=True)
        rendered=tok.apply_chat_template([{'role':'user','content':'test'}],tokenize=False,add_generation_prompt=True)
        actual=rendered.rstrip().endswith('<think>')
        assert actual==spec['think_is_prefilled'],(tag,rendered[-100:])
        audit['models'][tag]={'path':spec['path'],'benchmark':spec['benchmark'],'benchmark_sha256':hashlib.sha256(Path(spec['benchmark']).read_bytes()).hexdigest(),'chat_template_sha256':hashlib.sha256(tok.chat_template.encode()).hexdigest(),'think_prefilled':actual}
        benches[tag]=bench
    out=[]
    for i,t in enumerate(test):
        r={'eval_idx':i,'source_idx':i,'probe_idx':None,'reaction_class':labels[str(i)]['corrected'],'reaction_class_published':labels[str(i)]['published'],**t,'turn1':{}}
        for tag,bench in benches.items():
            q=bench[i]
            assert q['answer']==t['original_smiles']['End Product']
            assert t['original_smiles']['Starting Material'] in q['task_prompt']
            r['turn1'][tag]={**q,'prompt_variant':'archived_benchmark_prompt','decode':{'temperature':0.6,'top_p':0.95,'top_k':20}}
        out.append(r)
    (run/'data/eval_greedy.json').write_text(json.dumps(out))
    (run/'data/input_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Validated',len(out),'items for',list(benches))
