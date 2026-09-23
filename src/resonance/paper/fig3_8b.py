"""Rebuild the no-ID 8B Kimi figure's exact counts using resonance's scorer.

Ported from run_dir('notebooks')/scripts/build_fig3_kimi_breakdown.py. Reads the
Figure 2 counts written by `fig2_counts` (for the teacher and base sources).

    python -m resonance.paper.fig3_8b --inputs <dir>   # reads/writes <dir>/fig2_staged_counts.json, fig3_kimi_breakdown.json
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from rdkit import RDLogger

from resonance.evaluation.extractors import TASK_PARSERS
from resonance.paper.labels import RELABEL, corrected_test_classes
from resonance.paths import DATA_ROOT, run_dir

RDLogger.DisableLog('rdApp.*')
ROOT = DATA_ROOT
EXTRACT, MATCH = TASK_PARSERS['forward_rxn']
FAMILIES = [
 ('heteroatom alkylation/arylation', ['heteroatom alkylation/arylation']),
 ('protection chemistry', ['protections', 'deprotections']),
 ('acylation', ['acylation']),
 ('C-C bond formation', ['C-C bond formation']),
 ('redox', ['reductions', 'oxidations']),
 ('FG changes', ['functional group interconversion', 'functional group addition']),
 ('heterocycle formation', ['heterocycle formation']),
]
TRAIN_MAP = {'heteroatom_alkylation_arylation':'heteroatom alkylation/arylation',
             'cc_bond_formation':'C-C bond formation', 'fgi':'functional group interconversion',
             'fga':'functional group addition', 'heterocycle_formation':'heterocycle formation'}


def rel(key, path):
    return str((run_dir(key) / path).relative_to(ROOT))


def build(inputs):
    manifest = json.loads((Path(inputs)/'fig2_staged_counts.json').read_text())['runs']
    s2 = json.loads((run_dir('s2_8b_bench')/'outputs/evaluation.json').read_text())
    s3 = json.loads((run_dir('s3_8b')/'outputs/benchmark/evaluation.json').read_text())
    sources = {'teacher': manifest['kimi_teacher'], 'base': manifest['base']}
    for stage, path, count in [
        ('s1', rel('s1_8b', 'outputs/benchmark/results.jsonl'),
         sum(row['s1_correct'] for row in s3['classes'])),
        ('s2', rel('s2_8b_bench', 'outputs/results.jsonl'), s2['correct']),
        ('s3', rel('s3_8b', 'outputs/benchmark/results.jsonl'), s3['correct']),
    ]:
        sources[stage] = dict(path=path, correct=count, n=5000,
                             sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
    scored, runs, gold = {}, {}, None
    for key, meta in sources.items():
        raw = (ROOT/meta['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == meta['sha256'], 'Figure 2 source changed: regenerate it first'
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        rows = {int(r['idx']):r for r in records}
        assert len(records) == len(rows) == 5000 and set(rows) == set(range(5000))
        answers = {i:r['answer'] for i,r in rows.items()}
        if gold is None: gold = answers
        else: assert answers == gold, f'Gold answer alignment differs: {key}'
        correct = {}
        for i,r in rows.items():
            pred = EXTRACT(r.get('response') or '')
            try: correct[i] = bool(pred and MATCH(pred,r['answer'],'smiles'))
            except Exception: correct[i] = False
        assert sum(correct.values()) == meta['correct']
        scored[key] = correct
        runs[key] = meta
        print(key, sum(correct.values()), flush=True)
    test_path = rel('eval_rigor', 'outputs/audit/test_reaction_classes.json')
    train_path = rel('train_classes', 'outputs/train_reaction_classes.json')
    test = corrected_test_classes(json.loads((ROOT/test_path).read_text())['classes'])
    train = json.loads((ROOT/train_path).read_text())['classes']
    corpus_path = rel('traces', 'outputs/traces_stage1a_no_reaction_id.jsonl')
    corpus_hash = hashlib.sha256()
    train_indices = set()
    train_counts = collections.Counter()
    with (ROOT/corpus_path).open('rb') as stream:
        for line in stream:
            corpus_hash.update(line)
            row = json.loads(line)
            i = row['row_idx']
            assert i not in train_indices
            train_indices.add(i)
            train_counts[TRAIN_MAP.get(train[i], train[i])] += 1
    assert train_indices == set(range(40000))
    fams = []
    for name, members in FAMILIES:
        indices = [i for i,c in enumerate(test) if c in members]
        fams.append(dict(name=name,members=members,n_test=len(indices),
                         n_train=sum(train_counts[c] for c in members),
                         correct={k:sum(v[i] for i in indices) for k,v in scored.items()}))
    kept = [i for i,c in enumerate(test) if c in {m for _,ms in FAMILIES for m in ms}]
    assert len(kept) == sum(f['n_test'] for f in fams)
    assert sum(f['n_train'] for f in fams) == 38333
    data = dict(provenance=dict(runs=runs,training_labels=train_path,test_labels=test_path,test_label_correction=RELABEL,
                training_corpus=corpus_path,training_corpus_sha256=corpus_hash.hexdigest(),
                lineage='Kimi K3 no-ID Qwen3-8B',scorer='resonance.evaluation.extractors.TASK_PARSERS[forward_rxn]',
                note='No-ID S1/S2/S3 benchmarks; historical base reference retained. Teacher API prompt and generation settings differ. Family labels retain the established Figure 3 classification.'),
                n_train=40000,n_train_unclassified=1667,n_test=5000,n_test_unclassified=5000-len(kept),
                families=fams,overall=dict(n=len(kept),correct={k:sum(v[i] for i in kept) for k,v in scored.items()}))
    for f in fams:
        print(f['name'],f['n_test'], {k:round(100*v/f['n_test'],2) for k,v in f['correct'].items()})
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--inputs', type=Path, required=True)
    args = ap.parse_args()
    out = args.inputs/'fig3_kimi_breakdown.json'
    out.write_text(json.dumps(build(args.inputs),indent=2)+'\n')
    print('wrote',out)


if __name__ == '__main__':
    main()
