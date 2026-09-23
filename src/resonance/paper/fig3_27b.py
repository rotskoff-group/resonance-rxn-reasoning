"""Refresh 27B class scores from completed evaluations; missing stages stay null.

Ported from run_dir('notebooks')/scripts/build_fig3_27b_breakdown.py. Reads the 8B
breakdown written by `fig3_8b` (family definitions, teacher counts, label paths).

    python -m resonance.paper.fig3_27b --inputs <dir>   # reads fig3_kimi_breakdown.json, writes fig3_27b_breakdown.json
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

from resonance.paper.labels import RELABEL, corrected_test_classes
from resonance.paths import DATA_ROOT, run_dir
from resonance.stages.common import canonical, prediction

ROOT = DATA_ROOT
PIPELINE = run_dir('pipeline_27b')

STAGES = ('base', 's1', 's2', 's3', 'skip_s2', 'base_s3')
PAIRS = (('base', 's1'), ('s1', 's2'), ('s2', 's3'), ('s1', 's3'), ('s1', 'skip_s2'),
         ('skip_s2', 's3'), ('base', 'base_s3'))


def build(inputs):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    previous = json.loads((Path(inputs)/'fig3_kimi_breakdown.json').read_text())
    families = previous['families']
    provenance = previous['provenance']
    test_path = ROOT/provenance['test_labels']
    train_path = ROOT/provenance['training_labels']
    test_classes = corrected_test_classes(json.loads(test_path.read_text())['classes'])
    train_classes = json.loads(train_path.read_text())['classes']
    test = json.loads((run_dir('uspto')/'test.json').read_text())
    assert len(test_classes) == len(test) == 5000
    gold = {i:canonical(r['original_smiles']['End Product']) for i,r in enumerate(test)}
    assert all(gold.values())

    teacher = provenance['runs']['teacher']
    teacher_raw = (ROOT/teacher['path']).read_bytes()
    assert hashlib.sha256(teacher_raw).hexdigest() == teacher['sha256']
    teacher_rows = [json.loads(line) for line in teacher_raw.splitlines() if line.strip()]
    assert len(teacher_rows) == 5000
    assert {int(r['idx']):canonical(r['answer']) for r in teacher_rows} == gold

    corpus_path = run_dir('traces')/'outputs/traces_stage1a_no_reaction_id.jsonl'
    corpus_hash = hashlib.sha256()
    train_indices = set()
    train_counts = collections.Counter()
    train_map = {'heteroatom_alkylation_arylation':'heteroatom alkylation/arylation',
                 'cc_bond_formation':'C-C bond formation',
                 'fgi':'functional group interconversion', 'fga':'functional group addition',
                 'heterocycle_formation':'heterocycle formation'}
    with corpus_path.open('rb') as f:
        for line in f:
            corpus_hash.update(line)
            row = json.loads(line)
            i = row['row_idx']
            assert i not in train_indices
            train_indices.add(i)
            label = train_classes[i]
            train_counts[train_map.get(label, label)] += 1
    assert train_indices == set(range(40000))
    run_id = 'no-id-' + corpus_hash.hexdigest()[:16]
    scored, finished, runs, identities = {}, {}, {}, []
    variants = {'base': 'base', 's1': 's1',
                's2': 's2-v5-difficulty-mix-1ep',
                's3': 's3-from-s2-v5-difficulty-mix-1ep',
                'skip_s2': 'skip_s2/s3', 'base_s3': None}
    for stage in STAGES:
        # Base -> S3 is a separate experiment, not the skip-S1 Base -> S2 -> S3 arm.
        # No completed Base -> S3 evaluation has been registered.
        if variants[stage] is None:
            continue
        eval_root = (run_dir('ablation_27b')/'outputs/skip_s2/local_eval/generation'
                     if stage == 'skip_s2' else PIPELINE/'outputs'/run_id/variants[stage]/'eval')
        summary_path = eval_root/'COMPLETE.json'
        if stage == 'skip_s2':
            summary_path = eval_root.parent/'report.json'
        if stage in ('s2', 's3'):
            summary_path = eval_root/'report.json'
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        if stage == 'skip_s2':
            assert summary['arm'] == 'skip_s2' and summary['rows'] == 5000
            assert summary['training']['stage'] == 's3'
        if stage in ('s2', 's3'):
            assert summary['variant'] == variants[stage] and summary['rows'] == 5000
            parts = [eval_root/'shards'/str(i) for i in range(summary['shards'])]
        else:
            parts = [eval_root]
        raw_hash = hashlib.sha256()
        correct, done = {}, {}
        part_sources = []
        for part in parts:
            complete_path = part/'COMPLETE.json'
            complete = json.loads(complete_path.read_text())
            identity = complete['identity']
            assert identity['limit'] == 0 and identity['n'] == 1
            identities.append(identity)
            if stage in ('s2', 's3'):
                assert identity['data_sha256'] == summary['test_sha256']
                assert identity['prompts_sha256'] == summary['prompts_sha256']
                assert identity['model'] == summary['model']
            part_correct = {}
            part_hash = hashlib.sha256()
            with (part/'rollouts.jsonl').open('rb') as f:
                for line in f:
                    raw_hash.update(line)
                    part_hash.update(line)
                    row = json.loads(line)
                    i = row['row_idx']
                    assert i not in correct and row['ground_truth'] == gold[i]
                    assert len(row['completions']) == 1
                    c = row['completions'][0]
                    pred = prediction(c['text'])
                    hit = pred is not None and pred == gold[i]
                    assert hit == c['correct']
                    correct[i] = part_correct[i] = hit
                    done[i] = c['finish_reason'] == 'stop'
            assert len(part_correct) == complete['rows'] == complete['completions']
            assert math.isclose(sum(part_correct.values())/len(part_correct), complete['accuracy'], abs_tol=1e-12)
            part_sources.append({'path':str((part/'rollouts.jsonl').relative_to(ROOT)),
                                 'sha256':part_hash.hexdigest(), 'evaluation':identity})
        assert set(correct) == set(range(5000))
        assert math.isclose(sum(correct.values())/5000, summary['accuracy'], abs_tol=1e-12)
        scored[stage] = correct
        finished[stage] = done
        runs[stage] = {'correct':sum(correct.values()), 'n':5000,
                       'variant':variants[stage], 'parts':part_sources,
                       'sha256':raw_hash.hexdigest(), 'evaluation':identity}
    for key in ('data_sha256', 'prompts_sha256', 'temperature', 'max_tokens',
                'max_model_len', 'seed', 'top_p', 'top_k', 'thinking', 'n', 'limit'):
        assert len({identity[key] for identity in identities}) <= 1, f'Evaluation mismatch: {key}'

    output_families, classified = [], []
    for family in families:
        indices = [i for i,label in enumerate(test_classes) if label in family['members']]
        assert len(indices) == family['n_test']
        classified.extend(indices)
        counts = {'teacher':family['correct']['teacher']}
        counts.update({stage:sum(scored[stage][i] for i in indices) if stage in scored else None
                       for stage in STAGES})
        # Paired outcomes for McNemar tests: [a correct only, b correct only].
        pairs = {f'{a}->{b}':[sum(scored[a][i] and not scored[b][i] for i in indices),
                              sum(scored[b][i] and not scored[a][i] for i in indices)]
                 for a, b in PAIRS if a in scored and b in scored}
        output_families.append({'name':family['name'], 'members':family['members'],
                               'n_test':len(indices),
                               'n_train':sum(train_counts[m] for m in family['members']),
                               'correct':counts,
                               'finished':{s:sum(finished[s][i] for i in indices) for s in scored},
                               'correct_finished':{s:sum(scored[s][i] and finished[s][i] for i in indices)
                                                   for s in scored},
                               'pairs':pairs})
    assert len(classified) == len(set(classified))
    overall = {'teacher':previous['overall']['correct']['teacher']}
    overall.update({stage:sum(scored[stage][i] for i in classified) if stage in scored else None
                    for stage in STAGES})
    result = {'model':'Qwen3.8-27B', 'run_id':run_id, 'available_stages':list(scored),
              'pending_stages':[s for s in STAGES if s not in scored],
              'n_train':40000, 'n_train_unclassified':40000-sum(f['n_train'] for f in output_families),
              'n_test':5000, 'n_test_unclassified':5000-len(classified), 'test_label_correction':RELABEL,
              'families':output_families, 'overall':{'n':len(classified), 'correct':overall},
              'provenance':{'runs':{'teacher':teacher, **runs},
                            'training_corpus':str(corpus_path.relative_to(ROOT)),
                            'training_corpus_sha256':corpus_hash.hexdigest(),
                            'training_labels':str(train_path.relative_to(ROOT)),
                            'test_labels':str(test_path.relative_to(ROOT)),
                            'note':'S2 uses the one-epoch difficulty-balanced variant; S3 must descend from that variant. '
                                   'skip_s2 is the completed S1 -> S3 ablation; base_s3 has no registered result. '
                                   'Only full completed evaluations are included; missing stages are null. '
                                   'Student stages share evaluation settings. Teacher settings differ. '
                                   'Training panel describes the S1 corpus, not S2/S3 resampling.'}}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--inputs', type=Path, required=True)
    args = ap.parse_args()
    result = build(args.inputs)
    out = args.inputs/'fig3_27b_breakdown.json'
    tmp = out.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result,indent=2)+'\n')
    tmp.replace(out)
    print('Completed:', result['available_stages'], 'Pending:', result['pending_stages'])


if __name__ == '__main__':
    main()
