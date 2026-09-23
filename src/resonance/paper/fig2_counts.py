"""Score the base and teacher benchmark results behind Figure 2.

Ported from run_dir('notebooks')/scripts/build_fig2_staged_counts.py. The original also
scored earlier (with-ID and Sonnet-teacher) lineages; only `base` and `kimi_teacher`
reach a paper figure, so only those are kept.

    python -m resonance.paper.fig2_counts --out <dir>    # writes <dir>/fig2_staged_counts.json
"""
import argparse
import hashlib
import json
from pathlib import Path

from rdkit import RDLogger

from resonance.evaluation.extractors import TASK_PARSERS
from resonance.paths import DATA_ROOT, run_dir

RDLogger.DisableLog('rdApp.*')
extract, match = TASK_PARSERS['forward_rxn']
PATHS = {
    'base': run_dir('eval_rigor') / 'outputs/bench/bench_base_rules/results.jsonl',
    'kimi_teacher': run_dir('teacher_bench') / 'outputs/bench_kimik3_modal/results.jsonl',
}


def build():
    result = {}
    for name, path in PATHS.items():
        rel = str(path.relative_to(DATA_ROOT))
        raw = path.read_bytes(); rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
        assert len(rows) == 5000 and len({r['idx'] for r in rows}) == 5000
        correct = 0
        for r in rows:
            pred = extract(r.get('response', '') or '')
            try: ok = bool(pred and match(pred, r.get('answer'), 'smiles'))
            except Exception: ok = False
            correct += ok
        result[name] = {'correct': correct, 'n': len(rows), 'path': rel,
                        'sha256': hashlib.sha256(raw).hexdigest(),
                        'system_prompts': list({r.get('system_prompt', '') for r in rows})}
        print(name, correct, len(rows), 100*correct/len(rows), flush=True)
    return {'scorer': 'resonance.evaluation.extractors.TASK_PARSERS[forward_rxn]', 'runs': result}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'fig2_staged_counts.json').write_text(json.dumps(build(), indent=2) + '\n')


if __name__ == '__main__':
    main()
