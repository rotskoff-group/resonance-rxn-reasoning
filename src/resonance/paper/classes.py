"""Per-class correct counts for every model in the paper (appendix table "Reaction classes in USPTO-50k").

Ported from run_dir('paper_figures')/scripts/class11.py. The 11 classes, their
abbreviations and colours are shared by Figs 3 and 7; the overall counts are the
headline accuracies of Fig 2, Fig 7 and appendix table "Top-1 accuracy by class".

Every model is scored with `resonance.evaluation.extractors` (the lenient scorer: the last
<answer> block anywhere in the response). Test labels carry the benzyl-ether correction
(`resonance.paper.labels`); training labels get the same fix
(`resonance.pipelines.labels.relabel_train`, written to run_dir('paper_figures')/data).

    python -m resonance.paper.classes --out <dir>     # writes <dir>/class11_counts.json
"""
import argparse
import collections
import json
from pathlib import Path

from resonance.paths import DATA_ROOT, run_dir

# key, abbreviation (every figure), full name (captions), colour.
CLASSES = [
    ("heteroatom alkylation/arylation", "Alk./aryl.", "Heteroatom alkylation/arylation", "#1f5fa8"),
    ("acylation", "Acyl.", "Acylation", "#f28e2b"),
    ("deprotections", "Deprot.", "Deprotections", "#7b3fa0"),
    ("C-C bond formation", "C–C", "C–C bond formation", "#e377c2"),
    ("reductions", "Red.", "Reductions", "#8a8a1e"),
    ("unclassified", "Unclass.", "Unclassified", "#7f7f7f"),
    ("functional group interconversion", "FGI", "FG interconversion", "#a0522d"),
    ("protections", "Prot.", "Protections", "#c9a100"),
    ("heterocycle formation", "Heterocyc.", "Heterocycle formation", "#2a9d8f"),
    ("oxidations", "Ox.", "Oxidations", "#d62728"),
    ("functional group addition", "FGA", "FG addition", "#17becf"),
]
# corrected_test_classes() names -> CLASSES keys
TEST_NAME = {"0": "unclassified"}
TRAIN_NAME = {"heteroatom_alkylation_arylation": "heteroatom alkylation/arylation",
              "cc_bond_formation": "C-C bond formation", "fgi": "functional group interconversion",
              "fga": "functional group addition", "heterocycle_formation": "heterocycle formation"}

RUN_27B = "outputs/no-id-b97ef67b9ba65863"
# model -> results.jsonl (idx, response, answer)
RESULTS = {
    "8b_base": ("eval_rigor", "outputs/bench/bench_base_rules/results.jsonl"),
    "8b_s1": ("s1_8b", "outputs/benchmark/results.jsonl"),
    "8b_s2": ("s2_8b_bench", "outputs/results.jsonl"),
    "8b_s3": ("s3_8b", "outputs/benchmark/results.jsonl"),
    "kimi": ("teacher_bench", "outputs/bench_kimik3_modal/results.jsonl"),
    "opus": ("opus_bench", "outputs/claude-opus-5/forward_smiles/results_merged.jsonl"),
    "gpt6": ("gpt6_bench", "outputs/gpt-6-astra/forward_smiles/results.jsonl"),
}
# model -> 27B rollouts.jsonl globs (row_idx, completions[0].text, ground_truth)
ROLLOUTS = {
    "27b_base": ("pipeline_27b", f"{RUN_27B}/base/eval/rollouts.jsonl"),
    "27b_s1": ("pipeline_27b", f"{RUN_27B}/s1/eval/rollouts.jsonl"),
    "27b_s2": ("pipeline_27b", f"{RUN_27B}/s2-v5-difficulty-mix-1ep/eval/shards/*/rollouts.jsonl"),
    "27b_s3": ("pipeline_27b", f"{RUN_27B}/s3-from-s2-v5-difficulty-mix-1ep/eval/shards/*/rollouts.jsonl"),
    "27b_skip_s2": ("ablation_27b", "outputs/skip_s2/local_eval/generation/rollouts.jsonl"),
}
TEST_LABELS = ("eval_rigor", "outputs/audit/test_reaction_classes.json")
TRAIN_LABELS = ("paper_figures", "data/train_class_relabel.json")   # relabel_train output


def path(key, rel):
    return run_dir(key) / rel


def score_results(extract, match, file):
    correct = {}
    for line in open(file):
        r = json.loads(line)
        pred = extract(r["response"] or "")
        correct[int(r["idx"])] = bool(pred and match(pred, r["answer"], r.get("answer_type", "smiles")))
    return correct


def score_rollouts(extract, match, files):
    correct = {}
    for file in files:
        for line in open(file):
            r = json.loads(line)
            pred = extract(r["completions"][0]["text"])
            correct[r["row_idx"]] = bool(pred and match(pred, r["ground_truth"], "smiles"))
    return correct


def main():
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from rdkit import rdBase
    rdBase.DisableLog("rdApp.*")
    from resonance.evaluation.extractors import TASK_PARSERS
    from resonance.paper.labels import RELABEL, corrected_test_classes
    extract, match = TASK_PARSERS["forward_rxn"]

    test = [TEST_NAME.get(c, c) for c in
            corrected_test_classes(json.loads(path(*TEST_LABELS).read_text())["classes"])]
    train = [TRAIN_NAME.get(c, c) for c in json.loads(path(*TRAIN_LABELS).read_text())["classes"]]
    keys = [c[0] for c in CLASSES]
    assert set(test) == set(train) == set(keys), (set(test) ^ set(keys), set(train) ^ set(keys))

    correct = {m: score_results(extract, match, path(*src)) for m, src in RESULTS.items()}
    for m, (key, pattern) in ROLLOUTS.items():
        files = sorted(run_dir(key).glob(pattern))
        assert files, f"{m}: nothing matches {run_dir(key) / pattern}"
        correct[m] = score_rollouts(extract, match, files)
    for m, c in correct.items():
        assert set(c) == set(range(5000)), m

    n_test, n_train = collections.Counter(test), collections.Counter(train)
    rel = lambda key, p: str((run_dir(key) / p).relative_to(DATA_ROOT))  # noqa: E731
    out = {"classes": [{"name": k, "short": s, "long": lng, "color": col,
                        "n_test": n_test[k], "n_train": n_train[k],
                        "correct": {m: sum(c[i] for i in range(5000) if test[i] == k)
                                    for m, c in correct.items()}}
                       for k, s, lng, col in CLASSES],
           "overall": {"n": 5000, "correct": {m: sum(c.values()) for m, c in correct.items()}},
           "scorer": "resonance.evaluation.extractors (lenient)",
           "source": {"test_labels": [rel(*TEST_LABELS), RELABEL], "train_labels": rel(*TRAIN_LABELS),
                      **{m: rel(*src) for m, src in RESULTS.items()},
                      **{m: [rel(*src)] for m, src in ROLLOUTS.items()}}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "class11_counts.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for m, v in out["overall"]["correct"].items():
        print(f"{m:12s} {v:5d} {v / 50:.2f}")


if __name__ == "__main__":
    main()
