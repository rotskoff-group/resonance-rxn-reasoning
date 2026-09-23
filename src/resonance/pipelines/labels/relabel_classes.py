#!/usr/bin/env python3
"""Re-derive the eval set's reaction_class with the classifier defects fixed.

Two defects in classify_reactions.py (this package) produced the
`protections` trend that later failed its audit:

  1. benzyl_ether is "[OX2]Cc1ccccc1". X is TOTAL connectivity including hydrogen, so an
     -OH oxygen is also X2 and the pattern matches benzyl ALCOHOL, not just benzyl ethers.
     Any reduction that yields a benzyl alcohol therefore GAINS a protecting group
     (pg_delta +1) and is labelled `protections`. Fixed by requiring H0 on the oxygen.
  2. The protections/deprotections branch is tested before reductions, so whenever a
     reduction incidentally moves a PG count it is captured by the earlier branch. Left
     as-is and reported: with defect 1 fixed, the ordering question is measurable rather
     than assumed.

Writes both labels per item so any figure can show as-published vs corrected, and nothing
overwrites the published eval file.

Ported from run_dir('relabel')/scripts/relabel_classes.py (byte-identical to
resonance/analysis/metareasoning/frozen/relabel_classes.py apart from paths/CLI). The
classifier module is imported fresh and its FG tables patched in place, as before.

CLI: python -m resonance.pipelines.labels.relabel_classes [--eval E] [--out O]
"""
import argparse
import collections
import json
from pathlib import Path

from resonance.paths import run_dir

RUN = run_dir("relabel")
EVAL = RUN / "data/eval_5000_t06.json"
OUT = RUN / "data/class_relabel_v2.json"

FIXED_BENZYL = "[OX2;H0]Cc1ccccc1"   # a true ether oxygen carries no hydrogen


def main():
    global EVAL, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=EVAL)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    EVAL, OUT = a.eval, a.out
    from resonance.pipelines.labels import classify_reactions as C
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    ev = json.load(open(EVAL))
    rows = {}
    published = collections.Counter()
    for r in ev:
        published[r.get("reaction_class")] += 1

    # as-published labels, recomputed with the ORIGINAL classifier, to confirm this
    # reproduction matches the file before trusting any diff
    def run(label):
        out = {}
        for r in ev:
            o = r["original_smiles"]
            _, name = C.classify(o.get("Starting Material"), o.get("Reagents"),
                                 o.get("End Product"))
            out[r["eval_idx"]] = name
        return out

    orig = run("original")
    agree = sum(1 for r in ev if orig[r["eval_idx"]] == r.get("reaction_class"))
    print(f"reproduction check: {agree}/{len(ev)} recomputed labels match the published file")
    if agree < 0.98 * len(ev):
        print("  WARNING: the classifier no longer reproduces the published labels; "
              "the diff below mixes the SMARTS fix with that drift")

    # now with the benzyl_ether pattern fixed
    C.FG["benzyl_ether"] = FIXED_BENZYL
    C._FGM["benzyl_ether"] = Chem.MolFromSmarts(FIXED_BENZYL)
    fixed = run("fixed")

    changed = [r for r in ev if fixed[r["eval_idx"]] != orig[r["eval_idx"]]]
    print(f"\nbenzyl_ether H0 fix changes {len(changed)} of {len(ev)} labels "
          f"({100*len(changed)/len(ev):.2f}%)\n")
    moves = collections.Counter((orig[r["eval_idx"]], fixed[r["eval_idx"]]) for r in changed)
    print(f"{'from':<26}{'to':<34}{'n':>6}")
    for (a, b), n in moves.most_common():
        print(f"{a:<26}{b:<34}{n:>6}")

    after = collections.Counter(fixed.values())
    print(f"\n{'class':<34}{'published':>11}{'corrected':>11}{'delta':>8}")
    for c in sorted(set(published) | set(after), key=lambda c: -published.get(c, 0)):
        p, a = published.get(c, 0), after.get(c, 0)
        print(f"{str(c):<34}{p:>11}{a:>11}{a-p:>+8}")

    for r in ev:
        i = r["eval_idx"]
        rows[str(i)] = {"published": r.get("reaction_class"),
                        "recomputed_original": orig[i], "corrected": fixed[i],
                        "changed": fixed[i] != orig[i]}
    json.dump({"fix": {"benzyl_ether": FIXED_BENZYL},
               "n": len(ev), "n_changed": len(changed),
               "reproduction_agreement": agree,
               "moves": {f"{a}->{b}": n for (a, b), n in moves.items()},
               "labels": rows}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
