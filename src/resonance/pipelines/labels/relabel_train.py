#!/usr/bin/env python3
"""Benzyl-ether-corrected classes for the 40,000 USPTO-50k training reactions.

Same classifier and fix as the test labels (`relabel_classes`): benzyl_ether
"[OX2]Cc1ccccc1" -> "[OX2;H0]Cc1ccccc1". The original classifier must first reproduce
the published training labels (run_dir('train_classes')/outputs/train_reaction_classes.json,
written by `classify_train`); 2,247 of 40,000 labels then change.

Ported from run_dir('paper_figures')/scripts/relabel_train.py. About 20 minutes per pass
on 16 processes (rdFMCS); `resonance.paper.run_all` skips it when the output exists.

CLI: python -m resonance.pipelines.labels.relabel_train [--out O] [--limit N] [--procs P]
"""
import argparse
import collections
import json
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from resonance.paths import DATA_ROOT, run_dir

FIXED_BENZYL = "[OX2;H0]Cc1ccccc1"
OUT = run_dir("paper_figures") / "data/train_class_relabel.json"
C: Any = None   # the classifier module, imported by init() in each worker


def init(fixed):
    global C
    from rdkit import Chem, rdBase
    rdBase.DisableLog("rdApp.*")
    from resonance.pipelines.labels import classify_reactions
    C = classify_reactions
    if fixed:
        C.FG["benzyl_ether"] = FIXED_BENZYL
        C._FGM["benzyl_ether"] = Chem.MolFromSmarts(FIXED_BENZYL)


def label(o):
    return C.classify(o.get("Starting Material"), o.get("Reagents"), o.get("End Product"))[1]


def run(rows, fixed, procs):
    # A fresh pool per pass, so the unpatched and patched classifiers never share a worker.
    with Pool(procs, initializer=init, initargs=(fixed,)) as pool:
        return pool.map(label, [r["original_smiles"] for r in rows], chunksize=200)


def main():
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()
    train = run_dir("uspto") / "train.json"
    published_path = run_dir("train_classes") / "outputs/train_reaction_classes.json"
    rows = json.loads(train.read_text())
    published = json.loads(published_path.read_text())["classes"]
    if args.limit:
        rows, published = rows[:args.limit], published[:args.limit]
    orig = run(rows, False, args.procs)
    agree = sum(a == b for a, b in zip(orig, published, strict=True))
    print(f"reproduction: {agree}/{len(rows)} original-classifier labels match the published train labels")
    assert agree == len(rows), "classifier drift: fix before relabelling"
    fixed = run(rows, True, args.procs)
    moves = collections.Counter((a, b) for a, b in zip(orig, fixed, strict=True) if a != b)
    print(f"benzyl_ether H0 fix changes {sum(moves.values())} of {len(rows)} labels")
    for (a, b), n in moves.most_common():
        print(f"  {a} -> {b}: {n}")
    counts = collections.Counter(fixed)
    print({k: counts[k] for k in sorted(counts, key=lambda k: -counts[k])})
    if not args.limit:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "fix": {"benzyl_ether": FIXED_BENZYL}, "n": len(rows),
            "source": str(train.relative_to(DATA_ROOT)),
            "published": str(published_path.relative_to(DATA_ROOT)),
            "n_changed": sum(moves.values()),
            "moves": {f"{a}->{b}": n for (a, b), n in moves.items()},
            "counts": dict(counts), "classes": fixed}, indent=1) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
