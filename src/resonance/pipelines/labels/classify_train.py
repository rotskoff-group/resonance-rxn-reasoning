#!/usr/bin/env python3
"""Cache the 10-class label for every record in uspto50k train.json (the stage-1 pool).

rdFMCS makes this ~20 minutes for 40k, so it is cached once. Labels are positional:
classes[i] labels train.json row i.

Ported from run_dir('train_classes')/scripts/classify_train.py.
CLI: python -m resonance.pipelines.labels.classify_train [--train-json T] [--out O]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from resonance.paths import run_dir
from resonance.pipelines.labels import classify_reactions as clf

OUT = run_dir("train_classes") / "outputs" / "train_reaction_classes.json"


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", type=Path, default=run_dir("uspto") / "train.json")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    OUT = a.out

    recs = json.load(open(a.train_json))
    labels = []
    for i, r in enumerate(recs):
        o = r["original_smiles"]
        try:
            lab = clf.classify(o.get("Starting Material", ""), o.get("Reagents", "") or "",
                               o.get("End Product", ""))
            lab = lab[1] if isinstance(lab, (tuple, list)) else lab
        except Exception:                                              # noqa: BLE001
            lab = "unclassified"
        labels.append(lab or "unclassified")
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(recs)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": len(labels), "classes": labels,
               "source": "uspto50k_json_files/train.json"}, open(OUT, "w"))
    print("\nstage-1 pool class distribution:")
    for c, n in Counter(labels).most_common():
        print(f"  {c:<34}{n:>7}  {100*n/len(labels):>6.2f}%")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
