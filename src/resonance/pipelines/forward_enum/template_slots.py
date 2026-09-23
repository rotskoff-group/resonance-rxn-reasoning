#!/usr/bin/env python3
"""Partition the 110 templates by how many reactant slots their forward SMARTS has.

Two branches need different treatment:
  1 slot  -> apply_chunk_forward.py applies the template to each molecule directly.
  2 slots -> screen the pool for molecules matching each slot, then pair them
             (apply_chunk_screen -> collect_screen -> apply_pairs). Pairing beats
             random within-chunk pairs because every pair is guaranteed to match.
  3+ slots / unparseable slot SMARTS -> not reachable by this pipeline; reported so
             the coverage ceiling is explicit rather than silent.

Ported from run_dir('forward_enum')/scripts/template_slots.py.
CLI: python -m resonance.pipelines.forward_enum.template_slots [--templates T] [--out OUT]
"""
import argparse, json
from resonance.paths import run_dir
from resonance.pipelines.forward_enum.audit_ground_truth import TEMPLATES
from resonance.pipelines.synthetic_data.apply_chunk_screen import _split_lhs
from rdkit import Chem, rdBase
rdBase.DisableLog("rdApp.*")

if __name__ == '__main__':
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--templates", default=str(TEMPLATES))
    _ap.add_argument("--out", default=str(run_dir("forward_enum") / "outputs/template_ids.json"))
    _a = _ap.parse_args()
    OUT = _a.out
    T = json.load(open(_a.templates))

    one, two, unreachable = [], [], []
    for t in T:
        parts = _split_lhs(t["smarts"])
        if not all(Chem.MolFromSmarts(p) is not None for p in parts):
            unreachable.append({"id": t["id"], "name": t["name"], "why": "slot SMARTS does not parse"})
        elif len(parts) == 1:
            one.append(t["id"])
        elif len(parts) == 2:
            two.append(t["id"])
        else:
            unreachable.append({"id": t["id"], "name": t["name"], "why": f"{len(parts)} reactant slots"})

    json.dump({"one_slot": one, "two_slot": two, "unreachable": unreachable},
              open(OUT, "w"), indent=2)
    print(f"1-slot: {len(one)}  2-slot: {len(two)}  unreachable: {len(unreachable)}")
    print(f"coverage ceiling: {len(one)+len(two)} / {len(T)} templates")
    for u in unreachable:
        print(f"  [{u['id']:>3}] {u['name']:<38} {u['why']}")
    print(f"wrote {OUT}")
