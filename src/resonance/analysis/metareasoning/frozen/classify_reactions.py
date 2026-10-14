#!/usr/bin/env python3
"""
Robust disconnection-based reaction classifier for USPTO-50k.

Fixes the three defects in sonnet_s3/scripts/analyze_reaction_errors.py:
  1. Uses the UNION of all provided SMILES (Starting Material + Reagents) as the
     reactant side, NOT the single (and often mislabelled) "Starting material".
  2. No dead O-branch.
  3. Heterocycle formation is gated on a genuine ring-count increase, not merely
     the presence of a heteroaromatic ring (the old rule mislabelled 39%).

Classifies each reaction into the 10 standard USPTO-50k (Schneider) superclasses,
derived from the reaction's own disconnection (reactants -> product). Does NOT use
the 110-template Hartenfeller/Button library (a different reaction set).

Run:
  python classify_reactions.py [--limit N]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS

RDLogger.DisableLog("rdApp.*")

BASE = Path("data")
TEST_JSON = BASE / "uspto50k_json_files/test.json"
RESULTS = BASE / "teacher_bench/outputs/bench_kimik3_modal/results.jsonl"
OUT = BASE / "teacher_bench/outputs/bench_kimik3_modal/reaction_labels.json"

CLASS_NAMES = {
    1: "heteroatom_alkylation_arylation",
    2: "acylation",
    3: "cc_bond_formation",
    4: "heterocycle_formation",
    5: "protections",
    6: "deprotections",
    7: "reductions",
    8: "oxidations",
    9: "fgi",
    10: "fga",
    0: "unclassified",
    -1: "insufficient_reactants",
}

# ── Functional-group SMARTS ──────────────────────────────────────────────────
FG = {
    "amide":          "[NX3][CX3](=[OX1])",
    "ester":          "[#6][OX2][CX3](=[OX1])[#6]",
    "carboxylic_acid":"[CX3](=[OX1])[OX2H1]",
    "acid_halide":    "[CX3](=[OX1])[F,Cl,Br,I]",
    "sulfonamide":    "[SX4](=[OX1])(=[OX1])[NX3]",
    "alcohol":        "[OX2H1][CX4]",
    "phenol":         "[OX2H1][c]",
    "ketone":         "[#6][CX3](=[OX1])[#6]",
    "aldehyde":       "[CX3H1](=[OX1])",
    "nitro":          "[$([N+](=O)[O-]),$(N(=O)=O)]",
    "nitrile":        "[CX2]#[NX1]",
    "azide":          "[$(N=[N+]=[N-]),$([N-]=[N+]=N)]",
    "halide_c":       "[#6][F,Cl,Br,I]",
    "amine_nh":       "[NX3;H1,H2;!$(NC=O);!$(N=*);!$([N+])]",
    "alkene":         "[CX3]=[CX3]",
    "alkyne":         "[CX2]#[CX2]",
    "imine":          "[CX3]=[NX2]",
    # protecting groups
    "boc":            "[CX3](=O)OC(C)(C)C",
    "cbz":            "[CX3](=O)OCc1ccccc1",
    "fmoc":           "O=C(O*)OCC1c2ccccc2-c2ccccc21",
    "benzyl_ether":   "[OX2]Cc1ccccc1",
    "tosyl":          "[SX4](=O)(=O)c1ccc(C)cc1",
    "silyl":          "[Si]",
    "acetyl_o":       "[OX2][CX3](=O)[CH3]",
}
_FGM = {k: Chem.MolFromSmarts(v) for k, v in FG.items()}
PROTECTING = ["boc", "cbz", "fmoc", "benzyl_ether", "tosyl", "silyl", "acetyl_o"]

MCS_P = rdFMCS.MCSParameters()
MCS_P.Timeout = 1
MCS_P.AtomTyper = rdFMCS.AtomCompare.CompareElements
MCS_P.BondTyper = rdFMCS.BondCompare.CompareOrder


def to_mol(smi):
    if not smi or not smi.strip():
        return None
    return Chem.MolFromSmiles(smi.strip())


def combined_reactants(sm, reagents):
    """Union of every non-empty fragment across SM + reagents."""
    parts = []
    for field in (sm, reagents):
        if field and field.strip() and field.strip().upper() not in ("NA", "N/A", "NONE SPECIFIED"):
            parts.extend(p for p in field.strip().split(".") if p)
    frags = [m for m in (to_mol(p) for p in parts) if m is not None]
    combo = to_mol(".".join(parts)) if parts else None
    return combo, frags


def fg_count(mol, key):
    q = _FGM[key]
    if mol is None or q is None:
        return 0
    return len(mol.GetSubstructMatches(q))


def hetero_ring_count(mol):
    """Number of rings containing at least one N/O/S atom."""
    if mol is None:
        return 0
    ri = mol.GetRingInfo()
    n = 0
    for ring in ri.AtomRings():
        if any(mol.GetAtomWithIdx(a).GetAtomicNum() in (7, 8, 16) for a in ring):
            n += 1
    return n


def ring_count(mol):
    return mol.GetRingInfo().NumRings() if mol is not None else 0


def formed_bond_elements(product, frags):
    """
    Assign each product atom to a reactant fragment via per-fragment MCS, then
    return the multiset of element-pairs for bonds whose two atoms come from
    DIFFERENT fragments (the intermolecular bond(s) formed). Empty if none found.
    """
    if product is None or len(frags) < 2:
        return []
    assign = {}
    for fi, frag in enumerate(frags):
        if frag is None or frag.GetNumAtoms() < 2:
            continue
        try:
            mcs = rdFMCS.FindMCS([frag, product], MCS_P)
            if mcs.numAtoms == 0:
                continue
            pat = Chem.MolFromSmarts(mcs.smartsString)
            if pat is None:
                continue
            for a in product.GetSubstructMatch(pat):
                assign.setdefault(a, fi)
        except Exception:
            continue
    pairs = []
    for b in product.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a1 in assign and a2 in assign and assign[a1] != assign[a2]:
            e1 = product.GetAtomWithIdx(a1).GetSymbol()
            e2 = product.GetAtomWithIdx(a2).GetSymbol()
            pairs.append(tuple(sorted((e1, e2))))
    return pairs


def classify(sm, reagents, product_smi):
    combo, frags = combined_reactants(sm, reagents)
    prod = to_mol(product_smi)
    if prod is None:
        return 0, "unclassified"
    if combo is None:
        return -1, "insufficient_reactants"

    def d(key):
        return fg_count(prod, key) - fg_count(combo, key)

    def n_carbons(m):
        return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 6) if m else 0

    # Did two carbon-bearing fragments merge into one skeleton? (Grignard/aldol/
    # coupling do; a unimolecular reduction/oxidation/FGI does not.)
    prod_c = n_carbons(prod)
    max_frag_c = max((n_carbons(f) for f in frags), default=0)
    merged = len(frags) >= 2 and prod_c > max_frag_c + 1

    # 5/6 Protections & deprotections — PG substructure appears / disappears,
    #     OR ester/amide hydrolysis revealing an acid (a "deprotection" in Schneider)
    pg_delta = sum(fg_count(prod, p) - fg_count(combo, p) for p in PROTECTING)
    if pg_delta <= -1:
        return 6, "deprotections"
    if d("carboxylic_acid") > 0 and (d("ester") < 0 or d("amide") < 0):
        return 6, "deprotections"
    if pg_delta >= 1:
        return 5, "protections"

    # 7 Reductions — gated on NO carbon-skeleton merge (else it's a C-C addition
    #   such as Grignard/aldol that also produces an alcohol)
    if not merged and (
       (d("alcohol") > 0 and (d("ketone") < 0 or d("aldehyde") < 0))
       or (d("alcohol") > 0 and (d("ester") < 0 or d("carboxylic_acid") < 0))
       or (d("amine_nh") > 0 and d("nitro") < 0)
       or (d("amine_nh") > 0 and (d("nitrile") < 0 or d("amide") < 0))
       or (d("alkene") < 0 and d("alcohol") == 0 and d("halide_c") == 0 and d("ketone") == 0)):
        return 7, "reductions"

    # 8 Oxidations
    if ((d("ketone") > 0 or d("aldehyde") > 0 or d("carboxylic_acid") > 0) and d("alcohol") < 0):
        return 8, "oxidations"

    # 4 Heterocycle formation — a new heteroatom-containing ring was built
    if hetero_ring_count(prod) > hetero_ring_count(combo) and ring_count(prod) > ring_count(combo):
        return 4, "heterocycle_formation"

    # 2 Acylation — new amide / ester / sulfonamide
    if d("amide") > 0 or d("ester") > 0 or d("sulfonamide") > 0:
        return 2, "acylation"

    # 9 FGI (halogenation) — a C-halide is GAINED (a leaving-group halide would be
    #   consumed, d<0); catches NBS/bromination before the noisy bond logic
    if d("halide_c") > 0 and not merged:
        return 9, "fgi"

    # 3 vs 1 — decide from the atoms flanking the formed inter-fragment bond
    pairs = formed_bond_elements(prod, frags)
    has_cc = any(p == ("C", "C") for p in pairs)
    has_heteroC = any(("C" in p and (("N" in p) or ("O" in p) or ("S" in p))) for p in pairs)
    if has_cc and not has_heteroC:
        return 3, "cc_bond_formation"
    if has_heteroC:
        return 1, "heteroatom_alkylation_arylation"

    # Fallbacks when bond assignment was inconclusive: use FG deltas
    if d("amide") == 0 and d("ester") == 0:
        # heteroatom substitution signalled by consumed N-H/O-H
        if d("amine_nh") < 0 or d("phenol") < 0 or d("alcohol") < 0:
            return 1, "heteroatom_alkylation_arylation"

    # 9 FGI — interconversion of a functional group (halide/nitrile/nitro/azide)
    if abs(d("halide_c")) > 0 or abs(d("nitrile")) > 0 or abs(d("nitro")) > 0 or abs(d("azide")) > 0:
        return 9, "fgi"

    # 10 FGA — an unsaturation / sulfonyl introduced, skeleton otherwise intact
    if d("alkyne") > 0 or d("alkene") > 0 or d("imine") > 0:
        return 10, "fga"

    return 0, "unclassified"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    test = json.load(open(TEST_JSON))
    res = [json.loads(l) for l in open(RESULTS)]
    if args.limit:
        res = res[:args.limit]

    counts = defaultdict(int)
    labels = {}
    for i, r in enumerate(res):
        t = test[r["idx"]]["original_smiles"]
        cid, cname = classify(t.get("Starting Material", ""),
                              t.get("Reagents", ""),
                              t.get("End Product", ""))
        counts[cname] += 1
        labels[r["idx"]] = {"class_id": cid, "class_name": cname}
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(res)}")

    n = len(res)
    print(f"\nReaction-class distribution (n={n})")
    print(f"{'class':<34}{'count':>7}{'%':>8}   {'Schneider ref':>14}")
    ref = {"heteroatom_alkylation_arylation": 30.3, "acylation": 23.8,
           "deprotections": 16.5, "cc_bond_formation": 11.3, "reductions": 9.2,
           "fgi": 3.7, "heterocycle_formation": 1.8, "oxidations": 1.6,
           "protections": 1.6, "fga": 0.5}
    for cid in [1, 2, 6, 3, 7, 9, 4, 8, 5, 10, 0, -1]:
        name = CLASS_NAMES[cid]
        c = counts.get(name, 0)
        r_ = ref.get(name, None)
        rs = f"{r_:>13.1f}%" if r_ is not None else ""
        print(f"{name:<34}{c:>7}{c/n*100:>7.1f}%   {rs}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": n, "counts": dict(counts), "labels": labels}, open(OUT, "w"), indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
