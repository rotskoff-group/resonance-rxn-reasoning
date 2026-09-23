"""Forward-template oracle used by the validity gate in build_forward_subsets.

Ported verbatim (canon, load_templates, forward_products) from
run_dir('synthetic_audit')/scripts/audit_ground_truth.py. TEMPLATES is the repo copy of
reaction_templates.json, byte-identical to the one the published run read.
"""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

TEMPLATES = Path(__file__).resolve().parents[2] / "data" / "datasets" / "reaction_templates.json"


def canon(smi):
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def load_templates():
    """template_id -> compiled forward reaction (the `smarts` field, not retro)."""
    rxns, names = {}, {}
    for t in json.load(open(TEMPLATES)):
        names[t["id"]] = t["name"]
        try:
            rxn = AllChem.ReactionFromSmarts(t["smarts"])
            rxn.Initialize()
            rxns[t["id"]] = rxn
        except Exception:
            rxns[t["id"]] = None
    return rxns, names


def forward_products(rxn, reactant_smiles):
    """Every distinct product the forward template emits from these reactants.

    Reactant ORDER matters to RunReactants and the retro step gives no guarantee it
    matches the forward template's operand order, so all permutations are tried and
    the results unioned. Returns (product_set, fired) — fired distinguishes "the
    template produced nothing" from "the template produced something unparseable".
    """
    if rxn is None:
        return set(), False
    mols = [Chem.MolFromSmiles(s) for s in reactant_smiles]
    if any(m is None for m in mols):
        return set(), False

    n = rxn.GetNumReactantTemplates()
    products, fired = set(), False
    orderings = permutations(mols, n) if len(mols) >= n else ([tuple(mols)] if n == 1 else [])
    for combo in orderings:
        try:
            outcomes = rxn.RunReactants(combo)
        except Exception:
            continue
        for outcome in outcomes:
            fired = True
            for p in outcome:
                try:
                    Chem.SanitizeMol(p)
                except Exception:
                    continue
                smi = Chem.MolToSmiles(p)
                if smi:
                    products.add(smi)
    return products, fired
