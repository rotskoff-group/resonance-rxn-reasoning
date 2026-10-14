"""SLURM array worker: apply templates in FORWARD direction to a molecule chunk.

For 1-reactant templates, applies the forward SMARTS to every molecule in the
chunk.  For 2-reactant templates, enumerates all ordered pairs from a random
subsample of the chunk (default: 300 molecules → up to 90,000 ordered pairs per
template).  Templates requiring 3+ reactants are skipped.

Output schema is identical to apply_chunk.py retro records:
  product_smiles  — the synthesized product (NOT a PubChem molecule)
  product_cid     — null (product is synthesized, not from PubChem directly)
  reactant_smiles — list of PubChem input molecules used in the reaction
  template_id / template_name / template_smarts
  depth           — 0 (forward step, one synthesis step from PubChem roots)
  parent_smiles   — null

Use merge.py (same script as retro) to merge hits_*.jsonl → merged.jsonl + stats.
"""

import argparse
import json
import os
import random
import warnings
from pathlib import Path

from loguru import logger
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdChemReactions

from resonance.pipelines.synthetic_data.apply_chunk import _read_chunk

rdBase.DisableLog("rdApp.*")


def _count_reactants(forward_smarts: str) -> int:
    """Count distinct reactant fragments on the LHS of a forward SMARTS."""
    lhs = forward_smarts.split(">>")[0]
    return lhs.count(".") + 1


def _run_forward(rxn: AllChem.ChemicalReaction, reactants: tuple) -> list[str]:
    """Run a forward reaction; return canonical SMILES of distinct valid products.

    When a reaction produces multiple fragments (main product + leaving groups /
    byproducts), only the largest fragment by heavy-atom count is kept.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = rxn.RunReactants(reactants)
    except Exception:
        return []

    products: list[str] = []
    seen: set[str] = set()
    for product_tuple in results:
        if not product_tuple:
            continue
        valid = []
        for frag in product_tuple:
            try:
                Chem.SanitizeMol(frag)
                valid.append(frag)
            except Exception:
                pass
        if not valid:
            continue
        main = max(valid, key=lambda m: m.GetNumHeavyAtoms())
        smi = Chem.MolToSmiles(main, canonical=True)
        if not smi or "*" in smi or smi in seen:
            continue
        seen.add(smi)
        products.append(smi)
    return products


def process_forward_chunk(
    parquet_path: str,
    templates_path: str,
    chunk_id: int,
    chunk_size: int,
    output_dir: str,
    subsample_size: int,
    depth: int = 0,
    template_ids: set[int] | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"hits_{chunk_id:04d}.jsonl")

    with open(templates_path) as f:
        templates = json.load(f)

    if template_ids is not None:
        templates = [t for t in templates if t["id"] in template_ids]

    one_rxn: list[tuple[dict, AllChem.ChemicalReaction]] = []
    two_rxn: list[tuple[dict, AllChem.ChemicalReaction]] = []
    skipped_ids: list[int] = []

    for t in templates:
        n = _count_reactants(t["smarts"])
        if n > 2:
            skipped_ids.append(t["id"])
            continue
        try:
            rxn = rdChemReactions.ReactionFromSmarts(t["smarts"])
        except Exception as e:
            logger.warning(f"Template {t['id']} ({t['name']}): compile error: {e}")
            skipped_ids.append(t["id"])
            continue
        if rxn is None:
            skipped_ids.append(t["id"])
            continue
        (one_rxn if n == 1 else two_rxn).append((t, rxn))

    logger.info(
        f"Compiled {len(one_rxn)} one-reactant, {len(two_rxn)} two-reactant templates. "
        f"Skipped {len(skipped_ids)} (3+ reactants or compile fail): {skipped_ids}"
    )

    mols = _read_chunk(parquet_path, chunk_id, chunk_size)
    if not mols:
        logger.info(f"Chunk {chunk_id}: empty, writing no hits")
        Path(output_path).touch()
        return

    logger.info(f"Chunk {chunk_id}: {len(mols)} valid molecules")

    mol_objects: list[tuple[str, int | None, Chem.Mol]] = []
    for smi, cid, _parent in mols:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol_objects.append((smi, cid, mol))

    subsample = (
        mol_objects
        if len(mol_objects) <= subsample_size
        else random.sample(mol_objects, subsample_size)
    )

    n_hits = 0
    with open(output_path, "w") as out_f:
        # 1-reactant templates: every molecule in chunk
        for mol_smi, _cid, mol in mol_objects:
            for t, rxn in one_rxn:
                for product_smi in _run_forward(rxn, (mol,)):
                    record = {
                        "product_smiles": product_smi,
                        "product_cid": None,
                        "reactant_smiles": [mol_smi],
                        "template_id": t["id"],
                        "template_name": t["name"],
                        "template_smarts": t["smarts"],
                        "depth": depth,
                        "parent_smiles": None,
                    }
                    out_f.write(json.dumps(record) + "\n")
                    n_hits += 1

        # 2-reactant templates: all ordered pairs from within-chunk subsample.
        # i:  so each unordered pair is tried once in each order.
        # Self-pairs (i == j) are included to catch bifunctional / intramolecular cases.
        for i, (smi_a, _cid_a, mol_a) in enumerate(subsample):
            for smi_b, _cid_b, mol_b in subsample[i:]:
                is_self = smi_a == smi_b
                for t, rxn in two_rxn:
                    orderings = (
                        [((mol_a, mol_b), [smi_a, smi_b])]
                        if is_self
                        else [
                            ((mol_a, mol_b), [smi_a, smi_b]),
                            ((mol_b, mol_a), [smi_b, smi_a]),
                        ]
                    )
                    for reactants, r_smiles in orderings:
                        for product_smi in _run_forward(rxn, reactants):
                            record = {
                                "product_smiles": product_smi,
                                "product_cid": None,
                                "reactant_smiles": r_smiles,
                                "template_id": t["id"],
                                "template_name": t["name"],
                                "template_smarts": t["smarts"],
                                "depth": depth,
                                "parent_smiles": None,
                            }
                            out_f.write(json.dumps(record) + "\n")
                            n_hits += 1

    logger.info(f"Chunk {chunk_id}: {n_hits} forward hits → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply templates in FORWARD direction to a molecule chunk (SLURM array worker)"
    )
    parser.add_argument(
        "--input", required=True, help="Parquet path (CID+CanonicalSMILES)"
    )
    parser.add_argument(
        "--templates", required=True, help="reaction_templates.json path"
    )
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--subsample-size",
        type=int,
        default=300,
        help="Max molecules used for within-chunk pair enumeration (2-reactant templates). "
        "Pairs = subsample_size^2 ordered pairs per template.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="Synthesis depth to stamp on output records (0 = direct from PubChem roots).",
    )
    parser.add_argument(
        "--template-ids",
        type=str,
        default=None,
        help="Comma-separated list of template IDs to run (default: all templates).",
    )
    args = parser.parse_args()
    tid_filter = (
        {int(x) for x in args.template_ids.split(",")}
        if args.template_ids
        else None
    )
    process_forward_chunk(
        parquet_path=args.input,
        templates_path=args.templates,
        chunk_id=args.chunk_id,
        chunk_size=args.chunk_size,
        output_dir=args.output_dir,
        subsample_size=args.subsample_size,
        depth=args.depth,
        template_ids=tid_filter,
    )


if __name__ == "__main__":
    main()
