"""Phase 3 of forward screening: enumerate valid reactant pairs and run forward reactions.

Given slot_0.parquet (molecules matching reactant slot 0) and slot_1.parquet
(molecules matching reactant slot 1) for a single template, samples up to
--max-pairs pairs from S0 × S1 and runs RunReactants((mol_a, mol_b)).

Because every molecule in slot_0 is guaranteed to match the template's first
reactant pattern, and every molecule in slot_1 matches the second pattern, all
pairs here are chemically valid inputs — no wasted reaction attempts.

Output: hits_NNNN.jsonl (using template_id as the index) in the standard schema,
compatible with merge.py for aggregation across templates.
"""

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from loguru import logger
from rdkit import Chem, rdBase
from rdkit.Chem import rdChemReactions

from resonance.pipelines.synthetic_data.apply_chunk_forward import _run_forward

rdBase.DisableLog("rdApp.*")


def apply_pairs(
    template_id: int,
    templates_path: str,
    slot0_path: str,
    slot1_path: str,
    output_dir: str,
    max_pairs: int,
    seed: int = 42,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"hits_{template_id:04d}.jsonl"

    with open(templates_path) as f:
        templates = json.load(f)
    t = next((t for t in templates if t["id"] == template_id), None)
    if t is None:
        raise ValueError(f"Template id={template_id} not found in {templates_path}")

    try:
        rxn = rdChemReactions.ReactionFromSmarts(t["smarts"])
    except Exception as e:
        logger.error(f"Template {template_id}: compile error: {e}")
        return
    if rxn is None:
        logger.error(f"Template {template_id}: ReactionFromSmarts returned None")
        return

    s0 = pq.read_table(slot0_path).to_pydict()
    s1 = pq.read_table(slot1_path).to_pydict()
    smiles0, smiles1 = s0["smiles"], s1["smiles"]
    logger.info(
        f"Template {template_id} ({t['name']}): |S0|={len(smiles0):,}, |S1|={len(smiles1):,}"
    )

    # Build pair index list: sample without replacement up to max_pairs
    rng = random.Random(seed)
    total_possible = len(smiles0) * len(smiles1)
    if total_possible <= max_pairs:
        pairs = [(i, j) for i in range(len(smiles0)) for j in range(len(smiles1))]
    else:
        # Random index sampling — draw (i, j) pairs uniformly without full enumeration
        seen_pairs: set[tuple[int, int]] = set()
        pairs = []
        attempts = 0
        max_attempts = max_pairs * 5
        while len(pairs) < max_pairs and attempts < max_attempts:
            i = rng.randrange(len(smiles0))
            j = rng.randrange(len(smiles1))
            if (i, j) not in seen_pairs:
                seen_pairs.add((i, j))
                pairs.append((i, j))
            attempts += 1
        if len(pairs) < max_pairs:
            logger.warning(
                f"Template {template_id}: only drew {len(pairs)} unique pairs after {attempts} attempts"
            )

    logger.info(f"Trying {len(pairs):,} pairs")

    n_hits = 0
    with open(output_path, "w") as out_f:
        for i, j in pairs:
            smi_a, smi_b = smiles0[i], smiles1[j]
            mol_a = Chem.MolFromSmiles(smi_a)
            mol_b = Chem.MolFromSmiles(smi_b)
            if mol_a is None or mol_b is None:
                continue
            for product_smi in _run_forward(rxn, (mol_a, mol_b)):
                record = {
                    "product_smiles": product_smi,
                    "product_cid": None,
                    "reactant_smiles": [smi_a, smi_b],
                    "template_id": t["id"],
                    "template_name": t["name"],
                    "template_smarts": t["smarts"],
                    "depth": 0,
                    "parent_smiles": None,
                }
                out_f.write(json.dumps(record) + "\n")
                n_hits += 1

    logger.info(f"Template {template_id}: {n_hits} hits → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate valid reactant pairs for one template and run forward reactions"
    )
    parser.add_argument("--template-id", type=int, required=True)
    parser.add_argument(
        "--templates", required=True, help="reaction_templates.json path"
    )
    parser.add_argument("--slot0", required=True, help="slot_0.parquet path")
    parser.add_argument("--slot1", required=True, help="slot_1.parquet path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=100_000,
        help="Max (S0, S1) pairs to try per template",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    apply_pairs(
        template_id=args.template_id,
        templates_path=args.templates,
        slot0_path=args.slot0,
        slot1_path=args.slot1,
        output_dir=args.output_dir,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
