"""SLURM array worker: apply retro templates to a chunk of molecules."""

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, rdChemReactions

from resonance.chemistry.constants import R_CARBON_POOL, R_HETERO_POOL
from resonance.utils import reverse_smarts

rdBase.DisableLog("rdApp.*")

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402  # type: ignore

_DUMMY_QUERY = Chem.MolFromSmarts("[#0]")


def _resolve_wildcards(mol: Chem.Mol) -> Chem.Mol | None:
    """Replace dummy (*) atoms with randomly selected inert R groups.

    Iterates until no dummy atoms remain (capped at 20 to guard against
    unexpected loops). Returns None if replacement or sanitization fails.
    """
    if not mol.HasSubstructMatch(_DUMMY_QUERY):
        return mol
    result = mol
    for _ in range(20):
        match = result.GetSubstructMatch(_DUMMY_QUERY)
        if not match:
            break
        atom = result.GetAtomWithIdx(match[0])
        neighbor_nums = [n.GetAtomicNum() for n in atom.GetNeighbors()]
        pool = (
            R_HETERO_POOL
            if any(n in (7, 8, 16) for n in neighbor_nums)
            else R_CARBON_POOL
        )
        r_mol = Chem.MolFromSmiles(random.choice(pool))
        if r_mol is None:
            return None
        replaced = Chem.ReplaceSubstructs(result, _DUMMY_QUERY, r_mol, replaceAll=False)
        if not replaced:
            return None
        result = replaced[0]
    try:
        Chem.SanitizeMol(result)
        smi = Chem.MolToSmiles(result, canonical=True)
        return result if (smi and "*" not in smi) else None
    except Exception:
        return None


def compute_sa_scores(smiles_list: list[str]) -> list[float] | None:
    """Return SA scores (1=easy, 10=hard) for a list of SMILES, or None on failure."""
    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        try:
            scores.append(float(sascorer.calculateScore(mol)))
        except Exception:
            return None
    return scores


def apply_retro_template(
    mol_smiles: str,
    template_smarts: str,
) -> list[list[str]]:
    """Apply a template in retro direction; return list of reactant-SMILES sets.

    Each element is a list of canonical SMILES for one set of starting
    materials. Returns empty list on no match or any parse failure.
    """
    mol = Chem.MolFromSmiles(mol_smiles)
    if mol is None:
        return []
    retro = reverse_smarts(template_smarts)
    if retro is None:
        return []
    try:
        rxn = rdChemReactions.ReactionFromSmarts(retro)
    except Exception:
        return []
    if rxn is None:
        return []
    return _run_rxn(rxn, mol)


def _run_rxn(rxn: AllChem.ChemicalReaction, mol: Chem.Mol) -> list[list[str]]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = rxn.RunReactants((mol,))
    except Exception:
        return []

    reactant_sets: list[list[str]] = []
    seen: set[frozenset] = set()
    for product_tuple in results:
        smiles_set: list[str] = []
        valid = True
        for frag in product_tuple:
            try:
                Chem.SanitizeMol(frag)
                smi = Chem.MolToSmiles(frag, canonical=True)
                if not smi:
                    valid = False
                    break
                if "*" in smi:
                    frag = _resolve_wildcards(frag)
                    if frag is None:
                        valid = False
                        break
                    smi = Chem.MolToSmiles(frag, canonical=True)
                    if not smi or "*" in smi:
                        valid = False
                        break
                smiles_set.append(smi)
            except Exception:
                valid = False
                break
        if not valid or not smiles_set:
            continue
        key = frozenset(smiles_set)
        if key not in seen:
            seen.add(key)
            reactant_sets.append(smiles_set)
    return reactant_sets


def _read_chunk(
    parquet_path: str, chunk_id: int, chunk_size: int
) -> list[tuple[str, int | None, str | None]]:
    """Read a chunk of molecules from parquet using row-group-level reads.

    Reads only the row groups that overlap with [start, end), avoiding loading
    the entire file into memory. Critical for the 110M-molecule PubChem parquet
    where loading the full file per task would exhaust available RAM.

    Supports two parquet schemas:
      - Round 1 (PubChem source): columns CID + CanonicalSMILES
        → cid is set, parent_smiles is None (these molecules are tree roots)
      - Round 2+ (collect_reactants output): columns smiles + parent_smiles
        → cid is None, parent_smiles is the immediate parent from the prior round
    """
    start = chunk_id * chunk_size
    end = start + chunk_size

    pf = pq.ParquetFile(parquet_path)
    all_cols = pf.schema_arrow.names
    has_parent_col = "parent_smiles" in all_cols
    cols = ["smiles", "parent_smiles"] if has_parent_col else ["CID", "CanonicalSMILES"]

    cumulative = 0
    batches: list = []
    for rg_idx in range(pf.metadata.num_row_groups):
        rg_size = pf.metadata.row_group(rg_idx).num_rows
        if cumulative + rg_size <= start:
            cumulative += rg_size
            continue
        if cumulative >= end:
            break
        table = pf.read_row_group(rg_idx, columns=cols)
        local_start = max(0, start - cumulative)
        local_end = min(rg_size, end - cumulative)
        batches.append(table.slice(local_start, local_end - local_start))
        cumulative += rg_size

    if not batches:
        return []

    df = (pa.concat_tables(batches) if len(batches) > 1 else batches[0]).to_pandas()

    results: list[tuple[str, int | None, str | None]] = []
    for _, row in df.iterrows():
        if has_parent_col:
            smi_raw = str(row["smiles"])
            cid = None
            parent = (
                row["parent_smiles"] if pd.notna(row.get("parent_smiles")) else None
            )
        else:
            smi_raw = str(row["CanonicalSMILES"])
            cid = int(row["CID"])
            parent = None

        mol = Chem.MolFromSmiles(smi_raw)
        if mol is None:
            logger.warning(f"Invalid SMILES, skipping: {smi_raw}")
            continue
        results.append((Chem.MolToSmiles(mol, canonical=True), cid, parent))

    return results


def process_chunk(
    parquet_path: str,
    templates_path: str,
    chunk_id: int,
    chunk_size: int,
    output_dir: str,
    depth: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"hits_{chunk_id:04d}.jsonl")

    with open(templates_path) as f:
        templates = json.load(f)

    # Pre-compile retro reactions once per chunk (not once per mol).
    # Use retro_smarts directly when present (pre-written for retro direction);
    # otherwise reverse the forward smarts.
    compiled: list[tuple[dict, AllChem.ChemicalReaction]] = []
    failed: list[str] = []
    for t in templates:
        if "retro_smarts" in t:
            retro = t["retro_smarts"]
        else:
            retro = reverse_smarts(t["smarts"])
        if retro is None:
            failed.append(f"  template {t['id']} ({t['name']}): unparseable SMARTS")
            continue
        try:
            rxn = rdChemReactions.ReactionFromSmarts(retro)
            if rxn is None:
                failed.append(
                    f"  template {t['id']} ({t['name']}): ReactionFromSmarts returned None"
                )
            else:
                compiled.append((t, rxn))
        except Exception as e:
            failed.append(f"  template {t['id']} ({t['name']}): {e}")

    if failed:
        raise RuntimeError(
            f"{len(failed)}/{len(templates)} templates failed to compile:\n"
            + "\n".join(failed)
        )

    logger.info(f"All {len(compiled)} templates compiled successfully")

    mols = _read_chunk(parquet_path, chunk_id, chunk_size)
    if not mols:
        logger.info(f"Chunk {chunk_id}: empty, writing no hits")
        Path(output_path).touch()
        return

    logger.info(f"Chunk {chunk_id}: {len(mols)} valid molecules (depth={depth})")

    n_hits = 0
    n_pairs = 0
    with open(output_path, "w") as out_f:
        for mol_smiles, cid, parent_smiles in mols:
            mol = Chem.MolFromSmiles(mol_smiles)
            if mol is None:
                continue
            for t, rxn in compiled:
                n_pairs += 1
                reactant_sets = _run_rxn(rxn, mol)
                for smiles_set in reactant_sets:
                    record = {
                        "product_smiles": mol_smiles,
                        "product_cid": cid,
                        "reactant_smiles": smiles_set,
                        "template_id": t["id"],
                        "template_name": t["name"],
                        "template_smarts": t["smarts"],
                        "depth": depth,
                        "parent_smiles": parent_smiles,
                    }
                    out_f.write(json.dumps(record) + "\n")
                    n_hits += 1

    logger.info(
        f"Chunk {chunk_id}: {n_pairs} (mol,template) pairs → {n_hits} hits → {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply retro templates to a molecule chunk (SLURM array worker)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Parquet path (CID+CanonicalSMILES or smiles+parent_smiles)",
    )
    parser.add_argument(
        "--templates", required=True, help="reaction_templates.json path"
    )
    parser.add_argument(
        "--chunk-id", type=int, required=True, help="Zero-based chunk index"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="Molecules per chunk"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for hits_XXXX.jsonl files"
    )
    parser.add_argument(
        "--depth",
        type=int,
        required=True,
        help="Retrosynthetic depth of this round (1-indexed)",
    )
    args = parser.parse_args()
    process_chunk(
        parquet_path=args.input,
        templates_path=args.templates,
        chunk_id=args.chunk_id,
        chunk_size=args.chunk_size,
        output_dir=args.output_dir,
        depth=args.depth,
    )


if __name__ == "__main__":
    main()
