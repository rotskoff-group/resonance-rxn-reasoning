"""Phase 1 of forward screening: substructure scan for no_match 2-reactant templates.

For each molecule in a chunk, tests which (template, reactant-slot) patterns it
matches using substructure queries. Specifically focuses on the 2-reactant templates
that had no hits in the original retro-enumeration scheme.

Output: a parquet with columns [smiles, cid, template_id, slot_idx], one row per
(molecule, slot) match. Used by collect_screen.py to group matches per template.
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from rdkit import Chem, rdBase

from resonance.pipelines.synthetic_data.apply_chunk import _read_chunk

rdBase.DisableLog("rdApp.*")

NO_MATCH_2RXN_IDS: frozenset[int] = frozenset(
    [10, 11, 16, 23, 29, 31, 34, 71, 73, 74, 75, 76, 77, 78, 79, 81, 83, 85]
)


def _split_lhs(forward_smarts: str) -> list[str]:
    """Split LHS of a forward SMARTS into individual reactant SMARTS fragments."""
    lhs = forward_smarts.split(">>")[0]
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in lhs:
        if ch in "([":
            depth += 1
            current.append(ch)
        elif ch in ")]":
            depth -= 1
            current.append(ch)
        elif ch == "." and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def screen_chunk(
    parquet_path: str,
    templates_path: str,
    chunk_id: int,
    chunk_size: int,
    output_dir: str,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"screen_{chunk_id:04d}.parquet"

    with open(templates_path) as f:
        templates = json.load(f)

    # Compile one query mol per (template_id, slot_idx)
    queries: list[tuple[int, int, Chem.Mol]] = []
    for t in templates:
        if t["id"] not in NO_MATCH_2RXN_IDS:
            continue
        slot_smarts = _split_lhs(t["smarts"])
        if len(slot_smarts) != 2:
            logger.warning(
                f"Template {t['id']}: expected 2 reactant slots, got {len(slot_smarts)}, skipping"
            )
            continue
        for slot_idx, pat in enumerate(slot_smarts):
            q = Chem.MolFromSmarts(pat)
            if q is None:
                logger.warning(
                    f"Template {t['id']} slot {slot_idx}: SMARTS parse failed: {pat!r}"
                )
                continue
            queries.append((t["id"], slot_idx, q))

    logger.info(f"Chunk {chunk_id}: screening {len(queries)} (template, slot) queries")

    mols = _read_chunk(parquet_path, chunk_id, chunk_size)

    smiles_col: list[str] = []
    cid_col: list[int] = []
    tid_col: list[int] = []
    slot_col: list[int] = []

    for mol_smi, cid, _parent in mols:
        mol = Chem.MolFromSmiles(mol_smi)
        if mol is None:
            continue
        for tid, slot_idx, query in queries:
            if mol.HasSubstructMatch(query):
                smiles_col.append(mol_smi)
                cid_col.append(cid if cid is not None else -1)
                tid_col.append(tid)
                slot_col.append(slot_idx)

    table = pa.table(
        {
            "smiles": pa.array(smiles_col, type=pa.string()),
            "cid": pa.array(cid_col, type=pa.int64()),
            "template_id": pa.array(tid_col, type=pa.int32()),
            "slot_idx": pa.array(slot_col, type=pa.int32()),
        }
    )
    pq.write_table(table, output_path)
    logger.info(
        f"Chunk {chunk_id}: {len(mols)} molecules → {len(smiles_col)} matches → {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Substructure screen: find molecules matching each no_match template reactant slot"
    )
    parser.add_argument("--input", required=True, help="Parquet path")
    parser.add_argument(
        "--templates", required=True, help="reaction_templates.json path"
    )
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    screen_chunk(
        parquet_path=args.input,
        templates_path=args.templates,
        chunk_id=args.chunk_id,
        chunk_size=args.chunk_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
