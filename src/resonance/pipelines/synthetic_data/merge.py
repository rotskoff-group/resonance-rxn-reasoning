"""Merge retro-enumeration hits, deduplicate, and compute per-template stats."""

import argparse
import gzip
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from loguru import logger


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"{path}:{i + 1}: JSON parse error: {e}, skipping")


def merge_hits(
    hits_dir: str,
    output_path: str,
    stats_output_path: str,
    total_mols: int = 100_000,
) -> None:
    """Merge hits_*.jsonl files, dedup by (product_smiles, template_id), write stats.

    Files are shuffled before merging so that when two chunks produce the same
    (product_smiles, template_id) with different reactant_smiles (possible in the
    forward direction), the kept record is chosen uniformly at random rather than
    always favouring lower-indexed chunks.
    """
    seen: set[int] = set()
    template_meta: dict[int, str] = {}
    template_hits: dict[int, int] = defaultdict(int)
    template_sa_sum: dict[int, float] = defaultdict(float)
    template_sa_count: dict[int, int] = defaultdict(int)

    hit_files = sorted(Path(hits_dir).glob("hits_*.jsonl"))
    random.shuffle(hit_files)
    logger.info(
        f"Found {len(hit_files)} hit files in {hits_dir} (shuffled for unbiased dedup)"
    )

    opener = gzip.open if output_path.endswith(".gz") else open
    with opener(output_path, "wt") as out_f:
        for hit_file in hit_files:
            for record in iter_jsonl(str(hit_file)):
                key_bytes = (
                    f"{record['product_smiles']}|{record['template_id']}".encode()
                )
                key_hash = int.from_bytes(hashlib.md5(key_bytes).digest()[:8], "big")
                if key_hash in seen:
                    continue
                seen.add(key_hash)

                tid = record["template_id"]
                template_meta[tid] = record.get("template_name", "")
                template_hits[tid] += 1
                for sa in record.get("sa_scores", []):
                    template_sa_sum[tid] += sa
                    template_sa_count[tid] += 1

                out_f.write(json.dumps(record) + "\n")

    logger.info(f"Total unique hits: {len(seen):,} → {output_path}")

    stats = []
    for tid in sorted(template_hits.keys()):
        n_hits = template_hits[tid]
        n_sa = template_sa_count[tid]
        stats.append(
            {
                "template_id": tid,
                "template_name": template_meta.get(tid, ""),
                "n_hits": n_hits,
                "hit_rate": n_hits / total_mols if total_mols > 0 else 0.0,
                "mean_sa_score": template_sa_sum[tid] / n_sa if n_sa > 0 else None,
            }
        )

    with open(stats_output_path, "w") as sf:
        json.dump(stats, sf, indent=2)

    logger.info(f"Per-template stats ({len(stats)} templates) → {stats_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge retro-enumeration hits, deduplicate, and compute stats"
    )
    parser.add_argument(
        "--hits-dir", required=True, help="Directory with hits_*.jsonl files"
    )
    parser.add_argument("--output", required=True, help="Merged output .jsonl path")
    parser.add_argument(
        "--stats-output", required=True, help="Per-template stats .json path"
    )
    parser.add_argument(
        "--total-mols",
        type=int,
        default=100_000,
        help="Total molecules tried (denominator for hit_rate)",
    )
    args = parser.parse_args()
    merge_hits(
        hits_dir=args.hits_dir,
        output_path=args.output,
        stats_output_path=args.stats_output,
        total_mols=args.total_mols,
    )


if __name__ == "__main__":
    main()
