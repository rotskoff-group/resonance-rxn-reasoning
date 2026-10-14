#!/usr/bin/env python3
"""Memory-bounded replacement for resonance...collect_screen.

The repo version does `pa.concat_tables([read_table(f) for f in files]).to_pandas()`.
That is fine for the original 18-template screen, but this run screens 92 templates
over 110M molecules -> ~812M match rows, roughly 60 GB as Arrow and 150 GB+ once
converted to pandas objects. It would be OOM-killed.

This version streams: one input parquet at a time, appending rows to a per-(template,
slot) ParquetWriter. Peak memory is one chunk plus writer buffers, so it runs in a few
GB regardless of total size. Output layout is byte-for-byte compatible with the repo
version, so apply_pairs.py consumes it unchanged:

    {output_dir}/template_{TID:03d}/slot_{0,1}.parquet

Ported from run_dir('forward_enum')/scripts/collect_screen_streaming.py.
CLI: python -m resonance.pipelines.forward_enum.collect_screen_streaming --screen-dir
     --output-dir [--max-per-slot 200000] [--seed 42]
"""
import argparse
import random
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([("smiles", pa.string()), ("cid", pa.int64())])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--screen-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-per-slot", type=int, default=200_000,
                   help="Cap molecules kept per (template, slot). apply_pairs samples at most "
                        "--max-pairs pairs, and 200k x 200k possible pairs already dwarfs any "
                        "cap we use, so keeping every match only costs disk. 0 = keep all.")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    files = sorted(Path(a.screen_dir).glob("screen_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no screen_*.parquet in {a.screen_dir}")
    print(f"streaming {len(files)} screen files", flush=True)

    # Reservoir-sample per (template, slot): uniform sample of the matches, bounded
    # memory and bounded disk. Writing every match would cost ~34 GB at 110M molecules
    # for no gain, since pairing only ever draws --max-pairs of them.
    cap = a.max_per_slot
    rng = random.Random(a.seed)
    keep: dict[tuple[int, int], list] = {}
    counts: Counter = Counter()   # total matches SEEN (not kept)
    for n, f in enumerate(files, 1):
        tbl = pq.read_table(f)
        if tbl.num_rows == 0:
            continue
        d = tbl.to_pydict()
        for smi, cid, tid, slot in zip(d["smiles"], d["cid"], d["template_id"], d["slot_idx"], strict=True):
            key = (int(tid), int(slot))
            counts[key] += 1
            res = keep.setdefault(key, [])
            if cap <= 0 or len(res) < cap:
                res.append((smi, cid))
            else:
                j = rng.randrange(counts[key])
                if j < cap:
                    res[j] = (smi, cid)
        if n % 50 == 0:
            print(f"  {n}/{len(files)} files, {sum(counts.values()):,} matches seen, "
                  f"{sum(len(v) for v in keep.values()):,} kept", flush=True)

    n_files = 0
    for (tid, slot), rows in sorted(keep.items()):
        outd = Path(a.output_dir) / f"template_{tid:03d}"
        outd.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({"smiles": pa.array([r[0] for r in rows], pa.string()),
                      "cid": pa.array([r[1] for r in rows], pa.int64())}, schema=SCHEMA),
            outd / f"slot_{slot}.parquet", compression="zstd")
        n_files += 1

    tids = sorted({k[0] for k in counts})
    print(f"\ndone: {sum(counts.values()):,} matches seen, "
          f"{sum(len(v) for v in keep.values()):,} kept (cap {cap}/slot), "
          f"{len(tids)} templates, {n_files} slot files")
    both = [t for t in tids if (t, 0) in counts and (t, 1) in counts]
    print(f"templates with BOTH slots populated (pairable): {len(both)}")
    missing = [t for t in tids if t not in both]
    if missing:
        print(f"only one slot matched (not pairable): {missing}")


if __name__ == "__main__":
    main()
