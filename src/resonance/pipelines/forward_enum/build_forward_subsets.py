#!/usr/bin/env python3
"""Build alpha-tempered 60k subsets (80/10/10) from the forward-enumerated data.

Same recipe as the retro subset builder (subset_sample.py here) — sample template
i in proportion to n_i^alpha for alpha in {0, 0.5, 1}, water-fill quotas that exceed a
template's supply, nest the splits, and assign train/val/test by position so any prefix
holds the ratio — but sourced from the forward data and with a validity gate.

Why the gate: the retro splits are only 18.8% chemically derivable, because reversing a
forward SMARTS drops leaving groups. Forward-enumerated records are ~99% derivable by
construction, and this script keeps only the subset where the template emits EXACTLY ONE
product, so every prompt has one answer that follows from the inputs.

Three passes, sized so the expensive one stays small:
  1. reservoir-sample raw lines per template (3x what any alpha needs) — byte-scan for
     template_id, no JSON parse, so 125M records stream quickly
  2. round-trip check only until each template has the valid records it needs — a few
     hundred thousand checks rather than 125M
  3. allocate, split, write

Weights come from the TRUE per-template counts (both stats.json files), not the sampled
pool, so alpha=1 still means "proportional to the real distribution".

Ported from run_dir('forward_enum')/scripts/build_forward_subsets.py.
CLI: python -m resonance.pipelines.forward_enum.build_forward_subsets [--run-dir D] [--out O]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import subprocess
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from resonance.paths import run_dir
from resonance.pipelines.forward_enum.audit_ground_truth import canon, forward_products, load_templates
from resonance.pipelines.forward_enum.subset_sample import allocate, split_for_position

RUN_DIR = run_dir("forward_enum")
SOURCES = [RUN_DIR / "outputs" / "full_110M" / b / "merged.jsonl.gz" for b in ("pairs", "uni")]
STATS = [RUN_DIR / "outputs" / "full_110M" / b / "stats.json" for b in ("pairs", "uni")]
OUT = RUN_DIR / "outputs" / "subsets_60k"
SIZE = 60_000
ALPHAS = [0.0, 0.5, 1.0]
SEED = 42
TID_KEY = b'"template_id":'
SPLITS = ("train", "val", "test")

_RXNS: dict | None = None


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def true_counts():
    c = defaultdict(int)
    for p in STATS:
        for r in json.load(open(p)):
            c[r["template_id"]] += r["n_hits"]
    return dict(c)


def parse_tid(line: bytes) -> int:
    p = line.find(TID_KEY)
    if p < 0:
        return int(json.loads(line)["template_id"])
    p += len(TID_KEY)
    return int(line[p : line.find(b",", p)])


def _init():
    global _RXNS
    _RXNS, _ = load_templates()


def check(line_bytes):
    """-> raw line if the template emits exactly one product and it is the recorded one."""
    rec = json.loads(line_bytes)
    assert _RXNS is not None, "init() loads the templates in every worker"
    prods, fired = forward_products(_RXNS.get(rec["template_id"]), rec["reactant_smiles"])
    gt = canon(rec["product_smiles"])
    if fired and gt in prods and len(prods) == 1:
        return line_bytes
    return None


def main() -> None:
    global SOURCES, STATS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR,
                    help="reads <run-dir>/outputs/full_110M/{pairs,uni}/{merged.jsonl.gz,stats.json}")
    ap.add_argument("--out", type=Path, default=None, help="default: <run-dir>/outputs/subsets_60k")
    a = ap.parse_args()
    SOURCES = [a.run_dir / "outputs" / "full_110M" / b / "merged.jsonl.gz" for b in ("pairs", "uni")]
    STATS = [a.run_dir / "outputs" / "full_110M" / b / "stats.json" for b in ("pairs", "uni")]
    OUT = a.out or a.run_dir / "outputs" / "subsets_60k"
    OUT.mkdir(parents=True, exist_ok=True)
    counts = true_counts()
    log(f"{len(counts)} templates, {sum(counts.values()):,} forward records")

    # per-template need = the largest quota any alpha asks for at this size
    quotas = {a: allocate(counts, SIZE, a) for a in ALPHAS}
    need = {t: max(quotas[a].get(t, 0) for a in ALPHAS) for t in counts}
    cap = {t: min(3 * max(need[t], 1), counts[t]) for t in counts}
    log(f"need {sum(need.values()):,} valid records; reservoir cap {sum(cap.values()):,}")

    # ---- pass 1: reservoir per template over the raw stream
    rand = random.Random(SEED).random
    res: dict[int, list[bytes]] = defaultdict(list)
    seen: dict[int, int] = defaultdict(int)
    total = 0
    t0 = time.time()
    for src in SOURCES:
        proc = subprocess.Popen(["zcat", str(src)], stdout=subprocess.PIPE, bufsize=1 << 20)
        assert proc.stdout is not None
        for line in proc.stdout:
            if line == b"\n":
                continue
            tid = parse_tid(line)
            seen[tid] += 1
            k = cap.get(tid, 0)
            if k:
                r = res[tid]
                if len(r) < k:
                    r.append(line.rstrip(b"\n"))
                else:
                    j = int(rand() * seen[tid])
                    if j < k:
                        r[j] = line.rstrip(b"\n")
            total += 1
            if total % 20_000_000 == 0:
                log(f"  {total:,} records streamed ({total/(time.time()-t0)/1e6:.2f}M/s)")
        proc.stdout.close()
        proc.wait()
    log(f"pass 1 done: {total:,} records, reservoir {sum(len(v) for v in res.values()):,}")

    # ---- pass 2: validity-check per template only until `need` is met
    valid: dict[int, list[bytes]] = {}
    n_checked = 0
    with Pool(min(16, os.cpu_count() or 4), initializer=_init) as pool:
        for tid in sorted(res):
            want = need.get(tid, 0)
            if want <= 0:
                continue
            lines = res[tid][:]
            random.Random(SEED + tid).shuffle(lines)
            keep: list[bytes] = []
            for i in range(0, len(lines), 2000):
                batch = lines[i : i + 2000]
                n_checked += len(batch)
                keep.extend(x for x in pool.map(check, batch, chunksize=25) if x)
                if len(keep) >= want:
                    break
            valid[tid] = keep[: max(want, 0)]
            if len(keep) < want:
                log(f"  template {tid}: only {len(keep)} valid of {want} needed")
    log(f"pass 2 done: checked {n_checked:,}, kept {sum(len(v) for v in valid.values()):,}")

    # ---- pass 3: allocate against what we actually have, then split
    avail = {t: len(v) for t, v in valid.items() if v}
    weights = {t: counts[t] for t in avail}          # alpha weighting uses TRUE counts
    summary = {}
    for ai, alpha in enumerate(ALPHAS):
        q = allocate({t: min(weights[t], avail[t]) for t in avail}, SIZE, alpha)
        rng = random.Random(SEED + 1000 * (ai + 1))
        adir = OUT / f"alpha_{alpha:g}" / "60k"
        adir.mkdir(parents=True, exist_ok=True)
        handles = {s: gzip.open(adir / f"{s}.jsonl.gz", "wb", compresslevel=6) for s in SPLITS}
        per_split = {s: 0 for s in SPLITS}
        per_template = {}
        try:
            for tid in sorted(q):
                take = min(q[tid], len(valid.get(tid, [])))
                if take <= 0:
                    continue
                idx = list(range(len(valid[tid])))
                rng.shuffle(idx)
                off = rng.randrange(10)
                tally = {s: 0 for s in SPLITS}
                for i in range(take):
                    s = split_for_position(i, off)
                    handles[s].write(valid[tid][idx[i]] + b"\n")
                    tally[s] += 1
                    per_split[s] += 1
                per_template[str(tid)] = tally
        finally:
            for h in handles.values():
                h.close()
        n = sum(per_split.values())
        summary[f"alpha_{alpha:g}"] = {
            "n_records": n, "split_totals": per_split,
            "n_templates": len(per_template), "per_template": per_template,
        }
        log(f"alpha={alpha:g}: {n:,} records "
            f"({per_split['train']:,}/{per_split['val']:,}/{per_split['test']:,}) "
            f"over {len(per_template)} templates")

    json.dump({"source": [str(s) for s in SOURCES], "size": SIZE, "alphas": ALPHAS,
               "seed": SEED, "validity_gate": "derivable_unique (forward round-trip, single product)",
               "configs": summary},
              open(OUT / "manifest.json", "w"), indent=2)
    log(f"wrote {OUT}/manifest.json")


if __name__ == "__main__":
    main()
