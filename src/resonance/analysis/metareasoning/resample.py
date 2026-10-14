#!/usr/bin/env python3
"""Regenerate turn 1 under the sampled decode, and get the exact resampling null with it.

Draws K samples per item at T=0.6/0.95/20. Sample 0 becomes turn 1; samples 1..K-1 become
that arm's C5_fresh pool. The two are DISJOINT draws from one sampler, which matters:
writing sample 0 into both would make "turn-1 accuracy vs its own C5_fresh" pass by
construction, and that comparison is the check that decode is matched.

K=9 is the default so the null keeps 7 usable draws -- transition_matrix.py
--null-source fresh drops the C5_fresh record's own sample 0 as the answer it reports.

Every archived 8B turn-1 is greedy, and a greedy turn 1 scored against a sampled null
flips the null's sign rather than merely biasing it. Nothing downstream is interpretable
until this has run.

Outputs per tag (never a single shared file -- a SLURM array runs the tags concurrently):
    data/turn1_t06_<tag>.json              turn 1, merged later by merge_turn1.py
    <fresh-dir>/<tag>/C5_fresh.jsonl       the null pool, in the runner's own format

PER_TAG_BUDGET carries q8_base's larger caps; it is the one arm the old 16k cap truncated.

  python scripts/build_eval_set_resampled.py --tags q8_s3 --k 9 --fresh-dir outputs/fresh_t06
"""
import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent / "frozen"))
from selfcorrect_runner import turn1_messages
from vllm import LLM, SamplingParams

RUN = Path(__file__).resolve().parents[1]

# turn-1 tag -> model tag whose weights generate it. q8_s1 appears twice because S1 was
# trained on a bespoke prompt worth 16.2 pp (45.76% native vs 29.52% under `rules`); run
# it both ways or the arm measures prompt mismatch rather than self-correction.
TAG_MODEL = {
    "q8_base": "q8_base",
    "q8_s1": "q8_s1",
    "q8_s1_native": "q8_s1",
    "q8_s2": "q8_s2",
    "q8_s3": "q8_s3",
}

# q8_base is the one arm the old 16k cap truncated (17.6% of its turn-1s never emit an
# answer), so it needs a larger budget or the regeneration inherits the defect.
PER_TAG_BUDGET = {
    "q8_base": (32768, 40960),
}
DEFAULT_BUDGET = (4096, 16384)


def stratified(records, n, seed=42):
    """N items spread over reaction classes, largest-remainder by class frequency.

    A contiguous head of the test set is not a sample: USPTO-50k test is not shuffled by
    class, and the pilot exists to set the restate floor PER ARM, which is compared across
    classes later. Rare classes are floored at 1 so they appear at all, but note that a
    300-item pilot cannot support per-class rates -- it sizes a single pooled gate.
    """
    import random
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in records:
        by[r.get("reaction_class") or "unclassified"].append(r)
    tot = len(records)
    quota = {c: max(1, round(n * len(v) / tot)) for c, v in by.items()}
    out = []
    for c, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        out += rng.sample(v, min(quota[c], len(v)))
    rng.shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=",".join(TAG_MODEL),
                    help="comma-list of turn-1 tags to regenerate")
    ap.add_argument("--base-data", default="data/eval_5000_greedy_turn1.json",
                    help="source of each item's per-tag system_prompt/task_prompt")
    ap.add_argument("--out", default="",
                    help="merged eval set. Leave EMPTY under a SLURM array: each task "
                         "writes only its own data/turn1_t06_<tag>.json shard, and "
                         "merge_turn1.py combines them. Passing --out from two concurrent "
                         "tasks makes each overwrite the other's merge.")
    ap.add_argument("--fresh-dir", default="outputs",
                    help="C5_fresh.jsonl is written to <fresh-dir>/<tag>/")
    ap.add_argument("--k", type=int, default=9,
                    help="draws per item: 1 for turn 1 + (k-1) for the null pool")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = per-tag default")
    ap.add_argument("--max-model-len", type=int, default=0, help="0 = per-tag default")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed-mode", choices=["per_reaction", "shared"], default="per_reaction")
    ap.add_argument("--limit", type=int, default=0, help="first N items (debug only)")
    ap.add_argument("--stratified-n", type=int, default=0,
                    help="class-stratified subset of N items; prefer this to --limit, "
                         "which takes a contiguous head and inherits the test set's "
                         "ordering")
    ap.add_argument("--stratify-seed", type=int, default=42)
    args = ap.parse_args()

    if args.k < 3:
        raise SystemExit("--k must be >= 3: 1 draw for turn 1, and the null needs at least "
                         "2 more so that dropping C5_fresh's own reported answer leaves one")

    import models as M
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    for t in tags:
        if t not in TAG_MODEL:
            raise SystemExit(f"unknown turn-1 tag {t!r}; known: {sorted(TAG_MODEL)}")

    records = json.load(open(RUN / args.base_data))
    if args.stratified_n:
        records = stratified(records, args.stratified_n, args.stratify_seed)
    elif args.limit:
        records = records[: args.limit]
    logger.info(f"{len(records)} items | tags {tags} | K={args.k} "
                f"@ T={args.temperature}/top_p={args.top_p}/top_k={args.top_k}")

    decode = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}
    # Accumulated across tags so one output file carries the whole lineage.
    turn1_new = {}

    for tag in tags:
        src = [r for r in records if tag in r["turn1"]]
        if not src:
            logger.warning(f"{tag}: no item carries an archived turn-1 prompt; skipping")
            continue
        spec = M.get(TAG_MODEL[tag])
        mt, mml = PER_TAG_BUDGET.get(tag, DEFAULT_BUDGET)
        mt = args.max_tokens or mt
        mml = args.max_model_len or mml
        logger.info(f"=== {tag} ({spec.path}) | {len(src)} items | "
                    f"max_tokens={mt} max_model_len={mml}")

        llm_kwargs: dict[str, Any] = dict(model=spec.path, tensor_parallel_size=args.tp, dtype="bfloat16",
                          max_model_len=mml, gpu_memory_utilization=args.gpu_mem,
                          seed=args.seed, enable_prefix_caching=True)
        lora_req = None
        lora_path = getattr(spec, "lora_path", None)
        if lora_path:
            from vllm.lora.request import LoRARequest
            llm_kwargs.update(enable_lora=True, max_lora_rank=128)
            lora_req = LoRARequest("arm", 1, lora_path)
        llm = LLM(**llm_kwargs)

        convs = [turn1_messages(r["turn1"][tag]) for r in src]
        sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                            top_k=args.top_k, max_tokens=mt, seed=args.seed)
        if args.seed_mode == "per_reaction":
            # The frozen script gave every request one seed, so draw j shared a random
            # stream across all reactions; seed each reaction separately instead.
            base_sp, sp = sp, []
            for r in src:
                p = base_sp.clone()
                p.seed = args.seed + int(r["eval_idx"])
                sp.append(p)
        kw: dict[str, Any] = {"lora_request": lora_req} if lora_req else {}
        outs = llm.chat(convs, sampling_params=sp, **kw)

        fresh_dir = RUN / args.fresh_dir / tag
        fresh_dir.mkdir(parents=True, exist_ok=True)
        fresh_path = fresh_dir / "C5_fresh.jsonl"
        n_short = 0
        with open(fresh_path, "w") as fh:
            for r, o in zip(src, outs, strict=True):
                t1 = r["turn1"][tag]
                draws = list(o.outputs)
                if len(draws) < args.k:
                    n_short += 1
                texts = [d.text for d in draws]
                ntok = [len(d.token_ids) for d in draws]

                # sample 0 -> turn 1
                turn1_new.setdefault(r["eval_idx"], {})[tag] = {
                    "response": texts[0],
                    "answer": t1.get("answer"),
                    "answer_type": t1.get("answer_type"),
                    "question_type": t1.get("question_type"),
                    "task_prompt": t1.get("task_prompt"),
                    "system_prompt": t1.get("system_prompt"),
                    # Sampled, not the archived greedy response: not comparable to a
                    # published accuracy.
                    "prompt_variant": f"{t1.get('prompt_variant') or 'unknown'}_t06_sample0",
                    "decode": decode,
                    "n_output_tokens": ntok[0],
                }

                # samples 1.. -> the null pool, in the shape the runner emits for C5_fresh
                fh.write(json.dumps({
                    "idx": r["source_idx"],
                    "system_prompt": t1.get("system_prompt"),
                    "task_prompt": t1.get("task_prompt"),
                    "response": texts[1],
                    "answer": t1.get("answer"),
                    "answer_type": t1.get("answer_type"),
                    "question_type": t1.get("question_type"),
                    "eval_idx": r["eval_idx"],
                    "probe_idx": r["probe_idx"],
                    "reaction_class": r["reaction_class"],
                    "condition": "C5_fresh",
                    "model_tag": TAG_MODEL[tag],
                    "turn1_tag": tag,
                    "turn1_prompt_variant": t1.get("prompt_variant"),
                    "turn1_response": texts[0],
                    "n_output_tokens": ntok[1],
                    "think_prefilled": spec.think_is_prefilled,
                    "max_tokens": mt,
                    "decode": decode,
                    "n_rollouts": len(texts) - 1,
                    "skipped": None,
                    "responses": texts[1:],
                    "n_output_tokens_all": ntok[1:],
                }) + "\n")
        logger.info(f"wrote {fresh_path} ({len(src)} records"
                    + (f", {n_short} returned < K draws" if n_short else "") + ")")

        # One file per tag so a SLURM array cannot clobber a shared merge; also the
        # artifact a requeue gates on.
        shard = RUN / f"data/turn1_t06_{tag}.json"
        json.dump({str(i): v[tag] for i, v in turn1_new.items() if tag in v},
                  open(shard, "w"))
        logger.info(f"wrote {shard}")
        del llm

    if not args.out:
        logger.info("no --out given: per-tag shards only, run merge_turn1.py to combine")
        return
    out = []
    for r in records:
        got = turn1_new.get(r["eval_idx"])
        if not got:
            continue
        d = dict(r)
        d["turn1"] = dict(got)          # ONLY resampled turn-1s; never mix in greedy ones
        out.append(d)
    dest = RUN / args.out
    json.dump(out, open(dest, "w"))
    cov = {t: sum(1 for r in out if t in r["turn1"]) for t in tags}
    logger.info(f"wrote {dest} ({len(out)} records) | turn-1 coverage {cov}")
    json.dump({"n": len(out), "k": args.k, "decode": decode, "coverage": cov,
               "base_data": args.base_data},
              open(RUN / (args.out.rsplit(".", 1)[0] + "_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
