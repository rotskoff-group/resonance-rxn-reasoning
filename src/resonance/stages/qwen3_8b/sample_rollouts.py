#!/usr/bin/env python3
"""
Rejection-sampling rollout generation (S1 model -> S2 training candidates).

Ported from run_dir('s1_8b')/scripts/sample_rollouts.py. The published run launched four
workers through `run_gpu rollouts`:

    python -m resonance.stages.qwen3_8b.sample_rollouts --shard i --nshards 4 \
        --out <rollouts.shard{i}-of-4.jsonl>

Loads the USPTO-50k train set, generates N=8 completions per reaction at
temperature=0.9 from the SFT model, and writes every response to a JSONL
for later filtering.

Output schema (one line per reaction):
{
  "reaction_id": str,
  "starting_material": str,
  "reagent_list": str,
  "gold_product": str,
  "responses": [str, ...]   # length == N_SAMPLES
}
"""

import argparse
import json
import time
from pathlib import Path

from resonance.paths import run_dir

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH = str(run_dir("traces") / "outputs/model_0a_no_reaction_id")
TRAIN_JSON = str(run_dir("uspto") / "train.json")
OUT_PATH   = str(run_dir("s1_8b") / "outputs/rollouts/rollouts.jsonl")

N_SAMPLES   = 8
TEMPERATURE = 0.9
MAX_TOKENS  = 1024   # output tokens; prompt is ~300-400 tokens, so total < 1500
MAX_MODEL_LEN = 2048

# TP=1, sharded. Measured on this box: TP=1 on one card is 11,736 out-tok/s vs 1,949 for
# TP=4 across four -- a 6.0x loss. An 8B at max_model_len=2048 fits in one 95 GB card, so
# TP=4 bought no memory and paid a per-layer all-reduce, which the NVLink pairing (0-1, 2-3,
# 4-5, 6-7, no NVSwitch) then routed over PCIe. Sharding removes cross-GPU traffic entirely,
# so GPU placement stops mattering.
TENSOR_PARALLEL = 1

SYSTEM_PROMPT = """\
You are an expert organic chemist specializing in the forward prediction of organic reactions.
Correctly predicting the product of an organic reaction involves understanding of several key concepts.
Some of the most important concepts to consider include:

1. Acid-base chemistry: these reactions will proceed from a stronger acid to a weaker one.
2. Single displacement reactions: Similarly, these reactions will proceed by a more reactive element displacing a less reactive one.
3. Redox reactions: identify the relationship between reagents and their potential for reducing/oxidizing specific functional groups on a molecule.
4. Mechanistic analysis: leveraging knowledge of common reaction mechanisms, one can infer key functional groups in a molecule and their potential relationships to reagents and other key intramolecular functional groups

Your task is to predict the major product(s)
of organic reactions given starting materials and reagents."""

USER_TEMPLATE = """\
Here are the SMILES of the starting material and reagent(s).
Starting material: {starting_material}
Reagents: {reagent_list}

Predict the single most likely primary product of the chemical reaction in SMILES format.
The reaction should be chemically reasonable, following typical organic reactivity principles.
Instructions:
1. Identify the functional groups in the starting material
2. Determine which reagent(s) will react with these functional groups
3. Apply your knowledge of organic reaction mechanisms to predict the major product
4. Consider regioselectivity and stereoselectivity where applicable
If your proposed mechanism or product seems chemically implausible, revise your
reasoning and try a different mechanistic path.

Return the answer between <answer> and </answer> tags."""


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0, help="This worker's shard index.")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=None,
                    help="Defaults to OUT_PATH with a .shard<i>-of-<n> suffix.")
    ap.add_argument("--limit", type=int, default=0, help="Smoke test: first N of the shard.")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--train-json", default=TRAIN_JSON)
    a = ap.parse_args()
    if not 0 <= a.shard < a.nshards:
        ap.error(f"--shard must be in [0,{a.nshards})")
    return a


def build_messages(r):
    user_msg = USER_TEMPLATE.format(
        starting_material=r["starting_material"],
        reagent_list=r["reagent_list"] or "None specified",
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]


def main():
    # Imported here so filter_rollouts can read the prompt constants without vLLM.
    from vllm import LLM, SamplingParams

    args = parse_args()
    out_path = Path(args.out) if args.out else Path(
        OUT_PATH).with_suffix(f".shard{args.shard}-of-{args.nshards}.jsonl")

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading train set...", flush=True)
    with open(args.train_json) as f:
        raw = json.load(f)

    records = []
    for row_idx, item in enumerate(raw):
        sm   = item["original_smiles"]["Starting Material"]
        reag = item["original_smiles"].get("Reagents", "") or ""
        prod = item["original_smiles"]["End Product"]
        rid  = item["metadata"]["reaction_id"]
        records.append({"row_idx": row_idx, "reaction_id": rid, "starting_material": sm,
                        "reagent_list": reag, "gold_product": prod})

    print(f"Loaded {len(records):,} reactions", flush=True)

    # Contiguous slice, so shards partition the train set exactly and the concatenation is
    # in original train order when the shards are joined in index order.
    total = len(records)
    per = -(-total // args.nshards)
    lo, hi = args.shard * per, min(total, (args.shard + 1) * per)
    records = records[lo:hi]
    if args.limit:
        records = records[: args.limit]
    print(f"shard {args.shard}/{args.nshards}: rows [{lo},{hi}) -> {len(records):,} reactions",
          flush=True)
    print(f"output: {out_path}", flush=True)

    # ── build prompts ──────────────────────────────────────────────────────────
    messages_list = [build_messages(r) for r in records]

    # ── load model ────────────────────────────────────────────────────────────
    print("Loading vLLM model...", flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=TENSOR_PARALLEL,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        enable_prefix_caching=True,
    )

    sampling_params = SamplingParams(
        n=N_SAMPLES,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    # ── run inference ──────────────────────────────────────────────────────────
    print(f"Generating {N_SAMPLES} samples per reaction for {len(messages_list):,} reactions...", flush=True)
    t0 = time.time()
    outputs = llm.chat(messages_list, sampling_params=sampling_params)
    el = time.time() - t0
    ntok = sum(len(c.token_ids) for o in outputs for c in o.outputs)
    print(f"Inference complete in {el/60:.1f} min  "
          f"({len(messages_list)/el:.3f} prompt/s, {ntok/el:.0f} out-tok/s)", flush=True)

    # ── write output ───────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w") as out_f:
        for record, output in zip(records, outputs):
            responses = [o.text for o in output.outputs]
            row = {
                "row_idx":           record["row_idx"],
                "reaction_id":       record["reaction_id"],
                "starting_material": record["starting_material"],
                "reagent_list":      record["reagent_list"],
                "gold_product":      record["gold_product"],
                "responses":         responses,
                "finish_reasons":    [o.finish_reason for o in output.outputs],
            }
            out_f.write(json.dumps(row) + "\n")
            written += 1
            if written % 5000 == 0:
                print(f"  Written {written:,} / {len(records):,}", flush=True)

    print(f"Done. Wrote {written:,} records to {out_path}", flush=True)


if __name__ == "__main__":
    main()
