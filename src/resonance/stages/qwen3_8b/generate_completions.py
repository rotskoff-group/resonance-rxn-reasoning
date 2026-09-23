"""Generate G=16 completions from the SFT model on all 40k USPTO-50k training prompts.

Ported from run_dir('s3_8b')/scripts/generate_completions.py. The published run used eight
single-GPU shards (launchers/qwen3_8b/s3_generate.sbatch):

    python -m resonance.stages.qwen3_8b.generate_completions --model <S2 checkpoint-2229> \
        --train_json <train.json> --out <completions.shard{i}-of-8.jsonl> \
        --g 16 --temperature 0.7 --max_tokens 8192 --shard i --nshards 8

Prompts use the "rules" system prompt, while every benchmark ran with "none" (see README.md).

Output: completions.jsonl — one record per reaction with 16 sampled completions.
Schema: {reaction_id, prompt_messages, completions: [str x16], ground_truth}
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
from resonance.stages.qwen3_8b._common import S2_MODEL, TRAIN_JSON as _TRAIN_JSON
from resonance.paths import run_dir

SFT_MODEL = str(S2_MODEL)
TRAIN_JSON = str(_TRAIN_JSON)
OUT_PATH   = str(run_dir("s3_8b") / "outputs/completions.jsonl")

# ── Prompt building ───────────────────────────────────────────────────────────
from resonance.data.parsers.benchmark.uspto_50k import ReactionPredictionUSPTO
from resonance.data.prompts.benchmark.forward_rxn import ForwardRxnPrompt


def build_prompts(train_json_path: str) -> tuple[list[dict], list[str], list[str]]:
    """Returns (messages_list, ground_truths, reaction_ids)."""
    with open(train_json_path) as f:
        raw = json.load(f)

    parser = ReactionPredictionUSPTO(train_json_path)
    reactions = parser.load_reactions(representation="smiles")

    prompt_obj = ForwardRxnPrompt()
    system_content = prompt_obj.get_system_prompt(system_prompt_option="rules").strip()
    task_template = prompt_obj.get_task_template(representation="smiles")

    messages_list = []
    ground_truths = []
    reaction_ids = []

    for rxn, raw_rxn in zip(reactions, raw):
        user_content = parser.parse_reaction(task_template, rxn).strip()
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
        ]
        messages_list.append(messages)
        ground_truths.append(rxn["true_product_smiles"])
        reaction_ids.append(raw_rxn["metadata"]["reaction_id"])

    return messages_list, ground_truths, reaction_ids


def run_vllm(
    model_path: str,
    formatted_prompts: list[str],
    g: int,
    temperature: float,
    max_tokens: int,
) -> list[list[str]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=16384,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=g)
    outputs = llm.generate(formatted_prompts, sampling)
    return [[o.text for o in out.outputs] for out in outputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default=SFT_MODEL)
    parser.add_argument("--train_json",  default=TRAIN_JSON)
    parser.add_argument("--out",         default=OUT_PATH)
    parser.add_argument("--g",           type=int,   default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens",  type=int,   default=8192)
    parser.add_argument("--shard",       type=int,   default=0)
    parser.add_argument("--nshards",     type=int,   default=1)
    args = parser.parse_args()
    assert 0 <= args.shard < args.nshards

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert not out_path.exists(), f"Output already exists: {out_path}"
    model_dir = Path(args.model)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    assert all((model_dir / name).is_file() for name in set(index["weight_map"].values()))
    assert json.loads((model_dir / "config.json").read_text())["model_type"] == "qwen3"

    print("Building prompts from USPTO-50k train split...")
    messages_list, ground_truths, reaction_ids = build_prompts(args.train_json)
    print(f"  {len(messages_list)} reactions loaded")
    per_shard = (len(messages_list) + args.nshards - 1) // args.nshards
    lo = args.shard * per_shard
    hi = min(len(messages_list), lo + per_shard)
    messages_list = messages_list[lo:hi]
    ground_truths = ground_truths[lo:hi]
    reaction_ids = reaction_ids[lo:hi]
    print(f"  shard {args.shard}/{args.nshards}: rows [{lo},{hi})", flush=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Applying chat template...")
    formatted = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]

    print(f"Running vLLM (G={args.g}, T={args.temperature}, gpus={torch.cuda.device_count()})...")
    all_completions = run_vllm(args.model, formatted, args.g, args.temperature, args.max_tokens)

    print(f"Writing → {out_path}")
    written = 0
    with open(out_path, "w") as f:
        for rid, msgs, completions, gt in zip(reaction_ids, messages_list, all_completions, ground_truths):
            record = {
                "row_idx":       lo + written,
                "reaction_id":    rid,
                "prompt_messages": msgs,
                "completions":    completions,
                "ground_truth":   gt,
            }
            f.write(json.dumps(record) + "\n")
            written += 1
            if written % 1000 == 0:
                print(f"  Written {written}/{len(messages_list)}")

    print(f"Done. {written} records → {out_path}")


if __name__ == "__main__":
    main()
