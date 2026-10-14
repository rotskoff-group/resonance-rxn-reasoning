"""Build the S1 SFT dataset from the Kimi K3 no-ID teacher traces.

Ported from run_dir('traces')/scripts/prep_data.py. The published S1 used arm 0a:

    python -m resonance.stages.qwen3_8b.s1_prep --arm 0a \
        --input <traces_stage1a_no_reaction_id.jsonl> --output <sft_dataset_0a_no_reaction_id> \
        --min-records 40000 --max-length 2048

    --arm 0a   assistant think block = stage-1a trace (the published arm)
    --arm 0b   assistant think block = stage-1b think_body (reformatted; not used in the paper)

Relative to the earlier S1 prep, `Reaction ID` is dropped from the user turn, and the user
turn matches the rollout/benchmark wording so S1 and S2 train on one input format
(--legacy-user-template reproduces the old asymmetry).

Guards: --min-records aborts rather than training on a half-finished generation run; the p99
length report warns when examples exceed --max-length, since silent truncation would cut
<answer> off the end of the label. `s1_check_lengths` then checks every example.
"""
import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from resonance.paths import run_dir

RUN = run_dir("traces")
MODEL_NAME = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = (
    "You are an expert organic chemist specializing in the forward prediction of organic reactions. "
    "Your task is to predict the major product of organic reactions given starting materials and reagents. "
    "Return the answer between <answer> and </answer> tags."
)

# Matches the first lines of the rollout user template (sample_rollouts.py).
USER_TEMPLATE = (
    "Here are the SMILES of the starting material and reagent(s).\n"
    "Starting material: {starting_material}\n"
    "Reagents: {reagents}"
)

LEGACY_USER_TEMPLATE = (
    "Reaction ID: {reaction_id}\n\n"
    "Starting Material: {starting_material}\n"
    "Reagents: {reagents}\n"
    "Reaction Conditions: {conditions}"
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["0a", "0b"])
    ap.add_argument("--input", default=None, help="Defaults to the arm's stage output.")
    ap.add_argument("--output", default=None)
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--max-length", type=int, default=1024,
                    help="Must match training_args.max_length in the SFT config.")
    ap.add_argument("--min-records", type=int, default=1000)
    ap.add_argument("--legacy-user-template", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input) if args.input else (
        RUN / ("outputs/traces_stage1a.jsonl" if args.arm == "0a"
               else "outputs/traces_stage1b.jsonl"))
    out_dir = Path(args.output) if args.output else RUN / f"outputs/sft_dataset_{args.arm}"

    if not in_path.exists():
        print(f"ERROR: {in_path} does not exist.")
        return 2

    records = load_jsonl(in_path)
    print(f"Loaded {len(records):,} records from {in_path}")

    tmpl = LEGACY_USER_TEMPLATE if args.legacy_user_template else USER_TEMPLATE
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    texts, skipped, rejected = [], 0, 0
    for r in records:
        product = (r.get("product") or "").strip()
        if args.arm == "0a":
            think = (r.get("trace") or "").strip()
        else:
            # Only accepted rewrites are trainable: the rest failed answer-preservation or
            # the leak scan and are kept in the file purely so the rates stay measurable.
            if not (r.get("verify") or {}).get("accept"):
                rejected += 1
                continue
            think = (r.get("think_body") or "").strip()

        if not think or not product:
            skipped += 1
            continue

        user_msg = tmpl.format(
            reaction_id=r.get("reaction_id", ""),
            starting_material=r.get("starting_material", ""),
            reagents=r.get("reagents", ""),
            conditions=r.get("conditions", ""),
        )
        asst_msg = f"<think>\n{think}\n</think>\n<answer>\n{product}\n</answer>"
        text = tokenizer.apply_chat_template(
            [{"role": "system",    "content": SYSTEM_PROMPT},
             {"role": "user",      "content": user_msg},
             {"role": "assistant", "content": asst_msg}],
            tokenize=False, add_generation_prompt=False)
        texts.append(text)

    print(f"Built {len(texts):,} examples  (skipped {skipped:,}, "
          f"rejected-unaccepted {rejected:,})")

    if len(texts) < args.min_records:
        print(f"ERROR: only {len(texts)} examples < --min-records {args.min_records}. "
              f"Refusing to write -- this looks like a half-finished generation run.")
        return 1

    # Length report. p99 over the max_length is the failure that silently truncates the label.
    sample = texts[:2000]
    lens = sorted(len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in sample)
    p = lambda f: lens[min(len(lens) - 1, int(f * len(lens)))]  # noqa: E731
    print(f"Token length over first {len(sample):,}: "
          f"p50={p(0.50)} p90={p(0.90)} p99={p(0.99)} max={lens[-1]}")
    if p(0.99) > args.max_length:
        print(f"  *** WARNING: p99={p(0.99)} exceeds --max-length {args.max_length}. "
              f"Raise training_args.max_length or <answer> will be truncated away for ~1% "
              f"of examples, which trains the model to never close the tag. ***")
    else:
        print(f"  p99 fits within max_length={args.max_length}")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict({"text": texts}).save_to_disk(str(out_dir))
    print(f"Saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
