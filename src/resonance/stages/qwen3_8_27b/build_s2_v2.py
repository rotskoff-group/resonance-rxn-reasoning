"""Build validation-targeted S2 data with one reaction per example and assistant-only loss.

Ported from run_dir('pipeline_27b')/scripts/build_s2_v2.py. Only its manifest's `inputs`
(train.json and train class labels, with hashes) are read downstream by build_s2_full.
CLI: python -m resonance.stages.qwen3_8_27b.build_s2_v2 --run-root RUN/s2 --source TRAIN_JSON
     --labels TRAIN_LABELS --validation-report REPORT --tokenizer S1_MODEL --out RUN/S2_V2_dataset
"""
import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocate(total, classes):
    quotient, remainder = divmod(total, len(classes))
    return {name: quotient + int(i < remainder) for i, name in enumerate(sorted(classes))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hard-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import Dataset
    from transformers import AutoTokenizer

    run_root = Path(args.run_root)
    output = Path(args.out)
    report = json.loads(Path(args.validation_report).read_text())
    threshold = report["accuracy"]
    weak = sorted(row["class"] for row in report["classes"] if row["accuracy"] < threshold)
    strong = sorted(row["class"] for row in report["classes"] if row["accuracy"] >= threshold)
    assert weak and strong

    source = json.loads(Path(args.source).read_text())
    labels = json.loads(Path(args.labels).read_text())["labels"]
    assert len(source) == len(labels) == 40000
    by_class = defaultdict(list)
    for idx in range(len(source)):
        by_class[labels[str(idx)]["level1"]].append(idx)

    weak_ids = {idx for cls in weak for idx in by_class[cls]}
    replay_total = round(len(weak_ids) * (1 - args.hard_fraction) / args.hard_fraction)
    replay_allocation = allocate(replay_total, strong)
    rng = random.Random(args.seed)
    replay_ids = set()
    for cls in strong:
        candidates = list(by_class[cls])
        rng.shuffle(candidates)
        replay_ids.update(candidates[:replay_allocation[cls]])
    selected_ids = weak_ids | replay_ids
    assert not weak_ids & replay_ids

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tokenizer.eos_token_id
    rows = []
    source_types = Counter()
    class_counts = Counter()
    pass_counts = Counter()
    token_counts = Counter()
    seen = set()
    for shard in (0, 1):
        with (run_root / f"rollouts/{shard}/rollouts.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                idx = row["row_idx"]
                if idx not in selected_ids:
                    continue
                assert idx not in seen
                seen.add(idx)
                prompt_ids = list(row["prompt_ids"])
                usable = []
                for completion in row["completions"]:
                    ids = list(completion["token_ids"])
                    if not ids or ids[-1] != eos:
                        ids.append(eos)
                    if (completion["correct"] and completion["finish_reason"] == "stop"
                            and len(prompt_ids) + len(ids) <= 2048):
                        usable.append((len(ids), ids))
                pass_counts[len(usable)] += 1
                if usable:
                    completion_ids = min(usable, key=lambda item: item[0])[1]
                    source_type = "s1_shortest_correct"
                else:
                    answer = f"</think>\n\n<answer>\n{row['ground_truth']}\n</answer>"
                    completion_ids = tokenizer.encode(answer, add_special_tokens=False)
                    if not completion_ids or completion_ids[-1] != eos:
                        completion_ids.append(eos)
                    assert len(prompt_ids) + len(completion_ids) <= 2048
                    source_type = "gold_answer_only"
                input_ids = prompt_ids + completion_ids
                labels_row = [-100] * len(prompt_ids) + completion_ids
                cls = labels[str(idx)]["level1"]
                rows.append({
                    "input_ids": input_ids,
                    "labels": labels_row,
                    "row_idx": idx,
                    "reaction_id": source[idx]["metadata"]["reaction_id"],
                    "reaction_class": cls,
                    "target_group": "weak" if cls in weak else "replay",
                    "source_type": source_type,
                    "s1_pass_at_8": len(usable),
                    "prompt_tokens": len(prompt_ids),
                    "assistant_tokens": len(completion_ids),
                })
                source_types[source_type] += 1
                class_counts[cls] += 1
                token_counts["prompt"] += len(prompt_ids)
                token_counts["assistant"] += len(completion_ids)
    assert seen == selected_ids, (len(seen), len(selected_ids), sorted(selected_ids - seen)[:10])
    rows.sort(key=lambda row: row["row_idx"])

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    Dataset.from_list(rows).save_to_disk(str(output / "dataset"))
    total_tokens = token_counts["prompt"] + token_counts["assistant"]
    manifest = {
        "name": "S2_V2_dataset",
        "policy": {
            "weak_class_rule": "validation class accuracy < overall validation accuracy",
            "validation_accuracy_threshold": threshold,
            "weak_classes": weak,
            "strong_replay_classes": strong,
            "hard_fraction_requested": args.hard_fraction,
            "one_example_per_reaction": True,
            "correct_rollout_choice": "shortest usable exact-correct S1 rollout",
            "zero_success_fallback": "gold product, answer-only",
            "loss": "assistant tokens only",
            "seed": args.seed,
        },
        "examples": len(rows),
        "weak_examples": len(weak_ids),
        "replay_examples": len(replay_ids),
        "hard_fraction_actual": len(weak_ids) / len(rows),
        "source_types": dict(sorted(source_types.items())),
        "s1_usable_correct_count": dict(sorted(pass_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "tokens": {
            **token_counts,
            "total": total_tokens,
            "assistant_fraction": token_counts["assistant"] / total_tokens,
            "three_epoch_total": 3 * total_tokens,
        },
        "inputs": {
            "source": str(Path(args.source).resolve()),
            "source_sha256": sha256(args.source),
            "labels": str(Path(args.labels).resolve()),
            "labels_sha256": sha256(args.labels),
            "validation_report": str(Path(args.validation_report).resolve()),
            "validation_report_sha256": sha256(args.validation_report),
            "rollout_root": str(run_root.resolve()),
            "tokenizer": str(Path(args.tokenizer).resolve()),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
