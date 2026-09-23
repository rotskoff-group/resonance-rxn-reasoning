"""Mine (chosen, rejected) DPO pairs: self-play positives + hard negatives.

Design:
  - chosen:   one of the model's own correct completions (same format/length as
              rejected — avoids the oracle format-collapse failure mode).
  - rejected: the wrong completion with HIGHEST Tanimoto to ground truth
              (hardest plausible wrong answer → sharpest DPO gradient).

Coverage: prompts with 1 ≤ n_correct ≤ 15 (mixed-outcome only). Expects ~15k
pairs from 40k reactions at 44% accuracy.

Input:  completions.jsonl  — produced by generate_completions.py
Output: dpo_pairs.jsonl    — {prompt, chosen, rejected} in TRL conversational format

Ported from run_dir('s3_8b')/scripts/mine_dpo_pairs.py.

    python -m resonance.stages.qwen3_8b.mine_dpo_pairs --input <completions.jsonl> \
        --output <dpo_pairs.jsonl> --seed 42
"""

import argparse
import json
import random
import re
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from resonance.paths import run_dir

IN_PATH  = str(run_dir("s3_8b") / "outputs/completions.jsonl")
OUT_PATH = str(run_dir("s3_8b") / "outputs/dpo_pairs.jsonl")

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ── Chemistry helpers ─────────────────────────────────────────────────────────

def extract_smiles(text: str) -> str:
    m = re.search(r"</think>\s*(.*)", text, flags=re.DOTALL)
    if m:
        after = m.group(1).strip()
        am = re.search(r"<answer>(.*?)</answer>", after, re.DOTALL)
        if am:
            return am.group(1).strip()
        pm = re.search(r"<product>(.*?)</product>", after, re.DOTALL | re.IGNORECASE)
        if pm:
            return pm.group(1).strip()
        return after
    if "<think>" in text:
        return ""
    am = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if am:
        return am.group(1).strip()
    return text.strip()


def canonical(smiles: str) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def is_exact_match(pred: str, gt: str) -> bool:
    c_pred = canonical(pred)
    c_gt   = canonical(gt)
    return c_pred is not None and c_gt is not None and c_pred == c_gt


def tanimoto(smiles_pred: str, smiles_gt: str) -> float:
    m_pred = Chem.MolFromSmiles(smiles_pred) if smiles_pred else None
    m_gt   = Chem.MolFromSmiles(smiles_gt)   if smiles_gt  else None
    if m_pred is None or m_gt is None:
        return 0.0
    fp_pred = _MORGAN.GetFingerprint(m_pred)
    fp_gt   = _MORGAN.GetFingerprint(m_gt)
    return float(DataStructs.TanimotoSimilarity(fp_pred, fp_gt))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=IN_PATH)
    ap.add_argument("--output", default=OUT_PATH)
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    in_path  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = skipped_no_correct = skipped_all_correct = written = 0

    print(f"Reading {in_path} ...")
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1

            gt          = record["ground_truth"]
            completions = record["completions"]
            prompt_msgs = record["prompt_messages"]

            # Collect wrong completions with their Tanimoto to GT
            wrong_with_tan: list[tuple[float, str]] = []
            for comp in completions:
                pred = extract_smiles(comp)
                if not is_exact_match(pred, gt):
                    tan = tanimoto(pred, gt)
                    wrong_with_tan.append((tan, comp))

            # Bucket completions
            correct = []
            for comp in completions:
                pred = extract_smiles(comp)
                if is_exact_match(pred, gt):
                    correct.append(comp)

            # Skip all-wrong (no chosen) or all-correct (no rejected)
            if not correct:
                skipped_no_correct += 1
                continue
            if not wrong_with_tan:
                skipped_all_correct += 1
                continue

            # Self-play chosen — random correct completion (same format as rejected)
            chosen_text = rng.choice(correct)

            # Hard negative — highest Tanimoto wrong completion
            wrong_with_tan.sort(key=lambda x: x[0], reverse=True)
            hard_rejected_text = wrong_with_tan[0][1]

            pair = {
                "prompt":   prompt_msgs,
                "chosen":   [{"role": "assistant", "content": chosen_text}],
                "rejected": [{"role": "assistant", "content": hard_rejected_text}],
            }
            fout.write(json.dumps(pair) + "\n")
            written += 1

            if written % 1000 == 0:
                print(f"  Pairs written: {written}  (processed {total})")

    print(f"\n=== Mining complete ===")
    print(f"  Total records:             {total}")
    print(f"  Skipped (no correct):      {skipped_no_correct}  "
          f"({100*skipped_no_correct/max(total,1):.1f}%)")
    print(f"  Skipped (all correct):     {skipped_all_correct}  "
          f"({100*skipped_all_correct/max(total,1):.1f}%)")
    print(f"  DPO pairs written:         {written}  "
          f"({100*written/max(total,1):.1f}%)")
    print(f"  Self-play chosen:          random correct model completion")
    print(f"  Hard negative:             highest-Tanimoto wrong completion")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
