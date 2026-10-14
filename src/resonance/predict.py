"""Predict reaction products with a Resonance model, using the paper's evaluation prompt.

    python -m resonance.predict --model anaghaa/Resonance-8B-S3 \
        --starting-material "Fc1cc(I)ccn1" --reagents "OB(O)c1ccc2ccccc2c1"
    python -m resonance.predict --model anaghaa/Resonance-27B-S3 --input reactions.jsonl --out predictions.jsonl

`--input` is a JSON-lines file of {"starting_material": ..., "reagents": [...]}; an optional
"product" field is scored. Reagents are a list of SMILES; a string is split on "." as in
USPTO-50k, where the reactants after the first are joined with ".".

The prompt and decoding are the paper's (appendix "Evaluation prompt" and Table
appendix table "Sampling settings"): the short system prompt for 8B models and the long one for 27B models
(`--family`, default from the model name), thinking enabled, T 0.6, top-p 0.95, top-k 20,
16,384 new tokens. The prediction is the last <answer> block, canonicalised with RDKit (the
paper's scorer). Needs vLLM and a GPU (8B: one 80 GB card; 27B: two).
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from resonance.data.parsers.benchmark.forward_rxn_parser import ForwardRxnParser
from resonance.data.prompts.benchmark.forward_rxn import ForwardRxnPrompt
from resonance.evaluation.extractors import extract_forward_rxn, match_forward_rxn
from resonance.stages.common import canonical

PROMPTS_27B = Path(__file__).resolve().parent / "configs/paper/qwen3_8_27b/prompts.json"


def messages(starting_material, reagents, family="8b"):
    """[system, user] messages exactly as the paper's USPTO-50k evaluation built them.

    8b: the short system prompt (ForwardRxnPrompt "none"), reagents joined with ", ".
    27b: the long system prompt and the same user turn, from configs/paper/qwen3_8_27b/
    prompts.json (leading/trailing whitespace removed), reagents joined with ".".
    """
    if family == "27b":
        prompts = json.loads(PROMPTS_27B.read_text())
        user = prompts["user"].format(starting_material=starting_material,
                                      reagent_list=".".join(reagents) or "No reagents").strip()
        return [{"role": "system", "content": prompts["system"]}, {"role": "user", "content": user}]
    prompt = ForwardRxnPrompt()
    task = {"sm": starting_material, "rgts": reagents}
    user = ForwardRxnParser(data_path="").parse_reaction(prompt.get_task_template(representation="smiles"), task)
    return [{"role": "system", "content": prompt.get_system_prompt("none")},
            {"role": "user", "content": user}]


def main(argv=None):
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="Hub id or local directory")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--starting-material")
    src.add_argument("--input", help="JSON-lines file of reactions")
    ap.add_argument("--reagents", default="", help="reagent SMILES, several joined with '.' (default: none)")
    ap.add_argument("--out", help="write one JSON line per reaction here (default: stdout)")
    ap.add_argument("--family", choices=["8b", "27b"], help="prompt family (default: 27b if the name says so)")
    ap.add_argument("--tp", type=int, default=None, help="tensor parallel size (default: 2 for 27B, 1 otherwise)")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if args.input:
        rows = [json.loads(line) for line in open(args.input) if line.strip()]
    else:
        rows = [{"starting_material": args.starting_material, "reagents": args.reagents}]
    for r in rows:
        rg = r.get("reagents") or []
        r["reagents"] = [x for x in rg.split(".") if x] if isinstance(rg, str) else list(rg)

    # vLLM logs INFO to stdout; keep stdout for the JSON lines unless the user chose a level.
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    from vllm import LLM, SamplingParams
    family = args.family or ("27b" if "27b" in args.model.lower() else "8b")
    print(f"prompt family: {family}" + ("" if args.family else " (from the model name; set --family to override)"),
          file=sys.stderr)
    big = family == "27b"
    # Qwen3.8-27B is a vision-language checkpoint; the paper used it text-only.
    extra: dict[str, Any] = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if big else {}
    llm = LLM(model=args.model, tensor_parallel_size=args.tp or (2 if big else 1),
              max_model_len=20480 if big else 16384, seed=args.seed, **extra)
    params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.max_tokens, seed=args.seed)
    outputs = llm.chat([messages(r["starting_material"], r["reagents"], family) for r in rows], params,
                       chat_template_kwargs={"enable_thinking": True})

    out = open(args.out, "w") if args.out else sys.stdout
    scored = []
    for r, o in zip(rows, outputs, strict=True):
        text = o.outputs[0].text
        answer = extract_forward_rxn(text)
        # answer: the last <answer> block as written; prediction: its canonical SMILES
        # (None when the model gave no answer or an invalid SMILES).
        rec = {**r, "answer": answer, "prediction": canonical(answer) if answer else None, "response": text}
        if r.get("product"):
            rec["correct"] = bool(answer) and match_forward_rxn(answer, r["product"], "smiles")
            scored.append(rec["correct"])
        out.write(json.dumps(rec) + "\n")
    if args.out:
        out.close()
    if scored:
        print(f"Top-1 exact match: {sum(scored)}/{len(scored)} = {100 * sum(scored) / len(scored):.1f}%",
              file=sys.stderr)


if __name__ == "__main__":
    main()
