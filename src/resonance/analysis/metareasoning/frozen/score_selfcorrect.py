#!/usr/bin/env python3
"""Scorer for the self-correction conditions. Forked from
metareasoning_pilot/scripts/score.py, with ONE deliberate behavioural change.

WHY THE FORK. score.py declares `no_answer` whenever a response opened `<think>` and
never closed it, BEFORE attempting extraction:

    opened = "<think>" in resp or "</think>" in resp
    if opened and "</think>" not in resp:
        return "no_answer", None, None

Measured on bench_s1_rules (2026-09-10): 3/5000 records open `<think>`, never close it,
and still emit a complete and CORRECT `<answer>`. score.py scores them no_answer,
score_bench.py scores them correct -- which is the whole 29.46 vs 29.52 gap.

Three records is immaterial for a headline accuracy. It is NOT immaterial here. Turn-2
revision prompts are off-distribution by construction for these single-turn SFT
checkpoints, so malformed think blocks are expected to be MORE common in exactly the
conditions being compared against turn 1. A guard that discards an answer the model
actually produced would show up as "review destroys accuracy" -- a format artifact
wearing the costume of the result. So:

  * extraction is attempted FIRST, and an extractable answer is always scored;
  * an unclosed think block is recorded in a separate `think_unclosed` FLAG rather than
    as a status, so it can be reported and conditioned on instead of silently dropped;
  * `no_answer` now means what it says: nothing extractable.

Statuses are otherwise score.py's, including `copied_sm` -- a revision that degenerates
into echoing the starting material must not be pooled with ordinary misses.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from resonance.evaluation.extractors import TASK_PARSERS

RDLogger.DisableLog("rdApp.*")

EXTRACT, MATCH = TASK_PARSERS["forward_rxn"]
STATUSES = ("correct", "wrong", "copied_sm", "invalid", "no_answer")


def canon(smi: Optional[str]) -> Optional[str]:
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    return None if m is None else Chem.MolToSmiles(m)


def _fp(smi: Optional[str]):
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    return None if m is None else AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def classify(rec: Dict[str, Any]) -> Tuple[str, Optional[float], Optional[str], bool]:
    """(status, tanimoto, canonical_prediction, think_unclosed)."""
    resp = rec.get("response", "") or ""
    gt = rec.get("answer")

    # A PREFILLED family (Qwen3-*-Thinking-2507 and its derivatives) receives the opening
    # <think> in the PROMPT, so it never appears in the response. Both the flag below and
    # extract_forward_rxn's truncation guard test `"<think>" in text`, so on those models
    # the guard silently never fires: a response truncated mid-think falls through to the
    # extractor's full-response fallback and comes back as the raw reasoning prose, which
    # then scores `invalid` -- a chemistry attempt -- instead of `no_answer`.
    #
    # Measured on t30_armA/Cn_restate (2026-09-10): 0/2400 responses contain a literal
    # <think>, 807 hit the 4,096-token cap, and all 807 were scored `invalid`. That inverts
    # the format/chemical split in RESULTS.md 4a, which files `invalid` as chemical.
    prefilled = bool(rec.get("think_prefilled"))
    opened = ("<think>" in resp) or prefilled
    think_unclosed = opened and ("</think>" not in resp)

    pred = EXTRACT(resp)
    if think_unclosed and prefilled:
        # EXTRACT cannot see the prefill, so override it here rather than forking the
        # shared extractor. An explicit answer tag still wins: a model can close the answer
        # without closing the think block.
        if not re.search(r"<answer>|<product>|<forward_reaction>|\\boxed\{", resp):
            return "no_answer", None, None, True
    if pred is None:
        return "no_answer", None, None, think_unclosed

    pred_c = canon(pred)
    if pred_c is None:
        return "invalid", None, None, think_unclosed

    gt_c = canon(gt)
    tan = None
    if gt_c is not None:
        a, b = _fp(pred_c), _fp(gt_c)
        if a is not None and b is not None:
            tan = DataStructs.TanimotoSimilarity(a, b)

    if gt is not None and MATCH(pred, gt, rec.get("answer_type")):
        return "correct", tan, pred_c, think_unclosed

    sm = re.search(r"Starting material: (\S+)", rec.get("task_prompt", "") or "")
    if sm and canon(sm.group(1)) == pred_c:
        return "copied_sm", tan, pred_c, think_unclosed

    return "wrong", tan, pred_c, think_unclosed


def score_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        status, tan, pred_c, unclosed = classify(r)
        d = dict(r)
        d.update(status=status, correct=(status == "correct"), tanimoto=tan,
                 pred_canon=pred_c, think_unclosed=unclosed)
        out.append(d)
    return out


def summarize(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(scored)
    if n == 0:
        return {"n": 0}
    counts = {s: 0 for s in STATUSES}
    for r in scored:
        counts[r["status"]] += 1
    tans = [r["tanimoto"] for r in scored if r["tanimoto"] is not None]
    return {
        "n": n,
        "correct": counts["correct"],
        "accuracy": 100.0 * counts["correct"] / n,
        "status_counts": counts,
        "parse_rate": 100.0 * (n - counts["no_answer"]) / n,
        "think_unclosed_rate": 100.0 * sum(r["think_unclosed"] for r in scored) / n,
        "mean_tanimoto": (sum(tans) / len(tans)) if tans else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="results.jsonl paths")
    ap.add_argument("--out", default="", help="write a {label: summary} json here")
    args = ap.parse_args()

    report = {}
    for path in args.results:
        recs = [json.loads(l) for l in open(path)]
        s = summarize(score_records(recs))
        label = Path(path).parent.name if Path(path).name == "results.jsonl" else Path(path).stem
        report[label] = {"source": path, **s}
        print(f"{label:28s} n={s['n']:5d} acc={s['accuracy']:6.2f}%  "
              f"parse={s['parse_rate']:6.2f}%  unclosed={s['think_unclosed_rate']:5.2f}%  "
              f"{s['status_counts']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
