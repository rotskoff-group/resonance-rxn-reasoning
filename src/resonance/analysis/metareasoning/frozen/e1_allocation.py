#!/usr/bin/env python3
"""E1: accuracy vs sampling budget, uniform and selectively allocated. No GPU.

Nine T=0.6 draws exist per item (turn 1 plus the 8 in the C5_fresh pool), so allocation
policies can be simulated offline.

The distinction the simulation has to keep: `pass@k` needs an oracle to pick the right
draw, so it is a ceiling and not a policy. What a policy can actually do is majority vote,
or use a verifier score to decide where to spend. Both are reported against pass@k.

Budget is mean draws per item, so a selective policy is only interesting if it beats the
uniform curve at the SAME budget.

  python scripts/e1_allocation.py --arm q8_s3
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_selfcorrect import canon, classify

RUN = Path(__file__).resolve().parents[1]


# Statuses that are knowable to be wrong WITHOUT the ground truth: an unparseable SMILES,
# no answer at all, or the starting material handed back. A free verifier.
RULE_WRONG = {"no_answer", "invalid", "copied_sm"}


def draws_for(arm, tag, data):
    """eval_idx -> {'preds', 'ok', 'st', 'rule_wrong', 'gt'} with 9 draws each."""
    fresh = RUN / f"outputs/fresh_t06/{tag}/C5_fresh.jsonl"
    if not fresh.is_file():
        fresh = RUN / f"outputs_modal/t06_{arm}/C5_fresh.jsonl"
    turn1 = {r["eval_idx"]: r["turn1"][tag] for r in data if tag in r["turn1"]}
    out = {}
    for line in open(fresh):
        r = json.loads(line)
        i = r["eval_idx"]
        if i not in turn1:
            continue
        base = {"answer": r["answer"], "answer_type": r["answer_type"],
                "task_prompt": r["task_prompt"],
                "think_prefilled": r.get("think_prefilled", False)}
        texts = [turn1[i]["response"]] + list(r.get("responses") or [])
        preds, ok, sts = [], [], []
        for t in texts:
            st, _, pred, _ = classify({**base, "response": t})
            preds.append(pred)
            ok.append(st == "correct")
            sts.append(st)
        out[i] = {"preds": preds, "ok": ok, "st": sts,
                  "rule_wrong": [x in RULE_WRONG for x in sts],
                  "gt": canon(r["answer"])}
    return out


def majority(preds):
    """Most common non-None prediction; ties go to the earliest draw. None if all None."""
    seen = [p for p in preds if p]
    if not seen:
        return None
    c = collections.Counter(seen)
    top = max(c.values())
    for p in seen:                      # earliest draw among the tied winners
        if c[p] == top:
            return p
    return None


def majority_filtered(d, k):
    """Majority over the first k draws, dropping rule-detectable failures first."""
    keep = [p for p, bad in zip(d["preds"][:k], d["rule_wrong"][:k]) if not bad]
    return majority(keep) if keep else majority(d["preds"][:k])


def verifier_scores(arm):
    p = RUN / f"outputs/t06_{arm}/C3_verify.jsonl"
    if not p.is_file():
        return {}
    out = {}
    for line in open(p):
        r = json.loads(line)
        lp = (r.get("verify_logprobs") or {}).get("p_true")
        if lp is not None:
            out[r["eval_idx"]] = lp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="q8_s3")
    ap.add_argument("--turn1-tag", default="")
    ap.add_argument("--data", default=str(RUN / "data/eval_5000_t06.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    tag = args.turn1_tag or args.arm

    data = json.load(open(args.data))
    D = draws_for(args.arm, tag, data)
    idx = sorted(D)
    n = len(idx)
    ver = verifier_scores(args.arm)
    rng = random.Random(args.seed)
    rep = {"arm": args.arm, "turn1_tag": tag, "n": n, "uniform": [], "selective": []}

    def acc(fn):
        return 100.0 * sum(fn(D[i]) for i in idx) / n

    print(f"arm {args.arm}  n={n}  draws/item=9  (turn 1 + 8 fresh, all T=0.6)\n")
    print("UNIFORM — every item gets the same budget")
    print(f"{'policy':<18} {'budget':>7} {'acc%':>7}")
    for k in (1, 3, 5, 7, 9):
        a_maj = acc(lambda d, k=k: (m := majority(d["preds"][:k])) is not None
                    and m == d["gt"])
        a_pass = acc(lambda d, k=k: any(d["ok"][:k]))
        print(f"{'majority@'+str(k):<18} {k:>7} {a_maj:>7.2f}")
        rep["uniform"].append({"policy": f"majority@{k}", "budget": k,
                               "accuracy": round(a_maj, 2)})
        if k > 1:
            print(f"{'  (pass@'+str(k)+' ceiling)':<18} {k:>7} {a_pass:>7.2f}")
            rep["uniform"].append({"policy": f"pass@{k}", "budget": k,
                                   "accuracy": round(a_pass, 2), "ceiling": True})

    # selective: keep turn 1 unless flagged; on flagged spend 2 more and take majority@3
    def selective(flagged):
        hit = 0
        for i in idx:
            d = D[i]
            if flagged(i, d):
                m = majority(d["preds"][:3])
                hit += (m is not None and m == d["gt"])
            else:
                hit += d["ok"][0]
        return 100.0 * hit / n

    sels = {
        "oracle (turn 1 wrong)": lambda i, d: not d["ok"][0],
        "self-disagree (draw2)": lambda i, d: d["preds"][1] != d["preds"][0],
        "verifier p_true<0.5": lambda i, d: ver.get(i, 1.0) < 0.5,
        "verifier p_true<0.2": lambda i, d: ver.get(i, 1.0) < 0.2,
    }
    print("\nSELECTIVE — turn 1 everywhere; flagged items get majority@3")
    print(f"{'selector':<24} {'flagged%':>9} {'budget':>7} {'acc%':>7} {'vs uniform':>11}")
    # uniform curve for interpolation
    ucur = [(u["budget"], u["accuracy"]) for u in rep["uniform"] if not u.get("ceiling")]
    def uniform_at(b):
        for (b0, a0), (b1, a1) in zip(ucur, ucur[1:]):
            if b0 <= b <= b1:
                return a0 + (a1 - a0) * (b - b0) / (b1 - b0)
        return ucur[-1][1]
    for name, fn in sels.items():
        f = sum(1 for i in idx if fn(i, D[i])) / n
        if name.startswith("verifier") and not ver:
            continue
        a = selective(fn)
        b = 1 + 2 * f
        print(f"{name:<24} {100*f:>9.1f} {b:>7.2f} {a:>7.2f} {a-uniform_at(b):>+11.2f}")
        rep["selective"].append({"selector": name, "flagged_pct": round(100 * f, 2),
                                 "budget": round(b, 3), "accuracy": round(a, 2),
                                 "vs_uniform_pp": round(a - uniform_at(b), 2)})
    # random selector at the oracle's flag rate, as the null
    f0 = sum(1 for i in idx if not D[i]["ok"][0]) / n
    a = selective(lambda i, d: rng.random() < f0)
    b = 1 + 2 * f0
    print(f"{'random at same rate':<24} {100*f0:>9.1f} {b:>7.2f} {a:>7.2f} "
          f"{a-uniform_at(b):>+11.2f}")
    rep["selective"].append({"selector": "random", "flagged_pct": round(100 * f0, 2),
                             "budget": round(b, 3), "accuracy": round(a, 2),
                             "vs_uniform_pp": round(a - uniform_at(b), 2)})

    # ---- test 1: how much of the answer pool is rule-detectably wrong
    npool = 9
    rw = [sum(D[i]["rule_wrong"][j] for i in idx) / n for j in range(npool)]
    t1_rule = 100 * rw[0]
    print(f"\nRULE-DETECTABLE FAILURES (no_answer / invalid / copied_sm)")
    print(f"  turn 1: {t1_rule:.2f}% of items")
    print(f"  pooled over all 9 draws: {100*sum(rw)/npool:.2f}% of draws")
    st_counts = collections.Counter(s for i in idx for s in D[i]["st"])
    tot_draws = sum(st_counts.values())
    print("  draw status mix: " + "  ".join(
        f"{k} {100*v/tot_draws:.2f}%" for k, v in st_counts.most_common()))
    rep["rule_wrong_turn1_pct"] = round(t1_rule, 2)
    rep["rule_wrong_all_draws_pct"] = round(100 * sum(rw) / npool, 2)
    rep["draw_status_pct"] = {k: round(100 * v / tot_draws, 2)
                              for k, v in st_counts.most_common()}

    # decomposition of the pass/majority gap at k=9
    miss = [i for i in idx if any(D[i]["ok"]) and
            not ((m := majority(D[i]["preds"])) is not None and m == D[i]["gt"])]
    fixed = [i for i in miss if (m := majority_filtered(D[i], 9)) is not None
             and m == D[i]["gt"]]
    print(f"\n  recoverable-but-missed by majority@9: {len(miss)} items "
          f"({100*len(miss)/n:.2f} pp of the gap)")
    print(f"  of those, fixed by dropping rule-wrong draws: {len(fixed)} "
          f"({100*len(fixed)/max(1,len(miss)):.1f}%)")
    rep["gap_items"] = len(miss)
    rep["gap_items_fixed_by_rules"] = len(fixed)

    # ---- test 2: validity-filtered majority curve
    print("\nVALIDITY-FILTERED MAJORITY (drop rule-wrong draws before voting)")
    print(f"{'policy':<22} {'budget':>7} {'acc%':>7} {'vs plain':>9}")
    plain = {u["budget"]: u["accuracy"] for u in rep["uniform"] if not u.get("ceiling")}
    rep["filtered"] = []
    for k in (1, 3, 5, 7, 9):
        a = acc(lambda d, k=k: (m := majority_filtered(d, k)) is not None and m == d["gt"])
        print(f"{'maj@'+str(k)+' filtered':<22} {k:>7} {a:>7.2f} {a-plain[k]:>+9.2f}")
        rep["filtered"].append({"policy": f"majority_filtered@{k}", "budget": k,
                                "accuracy": round(a, 2),
                                "vs_plain_pp": round(a - plain[k], 2)})

    # ---- test 2b: rule failure as a selector, and combined with disagreement
    print("\nSELECTIVE with the free chemical check as the selector")
    print(f"{'selector':<34} {'flagged%':>9} {'budget':>7} {'acc%':>7} {'vs uniform':>11}")
    extra = {
        "turn-1 fails a rule check": lambda i, d: d["rule_wrong"][0],
        "rule check OR draw-2 disagree": lambda i, d: (d["rule_wrong"][0]
                                                       or d["preds"][1] != d["preds"][0]),
    }
    for name, fn in extra.items():
        f = sum(1 for i in idx if fn(i, D[i])) / n
        # on flagged items vote over 3 draws, filtered
        hit = 0
        for i in idx:
            d = D[i]
            if fn(i, d):
                m = majority_filtered(d, 3)
                hit += (m is not None and m == d["gt"])
            else:
                hit += d["ok"][0]
        a = 100.0 * hit / n
        b = 1 + 2 * f
        print(f"{name:<34} {100*f:>9.1f} {b:>7.2f} {a:>7.2f} {a-uniform_at(b):>+11.2f}")
        rep["selective"].append({"selector": name, "flagged_pct": round(100 * f, 2),
                                 "budget": round(b, 3), "accuracy": round(a, 2),
                                 "vs_uniform_pp": round(a - uniform_at(b), 2)})

    out = Path(args.out) if args.out else RUN / f"outputs/e1_allocation_{args.arm}.json"
    json.dump(rep, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
