#!/usr/bin/env python3
"""Six-cell transition matrix per arm x condition x reaction class, plus Branch 4.

Cells, from (turn-1 correct?, turn-2 correct?, answer changed?):
    wrong   -> correct              improvement
    wrong   -> wrong,   changed     churn
    wrong   -> wrong,   unchanged   failed_correction
    correct -> correct, unchanged   stable
    correct -> correct, changed     unstable      (raw-string change; see below)
    correct -> wrong                regression

Every rate is printed beside its raw count and normalised by the population named in its
own column, because arms start with different numbers of wrong answers.

Four things this handles that a naive version gets wrong (README.md has the evidence):
  1. `answer_change_rate` counts any change of the predicted molecule. The predecessor's
     `change_rate_pct` is (b+c)/n, a correctness-FLIP rate, and is 2.8-6.3x smaller.
     Both are available; never compare one against the other.
  2. `unstable` keys off the RAW emitted string, not the canonical one. Two correct
     answers have identical canonical SMILES by definition, so a canonical comparison
     leaves that cell structurally empty.
  3. Every cell is emitted twice, `pooled` and `chemical`. A flip with no_answer on
     either side is a lost answer, not a revision, and its share is arm-specific --
     pooling makes a lineage table compare format robustness, not judgement.
     Do NOT also exclude `invalid`; on the 8B it is a real baseline property.
  4. The null comes from the arm's OWN C5_fresh. Another model's rollouts import that
     model's per-item difficulty, and the table still looks fine.

Branch 4 metrics come from splitting a feedback condition on turn-1 correctness:
false_correction / persistence / productive_correction / entrenchment, plus the
truth-sensitivity gap and net robustness.

  python scripts/transition_matrix.py --dir outputs/t06_q8_s3 --turn1-tag q8_s3 \
      --null outputs/fresh_t06/q8_s3/C5_fresh.jsonl --out outputs/transition_q8_s3.json
"""
import argparse
import collections
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import conditions as C
from score_selfcorrect import EXTRACT, classify

RUN = Path(__file__).resolve().parents[1]
CELLS = ["improvement", "churn", "failed_correction", "stable", "unstable", "regression"]


# ---------------------------------------------------------------------------- stats ---
def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (round(100 * max(0.0, c - h), 2), round(100 * min(1.0, c + h), 2))


def mcnemar(b, c):
    """b = correct->wrong, c = wrong->correct. Paired: the conditions share items."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0, "test": "none"}
    if n < 25:
        p = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n) * 2
        return {"b": b, "c": c, "p": round(min(1.0, p), 6), "test": "exact"}
    chi = (abs(b - c) - 1) ** 2 / n
    return {"b": b, "c": c, "chi2_cc": round(chi, 4),
            "p": round(math.erfc(math.sqrt(chi / 2)), 6), "test": "chi2_cc"}


def pct(k, n):
    return None if not n else round(100.0 * k / n, 2)


# ----------------------------------------------------------------------- transitions ---
# eval_idx -> corrected reaction_class, set by --labels. Empty means use the class
# recorded in the per-item file (the as-published label).
CLASS_OVERRIDE = {}


def class_of(r):
    """The record's reaction class, after any --labels correction."""
    return (CLASS_OVERRIDE.get(str(r["eval_idx"]))
            or r.get("reaction_class") or "unclassified")


def transitions(path, prefilled_default=False):
    """One row per item: turn-1 and turn-2 status, whether the answer moved, and which
    channel the move belongs to."""
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("skipped"):
            continue
        pre = r.get("think_prefilled", prefilled_default)
        t1 = dict(r, response=r["turn1_response"], think_prefilled=pre)
        s1, _, p1, u1 = classify(t1)
        s2, _, p2, u2 = classify(dict(r, think_prefilled=pre))
        # `changed` = the canonical molecule differs (a chemical revision).
        # `changed_repr` = the emitted string differs, molecule or not. The `unstable` cell
        # needs the latter: two correct answers canonicalise identically.
        changed = (p1 != p2)
        r1, r2 = EXTRACT(t1["response"]) or "", EXTRACT(r.get("response") or "") or ""
        changed_repr = (r1.strip() != r2.strip())
        c1, c2 = s1 == "correct", s2 == "correct"
        if not c1 and c2:
            cell = "improvement"
        elif not c1 and not c2:
            cell = "churn" if changed else "failed_correction"
        elif c1 and c2:
            cell = "unstable" if changed_repr else "stable"
        else:
            cell = "regression"
        rows.append({
            "eval_idx": r["eval_idx"], "reaction_class": class_of(r),
            "s1": s1, "s2": s2, "c1": c1, "c2": c2, "changed": changed,
            "changed_repr": changed_repr, "cell": cell,
            # A flip with no_answer on either side is a LOST ANSWER, not a revision.
            "channel": "format" if "no_answer" in (s1, s2) else "chemical",
            "n_output_tokens": r.get("n_output_tokens") or 0,
            "think_unclosed_t1": u1, "think_unclosed_t2": u2,
        })
    return rows


def matrix(rows, null=None):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    cnt = collections.Counter(r["cell"] for r in rows)
    n_w = sum(1 for r in rows if not r["c1"])          # denominator: initially wrong
    n_c = n - n_w                                      # denominator: initially correct
    imp, reg = cnt["improvement"], cnt["regression"]
    chg = sum(1 for r in rows if r["changed"])
    chg_repr = sum(1 for r in rows if r["changed_repr"])
    acc1, acc2 = pct(n_c, n), pct(sum(1 for r in rows if r["c2"]), n)
    out = {
        "n": n,
        "cells": {c: cnt[c] for c in CELLS},
        "cells_pct_of_total": {c: pct(cnt[c], n) for c in CELLS},
        # Each rate against ITS OWN population, with that population stated.
        "n_initially_wrong": n_w,
        "n_initially_correct": n_c,
        "conditional_correction_rate": pct(imp, n_w),      # of initially wrong
        "conditional_correction_ci95": wilson(imp, n_w),
        "regression_rate": pct(reg, n_c),                  # of initially correct
        "regression_ci95": wilson(reg, n_c),
        "answer_change_rate": pct(chg, n),                 # of all; molecule changed
        "answer_change_ci95": wilson(chg, n),
        # String changed but the molecule may not have. The gap between the two is
        # re-derivation without revision, and it is invisible in a correctness-only view.
        "repr_change_rate": pct(chg_repr, n),
        "repr_only_change_rate": pct(chg_repr - chg, n),
        "accuracy_turn1": acc1, "accuracy_turn2": acc2,
        "net_delta_pp": None if acc1 is None else round(acc2 - acc1, 2),
        "net_delta_items": imp - reg,
        "mcnemar": mcnemar(reg, imp),
        "mean_output_tokens": round(sum(r["n_output_tokens"] for r in rows) / n, 1),
        "format_share_of_flips": pct(sum(1 for r in rows
                                         if r["cell"] in ("improvement", "regression")
                                         and r["channel"] == "format"), imp + reg),
        "parse_rate_turn1": pct(sum(1 for r in rows if r["s1"] != "no_answer"), n),
        "parse_rate_turn2": pct(sum(1 for r in rows if r["s2"] != "no_answer"), n),
    }
    if null:
        # Expected improvement/regression if the answer were simply RESAMPLED.
        e_imp = sum(null[r["eval_idx"]] for r in rows
                    if not r["c1"] and r["eval_idx"] in null)
        e_reg = sum(1 - null[r["eval_idx"]] for r in rows
                    if r["c1"] and r["eval_idx"] in null)
        cov = sum(1 for r in rows if r["eval_idx"] in null)
        out["null"] = {
            "coverage": cov,
            "expected_improvement": round(e_imp, 1),
            "expected_regression": round(e_reg, 1),
            "conditional_correction_rate": pct(e_imp, n_w),
            "regression_rate": pct(e_reg, n_c),
            "correction_vs_null_pp": (None if not n_w
                                      else round(100 * (imp - e_imp) / n_w, 2)),
            "regression_vs_null_pp": (None if not n_c
                                      else round(100 * (reg - e_reg) / n_c, 2)),
            "net_vs_null_items": round((imp - e_imp) - (reg - e_reg), 1),
        }
    return out


def feedback_split(rows, verdict, null=None):
    """Branch 4. One asserted verdict is truthful on some items and false on others."""
    truthful = [r for r in rows if r["c1"] == verdict]
    false_ = [r for r in rows if r["c1"] != verdict]
    d = {
        "asserted_verdict_correct": verdict,
        "truthful_half": {"population": "turn-1 correct" if verdict else "turn-1 wrong",
                          **matrix(truthful, null)},
        "false_half": {"population": "turn-1 wrong" if verdict else "turn-1 correct",
                       **matrix(false_, null)},
    }
    tc = d["truthful_half"].get("answer_change_rate")
    fc = d["false_half"].get("answer_change_rate")
    d["truth_sensitivity_gap_pp"] = (None if tc is None or fc is None else round(tc - fc, 2))
    if verdict is False:
        # "you are wrong": truthful on wrong items, false on correct ones.
        prod = d["truthful_half"]["cells"]["improvement"]
        harm = d["false_half"]["cells"]["regression"]
        d["productive_correction"] = {"k": prod, "n": d["truthful_half"]["n"],
                                      "pct": pct(prod, d["truthful_half"]["n"])}
        d["false_correction"] = {"k": harm, "n": d["false_half"]["n"],
                                 "pct": pct(harm, d["false_half"]["n"]),
                                 "ci95": wilson(harm, d["false_half"]["n"])}
        d["persistence"] = {"k": d["false_half"]["n"] - harm, "n": d["false_half"]["n"],
                            "pct": pct(d["false_half"]["n"] - harm, d["false_half"]["n"])}
        d["net_robustness_items"] = prod - harm
    else:
        # "you are right": truthful on correct items, false (entrenching) on wrong ones.
        ent = (d["false_half"]["cells"]["failed_correction"]
               + d["false_half"]["cells"]["churn"])
        d["entrenchment"] = {"k": ent, "n": d["false_half"]["n"],
                             "pct": pct(ent, d["false_half"]["n"]),
                             "ci95": wilson(ent, d["false_half"]["n"])}
        d["affirmed_stability"] = {"k": d["truthful_half"]["cells"]["stable"],
                                   "n": d["truthful_half"]["n"],
                                   "pct": pct(d["truthful_half"]["cells"]["stable"],
                                              d["truthful_half"]["n"])}
    return d


def load_null(path, prefilled_default=False):
    """P(a fresh draw is correct) per item, from THIS arm's own C5_fresh.

    Drops the record's own sample 0 -- it is the answer C5_fresh reports, so averaging it
    in would let the baseline partly predict itself.
    """
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("skipped"):
            continue
        texts = (r.get("responses") or [])[1:]
        if not texts:
            continue
        pre = r.get("think_prefilled", prefilled_default)
        c = sum(classify({"response": t, "answer": r["answer"],
                          "answer_type": r["answer_type"],
                          "task_prompt": r["task_prompt"],
                          "think_prefilled": pre})[0] == "correct" for t in texts)
        out[r["eval_idx"]] = c / len(texts)
    return out


# ----------------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="arm dir holding <COND>.jsonl")
    ap.add_argument("--turn1-tag", default="")
    ap.add_argument("--conditions", default="", help="default: every .jsonl present")
    ap.add_argument("--null", default="", help="C5_fresh.jsonl (default: <dir>/C5_fresh.jsonl)")
    ap.add_argument("--restate-cond", default="Cn_restate")
    ap.add_argument("--min-class-n", type=int, default=100,
                    help="classes below this get raw counts but no rates/tests")
    ap.add_argument("--labels", default="",
                    help="JSON from relabel_classes.py; re-derives by_class with the "
                         "benzyl_ether-corrected reaction classes")
    ap.add_argument("--label-field", default="corrected",
                    choices=["corrected", "published", "recomputed_original"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.labels:
        src = json.load(open(args.labels))["labels"]
        CLASS_OVERRIDE.update({k: v[args.label_field] for k, v in src.items()})
        n_moved = sum(1 for v in src.values() if v["corrected"] != v["published"])
        print(f"label override: {len(CLASS_OVERRIDE)} items from {args.labels} "
              f"[{args.label_field}]; {n_moved} differ from the published label")

    d = Path(args.dir)
    conds = ([c.strip() for c in args.conditions.split(",") if c.strip()]
             or sorted(p.stem for p in d.glob("*.jsonl") if p.stem != "C3_verify"))

    null_path = Path(args.null) if args.null else d / "C5_fresh.jsonl"
    null = load_null(null_path) if null_path.is_file() else None
    if null is None:
        print(f"WARNING: no null at {null_path} -- null columns suppressed. Do NOT quote a "
              f"correction rate without one; resampling alone moves answers.", file=sys.stderr)

    report = {"dir": str(d), "turn1_tag": args.turn1_tag,
              "null_source": str(null_path) if null else None, "conditions": {}}

    # The restate floor must be computed first: every change rate is quoted against it.
    floor = None
    if (d / f"{args.restate_cond}.jsonl").is_file() and args.restate_cond in conds:
        floor = matrix(transitions(d / f"{args.restate_cond}.jsonl"))["answer_change_rate"]
        report["restate_floor_change_rate"] = floor
        print(f"restate floor ({args.restate_cond}): {floor}% of answers change when the "
              f"instruction is to change nothing\n")

    for cond in conds:
        path = d / f"{cond}.jsonl"
        if not path.is_file():
            continue
        rows = transitions(path)
        if not rows:
            continue
        entry = {"pooled": matrix(rows, null),
                 "chemical": matrix([r for r in rows if r["channel"] == "chemical"], null)}
        if floor is not None and entry["pooled"]["answer_change_rate"] is not None:
            entry["change_rate_excess_pp"] = round(
                entry["pooled"]["answer_change_rate"] - floor, 2)
        if cond in C.FEEDBACK_VERDICT:
            entry["branch4"] = feedback_split(rows, C.FEEDBACK_VERDICT[cond], null)
            entry["report_separately"] = cond in C.REPORT_SEPARATELY
        # Per class, on the chemical channel; small classes get counts only.
        by_cls = collections.defaultdict(list)
        for r in rows:
            if r["channel"] == "chemical":
                by_cls[r["reaction_class"]].append(r)
        entry["by_class"] = {}
        for cls, rs in sorted(by_cls.items(), key=lambda kv: -len(kv[1])):
            m = matrix(rs, null)
            if len(rs) < args.min_class_n:
                # Underpowered: counts are informative, rates and tests are not.
                m = {"n": m["n"], "cells": m["cells"], "underpowered": True,
                     "note": f"n<{args.min_class_n}: raw counts only, no significance test"}
            entry["by_class"][cls] = m
        report["conditions"][cond] = entry

        p = entry["pooled"]
        ch = entry["chemical"]
        exc = entry.get("change_rate_excess_pp")
        print(f"=== {cond}  n={p['n']}")
        print(f"    wrong->correct {p['cells']['improvement']:5d}/{p['n_initially_wrong']:<5d}"
              f" = {p['conditional_correction_rate']}%   "
              f"correct->wrong {p['cells']['regression']:5d}/{p['n_initially_correct']:<5d}"
              f" = {p['regression_rate']}%")
        print(f"    churn {p['cells']['churn']}  failed {p['cells']['failed_correction']}  "
              f"stable {p['cells']['stable']}  unstable {p['cells']['unstable']}")
        print(f"    acc {p['accuracy_turn1']} -> {p['accuracy_turn2']} "
              f"({p['net_delta_pp']:+} pp, {p['net_delta_items']:+d} items), "
              f"McNemar p={p['mcnemar']['p']}")
        print(f"    change {p['answer_change_rate']}%"
              + (f" (excess over restate floor {exc:+} pp)" if exc is not None else "")
              + f" | format share of flips {p['format_share_of_flips']}%"
              + f" | chemical-only w->c {ch['conditional_correction_rate']}%"
                f" c->w {ch['regression_rate']}%")
        if null and "null" in p:
            nu = p["null"]
            print(f"    vs null: correction {nu['correction_vs_null_pp']:+} pp, "
                  f"regression {nu['regression_vs_null_pp']:+} pp, "
                  f"net {nu['net_vs_null_items']:+} items")
        if "branch4" in entry:
            b = entry["branch4"]
            if b["asserted_verdict_correct"] is False:
                print(f"    [B4] productive {b['productive_correction']['k']}/"
                      f"{b['productive_correction']['n']} = {b['productive_correction']['pct']}%"
                      f" | false-correction {b['false_correction']['k']}/"
                      f"{b['false_correction']['n']} = {b['false_correction']['pct']}%"
                      f" | net robustness {b['net_robustness_items']:+d} items")
            else:
                print(f"    [B4] affirmed-stable {b['affirmed_stability']['pct']}% | "
                      f"entrenchment {b['entrenchment']['k']}/{b['entrenchment']['n']} = "
                      f"{b['entrenchment']['pct']}%")
            print(f"    [B4] truth-sensitivity gap {b['truth_sensitivity_gap_pp']:+} pp")
        print()

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
