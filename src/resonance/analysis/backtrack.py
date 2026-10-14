"""Backtracking metric over the thinking block (appendix Figs A1-A2, appendix table "Backtracking by source").

Ported from run_dir('relabel')/scripts/backtrack.py; the definition (PATTERNS, LEGACY,
WAIT_ONLY, measure, summarize) is unchanged.

Definition version 2026-09-12.1, FROZEN before any Branch-2 intervention was generated --
a metric tuned after seeing the intervention can always be made to detect it. If it
changes, re-run Step B and say so in RESULTS.md.

Four regex categories, counted separately and pooled:
    reversal   explicit repudiation      ("wait", "actually", "that's wrong")
    revision   proposal then replacement ("instead", "let me redo", "on closer inspection")
    branching  entertaining alternatives ("alternatively", "another possibility")
    marker     structured tags, only if a prompt asked for them

PRIMARY METRIC is `events_per_1k` (events per 1,000 reasoning tokens), not the raw count:
raw counts are confounded with trace length, and length is what these arms differ in most
(t30_base ~6,500 output tokens vs t30_s1 ~435).

Two guards, both reported by default:
  * `events_per_1k_no_wait` -- the token "wait" is 87% of reversal hits on uncollapsed
    arms, so a prompt that suppresses one word could fake an ablation. A Step C/D result
    that moves `events_per_1k` but not this twin is lexical, not structural.
  * `legacy_*` -- the 0901/0903 regex verbatim, so numbers stay comparable to published
    trace statistics.

This is a lexical detector: it cannot tell a real reconsideration from "actually quite
stable", and `--audit-sample` exists to quantify that. The audit is mandatory before any
accuracy claim is attached. Never report a backtracking-vs-correctness correlation without
conditioning on leave-one-out difficulty -- that confound has flipped three analyses here.

  python -m resonance.analysis.backtrack --dir <condition dir> --out report.json
  python -m resonance.analysis.backtrack --turn1 <eval set .json> --tag q8_base
  python -m resonance.analysis.backtrack --dir <condition dir> --audit-sample 150
"""
import argparse
import collections
import glob
import json
import os
import random
import re

# --------------------------------------------------------------------------------------
# The frozen definition
# --------------------------------------------------------------------------------------
PATTERNS = {
    "reversal": re.compile(
        r"\b(wait|hold on|hmm+|oops|scratch that|"
        r"(?:that|this|the above)(?:'s| is| was| would be)\s+(?:wrong|incorrect|not right|a mistake)|"
        r"i (?:made|"r"made an) (?:a )?(?:mistake|error)|"
        r"no[,.]? (?:that|this)(?:'s| is)\s+(?:wrong|incorrect|not right)|"
        r"i was (?:wrong|mistaken)|"
        r"on second thought|actually)\b", re.I),
    "revision": re.compile(
        r"\b(instead(?: of)?|rather than|let me (?:re|try again|start over|redo|rethink|"
        r"reconsider|revisit)|reconsider(?:ing)?|revis(?:e|ing|ed|ion)|"
        r"correction|on (?:closer|further) (?:inspection|reflection|thought)|"
        r"going back|back up|revers(?:e|ing))\b", re.I),
    "branching": re.compile(
        r"\b(alternatively|another (?:possibility|option|route|candidate|way)|"
        r"other (?:possibilit|option)|a second (?:possibility|option)|"
        r"option [b-z]\b|could also be|might also be|or it (?:could|might) be|"
        r"two possibilit|both (?:are|seem) plausible|on the other hand)\b", re.I),
    "marker": re.compile(
        r"<\s*/?\s*(candidate|reconsideration|revision|alternative)\s*>", re.I),
}

# The 0910/0901 regex, kept VERBATIM so new numbers stay comparable with published trace
# statistics (search_metrics.py, trace_dynamics.py). Reported alongside, never instead of:
# it is narrower than `reversal` + `revision` and conflates the two.
LEGACY = re.compile(r"\b(wait|hmm|actually|reconsider|on second thought|let me re|"
                    r"however,? (?:this|that) (?:is|would be) (?:wrong|incorrect)|"
                    r"correction|i made an error|revis(?:e|ing|ed))\b", re.I)

# "wait" is 87% of reversal hits on uncollapsed arms, so this metric is close to a count of
# one token there. `events_per_1k_no_wait` is the robustness twin -- an intervention that
# moves the metric but not the twin changed vocabulary, not search. See RESULTS.md 6.
WAIT_ONLY = re.compile(r"\bwait\b", re.I)

DEFINITION_VERSION = "2026-09-12.1"


def split_trace(response, prefilled):
    """(thinking, post_think, closed).

    The original imported `split_trace` from the metareasoning pipeline's common.py when it
    was importable; that function is identical to this one.
    """
    # Test the CLOSING tag alone: a prefilled family's response never contains
    # "<think>", so an opening-tag test is always False and silently mis-bins every
    # truncated generation.
    if "</think>" not in response:
        body = response
        if not prefilled and "<think>" in body:
            body = body.split("<think>", 1)[1]
        return body, "", False
    head, post = response.split("</think>", 1)
    if not prefilled and "<think>" in head:
        head = head.split("<think>", 1)[1]
    return head, post, True


def measure(response, prefilled=False, n_output_tokens=None):
    """Per-trace backtracking metrics over the THINKING BLOCK only.

    Restricting to the thinking block matters: the post-</think> answer section is
    boilerplate in these checkpoints ('## Reaction Trace: ...'), and counting it would add
    a near-constant to every trace, compressing the between-arm contrast.
    """
    think, _, closed = split_trace(response or "", prefilled)
    n = len(think)
    hits = {k: [m.start() for m in p.finditer(think)] for k, p in PATTERNS.items()}
    total = sum(len(v) for v in hits.values())
    first = min((v[0] for v in hits.values() if v), default=None)
    # Token count for normalisation. Prefer the recorded generation length; fall back to a
    # chars/4 estimate, flagged, so a mixed corpus never silently mixes the two scales.
    if n_output_tokens:
        tok, est = n_output_tokens, False
    else:
        tok, est = max(1, n // 4), True
    return {
        "n_think_chars": n,
        "n_output_tokens": tok,
        "tokens_estimated": est,
        "think_closed": closed,
        "counts": {k: len(v) for k, v in hits.items()},
        "n_events": total,
        "any_backtrack": total > 0,
        "events_per_1k": round(1000.0 * total / tok, 3),
        "n_wait": len(WAIT_ONLY.findall(think)),
        "events_per_1k_no_wait": round(
            1000.0 * (total - len(WAIT_ONLY.findall(think))) / tok, 3),
        "first_event_frac": (None if first is None or n == 0 else round(first / n, 3)),
        "legacy_count": len(LEGACY.findall(think)),
        "legacy_per_1k": round(1000.0 * len(LEGACY.findall(think)) / tok, 3),
    }


# --------------------------------------------------------------------------------------
def _rows_from_conditions(d, skip=("C3_verify",)):
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        cond = os.path.basename(f)[:-6]
        if cond in skip:
            continue
        for line in open(f):
            r = json.loads(line)
            if r.get("skipped"):
                continue
            yield cond, r.get("response") or "", r.get("think_prefilled", False), \
                r.get("n_output_tokens"), r


def _rows_from_turn1(path, tag):
    d = json.load(open(path))
    if isinstance(d, dict) and "items" not in d:
        for v in d.values():
            yield "turn1", v["response"], v.get("think_prefilled", tag.startswith("t30_")), \
                v.get("n_output_tokens"), v
    else:
        items = d["items"] if isinstance(d, dict) and "items" in d else d
        for it in items:
            if tag in it.get("turn1", {}):
                v = it["turn1"][tag]
                yield "turn1", v["response"], v.get("think_prefilled", tag.startswith("t30_")), \
                    v.get("n_output_tokens"), v


def summarize(ms):
    n = len(ms)
    if not n:
        return {"n": 0}
    per1k = sorted(m["events_per_1k"] for m in ms)
    tot = collections.Counter()
    for m in ms:
        tot.update(m["counts"])
    return {
        "n": n,
        "any_backtrack_pct": round(100.0 * sum(m["any_backtrack"] for m in ms) / n, 2),
        "mean_events": round(sum(m["n_events"] for m in ms) / n, 3),
        "mean_events_per_1k": round(sum(per1k) / n, 3),
        "mean_events_per_1k_no_wait": round(
            sum(m["events_per_1k_no_wait"] for m in ms) / n, 3),
        "wait_share_of_events_pct": round(
            100.0 * sum(m["n_wait"] for m in ms) / max(1, sum(m["n_events"] for m in ms)), 1),
        "median_events_per_1k": per1k[n // 2],
        "mean_output_tokens": round(sum(m["n_output_tokens"] for m in ms) / n, 1),
        "tokens_estimated": any(m["tokens_estimated"] for m in ms),
        "by_category": dict(tot),
        "mean_first_event_frac": (
            round(sum(m["first_event_frac"] for m in ms if m["first_event_frac"] is not None)
                  / max(1, sum(m["first_event_frac"] is not None for m in ms)), 3)),
        "legacy_mean_per_1k": round(sum(m["legacy_per_1k"] for m in ms) / n, 3),
        "think_unclosed_pct": round(100.0 * sum(not m["think_closed"] for m in ms) / n, 2),
    }


def audit_sample(rows, n, seed=42, out=None):
    """Write a stratified sample for the MANDATORY manual audit.

    Deliberately mixes detected and undetected traces so the rubric can estimate BOTH
    false positives and false negatives. A sample of only detected traces measures
    precision and silently assumes recall is 1.
    """
    rng = random.Random(seed)
    hit = [r for r in rows if r["metrics"]["any_backtrack"]]
    miss = [r for r in rows if not r["metrics"]["any_backtrack"]]
    half = n // 2
    sel = (rng.sample(hit, min(half, len(hit)))
           + rng.sample(miss, min(n - half, len(miss))))
    rng.shuffle(sel)
    for r in sel:
        r["AUDIT_human_label"] = None      # fill in: true | false | unclear
        r["AUDIT_notes"] = ""
    if out:
        json.dump({"definition_version": DEFINITION_VERSION,
                   "rubric": "Label `true` if the trace explicitly repudiates, replaces or "
                             "weighs an alternative to something it already proposed. "
                             "Label `false` for a cue word used without that function "
                             "(e.g. 'actually quite stable'). `unclear` if ambiguous.",
                   "n_detected": len(hit), "n_undetected": len(miss),
                   "sample": sel}, open(out, "w"), indent=1)
        print(f"wrote {out} ({len(sel)} traces for manual labelling)")
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="", help="arm dir of <COND>.jsonl")
    ap.add_argument("--turn1", default="", help="eval set or per-tag shard")
    ap.add_argument("--tag", default="", help="turn-1 tag, with --turn1")
    ap.add_argument("--audit-sample", type=int, default=0)
    ap.add_argument("--audit-out", default="")
    ap.add_argument("--keep-traces", action="store_true",
                    help="keep response text in the audit sample (large)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not args.dir and not args.turn1:
        ap.error("pass --dir or --turn1")

    src = (_rows_from_conditions(args.dir) if args.dir
           else _rows_from_turn1(args.turn1, args.tag))
    rows, by_cond = [], collections.defaultdict(list)
    for cond, resp, pre, ntok, rec in src:
        m = measure(resp, pre, ntok)
        by_cond[cond].append(m)
        rows.append({"condition": cond, "eval_idx": rec.get("eval_idx"),
                     "reaction_class": rec.get("reaction_class"),
                     "metrics": m,
                     **({"response": resp} if args.keep_traces or args.audit_sample else {})})

    report = {"definition_version": DEFINITION_VERSION,
              "source": args.dir or f"{args.turn1}:{args.tag}",
              "by_condition": {c: summarize(v) for c, v in sorted(by_cond.items())}}

    print(f"definition {DEFINITION_VERSION}   source {report['source']}")
    print(f"{'condition':<24}{'n':>6}{'any%':>7}{'ev/1k':>8}{'-wait':>8}{'wait%':>7}"
          f"{'events':>8}{'tok':>8}{'first%':>8}{'legacy':>8}")
    for c, s in report["by_condition"].items():
        print(f"{c:<24}{s['n']:>6}{s['any_backtrack_pct']:>7.2f}"
              f"{s['mean_events_per_1k']:>8.3f}{s['mean_events_per_1k_no_wait']:>8.3f}"
              f"{s['wait_share_of_events_pct']:>7.1f}{s['mean_events']:>8.2f}"
              f"{s['mean_output_tokens']:>8.0f}"
              f"{(s['mean_first_event_frac'] or 0) * 100:>8.1f}{s['legacy_mean_per_1k']:>8.3f}")
        print(f"{'':24}  {s['by_category']}")

    if args.audit_sample:
        audit_sample(rows, args.audit_sample,
                     out=args.audit_out or "outputs/backtrack_audit_sample.json")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
