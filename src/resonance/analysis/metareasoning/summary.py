"""Paper numbers for the metareasoning figures, computed from condition outputs.

    python -m resonance.analysis.metareasoning.summary --config summary_config.json --out OUT

Writes OUT/summary.json (every statistic) and the resonance.plots inputs named in
the config (intervention_scatter / feedback_delta / plurality formats).

Definitions (fixed; see README.md):
  * Accuracy counts every non-skipped reaction; a response without a parsable
    product is wrong. `chemical_channel` repeats the comparison on reactions where
    both the prior-turn and the new response parse (the earlier figure definition).
  * Delta is paired by reaction: mean over reactions of (condition accuracy -
    baseline accuracy), each averaged over that reaction's draws. SE is the
    question-level SD / sqrt(n); p is two-sided normal on Delta/SE, plus exact
    McNemar when both sides are single draws. `se_draws_pp` is the draw-level
    alternative: SD of the per-draw accuracies / sqrt(draws), combined for the
    condition and baseline in quadrature (a figure's `error_bars: "draws"`).
  * Reasoning tokens: tokenizer offsets inside <think>...</think>; a prefilled
    think starts at 0; unclosed reasoning ends at the first answer marker or the
    end of the response; a response with neither <think> nor a prefill counts 0.
  * Reaction classes come from the `corrected` field of the relabel file. The
    plurality block may merge them into families (`plurality.families`: family ->
    member classes), e.g. protections + deprotections -> protection chemistry.
"""
import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

FROZEN = Path(__file__).resolve().parent / "frozen"
sys.path.insert(0, str(FROZEN))

from e1_allocation import majority  # noqa: E402
from score_selfcorrect import EXTRACT, canon, classify  # noqa: E402

ANSWER_MARKER = re.compile(r"<answer>|<product>|<forward_reaction>|\\boxed\{")


def load_jsonl(path):
    with open(path) as handle:
        return {r["eval_idx"]: r for r in map(json.loads, handle) if not r.get("skipped")}


def texts(record):
    return list(record.get("responses") or [record["response"]])


def correct(record, text):
    return classify({**record, "response": text})[0] == "correct"


def has_answer(text, prefilled):
    if prefilled and "</think>" not in text and not ANSWER_MARKER.search(text):
        return False
    return EXTRACT(text) is not None


def reasoning_tokens(text, prefilled, tokenizer):
    start = text.find("<think>")
    if start >= 0:
        start += len("<think>")
    elif prefilled:
        start = 0
    else:
        return 0
    end = text.find("</think>", start)
    if end < 0:
        marker = ANSWER_MARKER.search(text[start:])
        end = start + marker.start() if marker else len(text)
    offsets = tokenizer.encode(text, add_special_tokens=False).offsets
    return sum(a >= start and b <= end and b > a for a, b in offsets)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def se(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def normal_p(delta, err):
    if not err or math.isnan(err):
        return float("nan")
    return math.erfc(abs(delta / err) / math.sqrt(2))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired(cond, base, ids):
    """Paired statistics over `ids` (percentage points)."""
    deltas = [100 * (mean(cond[i]) - mean(base[i])) for i in ids]
    out = {"n": len(ids), "accuracy_pct": 100 * mean([mean(cond[i]) for i in ids]),
           "baseline_accuracy_pct": 100 * mean([mean(base[i]) for i in ids]),
           "delta_pp": mean(deltas), "se_pp": se(deltas)}
    out["p_normal"] = normal_p(out["delta_pp"], out["se_pp"])
    if all(len(cond[i]) == 1 and len(base[i]) == 1 for i in ids):
        b = sum(base[i][0] and not cond[i][0] for i in ids)
        c = sum(cond[i][0] and not base[i][0] for i in ids)
        out.update(mcnemar_discordant=[b, c], p_mcnemar=mcnemar(b, c))
    return out


def condition_file(stage, cond):
    eight = stage.get("eight_rollouts_dir")
    if eight and (Path(eight) / f"{cond}.jsonl").is_file():
        return Path(eight) / f"{cond}.jsonl"
    return Path(stage["conditions_dir"]) / f"{cond}.jsonl"


def baseline_file(stage, cond):
    return Path(stage["fresh"]) if cond == "C5_fresh" else condition_file(stage, cond)


_SCORED = {}


def scored(path):
    """eval_idx -> (record, per-draw correctness), cached per file."""
    key = str(path)
    if key not in _SCORED:
        recs = load_jsonl(path)
        _SCORED[key] = {i: (r, [correct(r, t) for t in texts(r)]) for i, r in recs.items()}
    return _SCORED[key]


def summarize_condition(stage, cond, baseline, labels, tokenizer, min_class_n):
    path = condition_file(stage, cond)
    cur, base = scored(path), scored(baseline_file(stage, baseline))
    ids = sorted(set(cur) & set(base))
    recs = {i: cur[i][0] for i in ids}
    ok = {i: cur[i][1] for i in ids}
    base_ok = {i: base[i][1] for i in ids}
    base_recs = {i: base[i][0] for i in ids}
    toks = {i: [reasoning_tokens(t, recs[i].get("think_prefilled", False), tokenizer)
                for t in texts(recs[i])] for i in ids}
    token_means = [mean(toks[i]) for i in ids]
    # Draw-level SE: SD of the per-draw means over draws / sqrt(draws) (error_bars: "draws").
    def draw_means(per_row):
        n_draws = min(len(per_row[i]) for i in ids)
        return [mean([per_row[i][j] for i in ids]) for j in range(n_draws)]
    se_cond, se_base = se(draw_means(ok)), se(draw_means(base_ok))
    draws_se = {"se_draws_pp": 100 * math.sqrt(se_cond ** 2 + se_base ** 2),
                "se_draws_reasoning_tokens": se(draw_means(toks))}
    result = {"source": str(path), "baseline": baseline, "draws": len(texts(recs[ids[0]])),
              "baseline_draws": len(texts(base_recs[ids[0]])),
              **paired(ok, base_ok, ids),
              "mean_reasoning_tokens": mean(token_means), "se_reasoning_tokens": se(token_means),
              **draws_se}
    # Earlier figure definition: both turns of the (first) draw must parse.
    chem = [i for i in ids if all(has_answer(recs[i][k], recs[i].get("think_prefilled", False))
                                  for k in ("response", "turn1_response"))]
    first = {i: ok[i][:1] for i in chem}
    result["chemical_channel"] = paired(first, {i: base_ok[i] for i in chem}, chem)
    by_class = collections.defaultdict(list)
    for i in ids:
        by_class[labels.get(i, "unclassified")].append(i)
    result["by_class"] = {cls: paired(ok, base_ok, members)
                          for cls, members in sorted(by_class.items(), key=lambda kv: -len(kv[1]))
                          if len(members) >= min_class_n}
    return result


def plurality(stage, labels, ks, min_class_n):
    """plurality@k by class over the turn-1 draw followed by the fresh draws."""
    tag = stage["tag"]
    turn1 = {r["eval_idx"]: r["turn1"][tag]["response"] for r in json.loads(Path(stage["eval"]).read_text())
             if tag in r["turn1"]}
    hits = collections.defaultdict(lambda: collections.Counter())
    sizes = collections.Counter()
    for i, r in load_jsonl(stage["fresh"]).items():
        if i not in turn1:
            continue
        base = {"answer": r["answer"], "answer_type": r["answer_type"], "task_prompt": r["task_prompt"],
                "think_prefilled": r.get("think_prefilled", False)}
        preds = [classify({**base, "response": t})[2] for t in [turn1[i]] + list(r.get("responses") or [])]
        gt = canon(r["answer"])
        cls = labels.get(i, "unclassified")
        sizes[cls] += 1
        for k in ks:
            m = majority(preds[:k])
            hits[cls][k] += m is not None and m == gt
    return {cls: {"n": n, **{f"maj@{k}": round(100 * hits[cls][k] / n, 2) for k in ks}}
            for cls, n in sizes.items() if n >= min_class_n}


def main(argv=None):
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    from tokenizers import Tokenizer

    labels = {int(k): v["corrected"] for k, v in json.loads(Path(cfg["labels"]).read_text())["labels"].items()}
    min_n = cfg.get("min_class_n", 40)
    report = {"definitions": str(__doc__).split("Definitions (fixed; see README.md):")[1].strip(),
              "figures": {}, "plurality": {}}
    args.out.mkdir(parents=True, exist_ok=True)

    for fig_name, fig in cfg["figures"].items():
        report["figures"][fig_name] = {}
        points = []
        for stage_name, stage in cfg["stages"].items():
            tokenizer = Tokenizer.from_file(stage["tokenizer"])
            report["figures"][fig_name][stage_name] = {}
            for cond, *_ in fig["conditions"]:
                s = summarize_condition(stage, cond, fig["baseline"], labels, tokenizer, min_n)
                report["figures"][fig_name][stage_name][cond] = s
                points.append({"stage": stage_name, "condition": cond, "delta_accuracy_pp": s["delta_pp"],
                               "mean_reasoning_tokens": s["mean_reasoning_tokens"],
                               "baseline_accuracy_pct": s["baseline_accuracy_pct"],
                               **({"delta_se_pp": s["se_draws_pp"], "tokens_se": s["se_draws_reasoning_tokens"]}
                                  if fig.get("error_bars") == "draws" else
                                  {"delta_se_pp": s["se_pp"], "tokens_se": s["se_reasoning_tokens"]}
                                  if fig.get("error_bars") else {})})
                print(f"{fig_name} {stage_name} {cond}: {s['delta_pp']:+.2f} ± {s['se_pp']:.2f} pp "
                      f"(draws={s['draws']}, n={s['n']}), {s['mean_reasoning_tokens']:.0f} tokens", flush=True)
        plot = {"stages": list(cfg["stages"]), "points": points,
                # Fig 4 conditions carry a marker, Fig 5 conditions do not.
                "conditions": [dict(zip(("id", "label", "color", "marker"), c, strict=False))
                               for c in fig["conditions"]],
                **fig.get("plot", {})}
        (args.out / f"{fig['plot_file']}.json").write_text(json.dumps(plot, indent=2, ensure_ascii=False) + "\n")

    if "plurality" in cfg:
        pl = cfg["plurality"]
        family_of = {m: fam for fam, members in pl.get("families", {}).items() for m in members}
        pl_labels = {i: family_of.get(c, c) for i, c in labels.items()}
        series = []
        for stage_name, stage in cfg["stages"].items():
            table = plurality(stage, pl_labels, pl["ks"], pl.get("min_class_n", min_n))
            report["plurality"][stage_name] = table
            for key, label, color in pl["classes"]:
                m = table[key]
                series.append({"stage": stage_name, "label": label, "n": m["n"], "color": color,
                               "accuracy_pct": [m[f"maj@{k}"] for k in pl["ks"]]})
        plot = {"ks": pl["ks"], "stages": list(cfg["stages"]), "series": series, **pl.get("plot", {})}
        (args.out / f"{pl['plot_file']}.json").write_text(json.dumps(plot, indent=2, ensure_ascii=False) + "\n")

    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", args.out / "summary.json")


if __name__ == "__main__":
    main()
