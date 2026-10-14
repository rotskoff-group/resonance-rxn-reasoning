"""Backtracking rates behind appendix Figs A1-A2 and appendix table "Backtracking by source".

Ported from run_dir('paper_figures')/scripts/{backtrack_draws,backtrack_appendix}.py. The
detector is `resonance.analysis.backtrack` (definition 2026-09-12.1), applied to every draw
of every multi-draw record: the 8-draw feedback run first, then the length prompts, then the
fresh resamples (a condition found in several runs is taken from the first). References:
Qwen3-8B base at T 0.6, and the Kimi K3 SFT corpus with tokens counted by the Qwen3-8B
tokenizer.

    python -m resonance.paper.backtrack --out <dir>   # needs <dir>/Fig_{4,5}.json; writes
                                                      # <dir>/backtrack/kimi_s*_draws.json, <dir>/Fig_A_backtrack.json
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from resonance.analysis.backtrack import DEFINITION_VERSION, _rows_from_turn1, measure, summarize
from resonance.paths import run_dir

SKIP = {"C3_verify"}
KEYS = ["n", "any_backtrack_pct", "mean_events_per_1k", "mean_events_per_1k_no_wait",
        "mean_output_tokens"]


def stage_dirs(stage):
    runs = run_dir("paper_figures") / "runs"
    return [runs / f"per_reaction_seed_feedback8/full/outputs/t06_kimi_s{stage}",
            runs / f"per_reaction_seed/full/outputs/t06_kimi_s{stage}",
            runs / f"per_reaction_seed/full/outputs/fresh_t06/kimi_s{stage}"]


def base_turn1():
    return run_dir("relabel") / "data/turn1_t06_q8_base.json"


def corpus():
    return run_dir("traces") / "outputs/traces_stage1a_no_reaction_id.jsonl"


def tokenizer():
    return run_dir("traces") / "outputs/model_0a_no_reaction_id"


def measure_file(path):
    ms, draws = [], set()
    for line in open(path):
        r = json.loads(line)
        if r.get("skipped"):
            continue
        texts = r.get("responses") or [r.get("response") or ""]
        toks = r.get("n_output_tokens_all") or [r.get("n_output_tokens")] * len(texts)
        assert len(toks) == len(texts), path
        draws.add(len(texts))
        pre = r.get("think_prefilled", False)
        ms.extend(measure(t or "", pre, n) for t, n in zip(texts, toks, strict=True))
    return path.stem, ms, sorted(draws)


def draws_report(dirs, procs):
    files, seen = [], set()
    for d in dirs:
        for f in sorted(Path(d).glob("*.jsonl")):
            if f.stem not in SKIP and f.stem not in seen:
                seen.add(f.stem)
                files.append(f)
    with Pool(procs) as pool:
        results = pool.map(measure_file, files)
    return {"definition_version": DEFINITION_VERSION, "source": [str(d) for d in dirs], "all_draws": True,
            "files": {f.stem: str(f) for f in files},
            "draws": {c: d for c, _, d in results},
            "by_condition": {c: summarize(ms) for c, ms, _ in sorted(results)}}


def pick(s):
    return {k: s[k] for k in KEYS}


def corpus_stats():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer())
    traces = [json.loads(line)["trace"] for line in open(corpus())]
    lens = [len(ids) for ids in tok(traces, add_special_tokens=False)["input_ids"]]
    # No think tags in a corpus trace: prefilled=True scores the whole trace as thinking.
    return summarize([measure(t, True, n) for t, n in zip(traces, lens, strict=True)])


def main():
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=12)
    args = ap.parse_args()
    bt = args.out / "backtrack"
    bt.mkdir(parents=True, exist_ok=True)
    stages = {}
    for i in (1, 2, 3):
        report = draws_report(stage_dirs(i), args.procs)
        (bt / f"kimi_s{i}_draws.json").write_text(json.dumps(report, indent=1) + "\n")
        stages[f"S{i}"] = {c: pick(s) for c, s in report["by_condition"].items()}

    base = summarize([measure(r, p, n) for _, r, p, n, _ in _rows_from_turn1(base_turn1(), "q8_base")])
    out = {"definition_version": DEFINITION_VERSION,
           "reference": {"Qwen3-8B base": pick(base), "SFT corpus (Kimi K3)": pick(corpus_stats())},
           "stages": stages,
           "accuracy": {f: json.loads((args.out / f"{f}.json").read_text()) for f in ("Fig_4", "Fig_5")}}
    path = args.out / "Fig_A_backtrack.json"
    json.dump(out, open(path, "w"), indent=1)
    for k, v in out["reference"].items():
        print(f"{k:<24} {v}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
