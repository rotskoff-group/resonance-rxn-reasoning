"""Fail if draw position predicts accuracy (the shared-seed symptom).

Ported from run_dir('paper_figures')/scripts/check_draw_independence.py.

For every multi-draw file in the summary config (fresh pool and 8-rollout
conditions), tests whether per-position accuracy is homogeneous with a chi-square
test over draw positions. Writes a JSON report; exits 1 if any file has p < --alpha.

    python -m resonance.paper.draw_independence --config kimi_no_id_paper.json --out <dir>
"""
import argparse
import json
import math
import sys
from pathlib import Path

from resonance.analysis.metareasoning.summary import classify  # frozen/score_selfcorrect.classify
from resonance.paper import config as paper_config


def chi2_sf(x, k):
    """Upper tail of chi-square with k degrees of freedom (k even or odd), via series."""
    if x <= 0:
        return 1.0
    # Regularized upper incomplete gamma Q(k/2, x/2) by continued fraction / series.
    a, z = k / 2, x / 2
    if z < a + 1:
        term = total = 1 / a
        n = a
        for _ in range(1000):
            n += 1
            term *= z / n
            total += term
            if term < total * 1e-12:
                break
        return 1 - total * math.exp(-z + a * math.log(z) - math.lgamma(a))
    b, c, d = z + 1 - a, 1e300, 1 / (z + 1 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = 1 / (an * d + b)
        c = b + an / c
        h *= d * c
        if abs(d * c - 1) < 1e-12:
            break
    return h * math.exp(-z + a * math.log(z) - math.lgamma(a))


def per_position(path):
    hits, n = None, 0
    for line in open(path):
        r = json.loads(line)
        if r.get("skipped") or not r.get("responses"):
            continue
        ok = [classify({**r, "response": t})[0] == "correct" for t in r["responses"]]
        hits = hits or [0] * len(ok)
        hits = [h + o for h, o in zip(hits, ok, strict=True)]
        n += 1
    return hits, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    ap.add_argument("--config", type=Path, required=True,
                    help="summary config (path, or a name in configs/paper/figures)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=1e-3)
    args = ap.parse_args(argv)
    cfg = paper_config.load(args.config)
    report, failed = {}, []
    for stage_name, stage in cfg["stages"].items():
        files = [Path(stage["fresh"])] + [Path(stage["conditions_dir"]) / f"{c}.jsonl"
                                           for c, *_ in cfg["figures"]["length_prompts"]["conditions"]]
        for path in files:
            hits, n = per_position(path)
            if not hits or len(hits) < 2:
                continue
            p_bar = sum(hits) / (n * len(hits))
            stat = sum((h - n * p_bar) ** 2 / (n * p_bar * (1 - p_bar)) for h in hits)
            p = chi2_sf(stat, len(hits) - 1)
            key = f"{stage_name}/{path.stem}"
            report[key] = {"n": n, "accuracy_by_draw_pct": [round(100 * h / n, 2) for h in hits],
                           "chi2": round(stat, 2), "df": len(hits) - 1, "p": p}
            print(f"{key:28s} {report[key]['accuracy_by_draw_pct']}  chi2={stat:.1f} p={p:.3g}", flush=True)
            if p < args.alpha:
                failed.append(key)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "draw_independence.json").write_text(
        json.dumps({"alpha": args.alpha, "failed": failed, "files": report}, indent=2) + "\n")
    if failed:
        print("FAIL: draw position predicts accuracy in " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
