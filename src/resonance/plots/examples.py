"""Write synthetic inputs for every figure, as format templates and a smoke test.

    python -m resonance.plots.examples --out /tmp/plot_examples
    python -m resonance.plots.run --data-dir /tmp/plot_examples --out-dir /tmp/plot_examples/figures

Values are made up; they only exercise every field of each format.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def examples(seed=0):
    rng = np.random.default_rng(seed)
    stages = ["base", "S1", "S2", "S3"]

    staged = {"groups": [
        {"name": "Student", "kind": "stages",
         "bars": [{"label": s, "correct": c, "n": 5000}
                  for s, c in zip(stages, [60, 1550, 1950, 2140], strict=True)]},
        {"name": "Frontier", "kind": "references",
         "bars": [{"label": "Model\nA", "correct": 2870, "n": 5000},
                  {"label": "Model\nB", "correct": 3930, "n": 5000}]}]}

    def scatter(conditions, spread, with_se):
        points = []
        for stage in ("S1", "S2", "S3"):
            for cid, _ in conditions:
                p = {"stage": stage, "condition": cid,
                     "delta_accuracy_pp": float(rng.normal(0, spread)),
                     "mean_reasoning_tokens": float(rng.uniform(400, 800)),
                     "baseline_accuracy_pct": {"S1": 31.0, "S2": 41.0, "S3": 44.0}[stage]}
                if with_se:
                    p.update(delta_se_pp=float(rng.uniform(0.2, 0.6)),
                             tokens_se=float(rng.uniform(3, 10)))
                points.append(p)
        return {"stages": ["S1", "S2", "S3"],
                "conditions": [{"id": cid, "label": label} for cid, label in conditions],
                "points": points}

    length = scatter([("commit", "Commit"), ("consider", "Consider"), ("longer", "Think more")],
                     0.4, with_se=True)
    told = scatter([("mild", "Mild doubt"), ("told_wrong", "Told wrong"),
                    ("told_correct", "Told correct")], 5, with_se=False)
    feedback = {k: told[k] for k in ("stages", "conditions")}
    feedback["points"] = [{"stage": p["stage"], "condition": p["condition"],
                           "delta_accuracy_pp": p["delta_accuracy_pp"],
                           "delta_se_pp": float(rng.uniform(0.2, 0.7))}
                          for p in told["points"]]
    feedback.update(bound=20, xlabel="$\\Delta$Acc (pp)")

    ks = [1, 3, 5, 7, 9]
    plurality = {"ks": ks, "stages": ["S1", "S2", "S3"], "series": [
        {"stage": stage, "label": label, "n": n,
         "accuracy_pct": list(np.round(base + gain * np.log(ks), 2))}
        for stage, lift in (("S1", 0), ("S2", 8), ("S3", 12))
        for label, n, base, gain in (("class A", 1100, 35 + lift, 6), ("class B", 800, 25 + lift, 4),
                                     ("class C", 100, 0, 0.2))]}

    return {"staged_accuracy": staged, "length_prompts": length, "feedback_delta": feedback,
            "plurality": plurality}


def main():
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in examples().items():
        (args.out / f"{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(examples())} example inputs to {args.out}")


if __name__ == "__main__":
    main()
