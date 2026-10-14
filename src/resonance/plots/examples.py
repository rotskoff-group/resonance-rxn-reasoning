"""Write synthetic inputs for every figure, as format templates and a smoke test.

    python -m resonance.plots.examples --out /tmp/plot_examples
    python -m resonance.plots.run --data-dir /tmp/plot_examples --out-dir /tmp/plot_examples/figures

Values are made up; they only exercise every field of each format.
"""
import argparse
import json
from pathlib import Path

import numpy as np

FAMILIES = ["heteroatom alkylation/arylation", "protection chemistry", "acylation",
            "C-C bond formation", "redox", "FG changes", "heterocycle formation"]


def examples(seed=0):
    rng = np.random.default_rng(seed)
    n_test = [1479, 1131, 969, 521, 473, 144, 102]
    stages = ["base", "S1", "S2", "S3"]

    staged = {"groups": [
        {"name": "Student", "kind": "stages",
         "bars": [{"label": s, "correct": c, "n": 5000}
                  for s, c in zip(stages, [60, 1550, 1950, 2140], strict=True)]},
        {"name": "Frontier", "kind": "references",
         "bars": [{"label": "Model\nA", "correct": 2870, "n": 5000},
                  {"label": "Model\nB", "correct": 3930, "n": 5000}]}]}

    fams = []
    for name, n in zip(FAMILIES, n_test, strict=True):
        rates = np.sort(rng.uniform(0.0, 0.6, 4)) * [0.05, 1, 1, 1]
        correct = {s: int(r * n) for s, r in zip(stages, rates, strict=True)}
        correct["teacher"] = int(rng.uniform(0.2, 0.8) * n)
        fams.append({"name": name, "short": name.split()[0][:8], "n_train": int(n * 8),
                     "n_test": n, "correct": correct})
    overall = {"n": sum(n_test), "correct": {s: sum(f["correct"][s] for f in fams) for s in stages}}
    progression = {"stages": stages, "teacher": "Teacher", "families": fams, "overall": overall}

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
    counterfactual = scatter([("mild", "Mild doubt"), ("told_wrong", "Told wrong"),
                              ("told_correct", "Told correct")], 5, with_se=False)
    counterfactual["baseline_label"] = "Control"
    feedback = {k: counterfactual[k] for k in ("stages", "conditions")}
    feedback["points"] = [{"stage": p["stage"], "condition": p["condition"],
                           "delta_accuracy_pp": p["delta_accuracy_pp"],
                           "delta_se_pp": float(rng.uniform(0.2, 0.7))}
                          for p in counterfactual["points"]]
    feedback.update(bound=20, xlabel="$\\Delta$Acc (pp)")

    ks = [1, 3, 5, 7, 9]
    plurality = {"ks": ks, "stages": ["S1", "S2", "S3"], "series": [
        {"stage": stage, "label": label, "n": n,
         "accuracy_pct": list(np.round(base + gain * np.log(ks), 2))}
        for stage, lift in (("S1", 0), ("S2", 8), ("S3", 12))
        for label, n, base, gain in (("class A", 1100, 35 + lift, 6), ("class B", 800, 25 + lift, 4),
                                     ("class C", 100, 0, 0.2))]}

    ablation_fams = []
    for name, n in zip(FAMILIES, n_test, strict=True):
        base = int(rng.uniform(0, 0.4) * n)
        s1 = int(rng.uniform(0.6, 0.9) * n)
        ablation_fams.append({
            "name": name, "short": name, "n_test": n,
            "correct": {"base": base, "s1": s1, "s2": s1 + 3, "s3": s1 + 5},
            "base_finished": n // 2, "base_correct_finished": min(n // 2, base),
            "pairs": {"s1->s2": [int(rng.integers(5, 40)), int(rng.integers(5, 40))],
                      "s2->s3": [int(rng.integers(2, 20)), int(rng.integers(2, 20))]}})
    ablation = {"paths": [{"key": "s1", "label": "S1"}, {"key": "s2", "label": "S1→S2"},
                          {"key": "s3", "label": "S1→S2→S3"},
                          {"key": "base_s3", "label": "S3 (pending)"}],
                "increments": [{"pair": "s1->s2", "label": "S2−S1"},
                               {"pair": "s2->s3", "label": "S3−S2"}],
                "families": ablation_fams}

    return {"staged_accuracy": staged, "class_progression": progression,
            "length_prompts": length, "counterfactual": counterfactual, "feedback_delta": feedback,
            "plurality": plurality, "stage_ablation": ablation}


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
