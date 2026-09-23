"""Score Opus results (workflow output plus resume fill-in), overall and per template.

Safe to run while generation is in progress: it reads whatever lines exist, and marks the
report partial until every test row has a response. Also the scorer for the 8B stages
(run_kimi8b.py passes --out-dir and --model).

Ported from run_dir('synthetic_ood')/scripts/score_opus.py.
CLI: python -m resonance.pipelines.synthetic_ood.score_opus [--suite S] [--out-dir D]
     [--model M] [--root R]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from resonance.evaluation.extractors import extract_forward_rxn
from resonance.paths import run_dir

RDLogger.DisableLog("rdApp.*")
ROOT = run_dir("synthetic_ood")


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles.strip()) if smiles else None
    return Chem.MolToSmiles(mol) if mol else None


def load(out_dir):
    records = {}
    for name in ("results.jsonl", "results_resume.jsonl"):
        path = out_dir / name
        if not path.exists():
            continue
        for line in path.open():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("response") is not None and "error" not in rec:
                records.setdefault(rec["idx"], rec)
    return records


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="synthetic_alpha_0.5")
    ap.add_argument("--run", default="synthetic_alpha_0.5_full")
    ap.add_argument("--out-dir", help="results directory; overrides outputs/opus5/<run>")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--root", type=Path, default=ROOT, help="holds data/<suite>/")
    args = ap.parse_args()
    ROOT = args.root
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs/opus5" / args.run
    data = json.loads((ROOT / "data" / args.suite / "test.json").read_text())
    labels = json.loads((ROOT / "data" / args.suite / "labels.json").read_text())["labels"]
    records = load(out_dir)
    groups = defaultdict(lambda: {"rows": 0, "correct": 0, "valid": 0})
    rows = []
    for idx, rec in sorted(records.items()):
        gold = canonical(data[idx]["original_smiles"]["End Product"])
        pred = canonical(extract_forward_rxn(rec["response"]) or "")
        correct = pred is not None and pred == gold
        g = groups[labels[str(idx)]["level1"]]
        g["rows"] += 1
        g["correct"] += int(correct)
        g["valid"] += int(pred is not None)
        rows.append({"idx": idx, "template": labels[str(idx)]["level1"], "prediction": pred,
                     "gold": gold, "correct": correct})
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    for g in groups.values():
        g["accuracy"] = g["correct"] / g["rows"]
    report = {"suite": args.suite, "model": args.model, "complete": n == len(data),
              "scored": n, "total": len(data), "correct": correct,
              "accuracy": correct / n if n else None,
              "valid_smiles_fraction": sum(r["prediction"] is not None for r in rows) / n if n else None,
              "templates": dict(sorted(groups.items()))}
    name = "report.json" if report["complete"] else "report_partial.json"
    (out_dir / name).write_text(json.dumps(report, indent=2) + "\n")
    with (out_dir / "scored_rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    acc = f"{report['accuracy']:.4f}" if n else "n/a"
    print(f"{args.suite}: {n}/{len(data)} scored, accuracy {acc} -> {out_dir / name}", flush=True)


if __name__ == "__main__":
    main()
