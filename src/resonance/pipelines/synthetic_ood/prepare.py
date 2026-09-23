"""Rebuild the forward synthetic test splits byte-identically to the 27B S1 synthetic eval.

Ported from run_dir('synthetic_ood')/scripts/prepare.py.
CLI: python -m resonance.pipelines.synthetic_ood.prepare [--out-root DIR]
     writes DIR/data/synthetic_alpha_{0,0.5,1}/{test,labels}.json (default DIR: run_dir('synthetic_ood'))
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from resonance.paths import run_dir

ROOT = run_dir("synthetic_ood")
MANIFEST = run_dir("forward_enum") / "outputs/subsets_60k/manifest.json"
S1_EVAL = run_dir("synthetic_s2_27b") / "outputs/s1-synthetic-baseline/eval"


def build(alpha):
    # Same row construction as the 27B S1 synthetic benchmark (benchmark_s1_synthetic.py).
    path = MANIFEST.parent / f"alpha_{alpha}/60k/test.jsonl.gz"
    rows, labels = [], {}
    with gzip.open(path, "rt") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            reactants = row["reactant_smiles"]
            assert isinstance(reactants, list) and reactants
            rows.append({"original_smiles": {
                "Starting Material": reactants[0], "Reagents": ".".join(reactants[1:]),
                "End Product": row["product_smiles"]}})
            labels[str(idx)] = {"level1": f"{row['template_id']}: {row['template_name']}",
                                "template_id": row["template_id"], "template_name": row["template_name"]}
    return json.dumps(rows).encode(), json.dumps({"labels": labels}).encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT)
    root = ap.parse_args().out_root
    for alpha in ("0", "0.5", "1"):
        suite = f"synthetic_alpha_{alpha}"
        test_raw, labels_raw = build(alpha)
        reference = json.loads((S1_EVAL / suite / "input.json").read_text())
        assert hashlib.sha256(test_raw).hexdigest() == reference["test_sha256"], suite
        assert hashlib.sha256(labels_raw).hexdigest() == reference["labels_sha256"], suite
        out = root / "data" / suite
        out.mkdir(parents=True, exist_ok=True)
        (out / "test.json").write_bytes(test_raw)
        (out / "labels.json").write_bytes(labels_raw)
        print(suite, reference["rows"], "rows, hashes match S1 eval")


if __name__ == "__main__":
    main()
