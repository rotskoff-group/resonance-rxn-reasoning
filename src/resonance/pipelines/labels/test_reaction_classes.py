"""Reaction class of every USPTO-50k test reaction (the cached test labels).

Ported from the class-cache part of run_dir('eval_rigor')/scripts/error_trajectory.py:
CLASS_NAMES, class_label() and reaction_classes() verbatim; the per-stage error trajectory
in that script is not part of the paper. Note the code path: classify() returns
(code, name) and label[0] -- the numeric code -- is mapped through CLASS_NAMES.

CLI: python -m resonance.pipelines.labels.test_reaction_classes [--test-json T] [--out O]
"""

import argparse
import json
import os

from rdkit import rdBase

from resonance.paths import run_dir
from resonance.pipelines.labels import classify_reactions

rdBase.DisableLog("rdApp.*")

TEST_JSON = str(run_dir("uspto") / "test.json")
CLASS_CACHE = str(run_dir("eval_rigor") / "outputs" / "audit" / "test_reaction_classes.json")

# classify_reactions.classify() returns the numeric Schneider superclass. Map to names so the
# tables and figures are readable. Verified against the known USPTO-50k proportions: class 1
# is the largest (1479 / 29.6% of test) and class 4 has n=102.
CLASS_NAMES = {
    "0": "unclassified",          # classify() returns 0 when no superclass rule fires
    "1": "heteroatom alkylation/arylation",
    "2": "acylation",
    "3": "C-C bond formation",
    "4": "heterocycle formation",
    "5": "protections",
    "6": "deprotections",
    "7": "reductions",
    "8": "oxidations",
    "9": "functional group interconversion",
    "10": "functional group addition",
}


def class_label(code):
    return CLASS_NAMES.get(str(code), str(code))


def reaction_classes(records):
    if os.path.exists(CLASS_CACHE):
        cached = json.load(open(CLASS_CACHE))
        if cached.get("n") == len(records):
            print(f"  using cached classes ({CLASS_CACHE})")
            # older caches stored the bare numeric code for the no-rule-fired case
            return [class_label(c) if c in ("0", 0) else c for c in cached["classes"]]

    print(f"  classifying {len(records)} test reactions ...")
    classes = []
    for i, rec in enumerate(records):
        o = rec["original_smiles"]
        try:
            label = classify_reactions.classify(
                o.get("Starting Material", ""),
                o.get("Reagents", "") or "",
                o.get("End Product", ""),
            )
            if isinstance(label, (tuple, list)):
                label = label[0]
        except Exception:                                   # noqa: BLE001
            label = "unclassified"
        classes.append(class_label(label) if label is not None else "unclassified")
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{len(records)}")
    os.makedirs(os.path.dirname(CLASS_CACHE), exist_ok=True)
    with open(CLASS_CACHE, "w") as f:
        json.dump({"n": len(records), "classes": classes}, f)
    return classes


def main():
    global CLASS_CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-json", default=TEST_JSON)
    ap.add_argument("--out", default=CLASS_CACHE,
                    help="an existing file with matching n is reused, not recomputed")
    ap.add_argument("--encoding", choices=["published", "current"], default="published",
                    help="published: write the no-rule-fired class as the bare code \"0\", as the "
                         "published cache (written by an earlier revision) does and as the figure "
                         "builders' corrected_labels mapping expects; current: \"unclassified\"")
    a = ap.parse_args()
    CLASS_CACHE = a.out
    records = json.load(open(a.test_json))
    classes = reaction_classes(records)
    if a.encoding == "published" and "unclassified" in classes:
        classes = ["0" if c == "unclassified" else c for c in classes]
        with open(CLASS_CACHE, "w") as f:
            json.dump({"n": len(records), "classes": classes}, f)
    print(f"{len(classes)} labels -> {CLASS_CACHE}")


if __name__ == "__main__":
    main()
