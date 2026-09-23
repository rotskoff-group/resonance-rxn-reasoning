"""Benzyl-ether-corrected test reaction classes, in test_reaction_classes.json naming.

Ported from run_dir('notebooks')/scripts/corrected_labels.py.

The classifier's benzyl_ether SMARTS matched benzyl alcohols; relabel_classes.py
(resonance.analysis.metareasoning.frozen) re-derives the labels. 282/5,000 test
reactions change class. Training-corpus labels are not corrected.
"""
import json

from resonance.paths import DATA_ROOT, run_dir

RELABEL = str((run_dir("relabel") / "data/class_relabel_v2.json").relative_to(DATA_ROOT))
# relabel names -> test_reaction_classes.json names (one-to-one on the published labels).
NAME = {"acylation": "acylation", "cc_bond_formation": "C-C bond formation",
        "deprotections": "deprotections", "fga": "functional group addition",
        "fgi": "functional group interconversion",
        "heteroatom_alkylation_arylation": "heteroatom alkylation/arylation",
        "heterocycle_formation": "heterocycle formation", "oxidations": "oxidations",
        "protections": "protections", "reductions": "reductions", "unclassified": "0"}


def corrected_test_classes(published):
    """Corrected class per test index; checks the published labels match `published`."""
    labels = json.loads((DATA_ROOT / RELABEL).read_text())["labels"]
    assert len(labels) == len(published) == 5000
    for i, c in enumerate(published):
        assert NAME[labels[str(i)]["published"]] == c, f"published label mismatch at {i}"
    return [NAME[labels[str(i)]["corrected"]] for i in range(len(published))]
