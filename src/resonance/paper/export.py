"""Write the resonance.plots inputs for Figures 2, 3 and 7.

Ported from run_dir('paper_figures')/scripts/export_main_text.py and the three exports
it used from run_dir('notebooks')/scripts/export_paper_plot_data.py (staged_accuracy,
class_progression, stage_ablation). The intermediate count files are rebuilt from the
raw results into <out>/inputs/ first, instead of being read from, and written back to,
the notebooks directory.

    python -m resonance.paper.export --out <dir>   # <dir>/Fig_{2,3,7}.json, <dir>/inputs/*.json
"""
import argparse
import json
from pathlib import Path

from resonance.paper import fig2_counts, fig3_27b, fig3_8b
from resonance.paths import DATA_ROOT, run_dir

# Family display names: short (axis) and long (legend).
FAMILY_SHORT = {
    "heteroatom alkylation/arylation": ("Alk./aryl.", "Heteroatom alk./aryl."),
    "protection chemistry": ("Protect.", "Protection chemistry"),
    "acylation": ("Acyl.", "Acylation"),
    "C-C bond formation": ("C–C", "C-C bond formation"),
    "redox": ("Redox", "Redox"),
    "FG changes": ("FG", "FG"),
    "heterocycle formation": ("Heterocyc.", "Heterocycle formation"),
}


def read(path):
    return json.loads(Path(path).read_text())


def rel(path):
    return str(Path(path).relative_to(DATA_ROOT))


def staged_accuracy(inputs):
    counts = read(inputs / "fig2_staged_counts.json")["runs"]
    s2_path = run_dir("s2_8b_bench") / "outputs/evaluation.json"
    s3_path = run_dir("s3_8b") / "outputs/benchmark/evaluation.json"
    s2, s3 = read(s2_path), read(s3_path)
    stages = [
        {"label": "base", "correct": counts["base"]["correct"], "n": counts["base"]["n"]},
        # The S3 report rescores the no-ID S1 predictions on the same 5,000 reactions.
        {"label": "S1", "correct": sum(r["s1_correct"] for r in s3["classes"]),
         "n": sum(r["rows"] for r in s3["classes"])},
        {"label": "S2", "correct": s2["correct"], "n": s2["total"]},
        {"label": "S3", "correct": s3["correct"], "n": s3["total"]},
    ]
    refs = [
        {"label": "Kimi\nK3", "correct": counts["kimi_teacher"]["correct"],
         "n": counts["kimi_teacher"]["n"]},
        # Frontier reference counts on the same 5,000 test reactions, as used in the
        # project's original figure notebook (resonance_figures.ipynb, REFS).
        {"label": "Opus\n5", "correct": 3910, "n": 5000},
        {"label": "GPT\n6", "correct": 3152, "n": 5000},
    ]
    return {"groups": [{"name": "Qwen3-8B (Kimi no-ID)", "kind": "stages", "bars": stages},
                       {"name": "Frontier", "kind": "references", "bars": refs}],
            "source": ["inputs/fig2_staged_counts.json", rel(s2_path), rel(s3_path)]}


def class_progression(inputs):
    d = read(inputs / "fig3_kimi_breakdown.json")
    stage_keys = {"base": "base", "s1": "S1", "s2": "S2", "s3": "S3"}
    fams = []
    for f in d["families"]:
        short, legend = FAMILY_SHORT[f["name"]]
        correct = {"teacher": f["correct"]["teacher"]}
        correct.update({label: f["correct"][key] for key, label in stage_keys.items()})
        fams.append({"name": f["name"], "short": short, "legend": legend,
                     "n_train": f["n_train"], "n_test": f["n_test"], "correct": correct})
    overall = {"n": d["overall"]["n"],
               "correct": {label: d["overall"]["correct"][key] for key, label in stage_keys.items()}}
    return {"stages": list(stage_keys.values()), "teacher": "Kimi K3", "families": fams,
            "overall": overall, "source": ["inputs/fig3_kimi_breakdown.json"]}


def stage_ablation(inputs):
    d = read(inputs / "fig3_27b_breakdown.json")
    short = {"heteroatom alkylation/arylation": "Heteroatom alk./aryl.",
             "protection chemistry": "Protection", "acylation": "Acylation",
             "C-C bond formation": "C–C bond formation", "redox": "Redox",
             "FG changes": "FG changes", "heterocycle formation": "Heterocycle form."}
    fams = [{"name": f["name"], "short": short[f["name"]], "n_test": f["n_test"],
             "correct": {k: v for k, v in f["correct"].items() if v is not None and k != "teacher"},
             "base_finished": f["finished"]["base"],
             "base_correct_finished": f["correct_finished"]["base"],
             "pairs": f["pairs"]} for f in d["families"]]
    return {"paths": [{"key": "s1", "label": "S1"}, {"key": "s2", "label": "S1→S2"},
                      {"key": "s3", "label": "S1→S2→S3"}, {"key": "skip_s2", "label": "S1→S3"},
                      {"key": "base_s3", "label": "S3"}, {"key": "skip_s1", "label": "S2→S3"}],
            "increments": [{"pair": "s1->s2", "label": "S2−S1"},
                           {"pair": "s2->s3", "label": "S3−S2"},
                           {"pair": "s1->s3", "label": "S3−S1"},
                           {"pair": "s1->skip_s2", "label": "S3−S1, no S2"}],
            "families": fams, "source": ["inputs/fig3_27b_breakdown.json"]}


# output stem -> exporter function
FIGURES = {"Fig_2": staged_accuracy,
           "Fig_3": class_progression,
           "Fig_7": stage_ablation}


def build_inputs(inputs):
    """Rebuild the three count files, each from the raw results and the previous one."""
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "fig2_staged_counts.json").write_text(json.dumps(fig2_counts.build(), indent=2) + "\n")
    (inputs / "fig3_kimi_breakdown.json").write_text(json.dumps(fig3_8b.build(inputs), indent=2) + "\n")
    (inputs / "fig3_27b_breakdown.json").write_text(json.dumps(fig3_27b.build(inputs), indent=2) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reuse-inputs", action="store_true",
                    help="use <out>/inputs/*.json as they are instead of rebuilding them")
    args = ap.parse_args(argv)
    inputs = args.out / "inputs"
    if not args.reuse_inputs:
        build_inputs(inputs)
    for stem, fn in FIGURES.items():
        (args.out / f"{stem}.json").write_text(
            json.dumps(fn(inputs), indent=2, ensure_ascii=False) + "\n")
        print("wrote", args.out / f"{stem}.json")


if __name__ == "__main__":
    main()
