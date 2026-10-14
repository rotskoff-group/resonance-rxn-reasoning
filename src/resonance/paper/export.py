"""Write Fig_2.json, the resonance.plots input for Figure 2, from the scored class counts.

Ported from run_dir('paper_figures')/scripts/export_main_text.py (its Fig 2 path). Every
bar is an overall count in class11_counts.json (`resonance.paper.classes`), so the 8B
stages, Kimi K3, Opus 5 and GPT-6-Astra are all scored by the same lenient extractor from
their raw responses. Figures 3 and 7 are drawn straight from class11_counts.json
(`resonance.paper.fig3`, `resonance.paper.fig7`).

    python -m resonance.paper.export --data <dir>   # reads <dir>/class11_counts.json, writes <dir>/Fig_2.json
"""
import argparse
import json
from pathlib import Path

STAGES = [("base", "8b_base"), ("S1", "8b_s1"), ("S2", "8b_s2"), ("S3", "8b_s3")]
REFERENCES = [("Kimi\nK3", "kimi"), ("Opus\n5", "opus"), ("GPT\n6", "gpt6")]
STAGE_GRADIENT = ["#9ecae1", "#4a98c9", "#1f5fa8", "#0b2e63"]   # base -> S3, as in Fig 7


def staged_accuracy(counts):
    overall = counts["overall"]
    stages = [{"label": label, "correct": overall["correct"][key], "n": overall["n"], "color": color}
              for (label, key), color in zip(STAGES, STAGE_GRADIENT, strict=True)]
    refs = [{"label": label, "correct": overall["correct"][key], "n": overall["n"]}
            for label, key in REFERENCES]
    return {"groups": [{"name": "Qwen3-8B (Kimi no-ID)", "kind": "stages", "bars": stages},
                       {"name": "Frontier", "kind": "references", "bars": refs}],
            "source": ["class11_counts.json"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args(argv)
    counts = json.loads((args.data / "class11_counts.json").read_text())
    out = args.data / "Fig_2.json"
    out.write_text(json.dumps(staged_accuracy(counts), indent=2, ensure_ascii=False) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
