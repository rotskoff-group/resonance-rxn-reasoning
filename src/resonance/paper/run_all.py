"""Rebuild every main-text figure from the underlying results.

Ported from run_dir('paper_figures')/run_all.sh; same steps, same order:

  1. frozen generation/scoring scripts match MANIFEST.json (and, with --protocol, a run's record)
  1b. regenerated samples are independent across reactions (fails on shared seeds)
  2. metareasoning statistics and plot inputs (Figs 4-6)
  3. stage accuracy and class figures (Figs 2, 3, 7; corrected class labels)
  4. render: Figs 2, 3, 7 in the 5.5 in house style, Figs 4-6 in the main-text style

    python -m resonance.paper.run_all --out <dir>     # <dir>/data, <dir>/figures

Metareasoning inputs come from the per-reaction-seed runs (`resonance.paper.regenerate`).
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from resonance.paper import config as paper_config



def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", default="metareasoning_summary.json",
                    help="summary config (path, or a name in configs/paper/figures)")
    ap.add_argument("--protocol", type=Path,
                    help="run protocol record to check frozen/ against (optional)")
    args = ap.parse_args(argv)
    data, figures = args.out / "data", args.out / "figures"
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "paper-figures-mpl"))
    config = paper_config.write_resolved(args.config, data / "metareasoning_summary.resolved.json")

    verify = ["-m", "resonance.analysis.metareasoning.verify"]
    run(*verify, *(["--protocol", args.protocol] if args.protocol else []))
    run("-m", "resonance.paper.draw_independence", "--config", config, "--out", data)
    run("-m", "resonance.analysis.metareasoning.summary", "--config", config, "--out", data)
    run("-m", "resonance.paper.export", "--out", data)
    run("-m", "resonance.plots.run", "--data-dir", data, "--out-dir", figures, "--style", "default",
        "--map", "Fig_2=staged_accuracy", "Fig_3=class_progression", "Fig_7=stage_ablation",
        "--only", "Fig_2", "Fig_3", "Fig_7")
    run("-m", "resonance.plots.run", "--data-dir", data, "--out-dir", figures, "--style", "main_text",
        "--map", "Fig_4=length_prompts", "Fig_5=feedback_delta", "Fig_6=plurality",
        "--only", "Fig_4", "Fig_5", "Fig_6")


if __name__ == "__main__":
    main()
