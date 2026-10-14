"""Rebuild every figure in the paper (Figs 2-7, A1, A2) from the underlying results.

Ported from run_dir('paper_figures')/run_all.sh; same steps, same order:

  1. frozen generation/scoring scripts match MANIFEST.json (and, with --protocol, a run's record)
  1b. regenerated samples are independent across reactions (fails on shared seeds)
  2. metareasoning statistics and plot inputs (Figs 4-6)
  3. per-class counts for every model (Figs 2, 3, 7; corrected class labels, lenient scorer)
  4. render Figs 2-7
  5. backtracking rates and the appendix figures (A1, A2)

    python -m resonance.paper.run_all --out <dir>     # <dir>/data, <dir>/figures

Everything is written under --out. Inputs resolve through `resonance.paths.run_dir`; the
figure inputs are fetched by `python -m resonance.download figures`. Fig 1 (the pipeline
schematic) is drawn by hand and is not generated.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from resonance.paper import config as paper_config
from resonance.paths import run_dir


def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", default="kimi_no_id_paper.json",
                    help="summary config (path, or a name in configs/paper/figures)")
    ap.add_argument("--protocol", type=Path,
                    help="run protocol record to check frozen/ against (optional)")
    args = ap.parse_args(argv)
    data, figures = args.out / "data", args.out / "figures"
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "paper-figures-mpl"))
    config = paper_config.write_resolved(args.config, data / "summary_config.resolved.json")

    # 1-2. Metareasoning (Figs 4-6).
    run("-m", "resonance.analysis.metareasoning.verify", *(["--protocol", args.protocol] if args.protocol else []))
    run("-m", "resonance.paper.draw_independence", "--config", config, "--out", data)
    run("-m", "resonance.analysis.metareasoning.summary", "--config", config, "--out", data)

    # 3. Per-class counts. The corrected training labels take ~40 min to derive, so they are
    #    used from run_dir('paper_figures') when present and otherwise derived into <out>/data.
    if not (run_dir("paper_figures") / "data/train_class_relabel.json").exists():
        sys.exit("missing run_dir('paper_figures')/data/train_class_relabel.json: "
                 "run `python -m resonance.pipelines.labels.relabel_train --out <path>` "
                 "or fetch it with `python -m resonance.download figures`")
    run("-m", "resonance.paper.classes", "--out", data)
    run("-m", "resonance.paper.export", "--data", data)

    # 4. Render. Figs 2, 3, 7 in the 5.5 in house style; Figs 4-6 in the main-text style.
    run("-m", "resonance.plots.run", "--data-dir", data, "--out-dir", figures, "--style", "default",
        "--map", "Fig_2=staged_accuracy", "--only", "Fig_2")
    run("-m", "resonance.paper.fig3", "--data", data, "--out", figures)
    run("-m", "resonance.paper.fig7", "--data", data, "--out", figures)
    run("-m", "resonance.plots.run", "--data-dir", data, "--out-dir", figures, "--style", "main_text",
        "--map", "Fig_4=length_prompts", "Fig_5=feedback_delta", "Fig_6=plurality",
        "--only", "Fig_4", "Fig_5", "Fig_6")

    # 5. Appendix backtracking (all 8 draws per reaction).
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    print("+ -m resonance.paper.backtrack --out", data, flush=True)
    subprocess.run([sys.executable, "-m", "resonance.paper.backtrack", "--out", str(data)], check=True, env=env)
    run("-m", "resonance.paper.fig_appendix", "--data", data / "Fig_A_backtrack.json", "--out", figures)


if __name__ == "__main__":
    main()
