"""Regenerate the Kimi no-ID 8B metareasoning samples with per-reaction seeds.

Ported from run_dir('paper_figures')/scripts/regenerate_metareasoning.py.

Same models, prompts, decode and budgets as the original metareasoning pipeline, but
calls resonance.analysis.metareasoning.{resample,runner}, which seed each reaction
separately (seed + eval_idx). The earlier samples shared one seed across all
reactions. Outputs go under --out-root; nothing is overwritten.

    python -m resonance.paper.regenerate fresh  --tag kimi_s1 [--smoke]
    python -m resonance.paper.regenerate review --tag kimi_s1 --group length [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

import resonance.analysis.metareasoning as meta
from resonance.paper import config as paper_config
from resonance.paths import run_dir

PKG = Path(meta.__file__).resolve().parent
CFG = paper_config.load("metareasoning_eval.json")
SRC = Path(CFG["run_dir"])  # holds data/eval_greedy.json (resonance.paper.prepare_eval)
DEFAULT_OUT = run_dir("paper_figures") / "runs/per_reaction_seed"
GROUPS = {
    # Reasoning-length prompts: eight rollouts each for question-level SEs.
    "length": ("B2_commit_first,B2_consider_alts,C6_longer", 8),
    # Feedback conditions and their neutral-review control: one draw each.
    "feedback": ("C1_review_full,C4a_mild,C4_oracle_wrong,C4c_authority,C7_told_correct,"
                 "C4t_oracle_wrong_trace", 1),
}


def load(name):
    # resample/runner put frozen/ on sys.path and import vllm at module load.
    import importlib
    return importlib.import_module(f"resonance.analysis.metareasoning.{name}")


def register_models():
    sys.path.insert(0, str(PKG / "frozen"))
    import models
    for name, spec in CFG["models"].items():
        models.REGISTRY[name] = models.ModelSpec(tag=name, path=spec["path"], family="q8",
                                                 think_is_prefilled=spec["think_is_prefilled"],
                                                 label=name + " no-ID")


def work_dir(root, smoke):
    work = root / ("smoke" if smoke else "full")
    (work / "data").mkdir(parents=True, exist_ok=True)
    return work


def fresh(tag, root, smoke):
    work = work_dir(root, smoke)
    register_models()
    resample = load("resample")
    records = json.loads((SRC / "data/eval_greedy.json").read_text())
    base = SRC / "data/eval_greedy.json"
    if smoke:
        base = work / f"data/input_{tag}.json"
        base.write_text(json.dumps(resample.stratified(records, 24, CFG["seed"])))
    resample.RUN = work
    resample.TAG_MODEL = {name: name for name in CFG["models"]}
    sys.argv = [str(PKG / "resample.py"), "--tags", tag, "--base-data", str(base),
                "--out", str(work / f"data/eval_{tag}.json"),
                "--fresh-dir", str(work / "outputs/fresh_t06"), "--k", str(CFG["k"]),
                "--tp", "1", "--gpu-mem", str(CFG["gpu_mem"]), "--seed", str(CFG["seed"]),
                "--seed-mode", "per_reaction",
                "--max-tokens", str(CFG["max_tokens"]), "--max-model-len", str(CFG["max_model_len"])]
    resample.main()


def review(tag, group, root, smoke):
    work = work_dir(root, smoke)
    conditions, rollouts = GROUPS[group]
    data = work / f"data/eval_{tag}.json"
    if not data.exists():
        raise SystemExit(f"{data} missing: run `fresh --tag {tag}` first")
    register_models()
    runner = load("runner")
    runner.RUN = work
    sys.argv = [str(PKG / "runner.py"), "--model-tag", tag, "--model-path", CFG["models"][tag]["path"],
                "--think-prefilled", "no", "--data", str(data), "--turn1-tag", tag,
                "--out-dir", str(work / f"outputs/t06_{tag}"), "--conditions", conditions,
                "--rollouts", str(rollouts), "--tp", "1", "--gpu-mem", str(CFG["gpu_mem"]),
                "--max-tokens", str(CFG["max_tokens"]), "--max-model-len", str(CFG["max_model_len"]),
                "--seed", str(CFG["seed"]), "--seed-mode", "per_reaction", "--verify-mode", "forced"]
    runner.main()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["fresh", "review"])
    ap.add_argument("--tag", required=True, choices=sorted(CFG["models"]))
    ap.add_argument("--group", choices=sorted(GROUPS))
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.action == "fresh":
        fresh(args.tag, args.out_root, args.smoke)
    else:
        if not args.group:
            ap.error("review needs --group")
        review(args.tag, args.group, args.out_root, args.smoke)


if __name__ == "__main__":
    main()
