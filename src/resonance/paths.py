"""Where the paper pipeline's inputs and outputs live.

Every stage script resolves its directories through `run_dir(key)`, never through a
literal path, so moving the data (or anonymizing it) means editing this file only.

    RESONANCE_DATA_ROOT   root holding every run directory (default: DATA_ROOT below)
"""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("RESONANCE_DATA_ROOT", "data"))

# key -> directory name under DATA_ROOT.
RUN_DIRS = {
    # data
    "uspto": "uspto",                           # USPTO-50k train/test json
    "traces": "traces",                         # Kimi K3 no-ID teacher traces + 8B S1
    # reaction classes
    "uspto_classes": "uspto_classes",
    "classifier": "classifier",
    "eval_rigor": "eval_rigor",                 # test labels + 8B base benchmark
    "train_classes": "train_classes",
    "relabel": "relabel",
    # references
    "teacher_bench": "teacher_bench",           # Kimi K3 teacher benchmark
    "opus_bench": "opus_bench",                 # Claude Opus 5 benchmark
    "gpt6_bench": "gpt6_bench",                 # GPT-6-Astra benchmark
    # Qwen3-8B lineage
    "s1_8b": "s1_8b",                           # S1 benchmark + rollouts
    "s2_8b": "s2_8b",                           # S2 filter + training
    "s2_8b_bench": "s2_8b_bench",               # S2 benchmark
    "s3_8b": "s3_8b",                           # S3 generation, mining, DPO, benchmark
    "ablation_8b": "ablation_8b",               # skip-S2 stage ablation (DPO on S1)
    # Qwen3.8-27B lineage
    "pipeline_27b": "pipeline_27b",             # base, S1 eval, S2, S3
    "ablation_27b": "ablation_27b",             # skip-S2 stage ablation (DPO on S1)
    # analysis
    "metareasoning": "metareasoning",
    "paper_figures": "paper_figures",
}


def run_dir(key: str) -> Path:
    return DATA_ROOT / RUN_DIRS[key]
