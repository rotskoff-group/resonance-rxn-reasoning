"""Where the paper pipeline's inputs and outputs live.

Every stage script resolves its directories through `run_dir(key)`, never through a
literal path, so moving the data (or anonymizing it) means editing this file only.

    RESONANCE_DATA_ROOT   root holding every run directory (default: DATA_ROOT below)
    RESONANCE_HF_CACHE    Hugging Face cache used by the GPU launchers
"""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("RESONANCE_DATA_ROOT", "data"))
HF_CACHE = Path(os.environ.get("RESONANCE_HF_CACHE", DATA_ROOT.parent / "hf_cache"))

# key -> directory name under DATA_ROOT.
RUN_DIRS = {
    # data
    "uspto": "uspto",                           # USPTO-50k train/test json
    "pubchem": "pubchem",                       # input molecules for enumeration
    "forward_enum": "forward_enum",             # synthetic forward-enumeration corpus
    "retro_enum_subsets": "retro_enum_subsets", # subset_sample helpers' origin
    "synthetic_audit": "synthetic_audit",       # ground-truth audit helpers' origin
    "synthetic_ood": "synthetic_ood",           # synthetic OOD evaluation
    "synthetic_s2_27b": "synthetic_s2_27b",     # OOD test-set hash reference
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
    "s1_27b": "s1_27b",
    "pipeline_27b": "pipeline_27b",             # base, S1 eval, S2, S3
    "ablation_27b": "ablation_27b",             # skip-S2 stage ablation (DPO on S1)
    "answer_only_27b": "answer_only_27b",
    # analysis
    "metareasoning": "metareasoning",
    "paper_figures": "paper_figures",
    "notebooks": "notebooks",
}


def run_dir(key: str) -> Path:
    return DATA_ROOT / RUN_DIRS[key]
