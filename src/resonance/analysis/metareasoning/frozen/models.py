"""Model registry for the meta-reasoning arm.

One place that knows, per model, (a) where the weights are and (b) whether the chat
template prefills the opening <think> tag. That second fact is not cosmetic: it decides
where the thinking block starts, and getting it wrong silently mis-slices every trace.

Both values were verified against results.jsonl files already on disk, not assumed:

  Qwen3-30B-A3B-Thinking-2507 -- template PREFILLS <think>.
    ablation_30b/outputs/bench/bench_uspto_thinking/results.jsonl
    has "<think>" in 0/5000 responses and "</think>" in 4985/5000.

  Qwen3-8B -- template does NOT prefill it, the model emits it.
    eval_rigor/outputs/bench/bench_base_none/results.jsonl
    has "<think>" in 5000/5000 responses.

scripts/gate1_template.py re-checks both against the live tokenizers before any GPU job.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

CHEM = Path("data")
RUN_DIR = CHEM / "metareasoning_pilot"


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    path: str
    family: str  # "t30_thinking" | "q8"
    think_is_prefilled: bool
    label: str
    # Decode settings. Qwen3-*-Thinking-2507's own generation_config recommendation, and
    # what ablation_30b used. Applied to the 8B lineage too so that K-sampling
    # is meaningful -- this DOES break byte-comparability with the old greedy 8B numbers,
    # which every table carrying both must say out loud.
    sampling: Dict[str, float] = field(
        default_factory=lambda: {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    )
    max_model_len: int = 16384
    tensor_parallel_size: int = 2
    lora_path: Optional[str] = None
    notes: str = ""


REGISTRY: Dict[str, ModelSpec] = {
    # ---- 30B Thinking lineage (ablation_30b) ----
    "t30_base": ModelSpec(
        tag="t30_base",
        path="Qwen/Qwen3-30B-A3B-Thinking-2507",
        family="t30_thinking",
        think_is_prefilled=True,
        label="30B Thinking base",
    ),
    "t30_s1": ModelSpec(
        tag="t30_s1",
        path=str(CHEM / "ablation_30b/outputs/armB_s1_model"),
        family="t30_thinking",
        think_is_prefilled=True,
        label="30B S1 (Sonnet 40k SFT)",
    ),
    "t30_armA": ModelSpec(
        tag="t30_armA",
        path=str(CHEM / "ablation_30b/outputs/armA_model_step800"),
        family="t30_thinking",
        think_is_prefilled=True,
        label="30B arm A (RS-SFT from base, 0.70ep)",
        notes="checkpoint-800/1140; arm A's final save was lost, so this UNDERSTATES it.",
    ),
    "t30_armB": ModelSpec(
        tag="t30_armB",
        path=str(CHEM / "ablation_30b/outputs/armB_s1_model"),
        lora_path=str(CHEM / "ablation_30b/outputs/armB_adapter"),
        family="t30_thinking",
        think_is_prefilled=True,
        label="30B arm B (S1 then RS-SFT)",
        notes=(
            "NEVER MERGED (checked 2026-09-10). outputs/armB_model does not and never "
            "did exist; arm B is the LoRA adapter at outputs/armB_adapter (r=128, "
            "alpha=32, q/k/v/o) over base outputs/armB_s1_model. So `path` is the BASE "
            "and `lora_path` carries the arm -- any loader must pass both, with "
            "max_lora_rank=128. The adapter_config.json base_model_name_or_path is "
            "/outputs/run_thinking/... , a dead Modal container path; ignore it. "
            "Published score 58.50% on USPTO-50k test full."
        ),
    ),
    # ---- 8B lineage (base -> SFT -> RS-SFT -> DPO) ----
    "q8_base": ModelSpec(
        tag="q8_base",
        path="Qwen/Qwen3-8B",
        family="q8",
        think_is_prefilled=False,
        label="8B base",
    ),
    "q8_s1": ModelSpec(
        tag="q8_s1",
        path=str(CHEM / "sonnet_s1/outputs/model"),
        family="q8",
        think_is_prefilled=False,
        label="8B S1 (SFT)",
        notes=(
            "WEIGHTS PRESENT (re-checked 2026-09-10): 16G merged model at outputs/model. The "
            "earlier `WEIGHTS DELETED` note was wrong -- only the checkpoint-NNNN "
            "SUBDIRS were reclaimed, and this path used to point at checkpoint-3750, so "
            "is_available() returned False and silently skipped the model. New inference "
            "IS possible. Caveat unchanged: S1 was trained on its OWN bespoke prompt "
            "(see the header of eval_rigor/scripts/run_bench_native.py), "
            "so bench_s1_rules numbers are under a prompt it never saw -- prefer "
            "bench_s1_native for turn-1 reuse, and always label which."
        ),
    ),
    "q8_s2": ModelSpec(
        tag="q8_s2",
        path=str(CHEM / "sonnet_s2/outputs/model"),
        family="q8",
        think_is_prefilled=False,
        label="8B S2 (RS-SFT)",
        notes=(
            "WEIGHTS PRESENT (re-checked 2026-09-10): 16G merged model at outputs/model. The "
            "earlier `WEIGHTS DELETED` note was wrong -- only the checkpoint-NNNN "
            "SUBDIRS were reclaimed, and this path used to point at checkpoint-5085. "
            "The full 8B lineage base -> S1 -> S2 -> S3 is runnable again."
        ),
    ),
    "q8_s3": ModelSpec(
        tag="q8_s3",
        path=str(CHEM / "sonnet_s3/merged_model"),
        family="q8",
        think_is_prefilled=False,
        label="8B S3 (DPO)",
    ),
}

# Order for every table and figure: lineage order, 30B first.
ORDER: List[str] = [
    "t30_base",
    "t30_s1",
    "t30_armA",
    "t30_armB",
    "q8_base",
    "q8_s1",
    "q8_s2",
    "q8_s3",
]

# Models that also get the expensive secondary passes (instrument B, synthetic split,
# API-generated corruptions): one per lineage endpoint plus the two bases.
#
# q8_s3 (the merged DPO model) is the 8B representative as the lineage endpoint. Filtered
# through available() at call time so a missing checkpoint degrades to a skip rather than a
# load failure. (Until 2026-09-10 this comment said q8_s2 was picked because its weights had
# been reclaimed; that was false -- see the q8_s2 notes. q8_s3 stays the headline on merit.)
HEADLINE: List[str] = ["t30_base", "t30_s1", "q8_base", "q8_s3"]


def get(tag: str) -> ModelSpec:
    if tag not in REGISTRY:
        raise KeyError(f"unknown model tag {tag!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[tag]


def is_available(tag: str) -> bool:
    """True if the weights are on disk (or are a HF hub id, which vLLM resolves itself).

    Mirrors the skip logic in ablation_30b/bash_scripts/bench_uspto.sbatch:
    a config whose checkpoint has not been downloaded yet should be skipped, not
    allowed to burn a queue slot failing at model load.
    """
    p = get(tag).path
    if not p.startswith("/"):
        return True
    return (Path(p) / "config.json").is_file()


def available(tags: Optional[List[str]] = None) -> List[str]:
    return [t for t in (tags or ORDER) if is_available(t)]
