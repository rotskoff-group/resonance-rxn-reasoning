"""Where the released models and data live on the Hugging Face Hub.

One place for every repository id, so the README, the model and dataset cards and
`resonance.download` cannot disagree.
"""

DATASET_REPO = "anaghaa/resonance"
RELEASE = "v1"   # every file of this release sits under <RELEASE>/ in DATASET_REPO; later releases add v2, ...

# Paper name -> (Hub model repo, base model, what it is). The USPTO-50k test accuracy is the
# paper's (appendix tables "Paired tests between stages" and "Skip-S2 ablation").
MODELS = {
    "Resonance-8B-S1": ("anaghaa/Resonance-8B-S1", "Qwen/Qwen3-8B", "teacher SFT", 31.0),
    "Resonance-8B-S2": ("anaghaa/Resonance-8B-S2", "Qwen/Qwen3-8B", "+ rejection-sampling SFT", 38.9),
    "Resonance-8B-S3": ("anaghaa/Resonance-8B-S3", "Qwen/Qwen3-8B", "+ DPO (final model)", 42.8),
    "Resonance-8B-skip-S2": ("anaghaa/Resonance-8B-skip-S2", "Qwen/Qwen3-8B", "ablation: DPO on S1", 37.1),
    "Resonance-27B-S1": ("anaghaa/Resonance-27B-S1", "Qwen/Qwen3.8-27B", "teacher SFT", 79.0),
    "Resonance-27B-S2": ("anaghaa/Resonance-27B-S2", "Qwen/Qwen3.8-27B", "+ rejection-sampling SFT", 79.3),
    "Resonance-27B-S3": ("anaghaa/Resonance-27B-S3", "Qwen/Qwen3.8-27B", "+ DPO (final model)", 79.6),
    "Resonance-27B-skip-S2": ("anaghaa/Resonance-27B-skip-S2", "Qwen/Qwen3.8-27B", "ablation: DPO on S1", 78.8),
}
