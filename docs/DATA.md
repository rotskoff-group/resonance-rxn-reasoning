# Data

## Released datasets

Everything is in the Hugging Face dataset repo
[`anaghaa/resonance`](https://huggingface.co/datasets/anaghaa/resonance), under `v1/`.
Later work built on this release adds its own top-level directory (`v2/`, ...), so `v1/` stays fixed.
The git tag `v1` on the dataset repo marks the version described here.

| Config | Rows | Contents |
|---|---|---|
| `v1_uspto50k` | 40,000 / 5,000 / 5,000 | USPTO-50k train / val / test with patent ids, the starting material (first reactant), the reagents (other reactants) and the recorded product. |
| `v1_reaction_classes` | 40,000 + 5,000 | Rule-based reaction class per reaction, as published and with the benzyl-ether correction (appendix table "Reaction classes in USPTO-50k"). |
| `v1_kimi_k3_traces` | 40,000 | The S1 training data: one Kimi K3 trace per training reaction (product visible, no reaction id), with the teacher prompt fingerprint. sha256 of the source file: `b97ef67b…5f27e9`. |
| `v1_s2_8b` | 47,492 | Resonance-8B S2 training set: exactly correct S1 samples of at most 1,024 tokens (19,668 reactions). |
| `v1_s2_27b` | 98,731 | Resonance-27B S2 training set: difficulty mix of correct S1 samples, plus 1,108 answer-only gold fallbacks, with per-example provenance. |
| `v1_s3_pairs_8b` | 28,747 | Resonance-8B S3 DPO pairs (chosen = random correct S2 sample, rejected = most Tanimoto-similar incorrect one). |
| `v1_s3_pairs_27b` | 12,079 | Resonance-27B S3 DPO pairs. |
| `v1_skip_s2_pairs_8b` | 26,661 | DPO pairs sampled from 8B S1 (skip-S2 ablation). |
| `v1_skip_s2_pairs_27b` | 13,698 | DPO pairs sampled from 27B S1 (skip-S2 ablation). |
| `v1_test_predictions` | 5,000 per model | Every model's full response, extracted answer and correctness on the test set: both students at every stage, skip-S2, and Kimi K3, Claude Opus 5 and GPT-6-Astra. |
| `v1_reasoning_probes` | 5,000 × condition × draws | Resonance-8B S1–S3 samples behind §4.3: fresh resamples, the length prompts and the feedback conditions (8 draws each, per-reaction seeds). |

```python
from datasets import load_dataset
traces = load_dataset("anaghaa/resonance", "v1_kimi_k3_traces", split="train")
```

`v1/raw/<run_dir key>/<path>` holds the original files, at the paths the runs wrote them to.
No script reads the tidy tables. `resonance.download` copies `raw/` into
`RESONANCE_DATA_ROOT`, where `run_dir(key)` finds every file:

| Command | Places | Enough for |
|---|---|---|
| `python -m resonance.download figures` | the 63 files listed in `src/resonance/release/figure_inputs.txt` (4.3 GB): every model's test outputs, the reasoning-probe samples, the class labels, the teacher traces and the S1–S3 tokenizers | `resonance.paper.run_all` |
| `python -m resonance.download all` | everything in `raw/`. Beyond the figure inputs: the USPTO-50k split (`uspto/`), the S1 SFT dataset (`traces: outputs/sft_dataset_0a_no_reaction_id`), the S2 training sets (`s2_8b: outputs/sft_dataset`, `pipeline_27b: .../S2_V5_difficulty_mix_dataset`), the S3 and skip-S2 DPO pairs (`s3_8b`, `ablation_8b`, and the 27B S3 / skip-S2 datasets) | the workflows' paper configs ([PIPELINE.md](../PIPELINE.md)): retraining any 8B stage, benchmarking a model |

Both accept `--from-dir <dir>` for a local copy of the dataset repo (the directory that
contains `v1/`).

## Run directories (`resonance.paths.RUN_DIRS`)

Every script reads and writes through `run_dir(key)` = `$RESONANCE_DATA_ROOT/<key>`, and the paper
configs spell locations as `${run_dir:<key>}`. `resonance.download` fills these directories from
`v1/raw/`; a workflow run writes its outputs to the same places the original run did.

| Key | Holds |
|---|---|
| `uspto` | USPTO-50k `{train,val,test}.json[l]` in forward-reaction format (`pipelines.uspto`) |
| `traces` | Kimi K3 traces (`outputs/traces_stage1a_no_reaction_id.jsonl`), the S1 SFT dataset (`outputs/sft_dataset_0a_no_reaction_id`), the 8B S1 tokenizer |
| `s1_8b` | 8B S1 test outputs |
| `s2_8b`, `s2_8b_bench` | 8B S2 training set (`outputs/sft_dataset`), its manifest, the S2 tokenizer; 8B S2 test outputs |
| `s3_8b` | 8B S3 DPO pairs (`outputs/dpo_pairs.jsonl`), test outputs, tokenizer |
| `ablation_8b` | 8B skip-S2 pairs and test outputs |
| `pipeline_27b` | 27B base/S1/S2/S3 test outputs, the S2 training set and the S3 pairs (`outputs/no-id-b97ef67b9ba65863/...`) |
| `ablation_27b` | 27B skip-S2 pairs and test outputs |
| `eval_rigor` | 8B base test outputs (also the 0-shot arm of the few-shot control), published test labels |
| `teacher_bench`, `opus_bench`, `gpt6_bench` | Kimi K3, Claude Opus 5 and GPT-6-Astra zero-shot test outputs |
| `uspto_classes`, `classifier`, `train_classes`, `relabel` | Reaction-class labels as published and corrected (`pipelines.labels`); `relabel` also holds the Qwen3-8B base sample used as the backtracking reference |
| `metareasoning`, `paper_figures` | Reasoning-probe samples (`paper_figures: runs/per_reaction_seed*`), the corrected label copy, figure data |

The few-shot control (appendix table "Few-shot control on base Qwen3-8B") was run outside this
repository. Its 4- and 8-shot outputs are released in `v1_test_predictions`; the code that built the
exemplars is not included.
