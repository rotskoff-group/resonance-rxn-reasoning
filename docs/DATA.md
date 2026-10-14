# Data

## Released datasets

Everything is in the Hugging Face dataset repo
[`anaghaa/resonance`](https://huggingface.co/datasets/anaghaa/resonance), under `iclr2027/`.
Later papers built on this work add their own top-level directory, so `iclr2027/` stays fixed.
The tag `iclr2027-v1` marks the version described here.

| Config | Rows | Contents |
|---|---|---|
| `iclr2027_uspto50k` | 40,000 / 5,000 / 5,000 | USPTO-50k train / val / test with patent ids, the starting material (first reactant), the reagents (other reactants) and the recorded product. |
| `iclr2027_reaction_classes` | 40,000 + 5,000 | Rule-based reaction class per reaction, as published and with the benzyl-ether correction (appendix table "Reaction classes in USPTO-50k"). |
| `iclr2027_kimi_k3_traces` | 40,000 | The S1 training data: one Kimi K3 trace per training reaction (product visible, no reaction id), with the teacher prompt fingerprint. sha256 of the source file: `b97ef67b…5f27e9`. |
| `iclr2027_s2_8b` | 47,492 | Resonance-8B S2 training set: exactly correct S1 samples of at most 1,024 tokens (19,668 reactions). |
| `iclr2027_s2_27b` | 98,731 | Resonance-27B S2 training set: difficulty mix of correct S1 samples, plus 1,108 answer-only gold fallbacks, with per-example provenance. |
| `iclr2027_s3_pairs_8b` | 28,747 | Resonance-8B S3 DPO pairs (chosen = random correct S2 sample, rejected = most Tanimoto-similar incorrect one). |
| `iclr2027_s3_pairs_27b` | 12,079 | Resonance-27B S3 DPO pairs. |
| `iclr2027_skip_s2_pairs_8b` | 26,661 | DPO pairs sampled from 8B S1 (skip-S2 ablation). |
| `iclr2027_skip_s2_pairs_27b` | 13,698 | DPO pairs sampled from 27B S1 (skip-S2 ablation). |
| `iclr2027_test_predictions` | 5,000 per model | Every model's full response, extracted answer and correctness on the test set: both students at every stage, skip-S2, and Kimi K3, Claude Opus 5 and GPT-6-Astra. |
| `iclr2027_reasoning_probes` | 5,000 × condition × draws | Resonance-8B S1–S3 samples behind §4.3: fresh resamples, the length prompts and the feedback conditions (8 draws each, per-reaction seeds). |

```python
from datasets import load_dataset
traces = load_dataset("anaghaa/resonance", "iclr2027_kimi_k3_traces", split="train")
```

`iclr2027/raw/<run_dir key>/<path>` holds the original files, at the paths the runs wrote them to.
No script reads the tidy tables. `resonance.download` copies `raw/` into
`RESONANCE_DATA_ROOT`, where `run_dir(key)` finds every file:

| Command | Places | Enough for |
|---|---|---|
| `python -m resonance.download figures` | the 63 files listed in `src/resonance/release/figure_inputs.txt` (4.3 GB): every model's test outputs, the reasoning-probe samples, the class labels, the teacher traces and the S1–S3 tokenizers | `resonance.paper.run_all` |
| `python -m resonance.download all` | everything in `raw/`. Beyond the figure inputs: the USPTO-50k split (`uspto/`), the S2 training sets (`s2_8b: outputs/sft_dataset`, `pipeline_27b: .../S2_V5_difficulty_mix_dataset`), the S3 and skip-S2 DPO pairs (`s3_8b`, `ablation_8b`, and the 27B S3 / skip-S2 datasets) | rerunning any training stage from the previous stage's released outputs |

Both accept `--from-dir <dir>` for a local copy of the dataset repo (the directory that
contains `iclr2027/`).

## Run directories (`resonance.paths.RUN_DIRS`)

Every script reads and writes through `run_dir(key)` = `$RESONANCE_DATA_ROOT/<key>`. If you
run a stage yourself, it writes to the same place the original run did, and the next stage
finds it there.

| Key | Holds | Written by | In the paper |
|---|---|---|---|
| `uspto` | `{train,val,test}.json[l]` in forward-reaction format | `pipelines.uspto.convert_to_forward_rxn_format` | yes |
| `traces` | Kimi K3 traces (`outputs/traces_stage1a_no_reaction_id.jsonl`), 8B S1 SFT data and model | `pipelines.traces`, `stages.qwen3_8b.s1_prep`, `workflow sft` | yes |
| `s1_8b` | 8B S1 benchmark and the 320,000 S2-data rollouts | `stages.qwen3_8b` | yes |
| `s2_8b`, `s2_8b_bench` | 8B S2 data, model (`outputs/model/checkpoint-2229`), benchmark | `stages.qwen3_8b` | yes |
| `s3_8b` | 8B S3 completions, DPO pairs, adapter, merged model, benchmark | `stages.qwen3_8b` | yes |
| `ablation_8b` | 8B skip-S2: S1 completions, pairs, adapter, merged model, benchmark | stage-ablation run | yes (appendix) |
| `s1_27b`, `pipeline_27b` | 27B S1 metadata; base / S1 evaluations, S2 data and runs, S3 rollouts and runs (`outputs/no-id-b97ef67b9ba65863/...`) | `stages.qwen3_8_27b` | yes |
| `ablation_27b` | 27B skip-S2 run and its local evaluation | `stages.qwen3_8_27b.ablation` | yes (appendix) |
| `eval_rigor` | 8B base benchmark (0-shot arm of the few-shot control), published test labels (`outputs/audit/test_reaction_classes.json`) | evaluation audit, `pipelines.labels.test_reaction_classes` | yes |
| `teacher_bench`, `opus_bench`, `gpt6_bench` | Kimi K3, Claude Opus 5 and GPT-6-Astra zero-shot test results | API / endpoint runs | yes |
| `uspto_classes` | Test-set class labels used by the stage evaluations' per-class reports (`outputs/uspto_labels_test.json`) | class-label study | yes |
| `classifier`, `train_classes`, `relabel` | Reaction classifier runs, published training labels, the benzyl-ether relabel; `relabel` also holds the Qwen3-8B base sample used as the backtracking reference | `pipelines.labels` | yes |
| `metareasoning`, `paper_figures` | Reasoning-probe samples (`paper_figures: runs/per_reaction_seed*`), relabel copy, figure data | `paper.regenerate`, `paper.run_all` | yes |
| `answer_only_27b` | 27B answer-only control | `stages.qwen3_8_27b` | no |
| `pubchem`, `forward_enum`, `retro_enum_subsets`, `synthetic_audit`, `synthetic_ood`, `synthetic_s2_27b` | Synthetic out-of-distribution benchmark (PubChem forward enumeration) | `pipelines.{forward_enum,synthetic_data,synthetic_ood}` | **no**, from an earlier draft |
| `notebooks` | Exploratory notebooks | – | no |

The few-shot control (appendix table "Few-shot control on base Qwen3-8B") was run outside this repository. Its 4- and
8-shot outputs and scores are released in `iclr2027_test_predictions`, but the code that built
the exemplars and scored them is not included.
