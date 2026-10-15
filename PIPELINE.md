# Training and evaluation

How the released models were trained and evaluated, and how to rerun each step with this
repository's workflows. Terms are defined in [docs/GLOSSARY.md](docs/GLOSSARY.md); the released
data and the `run_dir` layout are in [docs/DATA.md](docs/DATA.md).

```
USPTO-50k ──► teacher traces (Kimi K3) ──► S1 SFT ──► S2 rejection-sampling SFT ──► S3 DPO + NLL
                                              │                 │                        │
                                              └──── benchmark on the USPTO-50k test set ─┘
reaction-class labels ──► per-class analysis ──► Figs 2, 3, 7
8B S1/S2/S3 ──► reasoning probes (length prompts, feedback, resampling) ──► Figs 4–6, A1, A2
```

## What is in this repository

| Path | Contents |
|---|---|
| `src/resonance/workflows/` | Hydra entry points: `sft`, `dpo`, `benchmark`, `evaluate`. Importing the package registers the `${run_dir:<key>}` config resolver. |
| `src/resonance/configs/paper/qwen3_8b/` | The configs of the published 8B runs (S1 SFT, S2 SFT, S3 DPO, S1/S2 benchmarks). |
| `src/resonance/configs/paper/qwen3_8_27b/prompts.json` | The 27B evaluation prompts. |
| `src/resonance/data/`, `training/`, `inference/`, `evaluation/` | Parsers and prompts, trainer construction, vLLM inference, and the scorer used by every number in the paper. |
| `src/resonance/pipelines/uspto/` | USPTO-50k to the forward-reaction JSON the workflows read. |
| `src/resonance/pipelines/traces/` | Kimi K3 teacher-trace generation: frozen prompts with fingerprints, endpoints, corpus validation. |
| `src/resonance/pipelines/labels/` | Reaction-class labels for train and test, with the benzyl-ether correction (`relabel_classes`, `relabel_train`). |
| `src/resonance/analysis/`, `paper/`, `plots/` | The reasoning probes, the backtracking detector, and every paper figure ([docs/FIGURES.md](docs/FIGURES.md)). |
| `src/resonance/paths.py` | Every data location: `RESONANCE_DATA_ROOT` plus one key per run directory. |

The scripts that scheduled these runs on the authors' machines and on Modal are not included.
The released datasets cover every training set they produced (below).

## Rerunning the 8B stages

```bash
export RESONANCE_DATA_ROOT=$PWD/data
uv sync --frozen --extra train
python -m resonance.download all          # USPTO-50k, the S1/S2 training sets, the S3 pairs, ...
C=src/resonance/configs/paper/qwen3_8b
python -m resonance.workflows.sft       --config-path $PWD/$C --config-name s1_sft
python -m resonance.workflows.sft       --config-path $PWD/$C --config-name s2_sft resume_from_checkpoint=null
python -m resonance.workflows.dpo       --config-path $PWD/$C --config-name s3_dpo
python -m resonance.workflows.benchmark --config-path $PWD/$C --config-name s1_benchmark
```

Any config value can be overridden on the command line. For example,
`model_name=anaghaa/Resonance-8B-S1` starts S2 from the released S1 checkpoint instead of a
local one.

| Stage | Config | Reads (released in `v1/raw`) | Paper settings |
|---|---|---|---|
| S1 SFT | `s1_sft` | `traces: outputs/sft_dataset_0a_no_reaction_id` (the 40,000 teacher traces, chat-formatted) | full fine-tune, 3 epochs (3,750 steps), lr 2e-5, warmup 100, effective batch 32, max length 2,048 |
| S2 SFT | `s2_sft` | `s2_8b: outputs/sft_dataset` (47,492 examples) | full fine-tune, 3 epochs (2,229 steps), lr 1e-5, warmup 50, effective batch 64, max length 1,024 |
| S3 DPO | `s3_dpo` | `s3_8b: outputs/dpo_pairs.jsonl` (28,747 pairs) | LoRA r128 / α32 / dropout 0.05, 3,000 steps, lr 2e-6, β 0.1, DPO:NLL 1.0:0.2, 32 pairs per step, max length 4,096 |
| Benchmarks | `s1_benchmark`, `s2_benchmark` | `uspto: test.json` | T 0.6, top-p 0.95, top-k 20, 16,384 new tokens |

The S2 training set is the exactly correct S1 samples (8 per training reaction, T 0.9) of at most
1,024 tokens. The S3 pairs come from 16 S2 samples per reaction (T 0.7), pairing a random correct
sample with the most Tanimoto-similar incorrect one. The sampling and pair-mining scripts are not
included; their outputs are the released training sets.

All runs used 8×H100 GPUs, bfloat16, gradient checkpointing and seed 42. Training GPU-hours were
79.8 (S1), 46.3 (S2) and 20.4 (S3).

## The 27B stages

The Qwen3.8-27B stages ran on Modal (8×H200) with code specific to that setup, which is not
included. The checkpoints and every training set are released:

| Stage | Training data (released) | Settings |
|---|---|---|
| S1 SFT | the same 40,000 teacher traces | full fine-tune (text model), 3 epochs (3,750 steps), lr 2e-5, warmup 100, effective batch 32, max length 4,096 |
| S2 SFT | `v1_s2_27b` (98,731 examples: difficulty mix with 1,108 answer-only gold fallbacks; loss on the assistant completion only) | LoRA r128 / α32 / dropout 0.05, 1 epoch (3,086 steps), lr 2e-5, warmup 50, effective batch 32, max length 2,048 |
| S3 DPO | `v1_s3_pairs_27b` (12,079 pairs) | LoRA r128, 1 epoch (378 steps), lr 2e-6, β 0.1, DPO:NLL 1.0:0.2 |
| skip-S2 | `v1_skip_s2_pairs_27b` (13,698 pairs) | the S3 recipe on S1 (429 steps) |

Qwen3.8-27B is a vision-language checkpoint, used text-only. To serve it with vLLM, pass
`limit_mm_per_prompt={'image': 0, 'video': 0}`. Every 27B evaluation used the long system prompt in
`configs/paper/qwen3_8_27b/prompts.json`, with seed 42 + test row.

## Environment

| Variable | Used for |
|---|---|
| `RESONANCE_DATA_ROOT` | Root of every run directory (`resonance.paths`) |
| `HF_HOME`, `VLLM_CACHE_ROOT` | Model-weight and vLLM compile caches (see the README) |
| `KIMI_MODAL_API_BASE`, `MODAL_KEY`, `MODAL_SECRET`, `MOONSHOT_API_KEY`, `OPENROUTER_API_KEY` | Kimi K3 endpoints for trace generation (`pipelines/traces/endpoints.py`; see `.env.example`) |

## Recorded quirks

These settings are kept exactly as they ran; changing them changes the published numbers.

* **System prompts differ by step (8B).**
  - S1 SFT trained on a prep prompt without the instruction block (it is in the released S1 dataset).
  - S2 and S3 sampling used the "rules" prompt (the paper's *long* system prompt).
  - Every 8B student benchmark ran with the "none" prompt (the paper's *short* one), because the
    benchmark configs set `prompt_kwargs`, which `workflows/benchmark.py` does not read.
* **8B S2 changed world size** at checkpoint-500: from 4 to 8 GPUs, so the effective batch went
  from 32 to 64. `s2_sft.yaml` keeps `resume_from_checkpoint`; set it to `null` for a fresh run.
* **8B S2's benchmark sets no sampling seed**; S1 and S3 use seed 42.
* **27B S2 and S3 are the one-epoch variants** `s2-v5-difficulty-mix-1ep` and
  `s3-from-s2-v5-difficulty-mix-1ep`. The 27B S3 rollouts started on H200 and were resumed
  row-wise, with the same seeds, on H100.
* **Published test labels encode "no rule fired" as `"0"`**; `pipelines/labels/test_reaction_classes.py
  --encoding published` (the default) reproduces them.
* **Every Fig 2 bar is scored from raw responses** with the same lenient extractor
  (`resonance.paper.classes`), the frontier references included. Opus 5 is 3,934/5,000; an earlier
  partial Opus run gave 3,910, the number in some earlier drafts.

## Verification

| Step | Check |
|---|---|
| USPTO conversion | byte-identical to the published split |
| Teacher-trace prompts, corpus validation | fingerprints and corpus sha256 match |
| Train / test / corrected class labels | byte-identical |
| 8B and 27B training sets | row counts as in the paper (47,492; 98,731; 28,747; 12,079; 26,661; 13,698) |
| Hydra configs | equal to the runs' saved `.hydra/config.yaml` apart from wandb naming; every paper config resolves through `resonance.workflows` (CI) |
| `resonance.paper.run_all` | all 11 figure data files identical to the paper's, and Figs 2–7, A1, A2 pixel-identical, rebuilt from the released data on the Hugging Face Hub |
| Headline accuracies | every model's correct count, recomputed from its raw test outputs with the lenient scorer, equals the paper's |
| Released checkpoints (all 8) | training steps match; 8B S1/S2 weights byte-identical to the recorded checkpoints; adapter sha256 match the merge records; replaying 200 recorded test generations per model gives the same accuracy (paired McNemar p ≥ 0.34) |

Not rerun: training (GPU).
