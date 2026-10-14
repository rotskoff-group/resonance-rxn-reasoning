# Paper pipeline

The code that produced the paper's results: teacher traces from Kimi K3 (prompt without the
reaction ID) distilled into Qwen3-8B and Qwen3.8-27B through three stages, then evaluated,
analysed and plotted. Terms are defined in [docs/GLOSSARY.md](docs/GLOSSARY.md); every run
directory and the released copy of its contents are listed in [docs/DATA.md](docs/DATA.md).

```
USPTO-50k ──► teacher traces (Kimi K3) ──► S1 SFT ──► S2 rejection-sampling SFT ──► S3 DPO+NLL
                                              │                 │                        │
                                              └── benchmark + evaluate on the USPTO-50k test set ──┘
reaction-class labels ──► per-class analysis ──► Figs 2, 3, 7
8B S1/S2/S3 ──► reasoning probes (length prompts, feedback, resampling) ──► Figs 4–6, A1, A2
```

## Layout

| Path | Contents |
|---|---|
| `src/resonance/paths.py` | Every data location: `RESONANCE_DATA_ROOT` plus one key per run directory. Scripts call `run_dir(key)`; nothing else spells a path. |
| `src/resonance/pipelines/uspto/` | USPTO-50k → forward-reaction json |
| `src/resonance/pipelines/traces/` | Kimi K3 teacher traces: frozen prompts + fingerprint, endpoints, generation, corpus validation |
| `src/resonance/pipelines/labels/` | Reaction-class labels for train and test, and the benzyl-ether correction for both (`relabel_classes`, `relabel_train`) |
| `src/resonance/pipelines/{forward_enum,synthetic_data,synthetic_ood}/` | Synthetic out-of-distribution benchmark from an earlier draft; **not in the ICLR 2027 paper** |
| `src/resonance/stages/qwen3_8b/` | 8B S1 → S2 → S3, each benchmarked (see its `README.md`) |
| `src/resonance/stages/qwen3_8_27b/` | 27B S1 → S2 → S3 on Modal, base/S1 local evals, the skip-S2 stage ablation, synthetic S1 eval |
| `src/resonance/stages/common.py` | Checkpoint checks, durable json/jsonl, answer extraction and canonical SMILES |
| `src/resonance/paper/` | Every paper figure: `classes` (per-class counts for all models), `export` (Fig 2 input), `fig3`, `fig7`, `backtrack` + `fig_appendix` (Figs A1–A2), `regenerate` (reasoning-probe samples), `draw_independence`, and `run_all` |
| `src/resonance/release/` | Release tooling: Hub ids (`artifacts`), checkpoint replay check (`replay`, `launchers/release/replay.sbatch`) |
| `src/resonance/configs/paper/` | The configs the published runs used, with locations as `${run_dir:<key>}` |
| `launchers/` | sbatch / shell launchers per stage; submit from the repository root with `mkdir -p logs` |

The workflows (`resonance.workflows.{sft,dpo,benchmark,evaluate}`), `resonance.analysis.metareasoning`
and `resonance.plots` are the versions that produced the paper.

## Environment

| Variable | Used for |
|---|---|
| `RESONANCE_DATA_ROOT` | Root of every run directory (`resonance.paths`) |
| `RESONANCE_HF_CACHE` | Hugging Face cache for GPU launchers |
| `RESONANCE_PY`, `RESONANCE_BENCH_BIN` | Interpreter for launchers; `bin/` of the vLLM benchmark env |
| `RESONANCE_GPU_LOCK_DIR` | Shared GPU lock files (the scheduler did not track GPU memory) |
| `RESONANCE_WAIT_JOB_PREFIX` | Slurm job-name prefix the 27B local eval waits on (default `q8b_s1_sft`) |
| `RESONANCE_MODAL_*` | Modal app and volume names (`stages/qwen3_8_27b/config.py`) |
| `FLASHINFER_WORKSPACE_BASE` | 27B local inference: directory with prebuilt flashinfer kernels, if the vLLM environment cannot JIT-compile them (Qwen3.8's linear-attention layers need `ninja` and flashinfer's headers) |
| `KIMI_MODAL_API_BASE`, `MODAL_KEY`, `MODAL_SECRET` | Kimi K3 endpoint for trace generation (`pipelines/traces/endpoints.py`) |

The 27B training, merge and preflight scripts import `Qwen3_5*` classes and run inside the
Modal image (transformers 5.17), not the local venv. Qwen3.8-27B is a vision-language checkpoint
used text-only: vLLM needs `limit_mm_per_prompt={'image': 0, 'video': 0}`.

## Figures

```bash
python -m resonance.download figures              # inputs, if you do not have the run directories
python -m resonance.paper.run_all --out <dir>     # verify → draw independence → summary → class counts → Figs 2–7 → A1, A2
```

[docs/FIGURES.md](docs/FIGURES.md) lists each figure's module and inputs.

## Recorded quirks

These are kept exactly as they ran; changing them changes published numbers.

* **System prompts differ by step (8B).** S1 SFT uses the prep prompt in `stages/qwen3_8b/s1_prep.py`;
  S2 rollouts and S3 completions use the "rules" prompt (the paper's *long* system prompt);
  every 8B student benchmark ran with the "none" prompt (the paper's *short* one), because the
  benchmark configs set `prompt_kwargs`, which `workflows/benchmark.py` does not read.
* **8B S2 changed world size** at checkpoint-500 (4 → 8 GPUs, effective batch 32 → 64);
  `TRAIN_COMPLETE.json` records 32. The 4-GPU launcher was not preserved.
* **8B S2's benchmark sets no sampling seed**; S1 and S3 use seed 42.
* **27B S2 and S3 are the one-epoch variants** `s2-v5-difficulty-mix-1ep` (LoRA r128) and
  `s3-from-s2-v5-difficulty-mix-1ep` (378 DPO steps). `train_stage.py` is the version current
  after S2-V5 launched; the arguments S2-V5 recorded match it.
* **27B S3 rollouts** started on H200 and were resumed row-wise, same seeds, on H100
  (`modal_apps/resume_h100_rollouts.py`).
* **Published test labels encode "no rule fired" as `"0"`**; `pipelines/labels/test_reaction_classes.py
  --encoding published` (the default) reproduces them.
* Every Fig 2 bar, the frontier references included, is scored from raw responses with the
  same lenient extractor (`resonance.paper.classes`). Opus 5 is 3,934/5,000; an earlier
  partial Opus run gave 3,910, which is the number in some earlier drafts.

## Verification

Each ported CPU step was re-run on the original inputs and compared with the published
outputs, and each GPU step's rendered prompts and resolved config were compared with what
the runs recorded:

| Step | Check |
|---|---|
| USPTO conversion, template slots, forward subsets (all α × split) | byte-identical |
| Teacher-trace prompts, corpus validation | fingerprints and corpus sha256 match |
| Train / test / corrected class labels | byte-identical |
| Synthetic OOD sets and 8B OOD scores | byte-identical |
| 8B S1 dataset, S2 filter (47,492), S3 merge, S3 pairs (28,747), all 8B evaluations | identical |
| 27B S2 datasets (V4, V5 = 98,731), base-eval merge | identical |
| 27B S3 / skip-S2 pair selection | identical to the original `select` on 3,000 real rollouts (inputs only on Modal) |
| Prompts for 8B rollouts, S3 completions, benchmarks; 27B S1/S2/S3 generation | first 200 rows identical to saved outputs |
| Hydra configs | equal to saved `.hydra/config.yaml` apart from wandb naming |
| `resonance.paper.run_all` (all figures): `class11_counts.json`, `Fig_{2,4,5,6}.json`, `summary.json`, `draw_independence.json`, backtracking reports, `Fig_A_backtrack.json` | identical to the paper's figure data (ignoring provenance paths) |
| Rendered Figs 2–7, A1, A2 | PNGs pixel-identical (PDFs differ only in timestamps) |
| Headline accuracies | every model's correct count recomputed from its raw test outputs with the lenient scorer equals the paper's |
| Released checkpoints (all 8) | training steps match; 8B S1/S2 final weights byte-identical to the recorded checkpoints; adapter sha256 match the merge records; replaying 200 recorded test generations per model with the recorded settings gives the same accuracy (paired McNemar p ≥ 0.34; 27B answers agree on 92–95% of rows) |
| `resonance.predict` prompts | identical to the recorded 8B and 27B evaluation prompts on all 5,000 test reactions |
| Released datasets | row counts as in the paper (47,492; 98,731; 28,747; 12,079; 26,661; 13,698); teacher-trace sha256 matches |

Not run: training and Modal apps (GPU). The Modal apps' in-container import layout is
untested.
