# Qwen3-8B lineage (Kimi K3 no-ID teacher traces)

S1 SFT on teacher traces -> S2 rejection-sampling SFT on the S1 model's own correct rollouts
-> S3 DPO+NLL on self-play pairs, each benchmarked on the USPTO-50k test set (5,000 reactions).
Every location resolves through `resonance.paths.run_dir(key)`; configs spell them as
`${run_dir:<key>}` and are run through `python -m resonance.stages.qwen3_8b.workflow`.
Launchers live in `launchers/qwen3_8b/` (submit from the repository root, `mkdir -p logs`).

| Step | Launcher | Module / config | Output (under `run_dir(key)`) |
|---|---|---|---|
| S1 data + SFT | `s1_sft.sbatch` | `resonance.pipelines.traces.validate_no_id`, `s1_prep --arm 0a`, `s1_check_lengths`, `workflow sft` + `s1_sft.yaml` | `traces`: `outputs/sft_dataset_0a_no_reaction_id`, `outputs/model_0a_no_reaction_id` |
| S1 benchmark | `s1_benchmark.sbatch` | `run_gpu benchmark` -> `workflow benchmark` + `s1_benchmark.yaml` | `s1_8b`: `outputs/benchmark/results.jsonl` |
| S1 evaluate | `s1_evaluate.sbatch` | `resonance.workflows.evaluate` + `configs/evaluate.yaml` | `s1_8b`: `outputs/benchmark/plots` |
| S1 rollouts | `s1_rollouts.sbatch` | `run_gpu rollouts` -> 4 x `sample_rollouts` (N=8, T=0.9, 1024 tokens) | `s1_8b`: `outputs/rollouts/rollouts.jsonl` |
| S2 filter | `s2_filter.sbatch` | `filter_rollouts` (finished, exact-correct, <=1024 tokens) | `s2_8b`: `outputs/sft_dataset` (47,492 examples) |
| S2 SFT | `s2_train.sbatch` | `s2_train` -> `workflow sft` + `s2_sft.yaml` | `s2_8b`: `outputs/model/checkpoint-2229` |
| S2 benchmark | `s2_benchmark.sbatch` | `workflow benchmark` + `s2_benchmark.yaml` | `s2_8b_bench`: `outputs/results.jsonl` |
| S2 evaluate | `s2_evaluate.sbatch` | `s2_evaluate` | `s2_8b_bench`: `outputs/evaluation.json` |
| S3 completions | `s3_generate.sbatch` (array 0-7) | `generate_completions` (G=16, T=0.7, 8192 tokens) | `s3_8b`: `outputs/completions.shard*-of-8.jsonl` |
| S3 merge | `s3_merge_shards.sbatch` | `merge_shards` | `s3_8b`: `outputs/completions.jsonl` |
| S3 pairs | `s3_mine_pairs.sbatch` | `validate_rollouts`, `mine_dpo_pairs --seed 42` | `s3_8b`: `outputs/dpo_pairs.jsonl` (28,747 pairs) |
| S3 DPO | `s3_train.sbatch` | `workflow dpo` + `s3_dpo.yaml` (LoRA r128, 3000 steps) | `s3_8b`: `outputs/model` |
| S3 merge LoRA | `s3_merge_lora.sbatch` | `merge_lora` | `s3_8b`: `outputs/benchmark_model` |
| S3 benchmark | `s3_benchmark.sbatch` | `run_benchmark_gpu` -> `benchmark_replay` | `s3_8b`: `outputs/benchmark/results.jsonl` |
| S3 evaluate | `s3_evaluate.sbatch` | `evaluate_benchmark`, `resonance.workflows.evaluate` | `s3_8b`: `outputs/benchmark/evaluation.json` |
| S2 vs S3 | (by hand) | `compare_s2_s3` | `s3_8b`: `outputs/benchmark/s2_s3_comparison.json` |

GPU launchers take the shared locks in `RESONANCE_GPU_LOCK_DIR` and wait for free cards,
because the scheduler on the original machine did not track GPU memory.

## Recorded quirks (kept as run; changing them changes the published numbers)

* **System prompts differ by step.** S1 SFT uses the short prep prompt in `s1_prep.py`;
  S2 rollouts (`sample_rollouts.py`) and S3 completions (`generate_completions.py`, via
  `ForwardRxnPrompt(system_prompt_option="rules")`) use the "rules" prompt; every benchmark
  uses the "none" prompt. The benchmark configs set `prompt_kwargs`, which
  `workflows/benchmark.py` does not read (it reads `system_prompt_kwargs`), so "none" is what
  actually ran. S3's benchmark replays the S1 benchmark's saved prompts, so it matches.
* **S2 world size changed mid-run.** Steps 1-500 ran on 4 GPUs (2 x 4 x 4 = effective batch
  32); training was paused at checkpoint-500 and resumed on 8 GPUs (effective batch 64) to
  step 2229. `s2_train.py` is the 8-GPU version; the 4-GPU version was not preserved.
  `TRAIN_COMPLETE.json` records `effective_batch: 32`, which is wrong after step 500. On
  resume, the trainer skipped ~32k samples of epoch 1 of which ~16k had been trained.
  `s2_sft.yaml` keeps `resume_from_checkpoint`; set it to `null` for a fresh run.
* **S2's benchmark sets no sampling seed** (S1 and S3 use seed 42; S3 also batches by 64).
* S1 benchmark 31.0% (1551/5000), S2 38.92% (1946/5000), S3 42.80% (2140/5000).
