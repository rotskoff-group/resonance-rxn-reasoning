# Qwen3.8-27B lineage (Kimi K3 no-ID teacher traces)

The same teacher traces as the 8B lineage, on `Qwen/Qwen3.8-27B` (revision `1d4bf0f2…`):
- S1: full SFT.
- S2: LoRA r128 on a difficulty-mixed rejection-sampling set, 1 epoch.
- S3: LoRA r128 DPO + NLL, 1 epoch.
- skip-S2: the S3 recipe applied to S1.

Training and rollouts run on [Modal](https://modal.com) (8×H200 for training, 2×H200 per
rollout shard). The base and S1 test evaluations, the skip-S2 evaluation and the S2 data build
run locally (2 GPUs, vLLM).

Every location resolves through `config.py`:
- Local: `run_dir('pipeline_27b')/outputs/no-id-b97ef67b9ba65863/<stage>/`.
- Modal: the volumes named by the `RESONANCE_MODAL_*` variables.

`no-id-b97ef67b9ba65863` is derived from the teacher traces' sha256 and re-checked wherever
they are read.

## Before you start

- `pip install modal`, then authenticate with `modal token new` (or set `MODAL_TOKEN_ID` /
  `MODAL_TOKEN_SECRET`).
- Export `RESONANCE_DATA_ROOT`. The teacher traces must be at
  `run_dir('traces')/outputs/traces_stage1a_no_reaction_id.jsonl`; both
  `python -m resonance.download figures` and `download all` place them there.
- For local evaluation, `RESONANCE_BENCH_BIN` must point at the `bin/` of a vLLM environment
  that can serve Qwen3.8 (this repo's `.venv` from `uv sync --frozen` works). Qwen3.8-27B is a
  vision-language checkpoint used text-only, so vLLM needs `limit_mm_per_prompt={'image': 0, 'video': 0}`.
- The training scripts import `Qwen3_5*` classes and run inside the Modal images defined in
  `modal_apps/` (transformers 5.17), not in the local environment.
- Submit launchers from the repository root after `mkdir -p logs`. They are Slurm scripts, but
  the Modal ones only launch a detached Modal app and can also be run with plain `bash`.

## Steps

| Step | Launcher (`launchers/qwen3_8_27b/`) | Module | Output |
|---|---|---|---|
| S1 SFT (3 epochs, 3,750 steps) | `s1_full_sft.sbatch` | `modal_apps/full_sft.py` | S1 volume `runs/<RUN_ID>/model` |
| S1 checkpoint to local | `download.sbatch s1` | `download_checkpoint --stage s1` | `.../s1/model` |
| Base model, base evaluation | `download_base.sbatch`, `eval_base_shard.sbatch` (array 0–3), `merge_base_eval.sbatch` | `download_base`, `evaluate_base_shard`, `merge_base_eval_shards` | `.../base/eval/rollouts.jsonl` (26.3%) |
| S1 evaluation | `eval_s1_verified.sbatch` | `evaluate_local --stage s1 --verify-first` | `.../s1/eval/rollouts.jsonl` (79.0%) |
| S2 rollouts (8 per training reaction, T 0.9) | `modal_s2_rollouts.sbatch` | `modal_apps/pipeline.py --stage s2` | pipeline volume `s2/rollouts/{0,1}` |
| S2 data | (local, CPU) | `volume_transfer get-s2-rollouts`, `build_s2_full` (V4: every correct sample), `build_s2_difficulty_mix` (V5), `volume_transfer put-s2-v5` | `.../S2_V5_difficulty_mix_dataset` (98,731 examples) |
| S2 training (LoRA, 1 epoch, 3,086 steps), merge, evaluation | `modal_s2_difficulty_one_epoch.sbatch`, `modal_eval_v5.sbatch s2` | `modal_apps/s2_difficulty_one_epoch.py`, `modal_apps/evaluate_v5.py` | volume `s2-v5-difficulty-mix-1ep/{adapter,model,eval}` (79.3%) |
| S3 rollouts (16 per reaction, T 0.7) | `modal_s3_v5.sbatch`; the paper's run was resumed with `resume_h100_rollouts.sbatch s2_dpo` | `modal_apps/s3_from_s2_v5.py`, `modal_apps/resume_h100_rollouts.py` | volume `s3-from-s2-v5-difficulty-mix-1ep/rollouts` |
| S3 pairs, DPO (1 epoch, 378 steps, 12,079 pairs), merge, evaluation | `train_s3_one_epoch.sbatch`, `modal_eval_v5.sbatch s3` | `s3_from_s2_v5.py::train_after_rollouts`, `evaluate_v5.py` | volume `s3-from-s2-v5-difficulty-mix-1ep/{adapter,model,eval}` (79.6%) |
| Evaluations to local | (local) | `volume_transfer get-s2-eval`, `get-s3-eval` | `.../{s2-v5-…,s3-from-…}/eval/shards/*/rollouts.jsonl` |
| skip-S2: rollouts from S1, DPO, merge | `ablation_skip_s2.sbatch`, `resume_h100_rollouts.sbatch skip_s2`, `ablation_train_s3_one_epoch.sbatch` | `modal_apps/ablation_pipeline.py` | ablation volume `runs/<RUN_ID>/skip_s2/s3` (13,698 pairs, 429 steps) |
| skip-S2 evaluation | `ablation_benchmark_skip_s2_local.sbatch` | `ablation/benchmark_skip_s2_local` | `run_dir('ablation_27b')/outputs/skip_s2/local_eval` (78.8%) |

Released checkpoints: `Resonance-27B-{S1,S2,S3,skip-S2}` on the Hugging Face Hub
(`resonance.release.artifacts`), each merged, with the LoRA adapter alongside for S2, S3 and
skip-S2.

## Recorded quirks (kept as run)

- **S2 rebalancing.** S2 is the V5 difficulty mix rather than plain rejection sampling, because
  S1 already solves 85.8% of its training samples. Earlier variants (V2, V4, a 3-epoch LoRA)
  were run, and one was evaluated on the test set before V5 was fixed (paper appendix, "Why S2
  was rebalanced for the 27B model").
- **S3 rollout hardware.** The S3 rollouts started on H200 and were resumed row-wise, with the
  same per-row seeds, on H100.
- **Scorer.** The pipeline stores a strict correctness flag (text after `</think>` only). Every
  number in the paper uses the lenient scorer (`resonance.evaluation.extractors`); only the
  base model's score differs (22.8% strict, 26.3% lenient).
- **Prompts.** Every 27B evaluation uses the long system prompt
  (`configs/paper/qwen3_8_27b/prompts.json`), with seed 42 + test row.
