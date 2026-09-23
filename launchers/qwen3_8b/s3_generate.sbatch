#!/bin/bash
#SBATCH --job-name=q8b_s3_generate
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --array=0-7%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
source "${RESONANCE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/launchers/qwen3_8b/_env.sh"
# Eight single-GPU shards of G=16 completions over the 40k train prompts.
RUN=$(run_dir s3_8b)
export TOKENIZERS_PARALLELISM=false
"$PY" -m resonance.stages.qwen3_8b.generate_completions \
  --model "$(run_dir s2_8b)/outputs/model/checkpoint-2229" \
  --train_json "$(run_dir uspto)/train.json" \
  --out "$RUN/outputs/completions.shard${SLURM_ARRAY_TASK_ID}-of-8.jsonl" \
  --g 16 --temperature 0.7 --max_tokens 8192 \
  --shard "$SLURM_ARRAY_TASK_ID" --nshards 8
