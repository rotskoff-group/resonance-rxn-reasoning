#!/bin/bash
#SBATCH --job-name=q8b_s3_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
source "${RESONANCE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/launchers/qwen3_8b/_env.sh"
RUN=$(run_dir s3_8b)
export TOKENIZERS_PARALLELISM=false
test -s "$RUN/outputs/dpo_pairs.jsonl"
test ! -d "$RUN/outputs/model"
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$REPO"
"$PY" -m torch.distributed.run --nproc_per_node=8 --master_port=$((29500 + SLURM_JOB_ID % 1000)) \
  -m resonance.stages.qwen3_8b.workflow dpo --config-path "$CONFIGS" --config-name s3_dpo \
  "hydra.run.dir=$RUN/outputs/model/hydra"
