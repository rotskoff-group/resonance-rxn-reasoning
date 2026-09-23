#!/bin/bash
#SBATCH --job-name=q8b_s3_mine_pairs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
source "${RESONANCE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/launchers/qwen3_8b/_env.sh"
RUN=$(run_dir s3_8b)
export TOKENIZERS_PARALLELISM=false
"$PY" -m resonance.stages.qwen3_8b.validate_rollouts
test ! -e "$RUN/outputs/dpo_pairs.jsonl"
"$PY" -m resonance.stages.qwen3_8b.mine_dpo_pairs --input "$RUN/outputs/completions.jsonl" --output "$RUN/outputs/dpo_pairs.jsonl" --seed 42
test -s "$RUN/outputs/dpo_pairs.jsonl"
