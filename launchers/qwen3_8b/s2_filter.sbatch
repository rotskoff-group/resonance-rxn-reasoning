#!/bin/bash
#SBATCH --job-name=q8b_s2_filter
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
source "${RESONANCE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/launchers/qwen3_8b/_env.sh"
export TOKENIZERS_PARALLELISM=false
exec "$PY" -m resonance.stages.qwen3_8b.filter_rollouts
