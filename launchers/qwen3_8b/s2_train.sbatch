#!/bin/bash
#SBATCH --job-name=q8b_s2_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
source "${RESONANCE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/launchers/qwen3_8b/_env.sh"
exec "$PY" -m resonance.stages.qwen3_8b.s2_train
