#!/bin/bash
#SBATCH --job-name=q8b_s2_evaluate
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
source "${RESONANCE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/launchers/qwen3_8b/_env.sh"
# Run by hand in the published pipeline; wrapped here for convenience.
exec "$PY" -m resonance.stages.qwen3_8b.s2_evaluate
