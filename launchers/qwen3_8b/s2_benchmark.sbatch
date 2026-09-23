#!/bin/bash
#SBATCH --job-name=q8b_s2_benchmark
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
source "${RESONANCE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/launchers/qwen3_8b/_env.sh"
RUN=$(run_dir s2_8b_bench)
cd "$REPO"
mkdir -p "$RUN/logs" "$RUN/outputs"
exec "$PY" -m resonance.stages.qwen3_8b.workflow benchmark \
  --config-path "$CONFIGS" --config-name s2_benchmark \
  "hydra.run.dir=$RUN/outputs/hydra"
