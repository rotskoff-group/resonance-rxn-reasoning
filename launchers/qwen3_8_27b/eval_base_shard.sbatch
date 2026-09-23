#!/bin/bash
#SBATCH --job-name=q27_base_shard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:2
#SBATCH --array=0-3
# Submit from the repository root (logs/ must exist). Modal credentials come from
# `modal token new` or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET; RESONANCE_DATA_ROOT locates the data.
set -euo pipefail
PY=${RESONANCE_PY:-python}
# Base evaluation, one quarter of the test set per array task (RESONANCE_BENCH_BIN: vLLM env bin/).
exec "$PY" -m resonance.stages.qwen3_8_27b.evaluate_base_shard --shard "$SLURM_ARRAY_TASK_ID"
