#!/bin/bash
#SBATCH --job-name=q27_eval_s1_verified
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:2
# Submit from the repository root (logs/ must exist). Modal credentials come from
# `modal token new` or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET; RESONANCE_DATA_ROOT locates the data.
set -euo pipefail
PY=${RESONANCE_PY:-python}
# Local S1 test evaluation after a 32-reaction build check.
exec "$PY" -m resonance.stages.qwen3_8_27b.evaluate_local --stage s1 --gpu-pair "${GPU_PAIR:-6,7}" --verify-first
