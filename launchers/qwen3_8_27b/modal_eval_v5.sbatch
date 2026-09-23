#!/bin/bash
#SBATCH --job-name=q27_eval_v5
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Submit from the repository root (logs/ must exist). Modal credentials come from
# `modal token new` or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET; RESONANCE_DATA_ROOT locates the data.
set -euo pipefail
PY=${RESONANCE_PY:-python}
# Held-out test benchmark of the merged S2-V5 (s2) or S3 (s3) model.
RESONANCE_EVAL_STAGE="$1" exec "$PY" -m modal run --detach src/resonance/stages/qwen3_8_27b/modal_apps/evaluate_v5.py
