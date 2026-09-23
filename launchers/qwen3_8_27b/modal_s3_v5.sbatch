#!/bin/bash
#SBATCH --job-name=q27_s3_from_s2_v5
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=96:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Submit from the repository root (logs/ must exist). Modal credentials come from
# `modal token new` or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET; RESONANCE_DATA_ROOT locates the data.
set -euo pipefail
PY=${RESONANCE_PY:-python}
# S3: rollouts from S2-V5 (n=16, T=0.7), DPO pairs, one epoch of DPO+NLL, merge; then evaluate.
"$PY" -m modal run --detach src/resonance/stages/qwen3_8_27b/modal_apps/s3_from_s2_v5.py
RESONANCE_EVAL_STAGE=s3 exec "$PY" -m modal run --detach src/resonance/stages/qwen3_8_27b/modal_apps/evaluate_v5.py
