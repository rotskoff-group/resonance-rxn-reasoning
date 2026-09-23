#!/bin/bash
#SBATCH --job-name=q27_h100_rollouts
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=14:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Submit from the repository root (logs/ must exist). Modal credentials come from
# `modal token new` or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET; RESONANCE_DATA_ROOT locates the data.
set -euo pipefail
PY=${RESONANCE_PY:-python}
# Resume the S3 (s2_dpo) or skip_s2 rollouts on H100s with the same per-row seeds.
exec "$PY" -m modal run --detach src/resonance/stages/qwen3_8_27b/modal_apps/resume_h100_rollouts.py --target "$1"
