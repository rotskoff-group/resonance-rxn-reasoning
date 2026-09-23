#!/bin/bash
#SBATCH --job-name=q27_download
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Submit from the repository root (logs/ must exist). Modal credentials come from
# `modal token new` or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET; RESONANCE_DATA_ROOT locates the data.
set -euo pipefail
PY=${RESONANCE_PY:-python}
# Download a finished stage checkpoint (s1|s2|s3) from its Modal volume.
exec "$PY" -m resonance.stages.qwen3_8_27b.download_checkpoint --stage "$1"
