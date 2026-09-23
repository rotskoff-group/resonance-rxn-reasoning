#!/bin/bash
#SBATCH --job-name=q27_s2_v5_1ep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=52:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Submit from the repository root (logs/ must exist). Modal credentials come from
# `modal token new` or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET; RESONANCE_DATA_ROOT locates the data.
set -euo pipefail
PY=${RESONANCE_PY:-python}
# S2: one epoch LoRA RS-SFT from S1 on S2_V5_difficulty_mix_dataset (upload it first).
exec "$PY" -m modal run --detach src/resonance/stages/qwen3_8_27b/modal_apps/s2_difficulty_one_epoch.py
