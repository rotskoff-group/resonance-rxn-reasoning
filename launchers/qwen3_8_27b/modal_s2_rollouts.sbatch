#!/bin/bash
#SBATCH --job-name=q27_s2_rollouts
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Submit from the repository root (logs/ must exist). Modal credentials come from
# `modal token new` or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET; RESONANCE_DATA_ROOT locates the data.
set -euo pipefail
PY=${RESONANCE_PY:-python}
# S1 rollouts on the train set (n=8, T=0.9). Only the rollouts are used downstream;
# stop the app after they finish (its own S2 recipe is superseded by the V5 difficulty mix).
exec "$PY" -m modal run --detach src/resonance/stages/qwen3_8_27b/modal_apps/pipeline.py --stage s2
