#!/bin/bash
#SBATCH --job-name=q27_skip_s2_dpo_1ep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=30:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Submit from the repository root (logs/ must exist). Modal credentials come from
# `modal token new` or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET; RESONANCE_DATA_ROOT locates the data.
set -euo pipefail
PY=${RESONANCE_PY:-python}
# skip_s2 from completed rollouts: pairs, one DPO epoch, merge.
exec "$PY" -m modal run --detach src/resonance/stages/qwen3_8_27b/modal_apps/ablation_pipeline.py::train_after_rollouts
