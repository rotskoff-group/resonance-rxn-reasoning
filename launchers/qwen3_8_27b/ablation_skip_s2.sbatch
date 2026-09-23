#!/bin/bash
#SBATCH --job-name=q27_skip_s2_modal
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
# Fig 7 ablation skip_s2 (S1 -> S3). Pass --resume-failed to retry failed rollout shards.
exec "$PY" -m modal run --detach src/resonance/stages/qwen3_8_27b/modal_apps/ablation_pipeline.py --arm skip_s2 --s2-policy difficulty_mix "$@"
