#!/bin/bash
#SBATCH --job-name=q27_skip_s2_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Submit from the repository root (logs/ must exist). Modal credentials come from
# `modal token new` or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET; RESONANCE_DATA_ROOT locates the data.
set -euo pipefail
PY=${RESONANCE_PY:-python}
# Download the skip_s2 model and run the matched local test benchmark (takes shared GPU locks).
exec "$PY" -m resonance.stages.qwen3_8_27b.ablation.benchmark_skip_s2_local
