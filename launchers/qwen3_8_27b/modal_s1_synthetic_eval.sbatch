#!/bin/bash
#SBATCH --job-name=q27_s1_synth_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Submit from the repository root (logs/ must exist). Modal credentials come from
# `modal token new` or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET; RESONANCE_DATA_ROOT locates the data.
set -euo pipefail
PY=${RESONANCE_PY:-python}
# Synthetic OOD evaluation of the 27B S1 checkpoint on the three forward-enumeration test suites.
exec "$PY" -m modal run --detach src/resonance/stages/qwen3_8_27b/modal_apps/benchmark_s1_synthetic.py
