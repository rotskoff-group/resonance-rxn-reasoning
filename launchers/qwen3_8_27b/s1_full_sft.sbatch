#!/bin/bash
#SBATCH --job-name=q27_s1_full_sft
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
# S1: full SFT of Qwen3.8-27B on the Kimi K3 no-ID traces (Modal, 8x H200).
"$PY" -c "from resonance.pipelines.traces.validate_no_id import validate; from resonance.stages.qwen3_8_27b.config import traces_path; validate(traces_path())"
exec "$PY" -m modal run --detach src/resonance/stages/qwen3_8_27b/modal_apps/full_sft.py
