#!/bin/bash
#SBATCH --job-name=q27_base_merge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Submit from the repository root (logs/ must exist). Modal credentials come from
# `modal token new` or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET; RESONANCE_DATA_ROOT locates the data.
set -euo pipefail
PY=${RESONANCE_PY:-python}
# Merge the four base-evaluation shards (run after eval_base_shard.sbatch).
exec "${RESONANCE_BENCH_PY:-$PY}" -m resonance.stages.qwen3_8_27b.merge_base_eval_shards
