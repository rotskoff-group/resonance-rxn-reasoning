#!/bin/bash
#SBATCH --job-name=fwd_screen
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/screen_%A_%a.out
#SBATCH --error=logs/screen_%A_%a.err
# Phase 1 — substructure screen for all 92 two-slot templates.
# Required --export: RUN_DIR, TEMPLATES, INPUT_PARQUET, CHUNK_SIZE, OUT_SUBDIR
set -euo pipefail
"${RESONANCE_PY:-python}" -m resonance.pipelines.forward_enum.screen_templates \
    --input "${INPUT_PARQUET}" \
    --templates "${TEMPLATES}" \
    --template-ids-json "${RUN_DIR}/outputs/template_ids.json" \
    --chunk-id "${SLURM_ARRAY_TASK_ID}" --chunk-size "${CHUNK_SIZE}" \
    --output-dir "${RUN_DIR}/outputs/${OUT_SUBDIR}/screen/"
