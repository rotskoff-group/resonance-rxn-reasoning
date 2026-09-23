#!/bin/bash
#SBATCH --job-name=fwd_uni
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/uni_%A_%a.out
#SBATCH --error=logs/uni_%A_%a.err
# One-slot branch — apply each 1-reactant template directly to every molecule.
# No pairing needed, so this path never depends on the screen.
# Required --export: RUN_DIR, TEMPLATES, INPUT_PARQUET, CHUNK_SIZE, OUT_SUBDIR
set -euo pipefail
PY="${RESONANCE_PY:-python}"
IDS=$("$PY" -c "import json;print(','.join(map(str,json.load(open('${RUN_DIR}/outputs/template_ids.json'))['one_slot'])))")
"$PY" -m resonance.pipelines.synthetic_data.apply_chunk_forward \
    --input "${INPUT_PARQUET}" \
    --templates "${TEMPLATES}" \
    --chunk-id "${SLURM_ARRAY_TASK_ID}" --chunk-size "${CHUNK_SIZE}" \
    --output-dir "${RUN_DIR}/outputs/${OUT_SUBDIR}/uni/hits/" \
    --template-ids "${IDS}"
