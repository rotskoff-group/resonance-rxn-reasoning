#!/bin/bash
#SBATCH --job-name=fwd_pairs
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pairs_%A_%a.out
#SBATCH --error=logs/pairs_%A_%a.err
# Phase 3 — enumerate guaranteed-matching reactant pairs per template and react them.
# One array task per template directory that Phase 2 actually produced; a template
# whose second slot never matched has no slot_1.parquet and is skipped with a note.
# Required --export: RUN_DIR, TEMPLATES, OUT_SUBDIR, MAX_PAIRS
set -euo pipefail
COLLECTED="${RUN_DIR}/outputs/${OUT_SUBDIR}/screen_collected"
mapfile -t DIRS < <(ls -d "${COLLECTED}"/template_* 2>/dev/null | sort)
DIR="${DIRS[$SLURM_ARRAY_TASK_ID]}"
TID=$((10#$(basename "$DIR" | sed 's/template_//')))
if [ ! -s "$DIR/slot_0.parquet" ] || [ ! -s "$DIR/slot_1.parquet" ]; then
    echo "template ${TID}: missing a slot parquet (one slot never matched) — skipping"
    exit 0
fi
"${RESONANCE_PY:-python}" -m resonance.pipelines.synthetic_data.apply_pairs \
    --template-id "${TID}" \
    --templates "${TEMPLATES}" \
    --slot0 "$DIR/slot_0.parquet" --slot1 "$DIR/slot_1.parquet" \
    --output-dir "${RUN_DIR}/outputs/${OUT_SUBDIR}/pairs/hits/" \
    --max-pairs "${MAX_PAIRS}"
