#!/bin/bash
#SBATCH --job-name=fwd_merge
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err
# Merge + dedup by (product_smiles, template_id); writes merged.jsonl.gz + stats.json.
# Required --export: HITS_DIR, OUT_PREFIX, TOTAL_MOLS
set -euo pipefail
"${RESONANCE_PY:-python}" -m resonance.pipelines.synthetic_data.merge \
    --hits-dir "${HITS_DIR}" \
    --output "${OUT_PREFIX}/merged.jsonl.gz" \
    --stats-output "${OUT_PREFIX}/stats.json" \
    --total-mols "${TOTAL_MOLS}"
# The merged .jsonl.gz is the artifact; the per-task hits are ~50 GB of intermediates
# at 110M scale, so drop them once the merge has actually produced output.
if [ -s "${OUT_PREFIX}/merged.jsonl.gz" ]; then
    echo "merge OK -> removing ${HITS_DIR}"
    rm -rf "${HITS_DIR}"
else
    echo "merge produced no output; KEEPING ${HITS_DIR}" >&2
    exit 1
fi
