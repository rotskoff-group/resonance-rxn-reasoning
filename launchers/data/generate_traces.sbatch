#!/bin/bash
#SBATCH --job-name=traces_kimi
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/traces_%j.out
#SBATCH --error=logs/traces_%j.err
# Kimi K3 teacher traces over USPTO-50k train (no reaction ID).
# Ported from run_dir('traces')/bash_scripts/01_generate_1a.sbatch.
#
# NO --gres: this is a pure API job, network-bound not compute-bound.
# Required env (route=modal, the default): MODAL_KEY, MODAL_SECRET, KIMI_MODAL_API_BASE.
# route=native needs MOONSHOT_API_KEY or KIMI_API_KEY; route=openrouter needs OPENROUTER_API_KEY.
set -uo pipefail
PY=${RESONANCE_PY:-python}

# Check the frozen prompts and the intentional reaction-ID removal before generation.
"$PY" -m resonance.pipelines.traces.test_prompt_identity || { echo "PROMPT IDENTITY FAILED"; exit 1; }

OUT="${OUT:-$("$PY" -c 'from resonance.pipelines.traces.generate_traces_kimi import OUT_FILE; print(OUT_FILE)')}"
TARGET="${TARGET:-40000}"

# Gate on the artifact GROWING, not on the artifact existing. On a resume run the file is
# already non-empty, so `[ -s "$OUT" ]` is true before a single new record is written -- a
# dead job would then report success with its full prior record count.
BEFORE=0; [ -f "$OUT" ] && BEFORE=$(wc -l < "$OUT")
echo "records before: $BEFORE  (target $TARGET)"

"$PY" -m resonance.pipelines.traces.generate_traces_kimi \
    --route "${ROUTE:-modal}" \
    --concurrency "${CONCURRENCY:-16}" \
    --reasoning-effort "${EFFORT:-low}" \
    --output "$OUT" \
    ${LIMIT:+--limit $LIMIT}
rc=$?

AFTER=0; [ -f "$OUT" ] && AFTER=$(wc -l < "$OUT")
ADDED=$((AFTER - BEFORE))
echo "records after : $AFTER  (added $ADDED, rc=$rc)"

if [ "$ADDED" -le 0 ] && [ "$BEFORE" -lt "$TARGET" ]; then
    echo "trace generation FAILED: added 0 records with $((TARGET - BEFORE)) still to go"
    exit 1
fi

if [ "$AFTER" -lt "$TARGET" ]; then
    # Partial, not failed: the generator is resumable, so resubmitting picks up the rest.
    echo "trace generation PARTIAL: $AFTER/$TARGET -- resubmit to continue"
    exit 2
fi
"$PY" -m resonance.pipelines.traces.validate_no_id "$OUT"
echo "trace generation COMPLETE: $AFTER/$TARGET"
