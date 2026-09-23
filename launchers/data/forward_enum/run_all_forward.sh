#!/bin/bash
# Full forward-enumeration pipeline over all reachable templates.
# Ported from run_dir('forward_enum')/bash_scripts/run_all_forward.sh.
#
# WHY forward: retro-enumerated records are only ~19% chemically derivable, because
# reversing a forward SMARTS drops leaving groups. Building the product BY APPLYING the
# forward template makes the round-trip true by construction.
#
# Two independent branches, both ending in merge:
#   two-slot (92 templates): screen -> collect -> pairs -> merge
#   one-slot (14 templates): apply -> merge          (no pairing needed)
# 4 templates (6, 14, 25, 26) are unreachable: 3+ slots or unparseable slot SMARTS.
#
# Prerequisite: python -m resonance.pipelines.forward_enum.template_slots
# Then:        python -m resonance.pipelines.forward_enum.build_forward_subsets
#
# Usage:  bash run_all_forward.sh pilot     # 2M molecules, small caps, validates wiring
#         bash run_all_forward.sh full      # 110M molecules, production caps
set -euo pipefail

MODE="${1:-pilot}"
PY="${RESONANCE_PY:-python}"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${RUN_DIR:-$("$PY" -c 'from resonance.paths import run_dir; print(run_dir("forward_enum"))')}"
POOL_DIR="$("$PY" -c 'from resonance.paths import run_dir; print(run_dir("pubchem"))')"
TEMPLATES="$("$PY" -c 'from resonance.pipelines.forward_enum.audit_ground_truth import TEMPLATES; print(TEMPLATES)')"

case "$MODE" in
  pilot)
    INPUT_PARQUET="${POOL_DIR}/pubchem_random_sample_2M.parquet"
    N_MOLS=2000000; MAX_TASKS=200; MAX_PAIRS=2000; OUT_SUBDIR="pilot_2M" ;;
  full)
    INPUT_PARQUET="${POOL_DIR}/cid_iupac_smiles_clean.parquet"
    N_MOLS=110145061; MAX_TASKS=800; MAX_PAIRS=100000; OUT_SUBDIR="full_110M" ;;
  *) echo "usage: $0 pilot|full" >&2; exit 1 ;;
esac

CHUNK_SIZE=$(( (N_MOLS + MAX_TASKS - 1) / MAX_TASKS ))
N_TASKS=$(( (N_MOLS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
ARRAY_END=$(( N_TASKS - 1 ))
N_TWO=$("$PY" -c "import json;print(len(json.load(open('${RUN_DIR}/outputs/template_ids.json'))['two_slot']))")

mkdir -p logs "${RUN_DIR}/outputs/${OUT_SUBDIR}"/{screen,screen_collected,pairs/hits,uni/hits}

echo "=== forward enumeration [${MODE}] ==="
echo "  pool        : ${INPUT_PARQUET}"
echo "  molecules   : ${N_MOLS}  chunk ${CHUNK_SIZE} x ${N_TASKS} tasks"
echo "  two-slot    : ${N_TWO} templates, max ${MAX_PAIRS} pairs each"
echo "  output      : ${RUN_DIR}/outputs/${OUT_SUBDIR}"
echo ""

EXPORTS="RUN_DIR=${RUN_DIR},TEMPLATES=${TEMPLATES},INPUT_PARQUET=${INPUT_PARQUET},CHUNK_SIZE=${CHUNK_SIZE},OUT_SUBDIR=${OUT_SUBDIR},MAX_PAIRS=${MAX_PAIRS}"

SCREEN_JID=$(sbatch --parsable --array=0-${ARRAY_END} --export=ALL,${EXPORTS} "${SCRIPTS}/screen_apply.sbatch")
echo "  phase 1 screen  : ${SCREEN_JID}"

COLLECT_JID=$(sbatch --parsable --dependency=afterok:${SCREEN_JID} --export=ALL,${EXPORTS} "${SCRIPTS}/screen_collect.sbatch")
echo "  phase 2 collect : ${COLLECT_JID}"

PAIRS_JID=$(sbatch --parsable --array=0-$((N_TWO - 1)) --dependency=afterok:${COLLECT_JID} --export=ALL,${EXPORTS} "${SCRIPTS}/screen_pairs.sbatch")
echo "  phase 3 pairs   : ${PAIRS_JID}"

PMERGE_JID=$(sbatch --parsable --dependency=afterany:${PAIRS_JID} \
    --export=ALL,HITS_DIR=${RUN_DIR}/outputs/${OUT_SUBDIR}/pairs/hits/,OUT_PREFIX=${RUN_DIR}/outputs/${OUT_SUBDIR}/pairs,TOTAL_MOLS=${N_MOLS} \
    "${SCRIPTS}/merge_hits.sbatch")
echo "  phase 4 merge   : ${PMERGE_JID}"

# one-slot branch runs in parallel with the whole two-slot chain
UNI_JID=$(sbatch --parsable --array=0-${ARRAY_END} --export=ALL,${EXPORTS} "${SCRIPTS}/uni_apply.sbatch")
UMERGE_JID=$(sbatch --parsable --dependency=afterany:${UNI_JID} \
    --export=ALL,HITS_DIR=${RUN_DIR}/outputs/${OUT_SUBDIR}/uni/hits/,OUT_PREFIX=${RUN_DIR}/outputs/${OUT_SUBDIR}/uni,TOTAL_MOLS=${N_MOLS} \
    "${SCRIPTS}/merge_hits.sbatch")
echo "  one-slot apply  : ${UNI_JID}  merge: ${UMERGE_JID}"
echo ""
echo "final outputs:"
echo "  ${RUN_DIR}/outputs/${OUT_SUBDIR}/pairs/merged.jsonl.gz   (two-slot)"
echo "  ${RUN_DIR}/outputs/${OUT_SUBDIR}/uni/merged.jsonl.gz     (one-slot)"
