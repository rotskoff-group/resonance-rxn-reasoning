#!/bin/bash
#SBATCH --job-name=fwd_collect
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/collect_%j.out
#SBATCH --error=logs/collect_%j.err
# Phase 2 — group per-chunk matches into per-template slot parquets.
# Required --export: RUN_DIR, OUT_SUBDIR
set -euo pipefail
"${RESONANCE_PY:-python}" -m resonance.pipelines.forward_enum.collect_screen_streaming \
    --screen-dir "${RUN_DIR}/outputs/${OUT_SUBDIR}/screen/" \
    --output-dir "${RUN_DIR}/outputs/${OUT_SUBDIR}/screen_collected/" \
    --max-per-slot 200000
