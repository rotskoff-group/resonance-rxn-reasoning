#!/bin/bash
#SBATCH --job-name=paper_figures_render
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# CPU only: independence check, metareasoning summary, exports and all six figures.
# Usage (from the repo root): sbatch launchers/paper/render.sbatch <out_dir>
set -euo pipefail
exec "${RESONANCE_PY:-python}" -m resonance.paper.run_all --out "$1"
