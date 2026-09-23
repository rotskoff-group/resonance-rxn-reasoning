#!/bin/bash
#SBATCH --job-name=q8b_s3_evaluate
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
source "${RESONANCE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/launchers/qwen3_8b/_env.sh"
OUT=$(run_dir s3_8b)/outputs/benchmark
"$PY" -m resonance.stages.qwen3_8b.evaluate_benchmark
"$PY" -m resonance.workflows.evaluate --config-path "$REPO/src/resonance/configs" --config-name evaluate \
  "results_path=$OUT/results.jsonl" "plot.output_dir=$OUT/plots" "hydra.run.dir=$OUT/evaluation_hydra"
