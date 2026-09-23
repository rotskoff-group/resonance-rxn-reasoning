#!/bin/bash
#SBATCH --job-name=kimi8b_synth
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/kimi8b_%x_%j.out
#SBATCH --error=logs/kimi8b_%x_%j.err
# Usage: sbatch --job-name=kimi8b_synth_s1 launchers/data/synthetic_ood_kimi8b.sbatch s1
# Prerequisite: python -m resonance.pipelines.synthetic_ood.prepare
# GPU usage is coordinated by per-GPU locks and live memory checks in the runner.
# Ported from run_dir('synthetic_ood')/bash_scripts/kimi8b_synthetic.sbatch.
set -euo pipefail
exec "${RESONANCE_PY:-python}" -m resonance.pipelines.synthetic_ood.run_kimi8b "$1"
