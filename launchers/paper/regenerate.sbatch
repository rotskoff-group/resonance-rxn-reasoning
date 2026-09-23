#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Metareasoning samples for Figs 4-6 (resonance.paper.regenerate). Submit from the repo root.
# Usage: sbatch --job-name=... launchers/paper/regenerate.sbatch <fresh|review> <tag> [group] [--smoke]
#   fresh  kimi_s{1,2,3}                  9 draws: new turn-1 answer + 8 fresh
#   review kimi_s{1,2,3} length|feedback  after the stage's fresh job
set -euo pipefail
PY=${RESONANCE_PY:-python}
export HF_HOME=${RESONANCE_HF_CACHE:-$("$PY" -c 'from resonance.paths import HF_CACHE; print(HF_CACHE)')}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "no GPU allocated" >&2; exit 2; }
# Other jobs on this node can use GPUs without Slurm GRES; refuse to share one.
used=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)
[[ $used -lt 2000 ]] || { echo "GPU $CUDA_VISIBLE_DEVICES already has ${used} MiB in use" >&2; exit 3; }
action=$1 tag=$2; shift 2
args=(--tag "$tag")
if [[ $action == review ]]; then args+=(--group "$1"); shift; fi
exec "$PY" -m resonance.paper.regenerate "$action" "${args[@]}" "$@"
