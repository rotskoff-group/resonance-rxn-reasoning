#!/bin/bash
#SBATCH --job-name=release_replay
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Replay the first ROWS recorded test generations of one checkpoint (resonance.release.replay).
#   sbatch --export=ALL,MODEL=<dir>,RECORDED=<glob>,FORMAT=benchmark,SEED=fixed,OUT=<json> launchers/release/replay.sbatch
# Optional: ROWS (200), MAX_MODEL_LEN (16384), RESONANCE_PY (interpreter with vLLM),
#   FLASHINFER_WORKSPACE_BASE (27B: a directory holding prebuilt flashinfer kernels, if the
#   vLLM environment cannot JIT-compile them).
set -euo pipefail
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false MAX_JOBS=4
# The 27B linear-attention kernels JIT-compile with ninja from the vLLM environment's bin/
# (an absolute interpreter path does not put it on PATH).
PY="${RESONANCE_PY:-python}"
export PATH="$(dirname "$(command -v "$PY")"):$PATH"
"$PY" -m resonance.release.replay --model "$MODEL" --recorded "$RECORDED" \
  --format "$FORMAT" --seed "$SEED" --rows "${ROWS:-200}" --max-model-len "${MAX_MODEL_LEN:-16384}" --out "$OUT"
