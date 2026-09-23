#!/bin/bash
#SBATCH --job-name=q8b_s1_sft
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
source "${RESONANCE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/launchers/qwen3_8b/_env.sh"
# S1: validate the teacher traces, build the SFT dataset, check every length, train.
RUN=$(run_dir traces)
ARM=0a_no_reaction_id
TRACES="$RUN/outputs/traces_stage1a_no_reaction_id.jsonl"
DS="$RUN/outputs/sft_dataset_$ARM"
"$PY" -m resonance.pipelines.traces.validate_no_id "$TRACES"
"$PY" -m resonance.stages.qwen3_8b.s1_prep --arm 0a --input "$TRACES" --output "$DS" --min-records 40000 --max-length 2048
# Check every example, not only the prep script's length sample.
"$PY" -m resonance.stages.qwen3_8b.s1_check_lengths "$DS"

# Reserve room for two full optimizer checkpoints, an in-flight save, and final weights.
AVAIL=$(df --output=avail -BG "$RUN" | tail -1 | tr -dc '0-9')
echo "available: ${AVAIL}G"
[ "$AVAIL" -ge 300 ] || { echo "REFUSING: <300G free for full-state checkpointing"; exit 1; }

# GPU guard: the scheduler did not track GPU memory, so a card could already hold another
# process. Full fine-tune under plain accelerate DDP with torch AdamW: every rank holds the
# whole optimizer state (~74 GiB steady state); NEED_MIB leaves a margin on top.
NEED_MIB=${NEED_MIB:-78000}
while true; do
    mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    OK=()
    for row in "${FREE[@]}"; do
        idx="${row%%,*}"; free="${row##*, }"
        [ "$free" -ge "$NEED_MIB" ] && OK+=("$idx")
    done
    [ "${#OK[@]}" -eq 8 ] && break
    echo "$(date -Is): waiting for all 8 GPUs with >= ${NEED_MIB} MiB free; found ${#OK[@]}"
    sleep 60
done
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${OK[*]:0:8}")"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader

cd "$REPO"
"$PY" -m accelerate.commands.launch --multi_gpu --num_processes=8 --mixed_precision=bf16 \
    -m resonance.stages.qwen3_8b.workflow sft \
    --config-path "$CONFIGS" --config-name s1_sft \
    "hydra.run.dir=$RUN/outputs/hydra_sft_$ARM"
rc=$?

OUT="$RUN/outputs/model_$ARM"
if [ -d "$OUT" ] && ls "$OUT"/*.safetensors >/dev/null 2>&1; then
    echo "SFT $ARM COMPLETE -> $OUT"
else
    echo "SFT $ARM FAILED (rc=$rc): no safetensors in $OUT"; exit 1
fi
