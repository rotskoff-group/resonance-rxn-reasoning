# Sourced by every Qwen3-8B launcher. Submit from the repository root, e.g.
#   mkdir -p logs && sbatch launchers/qwen3_8b/s1_sft.sbatch
# (sbatch runs a copy of the script, so the repository is located from the submit dir.)
#
#   RESONANCE_REPO        repository root (default: the sbatch submit dir)
#   RESONANCE_PY          interpreter (default: <repo>/.venv/bin/python)
#   RESONANCE_DATA_ROOT   data root read by resonance.paths
#
# Credentials (wandb, HF) are not sourced here: export them in the submitting shell.
REPO=${RESONANCE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
PY=${RESONANCE_PY:-$REPO/.venv/bin/python}
export RESONANCE_PY=$PY
export PYTHONPATH=$REPO/src${PYTHONPATH:+:$PYTHONPATH}
CONFIGS=$REPO/src/resonance/configs/paper/qwen3_8b
run_dir() { "$PY" -c "from resonance.paths import run_dir; print(run_dir('$1'))"; }
