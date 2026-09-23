"""Locations and launch environment shared by the Qwen3-8B stage scripts.

    RESONANCE_PY             interpreter for child processes (default: this interpreter)
    RESONANCE_GPU_LOCK_DIR   directory holding the shared local GPU locks
                             (default: run_dir('pipeline_27b')/outputs, where the published
                             runs kept `local-evaluation.lock` and `gpu-<i>.lock`)
"""
import os
import sys
from pathlib import Path

from resonance.paths import run_dir

REPO = Path(__file__).resolve().parents[4]
CONFIGS = Path(__file__).resolve().parents[1].parent / "configs" / "paper" / "qwen3_8b"
PY = os.environ.get("RESONANCE_PY", sys.executable)
LOCK_DIR = Path(os.environ.get("RESONANCE_GPU_LOCK_DIR", run_dir("pipeline_27b") / "outputs"))

TRAIN_JSON = run_dir("uspto") / "train.json"
TEST_JSON = run_dir("uspto") / "test.json"
S1_MODEL = run_dir("traces") / "outputs/model_0a_no_reaction_id"
S2_MODEL = run_dir("s2_8b") / "outputs/model/checkpoint-2229"
USPTO_TEST_LABELS = run_dir("uspto_classes") / "outputs/uspto_labels_test.json"


def workflow_cmd(workflow: str, config_name: str, *overrides: str) -> list[str]:
    """argv that runs `resonance.workflows.<workflow>` on a config in CONFIGS."""
    return ["-m", "resonance.stages.qwen3_8b.workflow", workflow,
            "--config-path", str(CONFIGS), "--config-name", config_name, *overrides]
