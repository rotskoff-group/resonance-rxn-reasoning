"""Names and locations shared by the 27B scripts and Modal apps.

Modal app and volume names live in your own Modal workspace: the S1 app (`full_sft`) and the
S3 app create their volumes if missing, the other apps expect them to exist. Override any
name with the RESONANCE_MODAL_* variable named next to it. (The published runs used
volumes named chem-qwen38-27b-kimi-no-id-s1, chem-qwen38-27b-no-id-s2-s3 and
chem-qwen38-27b-stage-ablations in the authors' workspace; their checkpoints are released on
the Hugging Face Hub, see resonance.release.artifacts.) `CONTAINER_ENV` is baked into every
image so the names resolve identically inside containers.

Local directories come from `resonance.paths.run_dir`:
    pipeline_27b  base / S1-eval / S2 / S3 outputs (`outputs/<RUN_ID>/...`), GPU lock files
    ablation_27b  stage-ablation outputs (`outputs/<arm>/...`)
    s1_27b        S1 training metadata (`outputs/<RUN_ID>/...`)
    synthetic_s2_27b  S1 synthetic OOD benchmark (`outputs/s1-synthetic-baseline/eval/<suite>`)
"""
import os
import sys
from pathlib import Path

from resonance.paths import run_dir

STAGE_DIR = Path(__file__).resolve().parent
SRC = STAGE_DIR.parents[2]               # directory containing the `resonance` package
PROMPTS = SRC / 'resonance/configs/paper/qwen3_8_27b/prompts.json'

BASE_MODEL = 'Qwen/Qwen3.8-27B'
BASE_REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
# 'no-id-' + sha256(teacher traces)[:16]; re-derived and checked wherever the traces are read.
RUN_ID = 'no-id-b97ef67b9ba65863'
SOURCE_SHA = 'b97ef67b9ba658631c2c95a015d2166e32180c7d81ce583a9f1aeac9d45f27e9'

_NAMES = {
    # env var                              default
    'RESONANCE_MODAL_S1_VOLUME':            'resonance-27b-kimi-no-id-s1',
    'RESONANCE_MODAL_PIPELINE_VOLUME':      'resonance-27b-no-id-s2-s3',
    'RESONANCE_MODAL_ABLATION_VOLUME':      'resonance-27b-stage-ablations',
    'RESONANCE_MODAL_SYNTHETIC_VOLUME':     'resonance-27b-synthetic-s2',
    'RESONANCE_MODAL_APP_S1':               'resonance-27b-kimi-no-id-s1',
    'RESONANCE_MODAL_APP_S2':               'resonance-27b-s2-v5-difficulty-1ep',
    'RESONANCE_MODAL_APP_EVAL_S2':          'resonance-27b-s2-v5-difficulty-eval',
    'RESONANCE_MODAL_APP_S3':               'resonance-27b-s3-from-s2-v5',
    'RESONANCE_MODAL_APP_EVAL_S3':          'resonance-27b-s3-from-s2-v5-eval',
    'RESONANCE_MODAL_APP_RESUME':           'resonance-27b-rollout-resume',
    'RESONANCE_MODAL_APP_ABLATION':         'resonance-27b-stage-ablations',
    'RESONANCE_MODAL_APP_S1_SYNTHETIC':     'resonance-27b-s1-synthetic-eval',
}
_names = {key: os.environ.get(key, default) for key, default in _NAMES.items()}
S1_VOLUME = _names['RESONANCE_MODAL_S1_VOLUME']
PIPELINE_VOLUME = _names['RESONANCE_MODAL_PIPELINE_VOLUME']
ABLATION_VOLUME = _names['RESONANCE_MODAL_ABLATION_VOLUME']
SYNTHETIC_VOLUME = _names['RESONANCE_MODAL_SYNTHETIC_VOLUME']
APP_S1 = _names['RESONANCE_MODAL_APP_S1']
APP_PIPELINE = PIPELINE_VOLUME           # the rollout app is named after its volume
APP_S2 = _names['RESONANCE_MODAL_APP_S2']
APP_EVAL = {'s2': _names['RESONANCE_MODAL_APP_EVAL_S2'], 's3': _names['RESONANCE_MODAL_APP_EVAL_S3']}
APP_S3 = _names['RESONANCE_MODAL_APP_S3']
APP_RESUME = _names['RESONANCE_MODAL_APP_RESUME']
APP_ABLATION = _names['RESONANCE_MODAL_APP_ABLATION']
APP_S1_SYNTHETIC = _names['RESONANCE_MODAL_APP_S1_SYNTHETIC']
# Container code layout: the package at /pipeline_src (on PYTHONPATH), stage scripts at CODE_DIR.
CONTAINER_ENV = {**_names, 'PYTHONPATH': '/pipeline_src'}
_IGNORE = ['**/__pycache__', '**/.ruff_cache', '**/*.pyc']


def with_code(image, code_dir='/pipeline_code'):
    """Mount the resonance package and this stage directory (as `code_dir`) into `image`."""
    return (image.add_local_dir(SRC / 'resonance', '/pipeline_src/resonance', ignore=_IGNORE)
                 .add_local_dir(STAGE_DIR, code_dir, ignore=_IGNORE))


def traces_path():
    return run_dir('traces') / 'outputs/traces_stage1a_no_reaction_id.jsonl'


def pipeline_root():
    return run_dir('pipeline_27b')


def benchmark_bin():
    """bin/ of the vLLM benchmark environment (RESONANCE_BENCH_BIN; default: this interpreter's)."""
    return Path(os.environ.get('RESONANCE_BENCH_BIN', str(Path(sys.executable).parent)))
