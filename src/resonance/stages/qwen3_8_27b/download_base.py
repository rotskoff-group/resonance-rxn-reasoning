"""Download the pinned pre-SFT checkpoint for a matched base evaluation.

Ported from run_dir('pipeline_27b')/scripts/download_base.py.
CLI: python -m resonance.stages.qwen3_8_27b.download_base
"""
import json

from huggingface_hub import snapshot_download

from resonance.stages.common import checkpoint, write_json
from resonance.stages.qwen3_8_27b.config import BASE_MODEL, BASE_REVISION, pipeline_root, traces_path

ROOT = pipeline_root()
MODEL = BASE_MODEL
REVISION = BASE_REVISION


def main():
    from resonance.pipelines.traces.validate_no_id import validate
    identity = validate(traces_path())
    run_id = 'no-id-' + identity['source_sha256'][:16]
    root = ROOT/'outputs'/run_id/'base'
    expected = {'model_id': MODEL, 'revision': REVISION,
                'source_sha256': identity['source_sha256'], 'run_id': run_id}
    root.mkdir(parents=True, exist_ok=True)
    manifest = root/'DOWNLOAD_COMPLETE.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == expected
        checkpoint(root/'model')
        print('Pinned base checkpoint already downloaded', flush=True)
        return
    print(f'Downloading {MODEL}@{REVISION}', flush=True)
    snapshot_download(MODEL, revision=REVISION, local_dir=root/'model',
                      allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model'],
                      max_workers=4)
    checkpoint(root/'model')
    write_json(manifest, expected)
    print(json.dumps(expected), flush=True)


if __name__ == '__main__':
    main()
