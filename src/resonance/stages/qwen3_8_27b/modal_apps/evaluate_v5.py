"""Benchmark a merged 27B S2-V5 or S3 model and report held-out test accuracy by class.

Ported from run_dir('pipeline_27b')/modal/evaluate_s2_difficulty.py and
evaluate_s3_v5.py, which differed only in the variant, app name and messages; the stage is
chosen by RESONANCE_EVAL_STAGE (baked into the image so containers resolve it identically).
Launch: RESONANCE_EVAL_STAGE={s2,s3} python -m modal run --detach \
        src/resonance/stages/qwen3_8_27b/modal_apps/evaluate_v5.py
"""
import json
import os
import pathlib

import modal

from resonance.paths import run_dir
from resonance.stages.qwen3_8_27b.config import (
    APP_EVAL,
    CONTAINER_ENV,
    PIPELINE_VOLUME,
    PROMPTS,
    RUN_ID,
    pipeline_root,
    with_code,
)

ROOT = pipeline_root()
STAGE = os.environ["RESONANCE_EVAL_STAGE"]
# stage -> (variant, missing-model message, comparison key)
VARIANTS = {
    "s2": ("s2-v5-difficulty-mix-1ep", "Merged difficulty-balanced S2 model is not complete", "s2_v5"),
    "s3": ("s3-from-s2-v5-difficulty-mix-1ep", "Merged S3 model is not complete", "s3_from_s2_v5"),
}
VARIANT, MISSING_MESSAGE, COMPARISON_KEY = VARIANTS[STAGE]
VOLUME = PIPELINE_VOLUME
SHARDS = 4

app = modal.App(APP_EVAL[STAGE])
volume = modal.Volume.from_name(VOLUME, create_if_missing=False)

image = (modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("build-essential")
    .uv_pip_install("vllm==0.18.1", "transformers==4.57.6", "rdkit==2025.9.6")
    .env({"CUDA_HOME": "/usr/local/cuda", "TOKENIZERS_PARALLELISM": "false",
          "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    .env({**CONTAINER_ENV, "RESONANCE_EVAL_STAGE": STAGE}))
image = with_code(image)


def paths():
    root = pathlib.Path("/pipeline/runs") / RUN_ID / VARIANT
    return root, root / "model"


@app.function(image=image, volumes={"/pipeline": volume}, cpu=2, memory=8192, timeout=900)
def initialize(test_raw: bytes, prompts_raw: bytes, labels_raw: bytes):
    import hashlib

    from resonance.stages.common import checkpoint, write_json

    volume.reload()
    root, model = paths()
    assert (root / "COMPLETE.json").is_file(), MISSING_MESSAGE
    checkpoint(model)
    evaluation = root / "eval"
    evaluation.mkdir(parents=True, exist_ok=True)
    identity = {
        "run_id": RUN_ID,
        "variant": VARIANT,
        "model": str(model),
        "test_sha256": hashlib.sha256(test_raw).hexdigest(),
        "prompts_sha256": hashlib.sha256(prompts_raw).hexdigest(),
        "labels_sha256": hashlib.sha256(labels_raw).hexdigest(),
        "rows": len(json.loads(test_raw)),
        "shards": SHARDS,
    }
    if (evaluation / "input.json").exists():
        assert json.loads((evaluation / "input.json").read_text()) == identity
    (evaluation / "test.json").write_bytes(test_raw)
    (evaluation / "prompts.json").write_bytes(prompts_raw)
    (evaluation / "labels.json").write_bytes(labels_raw)
    write_json(evaluation / "input.json", identity)
    volume.commit()
    return identity


@app.function(image=image, volumes={"/pipeline": volume}, gpu="H200:2", cpu=16,
              memory=192 * 1024, timeout=4 * 3600, max_containers=SHARDS, retries=0)
def benchmark_shard(shard: int):
    import os
    import subprocess

    from resonance.stages.common import checkpoint

    volume.reload()
    root, model = paths()
    checkpoint(model)
    evaluation = root / "eval"
    out = evaluation / "shards" / str(shard)
    if not (out / "COMPLETE.json").exists():
        subprocess.run([
            "python", "/pipeline_code/generate.py",
            "--model", str(model), "--data", str(evaluation / "test.json"),
            "--prompts", str(evaluation / "prompts.json"), "--out", str(out),
            "--stage", "eval", "--shard", str(shard), "--shards", str(SHARDS),
            "--tp", "2",
        ], env={**os.environ, "PIPELINE_VOLUME": VOLUME}, check=True)
    volume.commit()
    return json.loads((out / "COMPLETE.json").read_text())


@app.function(image=image, volumes={"/pipeline": volume}, cpu=4, memory=16384,
              timeout=1800, retries=0)
def evaluate():
    from collections import defaultdict

    from resonance.stages.common import iter_jsonl, write_json

    volume.reload()
    root, _ = paths()
    evaluation = root / "eval"
    identity = json.loads((evaluation / "input.json").read_text())
    labels = json.loads((evaluation / "labels.json").read_text())["labels"]
    stats = defaultdict(lambda: {"rows": 0, "correct": 0, "valid": 0, "tokens": 0})
    seen = set()
    for shard in range(SHARDS):
        assert (evaluation / f"shards/{shard}/COMPLETE.json").is_file()
        for row in iter_jsonl(evaluation / f"shards/{shard}/rollouts.jsonl"):
            idx = row["row_idx"]
            assert idx not in seen
            seen.add(idx)
            cls = labels[str(idx)]["level1"]
            completion = row["completions"][0]
            stats[cls]["rows"] += 1
            stats[cls]["correct"] += int(completion["correct"])
            stats[cls]["valid"] += int(completion["prediction"] is not None)
            stats[cls]["tokens"] += len(completion["token_ids"])
    assert seen == set(range(identity["rows"]))
    total = sum(row["rows"] for row in stats.values())
    correct = sum(row["correct"] for row in stats.values())
    valid = sum(row["valid"] for row in stats.values())
    tokens = sum(row["tokens"] for row in stats.values())
    classes = []
    for cls, values in sorted(stats.items()):
        classes.append({
            "class": cls, **values,
            "accuracy": values["correct"] / values["rows"],
            "valid_fraction": values["valid"] / values["rows"],
            "mean_output_tokens": values["tokens"] / values["rows"],
        })
    report = {
        **identity,
        "accuracy": correct / total,
        "valid_smiles_fraction": valid / total,
        "mean_output_tokens": tokens / total,
        "classes": classes,
    }
    write_json(evaluation / "report.json", report)
    write_json(evaluation / "COMPLETE.json", report)
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    identity = initialize.remote(
        (run_dir("uspto") / "test.json").read_bytes(),
        PROMPTS.read_bytes(),
        (run_dir("uspto_classes") / "outputs/uspto_labels_test.json").read_bytes())
    results = list(benchmark_shard.map(range(SHARDS), return_exceptions=True))
    failures = [str(r) for r in results if isinstance(r, Exception)]
    assert not failures, f"Shard failures: {failures}"
    assert len(results) == SHARDS
    report = evaluate.remote()
    out = ROOT / "outputs" / RUN_ID / VARIANT / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "input.json").write_text(json.dumps(identity, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    s1 = json.loads((ROOT / "outputs" / RUN_ID / "s1/eval/COMPLETE.json").read_text())
    assert s1["identity"]["data_sha256"] == report["test_sha256"]
    assert s1["identity"]["prompts_sha256"] == report["prompts_sha256"]
    comparison = {
        "s1": {"accuracy": s1["accuracy"],
               "valid_smiles_fraction": s1["valid_smiles_fraction"],
               "mean_output_tokens": s1["mean_output_tokens"]},
        COMPARISON_KEY: {"accuracy": report["accuracy"],
                  "valid_smiles_fraction": report["valid_smiles_fraction"],
                  "mean_output_tokens": report["mean_output_tokens"]},
        "accuracy_delta": report["accuracy"] - s1["accuracy"],
    }
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps({"report": report, "comparison": comparison}, indent=2), flush=True)
