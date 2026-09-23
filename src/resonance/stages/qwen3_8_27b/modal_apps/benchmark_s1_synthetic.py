"""Benchmark the pre-synthetic 27B S1 checkpoint on the current forward synthetic test splits.

Ported from run_dir('synthetic_s2_27b')/modal/benchmark_s1_synthetic.py (the 27B S1 row of the
synthetic OOD evaluation). Local reports land in run_dir('synthetic_s2_27b')/outputs/
s1-synthetic-baseline/eval/<suite>, which resonance.pipelines.synthetic_ood.prepare hash-checks.
Launch: python -m modal run --detach src/resonance/stages/qwen3_8_27b/modal_apps/benchmark_s1_synthetic.py [--check-only]
"""
import json
import pathlib

import modal


from resonance.paths import run_dir
from resonance.stages.qwen3_8_27b.config import (APP_S1_SYNTHETIC, CONTAINER_ENV, PROMPTS, RUN_ID as S1_RUN_ID,
                                                 S1_VOLUME, SYNTHETIC_VOLUME, with_code)

ROOT = run_dir("synthetic_s2_27b")
RUN_ID = "synthetic-alpha0-60k"
VARIANT = "s1-synthetic-baseline"
VOLUME = SYNTHETIC_VOLUME
SHARDS = 4

app = modal.App(APP_S1_SYNTHETIC)
volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
s1_volume = modal.Volume.from_name(S1_VOLUME, create_if_missing=False)
mounts = {"/pipeline": volume, "/s1": s1_volume}

image = (modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("build-essential")
    .uv_pip_install("vllm==0.18.1", "transformers==4.57.6", "rdkit==2025.9.6")
    .env({"CUDA_HOME": "/usr/local/cuda", "TOKENIZERS_PARALLELISM": "false",
          "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    .env(CONTAINER_ENV))
image = with_code(image)


def paths():
    root = pathlib.Path("/pipeline/runs") / RUN_ID / VARIANT
    return root, pathlib.Path(f"/s1/runs/{S1_RUN_ID}/model")


@app.function(image=image, volumes=mounts, cpu=2, memory=8192, timeout=900, retries=0)
def initialize(suite: str, test_raw: bytes, prompts_raw: bytes, labels_raw: bytes, source_info: dict):
    import hashlib
    import sys
    from resonance.stages.common import checkpoint, write_json

    volume.reload()
    root, model = paths()
    s1_volume.reload()
    checkpoint(model)
    evaluation = root / "eval" / suite
    evaluation.mkdir(parents=True, exist_ok=True)
    identity = {
        "run_id": RUN_ID,
        "suite": suite,
        "source": source_info,
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


@app.function(image=image, volumes=mounts, gpu="H200:2", cpu=16,
              memory=192 * 1024, timeout=4 * 3600, max_containers=SHARDS, retries=0)
def benchmark_shard(task: tuple):
    suite, shard = task
    import os
    import subprocess
    import sys
    from resonance.stages.common import checkpoint

    volume.reload()
    root, model = paths()
    s1_volume.reload()
    checkpoint(model)
    evaluation = root / "eval" / suite
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


@app.function(image=image, volumes=mounts, cpu=4, memory=16384,
              timeout=1800, retries=0)
def evaluate(suite: str):
    import sys
    from collections import defaultdict
    from resonance.stages.common import iter_jsonl, write_json

    volume.reload()
    root, _ = paths()
    evaluation = root / "eval" / suite
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


def prepare_inputs():
    import gzip
    import hashlib
    manifest_path = run_dir("forward_enum") / "outputs/subsets_60k/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    suites = []
    path = run_dir("uspto") / "test.json"
    raw = path.read_bytes()
    suites.append(("uspto", raw,
                   (run_dir("uspto_classes") / "outputs/uspto_labels_test.json").read_bytes(),
                   {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "split": "test"}))
    for alpha in ("0", "0.5", "1"):
        path = manifest_path.parent / f"alpha_{alpha}/60k/test.jsonl.gz"
        rows, labels = [], {}
        with gzip.open(path, "rt") as handle:
            for idx, line in enumerate(handle):
                row = json.loads(line)
                reactants = row["reactant_smiles"]
                assert isinstance(reactants, list) and reactants
                rows.append({"original_smiles": {
                    "Starting Material": reactants[0], "Reagents": ".".join(reactants[1:]),
                    "End Product": row["product_smiles"]}})
                labels[str(idx)] = {"level1": f"{row['template_id']}: {row['template_name']}",
                                    "template_id": row["template_id"], "template_name": row["template_name"]}
        assert len(rows) == manifest["configs"][f"alpha_{alpha}"]["split_totals"]["test"]
        suites.append((f"synthetic_alpha_{alpha}", json.dumps(rows).encode(),
                       json.dumps({"labels": labels}).encode(),
                       {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                        "split": "test", "alpha": alpha, "grouping": "reaction template"}))
    return suites


@app.local_entrypoint()
def main(check_only: bool = False):
    suites = [item for item in prepare_inputs() if item[0] != "uspto"]
    if check_only:
        print(json.dumps({suite: len(json.loads(data)) for suite, data, _, _ in suites}, indent=2))
        return
    for suite, test_raw, labels_raw, source_info in suites:
        identity = initialize.remote(suite, test_raw, PROMPTS.read_bytes(), labels_raw, source_info)
        results = list(benchmark_shard.map([(suite, shard) for shard in range(SHARDS)], return_exceptions=True))
        failures = [str(result) for result in results if isinstance(result, Exception)]
        assert not failures, f"Shard failures (other shards allowed to finish): {failures}"
        assert len(results) == SHARDS
        report = evaluate.remote(suite)
        out = ROOT / "outputs" / VARIANT / "eval" / suite
        out.mkdir(parents=True, exist_ok=True)
        (out / "input.json").write_text(json.dumps(identity, indent=2) + "\n")
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"suite": suite, "report": report}, indent=2), flush=True)
