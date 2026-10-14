"""Full Qwen3.8-27B SFT on the regenerated no-reaction-ID corpus.

Ported from run_dir('s1_27b')/modal/full_sft.py.
Launch: python -m modal run --detach src/resonance/stages/qwen3_8_27b/modal_apps/full_sft.py
"""
import json
import pathlib

import modal

from resonance.paths import run_dir
from resonance.stages.qwen3_8_27b.config import (
    APP_S1,
    CONTAINER_ENV,
    S1_VOLUME,
    traces_path,
    with_code,
)
from resonance.stages.qwen3_8_27b.config import BASE_MODEL as MODEL
from resonance.stages.qwen3_8_27b.config import BASE_REVISION as REVISION

ROOT = run_dir('s1_27b')

SOURCE = traces_path()
app = modal.App(APP_S1)
volume = modal.Volume.from_name(S1_VOLUME, create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("build-essential", "git")
    .uv_pip_install("torch==2.10.0", extra_index_url="https://download.pytorch.org/whl/cu126")
    .uv_pip_install("transformers==5.17.0", "accelerate", "datasets", "deepspeed==0.19.0",
                    "flash-linear-attention", "ninja", "packaging", "wheel", "setuptools", "psutil")
    .run_commands("MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=9.0 pip install causal-conv1d --no-build-isolation")
    # FLA 0.5.2 rejects Triton 3.6's incorrect Hopper gated backward kernel.
    .uv_pip_install("tilelang==0.1.14")
    .env({"HF_HOME": "/pilot/hf", "TOKENIZERS_PARALLELISM": "false",
          "OMP_NUM_THREADS": "4", "PYTORCH_ALLOC_CONF": "expandable_segments:True"})
    .env(CONTAINER_ENV)
)
image = with_code(image, "/pilot_code")


@app.function(image=image, volumes={"/pilot": volume}, cpu=4, memory=16384,
              timeout=3600, retries=0)
def prepare(raw: bytes, cached: bool = False):
    import hashlib
    import importlib.metadata
    import random

    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    base = pathlib.Path("/pilot")
    if cached:
        stats = json.loads((base / "data_stats.json").read_text())
        assert stats["source_sha256"] == hashlib.sha256(raw).hexdigest()
        assert stats["revision"] == REVISION
        return stats
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION)
    assert config.model_type == "qwen3_5"
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(records) == 40000, len(records)
    system = ("You are an expert organic chemist specializing in the forward prediction of organic reactions. "
              "Your task is to predict the major product of organic reactions given starting materials and reagents. "
              "Return the answer between <answer> and </answer> tags.")
    texts = []
    for row in records:
        trace, product = row["trace"].strip(), row["product"].strip()
        assert trace and product
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Here are the SMILES of the starting material and reagent(s).\n"
             f"Starting material: {row.get('starting_material', '')}\nReagents: {row.get('reagents', '')}"},
            {"role": "assistant", "reasoning_content": trace,
             "content": f"<answer>\n{product}\n</answer>"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                             enable_thinking=True, preserve_thinking=True)
        assert trace in text and f"<answer>\n{product}\n</answer>" in text
        assert "<think>" in text and "</think>" in text
        texts.append(text)
    ids = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
    lengths = sorted(map(len, ids))
    stats = {"model": MODEL, "revision": REVISION, "source_sha256": hashlib.sha256(raw).hexdigest(),
             "examples": len(ids), "tokens_per_epoch": sum(lengths), "mean_tokens": sum(lengths)/len(ids),
             "p50": lengths[len(ids)//2], "p99": lengths[int(len(ids)*.99)], "max_tokens": lengths[-1],
             "thinking": True, "think_source": "trace",
             "loss": "all non-padding tokens (matches prior text-column SFT)",
             "packages": {p: importlib.metadata.version(p) for p in
                          ["torch", "transformers", "accelerate", "deepspeed", "flash-linear-attention",
                           "causal-conv1d"]}}
    assert lengths[-1] <= 4096, "Unexpected long traces: inspect before allocating GPUs"
    Dataset.from_dict({"input_ids": ids}).save_to_disk(str(base / "dataset"))
    # First optimizer step exercises the 32 longest real examples; subsequent data is shuffled.
    tail = sorted(range(len(ids)), key=lambda i: len(ids[i]), reverse=True)[:32]
    tail_set = set(tail)
    rest = [i for i in range(len(ids)) if i not in tail_set]
    random.Random(42).shuffle(rest)
    Dataset.from_dict({"input_ids": [ids[i] for i in tail + rest[:3168]]}).save_to_disk(str(base / "pilot_dataset"))
    (base / "data_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)
    snapshot_download(MODEL, revision=REVISION,
                      allow_patterns=["*.json", "*.jinja", "*.safetensors"])
    volume.commit()
    return stats


@app.function(image=image, volumes={"/pilot": volume}, gpu="H200:8", cpu=32,
              memory=256*1024, ephemeral_disk=2*1024*1024,
              timeout=24*3600, retries=0, max_containers=1)
def train(run_id: str):
    import os
    import subprocess
    import time
    volume.reload()
    out = pathlib.Path("/pilot/runs") / run_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE.json").exists():
        return (out / "COMPLETE.json").read_text()
    env = {**os.environ, "PILOT_OUTPUT": str(out), "PILOT_MODEL": MODEL,
           "PILOT_REVISION": REVISION, "HF_HUB_OFFLINE": "1", "PILOT_VOLUME": S1_VOLUME}
    started = time.monotonic()
    status = "failed"
    try:
        subprocess.run(["python", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
                        "/pilot_code/train_full.py"], env=env, check=True, timeout=23*3600)
        assert (out / "COMPLETE.json").exists(), "Training exited without final checkpoint"
        status = "completed"
    finally:
        (out / "execution.json").write_text(json.dumps({"status": status,
            "wall_seconds": time.monotonic()-started, "automatic_retries": 0}, indent=2))
        volume.commit()
    return (out / "COMPLETE.json").read_text()


@app.local_entrypoint()
def main():
    from resonance.pipelines.traces.validate_no_id import validate
    manifest = validate(SOURCE)
    run_id = "no-id-" + manifest["source_sha256"][:16]
    target = ROOT / "outputs" / run_id
    target.mkdir(parents=True, exist_ok=True)
    stats = prepare.remote(SOURCE.read_bytes())
    (target / "data_stats.json").write_text(json.dumps(stats, indent=2))
    (target / "run.json").write_text(json.dumps({"run_id": run_id, "volume": S1_VOLUME, **manifest}, indent=2))
    result = train.remote(run_id)
    (target / "COMPLETE.json").write_text(result)
    print(result)
