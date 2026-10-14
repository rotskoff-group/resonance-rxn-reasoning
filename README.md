# Resonance

Code, models and data for **Distillation, Self-Training, and the Fragility of Learned Chemical
Reasoning** (ICLR 2027).

Resonance trains open-weight models to predict the products of chemical reactions. Each
reaction is given as SMILES, and the model reasons step by step before answering. A teacher
(Kimi K3) writes reasoning traces for the USPTO-50k training reactions. The student is
fine-tuned on those traces (**S1**), then on its own verified-correct answers (**S2**,
rejection-sampling SFT), then on preference pairs drawn from its own correct and incorrect
answers (**S3**, DPO). The paper trains two students, Qwen3-8B and Qwen3.8-27B, and then
analyses how the 8B student reasons.

Top-1 exact-match accuracy on the USPTO-50k test set (N = 5,000):

| | base | S1 | S2 | S3 | skip-S2 (ablation) |
|---|---|---|---|---|---|
| Resonance-8B (Qwen3-8B) | 1.2 | 31.0 | 38.9 | **42.8** | 37.1 |
| Resonance-27B (Qwen3.8-27B) | 26.3 | 79.0 | 79.3 | **79.6** | 78.8 |
| *Zero-shot references* | Kimi K3 (teacher) 57.4 | Claude Opus 5 78.7 | GPT-6-Astra 63.0 | | |

New to the project? Start with [docs/GLOSSARY.md](docs/GLOSSARY.md), then this page.

## What is released

| | Where | What |
|---|---|---|
| Models | [Hugging Face collection](https://huggingface.co/anaghaa) | All 8 checkpoints in the paper: `Resonance-{8B,27B}-{S1,S2,S3,skip-S2}`. Each is a merged, standalone model; stages trained with LoRA (8B S3 and skip-S2, every 27B stage after S1) also ship their adapter. |
| Data | [`anaghaa/resonance`](https://huggingface.co/datasets/anaghaa/resonance) (under `iclr2027/`) | USPTO-50k split and corrected class labels, the 40,000 Kimi K3 teacher traces, every S2 and S3 training set, every model's test predictions, and the reasoning-analysis samples. See [docs/DATA.md](docs/DATA.md). |
| Code | this repository | Training, sampling, evaluation, the reasoning analysis and every figure in the paper. |

Model ids are listed in `src/resonance/release/artifacts.py`.

## Install

Linux, Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/rotskoff-group/resonance-rxn-reasoning.git && cd resonance-rxn-reasoning
uv python install 3.11
uv sync --frozen                  # add --extra train to train models
```

`--frozen` installs the exact versions the paper used (vLLM 0.18.0, TRL 1.3.0). The figures
need no GPU. Running a model needs one 80 GB GPU for 8B or two for 27B.

## 1. Predict reaction products

```bash
uv run python -m resonance.predict --model anaghaa/Resonance-8B-S3 \
    --starting-material "Fc1cc(I)ccn1" --reagents "OB(O)c1ccc2ccccc2c1"
```

The command prints one JSON line to stdout (vLLM's progress goes to stderr) containing:
- `answer`: the model's last `<answer>` block
- `prediction`: its canonical SMILES, or `null` if it is not valid SMILES
- `response`: the full response

This example is a Suzuki coupling from the USPTO-50k test set. Resonance-8B-S3 answers
`Fc1cc(-c2ccc3ccccc3c2)ccn1`, which is the recorded product. The models sample at T 0.6, like
the paper's evaluation, and Resonance-8B-S3 is right on 42.8% of test reactions, so expect
some wrong answers.

- **Batches:** pass `--input reactions.jsonl --out predictions.jsonl` with one
  `{"starting_material": ..., "reagents": [...], "product": ...}` object per line. `reagents` is
  a list of SMILES, and a string is split on "."; `product` is optional, and reactions that have
  one are scored.
- **Prompt and scoring:** the prompt, decoding (T 0.6, top-p 0.95, top-k 20) and answer
  extraction are the paper's. The 27B prompt is chosen when the model name contains "27b";
  for a local directory with another name, pass `--family 27b`.
- **Your own client:** `resonance.predict.messages()` returns the exact chat messages if you
  want to call the model some other way.
- **Caches:** model weights download to the Hugging Face cache (`HF_HOME`, default
  `~/.cache/huggingface`) and vLLM compiles into `VLLM_CACHE_ROOT` (default `~/.cache/vllm`);
  point both at a large disk if your home directory is small.
- **Warnings:** a tokenizer warning about an "incorrect regex pattern" is harmless.
- **27B kernel build errors:** the 27B models JIT-compile flashinfer kernels on first use, into
  `~/.cache/flashinfer` by default. If a "Ninja build failed" error mentions another Python
  environment's paths, that cache was built by a different vLLM install. Set
  `FLASHINFER_WORKSPACE_BASE` to an empty directory, or delete the stale cache.

## 2. Reproduce the paper's figures (CPU only)

```bash
export RESONANCE_DATA_ROOT=$PWD/data
uv run python -m resonance.download figures     # about 4 GB: every figure input
uv run python -m resonance.paper.run_all --out paper_out
```

If you already have a copy of the dataset repo (e.g. `git clone https://huggingface.co/datasets/anaghaa/resonance`),
pass `--from-dir <that directory>` to `resonance.download` instead of downloading again.

This takes about 40 minutes and 6 GB of RAM. It writes
`paper_out/figures/Fig_{2..7,A1,A2}.{pdf,png}`, pixel-identical to the paper's figures, and the
numbers behind them in `paper_out/data/`:
- `class11_counts.json`: per-class correct counts for every model
- `summary.json`: every reasoning-analysis statistic

[docs/FIGURES.md](docs/FIGURES.md) maps each figure and table to the module that makes it.

## 3. Retrain or regenerate

`python -m resonance.download all` places every released input (the USPTO-50k split, the
teacher traces, the S2 training sets and the S3 pairs) where the stage scripts read them, so any
stage can be rerun from the previous stage's released outputs.

[PIPELINE.md](PIPELINE.md) maps each part of the pipeline to its code and lists the run
settings that look odd but were kept exactly as they ran. The per-stage guides give the
launcher, inputs, outputs and hardware of every step:
- 8B: [`stages/qwen3_8b/README.md`](src/resonance/stages/qwen3_8b/README.md)
- 27B: [`stages/qwen3_8_27b/README.md`](src/resonance/stages/qwen3_8_27b/README.md)

In short:

| Stage | 8B (Slurm, 8×H100) | 27B (Modal, 8×H200) |
|---|---|---|
| Teacher traces | `resonance.pipelines.traces` | (same traces) |
| S1, S2, S3, benchmarks | `src/resonance/stages/qwen3_8b/` ([README](src/resonance/stages/qwen3_8b/README.md)), `launchers/qwen3_8b/` | `src/resonance/stages/qwen3_8_27b/`, `launchers/qwen3_8_27b/` |
| Reasoning probes | `resonance.paper.regenerate` | – |

## Repository map

| Path | Contents |
|---|---|
| `src/resonance/paths.py` | Every data location: `RESONANCE_DATA_ROOT` plus one key per run directory (see [docs/DATA.md](docs/DATA.md)) |
| `src/resonance/predict.py`, `download.py` | Run a model; fetch the released data |
| `src/resonance/pipelines/` | USPTO-50k conversion, teacher traces, reaction-class labels |
| `src/resonance/stages/` | Qwen3-8B and Qwen3.8-27B S1 → S2 → S3, benchmarks, stage ablation |
| `src/resonance/workflows/` | Hydra entry points for SFT, DPO, benchmarking and evaluation |
| `src/resonance/evaluation/` | The scorer (`extractors.py`): answer extraction and canonical SMILES matching |
| `src/resonance/analysis/` | Reasoning probes (`metareasoning/`) and the backtracking detector |
| `src/resonance/paper/`, `src/resonance/plots/` | Figure inputs and rendering |
| `src/resonance/release/` | Release tooling: artifact ids, checkpoint replay check |
| `src/resonance/configs/paper/` | Configs of the runs in the paper |
| `launchers/` | Slurm and shell launchers per stage (submit from the repository root after `mkdir -p logs`) |

`src/resonance/pipelines/forward_enum`, `synthetic_data` and `synthetic_ood` build a synthetic
out-of-distribution benchmark from an earlier draft of this work. It is **not** part of the
ICLR 2027 paper.

## Citation

```bibtex
@inproceedings{aneesh2027resonance,
  title     = {Distillation, Self-Training, and the Fragility of Learned Chemical Reasoning},
  author    = {Aneesh, Anagha and Ibarraran, Sebastian and Hu, Frank and Liu, Jason and Rotskoff, Grant},
  booktitle = {International Conference on Learning Representations},
  year      = {2027}
}
```

## Development

CI runs these checks on every push. They run the same way locally:

```bash
uv sync --frozen
uv run ruff check .          # pycodestyle, pyflakes, import order, bugbear, pyupgrade
uv run ty check .            # type check
uv run python -m pytest -q tests
uv run python -m resonance.analysis.metareasoning.verify   # frozen scripts unchanged
```

The rules are in `pyproject.toml`. `src/resonance/analysis/metareasoning/frozen/` is excluded on
purpose: it holds verbatim copies of the scripts that generated the reasoning-probe samples,
pinned by hash. Three prompt files are exempt from the line-length rule, because their line
breaks are part of the prompts.

## License

The code is released under the [MIT License](LICENSE). USPTO-50k is derived from public US
patent data (Lowe 2012; Schneider et al. 2016). The models are fine-tuned from Qwen3-8B and
Qwen3.8-27B and carry their licenses.
