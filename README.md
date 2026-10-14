# Resonance

Code, models and data for the preprint **Distillation, Self-Training, and the Fragility of Learned
Chemical Reasoning**.

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
| Data | [`anaghaa/resonance`](https://huggingface.co/datasets/anaghaa/resonance) (under `v1/`) | USPTO-50k split and corrected class labels, the 40,000 Kimi K3 teacher traces, every S2 and S3 training set, every model's test predictions, and the reasoning-analysis samples. See [docs/DATA.md](docs/DATA.md). |
| Code | this repository | The training and evaluation workflows, the data pipelines, the reasoning analysis and every figure in the paper. |

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

## 1. Evaluate a released model

The `benchmark` workflow runs a model on the USPTO-50k test set with the paper's prompt, decoding
(T 0.6, top-p 0.95, top-k 20) and scoring:

```bash
export RESONANCE_DATA_ROOT=$PWD/data
uv run python -m resonance.download all          # includes the USPTO-50k split
uv run python -m resonance.workflows.benchmark \
    --config-path $PWD/src/resonance/configs/paper/qwen3_8b --config-name s1_benchmark \
    inference.model_name=anaghaa/Resonance-8B-S3 output_path=$PWD/results/8b_s3.jsonl
uv run python -m resonance.workflows.evaluate results_path=$PWD/results/8b_s3.jsonl plot.output_dir=$PWD/results/plots
```

`results/8b_s3.jsonl` holds one response per reaction. The prediction is the last `<answer>` block,
scored by `resonance.evaluation.extractors` (canonical-SMILES exact match). This is the 8B
evaluation setup. The 27B evaluations used the long system prompt in
`configs/paper/qwen3_8_27b/prompts.json` and need vLLM's `limit_mm_per_prompt={'image': 0, 'video': 0}`
(Qwen3.8-27B is a vision-language checkpoint used text-only).

- **Caches:** model weights download to the Hugging Face cache (`HF_HOME`, default
  `~/.cache/huggingface`) and vLLM compiles into `VLLM_CACHE_ROOT` (default `~/.cache/vllm`);
  point both at a large disk if your home directory is small.
- **Warnings:** a tokenizer warning about an "incorrect regex pattern" is harmless.

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

## 3. Retrain

`python -m resonance.download all` places every released training set (the S1 teacher-trace
dataset, the S2 sets and the S3 pairs) where the paper configs read them. Each 8B stage then runs
through a workflow:

```bash
C=$PWD/src/resonance/configs/paper/qwen3_8b
uv run python -m resonance.workflows.sft --config-path $C --config-name s1_sft
uv run python -m resonance.workflows.sft --config-path $C --config-name s2_sft resume_from_checkpoint=null
uv run python -m resonance.workflows.dpo --config-path $C --config-name s3_dpo
```

[PIPELINE.md](PIPELINE.md) has the full guide:
- what each stage reads, and the paper's settings and hardware;
- the 27B hyperparameters;
- the run settings that look odd but were kept exactly as they ran.

## Repository map

| Path | Contents |
|---|---|
| `src/resonance/paths.py` | Every data location: `RESONANCE_DATA_ROOT` plus one key per run directory (see [docs/DATA.md](docs/DATA.md)) |
| `src/resonance/workflows/` | Hydra entry points for SFT, DPO, benchmarking and evaluation |
| `src/resonance/configs/` | Workflow defaults, and the configs of the paper's runs (`configs/paper/`) |
| `src/resonance/data/`, `training/`, `inference/` | Parsers and prompts, trainer construction, vLLM inference |
| `src/resonance/evaluation/` | The scorer (`extractors.py`): answer extraction and canonical SMILES matching |
| `src/resonance/pipelines/` | USPTO-50k conversion, teacher-trace generation, reaction-class labels |
| `src/resonance/analysis/` | Reasoning probes (`metareasoning/`) and the backtracking detector |
| `src/resonance/paper/`, `src/resonance/plots/` | Figure inputs and rendering |
| `src/resonance/download.py`, `release/` | Fetch the released data; Hugging Face ids |

## Citation

```bibtex
@misc{aneesh2026resonance,
  title  = {Distillation, Self-Training, and the Fragility of Learned Chemical Reasoning},
  author = {Aneesh, Anagha and Ibarraran, Sebastian and Hu, Frank and Liu, Jason and Rotskoff, Grant},
  year   = {2026},
  note   = {Preprint}
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
