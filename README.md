# Resonance

Code for staged distillation of reaction-prediction reasoning: teacher traces over USPTO-50k are
distilled into Qwen3-8B and Qwen3.8-27B by supervised fine-tuning (S1), rejection-sampling
fine-tuning on the student's own correct rollouts (S2) and DPO with an NLL term on self-play pairs
(S3). The repository also contains the synthetic out-of-distribution test sets (forward template
enumeration over PubChem molecules), the reaction-class labelling, the metareasoning experiments
and every main-text figure.

[`PIPELINE.md`](PIPELINE.md) maps each stage to its module, config and launcher, lists the
choices kept exactly as they ran, and records how each step was checked against the original
outputs.

## Installation

Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.11
uv sync --frozen --extra train
```

`--frozen` installs the locked versions; the 8B stages rely on TRL 1.3.0 and vLLM 0.18.0 as
locked. The 27B stages run in the Modal images defined next to them
(`src/resonance/stages/qwen3_8_27b/modal_apps/`), which pin their own versions.

## Data

Every script resolves its inputs and outputs through `resonance.paths`: one root,
`RESONANCE_DATA_ROOT` (default `./data`), holding one directory per key in `RUN_DIRS`
(`uspto`, `traces`, `s1_8b`, `s2_8b`, `s3_8b`, `pipeline_27b`, …).

```bash
export RESONANCE_DATA_ROOT=/path/to/data
python -m resonance.pipelines.uspto.convert_to_forward_rxn_format <USPTO-50k jsonl dir> -o $RESONANCE_DATA_ROOT/uspto
```

Credentials, where a step needs them, are listed in `.env.example`.

## Main-text figures

From the stage outputs:

```bash
python -m resonance.paper.run_all --out figures/
```

`resonance.plots` renders each figure from a single JSON input
(`src/resonance/plots/README.md`).

## Layout

| Path | Contents |
|---|---|
| `src/resonance/pipelines/` | USPTO-50k conversion, teacher traces, forward enumeration, class labels, synthetic OOD sets |
| `src/resonance/stages/` | Qwen3-8B and Qwen3.8-27B S1 → S2 → S3, benchmarks, stage ablation |
| `src/resonance/workflows/` | Hydra entry points for SFT, DPO, benchmarking and evaluation |
| `src/resonance/analysis/metareasoning/` | Metareasoning generation and statistics |
| `src/resonance/paper/`, `src/resonance/plots/` | Figure inputs and rendering |
| `src/resonance/configs/paper/` | Configs of the reported runs |
| `launchers/` | Slurm and shell launchers per stage (submit from the repository root) |
