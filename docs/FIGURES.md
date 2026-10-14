# Figures and tables

`python -m resonance.paper.run_all --out <dir>` rebuilds every generated figure into
`<dir>/figures` and its numbers into `<dir>/data`. The rebuild runs on a CPU and needs only the
released figure inputs (`python -m resonance.download figures`). The steps can also be run one
at a time, in this order:

| Output | Module | Reads (`run_dir` key: path) | Paper |
|---|---|---|---|
| (check) | `resonance.analysis.metareasoning.verify` | `frozen/` against `frozen/MANIFEST.json` | – |
| `draw_independence.json` | `resonance.paper.draw_independence --config kimi_no_id_paper.json` | reasoning-probe samples (below) | App. "Design" (draws independent, χ² p = 0.26–1.00) |
| `Fig_4.json`, `Fig_5.json`, `Fig_6.json`, `summary.json` | `resonance.analysis.metareasoning.summary --config kimi_no_id_paper.json` | `paper_figures: runs/per_reaction_seed{,_feedback8}/full/...`, `metareasoning: no_id/data/class_relabel_v2.json`, the S1–S3 `tokenizer.json` | Figs 4–6; appendix table "Change in top-1 accuracy vs. unprompted resample", appendix table "Change in top-1 accuracy vs. neutral review"; plurality@k of appendix table "Plurality@k and pass@k" |
| `class11_counts.json` | `resonance.paper.classes` | every model's test outputs (below), `eval_rigor: outputs/audit/test_reaction_classes.json`, `relabel: data/class_relabel_v2.json`, `paper_figures: data/train_class_relabel.json` | appendix table "Reaction classes in USPTO-50k", appendix table "Top-1 accuracy by class"; every headline accuracy |
| `Fig_2.json` | `resonance.paper.export --data <dir>/data` | `class11_counts.json` | Fig 2 |
| `Fig_2.{pdf,png}` | `resonance.plots.run --style default --map Fig_2=staged_accuracy` | `Fig_2.json` | Fig 2 |
| `Fig_3.{pdf,png}` | `resonance.paper.fig3` | `class11_counts.json` | Fig 3 |
| `Fig_7.{pdf,png}` | `resonance.paper.fig7` | `class11_counts.json` | Fig 7 |
| `Fig_{4,5,6}.{pdf,png}` | `resonance.plots.run --style main_text --map Fig_4=length_prompts Fig_5=feedback_delta Fig_6=plurality` | `Fig_{4,5,6}.json` | Figs 4–6 |
| `backtrack/kimi_s*_draws.json`, `Fig_A_backtrack.json` | `resonance.paper.backtrack` | the probe samples, `relabel: data/turn1_t06_q8_base.json`, `traces: outputs/traces_stage1a_no_reaction_id.jsonl` and the S1 tokenizer | appendix table "Backtracking by source" |
| `Fig_A{1,2}.{pdf,png}` | `resonance.paper.fig_appendix` | `Fig_A_backtrack.json` | Figs A1, A2 |

## What each figure shows

| Figure | Content |
|---|---|
| Fig 1 | The pipeline schematic: teacher traces → S1 → S2 → S3. Drawn by hand, not generated. |
| Fig 2 | Top-1 accuracy of Resonance-8B at each stage, next to Kimi K3, Claude Opus 5 and GPT-6-Astra, with 95% Wilson intervals. |
| Fig 3 | Resonance-8B by reaction class: (A) student minus teacher accuracy per class and stage; (B) accuracy per class across stages. |
| Fig 4 | Reasoning-length prompts (commit / consider alternatives / think more): change in accuracy against reasoning tokens, per 8B stage. |
| Fig 5 | Feedback after the first answer (mild doubt / told wrong / marked wrong / told correct): change in accuracy against a neutral review. |
| Fig 6 | Plurality@k (k = 1…9) accuracy by reaction class and 8B stage. |
| Fig 7 | Resonance-27B accuracy by reaction class: base, S1, S2, S3, skip-S2, Kimi K3, Claude Opus 5. |
| Fig A1 | Backtracking in the base model, the teacher corpus and the S1–S3 samples. |
| Fig A2 | Change in accuracy against backtracking rate for the Fig 4 and Fig 5 prompts. |

## Test outputs scored by `resonance.paper.classes`

| Model | `run_dir` key: file |
|---|---|
| Qwen3-8B base | `eval_rigor: outputs/bench/bench_base_rules/results.jsonl` |
| Resonance-8B S1 / S2 / S3 | `s1_8b: outputs/benchmark/results.jsonl`, `s2_8b_bench: outputs/results.jsonl`, `s3_8b: outputs/benchmark/results.jsonl` |
| Qwen3.8-27B base, Resonance-27B S1 / S2 / S3 | `pipeline_27b: outputs/no-id-b97ef67b9ba65863/{base,s1}/eval/rollouts.jsonl`, `.../{s2-v5-difficulty-mix-1ep,s3-from-s2-v5-difficulty-mix-1ep}/eval/shards/*/rollouts.jsonl` |
| Resonance-27B skip-S2 | `ablation_27b: outputs/skip_s2/local_eval/generation/rollouts.jsonl` |
| Kimi K3 / Claude Opus 5 / GPT-6-Astra | `teacher_bench: outputs/bench_kimik3_modal/results.jsonl`, `opus_bench: outputs/claude-opus-5/forward_smiles/results_merged.jsonl`, `gpt6_bench: outputs/gpt-6-astra/forward_smiles/results.jsonl` |

The 8B skip-S2 ablation (37.1%) appears only in appendix table "Skip-S2 ablation"; its outputs are at
`ablation_8b: outputs/skip_s2/s3/outputs/benchmark/results.jsonl`.

## Tables that are not generated here

`run_all` does not regenerate these appendix tables:
- paired stage tests (appendix table "Paired tests between stages")
- looser matching (appendix table "Accuracy under looser matching")
- error categories (appendix table "Incorrect predictions that repeat an input")
- student–teacher overlap (appendix table "Paired outcomes against the teacher")
- answer flips (appendix table "Answer flips relative to the first answer")
- the pass@k rows of appendix table "Plurality@k and pass@k" (its plurality@k rows come from `summary.json`)
- structure changes within a trace (appendix table "Structure changes within the trace")
- the few-shot control (appendix table "Few-shot control on base Qwen3-8B"), whose code is not in this repository (see [DATA.md](DATA.md))

All of them are functions of the released per-model test outputs and probe samples
([docs/DATA.md](DATA.md)).

## Error bars

Each figure's error bars are set in `configs/paper/figures/kimi_no_id_paper.json`:
- **Fig 4:** `"error_bars": "draws"`, the standard error across the 8 draws (`se_draws_pp` in `summary.json`).
- **Fig 5:** the reaction-clustered standard error `se_pp` × `error_z` = 1.96, i.e. a 95% interval.
- **Fig 2:** 95% Wilson intervals.
