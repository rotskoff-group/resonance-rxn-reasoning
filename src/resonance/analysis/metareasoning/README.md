# `resonance.analysis.metareasoning`

Generation, scoring and paper statistics for the metareasoning experiments
(reasoning-length prompts, feedback conditions, plurality voting).

| File | Role |
|---|---|
| `frozen/` | Copies of the scripts that produced the metareasoning results; since those runs only machine paths and directory names were replaced (hashes in `frozen/MANIFEST.json`). Never edit. |
| `verify.py` | Check `frozen/` against the manifest and, with `--protocol`, against a run's recorded hashes. |
| `resample.py` | `frozen/build_eval_set_resampled.py` + per-reaction seeds: k draws per reaction; draw 0 becomes the turn-1 answer, the rest the fresh baseline. |
| `runner.py` | `frozen/selfcorrect_runner.py` + per-reaction seeds: generates each condition. Prompts: `frozen/conditions.py`. |
| `summary.py` | Paper statistics and `resonance.plots` inputs from condition outputs. |

## Seeding

The frozen scripts pass one `seed` to every vLLM request, so draw *j* uses the
same random stream for every reaction and draws are correlated across reactions
(on the Kimi 8B data, per-draw accuracy varied by ~6 pp; independent draws give
±0.65 pp). `resample.py` and `runner.py` default to `--seed-mode per_reaction`
(seed + eval_idx) and record `seed_mode` in each output; `--seed-mode shared`
reproduces the frozen behaviour. `diff runner.py frozen/selfcorrect_runner.py`
shows the whole change.

## Statistics (`summary.py`)

- Accuracy counts every reaction; an unparsable response is wrong. The earlier
  definition (both turns must parse) is reported as `chemical_channel`.
- Delta is paired by reaction, averaging each reaction's draws; SE is question-level.
  With per-reaction seeds reactions are independent, so this SE is valid.
- p-values: two-sided normal on Delta/SE; exact McNemar when both sides are single draws.
- Reasoning tokens: tokenizer offsets inside `<think>...</think>`; prefilled think
  starts at 0; unclosed reasoning stops at the first answer marker.
- Classes: the `corrected` field of the relabel file (`frozen/relabel_classes.py`).

Config format and definitions: `summary.py` docstring. The paper's config is
`src/resonance/configs/paper/figures/kimi_no_id_paper.json`: Figs 4 and 6 read the
per-reaction-seed runs, Fig 5 the 8-draw feedback run (`eight_rollouts_dir`).

## Verification against the recorded runs

- Before machine paths were replaced, `frozen/` matched the hashes the generation run recorded
  in its `protocol.json`; `verify --protocol` therefore applies only to the original copies.
  `verify` alone checks `frozen/` against `MANIFEST.json`.
- `frozen/models.py` is the registry of an earlier lineage (Sonnet-teacher 8B, 30B Thinking);
  the paper's checkpoints are the stage paths in `kimi_no_id_paper.json` and
  `resonance.paper.regenerate`.
- On the shared-seed outputs, `summary.py`'s `chemical_channel` reproduces the
  previous figure values exactly (n and accuracy, all 18 single-draw cells), and its
  plurality table matches `e1_by_class_k13579.json` exactly.
