# `resonance.plots`: paper figures

Every paper figure is a pure function of one JSON file. This package holds only
the plotting code; the paper's inputs are produced by `resonance.paper.run_all`
(see [docs/FIGURES.md](../../../docs/FIGURES.md)).

```bash
python -m resonance.plots.run --list
python -m resonance.plots.run --data-dir DATA --out-dir FIGURES          # all present
python -m resonance.plots.run --data-dir DATA --out-dir FIGURES --only plurality
python -m resonance.plots.run --data-dir DATA --out-dir FIGURES --style main_text \
    --map Fig_5=feedback_delta --only Fig_5      # render Fig_5.json with feedback_delta
```

`--style` picks a type/width preset from `style.PRESETS` (`default`: 5.5 in, 8-9 pt;
`main_text`: 7.5 in, 14-16 pt). `--map FILE=FIGURE` renders an arbitrarily named
input with a registered figure's module.

`run` renders `<name>.pdf` and `<name>.png` for each `<name>.json` it finds in
`DATA`, skips figures whose file is absent, and exits non-zero if any input
fails validation (the message names the missing or malformed field).

To see a valid input for every figure, generate synthetic templates:

```bash
python -m resonance.plots.examples --out /tmp/plot_examples
python -m resonance.plots.run --data-dir /tmp/plot_examples --out-dir /tmp/plot_examples/figures
```

## Figures

| File | Module | Figure |
|---|---|---|
| `staged_accuracy.json` | `staged_accuracy` | Stage accuracy with 95% Wilson intervals next to reference models |
| `class_progression.json` | `class_progression` | (a) training reactions per family, (b) student − teacher accuracy, (c) family trajectories |
| `length_prompts.json` | `intervention_scatter` | Δ accuracy vs reasoning tokens for commit / consider / think more |
| `counterfactual.json` | `intervention_scatter` | Same form for told correct / told wrong feedback |
| `feedback_delta.json` | `feedback_delta` | Accuracy change per condition as labelled dots (±SE optional) |
| `plurality.json` | `plurality` | Plurality@k accuracy by reaction class |
| `stage_ablation.json` | `stage_ablation` | (a) base accuracy, (b) gain over base per path, (c) paired stage increments |

The exact schema, with an example, is in each module's docstring. Common rules:

- **Counts, not rates**, wherever an interval or test is drawn: `correct` and
  `n` (or `n_test`). Accuracies are computed in the module.
- **Order is display order.** Lists of stages, families, conditions and paths
  plot in the order given.
- **Missing results are allowed where noted.** In `stage_ablation`, a path key
  absent from a family's `correct` renders as "–", so pending ablations stay in
  the layout.
- **Axis labels, ticks and `figsize`** can be set in the input where the module
  docstring lists them, so paper-specific wording lives with the data.
- **Colours** default to the house palette in `style.py`. Families are coloured
  by name (`style.FAMILY_COLORS`); any series, bar or condition may set `color`
  (and `marker` for scatter conditions).

## Statistics

- `staged_accuracy`: 95% Wilson score interval per bar.
- `intervention_scatter` / `feedback_delta`: error bars are drawn only from supplied
  `delta_se_pp` / `tokens_se`; nothing is estimated here.
- `stage_ablation` (c): increment = (second-only − first-only correct) / n from
  paired per-reaction outcomes. `*` marks exact two-sided McNemar p < `alpha`
  after Holm correction over all cells of the panel.

## Adding a figure

1. Write `<module>.py` with a docstring giving the input format, plus
   `validate(data)` (raise `schema.DataError`) and `plot(data) -> matplotlib.Figure`.
   Use `style` constants so type and colour match the other figures.
2. Register it in `run.FIGURES` with the data-file stem.
3. Add a synthetic input to `examples.py`.

## Style

Figures are drawn at `style.TEXTWIDTH` (5.5 in, NeurIPS `\linewidth`) with STIX
serif type, so `\includegraphics[width=\linewidth]{...}` never rescales them.
`staged_accuracy` deliberately uses larger tick and label type.
