"""Render every paper figure whose input JSON is present in a data directory.

    python -m resonance.plots.run --data-dir DATA --out-dir FIGURES
    python -m resonance.plots.run --data-dir DATA --out-dir FIGURES --only plurality
    python -m resonance.plots.run --list

Each figure reads one ``<name>.json`` from DATA and writes ``<name>.pdf/.png``
to FIGURES. Formats are documented in each module's docstring and README.md.
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from . import (class_progression, feedback_delta, intervention_scatter, plurality,
               stage_ablation, staged_accuracy, style)
from .schema import DataError, load


@dataclass(frozen=True)
class Figure:
    name: str            # data file stem and output stem
    module: ModuleType   # provides validate(data) and plot(data) -> matplotlib Figure
    description: str


FIGURES = [
    Figure("staged_accuracy", staged_accuracy, "Stage accuracy vs frontier models"),
    Figure("class_progression", class_progression, "Three-panel per-family progression"),
    Figure("length_prompts", intervention_scatter, "Commit / consider / think more"),
    Figure("counterfactual", intervention_scatter, "Told correct / told wrong feedback"),
    Figure("feedback_delta", feedback_delta, "Accuracy change per feedback condition (dots)"),
    Figure("plurality", plurality, "Plurality@k by reaction class"),
    Figure("stage_ablation", stage_ablation, "Per-family stage-ablation heatmap"),
]


def render(figure, data_dir, out_dir, formats):
    data = load(Path(data_dir) / f"{figure.name}.json")
    figure.module.validate(data)
    return style.save(figure.module.plot(data), out_dir, figure.name, formats)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--only", nargs="+", help="figure (or --map file) names to render")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--style", default="default", choices=sorted(style.PRESETS))
    ap.add_argument("--map", nargs="+", default=[], metavar="FILE=FIGURE",
                    help="render FILE.json with FIGURE's module, e.g. fig6a=feedback_delta")
    ap.add_argument("--list", action="store_true", help="list figures and exit")
    args = ap.parse_args(argv)
    if args.list:
        for f in FIGURES:
            print(f"{f.name:<20} {f.description}")
        return 0
    if not args.data_dir or not args.out_dir:
        ap.error("--data-dir and --out-dir are required")

    style.use(args.style)
    figures = list(FIGURES)
    by_name = {f.name: f for f in FIGURES}
    for item in args.map:
        stem, _, kind = item.partition("=")
        if kind not in by_name:
            ap.error(f"--map {item}: unknown figure {kind!r}")
        figures.append(Figure(stem, by_name[kind].module, by_name[kind].description))
    failed = False
    for figure in figures:
        if args.only and figure.name not in args.only:
            continue
        path = args.data_dir / f"{figure.name}.json"
        if not path.exists():
            if args.only:
                print(f"FAIL {figure.name}: {path} not found", file=sys.stderr)
                failed = True
            else:
                print(f"skip {figure.name}: no {path.name}")
            continue
        try:
            outputs = render(figure, args.data_dir, args.out_dir, args.formats)
        except DataError as exc:
            print(f"FAIL {figure.name}: {exc}", file=sys.stderr)
            failed = True
            continue
        print(f"ok   {figure.name}: " + ", ".join(str(p) for p in outputs))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
