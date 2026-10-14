"""Per-family effect of each training stage across stage-ablation paths.

(a) base accuracy over all answers and over answers that finished before the
output cap, (b) accuracy gain over base for each training path, (c) paired
per-stage increments; ``*`` marks Holm-corrected exact McNemar p < alpha over
all cells of (c).

Input (``stage_ablation.json``)::

    {"paths": [{"key": "s1", "label": "S1"}, {"key": "s3", "label": "S1→S2→S3"},
               {"key": "base_s3", "label": "S3"}, ...],
     "increments": [{"pair": "s1->s2", "label": "S2−S1"}, ...],
     "alpha": 0.05,                                   # optional
     "families": [
        {"name": "acylation", "short": "Acylation", "n_test": 969,
         "correct": {"base": 365, "s1": 843, "s3": 850},  # a missing path is left blank
         "base_finished": 490, "base_correct_finished": 365,
         "pairs": {"s1->s2": [14, 21]}},                  # [first-only correct, second-only correct]
        ...]}

A path whose key is absent from ``correct`` renders as "–", so pending
ablations can stay in the figure. Base finished columns are optional.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import style
from .schema import DataError, require, require_list
from .stats import holm, mcnemar_exact


def validate(data):
    paths = require_list(data, "paths", "stage_ablation")
    increments = require_list(data, "increments", "stage_ablation")
    for i, p in enumerate(paths):
        require(p, ["key", "label"], f"paths[{i}]")
    for i, inc in enumerate(increments):
        require(inc, ["pair", "label"], f"increments[{i}]")
    for i, fam in enumerate(require_list(data, "families", "stage_ablation")):
        where = f"families[{i}]"
        require(fam, ["name", "n_test", "correct", "pairs"], where)
        require(fam["correct"], ["base"], f"{where}.correct")
        for inc in increments:
            pair = fam["pairs"].get(inc["pair"])
            if pair is not None and (len(pair) != 2 or min(pair) < 0):
                raise DataError(f"{where}.pairs[{inc['pair']!r}]: expected [b, c] counts")


def plot(data):
    fams, paths, increments = data["families"], data["paths"], data["increments"]
    alpha = data.get("alpha", 0.05)

    def acc(fam, key):
        c = fam["correct"].get(key)
        return np.nan if c is None else 100 * c / fam["n_test"]

    has_finished = data.get("show_finished", True) and all("base_finished" in f for f in fams)
    base_cols = [[acc(f, "base")] + ([100 * f["base_correct_finished"] / f["base_finished"]]
                                     if has_finished else []) for f in fams]
    base = np.array(base_cols)
    gain = np.array([[acc(f, p["key"]) - acc(f, "base") for p in paths] for f in fams])
    inc = np.full((len(fams), len(increments)), np.nan)
    pvals = np.ones_like(inc)
    for r, fam in enumerate(fams):
        for c, step in enumerate(increments):
            pair = fam["pairs"].get(step["pair"])
            if pair is not None:
                inc[r, c] = 100 * (pair[1] - pair[0]) / fam["n_test"]
                pvals[r, c] = mcnemar_exact(*pair)
    tested = ~np.isnan(inc)
    stars = np.zeros_like(tested)
    if tested.any():
        stars[tested] = np.array(holm(list(pvals[tested]))) < alpha

    # Fixed grid (not constrained layout) so the gaps between panels are equal.
    fig = plt.figure(figsize=data.get("figsize", (style.TEXTWIDTH, 3.1)))
    m = {"left": 0.25, "right": 0.99, "top": 0.9, "bottom": 0.27, **data.get("margins", {})}
    widths = [max(base.shape[1], data.get("base_width", 1.8)), len(paths), len(increments)]
    gs = fig.add_gridspec(1, 3, width_ratios=widths,
                          wspace=data.get("panel_gap", 0.1), **m)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))
    # Optional presentation keys: annot_size, tick_size, title_size, row_counts (default true).
    sizes = dict(annot_size=data.get("annot_size"), tick_size=data.get("tick_size"))
    rows = [f"{f.get('short', f['name'])} ({f['n_test']})" if data.get("row_counts", True)
            else f.get("short", f["name"]) for f in fams]
    style.annotated_heatmap(ax_a, base, style.SEQUENTIAL, mpl.colors.Normalize(0, 100),
                            ["All", "Finished"] if base.shape[1] == 2 else [""], rows, **sizes)
    finite = gain[np.isfinite(gain)]
    style.annotated_heatmap(ax_b, gain, style.SEQUENTIAL,
                            mpl.colors.Normalize(0, finite.max() if finite.size else 1),
                            [p["label"] for p in paths], fmt="{:+.0f}", **sizes)
    bound = max(2.0, float(np.ceil(np.nanmax(np.abs(inc))))) if tested.any() else 2.0
    norm_c = mpl.colors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    style.annotated_heatmap(ax_c, inc, style.DIVERGING, norm_c,
                            [s["label"] for s in increments], fmt="{:+.1f}", stars=stars, **sizes)
    if data.get("colorbar", False):
        # Diverging scale of panel (c), just right of it.
        box = ax_c.get_position()
        cax = fig.add_axes((box.x1 + 0.012, box.y0, 0.014, box.height))
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_c, cmap=style.DIVERGING), cax=cax,
                          ticks=[-bound, 0, bound])
        cb.ax.set_yticklabels([f"{-bound:+.0f}", "0", f"{bound:+.0f}"])
        cb.ax.tick_params(labelsize=data.get("tick_size", style.FS_TICK), colors="black", length=2)
        # matplotlib stub declares the colorbar outline as Spines, not Spine.
        cb.outline.set_visible(False)  # ty: ignore[call-non-callable]
    for ax, title in zip((ax_a, ax_b, ax_c),
                         data.get("titles", ("(a) Base (%)", "(b) Gain over base (pp)",
                                             "(c) Stage increment (pp)")), strict=False):
        ax.set_title(title, fontsize=data.get("title_size", style.FS_TICK), loc="left", pad=4)
    return fig
