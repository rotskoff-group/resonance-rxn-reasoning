"""House style shared by every paper figure.

Figures are drawn at the width they occupy in the PDF (TEXTWIDTH = NeurIPS
single-column \\linewidth), so \\includegraphics never rescales them and type
size is identical across figures.
"""
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TEXTWIDTH = 5.5
FS_LABEL = 9
FS_TICK = 8
FS_LEG = 7.5
FS_ANNOT = 7.5
FS_TITLE = 9
MARKER_SIZE = 5

# Named type/width presets; `use(name)` sets the module constants above.
PRESETS = {
    "default": dict(TEXTWIDTH=5.5, FS_LABEL=9, FS_TICK=8, FS_LEG=7.5, FS_ANNOT=7.5,
                    FS_TITLE=9, MARKER_SIZE=5),
    # Main-text metareasoning figures: drawn wide with large type.
    "main_text": dict(TEXTWIDTH=7.5, FS_LABEL=16, FS_TICK=14, FS_LEG=12.5, FS_ANNOT=13,
                      FS_TITLE=16, MARKER_SIZE=8),
}

BLACK = "#110d1b"
MUTED = "#9dadc4"
GREEN = "#0f6852"
DARKTAN = "#af9e73"
GRID = "#e7e9ec"
ZERO_LINE = "#9baec8"

# Stage ramp for a training lineage (base, S1, S2, S3, ...), then reference models.
STAGE_COLORS = ["#4dcbff", "#008eca", "#00538a", GREEN, "#001f4d"]
REFERENCE_COLORS = ["#b779a8", "#df7954", "#756bb1", DARKTAN, "#8a8981"]

# Reaction-family colours; a family keeps its colour in every figure.
FAMILY_COLORS = {
    "overall": MUTED,
    "heteroatom alkylation/arylation": "#104281",
    "protection chemistry": "#3987e5",
    "acylation": "#86b6ef",
    "C-C bond formation": "#f0a3a2",
    "redox": "#8a8981",
    "FG changes": DARKTAN,
    "heterocycle formation": "#e34948",
}
# Fallback categorical order for series without an assigned colour.
CATEGORICAL = ["#10735d", "#ad9b6b", "#803da0", "#00538a", "#d62728", "#5b7fa3"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

# Diverging (negative = purple, zero = sand, positive = teal) and sequential maps.
DIVERGING = mpl.colors.LinearSegmentedColormap.from_list(
    "purple_sand_teal", ["#714478", "#c3acc8", "#f2eadc", "#92c6bd", "#006b60"])
SEQUENTIAL = mpl.colors.LinearSegmentedColormap.from_list(
    "sand_teal", ["#f2eadc", "#92c6bd", "#006b60"])
TAN_GREEN = mpl.colors.LinearSegmentedColormap.from_list(
    "tan_white_green", [DARKTAN, "#ffffff", GREEN])
for _cmap in (DIVERGING, SEQUENTIAL, TAN_GREEN):
    _cmap.set_bad("#ffffff")


def use(name="default"):
    if name not in PRESETS:
        raise ValueError(f"unknown style {name!r}; choose from {sorted(PRESETS)}")
    globals().update(PRESETS[name])
    apply()


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.7,
        "axes.edgecolor": BLACK,
        "axes.labelcolor": BLACK,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": False,
        "axes.axisbelow": True,
        "xtick.color": "#25282a", "ytick.color": "#25282a",
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "font.size": FS_TICK,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEG,
        "legend.frameon": False,
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, out_dir, stem, formats=("pdf", "png")):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def open_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def series_style(i, item):
    """Colour and marker for the i-th series, honouring explicit overrides."""
    return (item.get("color") or CATEGORICAL[i % len(CATEGORICAL)],
            item.get("marker") or MARKERS[i % len(MARKERS)])


def annotated_heatmap(ax, values, cmap, norm, xlabels, ylabels=None, fmt="{:.0f}",
                      stars=None, rotate=40, annot_size=None, tick_size=None):
    """Heatmap with a value in every cell; NaN cells render as a grey dash.

    annot_size / tick_size override the cell-text and tick-label font sizes.
    """
    annot_size = FS_ANNOT - 1.5 if annot_size is None else annot_size
    tick_size = FS_TICK - 0.5 if tick_size is None else tick_size
    ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto",
              interpolation="nearest")
    diverging = isinstance(norm, mpl.colors.TwoSlopeNorm)
    for (r, c), v in np.ndenumerate(values):
        if np.isnan(v):
            ax.text(c, r, "–", ha="center", va="center", fontsize=annot_size, color="#9a9a9a")
            continue
        dark = abs(norm(v) - 0.5) > 0.35 if diverging else norm(v) > 0.6
        text = fmt.format(v) + ("*" if stars is not None and stars[r, c] else "")
        ax.text(c, r, text, ha="center", va="center", fontsize=annot_size,
                color="white" if dark else BLACK)
    ax.set_xticks(np.arange(values.shape[1]), xlabels, rotation=rotate,
                  ha="right" if rotate else "center", rotation_mode="anchor",
                  fontsize=tick_size, color="black")
    ax.set_yticks(np.arange(values.shape[0]), ylabels or [], fontsize=tick_size, color="black")
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
