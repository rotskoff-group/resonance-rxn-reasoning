"""House style shared by every paper figure.

Figures are drawn at the width they occupy in the PDF (TEXTWIDTH = NeurIPS
single-column \\linewidth), so \\includegraphics never rescales them and type
size is identical across figures.
"""
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
GREEN = "#0f6852"
DARKTAN = "#af9e73"
GRID = "#e7e9ec"
ZERO_LINE = "#9baec8"

# Stage ramp for a training lineage (base, S1, S2, S3, ...), then reference models.
STAGE_COLORS = ["#4dcbff", "#008eca", "#00538a", GREEN, "#001f4d"]
REFERENCE_COLORS = ["#b779a8", "#df7954", "#756bb1", DARKTAN, "#8a8981"]

# Fallback categorical order for series without an assigned colour.
CATEGORICAL = ["#10735d", "#ad9b6b", "#803da0", "#00538a", "#d62728", "#5b7fa3"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

TAN_GREEN = mpl.colors.LinearSegmentedColormap.from_list(
    "tan_white_green", [DARKTAN, "#ffffff", GREEN])
TAN_GREEN.set_bad("#ffffff")


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
