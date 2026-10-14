"""3-D waterfall of accuracy by training stage and reaction family, one axes.

x: reaction family (single-letter code),
y (depth): training stage, receding from Base in front through S1, S2, S3 to
the S1→S3 path that skips S2, z: test accuracy (%).
An optional reference model (e.g. Claude Opus) is drawn as a front row of
open wireframe bars.

Input (``stage_ablation.json``) — same file as the heatmap figure, plus the
optional ``stages`` / ``codes`` / ``reference`` keys::

    {"stages": [{"key": "base", "label": "Base"}, {"key": "s1", "label": "S1"},
                ...],                                  # optional, default below
     "codes": {"acylation": "A", ...},                 # optional, default below
     "reference": {"key": "opus", "label": "Opus"},    # optional
     "reference_position": "front" | "back",           # optional, default front
     "reference_style": "wire" | "solid",              # optional, default wire
     "overall": {"label": "All", "n_test": 5000,        # optional aggregate column, drawn
                 "correct": {"base": 1139, ...}},       # first and set apart by overall_gap
     "families": [
        {"name": "acylation", "short": "Acylation", "n_test": 1107,
         "correct": {"base": 366, "s1": 971, "s2": 976, "s3": 976,
                     "skip_s2": 968, "opus": 700}},     # "opus" -> reference bar
        ...]}

A stage whose key is missing for a family is left empty.
"""
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import style
from .schema import require, require_list

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.axis3d import Axis as Axis3D

DEFAULT_STAGES = [{"key": "base", "label": "Base"},
                  {"key": "s1", "label": "S1"},
                  {"key": "s2", "label": "S2"},
                  {"key": "s3", "label": "S3"},
                  {"key": "skip_s2", "label": "no S2"}]
DEFAULT_CODES = {"heteroatom alkylation/arylation": "X",
                 "protection chemistry": "P",
                 "acylation": "A",
                 "C-C bond formation": "C",
                 "redox": "R",
                 "FG changes": "F",
                 "heterocycle formation": "H"}


def validate(data):
    for i, fam in enumerate(require_list(data, "families", "stage_ablation")):
        require(fam, ["name", "n_test", "correct"], f"families[{i}]")


def plot(data):
    validate(data)
    if data.get("layout") == "class_depth":
        return plot_class_depth(data)
    fams = list(data["families"])
    overall = data.get("overall")
    # x positions: the optional aggregate column first, then a gap, then the families.
    xs = [float(i) for i in range(len(fams))]
    if overall:
        gap = data.get("overall_gap", 0.6)
        fams = [{"name": "__overall__", "n_test": overall["n_test"], "correct": overall["correct"]}] + fams
        xs = [0.0] + [1 + gap + x for x in xs]
    stages = data.get("stages", DEFAULT_STAGES)
    codes = {**DEFAULT_CODES, **data.get("codes", {})}
    ref = data.get("reference")
    fs = data.get("tick_size", style.FS_TICK)          # stage / z tick labels
    fl = data.get("label_size", style.FS_LABEL)        # axis titles and family codes

    # House stage ramp for the lineage (Base, S1, S2, ...). The S1→S3 ablation
    # ends at S3, so it gets a lightened S3 colour rather than the next ramp
    # step (which would read as a stage after S3). Reference models take the
    # reference palette. Any stage/reference may override with "color".
    main = [s["key"] for s in stages if s["key"] != "skip_s2"]
    shade: dict[str, Any] = {k: style.STAGE_COLORS[i % len(style.STAGE_COLORS)] for i, k in enumerate(main)}
    s3 = np.array(mpl.colors.to_rgb(shade.get("s3", style.GREEN)))
    shade["skip_s2"] = tuple(s3 + (1 - s3) * 0.5)
    shade.update({s["key"]: s["color"] for s in stages if "color" in s})
    c_ref = (ref or {}).get("color", style.REFERENCE_COLORS[0])

    fig = plt.figure(figsize=data.get("waterfall_figsize", (style.TEXTWIDTH, 3.0)))
    ax = cast("Axes3D", fig.add_axes(data.get("axes_rect", [-0.26, -0.17, 1.36, 1.34]),
                                     projection="3d"))
    # mplot3d's automatic depth sort gets overlapping bars wrong (back bars
    # painted over front ones); draw strictly back-to-front instead.
    ax.computed_zorder = False
    zo = 0
    # Bar footprint in data units. "square_bars" sizes the depth so the footprint is square on
    # screen (width * x_stretch == depth * y_stretch): columns rather than flat slabs.
    dx = data.get("bar_width", 0.62)
    dy = data.get("bar_depth", 0.42)
    if data.get("square_bars"):
        dy = dx * data.get("x_stretch", 1.5) / data.get("y_stretch", 0.9)
    # The reference row is open wireframe, so it goes in front without hiding
    # the stage rows behind it.
    ref_row = [{"key": ref["key"], "label": ref["label"]}] if ref else []
    ref_back = data.get("reference_position", "front") == "back"
    rows = list(stages) + ref_row if ref_back else ref_row + list(stages)
    ref_j = (len(rows) - 1 if ref_back else 0) if ref else None
    solid_ref = data.get("reference_style", "wire") == "solid"

    # Stage rows recede from the viewer (Base in front); draw back to front so
    # mplot3d's painter ordering stays right.
    for j in reversed(range(len(rows))):
        st = rows[j]
        is_ref = j == ref_j
        for i, fam in zip(xs, fams, strict=True):
            c = fam["correct"].get(st["key"])
            if c is None:
                continue
            z = 100 * c / fam["n_test"]
            if is_ref and solid_ref:
                zo += 1
                ax.bar3d(i - dx / 2, j - dy / 2, 0, dx, dy, z, color=c_ref,
                         edgecolor=(1, 1, 1, 0.35), linewidth=0.25, shade=True, zorder=zo)
            elif is_ref:
                zo += 1
                ax.bar3d(i - dx / 2, j - dy / 2, 0, dx, dy, z, color=(0, 0, 0, 0),
                         edgecolor=c_ref, linewidth=0.8, zorder=zo)
            else:
                # footprint on the floor, visible even where a front bar is short
                ax.plot([i - dx / 2, i + dx / 2, i + dx / 2, i - dx / 2, i - dx / 2],
                        [j - dy / 2, j - dy / 2, j + dy / 2, j + dy / 2, j - dy / 2],
                        0, color="0.6", lw=0.3, zorder=zo)
                zo += 1
                ax.bar3d(i - dx / 2, j - dy / 2, 0, dx, dy, z, color=shade[st["key"]],
                         edgecolor=(1, 1, 1, 0.35), linewidth=0.25, shade=True,
                         zorder=zo)

    codes["__overall__"] = (overall or {}).get("label", "All")
    ax.set_xticks(xs, [codes.get(f["name"], f["name"][0].upper()) for f in fams],
                  fontsize=fl)
    if data.get("family_rotation"):
        for t in ax.get_xticklabels():
            t.set_rotation(data["family_rotation"])
            t.set_horizontalalignment("right")
            t.set_rotation_mode("anchor")
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows], fontsize=fs)
    for t in ax.get_yticklabels():
        t.set_horizontalalignment("left")
    ax.set_zticks([0, 25, 50, 75, 100])  # ty: ignore[call-non-callable]  (mplot3d axis-method wrapper)
    ax.tick_params(axis="z", labelsize=fs, pad=1)
    ax.tick_params(axis="x", pad=-4)
    ax.tick_params(axis="y", pad=-3)
    ax.set_zlim(0, 100)
    # Extra room past the last family/row: a bar flush with the box edge loses its side face.
    pad = data.get("edge_pad", 0.25)
    ax.set_xlim(-0.5, xs[-1] + 0.5 + pad)
    ax.set_ylim(-0.5, len(rows) - 0.5 + pad)
    ax.set_zlabel("Accuracy (%)", fontsize=fl, labelpad=data.get("zlabel_pad", -3))
    ax.set_xlabel("Reaction family", fontsize=fl, labelpad=data.get("xlabel_pad", -6))
    ax.set_ylabel("Training stage", fontsize=fl, labelpad=data.get("ylabel_pad", -2))
    ax.view_init(elev=data.get("elev", 26), azim=data.get("azim", -62))
    # Box stretched along x and axes overhanging the figure so the plot fills
    # the full text width with little white margin.
    # y_stretch: depth per row; smaller keeps rows of one family visually stacked at wide azimuths.
    ax.set_box_aspect(((xs[-1] + 1) * data.get("x_stretch", 1.5), len(rows) * data.get("y_stretch", 0.9),
                       data.get("z_height", 3.6)), zoom=data.get("zoom", 1.12))
    # Orthographic view + a shaded floor so every bar visibly stands on z=0
    # (with perspective and a blank floor, back rows look like they float).
    ax.set_proj_type("ortho")
    for axis in cast("list[Axis3D]", [ax.xaxis, ax.yaxis, ax.zaxis]):
        axis.set_pane_color((0.93, 0.93, 0.93, 1) if axis is ax.zaxis else (1, 1, 1, 0))
        cast(dict, axis._axinfo["grid"]).update(color=style.GRID, linewidth=0.5)
        axis.line.set_linewidth(0.5)

    # Family-code key is meant for the caption; set "show_key": true to draw it.
    if data.get("show_key", False):
        key = "\n".join(f"{codes.get(f['name'], '?')}   {f.get('short', f['name'])}"
                        for f in fams if f["name"] != "__overall__")
        fig.text(0.01, 0.97, key, va="top", ha="left", fontsize=style.FS_LEG,
                 color=style.BLACK, linespacing=1.3)
    return fig


def plot_class_depth(data):
    """Classic waterfall: one slice per reaction class (depth), stages along x.

    Each slice is a bar chart of the training stages for that class, with the
    reference model (e.g. Opus) as a horizontal line across the slice at its
    accuracy on the class. ``overall`` (if given) is the front slice.
    Keys: ``stages`` (default Base, S1, S2, S3, S1→S3; extra stages such as a
    teacher are dropped unless ``keep_extra_stages``), ``reference``, ``overall``,
    ``slice_labels`` ("short" | "code"), ``elev``/``azim``, ``x_stretch``/
    ``y_stretch``/``z_height``/``zoom``, ``axes_rect``, ``waterfall_figsize``,
    ``xlabel_pad``/``ylabel_pad``, ``legend_anchor``.
    """
    from matplotlib.lines import Line2D
    base_keys = {s["key"] for s in DEFAULT_STAGES}
    stages = [s for s in data.get("stages", DEFAULT_STAGES)
              if s["key"] in base_keys or data.get("keep_extra_stages")]
    ref = data.get("reference")
    overall = data.get("overall")
    slices: list[dict[str, Any]] = ([{"name": "__overall__", "short": overall.get("label", "All reactions"),
                "n_test": overall["n_test"], "correct": overall["correct"]}] if overall else [])
    slices += list(data["families"])
    codes = {**DEFAULT_CODES, **data.get("codes", {}), "__overall__": "All"}
    use_codes = data.get("slice_labels", "short") == "code"

    main = [s["key"] for s in stages if s["key"] != "skip_s2"]
    shade: dict[str, Any] = {k: style.STAGE_COLORS[i % len(style.STAGE_COLORS)] for i, k in enumerate(main)}
    s3 = np.array(mpl.colors.to_rgb(shade.get("s3", style.GREEN)))
    shade["skip_s2"] = tuple(s3 + (1 - s3) * 0.5)
    shade.update({s["key"]: s["color"] for s in stages if "color" in s})
    c_ref = (ref or {}).get("color", style.REFERENCE_COLORS[1])

    fig = plt.figure(figsize=data.get("waterfall_figsize", (style.TEXTWIDTH, 3.4)))
    ax = cast("Axes3D", fig.add_axes(data.get("axes_rect", [-0.2, -0.08, 1.32, 1.28]), projection="3d"))
    ax.computed_zorder = False
    zo = 0
    dx, dy = 0.7, 0.45
    nx = len(stages)
    # Slices recede from the viewer (first slice in front); draw back to front.
    for j in reversed(range(len(slices))):
        fam = slices[j]
        for i, st in enumerate(stages):
            c = fam["correct"].get(st["key"])
            if c is None:
                continue
            zo += 1
            ax.bar3d(i - dx / 2, j - dy / 2, 0, dx, dy, 100 * c / fam["n_test"],
                     color=shade[st["key"]], edgecolor=(1, 1, 1, 0.35), linewidth=0.25,
                     shade=True, zorder=zo)
        if ref and fam["correct"].get(ref["key"]) is not None:
            zr = 100 * fam["correct"][ref["key"]] / fam["n_test"]
            zo += 1
            # Across the slice, over its own bars; slices in front still occlude it.
            ax.plot([-0.5, nx - 0.5], [j, j], [zr, zr], color=c_ref, lw=1.8, zorder=zo,
                    solid_capstyle="butt")

    ax.set_xticks(range(nx), [s["label"] for s in stages], fontsize=style.FS_TICK)
    ax.set_yticks(range(len(slices)), [codes.get(f["name"], "?") if use_codes
                                       else f.get("short", f["name"]) for f in slices],
                  fontsize=style.FS_TICK)
    for t in ax.get_yticklabels():
        t.set_horizontalalignment("left")
    ax.set_zticks([0, 25, 50, 75, 100])  # ty: ignore[call-non-callable]  (mplot3d axis-method wrapper)
    ax.tick_params(axis="z", labelsize=style.FS_TICK, pad=1)
    ax.tick_params(axis="x", pad=-3)
    ax.tick_params(axis="y", pad=-2)
    ax.set_zlim(0, 100)
    pad = data.get("edge_pad", 0.25)
    ax.set_xlim(-0.5, nx - 0.5 + pad)
    ax.set_ylim(-0.5, len(slices) - 0.5 + pad)
    ax.set_zlabel("Accuracy (%)", fontsize=style.FS_LABEL, labelpad=-3)
    ax.set_xlabel("Training stage", fontsize=style.FS_LABEL, labelpad=data.get("xlabel_pad", 0))
    ax.set_ylabel("Reaction class", fontsize=style.FS_LABEL, labelpad=data.get("ylabel_pad", 10))
    ax.view_init(elev=data.get("elev", 24), azim=data.get("azim", -58))
    ax.set_box_aspect((nx * data.get("x_stretch", 1.0), len(slices) * data.get("y_stretch", 0.8),
                       data.get("z_height", 3.2)), zoom=data.get("zoom", 1.0))
    ax.set_proj_type("ortho")
    for axis in cast("list[Axis3D]", [ax.xaxis, ax.yaxis, ax.zaxis]):
        axis.set_pane_color((0.93, 0.93, 0.93, 1) if axis is ax.zaxis else (1, 1, 1, 0))
        cast(dict, axis._axinfo["grid"]).update(color=style.GRID, linewidth=0.5)
        axis.line.set_linewidth(0.5)
    if ref:
        fig.legend(handles=[Line2D([], [], color=c_ref, lw=1.8, label=ref["label"])],
                   loc="upper left", bbox_to_anchor=data.get("legend_anchor", (0.02, 0.98)),
                   fontsize=style.FS_LEG, frameon=False)
    return fig


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    args = ap.parse_args()
    style.apply()
    with open(args.data) as f:
        plot(json.load(f)).savefig(args.out)
