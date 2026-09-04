"""Render a PSheet to PNG (matplotlib).

The PNG is the dataset image: the COCO labels are measured against it,
so it is the only output format the pipeline writes.
"""
from __future__ import annotations

import math

from .sheet import PArrow, PCircle, PFace, PLine, PPoly, PSheet, PText

LAYER_STYLE = {
    # Black ink throughout; weight and dash pattern carry the distinction
    # between construction, hidden and visible geometry.
    "VISIBLE": ("#000000", 0.5, None),
    "HIDDEN":  ("#000000", 0.3, "3,1.5"),
    "CENTER":  ("#000000", 0.25, "6,1.5,1,1.5"),
    "DIM":     ("#000000", 0.25, None),
    "EXT":     ("#000000", 0.2, None),
    "LEADER":  ("#000000", 0.25, None),
    "TEXT":    ("#000000", 0.25, None),
    "FRAME":   ("#000000", 0.4, None),
    "TABLE":   ("#000000", 0.25, None),
    "FEATURE": ("#000000", 0.22, "2,1.5"),
    "SHADE":   ("#000000", 0.0, None),
    "ISOEDGE": ("#000000", 0.22, None),
    # Section hatching: thin, so it reads as fill rather than as edges.
    "HATCH":   ("#000000", 0.18, None),
    # ISO 128 type K: long dash, two short dashes. Alternate positions,
    # adjacent parts and -- here -- the stock the part is machined from.
    "PHANTOM": ("#000000", 0.25, "8,1.5,1,1.5,1,1.5"),
}


def _arrow_pts(tip, tail, size):
    dx, dy = tail[0] - tip[0], tail[1] - tip[1]
    L = math.hypot(dx, dy) or 1
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    w = size * 0.16
    b = (tip[0] + ux * size, tip[1] + uy * size)
    return [tip, (b[0] + px * w, b[1] + py * w), (b[0] - px * w, b[1] - py * w)]


# --------------------------------------------------------------------------- #
#: Thinnest line matplotlib will be asked to draw, in points, measured on the
#: THINNEST layer of the sheet. Below roughly this a stroke is narrower than
#: one device pixel and anti-aliases to a pale grey instead of a line: at
#: stroke_scale 0.71 the EXT layer came out at 0.284 pt and 44% of the ink
#: pixels were faint rather than solid.
#:
#: When a sheet falls under the floor the WHOLE set of layer weights is
#: multiplied up (see _legibility_scale) rather than each layer being clamped
#: individually. A per-layer clamp was tried and REVERTED: at any floor high
#: enough to help, it pulled nearly every layer to the same width, so a light
#: drawing lost the visible / hidden / dimension hierarchy altogether -- and
#: because the floor is absolute it did so at normal scale too.
#:
#: 0.40 chosen by measurement over 3 parts x 3 resolutions on the lightest
#: sheet, counting faint (anti-aliased) against solid ink pixels, and
#: weighed against how much of the stroke variation survives:
#:
#:     floor        0.00  0.30  0.40  0.50  0.60
#:     mean faint   0.74  0.60  0.36  0.31  0.22
#:     sheets lifted   0     3     9    16    20   (of 23)
#:     distinct wts   23    22    15    10     4
#:
#: 0.60 renders best but flattens the variation to four distinct weights,
#: which defeats the point of varying it. 0.40 halves the faint-ink ratio
#: while leaving 15 distinct weights and touching under half the corpus.
_MIN_STROKE_PT = 0.40


def _legibility_scale(stroke_scale: float) -> float:
    """Lift a whole light drawing until its thinnest layer is renderable.

    A per-layer clamp was tried first and REVERTED: at a 0.80 pt floor it
    pulled almost every layer to the same width, so a light sheet lost the
    visible / hidden / dimension weight hierarchy entirely -- and because the
    floor is absolute it did that at normal scale too. Multiplying the whole
    set by one factor keeps every ratio intact while still getting the
    thinnest line onto a pixel.
    """
    thinnest = min(w for _c, w, _d in LAYER_STYLE.values() if w > 0)
    drawn_pt = thinnest * stroke_scale * 2
    if drawn_pt >= _MIN_STROKE_PT:
        return stroke_scale
    return stroke_scale * (_MIN_STROKE_PT / drawn_pt)


def to_png(sh: PSheet, path: str, dpi: int = 200):
    """Draw the sheet with matplotlib and save it as the dataset PNG.

    This is the primary deliverable: the COCO boxes are measured
    against these pixels, so a defect here is a defect in the labels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPoly

    _mpl_sw = _legibility_scale(getattr(sh, "stroke_scale", 1.0))

    MM = 1 / 25.4
    fig = plt.figure(figsize=(sh.w * MM, sh.h * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, sh.w); ax.set_ylim(0, sh.h)
    ax.axis("off")

    # Dash patterns are the third field of LAYER_STYLE ("3,1.5", "8,1.5,1,1.5").
    # matplotlib wants an offset + tuple rather than a string, so derive them
    # here — one source of truth instead of a second copy that can drift.
    DASH = {layer: (0, tuple(float(x) for x in dash.split(",")))
            for layer, (_c, _w, dash) in LAYER_STYLE.items()
            if dash is not None}
    for p in sh.prims:
        layer = getattr(p, "layer", "VISIBLE")
        c, lw, _ = LAYER_STYLE.get(layer, ("#000000", 0.3, None))
        # One factor for the whole sheet, lifted if need be so the thinnest
        # layer still lands on a pixel. See _legibility_scale.
        lw *= _mpl_sw
        ls = DASH.get(layer, "solid")
        if isinstance(p, PLine):
            ax.plot([p.a[0], p.b[0]], [p.a[1], p.b[1]], color=c, lw=lw * 2, ls=ls,
                    solid_capstyle="round")
        elif isinstance(p, PPoly):
            xs = [q[0] for q in p.pts]; ys = [q[1] for q in p.pts]
            if p.close:
                xs.append(xs[0]); ys.append(ys[0])
            ax.plot(xs, ys, color=c, lw=lw * 2, ls=ls, solid_capstyle="round")
        elif isinstance(p, PCircle):
            ax.add_patch(plt.Circle(p.c, p.r, fill=False, ec=c, lw=lw * 2))
        elif isinstance(p, PFace):
            g = 0.55 + 0.45 * p.shade
            ax.add_patch(MPoly(p.pts, closed=True, fc=(g, g, g), ec="none",
                               zorder=0.5))
        elif isinstance(p, PArrow):
            kind = getattr(p, "kind", "filled")
            pts = _arrow_pts(p.tip, p.tail, p.size)
            if kind == "open":
                # a vee: the two barbs drawn as strokes, no fill
                ax.plot([pts[1][0], pts[0][0], pts[2][0]],
                        [pts[1][1], pts[0][1], pts[2][1]],
                        color=c, lw=lw * 2, solid_capstyle="round")
            elif kind == "tick":
                # the 45-degree slash used on architectural and some
                # sheet-metal prints, drawn across the dimension line
                dx = p.tail[0] - p.tip[0]
                dy = p.tail[1] - p.tip[1]
                L = math.hypot(dx, dy) or 1.0
                ux, uy = dx / L, dy / L
                # rotate the line direction by 45 degrees
                k = p.size * 0.6
                rx = (ux - uy) * 0.7071 * k
                ry = (uy + ux) * 0.7071 * k
                ax.plot([p.tip[0] - rx, p.tip[0] + rx],
                        [p.tip[1] - ry, p.tip[1] + ry],
                        color=c, lw=lw * 2.4, solid_capstyle="butt")
            elif kind == "dot":
                ax.add_patch(plt.Circle(p.tip, p.size * 0.18, fc=c, ec="none"))
            else:
                ax.add_patch(MPoly(pts, closed=True, fc=c, ec="none"))
        elif isinstance(p, PText):
            va = {"bottom": "bottom", "center": "center", "top": "top"}[p.va]
            ax.text(p.p[0], p.p[1], p.s, fontsize=p.h * 2.6, color=c,
                    family=getattr(sh, "font_family", "DejaVu Sans"),
                    fontweight=("bold" if getattr(p, "bold", False)
                                else "normal"),
                    ha=p.ha, va=va, rotation=p.rot, rotation_mode="anchor",
                    linespacing=1.2)
    # An opaque white background is stated explicitly rather than inherited:
    # a transparent PNG renders black-on-black in many viewers, and this is
    # training data where the background must be deterministic.
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    # matplotlib writes RGBA. The alpha is fully opaque here, so the channel
    # is 25% of the file carrying no information. Flattened onto white so the
    # result is byte-identical to what a viewer would composite anyway.
    try:
        from PIL import Image
        im = Image.open(path)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel("A"))
            bg.save(path, optimize=True)
    except Exception:
        pass                # Pillow missing: the RGBA file is still correct
    return path
