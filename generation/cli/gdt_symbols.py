"""GD&T characteristic symbols drawn as vector geometry.

The renderer's font (DejaVu Sans, matplotlib's default and the only font
guaranteed present) does **not** contain most ASME Y14.5 characteristic
glyphs. Measured against the installed font's cmap:

    present: U+22A5 perpendicularity, U+23E5 flatness, U+2225 parallelism,
             U+2220 angularity, U+2300 diameter, U+25CE concentricity
    MISSING: U+2316 position, U+232D cylindricity, U+232F symmetry,
             U+2313 profile-of-surface, U+2312 profile-of-line,
             U+23E4 straightness, U+2330 total runout,
             U+24C2/U+24C1/U+24C5 circled M / L / P modifiers,
             U+2334 counterbore, U+2335 countersink

A missing glyph renders as a tofu box, and position is the single most common
characteristic on a real print, so typesetting the frames was not an option.
Each symbol is therefore returned as a list of polylines and circles in a
1x1 box, which the caller scales and translates into the frame's first
compartment, which keeps the symbols legible at the ~2.8 mm text heights used
on the sheet.

Coordinates are normalised to a unit cell with (0, 0) at the bottom-left. The
caller supplies the cell size, so a symbol always fills the same proportion of
the frame regardless of sheet size.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

Poly = List[Tuple[float, float]]

# A circle is (centre_x, centre_y, radius); everything else is a polyline.
Circle = Tuple[float, float, float]


def _arc(cx: float, cy: float, r: float, a0: float, a1: float,
         n: int = 24) -> Poly:
    return [(cx + math.cos(math.radians(a)) * r,
             cy + math.sin(math.radians(a)) * r)
            for a in [a0 + (a1 - a0) * i / n for i in range(n + 1)]]


def _symbols() -> Dict[str, Tuple[List[Poly], List[Circle]]]:
    """Every characteristic symbol, as (polylines, circles)."""
    s: Dict[str, Tuple[List[Poly], List[Circle]]] = {}

    # position: a cross inside a circle
    s["position"] = ([[(0.5, 0.05), (0.5, 0.95)], [(0.05, 0.5), (0.95, 0.5)]],
                     [(0.5, 0.5, 0.32)])

    # concentricity: two concentric circles
    s["concentricity"] = ([], [(0.5, 0.5, 0.34), (0.5, 0.5, 0.16)])

    # symmetry: three horizontal bars between two verticals
    s["symmetry"] = ([[(0.5, 0.05), (0.5, 0.95)],
                      [(0.18, 0.72), (0.82, 0.72)],
                      [(0.18, 0.28), (0.82, 0.28)]], [])

    # parallelism: two parallel slashes
    s["parallelism"] = ([[(0.28, 0.1), (0.62, 0.9)],
                         [(0.58, 0.1), (0.92, 0.9)]], [])

    # perpendicularity: an upright T rotated -- vertical on a baseline
    s["perpendicularity"] = ([[(0.5, 0.9), (0.5, 0.15)],
                              [(0.12, 0.15), (0.88, 0.15)]], [])

    # angularity: an acute angle opening right
    s["angularity"] = ([[(0.9, 0.15), (0.12, 0.15), (0.85, 0.85)]], [])

    # flatness: a parallelogram
    s["flatness"] = ([[(0.28, 0.2), (0.95, 0.2), (0.72, 0.8), (0.05, 0.8),
                       (0.28, 0.2)]], [])

    # straightness: a single horizontal line
    s["straightness"] = ([[(0.06, 0.5), (0.94, 0.5)]], [])

    # circularity / roundness: one circle
    s["circularity"] = ([], [(0.5, 0.5, 0.36)])

    # cylindricity: a circle between two slashes
    s["cylindricity"] = ([[(0.1, 0.12), (0.32, 0.88)],
                          [(0.68, 0.12), (0.9, 0.88)]],
                         [(0.5, 0.5, 0.26)])

    # profile of a line: an upward arc
    s["profile_line"] = ([_arc(0.5, 0.18, 0.42, 20, 160)], [])

    # profile of a surface: the same arc closed by a baseline
    s["profile_surface"] = ([_arc(0.5, 0.18, 0.42, 20, 160),
                             [(0.5 + math.cos(math.radians(160)) * 0.42, 0.18),
                              (0.5 + math.cos(math.radians(20)) * 0.42, 0.18)]],
                            [])

    # circular runout: a single arrow leaning right
    s["circular_runout"] = ([[(0.3, 0.1), (0.72, 0.9)],
                             [(0.72, 0.9), (0.52, 0.84)],
                             [(0.72, 0.9), (0.72, 0.66)]], [])

    # total runout: two such arrows
    s["total_runout"] = ([[(0.16, 0.1), (0.58, 0.9)],
                          [(0.58, 0.9), (0.38, 0.84)],
                          [(0.58, 0.9), (0.58, 0.66)],
                          [(0.5, 0.1), (0.92, 0.9)],
                          [(0.92, 0.9), (0.72, 0.84)],
                          [(0.92, 0.9), (0.92, 0.66)]], [])

    return s


SYMBOLS = _symbols()


def symbol(name: str, x: float, y: float, size: float):
    """Place a characteristic symbol with its lower-left corner at (x, y).

    Returns ``(polylines, circles)`` in sheet millimetres. An unknown name
    yields empty lists rather than raising, so a characteristic this module
    does not draw degrades to a text-only frame instead of losing the frame.
    """
    polys, circles = SYMBOLS.get(name, ([], []))
    out_p = [[(x + px * size, y + py * size) for px, py in poly]
             for poly in polys]
    out_c = [(x + cx * size, y + cy * size, r * size)
             for cx, cy, r in circles]
    return out_p, out_c

