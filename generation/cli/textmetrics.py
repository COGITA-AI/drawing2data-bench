"""True rendered extent of a text string, for tight detection boxes.

Why this exists
---------------
Everywhere else in AutoDraft a string's width is estimated as
``len(s) * height * Style.char_w``. That estimate is deliberately crude and
deliberately generous: it is used to RESERVE space during placement, where
over-estimating merely spreads labels out while under-estimating makes them
overlap. ``variation._measure_char_w`` even adds 4% headroom on purpose.

An exported detection box is a different job. It is a label: it must contain
the glyphs, and hold as little blank paper as possible -- in that order.

Getting this right needs the answer to one question: *where exactly does the
renderer put the glyphs for a PText?* Three separate things were got wrong by
reasoning about it from font tables instead of asking:

1. ``char_w`` is the MEAN width per character over a sample alphabet, so a
   string of narrow glyphs (``BODY 1: 0.89 X 0.67`` -- digits, dots, spaces,
   a colon) is much narrower than it predicts. ~2.96 mm on that string.
2. The exporter draws at ``fontsize = h * 2.6`` POINTS, an em of
   ``h * 2.6 * 25.4 / 72 = 0.9172 * h`` mm, while ``char_w`` is measured per
   em and the box maths assumed an em of exactly ``h``. Another ~2.90 mm.
3. ``ha``/``va`` do NOT align the ink. ``va="bottom"`` aligns the descent
   line, so a box anchored as if ``bottom`` meant the baseline sits a full
   descent (0.76 mm at h=3.5) too low -- which is what put the boxes visibly
   below their text. ``ha="center"``/``"right"`` align the ADVANCE width,
   which includes side bearings the ink does not fill.

Rather than model (3) from ``FT2Font`` ascender/descender -- which was tried
and left a 0.62 mm residual on ``va="top"``, because matplotlib's layout
metrics are not simply the font's -- this asks **matplotlib itself** for the
layout box via ``Text.get_window_extent``. Verified against rendered pixels
over 54 (string x ha x va) combinations: the layout box contains the ink with
a worst under-coverage of **0.06 mm**.

The exported box is therefore matplotlib's own layout box, intersected with
the string's ink extent so the small amount of line-height padding the layout
box carries (up to 0.61 mm) does not loosen the label.

This is used ONLY for exported boxes; placement keeps the conservative
estimate, because shrinking the reservation would let labels collide.

Everything is cached -- a probe figure costs milliseconds and a sheet has
dozens of strings, re-measured on every layout pass.
"""
from __future__ import annotations

from typing import Optional, Tuple

#: The sheet is drawn by matplotlib at ``fontsize = h * MPL_FONTSIZE_FACTOR``
#: points. Keep in step with ``export._to_mpl``; it is imported from here so
#: the two cannot drift.
MPL_FONTSIZE_FACTOR = 2.6

#: Millimetres of em per millimetre of nominal text height, as actually drawn
#: by the raster/vector exporter.
EM_PER_H = MPL_FONTSIZE_FACTOR * 25.4 / 72.0

#: Font size, in points, used by the measurement probe. Large enough that
#: rounding in the layout engine is negligible once divided back out.
_PROBE_PT = 100.0
_PROBE_DPI = 100.0

_ink_cache: dict = {}
_layout_cache: dict = {}
_vmetric_cache: dict = {}
_failed = False


def _font_ok(family: str, bold: bool) -> bool:
    """True when this family really resolves.

    matplotlib SILENTLY substitutes its default for a family it cannot load,
    which would hand back a confident measurement of the wrong typeface.
    """
    try:
        import logging
        from matplotlib.font_manager import FontProperties, findfont
        log = logging.getLogger("matplotlib.font_manager")
        prev = log.level
        log.setLevel(logging.ERROR)
        try:
            return bool(findfont(
                FontProperties(family=family,
                               weight=("bold" if bold else "normal")),
                fallback_to_default=False))
        finally:
            log.setLevel(prev)
    except Exception:
        return False


def ink_extent(s: str, family: str, bold: bool = False
               ) -> Optional[Tuple[float, float, float, float]]:
    """Ink bounding box of ``s`` at em = 1, relative to the pen origin.

    ``(x0, y0, x1, y1)`` in em units, y measured from the baseline (negative
    for descenders). None when it cannot be measured.
    """
    global _failed
    if _failed or not s:
        return None
    key = (s, family, bold)
    hit = _ink_cache.get(key)
    if hit is not None:
        return hit
    try:
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties
        if not _font_ok(family, bold):
            return None
        prop = FontProperties(family=family, size=1000,
                              weight=("bold" if bold else "normal"))
        e = TextPath((0, 0), s, prop=prop).get_extents()
        if e.width <= 0 or e.height <= 0:
            return None
        box = (e.x0 / 1000.0, e.y0 / 1000.0, e.x1 / 1000.0, e.y1 / 1000.0)
    except Exception:
        _failed = True
        return None
    _ink_cache[key] = box
    return box


def layout_box(s: str, family: str, ha: str, va: str, rot: float = 0.0,
               bold: bool = False) -> Optional[Tuple[float, float, float, float]]:
    """matplotlib's own layout box for this text, in em units.

    Relative to the anchor point, so it can be offset straight onto a PText's
    position. This is the authoritative answer to "where will the renderer
    put this string", including exactly what ``ha`` and ``va`` align -- which
    is not the ink and not the baseline.
    """
    global _failed
    if _failed or not s:
        return None
    key = (s, family, ha, va, round(float(rot), 3), bold)
    hit = _layout_cache.get(key)
    if hit is not None:
        return hit
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not _font_ok(family, bold):
            return None
        fig = plt.figure(figsize=(6, 4), dpi=_PROBE_DPI)
        try:
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, 600 * _PROBE_DPI / 100.0)
            ax.set_ylim(0, 400 * _PROBE_DPI / 100.0)
            ax.axis("off")
            r = fig.canvas.get_renderer()
            ax_x, ax_y = 300.0, 200.0
            t = ax.text(ax_x, ax_y, s, fontsize=_PROBE_PT, ha=ha, va=va,
                        family=family,
                        fontweight=("bold" if bold else "normal"),
                        rotation=rot, rotation_mode="anchor")
            bb = t.get_window_extent(renderer=r)
            em_px = _PROBE_PT * _PROBE_DPI / 72.0
            box = ((bb.x0 - ax_x) / em_px, (bb.y0 - ax_y) / em_px,
                   (bb.x1 - ax_x) / em_px, (bb.y1 - ax_y) / em_px)
        finally:
            plt.close(fig)
    except Exception:
        _failed = True
        return None
    _layout_cache[key] = box
    return box


def drawn_box(s: str, h: float, family: str, ha: str = "left",
              va: str = "bottom", rot: float = 0.0, bold: bool = False
              ) -> Optional[Tuple[float, float, float, float]]:
    """Ink box of ``s`` in millimetres, relative to the PText anchor.

    Combines matplotlib's layout box (which says where the renderer puts the
    string, honouring ha/va) with the string's ink extent (which says how
    much of that box the glyphs actually fill). The layout box carries a
    little line-height padding -- up to 0.61 mm measured -- so the ink extent
    is used to trim it where the two disagree, giving a box that both
    CONTAINS the glyphs and hugs them.

    Returns None if the font cannot be measured; the caller must then fall
    back to the nominal estimate.
    """
    lay = layout_box(s, family, ha, va, rot, bold)
    if lay is None:
        return None
    em = h * EM_PER_H
    lx0, ly0, lx1, ly1 = (v * em for v in lay)
    ink = ink_extent(s, family, bold)
    if ink is None or abs(round(float(rot)) % 360) not in (0, 360):
        # Unrotated ink offsets do not apply to a rotated run; the layout box
        # is already correct for it, just slightly loose.
        return (lx0, ly0, lx1, ly1)
    # Trim the layout box down to the ink.
    #
    # The layout box is the verified container but it spans the font's whole
    # line, so it is taller than the glyphs and a little wider. To trim it we
    # need the ink's position INSIDE it, and deriving that from the face's
    # ascender/descender fields does not work: matplotlib's laid-out line is
    # not the face's ascent-to-descent span (0.9282 em vs ~1.06 em measured
    # for DejaVu Sans), so one edge or the other came out ~0.5-0.6 mm wrong
    # depending on which metric was trusted.
    #
    # Instead the baseline is located empirically, once per (family, bold),
    # by laying out a probe string whose ink extent is known and seeing where
    # its layout box falls. Everything then follows from the ink extent,
    # which is measured in the same coordinate system.
    ix0, iy0, ix1, iy1 = ink
    base = _baseline_in_layout(family, bold)
    if base is None:
        return (lx0, ly0, lx1, ly1)
    # `base` is the baseline's height above the layout box bottom, in em.
    by = ly0 + base * em
    out = (lx0 + ix0 * em, by + iy0 * em, lx0 + ix1 * em, by + iy1 * em)
    # Never report ink outside the box matplotlib says it will use: if the
    # two disagree, the container wins.
    return (max(out[0], lx0), max(out[1], ly0),
            min(out[2], lx1), min(out[3], ly1))


#: Probe string for locating the baseline: has both a tall cap and a full
#: descender, so its ink spans the line and the fit is well conditioned.
_BASE_PROBE = "Hxgjpq"


def _baseline_in_layout(family: str, bold: bool) -> Optional[float]:
    """Height of the baseline above the layout box's bottom edge, in em.

    Measured rather than derived, because matplotlib's line metrics are not
    the face's ascender/descender.
    """
    key = (family, bold)
    hit = _vmetric_cache.get(key)
    if hit is not None:
        return hit
    lay = layout_box(_BASE_PROBE, family, "left", "baseline", 0.0, bold)
    if lay is None:
        return None
    # With va="baseline" the anchor IS the baseline, so its offset from the
    # layout box bottom is just -y0.
    out = -lay[1]
    _vmetric_cache[key] = out
    return out
