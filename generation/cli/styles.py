"""House styles: whole drawing-office conventions, picked per drawing.

A real corpus of drawings is not one template with the font swapped. Two
offices differ in what their border looks like, where the number sits on a
dimension line, whether a size carries a tolerance, what an arrowhead is, how
a repeated hole is counted off, how views are captioned and what the note
block says. A detector trained on one template learns the template.

Each entry here is a *bundle* of such conventions, taken from real drawings:

``iso_office``
    European ISO practice as on a typical issued sheet: zoned border with
    centering marks, filled arrowheads, the number floating above an unbroken
    dimension line, aligned (bidirectional) text, symmetric tolerances on some
    sizes, a numbered note block, revision table and parts list.

``asme_inch``
    US practice: zoned border, the number sitting *in* a gap in the dimension
    line, every number read horizontally (unidirectional), ``4X`` counts, an
    UNLESS OTHERWISE SPECIFIED tolerance block, third-angle projection.

``vintage_blueprint``
    An older, heavier, hand-drafted look: double border with no zones, large
    all-capital text, 45-degree tick terminators, limit dimensions stacked as
    a max over a min, generous spacing.

``minimal_cad``
    A CAD default nobody customised: thin uniform strokes, small text, plain
    border, open arrowheads, no tolerances, no notes, small title block.

``workshop_metric``
    A fabrication-shop print: plain border, big text, dot terminators are
    avoided but the sizes are coarse (1 decimal), a general tolerance note,
    hole tables preferred, title block bottom-left, views captioned PLAN /
    ELEVATION / END VIEW.

Only *presentation* is varied. Geometry, stated values, tolerance magnitudes
and view selection are the labels; perturbing those would corrupt the dataset
rather than augment it.
"""
from __future__ import annotations

from typing import Dict

#: name -> Style field overrides.
HOUSE_STYLES: Dict[str, dict] = {
    "iso_office": dict(
        frame_style="zoned", zone_cols=8, zone_rows=4, centering_marks=True,
        arrow_style="filled", dim_text_mode="above",
        tolerance_style="symmetric", tolerance_fraction=0.22,
        decimals=2, qty_prefix="X", center_style="line", stock_outline=True,
        notes_style="numbered", rev_table=True, parts_list=True,
        view_label_style="ortho", scale_format="ratio", table_side="tr",
        sheet_margin=10.0, title_block_w=150.0, title_block_h=42.0,
        title_block_rows=4, title_block_cols=3, title_block_corner="br",
        sheet_pref="A3", text_height=3.0, dim_text_height=2.8, note_text_height=2.6,
        arrow=2.4, stroke_scale=1.0, projection="first",
    ),
    "asme_inch": dict(
        frame_style="zoned", zone_cols=4, zone_rows=2, centering_marks=False,
        arrow_style="filled", dim_text_mode="inline",
        tolerance_style="bilateral", tolerance_fraction=0.30,
        decimals=3, qty_prefix="X", center_style="mark", tolerance_block="decimal",
        two_sections=True,
        notes_style="uos", rev_table=True, parts_list=False,
        view_label_style="letters", scale_format="worded", table_side="tl",
        sheet_margin=12.0, title_block_w=165.0, title_block_h=38.0,
        title_block_rows=3, title_block_cols=4, title_block_corner="br",
        sheet_pref="A3", text_height=2.6, dim_text_height=2.5, note_text_height=2.4,
        arrow=2.0, stroke_scale=0.9, projection="third",
    ),
    "vintage_blueprint": dict(
        frame_style="double", centering_marks=False,
        arrow_style="tick", dim_text_mode="above",
        tolerance_style="limits", tolerance_fraction=0.28,
        decimals=2, qty_prefix="HOLES", center_style="line", flag_notes=True,
        chamfer_style="c",
        ref_dim_style="ref", pcd_style="pcd",
        notes_style="numbered", rev_table=False, parts_list=False,
        view_label_style="ortho", scale_format="fraction",
        sheet_margin=14.0, title_block_w=140.0, title_block_h=50.0,
        title_block_rows=5, title_block_cols=2, title_block_corner="br",
        sheet_pref="A2", text_height=3.8, dim_text_height=3.4, note_text_height=3.2,
        arrow=3.0, stroke_scale=1.6, projection="first",
    ),
    "minimal_cad": dict(
        frame_style="plain", centering_marks=False,
        arrow_style="open", dim_text_mode="unidirectional",
        tolerance_style="none", tolerance_fraction=0.0,
        decimals=2, qty_prefix="X", center_style="mark", rev_marks=True,
        notes_style="none", rev_table=False, parts_list=False,
        view_label_style="none", scale_format="ratio",
        sheet_margin=8.0, title_block_w=110.0, title_block_h=30.0,
        title_block_rows=3, title_block_cols=2, title_block_corner="tr",
        sheet_pref="A4", text_height=2.2, dim_text_height=2.1, note_text_height=2.0,
        arrow=1.7, stroke_scale=0.75, projection="third",
    ),
    "workshop_metric": dict(
        frame_style="zoned", zone_cols=6, zone_rows=4, centering_marks=False,
        arrow_style="filled", dim_text_mode="unidirectional",
        tolerance_style="symmetric", tolerance_fraction=0.20,
        decimals=1, qty_prefix="OFF", center_style="mark",
        ref_dim_style="none", pcd_style="pcd_dots", chamfer_style="cham",
        notes_style="uos", rev_table=False, parts_list=True,
        view_label_style="building", scale_format="worded",
        sheet_margin=11.0, title_block_w=135.0, title_block_h=46.0,
        title_block_rows=4, title_block_cols=2, title_block_corner="bl",
        sheet_pref="A3", text_height=3.4, dim_text_height=3.2, note_text_height=3.0,
        arrow=2.8, stroke_scale=1.25, projection="first",
    ),
    "portrait_din": dict(
        frame_style="zoned", zone_cols=4, zone_rows=6, centering_marks=True,
        arrow_style="filled", dim_text_mode="above",
        tolerance_style="bilateral", tolerance_fraction=0.24,
        decimals=2, qty_prefix="X", center_style="line",
        notes_style="numbered", rev_table=True, parts_list=False,
        view_label_style="ortho", scale_format="ratio", table_side="tl",
        sheet_pref="A3P", sheet_margin=12.0,
        title_block_w=140.0, title_block_h=40.0,
        title_block_rows=4, title_block_cols=3, title_block_corner="tr",
        text_height=3.0, dim_text_height=2.8, note_text_height=2.6,
        arrow=2.4, stroke_scale=1.05, projection="first",
    ),
    "toolroom_tall": dict(
        frame_style="plain", centering_marks=False,
        arrow_style="open", dim_text_mode="inline",
        tolerance_style="limits", tolerance_fraction=0.30,
        decimals=3, qty_prefix="TYP", center_style="mark", pcd_style="long",
        two_sections=True,
        notes_style="uos", rev_table=False, parts_list=True,
        view_label_style="letters", scale_format="fraction", table_side="tr",
        sheet_pref="A4P", sheet_margin=9.0,
        title_block_w=120.0, title_block_h=44.0,
        title_block_rows=5, title_block_cols=2, title_block_corner="tl",
        text_height=2.8, dim_text_height=2.6, note_text_height=2.4,
        arrow=2.1, stroke_scale=0.95, projection="third",
    ),
    "aerospace_iso": dict(
        frame_style="zoned", zone_cols=8, zone_rows=4, centering_marks=True,
        arrow_style="filled", dim_text_mode="unidirectional",
        tolerance_style="bilateral", tolerance_fraction=0.30,
        decimals=3, qty_prefix="X", center_style="line", flag_notes=True,
        rev_marks=True,
        notes_style="uos", rev_table=True, parts_list=True,
        view_label_style="letters", scale_format="worded", table_side="right",
        sheet_pref="A2", sheet_margin=13.0,
        title_block_w=170.0, title_block_h=44.0,
        title_block_rows=4, title_block_cols=4, title_block_corner="br",
        text_height=2.8, dim_text_height=2.6, note_text_height=2.5,
        arrow=2.2, stroke_scale=0.95, projection="third",
        section_views=True, detail_views=True,
    ),
    "sheetmetal_shop": dict(
        frame_style="plain", centering_marks=False,
        arrow_style="tick", dim_text_mode="above",
        tolerance_style="symmetric", tolerance_fraction=0.18,
        decimals=1, qty_prefix="PLACES", center_style="mark", stock_outline=True,
        notes_style="numbered", rev_table=False, parts_list=False,
        view_label_style="building", scale_format="ratio", table_side="bottom",
        sheet_pref="A3", sheet_margin=10.0,
        title_block_w=130.0, title_block_h=38.0,
        title_block_rows=4, title_block_cols=2, title_block_corner="br",
        text_height=3.6, dim_text_height=3.3, note_text_height=3.0,
        arrow=2.9, stroke_scale=1.35, projection="first",
        # a flat part has nothing worth sectioning; the shop wants the
        # hole chart and the bend line instead
        section_views=False, detail_views=True,
    ),
    "microparts_lab": dict(
        frame_style="double", centering_marks=False,
        arrow_style="open", dim_text_mode="inline",
        tolerance_style="limits", tolerance_fraction=0.35,
        decimals=3, qty_prefix="X", center_style="mark",
        notes_style="uos", rev_table=True, parts_list=False,
        view_label_style="letters", scale_format="fraction", table_side="tr",
        sheet_pref="A4P", sheet_margin=8.0,
        title_block_w=115.0, title_block_h=40.0,
        title_block_rows=4, title_block_cols=2, title_block_corner="bl",
        text_height=2.4, dim_text_height=2.3, note_text_height=2.2,
        arrow=1.9, stroke_scale=0.85, projection="third",
        section_views=True, detail_views=True,
    ),
}

#: Sampling weights. ISO and ASME practice dominate real corpora; the vintage
#: and stripped-down CAD looks are real but less common.
STYLE_WEIGHTS = {
    "iso_office": 4,
    "asme_inch": 4,
    "workshop_metric": 3,
    "minimal_cad": 2,
    "vintage_blueprint": 2,
    "portrait_din": 3,
    "toolroom_tall": 2,
    "aerospace_iso": 3,
    "sheetmetal_shop": 3,
    "microparts_lab": 2,
}


def pick(rng) -> str:
    """Choose a house style with :data:`STYLE_WEIGHTS`, given a seeded RNG."""
    bag = [name for name, n in STYLE_WEIGHTS.items() for _ in range(n)]
    return rng.pick(tuple(sorted(bag)))


# --------------------------------------------------------------------------- #
# tolerances
# --------------------------------------------------------------------------- #
#: Tolerance magnitudes in millimetres, by nominal size. Loosely IT10-IT12 --
#: the grades a general machining note actually quotes -- so the numbers read
#: like a real drawing rather than being arbitrary.
_GRADES = ((6.0, (0.05, 0.1, 0.1, 0.2)),
           (30.0, (0.1, 0.1, 0.2, 0.3)),
           (120.0, (0.1, 0.2, 0.3, 0.5)),
           (1e9, (0.2, 0.3, 0.5, 0.8)))


def _band(nominal: float):
    """Plausible tolerance magnitudes for a nominal size.

    Capped at a tenth of the size as well as by the size band: a 0.18 mm
    feature cannot carry the +/-0.2 a 5 mm one does, and it produced limits
    like "0.18/-0.02" -- a negative lower limit, which is nonsense on a
    drawing. Sub-millimetre parts are real in this corpus, so the cap is not
    hypothetical.
    """
    for limit, grades in _GRADES:
        if abs(nominal) <= limit:
            break
    else:
        grades = _GRADES[-1][1]
    cap = max(abs(nominal) * 0.1, 1e-4)
    out = []
    for g in grades:
        t = min(g, cap)
        # keep it to a printable number of decimals
        t = round(t, 4 if t < 0.01 else (3 if t < 0.1 else 2))
        out.append(t)
    return tuple(out)


def tolerance_text(style, nominal: float, key: str) -> str:
    """Tolerance to print after a size, or ``""`` for an untoleranced one.

    A real drawing tolerances the sizes that matter and leaves the rest to the
    general note, so only ``Style.tolerance_fraction`` of dimensions get one,
    chosen deterministically from ``key`` (BLAKE2b of the part name and the
    dimension, never Python's salted ``hash()``).

    The three presentations are the ones drawings actually use:
    ``50 ±0.2`` (symmetric), ``50 +0.2/-0.1`` (bilateral) and ``50.2/49.8``
    (limits, the max over the min written on one line).
    """
    mode = getattr(style, "tolerance_style", "none")
    frac = float(getattr(style, "tolerance_fraction", 0.0) or 0.0)
    if mode == "none" or frac <= 0.0 or nominal <= 0:
        return ""
    import hashlib
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    n = int.from_bytes(h, "big")
    if (n % 1000) / 1000.0 >= frac:
        return ""
    grades = _band(nominal)
    t = grades[(n >> 12) % len(grades)]
    if t <= 0:
        return ""
    lower = grades[((n >> 20) + 1) % len(grades)]
    if abs(lower - t) < 1e-9:
        lower = t / 2.0
    # An office writes every number on its sheets to the same precision. This
    # used to choose its own (4 decimals under 0.01, 3 under 0.1), so a
    # two-decimal house style printed "3.75 +0.050/-0.025" next to "4.47" --
    # seen on 0000_00000126 SECTION A-A. Follow style.decimals, and only add
    # places when the tolerance would otherwise round away to zero.
    nd = int(getattr(style, "decimals", 2))
    smallest = t if mode != "bilateral" else min(t, lower)
    while nd < 4 and round(smallest, nd) <= 0:
        nd += 1
    if mode == "symmetric":
        return f" \u00b1{t:.{nd}f}"
    if mode == "bilateral":
        return f" +{t:.{nd}f}/-{lower:.{nd}f}"
    if mode == "limits":
        return ""          # handled by limit_text(), which replaces the size
    return ""


def limit_text(style, nominal: float, key: str, fmt) -> str:
    """A size written as its two limits, or ``""`` to keep the nominal.

    ``fmt`` formats a length the way the rest of the drawing does, so the
    limits carry the sheet's own precision.
    """
    if getattr(style, "tolerance_style", "none") != "limits":
        return ""
    frac = float(getattr(style, "tolerance_fraction", 0.0) or 0.0)
    if frac <= 0.0 or nominal <= 0:
        return ""
    import hashlib
    n = int.from_bytes(
        hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big")
    if (n % 1000) / 1000.0 >= frac:
        return ""
    grades = _band(nominal)
    t = grades[(n >> 12) % len(grades)]
    if t <= 0 or nominal - t <= 0:
        return ""                      # a limit below zero is not a size
    return f"{fmt(nominal + t)}/{fmt(nominal - t)}"


def qty_text(style, n: int) -> str:
    """How this office counts off a repeated feature: ``4X``, ``4 HOLES`` ..."""
    kind = getattr(style, "qty_prefix", "X")
    if kind == "HOLES":
        return f"{n} HOLES "
    if kind == "PLACES":
        return f"{n} PLACES "
    if kind == "OFF":
        return f"{n} OFF "
    if kind == "TYP":
        return ""                       # written as a suffix -- see counted()
    return f"{n}X "


def counted(style, n: int, note: str) -> str:
    """``note`` with this office's count-off wording applied.

    Most offices prefix the count (``4X R6``). A few write the note once and
    mark it TYPICAL instead (``R6 TYP``), which is the wording BS 8888 and
    plenty of shop drawings use for a feature repeated identically. It is a
    suffix, so it cannot go through :func:`qty_text` -- and a count-off that
    is silently dropped is a drawing that under-states how many holes to
    drill, which is why this is a function and not a format string at each
    call site.
    """
    if n <= 1:
        return note
    if getattr(style, "qty_prefix", "X") == "TYP":
        return f"{note} TYP"
    return qty_text(style, n) + note


def reference_text(style, text: str) -> str:
    """A size that repeats information stated elsewhere, marked as reference.

    A chain of dimensions plus the overall size over-dimensions the part:
    the last one is arithmetic, not an instruction, and both ASME Y14.5 and
    ISO 129 have a way of saying so -- ``(34)`` in parentheses, or ``34 REF``.
    Which one is a house convention.
    """
    kind = getattr(style, "ref_dim_style", "paren")
    if kind == "paren":
        return f"({text})"
    if kind == "ref":
        return f"{text} REF"
    return text


def bolt_circle_text(style, dia_text: str, equally_spaced: bool) -> str:
    """How this office states a hole pattern's pitch circle.

    All four wordings are on real drawings: "ON \u2300100 B.C." (ASME),
    "ON \u2300100 PCD" and "\u2300100 P.C.D." (BS/ISO shops), and the long
    "EQUALLY SPACED ON \u2300100 BOLT CIRCLE". Equal spacing is stated only
    when it is true -- it is an instruction to the machinist, not decoration.
    """
    kind = getattr(style, "pcd_style", "bc")
    eq = "EQ SP " if equally_spaced else ""
    if kind == "pcd":
        return f"{eq}ON {dia_text} PCD"
    if kind == "pcd_dots":
        return f"{eq}{dia_text} P.C.D."
    if kind == "long":
        return f"{'EQUALLY SPACED ' if equally_spaced else ''}ON {dia_text} BOLT CIRCLE"
    return f"{eq}ON {dia_text} B.C."
