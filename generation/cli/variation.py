"""Per-drawing style variation, so a corpus is not 24 copies of one template.

A detector trained on drawings that all use the same font, the same stroke
weights and a title block in the same place learns those incidentals as if
they were signal. This module perturbs the *presentation* of a drawing --
never its geometry and never a stated value -- so the same part can be
rendered in many plausible house styles.

What varies
-----------
* **Font family**, from the families actually installed (checked at import,
  not assumed).
* **Text heights**, scaled together so the drawing stays internally
  consistent.
* **Stroke weights**, as a single multiplier applied across every layer, so
  the *relative* hierarchy (visible > frame > dimension) is preserved.
* **Title block**: width, height, row count and which corner it sits in.
* **Tables**: which side of the sheet they go, their row height, whether they
  are ruled with full gridlines or only under the header.
* **Layout**: sheet margin, view spacing, and which free corner takes the
  isometric.

What must NOT vary
------------------
Anything that changes what the drawing *says*: dimension values, tolerances,
thread designations, feature recognition, the choice of views. Those are the
labels; perturbing them would corrupt the dataset rather than augment it.

The character-width factor
--------------------------
This is the subtle part. Text width is estimated as
``len(text) * height * CHAR_W`` in roughly twenty places, and that estimate
drives both annotation placement *and* the COCO bounding boxes. The
factor is font-specific -- measured across the installed families it ranges
from 0.457 (STIXGeneral) to 0.576 (DejaVu Sans Mono), a 26% spread. Changing
the font without changing the factor would leave every box the wrong width.

So each font carries its own measured ratio, obtained from matplotlib's
``TextPath`` over representative drawing strings rather than guessed, and
``Style.char_w`` travels with ``Style.font_family``.

Determinism
-----------
Seeded with BLAKE2b of the part name, like :mod:`autodraft.roughness` and
:mod:`autodraft.synth_pmi`. Python's ``hash()`` is salted per process and
would restyle a part on every run, making a dataset impossible to regenerate.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Dict, Optional

#: Representative strings from real drawings -- digits, the diameter glyph,
#: thread and roughness callouts -- so the measured ratio reflects the text
#: this program actually draws rather than English prose.
_SAMPLES = (
    "4X \u23006.6 THRU", "157.98", "M10 x 1-6H", "Ra 1.6", "R0.52",
    "12.5", "C'BORE \u230011 DP 6.5", "THICKNESS 12",
)

#: Families to draw from, in preference order. Filtered at import against
#: what is really installed, because a missing family silently falls back to
#: the default and would make the recorded font a lie.
_CANDIDATE_FONTS = (
    "DejaVu Sans",
    "DejaVu Sans Mono",
    "DejaVu Serif",
    "STIXGeneral",
    # Widely-installed faces that are NOT present in this container but are on
    # many workstations. Listing them costs nothing -- a family that cannot be
    # resolved is skipped -- and it means the variation gets richer on a
    # machine that has them rather than being capped at whatever ships here.
    "Arial",
    "Helvetica",
    "Liberation Sans",
    "Liberation Serif",
    "Nimbus Sans",
    "Tahoma",
    "Verdana",
)

#: Glyphs a mechanical drawing cannot do without. A font missing any of them
#: renders tofu boxes in place of real callouts -- DejaVu Serif has no
#: U+2300 DIAMETER SIGN, so every bore note came out as "[]6.6".
_REQUIRED_GLYPHS = ("\u2300", "\u00b0", "\u00b5", "_")


def _has_glyphs(family: str) -> bool:
    """True when the family can draw every glyph a drawing needs.

    The font's character map is inspected directly. Rendering is not a
    usable test here: matplotlib silently substitutes a glyph from another
    font and merely emits a warning, so the drawn path is non-empty even
    though the chosen font cannot draw the character.
    """
    try:
        from matplotlib.font_manager import FontProperties, findfont
        from fontTools.ttLib import TTFont, TTCollection
        path = findfont(FontProperties(family=family),
                        fallback_to_default=False)
        if not path:
            return False
        # A .ttc collection is a bundle of faces -- the CJK one here is tens
        # of megabytes and parsing it cost 180 MB, enough to have batch
        # workers OOM-killed on a 1 GB box. No drawing needs a CJK face, so
        # collections are declined rather than inspected.
        if path.lower().endswith(".ttc"):
            return False
        try:
            fonts = [TTFont(path)]
        except Exception:
            # .ttc collections hold several faces and must be opened as one
            fonts = list(TTCollection(path).fonts)
        try:
            covered = set()
            for font in fonts:
                for table in font["cmap"].tables:
                    covered |= set(table.cmap)
            return all(ord(ch) in covered for ch in _REQUIRED_GLYPHS)
        finally:
            # A parsed TTFont holds the whole face in memory and a CJK
            # collection is tens of megabytes. Screening five candidates and
            # keeping them cost ~130 MB, which on this 2-core/1 GB box was
            # enough to have batch workers OOM-killed mid-run.
            for font in fonts:
                try:
                    font.close()
                except Exception:
                    pass
    except Exception:
        return False


#: Characters that must render correctly, because drawings contain them.
#: The TeX Computer Modern faces bundled with matplotlib (cmr10, cmss10 ...)
#: use TeX's own encoding, not Unicode: they draw U+005F LOW LINE as a raised
#: quote, so "laser_panel" came out as "laser'panel" in the title block. They
#: are excluded by measurement rather than by name -- a font is rejected when
#: its underscore does not sit below the baseline.
_REQUIRED_GLYPH_CHECK = "_"


def _underscore_is_sane(family: str) -> bool:
    """True when U+005F renders as a low line rather than a raised mark."""
    try:
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties
        ext = TextPath((0, 0), _REQUIRED_GLYPH_CHECK,
                       prop=FontProperties(family=family, size=100)
                       ).get_extents()
        # A real underscore sits at or below the baseline; the TeX faces put
        # it around y=+56 at 100 pt.
        return ext.y0 < 10.0
    except Exception:
        return False

#: Fallback if measurement is impossible (headless without matplotlib).
_DEFAULT_CHAR_W = 0.62


def _measure_char_w(family: str) -> Optional[float]:
    """Mean width-per-character, in units of text height, for one family.

    Returns None when the family cannot be loaded. matplotlib silently
    substitutes a default for an unknown family, so the result is compared
    against the resolved font file to catch that.
    """
    try:
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties, findfont
        prop = FontProperties(family=family, size=100)
        # findfont falls back silently; verify we got what we asked for.
        resolved = findfont(prop, fallback_to_default=False)
        if not resolved:
            return None
        ratios = []
        for text in _SAMPLES:
            width = TextPath((0, 0), text, prop=prop).get_extents().width
            ratios.append(width / (len(text) * 100.0))
        if not ratios:
            return None
        # The estimator is used to RESERVE space, so a slight over-estimate is
        # safer than an under-estimate: too small and labels overlap, too
        # large and they are merely a little spread out. 4% headroom.
        return round(sum(ratios) / len(ratios) * 1.04, 4)
    except Exception:
        return None


def _available_fonts() -> Dict[str, float]:
    """{family: char_width_ratio} for the families really present here."""
    out: Dict[str, float] = {}
    # matplotlib logs "findfont: Font family ... not found" at WARNING for
    # every family it cannot resolve. Probing a deliberately optimistic list
    # would print one line per absent font on every run, which is noise about
    # a non-problem.
    import logging
    fm_log = logging.getLogger("matplotlib.font_manager")
    previous = fm_log.level
    fm_log.setLevel(logging.ERROR)
    try:
        return _screen(out)
    finally:
        fm_log.setLevel(previous)


def _screen(out: Dict[str, float]) -> Dict[str, float]:
    for family in _CANDIDATE_FONTS:
        if not _underscore_is_sane(family):
            continue
        if not _has_glyphs(family):
            continue
        ratio = _measure_char_w(family)
        if ratio:
            out[family] = ratio
    if not out:
        out["DejaVu Sans"] = _DEFAULT_CHAR_W
    return out


# Measured LAZILY, not at import. Screening the candidates costs ~6 s
# (loading each font and walking its character map), and every spawned batch
# worker re-imports this module -- on a 2-core box that stalled a parallel
# run for minutes before any drawing was produced. Computed once per process
# on first use instead, and skipped entirely when variation is off.
_FONTS_CACHE: Optional[Dict[str, float]] = None


def fonts() -> Dict[str, float]:
    """{family: char_width_ratio} for the usable families, measured once."""
    global _FONTS_CACHE
    if _FONTS_CACHE is None:
        _FONTS_CACHE = _available_fonts()
    return _FONTS_CACHE


def _seed(text: str) -> int:
    """Stable integer from a name. BLAKE2b: hash() is salted per process."""
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


class _Rng:
    """Deterministic choices from one seed.

    A plain modulo of the same seed for every decision correlates them, so
    the seed is advanced with an LCG between draws.
    """

    def __init__(self, seed: int) -> None:
        self._s = seed or 1

    def _next(self) -> int:
        self._s = (self._s * 6364136223846793005 + 1442695040888963407) \
            % (1 << 64)
        return self._s >> 16

    def pick(self, seq):
        seq = list(seq)
        return seq[self._next() % len(seq)]

    def uniform(self, lo: float, hi: float, places: int = 3) -> float:
        return round(lo + (hi - lo) * ((self._next() % 10_000) / 10_000.0),
                     places)

    def chance(self, num: int, den: int) -> bool:
        return (self._next() % den) < num


def vary_style(style, part_name: str, enabled: bool = True,
               has_holes: bool = True):
    """Return a copy of `style` with its presentation perturbed.

    ``enabled=False`` returns the style unchanged, so a caller can turn the
    whole feature off without branching at every use site.

    ``has_holes`` says whether the part has any closed bore. A hole table on
    a part with no holes is an empty box, so the pipeline never draws one --
    which meant a ``prefer_hole_table`` draw spent on such a part was simply
    wasted. On this corpus 10 of 24 parts have no bores at all, so four of
    the five parts that had chosen a TOP table could never show one, and top
    tables looked absent. The draw is now only made where it can take effect.
    """
    if not enabled:
        return style
    rng = _Rng(_seed("style:" + part_name))

    # A house style comes first: it decides the drawing CONVENTIONS (border,
    # arrowheads, where the number sits, tolerancing, note block, captions).
    # The per-drawing jitter below then varies the presentation inside that
    # office's house rules, so two ISO sheets differ without either of them
    # turning into an ASME sheet.
    from .styles import HOUSE_STYLES, pick as _pick_house
    house = _pick_house(rng)
    style = replace(style, house_style=house, **HOUSE_STYLES[house])

    table = fonts()
    family = rng.pick(sorted(table))
    char_w = table[family]

    # One multiplier for all text, so the drawing stays self-consistent: a
    # sheet with big dimensions and tiny notes looks wrong rather than varied.
    # Widened from 0.88-1.18: that span produced sheets that were hard to
    # tell apart, which defeats the point.
    text_scale = rng.uniform(0.85, 1.25)
    # One multiplier for all strokes, preserving the weight hierarchy that
    # distinguishes visible from hidden from dimension lines.
    stroke_scale = rng.uniform(0.75, 1.45)

    return replace(
        style,
        font_family=family,
        char_w=char_w,
        stroke_scale=round(style.stroke_scale * stroke_scale, 3),
        text_height=round(style.text_height * text_scale, 3),
        dim_text_height=round(style.dim_text_height * text_scale, 3),
        note_text_height=round(style.note_text_height * text_scale, 3),
        # Arrowheads track the text, or they look detached from it.
        arrow=round(style.arrow * rng.uniform(0.85, 1.2), 3),
        # Title block: real drawings differ widely in proportion. The house
        # style sets the shape; this nudges it, so an ISO block stays an ISO
        # block while no two are identical.
        title_block_w=round(style.title_block_w * rng.uniform(0.88, 1.12), 1),
        title_block_h=round(style.title_block_h * rng.uniform(0.85, 1.15), 1),
        title_block_rows=max(3, style.title_block_rows + rng.pick((0, 0, 1))),
        # 2-4 columns below the title row. With rows this gives blocks
        # holding anywhere from 5 to 15 fields, so the structure genuinely
        # differs rather than one or two cells being absent.
        title_block_cols=max(2, style.title_block_cols + rng.pick((-1, 0, 0, 1))),
        title_block_corner=style.title_block_corner,
        # The block's typography is independent of the views': a drawing
        # office's title block often uses a different size from the body.
        title_text_scale=rng.uniform(0.80, 1.25),
        title_block_bold=rng.chance(1, 2),
        sheet_margin=round(style.sheet_margin * rng.uniform(0.85, 1.15), 1),
        # "left" was offered here but never read by the renderer -- the
        # table was always drawn in the right-hand column. Top and bottom are
        # the placements a real drawing actually uses for a short table.
        # Weighted toward the bands. A right-hand table is the pipeline's
        # historic default, so leaving the three equally likely made top and
        # bottom look scarce on a corpus where only ~14 parts can host a
        # table at all.
        table_side=rng.pick(("top", "top", "bottom", "bottom", "right",
                             "tl", "tr", "tr")),
        # A hole table is normally reserved for parts too dense to letter
        # with leaders, which on this corpus meant it was drawn on ONE part
        # of 24 -- most have only 0-1 hole groups, so no density threshold
        # could trigger one, and the three table placements were unreachable.
        # Tabulating is a legitimate house style, so it is chosen directly.
        # This changes only the PRESENTATION (table vs leader notes); the
        # holes and their stated values are identical either way.
        # Only meaningful on a part that HAS holes -- see the docstring.
        prefer_hole_table=(has_holes and rng.chance(1, 2)),
        table_row_h=round(rng.uniform(4.4, 6.4), 2),
        table_ruled=rng.chance(2, 3),
        view_gap_scale=rng.uniform(0.85, 1.3),
        # Even inside one office these two move: a draughtsman tolerances the
        # sizes that matter on this part, and captions get dropped on a
        # single-view sheet.
        tolerance_fraction=round(style.tolerance_fraction
                                 * rng.uniform(0.7, 1.3), 3),
        zone_cols=style.zone_cols + rng.pick((0, 0, 2)),
        # the datum corner this drawing dimensions from
        dim_side_h=rng.pick(("bottom", "bottom", "top")),
        dim_side_v=rng.pick(("left", "left", "right")),
        # Which way up the paper goes. A house style has a usual orientation,
        # but no office issues every drawing the same way round -- a long
        # part goes on landscape paper and a tall one on portrait whatever
        # the letterhead says. Counted over the 24-part corpus the sheet
        # chooser alone gave 20 portrait / 4 landscape (83%), because a
        # two-view drawing stacks into a tall block and a turned sheet always
        # scored the better fill. This draw is the balance: half the drawings
        # start from the turned form of the house sheet, and _choose_sheet
        # now respects that choice unless the part genuinely will not fit.
        # Kept LAST so it does not shift any draw above it.
        sheet_pref=_turn_sheet(style.sheet_pref) if rng.chance(1, 2)
        else style.sheet_pref,
        # --- conventions that vary inside an office as well as between them.
        # Each of these is a real drafting choice a draughtsman makes per
        # drawing, and pinning each one to a single house style meant a
        # feature appeared on two or three sheets of the corpus and never in
        # combination with the others. Drawn LAST so no draw above shifts.
        #
        # A flag note needs somewhere to point and a stock outline needs
        # machining allowance, so both are only ever offered here -- the
        # drawing code still decides whether it can honestly draw them.
        flag_notes=(style.flag_notes or rng.chance(1, 4)),
        stock_outline=(style.stock_outline or rng.chance(1, 4)),
        rev_marks=(style.rev_marks or rng.chance(1, 3)),
        two_sections=(style.two_sections or rng.chance(1, 3)),
        chamfer_style=rng.pick(("long", "long", "c", "cham")),
        pcd_style=rng.pick(("bc", "pcd", "pcd_dots", "long")),
        ref_dim_style=rng.pick(("paren", "paren", "ref", "none")),
        # How densely this office letters a view before it reaches for a
        # bigger sheet -- see render._label_density.
        max_label_density=round(rng.uniform(0.7, 1.5), 2),
    )


def _turn_sheet(name: str) -> str:
    """The same sheet size, turned: "A3" <-> "A3P". Unknown names unchanged."""
    from .sheet import SHEETS
    if not name:
        return name
    other = name[:-1] if name.endswith("P") else name + "P"
    return other if other in SHEETS else name
