"""Sheet layout: scale selection, view placement, title block, primitive list."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SHEETS = {           # ISO / ANSI sheet sizes in mm (landscape)
    "A4": (297, 210), "A3": (420, 297), "A2": (594, 420),
    "A1": (841, 594), "A0": (1189, 841),
    "A": (279.4, 215.9), "B": (431.8, 279.4), "C": (558.8, 431.8), "D": (863.6, 558.8),
}
# Portrait forms. A tall part on a landscape sheet leaves two columns of blank
# paper; real offices simply turn the sheet, and plenty of house styles use
# portrait as their default.
SHEETS.update({f"{k}P": (h, w) for k, (w, h) in list(SHEETS.items())})

# Standard-ish drawing scales, finely graded so the fitted size rarely has to
# drop a whole step (a coarse ladder is what leaves half a sheet empty).
PREFERRED_SCALES = [
    (1000, 1), (500, 1), (200, 1), (150, 1),
    (100, 1), (75, 1), (50, 1), (40, 1), (30, 1), (20, 1), (15, 1),
    (10, 1), (7.5, 1), (5, 1), (4, 1), (3, 1), (2.5, 1),
    (2, 1), (1.5, 1), (1.25, 1), (1, 1),
    (1, 1.25), (1, 1.5), (1, 2), (1, 2.5), (1, 3), (1, 4), (1, 5), (1, 7.5),
    (1, 10), (1, 15), (1, 20), (1, 25), (1, 50), (1, 75),
    (1, 100), (1, 200), (1, 500), (1, 1000),
]


@dataclass
class Style:
    text_height: float = 3.0          # mm on paper
    dim_text_height: float = 2.8
    note_text_height: float = 2.8
    arrow: float = 2.2
    witness_gap: float = 0.8      # break between geometry and witness line
    witness_overshoot: float = 1.6  # how far a witness line passes the dim line
    dim_gap: float = 0.10             # fraction of view size (model units)
    max_position_dims: int = 6
    diameter_dims: bool = True   # a round view gets one diameter, not w x h
    body_dims: bool = True       # size each solid when a file holds several
    body_max_notes: int = 6      # ...unless there are more bodies than this
    fillet_max_notes: int = 4    # distinct radii called out on the primary view
    fillet_note_min_count: int = 1   # note a radius even when it occurs once
    chamfer_max_notes: int = 4   # distinct chamfers/countersinks per view
    thread_max_notes: int = 12   # thread specs placed (only ones the file states)
    gdt_max_frames: int = 6      # feature control frames drawn on a view
    gdt_max_datums: int = 4      # datum feature symbols drawn on a view
    # Annotations already placed on a view, above which it takes no SYNTHETIC
    # GD&T / thread / chamfer. Real PMI and recognised chamfers are always
    # drawn. Chosen by measurement on the dense 160x90 fixture (22 holes, 20
    # dimensions), seeded onto A4:
    #     threshold 15 -> 1:2      12 -> 1:2.5      10 -> 1:1.25
    # i.e. the un-budgeted drawing lost two scale steps making room for six
    # extra leaders. Note the result is NOT monotonic -- 12 is worse than 15 --
    # because the sheet fitter searches (scale, sheet, layout) jointly and a
    # slightly larger annotation set can tip it into a different basin. 10 is
    # the value that restores the scale this fixture reaches with no synthetic
    # PMI at all, so the feature costs no drawing quality.
    #
    # 12 was tried, to raise corpus-wide gdnts from 17 to 32: it regressed the
    # dense fixture to 1:2.5 and was rejected. Drawing legibility wins -- a
    # real corpus supplies coverage from parts that have room for it, whereas
    # an illegible sheet is bad training data whatever it is labelled.
    gdt_crowded_at: int = 10

    # Threads get their own crowding threshold, one step above the frame
    # budget. A thread callout is a single short leader ("4X M10 x 1-6H")
    # where a GD&T frame is a row of boxed compartments, so a view with no
    # room for a frame can still take a thread -- sharing the frame threshold
    # suppressed threads on every part that had tappable geometry.
    #
    # 11 is a measured boundary, not a guess. On the dense 160x90 fixture
    # (A4-seeded) 12 admits two thread notes and drops the drawing from
    # 1:1.25 to 1:2.5, two scale steps, because on that saturated view every
    # candidate placement crosses committed segments and the label has to go
    # ~60 mm clear of the geometry. 11 keeps 1:1.25 there while still letting
    # angled_bracket state its thread. Note the count this is compared
    # against under-reports congestion (it excludes dimension and witness
    # lines), which is why the useful value is far below the ~22 that the
    # "a thread is cheap" argument would suggest.
    # A view must offer at least this many genuinely free label placements --
    # no ink, no crossed segment, inside the view -- before it will take a
    # SYNTHETIC thread; a real thread read from the file is always drawn.
    #
    # This replaced a raw "annotations already placed" threshold, which could
    # not tell congestion apart from mere busyness. Measured: bracket_plate's
    # top view has 14 annotations but 26 of 96 candidate placements are
    # completely free, while the dense 160x90 fixture has 19 and its best
    # candidate still lands on ink. Any count threshold either admitted both
    # (dropping the dense sheet from 1:1.25 to 1:2, failing the scale test) or
    # rejected both (losing a thread with plenty of room for it).
    #
    # 14, lowered from 16: machined_block's top view offers 17 free slots on
    # the seeding pass but 14 on the final one (text is reserved in model
    # units, so a larger true scale eats slots), which lost a thread on a
    # view that plainly had room. Measured at 16/14/12 -- the dense fixture
    # holds 1:1.25 at all three, so 14 costs no drawing quality.
    # 12, lowered from 14: the module's own measurement showed the dense
    # 160x90 fixture keeps 1:1.25 at 16/14/12, so admitting views with at
    # least 12 genuinely free slots costs that drawing no scale. It widens
    # the set of views that will accept a SYNTHETIC thread, which raises the
    # corpus thread count without drawing an illegible leaf.
    thread_min_free_slots: int = 12

    # A GD&T frame is a row of boxed compartments, so it is probed at a wider
    # label than a thread note and needs fewer alternatives to count as
    # placeable -- there are simply fewer spots that fit one.
    gdt_frame_chars: int = 20
    gdt_min_free_slots: int = 2

    roughness_max_marks: int = 2  # per-feature finish symbols on a view
    roughness_general_note: bool = True   # the drawing-wide finish note
    angle_dims: bool = True        # dimension non-orthogonal corners
    angle_min_edge: float = 0.10   # min edge length, fraction of view span
    angle_ortho_tol: float = 2.0   # deg: treat as square, do not dimension
    angle_min_deg: float = 5.0     # ignore slivers and near-straight joins
    angle_max_dims: int = 4        # per view
    max_chain_dims: int = 10   # max segments in one dimension chain
    callout_repair_sweeps: int = 6  # re-placement sweeps to clear crossings
    leader_outward_weight: float = 90.0  # bias leaders away from view centre
    feature_boxes: bool = False   # draw a box around each recognised feature
    feature_box_margin: float = 1.0
    leader_angles: Tuple[float, ...] = tuple(range(0, 360, 10))
    leader_lengths: Tuple[float, ...] = (0.10, 0.15, 0.21, 0.28, 0.36,
                                        0.45, 0.55, 0.68)
    hidden_lines: bool = True
    iso_view: bool = True
    iso_shaded: bool = True   # fill the isometric so it reads as a solid
    #: Present the holes as a tagged table rather than leader callouts, even
    #: when leaders would fit. None = decide from density (the drafting rule);
    #: True/False = force. Varied per drawing, because the density rule alone
    #: meant a table was drawn on 1 of 24 sample parts and the three table
    #: placements were effectively unreachable.
    prefer_hole_table: bool = False
    max_callouts_per_sheet: int = 34  # above this many grouped notes, use a table
    symbols: bool = False       # ISO glyphs in notes (needs a capable font)
    table_max_cols: int = 3     # hole table wraps into at most this many columns
    auto_sheet: bool = True     # grow the sheet when a part is annotation-dense
    min_view_fill: float = 0.34  # target share of the sheet occupied by views
    #: Labels per square centimetre of view above which the drawing is judged
    #: too crowded to read and the sheet chooser steps up a size. Offices
    #: differ in how densely they letter, so the house styles vary it.
    max_label_density: float = 1.0
    min_scale: float = 0.5       # below this, prefer a larger sheet
    # A circular primary view needs enough paper diameter for its hole pattern
    # and radial callouts to remain readable. This is a paper-space minimum,
    # not a model-size threshold; the chooser can move the same view to the
    # next sheet instead of shrinking its annotations around a tiny disc.
    min_round_view_mm: float = 24.0
    auto_layout: bool = True     # search alternative view arrangements
    max_callout_reach: float = 0.32  # note distance outside the view box
    iso_max_frac: float = 0.40
    iso_max_zoom: float = 5.0  # iso may be drawn larger than the view scale   # max share of the drawing width the ISO may take
    iso_max_mm: float = 120.0     # and a hard cap in mm
    projection: str = "third"         # third | first angle

    # ---- presentation, varied per part by autodraft.variation ------------ #
    # These change how a drawing LOOKS, never what it says. Defaults reproduce
    # the original fixed house style exactly, so variation is purely additive.
    font_family: str = "DejaVu Sans"
    #: Width of one character in units of text height. Font-specific and
    #: MEASURED, not guessed: across the installed families it ranges 0.457
    #: to 0.576. It drives both annotation placement and the exported
    #: bounding boxes, so it must travel with `font_family` -- changing the
    #: font alone would leave every box the wrong width.
    char_w: float = 0.62
    #: Multiplier on every layer's stroke weight, applied uniformly so the
    #: visible > frame > dimension hierarchy survives.
    stroke_scale: float = 1.0
    title_block_w: float = 130.0
    title_block_h: float = 40.0
    title_block_rows: int = 4
    #: Columns in the title-block grid below the title row. More columns means
    #: more fields visible, which is what makes one block structurally
    #: different from the next.
    title_block_cols: int = 2
    title_block_corner: str = "br"   # br | bl | tr | tl     # br | bl
    #: Text scale for the title block alone, so its typography can differ
    #: from the dimensions on the same sheet.
    title_text_scale: float = 1.0
    #: Draw the title in a heavy weight.
    title_block_bold: bool = False
    #: Generate title-block content (drafter, checker, revision ...).
    vary_titleblock: bool = True
    sheet_margin: float = 10.0
    #: Which edge of the sheet the hole table sits against. A table is a tall
    #: narrow block, so "right" gives it a full-height column; "top" and
    #: "bottom" lay it out as a wide band across the sheet instead, which is
    #: how a short table is usually drawn.
    table_side: str = "right"          # right | top | bottom
    table_row_h: float = 5.0
    table_ruled: bool = True           # full gridlines, or header rule only
    view_gap_scale: float = 1.0

    # ---- drawing conventions, set as a bundle by autodraft.styles -------- #
    #: Named house style this drawing follows (see autodraft/styles.py).
    house_style: str = "iso_office"
    #: Sheet border. "plain" is a single rectangle; "zoned" adds the ruled
    #: zone strip with letters down the sides and numbers along the top and
    #: bottom, which is what nearly every issued drawing carries.
    frame_style: str = "plain"          # plain | zoned | double
    zone_cols: int = 8
    zone_rows: int = 4
    centering_marks: bool = False       # ISO 5457 marks at the mid-edges
    #: Arrowhead: a solid triangle (ISO/ASME default), an open vee, or the
    #: 45-degree tick used on architectural and some sheet-metal prints.
    arrow_style: str = "filled"         # filled | open | tick | dot
    #: Where the number sits. "above" floats it over an unbroken dimension
    #: line (ISO); "inline" centres it in a gap in the line (common in the
    #: US); both keep the text aligned with the line. "unidirectional" reads
    #: every number horizontally, whatever the line's direction.
    dim_text_mode: str = "above"        # above | inline | unidirectional
    #: Tolerancing shown beside a size: none, "50 ±0.2", "50 +0.10/-0.05" or
    #: stacked limits. Only a fraction of dimensions carry one.
    tolerance_style: str = "none"       # none | symmetric | bilateral | limits
    tolerance_fraction: float = 0.0     # share of linear dims that get one
    #: Decimal places printed. Real offices differ; 2 is the common default.
    decimals: int = 2
    #: How a repeated feature is counted off: "4X", "4 HOLES", "4 PLACES".
    qty_prefix: str = "X"               # X | HOLES | PLACES | OFF | TYP
    #: How an over-dimensioning size is marked: "(34)", "34 REF", or not at
    #: all. See styles.reference_text.
    ref_dim_style: str = "paren"        # paren | ref | none
    #: Wording for a hole pattern's pitch circle -- see
    #: styles.bolt_circle_text.
    pcd_style: str = "bc"               # bc | pcd | pcd_dots | long
    #: Draw numbered flag triangles on the views, pointing at where a general
    #: note applies. A house convention -- see render._emit_flag_notes.
    flag_notes: bool = False
    #: Draw the stock envelope round the primary view in phantom line.
    stock_outline: bool = False
    #: How the general-tolerance block is written: an ISO class, or an ASME
    #: table by decimal places. See render._general_tolerance_notes.
    tolerance_block: str = "class"      # class | decimal
    #: Mark the dimensions changed at this revision with a lettered triangle.
    rev_marks: bool = False
    #: Take a second section (B-B) when the part has the features for it.
    two_sections: bool = False
    #: How a 45-degree break is written: "2 X 45\u00b0", "C2" or with CHAM.
    chamfer_style: str = "long"         # long | c | cham
    #: Centre marks on bores: a short cross, full centre lines, or nothing.
    center_style: str = "line"          # line | mark | none
    #: General-note block: numbered list, an UNLESS OTHERWISE SPECIFIED
    #: tolerance block, or nothing at all.
    notes_style: str = "numbered"       # numbered | uos | none
    #: Small revision history table, as issued drawings carry.
    rev_table: bool = False
    #: Parts list above the title block.
    parts_list: bool = False
    #: View captions: TOP/FRONT, VIEW A/B, PLAN/ELEVATION, or none.
    view_label_style: str = "ortho"     # ortho | letters | building | none
    #: Sheet this office reaches for first. A "P" suffix is the portrait
    #: form: a tall part on a landscape sheet leaves two columns of blank
    #: paper, and real offices simply turn the sheet.
    sheet_pref: str = ""
    #: Ring a crowded region and redraw it magnified as DETAIL A.
    detail_views: bool = True
    #: Draw one secondary view as a full section when the part has internal
    #: features -- a section states what a hidden-line thicket cannot.
    section_views: bool = True
    #: Draw a labelled detail view of each solid in a multi-body file.
    subpart_views: bool = True
    #: Which edge of a view its dimension chains run along. An office
    #: dimensions consistently from one datum corner; which corner differs
    #: between offices, and fixing it made every sheet look alike.
    dim_side_h: str = "bottom"          # bottom | top
    dim_side_v: str = "left"            # left | right
    #: Size a sloping edge along its own direction (an aligned dimension), as
    #: against leaving it to the angular dimension alone.
    aligned_dims: bool = True
    #: How the scale is printed in the title block.
    scale_format: str = "ratio"         # ratio | worded | fraction


# --------------------------------------------------------------------------- #
# primitives (paper space, mm, origin lower-left of the sheet)
# --------------------------------------------------------------------------- #
@dataclass
class PLine:
    a: Tuple[float, float]
    b: Tuple[float, float]
    layer: str = "VISIBLE"


@dataclass
class PPoly:
    pts: List[Tuple[float, float]]
    layer: str = "VISIBLE"
    close: bool = False


@dataclass
class PFace:
    """A filled polygon: used to shade the isometric so it reads as solid."""
    pts: List[Tuple[float, float]]
    shade: float = 0.8          # 0 = black, 1 = white
    layer: str = "SHADE"


@dataclass
class PCircle:
    c: Tuple[float, float]
    r: float
    layer: str = "VISIBLE"


@dataclass
class PText:
    p: Tuple[float, float]
    s: str
    h: float
    layer: str = "TEXT"
    ha: str = "left"      # left | center | right
    va: str = "bottom"    # bottom | center | top
    rot: float = 0.0
    #: Width of one character in units of `h`, for the font this text will be
    #: rendered in. Carried on the primitive because the bounding box is
    #: computed from it -- a box measured with one font's ratio and drawn in
    #: another is simply wrong, and these boxes are the dataset's labels.
    char_w: float = 0.62
    #: Render this string in a heavy weight. Bold glyphs are ~1.06x wider, so
    #: the caller must widen char_w to match or the recorded box will be too
    #: narrow.
    bold: bool = False
    #: Family this text will be rendered in, stamped by PSheet.add alongside
    #: char_w. Needed to measure the string's true ink for a tight detection
    #: box: char_w alone is a mean over an alphabet and cannot say how wide
    #: THIS string is. None means "not stamped yet".
    font_family: Optional[str] = None


@dataclass
class PArrow:
    tip: Tuple[float, float]
    tail: Tuple[float, float]
    size: float
    layer: str = "DIM"
    #: filled triangle, open vee, 45-degree tick or a dot. A house style picks
    #: one and uses it for every terminator on the sheet.
    kind: str = "filled"


@dataclass
class PSheet:
    w: float
    h: float
    prims: List[object] = field(default_factory=list)
    scale: float = 1.0          # achieved drawing scale
    view_area: float = 0.0      # paper area occupied by the view block
    sheet_area: float = 1.0
    fill: float = 0.0           # view_area / sheet_area
    #: Pairs of annotation labels that touch. Readability, as opposed to
    #: "do the views fit" -- see render._count_label_clashes.
    label_clashes: int = 0
    #: Labels per square centimetre of the busiest view -- see
    #: render._label_density. This is the readability figure the sheet
    #: chooser reacts to.
    label_density: float = 0.0
    table_rows: int = 0
    dim_texts: List[tuple] = field(default_factory=list)
    placed_views: dict = field(default_factory=dict)  # view -> Placed
    iso_box: tuple = ()   # paper bbox of the isometric, if drawn
    # Every annotation actually drawn, captured as it is emitted: one dict per
    # dimension / callout / note with its exact rendered text and its paper
    # bounding box. The record used to store only integer counts per view, so
    # a drawing carrying four dimensions and a THICKNESS note serialised as
    # "dimension_count: 2, feature_boxes: []" and the annotation content was
    # unrecoverable without re-rendering the sheet.
    annotations: List[dict] = field(default_factory=list)
    #: Character-width factor of the font this sheet is drawn in. Stamped
    #: onto every PText by `add`, and used by render._text_box.
    char_w: float = 0.62
    #: Stroke-weight multiplier, applied by the exporters across all layers.
    stroke_scale: float = 1.0
    #: Font family name for the exporters.
    font_family: str = "DejaVu Sans"

    def note_annotation(self, **kw):
        self.annotations.append(kw)

    def add(self, *p):
        """Append primitives, stamping the sheet's font metrics onto text.

        Done centrally rather than at each of the ~22 PText call sites: the
        width factor is font-specific and the bounding boxes derived from it
        are the dataset's labels, so a single site that forgot to pass it
        would silently mislabel that annotation. Anything that set char_w
        explicitly is left alone.
        """
        for item in p:
            if isinstance(item, PText) and item.char_w == 0.62:
                item.char_w = self.char_w
            # The family is stamped unconditionally (unlike char_w, which has
            # a sentinel default that callers may have overridden on purpose):
            # a PText measured against the wrong family gets a wrong tight
            # box, and no call site sets it deliberately.
            if isinstance(item, PText) and item.font_family is None:
                item.font_family = getattr(self, "font_family", None)
        self.prims.extend(p)


def _num(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


def scale_text_styled(s: float, style: "Style") -> str:
    """The scale as this office writes it in the title block.

    Three real conventions: the bare ratio ``1:2``, the worded ``SCALE 1:2``
    and the fraction ``1/2``. The `scale_format` knob was set by every house
    style and read by nobody, which is the kind of thing that quietly turns a
    variation axis into dead weight.
    """
    txt = scale_text(s)
    kind = getattr(style, "scale_format", "ratio")
    if kind == "worded":
        return f"SCALE {txt}"
    if kind == "fraction":
        return txt.replace(":", "/")
    return txt


def scale_text(s: float) -> str:
    if s >= 1:
        if abs(s - round(s)) < 1e-9:
            return f"{int(round(s))}:1"
        return f"{_num(s)}:1"
    inv = 1 / s
    if abs(inv - round(inv)) < 1e-9:
        return f"1:{int(round(inv))}"
    return f"1:{_num(inv)}"


# --------------------------------------------------------------------------- #
# frame + title block
# --------------------------------------------------------------------------- #
#: Zone letters run bottom-to-top up both side margins (ISO 5457).
_ZONE_LETTERS = "ABCDEFGH"


def draw_frame(sh: PSheet, margin: float = 10.0, style: "Style" = None):
    """Sheet border, and the zone strip if the style calls for one.

    A zoned border is what an issued drawing carries: a ruled band around the
    border with numbers along the top and bottom and letters up the sides, so
    a revision note or a phone call can say "the chamfer in B3". Centering
    marks at the mid-edges (ISO 5457) let the sheet be folded and filed.
    """
    w, h = sh.w, sh.h
    kind = getattr(style, "frame_style", "plain") if style else "plain"
    x0, y0, x1, y1 = margin, margin, w - margin, h - margin

    if kind == "double":
        # an outer trim line as well as the border itself
        o = min(margin * 0.45, 4.0)
        sh.add(PPoly([(x0 - o, y0 - o), (x1 + o, y0 - o),
                      (x1 + o, y1 + o), (x0 - o, y1 + o)],
                     layer="FRAME", close=True))

    sh.add(PPoly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                 layer="FRAME", close=True))

    if kind != "zoned":
        return

    band = min(margin * 0.62, 6.0)
    cols = max(2, int(getattr(style, "zone_cols", 8)))
    rows = max(2, int(getattr(style, "zone_rows", 4)))
    th = band * 0.62
    # inner rule of the zone band, on all four sides
    sh.add(PPoly([(x0 + band, y0 + band), (x1 - band, y0 + band),
                  (x1 - band, y1 - band), (x0 + band, y1 - band)],
                 layer="FRAME", close=True))

    cw = (x1 - x0) / cols
    for i in range(cols):
        cx = x0 + (i + 0.5) * cw
        # numbers run right to left along the bottom, as ISO 5457 has them
        label = str(cols - i)
        for cy in (y0 + band * 0.5, y1 - band * 0.5):
            sh.add(PText((cx, cy - th * 0.42), label, th, ha="center",
                         layer="TEXT"))
        if i:
            xr = x0 + i * cw
            sh.add(PLine((xr, y0), (xr, y0 + band), layer="FRAME"),
                   PLine((xr, y1 - band), (xr, y1), layer="FRAME"))

    rh = (y1 - y0) / rows
    for j in range(rows):
        cy = y0 + (j + 0.5) * rh
        label = _ZONE_LETTERS[j % len(_ZONE_LETTERS)]
        for cx in (x0 + band * 0.5, x1 - band * 0.5):
            sh.add(PText((cx, cy - th * 0.42), label, th, ha="center",
                         layer="TEXT"))
        if j:
            yr = y0 + j * rh
            sh.add(PLine((x0, yr), (x0 + band, yr), layer="FRAME"),
                   PLine((x1 - band, yr), (x1, yr), layer="FRAME"))

    if getattr(style, "centering_marks", False):
        for (mx, my, dx, dy) in ((w / 2, y0, 0, -1), (w / 2, y1, 0, 1),
                                 (x0, h / 2, -1, 0), (x1, h / 2, 1, 0)):
            sh.add(PLine((mx, my), (mx + dx * margin, my + dy * margin),
                         layer="FRAME"))


def note_lines(notes: List[str], width: float, t: float,
               char_w: float) -> List[str]:
    """Numbered general notes, word-wrapped to ``width`` paper millimetres.

    A general note is free text of arbitrary length -- "2 SEPARATE BODIES IN
    THIS FILE - OVERALL DIMENSIONS ARE THE ENVELOPE; EACH BODY IS SIZED
    SEPARATELY" is 93 characters. Drawn as one unwrapped line at the title
    block's left edge it ran 56 mm past the right border of an A3 sheet and
    straight off the page, and the recorded detection box followed the ink
    out of the image. Wrapping is the fix: the box is not wrong, the drawing
    is.

    Continuation lines are indented to clear the "N. " numbering, so the
    wrapped text aligns under the first word rather than under the number.

    This is the single source of truth for what the notes block contains.
    ``draw_title_block`` draws exactly these strings and
    ``render._emit_general_notes`` measures exactly these strings, so the two
    cannot drift apart -- the same duplicated-constant trap that previously
    put the recorded boxes 13.7 mm and 26 mm off the ink.
    """
    out: List[str] = []
    # Characters that fit on one line. Guard against a degenerate width (a
    # very narrow title block with a very large font) by always allowing at
    # least one word per line, so this can never loop forever or return
    # empty.
    per_line = max(8, int(width / max(t * char_w, 1e-6)))
    for i, n in enumerate(notes):
        prefix = f"{i+1}. "
        indent = " " * len(prefix)
        words = str(n).split()
        if not words:
            out.append(prefix.rstrip())
            continue
        cur = prefix + words[0]
        for w in words[1:]:
            if len(cur) + 1 + len(w) <= per_line:
                cur += " " + w
            else:
                out.append(cur)
                cur = indent + w
        out.append(cur)
    return out


def title_block_origin(sh: PSheet, style: "Style", margin: float,
                       tb_w: float, tb_h: float):
    """Bottom-left corner of the title block, in sheet millimetres.

    Bottom-right is much the commoner, but a top corner is normal on portrait
    sheets and on plenty of landscape ones (the block runs along the top with
    the revision table under it). One helper, because the notes block, the
    parts list and the finish note all have to find the same corner -- three
    copies of this arithmetic is how they ended up on top of each other.
    """
    corner = getattr(style, "title_block_corner", "br")
    x = margin if corner in ("bl", "tl") else sh.w - margin - tb_w
    y = margin if corner in ("bl", "br") else sh.h - margin - tb_h
    return x, y


def draw_title_block(sh: PSheet, meta: Dict[str, str], margin: float = 10.0,
                     tb_w: float = 130.0, tb_h: float = 40.0, style: Style = Style()):
    # Which corner the block occupies is a real house-style difference, so it
    # follows the style. Bottom-right is much the commoner, and the weighting
    # in autodraft.variation reflects that.
    x0, y0 = title_block_origin(sh, style, margin, tb_w, tb_h)
    x1, y1 = x0 + tb_w, y0 + tb_h
    sh.add(PPoly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer="FRAME", close=True))
    rows = max(3, int(getattr(style, "title_block_rows", 4)))
    cols = max(2, int(getattr(style, "title_block_cols", 2)))
    rh = tb_h / rows

    # The block is a rows x cols grid. Both dimensions vary, so one drawing
    # carries eight cells and another eighteen -- previously it was always
    # the same eight fields in the same places, which is exactly the
    # incidental a detector would learn instead of reading the drawing.
    #
    # The title spans the full width of the top row, as it does on a real
    # block; every other field gets one cell.
    fields = meta.get("tb_fields") or {}
    order = list(meta.get("tb_order") or [])
    if not order:
        order = ["TITLE", "PART NUMBER", "MATERIAL", "SCALE", "UNITS",
                 "DRAWN", "DATE", "SHEET"]

    t = style.note_text_height * getattr(style, "title_text_scale", 1.0)
    label_h = t * 0.62
    bold = getattr(style, "title_block_bold", False)

    def cell(cx, cy, cw, label, value, big=False):
        sh.add(PText((cx + 1.5, cy + rh - label_h * 1.35), label, label_h,
                     layer="TEXT"))
        vh = t * (1.5 if big else 1.0)
        # A long value in a narrow cell must shrink rather than run past the
        # rule -- a varied grid makes some cells much narrower than others.
        room = cw - 3.0
        if value and len(value) * vh * sh.char_w > room:
            vh = max(label_h * 0.85, room / (len(value) * sh.char_w))
        sh.add(PText((cx + 1.5, cy + 1.4), value, vh, layer="TEXT",
                     bold=bold and big))

    # horizontal rules
    for i in range(1, rows):
        sh.add(PLine((x0, y0 + i * rh), (x1, y0 + i * rh), layer="FRAME"))

    # The top row is the title and is never subdivided.
    title_value = fields.get("TITLE", meta.get("title", ""))
    cell(x0, y0 + (rows - 1) * rh, tb_w, "TITLE", title_value, big=True)

    # Remaining fields fill the grid left-to-right, top-to-bottom.
    body = [f for f in order if f != "TITLE"]
    slots = (rows - 1) * cols
    cw = tb_w / cols
    for idx, name in enumerate(body[:slots]):
        r = (rows - 2) - idx // cols
        c = idx % cols
        if r < 0:
            break
        cx = x0 + c * cw
        cy = y0 + r * rh
        if c > 0:                      # vertical rule between columns
            sh.add(PLine((cx, cy), (cx, cy + rh), layer="FRAME"))
        value = fields.get(name)
        if value is None:
            value = {
                "PART NUMBER": meta.get("part_number", ""),
                "MATERIAL": meta.get("material", ""),
                "SCALE": meta.get("scale", ""),
                "UNITS": meta.get("units", "mm"),
                "DRAWN": meta.get("drawn", "AUTODRAFT"),
                "DATE": meta.get("date", ""),
                "SHEET": meta.get("sheet", "1/1"),
                "PROJECTION": str(getattr(style, "projection", "third")
                                  ).upper() + " ANGLE",
            }.get(name, "")
        cell(cx, cy, cw, name, str(value))

    # general notes, stacked above the title block (and above the parts list
    # when there is one -- the caller passes its height in meta)
    notes = meta.get("notes", [])
    heading = notes_heading(style)
    # ``notes_base`` is the baseline of the LOWEST note line; the heading sits
    # len(lines) pitches above it. The caller (render._draw_sheet_furniture)
    # is the single source of that number -- it already knows the parts-list
    # height and the title block corner. This function used to recompute it
    # for a top-corner block ("ny = y0 - 6 - (n+1)*t*1.6") while
    # render._emit_general_notes measured meta["notes_base"] unchanged, so
    # every notes box on a tl/tr sheet was recorded 18.5 mm above its ink
    # (measured on 0000_00000007). Never derive it twice.
    ny = float(meta.get("notes_base", y1 + 6))
    if notes and heading:
        # Wrap to the paper actually available to the right of the block's
        # left edge, so a long note can never run off the sheet.
        # nx, like ny, comes from the caller -- see the note above.
        nx = float(meta.get("notes_x", x0))
        lines = note_lines(notes, (sh.w - margin) - nx, t * 0.9, sh.char_w)
        sh.add(PText((nx, ny + len(lines) * t * 1.6), heading, t, layer="TEXT"))
        for i, ln in enumerate(lines):
            sh.add(PText((nx, ny + (len(lines) - 1 - i) * t * 1.6),
                         ln, t * 0.9, layer="TEXT"))


def notes_heading(style: "Style") -> str:
    """Heading over the general-note block, or "" when the office draws none.

    An ISO-style sheet heads a numbered list ``NOTES:``; a US-style sheet
    heads a tolerance block ``UNLESS OTHERWISE SPECIFIED:``. Both wordings are
    on real drawings and a reader recognises the block by them, so the style
    picks one and both the renderer and the label writer read it from here --
    two copies of this string would drift.
    """
    kind = getattr(style, "notes_style", "numbered")
    if kind == "none":
        return ""
    if kind == "uos":
        return "UNLESS OTHERWISE SPECIFIED:"
    return "NOTES:"


def draw_rev_table(sh: PSheet, x0: float, y1: float, w: float,
                   rows: List[List[str]], style: "Style"):
    """Revision history block, as issued drawings carry in a top corner.

    Columns are REV / DESCRIPTION / DATE / BY. The content is generated (a
    CAD file records no revision history) and flagged synthetic wherever it is
    reported, like everything else in the title block.
    """
    if not rows:
        return 0.0
    t = style.note_text_height * getattr(style, "title_text_scale", 1.0) * 0.8
    rh = max(t * 1.9, 4.0)
    cols = [w * 0.12, w * 0.52, w * 0.22, w * 0.14]
    n = len(rows) + 1
    y0 = y1 - n * rh
    sh.add(PPoly([(x0, y0), (x0 + w, y0), (x0 + w, y1), (x0, y1)],
                 layer="FRAME", close=True))
    for i in range(1, n):
        yy = y1 - i * rh
        sh.add(PLine((x0, yy), (x0 + w, yy), layer="FRAME"))
    cx = x0
    for cw in cols[:-1]:
        cx += cw
        sh.add(PLine((cx, y0), (cx, y1), layer="FRAME"))
    for r, row in enumerate([["REV", "DESCRIPTION", "DATE", "BY"]] + rows):
        cx = x0
        for cw, val in zip(cols, row):
            sh.add(PText((cx + 1.0, y1 - (r + 1) * rh + rh * 0.3), str(val), t,
                         layer="TEXT"))
            cx += cw
    return n * rh


def draw_parts_list(sh: PSheet, x0: float, y0: float, w: float,
                    rows: List[List[str]], style: "Style"):
    """Parts list sitting directly on top of the title block.

    Rows read bottom-up from item 1, which is the convention when the list
    grows upward out of the title block.
    """
    if not rows:
        return 0.0
    t = style.note_text_height * getattr(style, "title_text_scale", 1.0) * 0.8
    rh = max(t * 1.9, 4.2)
    cols = [w * 0.12, w * 0.14, w * 0.52, w * 0.22]
    n = len(rows) + 1
    y1 = y0 + n * rh
    sh.add(PPoly([(x0, y0), (x0 + w, y0), (x0 + w, y1), (x0, y1)],
                 layer="FRAME", close=True))
    for i in range(1, n):
        yy = y0 + i * rh
        sh.add(PLine((x0, yy), (x0 + w, yy), layer="FRAME"))
    cx = x0
    for cw in cols[:-1]:
        cx += cw
        sh.add(PLine((cx, y0), (cx, y1), layer="FRAME"))
    body = list(rows) + [["ITEM", "QTY", "DESCRIPTION", "PART NO"]]
    for r, row in enumerate(body):
        cx = x0
        for cw, val in zip(cols, row):
            sh.add(PText((cx + 1.0, y0 + r * rh + rh * 0.3), str(val), t,
                         layer="TEXT"))
            cx += cw
    return n * rh


def draw_table(sh: PSheet, x: float, y: float, header: List[str], rows: List[List[str]],
               col_w: List[float], row_h: float, style: Style, title: str = ""):
    """Draw a simple table with its top-left corner at (x, y)."""
    t = style.note_text_height * 0.9
    total_w = sum(col_w)
    n = len(rows) + 1
    if title:
        sh.add(PText((x, y + row_h * 0.4), title, t * 1.1, layer="TEXT"))
    top = y
    # Two real table conventions: fully ruled, or open with a rule under the
    # header only. Both are common on engineering drawings, and which one a
    # drawing uses is exactly the incidental a detector should not latch on
    # to, so it varies per part.
    ruled = getattr(style, "table_ruled", True)
    if ruled:
        for i in range(n + 1):
            yy = top - i * row_h
            sh.add(PLine((x, yy), (x + total_w, yy), layer="TABLE"))
        cx = x
        for w in col_w + [0]:
            sh.add(PLine((cx, top), (cx, top - n * row_h), layer="TABLE"))
            cx += w
    else:
        for yy in (top, top - row_h, top - n * row_h):
            sh.add(PLine((x, yy), (x + total_w, yy), layer="TABLE"))
    for j, htxt in enumerate(header):
        sh.add(PText((x + sum(col_w[:j]) + 1.2, top - row_h + row_h * 0.3), htxt, t, layer="TEXT"))
    for i, r in enumerate(rows, 1):
        for j, v in enumerate(r):
            sh.add(PText((x + sum(col_w[:j]) + 1.2, top - (i + 1) * row_h + row_h * 0.3),
                         str(v), t, layer="TEXT"))
    return total_w, n * row_h


def draw_table_paginated(sh: PSheet, x: float, y: float, header, rows,
                         col_w, row_h: float, style: Style, title: str = "",
                         max_h: float = 250.0, col_gap: float = 4.0,
                         max_cols: int = 3):
    """Draw a long table as several side-by-side column blocks.

    Returns (total_width, total_height) actually consumed.
    """
    if not rows:
        return 0.0, 0.0
    per_col = max(1, int(max_h / row_h) - 1)          # -1 for the header row
    n_blocks = min(max_cols, math.ceil(len(rows) / per_col))
    per_col = math.ceil(len(rows) / n_blocks)
    blocks = [rows[i:i + per_col] for i in range(0, len(rows), per_col)][:max_cols]

    tw = 0.0
    th = 0.0
    for bi, blk in enumerate(blocks):
        bx = x + bi * (sum(col_w) + col_gap)
        w, hh = draw_table(sh, bx, y, header, blk, col_w, row_h, style,
                           title=title if bi == 0 else "")
        tw = bi * (sum(col_w) + col_gap) + w
        th = max(th, hh)
    return tw, th
