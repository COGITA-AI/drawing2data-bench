"""Turn annotated projections into paper-space primitives on a sheet."""
from __future__ import annotations

import math
from datetime import date
from dataclasses import dataclass
from typing import Dict, Optional

from .annotate import (AngleDim, Annotated, BoltCircle, Callout, DatumFlag,
                       RadiusDim, FeatureFrame, LinearDim, Occupancy,
                       RoughnessMark, build_hole_table, summarise_hole_groups)
from .geometry import UNITS, PartInfo, fmt, to_display
from .projection import Projection
from .sheet import (PArrow, PCircle, PFace, PLine, PPoly, PSheet, PText, SHEETS, Style,
                    draw_frame, draw_parts_list, draw_rev_table,
                    scale_text_styled,
                    draw_table, draw_table_paginated,
                    draw_title_block)

#: Family assumed when a PText was never stamped with one (it was not added
#: through PSheet.add). Matches the exporters' own fallback.
_DEFAULT_FAMILY = "DejaVu Sans"

VIEW_LABEL = {"front": "FRONT", "top": "TOP", "right": "RIGHT SIDE",
              "left": "LEFT SIDE", "back": "BACK", "bottom": "BOTTOM", "iso": "ISOMETRIC"}


class Placed:
    def __init__(self, ann: Annotated, ox: float, oy: float, scale: float,
                 name: str = ""):
        self.ann, self.ox, self.oy, self.s = ann, ox, oy, scale
        self.name = name or getattr(ann.view, "name", "")

    def m2p(self, p):
        return (self.ox + p[0] * self.s, self.oy + p[1] * self.s)


GRID_THIRD = {"front": (0, 0), "top": (0, 1), "bottom": (0, -1),
              "right": (1, 0), "left": (-1, 0), "back": (2, 0)}
GRID_FIRST = {"front": (0, 0), "top": (0, -1), "bottom": (0, 1),
              "right": (-1, 0), "left": (1, 0), "back": (-2, 0)}

# Alternative arrangements tried when the strict projection grid wastes space.
# Projection convention is preserved in the first entry; the others trade
# strict alignment for a block aspect that fits the free area better.
def _layout_variants(order, style):
    grid = GRID_THIRD if style.projection == "third" else GRID_FIRST
    strict = {v: grid.get(v, (i, 0)) for i, v in enumerate(order)}
    variants = [strict]
    if len(order) >= 2:
        variants.append({v: (0, len(order) - 1 - i) for i, v in enumerate(order)})   # column
        variants.append({v: (i, 0) for i, v in enumerate(order)})                    # row
    if len(order) >= 3:                                                              # 2x2 block
        variants.append({order[0]: (0, 0), order[1]: (0, 1), order[2]: (1, 1)})
    return variants


def _emit_hole_tags(sh, placed, tags_by_view, style, margin, W, H, scale):
    """Letter each tagged hole, placed in the first free direction around it.

    Tags are tiny but numerous, so they are laid against a paper-space
    occupancy grid seeded with the geometry already drawn; dense patterns
    stay legible because no two tags can claim the same cells.
    """
    # Tags are tiny but numerous; place each one in the first free direction
    # around its hole so dense patterns stay legible.
    if tags_by_view:
        th = style.dim_text_height * 0.8
        occ_pap = Occupancy(margin, margin, W - margin, H - margin,
                            cell=max(1.2, th * 0.55))
        for pr in sh.prims:                    # geometry already on the sheet
            if isinstance(pr, PPoly):
                occ_pap.mark_poly(pr.pts)
            elif isinstance(pr, PLine):
                occ_pap.mark_line(pr.a, pr.b)
            elif isinstance(pr, PCircle):
                occ_pap.mark_rect(pr.c[0] - pr.r, pr.c[1] - pr.r,
                                  pr.c[0] + pr.r, pr.c[1] + pr.r)

        for v, tags in tags_by_view.items():
            pl = placed.get(v)
            if not pl:
                continue
            for tag, p, r in tags:
                pp = pl.m2p(p)
                rp = max(r * scale, 0.6)
                tw = len(tag) * th * 0.66
                best, best_cost = None, None
                for ang in (45, 315, 135, 225, 0, 90, 180, 270):
                    for ring in (1.0, 1.7, 2.6):
                        a = math.radians(ang)
                        off = rp + th * 0.5 * ring
                        cx = pp[0] + math.cos(a) * off
                        cy = pp[1] + math.sin(a) * off
                        x0 = cx if math.cos(a) >= 0 else cx - tw
                        cost = occ_pap.rect_cost(x0, cy - th * 0.5,
                                                 x0 + tw, cy + th * 0.5)
                        cost += ring * 2 + (0 if ang in (45, 315) else 1)
                        if best_cost is None or cost < best_cost:
                            best_cost, best = cost, (x0, cy, cx, cy, ang)
                x0, cy, cx, _cy2, ang = best
                occ_pap.mark_rect(x0, cy - th * 0.5, x0 + tw, cy + th * 0.5)
                sh.add(PText((x0, cy - th * 0.42), tag, th, layer="DIM"))


def _furniture_height(style, meta) -> float:
    """Paper the bottom-corner stack needs above the title block.

    parts list + note block + the general finish note. An estimate is enough:
    it only has to keep the view area off them, and over-reserving costs a
    little scale where under-reserving prints a dimension through a note.
    """
    t = style.note_text_height * getattr(style, "title_text_scale", 1.0)
    h = 0.0
    if getattr(style, "parts_list", False):
        h += 2 * max(t * 1.9 * 0.8, 4.2)
    notes_style = getattr(style, "notes_style", "numbered")
    if notes_style != "none":
        # heading + the notes themselves: 4 boilerplate lines for an UNLESS
        # OTHERWISE SPECIFIED block, 2-3 derived ones for a numbered list
        lines = 5 if notes_style == "uos" else 3
        h += (lines + 1) * t * 1.6
        if getattr(style, "roughness_general_note", True):
            # The symbol includes a tall ISO 1302 vee plus the Ra text and
            # ALL OVER tail. Reserve the complete footprint, not only the
            # one-line text height; otherwise the lowest edge of the note can
            # land on the top edge of a view on a top-corner title block.
            h += t * 4.5
    return h


@dataclass(frozen=True)
class _Page:
    """Fixed geometry of the sheet: size, margin and title-block reservation.

    Set once when the sheet is created and never changed. Passing this one
    object instead of five loose floats is what makes the layout steps
    extractable -- they were sharing eleven such values with build_sheet.
    """
    W: float
    H: float
    margin: float = 10.0
    tb_w: float = 130.0
    tb_h: float = 40.0
    gutter: float = 10.0
    #: Paper claimed along the TOP edge, when the title block lives up there.
    top_reserve: float = 0.0

    @property
    def ax0(self):
        """Left edge of the area available to views."""
        return self.margin + 5

    @property
    def ay0(self):
        """Bottom edge, clear of the title block."""
        if self.top_reserve:
            return self.margin + 8
        return self.margin + self.tb_h + 14

    @property
    def ax1(self):
        return self.W - self.margin - 5

    @property
    def ay1(self):
        return self.H - self.margin - 8 - self.top_reserve


def _place_isometric(pg, sh, style, iso_proj, projections, placed,
                     slots, cols, rows, col_x, row_y, s,
                     table_rows, summary_rows, table_w,
                     ax0, ay0, ax1, ay1):
    """Draw the isometric in the best free space, if there is any.

    ``ax0..ay1`` is the live drawing area, which is NOT ``pg.ax0..pg.ay1``:
    build_sheet shrinks the right edge when a hole table claims a strip,
    so the page defaults would let the isometric overlap the table.

    Preference order: an empty cell of the projection grid (free space,
    and conventionally top-right), otherwise the largest genuinely empty
    rectangle on the sheet. The placement is collision-checked against
    the padded view blocks, the notes and the title block -- a bare "use
    the top-right corner" fallback silently overlapped the views
    whenever the grid had no spare cell (e.g. a two-view drawing in a
    single column).
    """
    # Preference order: an empty cell of the projection grid (free space, and
    # conventionally top-right), otherwise the largest genuinely empty
    # rectangle on the sheet. The placement is collision-checked against the
    # padded view blocks, the notes and the title block -- a bare "use the
    # top-right corner" fallback silently overlapped the views whenever the
    # grid had no spare cell (e.g. a two-view drawing in a single column).
    if iso_proj is not None and iso_proj.width > 0:
        occupied = []
        for v, ann_v in projections.items():
            pl_v = placed.get(v)
            if pl_v is None:
                continue
            pv = ann_v.view
            pl_, pr_, pb_, pt_ = ann_v.pad
            a = pl_v.m2p((pv.xmin - pl_, pv.ymin - pb_))
            b = pl_v.m2p((pv.xmax + pr_, pv.ymax + pt_))
            occupied.append((min(a[0], b[0]), min(a[1], b[1]),
                             max(a[0], b[0]), max(a[1], b[1])))
        # notes + title block, bottom right
        occupied.append((pg.W - pg.margin - pg.tb_w - 6, pg.margin,
                         pg.W - pg.margin, pg.margin + pg.tb_h + 12 + 7 * style.note_text_height))
        if table_rows or summary_rows:
            occupied.append((pg.W - pg.margin - 4 - table_w, pg.margin,
                             pg.W - pg.margin, pg.H - pg.margin))

        def _free(x0, y0, x1, y1):
            for ox0, oy0, ox1, oy1 in occupied:
                if not (x1 <= ox0 or x0 >= ox1 or y1 <= oy0 or y0 >= oy1):
                    return False
            return True

        used_cells = set(slots.values())
        cand = []
        cs, rs = sorted(cols), sorted(rows)
        for r in reversed(rs):
            for c in reversed(cs):
                if (c, r) in used_cells:
                    continue
                cand.append((col_x[c], row_y[r],
                             (cols[c][1] - cols[c][0]) * s,
                             (rows[r][1] - rows[r][0]) * s))

        # Scan the sheet for empty rectangles. Free space is often a tall
        # narrow strip beside the views rather than a wide shallow one, so
        # candidates are generated over a range of *both* width and height
        # instead of a single fixed aspect -- a width-first sweep never finds
        # the strip and strands the isometric at a few percent of the sheet.
        band_w, band_h = ax1 - ax0, ay1 - ay0
        max_w = min(style.iso_max_mm, band_w * style.iso_max_frac)
        max_h = min(style.iso_max_mm * 1.2, band_h * 0.55)
        boxes = []
        w_try = max_w
        while w_try >= 28.0:
            h_try = max_h
            while h_try >= 22.0:
                boxes.append((w_try, h_try))
                h_try *= 0.75
            w_try *= 0.75
        # biggest usable drawing first, so the search keeps the best fit
        boxes.sort(key=lambda wh: -min(wh[0] / max(iso_proj.width, 1e-9),
                                       wh[1] / max(iso_proj.height, 1e-9)))
        # Test the rectangle the isometric will actually occupy, not the
        # candidate cell: the drawing is centred inside the cell, and a padded
        # view block can spill into that cell while leaving its centre free.
        # Keep the largest workable spot rather than the first one found.
        #
        # The drawn area depends only on the cell SHAPE, not on where the cell
        # sits, and `boxes` is already sorted largest-drawing-first. So once a
        # spot is found, every later shape is at best equal and can be skipped
        # whole -- and positions are generated lazily instead of building a
        # ~44k-entry list up front.
        spot, spot_area = None, 0.0
        pad_i = 6.0
        m_lbl = style.note_text_height * 2.2          # room for the ISO label
        for tw_, th_ in boxes:
            if tw_ < 22 or th_ < 18:
                continue
            t_s = min((tw_ - pad_i) / iso_proj.width,
                      (th_ - pad_i) / iso_proj.height, s * style.iso_max_zoom)
            if t_s <= 0:
                continue
            t_w, t_h = iso_proj.width * t_s, iso_proj.height * t_s
            area = t_w * t_h
            if area <= spot_area:
                continue          # this shape cannot beat what we already have
            step = max(6.0, min(tw_, th_) / 6)
            found = None
            y = ay1 - th_
            while y >= ay0 - 1e-9:
                x = ax1 - tw_
                while x >= ax0 - 1e-9:
                    t_x = x + (tw_ - t_w) / 2
                    t_y = y + (th_ - t_h) / 2
                    if _free(t_x - 2, t_y - m_lbl,
                             t_x + t_w + 2, t_y + t_h + 2):
                        found = (x, y, tw_, th_)
                        break
                    x -= step
                if found:
                    break
                y -= step
            if found:
                spot, spot_area = found, area
        if spot is not None:
            bx0, by0, bw, bh = spot
            pad_i = 6.0
            iso_s = min((bw - pad_i) / iso_proj.width,
                        (bh - pad_i) / iso_proj.height, s * style.iso_max_zoom)
            iso_s = max(iso_s, 1e-6)
            iw, ih = iso_proj.width * iso_s, iso_proj.height * iso_s
            ix = bx0 + (bw - iw) / 2
            iy = by0 + (bh - ih) / 2
            pl = Placed(Annotated(view=iso_proj), ix - iso_proj.xmin * iso_s,
                        iy - iso_proj.ymin * iso_s, iso_s)
            _emit_iso(sh, pl, style)
            cap = PText((ix + iw / 2, iy - 5.5), "ISO", style.note_text_height,
                        ha="center", layer="TEXT")
            sh.add(cap)
            cap_box = _text_box(cap, grow=0.0)
            sh.note_annotation(kind="view_caption", view="iso", text="ISO",
                               text_box=cap_box, extent=cap_box, attach=[])
            sh.iso_box = (ix, iy, ix + iw, iy + ih)


def build_sheet(info: PartInfo, projections: Dict[str, Annotated], *,
                sheet: str = "A3", style: Style = Style(), meta: Optional[dict] = None,
                fixed_scale: Optional[float] = None,
                iso_proj: Optional[Projection] = None,
                sections=None,
                hole_table_view=None,
                subparts=None,
                section_of=None,
                table_blocks_override: int = 0,
                slots_override=None) -> PSheet:
    """Place the views, tables and title block on a sheet.

    Layout strategy: reserve only what the side content (hole table, ISO view)
    actually needs, then fit the view grid into everything that is left. The
    scale is fitted to that real area, so a sparse drawing grows to fill the
    sheet instead of sitting small in a corner.
    """
    meta = dict(meta or {})
    W, H = SHEETS[sheet]
    # Sheet furniture follows the (possibly varied) style rather than fixed
    # constants, so the title block's size and the page margin differ between
    # drawings the way they do between real drawing offices.
    # A zoned border eats a band of paper: everything on the sheet has to sit
    # inside the zone strip, or the title block prints over the zone letters.
    frame_margin = style.sheet_margin
    band = (min(frame_margin * 0.62, 6.0)
            if getattr(style, "frame_style", "plain") == "zoned" else 0.0)
    # The title block is not the only thing along the bottom edge: the parts
    # list grows out of it, the note block grows out of that, and the general
    # finish note sits above the lot. Reserving only the block let a view's
    # dimension chain print through the notes. The stack is estimated from
    # the style, which is what decides how tall it is.
    furniture = _furniture_height(style, meta or {})
    top_furniture = (getattr(style, "title_block_corner", "br") in ("tl", "tr"))
    # the stack (block + parts list + notes + finish note) hangs downward from
    # a top block, so the whole of it has to come off the top of the area
    # The block has to FIT. style.title_block_w is a house dimension (up to
    # 170 mm, plus up to 12% of per-drawing jitter) and the paper can be an
    # A4 portrait 210 mm wide with a 13.4 mm margin and a 6 mm zone strip:
    # 181.5 mm of block into 171.2 mm of paper put the left edge of
    # 0000_00000386's title block at x=9.1, which is 4.3 mm OUTSIDE the
    # border and 10.3 mm outside the zone rule -- and took the general notes,
    # the finish note and the projection symbol out with it, because they all
    # hang off this corner.
    _tb_w = min(style.title_block_w, W - 2 * (frame_margin + band))
    pg = _Page(W, H, margin=frame_margin + band,
               top_reserve=(style.title_block_h + furniture + 10.0
                            if top_furniture else 0.0),
               tb_w=_tb_w,
               tb_h=(style.title_block_h if top_furniture
                     else style.title_block_h + furniture),
               gutter=round(10.0 * style.view_gap_scale, 2))
    sh = PSheet(W, H)
    # Font metrics travel with the sheet so `add` can stamp every PText and
    # the exporters know what to render in.
    sh.char_w = style.char_w
    sh.stroke_scale = style.stroke_scale
    sh.font_family = style.font_family
    margin, tb_w, gutter = pg.margin, pg.tb_w, pg.gutter
    tb_h = style.title_block_h          # the block itself, not the stack

    order = list(projections.keys())
    table_views = ([hole_table_view] if isinstance(hole_table_view, str)
                   else list(hole_table_view or []))

    ax0, ay0, ax1, ay1 = pg.ax0, pg.ay0, pg.ax1, pg.ay1

    # ---------------------------------------------------------------- table
    # Build it first so the space it needs is known, not guessed.
    row_h = style.table_row_h
    col_w = [11, 13, 15, 15, 44]
    header = ["TAG", "VIEW", "X", "Y", "DESCRIPTION"]
    table_rows, tags_by_view = [], {}
    summary_rows = []
    if table_views:
        table_rows, tags_by_view = build_hole_table(
            info, table_views, getattr(style, "symbols", False))
        # a table taller than the sheet is useless -> fall back to a QTY summary
        max_rows_on_sheet = int((H - 2 * margin - 30) / row_h) * style.table_max_cols
        if len(table_rows) > max_rows_on_sheet:
            summary_rows = summarise_hole_groups(
                info, table_views, getattr(style, "symbols", False))
            table_rows, tags_by_view = [], {}

    # A multi-body file gets a strip of per-item detail views along the
    # bottom. Reserved before the views are placed, like the table, so the
    # assembly views simply get less room rather than being drawn over.
    sub_h = 0.0
    subparts = list(subparts or [])
    if subparts:
        sub_h = min((ay1 - ay0) * 0.28, 62.0)
        ay0 += sub_h + 6

    table_side, table_w, table_h, n_blocks = _reserve_table_strip(
        style, table_rows, summary_rows, row_h, col_w, H, margin,
        ax0, ax1, table_blocks_override)
    if table_side == "right":
        ax1 -= table_w + 8 if (table_rows or summary_rows) else 0
    elif table_side in ("top", "tl", "tr"):
        # a corner table is a top band too: it has to be reserved, or the
        # isometric's free-space search happily places the pictorial under it
        ay1 -= table_h + 8 if (table_rows or summary_rows) else 0
    elif table_side == "bottom":
        ay0 += table_h + 8 if (table_rows or summary_rows) else 0

    # ------------------------------------------------------------- iso view
    # Give the ISO a real column only when there is width to spare.
    # The ISO is pinned to the top-right corner. Reserve a modest block there
    # so the views never run into it; the reservation is deliberately small
    # (and only applied when the sheet can spare it) so the dimensioned views
    # keep priority over decoration.
    if slots_override:
        slots = dict(slots_override)
    else:
        grid = GRID_THIRD if style.projection == "third" else GRID_FIRST
        slots = {v: grid.get(v, (i, 0)) for i, v in enumerate(order)}

    cols, rows = {}, {}
    for v, ann in projections.items():
        pl_, pr_, pb_, pt_ = ann.pad
        pv = ann.view
        c, r = slots[v]
        x0, x1 = pv.xmin - pl_, pv.xmax + pr_
        y0, y1 = pv.ymin - pb_, pv.ymax + pt_
        cols[c] = ((min(cols[c][0], x0), max(cols[c][1], x1))
                   if c in cols else (x0, x1))
        rows[r] = ((min(rows[r][0], y0), max(rows[r][1], y1))
                   if r in rows else (y0, y1))

    tot_w_model = sum(c[1] - c[0] for c in cols.values())
    tot_h_model = sum(r[1] - r[0] for r in rows.values())
    n_gut_x, n_gut_y = max(0, len(cols) - 1), max(0, len(rows) - 1)

    # ---- paper for the detail views ----------------------------------------
    # Reserving a column for them BEFORE the scale is solved was tried and
    # reverted: on 0000_00000413 it cost the drawing two rungs of the scale
    # ladder (1.25:1 -> 1:1.25), and at that size the annotator could no
    # longer place the GD&T frames at all -- the sheet lost four objects to
    # gain one view. A detail is drawn in paper that is genuinely spare, and
    # if there is none, the sheet is simply left without it.
    ax1_full = ax1

    avail_w = (ax1 - ax0) - n_gut_x * gutter
    avail_h = (ay1 - ay0) - n_gut_y * gutter
    need = min(avail_w / tot_w_model if tot_w_model else 1e9,
               avail_h / tot_h_model if tot_h_model else 1e9)
    # Always snap to a standard scale. The ladder is fine-grained enough that
    # the wasted area is small, and an arbitrary value like "32.9499:1" is not
    # a drawing scale anyone can read off a print.
    s = fixed_scale if fixed_scale else _snap_scale(need)

    span_w = tot_w_model * s + n_gut_x * gutter
    span_h = tot_h_model * s + n_gut_y * gutter
    # Spread the views over the paper they have. The grid used to be packed at
    # one fixed gutter and centred, so a sheet with room to spare showed a
    # tight cluster of views ringed by blank paper -- the thing that most
    # marks a drawing as machine-made. Leftover space is now shared out
    # BETWEEN the columns and rows (capped, so views never drift so far apart
    # that the projection stops reading as one drawing) and only the
    # remainder becomes margin. Third-angle alignment is untouched: every
    # column and row still moves as a unit.
    # The COMPACT size of the view block, before any of that spreading. This
    # is what sh.fill has to be measured from: the spreading below only moves
    # views apart, it adds no drawing, and measuring the spread block made a
    # half-empty A2 report the same fill as a well-used A3 -- which is how
    # 0000_00000007 ended up with two small views at opposite ends of a sheet
    # that passed the 0.34 fill bar.
    packed_w, packed_h = span_w, span_h
    x_free = max(0.0, (ax1 - ax0) - span_w)
    n_gaps_x = max(0, len(cols) - 1)
    extra_x = min(x_free / (n_gaps_x + 1), gutter * 2.2) if n_gaps_x else 0.0
    gutter_x = gutter + extra_x
    span_w += extra_x * n_gaps_x
    y_free_pre = max(0.0, (ay1 - ay0) - span_h)
    n_gaps_y = max(0, len(rows) - 1)
    extra_y = min(y_free_pre / (n_gaps_y + 1), gutter * 2.2) if n_gaps_y else 0.0
    gutter_y = gutter + extra_y
    span_h += extra_y * n_gaps_y

    x_start = ax0 + max(0.0, (ax1 - ax0 - span_w) * 0.5)
    # bias the view block toward the top of its band: the notes/title block sit
    # bottom-right, so leftover space reads better as a single margin than as
    # two half-gaps above and below
    y_free = max(0.0, (ay1 - ay0) - span_h)
    # Centre the block in the space it has, with a slight upward bias: the
    # notes and title block sit along the bottom, so leftover paper reads
    # better as one margin than as a band of nothing under the views. A 0.82
    # bias pushed the views into the top corner and left a third of the sheet
    # empty, which is the first thing that marks a drawing as machine-made.
    y_start = ay0 + y_free * (0.5 if y_free < 40 else 0.55)

    col_x, x = {}, x_start
    for c in sorted(cols):
        col_x[c] = x
        x += (cols[c][1] - cols[c][0]) * s + gutter_x
    row_y, y = {}, y_start
    for r in sorted(rows):
        row_y[r] = y
        y += (rows[r][1] - rows[r][0]) * s + gutter_y

    # ---- detail views: plan BEFORE anything is drawn -----------------------
    # The dimensions a detail view carries have to be left OFF the parent, so
    # the windows must be known before _emit_view runs. Previously the window
    # was picked afterwards, which is why the detail could only ever repeat
    # geometry the parent had already labelled.
    detail_plan, skip_dims = _plan_details(projections, style, s,
                                           bool(subparts), bool(section_of))

    placed: Dict[str, Placed] = {}
    for v, ann in projections.items():
        c, r = slots[v]
        placed[v] = Placed(ann, col_x[c] - cols[c][0] * s,
                           row_y[r] - rows[r][0] * s, s, name=v)
        sh.placed_views[v] = placed[v]
        _emit_view(sh, placed[v], style,
                   label=_view_label(v, order, style,
                                     projections[v].view),
                   skip=skip_dims)

    # the cutting-plane line goes on the PARENT view, not on the section.
    # One line per section, each lettered to match its view's caption.
    _cuts = list(sections or ([] if not section_of
                              else [(section_of[0], section_of[1], "A-A")]))
    for _pname, _sname, _sid in _cuts:
        if _pname in placed:
            _emit_cutting_plane(sh, placed[_pname], style, _sid.split("-")[0],
                                at_frac=(0.5 if _sid.startswith("A")
                                         else 0.32))

    _emit_hole_tags(sh, placed, tags_by_view, style, margin, W, H, s)

    _place_isometric(pg, sh, style, iso_proj, projections, placed,
                     slots, cols, rows, col_x, row_y, s,
                     table_rows, summary_rows, table_w,
                     ax0, ay0, ax1, ay1)

    if subparts:
        _draw_subparts(sh, style, subparts, ax0, ay0 - sub_h - 6, ax1, ay0 - 6)

    # ---- stock outline -----------------------------------------------------
    _emit_stock_outline(sh, style, info, placed)

    # ---- detail views ------------------------------------------------------
    # Each planned window is ringed on its parent view and redrawn enlarged in
    # free paper, carrying the dimensions that were withheld from the parent.
    from .detail import detail_geometry, detail_circles
    drawn_dims = set()
    for vname, letter, win, dims, crowded in detail_plan:
        pl = sh.placed_views.get(vname)
        if pl is None:
            continue
        # the reserved band lives between ax1 and ax1_full, so the search has
        # to cover the whole drawing area, not the shrunken view area
        spots = _free_rect(sh, style, ax0, ay0, ax1_full, ay1, margin,
                           many=True) or []
        if not spots:
            break
        polys = detail_geometry(pl.ann.view, win)
        circles = detail_circles(pl.ann.view, win)
        if not polys and not circles:
            continue
        for spot in spots:
            # A callout carried into the detail may stand anywhere inside the
            # DRAWING AREA that is still blank -- see _turn_into_bounds.
            # Bounding it by the frame is what keeps ink on the sheet; the
            # emptiness test is what keeps it off the other views.
            box = _draw_detail_view(sh, style, polys, circles, win, letter,
                                    *spot, s, pl.ann, dims,
                                    bounds=(ax0, ay0, ax1_full, ay1),
                                    drawn=drawn_dims,
                                    min_ratio=(1.2 if crowded else 2.0))
            if box is not None:
                _emit_detail(sh, pl, style, win, letter)
                break

    # A dimension is only withheld from its parent view when the detail that
    # was going to carry it actually got drawn. The plan is made before the
    # sheet is laid out, so a detail can still be dropped for want of free
    # paper -- and when that happened the dimension went with it and the
    # sheet simply lost a size ("4X R6" vanished from machined_block).
    # Put the orphans back where they came from.
    for vname, _letter, _win, dims, _crowded in detail_plan:
        pl = sh.placed_views.get(vname)
        if pl is None:
            continue
        for d in dims:
            if id(d) in drawn_dims:
                continue
            _draw_dim(sh, pl, d, style)

    # ---- draw the table in its reserved strip ------------------------------
    _draw_hole_table(sh, style, table_side, table_rows, summary_rows, header,
                     col_w, row_h, table_w, table_h, n_blocks,
                     W, H, margin, tb_h)

    # ---- frame, notes, title block ----------------------------------------
    draw_frame(sh, frame_margin, style)
    meta.setdefault("scale", scale_text_styled(s, style))
    meta.setdefault("units", UNITS)
    meta.setdefault("date", date.today().isoformat())
    notes = list(meta.get("notes", []))
    meta["_right_table"] = bool(table_side in ("right", "tr")
                                and (table_rows or summary_rows))
    _draw_sheet_furniture(sh, info, meta, notes, style, W, margin, tb_w, tb_h)

    # diagnostics used by the auto sheet-size search
    sh.scale = s
    sh.view_area = packed_w * packed_h
    sh.sheet_area = max(1.0, (W - 2 * margin) * (H - 2 * margin))
    sh.fill = sh.view_area / sh.sheet_area
    sh.table_rows = len(table_rows)
    # How crowded the finished sheet is: pairs of LABELS that touch. This is
    # the readability number the sheet chooser needs -- "the views fit" and
    # "the drawing can be read" are different questions, and a sheet can pass
    # the first while its callouts pile into each other (0000_00000093 had
    # eleven of them on one 27 mm view).
    sh.label_clashes = _count_label_clashes(sh)
    sh.label_density = _label_density(sh)
    return sh


def _label_density(sh: PSheet) -> float:
    """Labels per square centimetre of view, worst view on the sheet.

    Overlapping labels are rare -- the placer works hard to avoid them -- so
    counting collisions says almost nothing about whether a drawing can be
    read. What makes 0000_00000093 unreadable is eleven callouts ringing a
    27 x 25 mm view: 1.6 labels per square centimetre, every one of them
    correctly placed. Density is the number that catches it, and it falls as
    the drawing is scaled up, which is exactly what the sheet chooser needs.
    """
    per_view = {}
    for a in sh.annotations:
        if a.get("kind") in ("table", "feature"):
            continue
        v = a.get("view") or ""
        if v in sh.placed_views:
            per_view[v] = per_view.get(v, 0) + 1
    worst = 0.0
    for v, n in per_view.items():
        pl = sh.placed_views[v]
        pv = pl.ann.view
        area_cm2 = max(pv.width * pl.s, 1.0) * max(pv.height * pl.s, 1.0) / 100.0
        worst = max(worst, n / max(area_cm2, 1e-6))
    return round(worst, 3)


def _count_label_clashes(sh: PSheet, gap: float = 0.6) -> int:
    """Pairs of annotation labels closer than ``gap`` millimetres.

    Table objects are skipped: the title block's box legitimately contains
    every field inside it, and a parts list sits against it by design.
    """
    boxes = []
    for a in sh.annotations:
        if a.get("kind") == "table":
            continue
        b = a.get("text_box")
        if b:
            boxes.append(b)
    n = 0
    for i in range(len(boxes)):
        x0, y0, x1, y1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            u0, v0, u1, v1 = boxes[j]
            if (u0 - gap < x1 and x0 < u1 + gap
                    and v0 - gap < y1 and y0 < v1 + gap):
                n += 1
    return n


def _snap_scale(s: float) -> float:
    from .sheet import PREFERRED_SCALES
    best = None
    for n, d in PREFERRED_SCALES:
        v = n / d
        if v <= s and (best is None or v > best):
            best = v
    return best or s


# --------------------------------------------------------------------------- #
def _emit_iso(sh: PSheet, pl: Placed, style: Style):
    """Draw the isometric: shaded facets under the visible outline.

    Filling the facets is what makes the pictorial view read as a solid; the
    outline is still drawn on top so silhouette and feature edges stay crisp.
    """
    facets = getattr(pl.ann.view, "facets", None)
    if style.iso_shaded and facets:
        for pts2d, _depth, shade in facets:
            sh.add(PFace([pl.m2p(q) for q in pts2d], shade=shade))
        # Facets already convey the surfaces; overlaying the full wireframe
        # just puts every through-edge back on top of them. Keep the outline
        # on a light layer so the silhouette stays crisp without the clutter.
        for seg in pl.ann.view.segs + pl.ann.view.circles:
            if seg.kind != "visible":
                continue
            if seg.is_circle and seg.full:
                sh.add(PCircle(pl.m2p(seg.center), seg.radius * pl.s,
                               layer="ISOEDGE"))
            else:
                sh.add(PPoly([pl.m2p(q) for q in seg.pts], layer="ISOEDGE"))
        return
    _emit_geometry(sh, pl, style, hidden=False)


def _free_rect(sh: PSheet, style: Style, ax0, ay0, ax1, ay1, margin,
               want=0.16, many=False):
    """The largest genuinely empty rectangle in the drawing area, if any.

    Coarse on purpose: a 24x16 occupancy grid over the drawing area, scanning
    for the biggest empty block. Exact free-space decomposition is not worth
    it -- the caller only needs somewhere a detail view will not land on ink.
    """
    nx, ny = 24, 16
    cw, ch = (ax1 - ax0) / nx, (ay1 - ay0) / ny
    if cw <= 0 or ch <= 0:
        return [] if many else None
    grid = [[0] * nx for _ in range(ny)]

    def mark(x0, y0, x1, y1):
        i0 = max(0, int((x0 - ax0) / cw)); i1 = min(nx - 1, int((x1 - ax0) / cw))
        j0 = max(0, int((y0 - ay0) / ch)); j1 = min(ny - 1, int((y1 - ay0) / ch))
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                grid[j][i] = 1

    for p in sh.prims:
        if isinstance(p, PLine):
            mark(min(p.a[0], p.b[0]), min(p.a[1], p.b[1]),
                 max(p.a[0], p.b[0]), max(p.a[1], p.b[1]))
        elif isinstance(p, (PPoly, PFace)):
            xs = [q[0] for q in p.pts]; ys = [q[1] for q in p.pts]
            mark(min(xs), min(ys), max(xs), max(ys))
        elif isinstance(p, PCircle):
            mark(p.c[0] - p.r, p.c[1] - p.r, p.c[0] + p.r, p.c[1] + p.r)
        elif isinstance(p, PText):
            w = len(p.s) * p.h * p.char_w
            mark(p.p[0] - w, p.p[1] - p.h, p.p[0] + w, p.p[1] + p.h * 1.6)

    found = []
    for j in range(ny):
        for i in range(nx):
            if grid[j][i]:
                continue
            i1 = i
            while i1 + 1 < nx and not grid[j][i1 + 1]:
                i1 += 1
            j1 = j
            while j1 + 1 < ny and all(not grid[j1 + 1][k]
                                      for k in range(i, i1 + 1)):
                j1 += 1
            found.append(((i1 - i + 1) * (j1 - j + 1), i, j, i1, j1))
    if not found:
        return None
    found.sort(key=lambda f: -f[0])

    def rect(f):
        _a, i, j, i1, j1 = f
        return (ax0 + i * cw + 2, ay0 + j * ch + 2,
                ax0 + (i1 + 1) * cw - 2, ay0 + (j1 + 1) * ch - 2)

    if not many:
        area, i, j, i1, j1 = found[0]
        if area < nx * ny * want * 0.5:
            return None
        return rect(found[0])
    # Several candidates, because "largest area" is not the same question as
    # "will a 50 mm circle fit". A detail view needs a rectangle that is big
    # in its SHORT side; the widest strip on the sheet is often useless to
    # it. Offer the three biggest and the three squarest, best first.
    out, seen = [], set()
    by_side = sorted(found, key=lambda f: -min((f[3] - f[1] + 1) * cw,
                                               (f[4] - f[2] + 1) * ch))
    for f in list(by_side[:3]) + list(found[:3]):
        key = f[1:]
        if key in seen:
            continue
        seen.add(key)
        out.append(rect(f))
    return out


def _emit_detail(sh: PSheet, pl: Placed, style: Style, window, letter: str):
    """Ring the magnified region on the parent view and letter it."""
    cx, cy, r = window
    c = pl.m2p((cx, cy))
    sh.add(PCircle(c, r * pl.s, layer="CENTER"))
    t = PText((c[0] + r * pl.s * 0.75, c[1] + r * pl.s * 0.75), letter,
              style.dim_text_height, layer="TEXT")
    sh.add(t)
    box = _text_box(t, grow=0.0)
    sh.note_annotation(kind="view_caption", view=pl.name, text=letter,
                       text_box=box, extent=box, attach=[])


def _plan_details(projections, style: Style, scale: float,
                  has_subparts: bool, has_section: bool):
    """Pick the detail windows for a sheet and say which dims they take.

    Returns ``([(view_name, letter, window, dims)], skip_ids)``.

    Letters continue after the section's: a sheet with SECTION A-A whose
    detail was also called "A" reads as one thing pointing at another.
    Several details per sheet are normal on a real drawing and are allowed
    here -- the cap is two per view and three per sheet, above which the
    sheet stops being a drawing and becomes a contact sheet.
    """
    if has_subparts or not getattr(style, "detail_views", True):
        return [], set()
    from .detail import choose_windows
    letters = [c for c in "ABCDEFGH"]
    if has_section:
        letters = letters[1:]              # A-A is taken by the section
    cands = []
    for vname, ann in projections.items():
        if getattr(ann.view, "section_id", ""):
            continue                       # a section IS the close-up
        for w in choose_windows(ann, scale, max_n=2):
            cands.append((vname, w))
    # Two details per sheet, and the ones that CARRY something first. Three
    # bubbles over three holes of the same four-hole pattern is a contact
    # sheet, not a drawing -- and the "4 PLACES" note already says there are
    # four of them.
    cands.sort(key=lambda vw: (0 if vw[1][3] else 1, -len(vw[1][3])))
    plan, skip = [], set()
    for li, (vname, (cx, cy, r, dims, crowded)) in enumerate(cands[:2]):
        if li >= len(letters):
            break
        plan.append((vname, letters[li], (cx, cy, r), dims, crowded))
        skip.update(id(d) for d in dims)
    return plan, skip


def _rescaled_dim(d, cx, cy, shrink: float):
    """Copy of a dimension with its PAPER offsets preserved at detail scale.

    Leader lengths, text offsets and dimension-line offsets are stored in
    VIEW units, so drawing the same object at 4x pushes its label four times
    further from the feature -- straight off the detail and across whatever
    is next to it. ``shrink`` = parent_scale / detail_scale puts every such
    offset back where the draughtsman put it, in millimetres of paper.
    """
    import copy as _copy
    d2 = _copy.copy(d)
    # Every PLACEMENT length is in view units and so is magnified with the
    # geometry; every FEATURE length (RadiusDim.radius) is the part and must not
    # be. Missing AngleDim.radius here drew the 120 deg arc of 0000_00000413
    # at four times its size -- a 230 mm arc sweeping across the whole sheet.
    for name in ("leader_len", "offset"):
        if hasattr(d2, name) and isinstance(getattr(d2, name), (int, float)):
            setattr(d2, name, getattr(d2, name) * shrink)
    if isinstance(d2, AngleDim):
        d2.radius = d2.radius * shrink
    for name in ("tail",):
        p = getattr(d2, name, None)
        anc = getattr(d2, "anchor", None) or getattr(d2, "tip", None)
        if p and anc and len(p) == 2:
            setattr(d2, name, (anc[0] + (p[0] - anc[0]) * shrink,
                               anc[1] + (p[1] - anc[1]) * shrink))
    return d2


def _prim_box(p):
    """Paper-space bounding box of one primitive, or None."""
    if isinstance(p, PLine):
        return (min(p.a[0], p.b[0]), min(p.a[1], p.b[1]),
                max(p.a[0], p.b[0]), max(p.a[1], p.b[1]))
    if isinstance(p, (PPoly, PFace)):
        xs = [q[0] for q in p.pts]; ys = [q[1] for q in p.pts]
        return (min(xs), min(ys), max(xs), max(ys))
    if isinstance(p, PCircle):
        return (p.c[0] - p.r, p.c[1] - p.r, p.c[0] + p.r, p.c[1] + p.r)
    if isinstance(p, PText):
        w = len(p.s) * p.h * p.char_w
        x = p.p[0] - (w if p.ha == "right" else w / 2 if p.ha == "center" else 0)
        return (x, p.p[1] - p.h * 0.3, x + w, p.p[1] + p.h * 1.2)
    return None


def _seg_hits_box(a, b, box) -> bool:
    """Does segment a-b touch the axis-aligned rectangle ``box``?"""
    x0, y0, x1, y1 = box
    # trivial reject on the segment's own bounding box
    if max(a[0], b[0]) < x0 or min(a[0], b[0]) > x1:
        return False
    if max(a[1], b[1]) < y0 or min(a[1], b[1]) > y1:
        return False
    # an endpoint inside is a hit
    if (x0 <= a[0] <= x1 and y0 <= a[1] <= y1) or \
       (x0 <= b[0] <= x1 and y0 <= b[1] <= y1):
        return True
    # otherwise the segment must cross an edge: test the sign of the
    # cross product at the four corners
    dx, dy = b[0] - a[0], b[1] - a[1]
    signs = set()
    for cx, cy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        cross = dx * (cy - a[1]) - dy * (cx - a[0])
        signs.add(cross > 0)
    return len(signs) > 1


def _prim_hits_box(p, box) -> bool:
    """Does this primitive put ink inside ``box``?

    An OUTLINE is not a filled shape. The sheet border is a PPoly the size of
    the whole page, so testing bounding boxes made every square millimetre of
    paper look occupied -- 48 of 48 candidate positions rejected, which is
    why the flag notes drew nothing at all and the projection symbol kept
    being dropped. Polylines and lines are tested edge by edge; only faces
    (the shaded isometric) and text are solid.
    """
    if isinstance(p, PLine):
        return _seg_hits_box(p.a, p.b, box)
    if isinstance(p, PPoly):
        pts = list(p.pts)
        if getattr(p, "close", False) and len(pts) > 2:
            pts = pts + [pts[0]]
        return any(_seg_hits_box(u, v, box) for u, v in zip(pts, pts[1:]))
    b = _prim_box(p)
    if b is None:
        return False
    return not (b[2] < box[0] or b[0] > box[2]
                or b[3] < box[1] or b[1] > box[3])


def _ink_free(sh: PSheet, box, since: int = 0) -> bool:
    """True when nothing already drawn on the sheet puts ink in ``box``.

    ``since`` limits the test to the first N primitives, so a detail view can
    test against everything that was on the paper BEFORE it started drawing
    itself -- its own bubble and geometry are not obstacles to its own
    callout.
    """
    for p in (sh.prims[:since] if since else sh.prims):
        if _prim_hits_box(p, box):
            return False
    return True


def _callout_paper_box(d, pl, style):
    """Where a leader callout's text will land, in paper mm. None if unknown."""
    if not isinstance(d, (RadiusDim, Callout)):
        return None                  # no leader-and-shelf to predict
    ch = style.dim_text_height
    txt = getattr(d, "text", "") or ""
    tw = max(len(ln) for ln in txt.split("\n")) * ch * style.char_w
    th = len(txt.split("\n")) * ch * 1.35
    if isinstance(d, RadiusDim):
        a = math.radians(d.angle)
        knee = pl.m2p((d.center[0] + math.cos(a) * (d.radius + d.leader_len),
                       d.center[1] + math.sin(a) * (d.radius + d.leader_len)))
        sgn = 1 if math.cos(a) >= 0 else -1
        x_a, x_b = knee[0], knee[0] + sgn * tw
        return (min(x_a, x_b), knee[1] - ch, max(x_a, x_b), knee[1] + th)
    if isinstance(d, Callout):
        t = pl.m2p(d.tail)
        a = pl.m2p(d.anchor)
        sgn = 1 if t[0] >= a[0] else -1
        x_a, x_b = t[0], t[0] + sgn * tw
        return (min(x_a, x_b), t[1] - ch, max(x_a, x_b), t[1] + th)
    return None


def _kind_fits(d, pl, style, bounds) -> bool:
    """Rough test that a non-leader annotation lands inside ``bounds``."""
    bx0, by0, bx1, by1 = bounds
    pts = []
    for name in ("anchor", "tail", "vertex", "center"):
        p = getattr(d, name, None)
        if p and len(p) == 2:
            pts.append(pl.m2p(p))
    if not pts:
        return True
    pad = style.dim_text_height * 6.0        # frame / flag / arc, generously
    return all(bx0 + pad <= q[0] <= bx1 - pad and by0 + pad <= q[1] <= by1 - pad
               for q in pts)


def _turn_into_bounds(d, pl, style, bounds, avoid=(), sh=None,
                      since=0):
    """Point a detail-view callout somewhere that is still on the paper.

    A detail is drawn in whatever rectangle happens to be free, which is
    often against a border, and the leaders it carries were laid out for the
    parent view somewhere else entirely. On machined_block the C'BORE note
    came out ABOVE the top border -- ink outside the frame, which this corpus
    must never have. Try the leader the other way round (and at right angles)
    and keep the first direction that lands inside the drawing area; drop the
    note if none does, because half a note off the sheet is worse than none.
    """
    if bounds is None:
        return d
    bx0, by0, bx1, by1 = bounds

    def inside(box):
        if box is None:
            return False
        if not (box[0] >= bx0 and box[1] >= by0
                and box[2] <= bx1 and box[3] <= by1):
            return False
        # ...and clear of the view's own furniture -- the caption sits under
        # the bubble, and a note dropped on it printed "4X R6" through
        # "DETAIL B (2:1)" on machined_block.
        if not all(box[2] < a[0] or box[0] > a[2] or
                   box[3] < a[1] or box[1] > a[3] for a in avoid):
            return False
        # Anywhere on the sheet is allowed as long as it is EMPTY. Confining
        # the note to the detail's own free rectangle instead threw away
        # perfectly good placements and bounced the C'BORE note back onto the
        # parent view, where it landed on two other notes.
        return sh is None or _ink_free(sh, box, since)

    box = _callout_paper_box(d, pl, style)
    if box is None:
        # A kind with no leader to turn (a control frame, a datum flag, an
        # angular dimension): it either fits where it is or it stays on the
        # parent view.
        return d if _kind_fits(d, pl, style, bounds) else None
    if inside(box):
        return d
    import copy as _copy
    if isinstance(d, RadiusDim):
        for turn in (180, 90, 270, 45, 135, 225, 315):
            alt = _copy.copy(d)
            alt.angle = (d.angle + turn) % 360
            alt.tip = None               # recomputed from the new angle
            if inside(_callout_paper_box(alt, pl, style)):
                return alt
        return None
    # a plain leader note: mirror its tail about the anchor
    for sx, sy in ((-1, 1), (1, -1), (-1, -1)):
        alt = _copy.copy(d)
        alt.tail = (d.anchor[0] + (d.tail[0] - d.anchor[0]) * sx,
                    d.anchor[1] + (d.tail[1] - d.anchor[1]) * sy)
        if inside(_callout_paper_box(alt, pl, style)):
            return alt
    return None


def _draw_detail_view(sh: PSheet, style: Style, polys, circles, window, letter,
                      x0, y0, x1, y1, parent_scale, parent_ann, dims,
                      bounds=None, drawn=None, min_ratio=2.0):
    """The magnified view itself, in a free rectangle of the sheet.

    Scale is a real ratio against the sheet scale and is printed with the
    caption, as `DETAIL A (2:1)` -- a magnified view whose factor is not
    stated is unreadable.

    The dimensions listed in ``dims`` are drawn HERE rather than on the
    parent: that is the whole point of the view. They are re-offset by
    :func:`_rescaled_dim` so their labels sit the same few millimetres from
    the feature as they would anywhere else on the sheet.
    """
    cx, cy, r = window
    cap_h = style.note_text_height * 2.4
    # Room has to be left for the callouts, which stand outside the bubble.
    avail = min(x1 - x0, (y1 - y0) - cap_h) * (0.48 if dims else 0.5)
    # ...and a detail is a detail: filling every scrap of free paper with one
    # gave laser_panel a pair of bubbles 120 mm across, larger than the view
    # they came from. Real details are a fist-sized circle on the sheet.
    # A detail bubble is a fist-sized circle on any sheet: 0.16 of the short
    # side is 34 mm on A4 but 67 mm on A2, which stops being a detail and
    # starts being a second drawing.
    avail = min(avail, min(sh.w, sh.h) * 0.16, 30.0)
    if avail <= 0 or r <= 0:
        return None
    # The caption states the scale the detail is ACTUALLY drawn at, off the
    # standard ladder. Two bugs here before: the code rounded a ratio, then
    # quietly rescaled to fit and printed the rounded number anyway; and the
    # number it printed was the multiple of the SHEET scale, so a detail
    # drawn at 2 x 1.25:1 was captioned "2:1" when it is 2.5:1. A drawing
    # that lies about its own scale is worse than one with no detail view.
    #
    # ``min_ratio`` is 2 for a feature too small to letter -- magnifying it
    # is the whole point -- but only 1.2 for a crowded region, where the job
    # is to give the callouts a view of their own. A 1.5:1 close-up carrying
    # six annotations that were fighting each other at 1.25:1 is a better
    # drawing than no close-up at all, and the next standard scale up is
    # often all the paper allows.
    from .sheet import PREFERRED_SCALES, scale_text_styled
    ladder = sorted({n / d for n, d in PREFERRED_SCALES}, reverse=True)
    sc = None
    # Two passes: the magnification this kind of detail wants, then any
    # enlargement at all. A 1.5x close-up that fits the paper beats a 2x one
    # that does not get drawn -- and dropping it is not free, because the
    # dimensions it was going to carry have already been taken off the
    # parent view (on machined_block that put the C'BORE note back on top of
    # the roughness callout it had been moved away from).
    for floor in (min_ratio, 1.2):
        for cand in ladder:
            if cand < parent_scale * floor - 1e-9:
                continue
            if r * cand <= avail:
                sc = cand
                break
        if sc is not None:
            break
    if sc is None:
        return None                 # no room to draw it here
    ox = (x0 + x1) / 2
    oy = (y0 + y1) / 2 + cap_h * 0.3
    before = len(sh.prims)          # everything already on the paper
    for poly in polys:
        pts = [(ox + (q[0] - cx) * sc, oy + (q[1] - cy) * sc) for q in poly]
        if len(pts) > 1:
            sh.add(PPoly(pts, layer="VISIBLE"))
    for c, rad in circles:
        sh.add(PCircle((ox + (c[0] - cx) * sc, oy + (c[1] - cy) * sc),
                       rad * sc, layer="VISIBLE"))
        _center_mark(sh, (ox + (c[0] - cx) * sc, oy + (c[1] - cy) * sc),
                     max(rad * sc * 1.35, 2.0), style)
    sh.add(PCircle((ox, oy), r * sc, layer="CENTER"))

    # the dimensions this view was created to make room for
    if dims:
        dpl = Placed(parent_ann, ox - cx * sc, oy - cy * sc, sc,
                     name=f"detail_{letter}")
        shrink = parent_scale / sc if sc else 1.0
        cap_y = oy - r * sc - cap_h * 0.7
        cap_avoid = [(x0, cap_y - cap_h * 0.4, x1, cap_y + cap_h * 0.9)]
        for d in dims:
            d2 = _rescaled_dim(d, cx, cy, shrink)
            d2 = _turn_into_bounds(d2, dpl, style, bounds, cap_avoid,
                                   sh=sh, since=before)
            if d2 is None:
                # No direction leaves this note inside the bubble's free
                # paper. Leave it for the caller to draw on the parent view
                # -- silently dropping it lost "4X R6" from machined_block
                # altogether.
                continue
            if drawn is not None:
                drawn.add(id(d))
            _draw_dim(sh, dpl, d2, style)

    label = f"DETAIL {letter} ({scale_text_styled(sc, style)})"
    cap = PText((ox, oy - r * sc - cap_h * 0.7), label,
                style.note_text_height, ha="center", layer="TEXT")
    sh.add(cap)
    box = _text_box(cap, grow=0.0)
    sh.note_annotation(kind="view_caption", view=f"detail_{letter}",
                       text=label, text_box=box, extent=box, attach=[])
    return (ox - r * sc, oy - r * sc - cap_h, ox + r * sc, oy + r * sc)


def _emit_hatch(sh: PSheet, pl: Placed, style: Style):
    """45-degree hatching over the cut faces of a section view."""
    for a, b in getattr(pl.ann.view, "hatch", []) or []:
        sh.add(PLine(pl.m2p(a), pl.m2p(b), layer="HATCH"))


def _emit_cutting_plane(sh: PSheet, pl: Placed, style: Style, letter: str,
                        at_frac: float = 0.5):
    """The cutting-plane line on the PARENT view: chain line, arrows, letters.

    A section is only readable if the reader can see where it was taken. The
    line runs across the parent view at the cut, with arrows at both ends
    pointing the way the section is viewed and the identifying letter at each
    end.
    """
    v = pl.ann.view
    pad_l, pad_r, pad_b, _pad_t = pl.ann.pad
    y = v.ymin + (v.ymax - v.ymin) * at_frac
    # The line overhangs the view a little, as it does on a real drawing --
    # but only into space the layout already reserved for this view, or the
    # overhang prints outside the border frame.
    ext = (v.xmax - v.xmin) * 0.08
    x0 = v.xmin - min(ext, max(0.0, pad_l * 0.7))
    x1 = v.xmax + min(ext, max(0.0, pad_r * 0.7))
    a, b = pl.m2p((x0, y)), pl.m2p((x1, y))
    sh.add(PLine(a, b, layer="CENTER"))
    arrow = style.arrow * 1.4
    drop = min(arrow * 2.2, max(1.0, pad_b * pl.s * 0.5))
    for end, inward in ((a, 1), (b, -1)):
        tip = (end[0], end[1] - drop)
        sh.add(PLine(end, tip, layer="CENTER"),
               PArrow(tip, (tip[0], tip[1] + max(arrow, drop * 0.5)), arrow,
                      kind=getattr(style, "arrow_style", "filled")))
        t = PText((end[0] + inward * style.dim_text_height * 0.8,
                   end[1] + style.dim_text_height * 0.4), letter,
                  style.dim_text_height, ha="center", layer="TEXT")
        sh.add(t)
        box = _text_box(t, grow=0.0)
        sh.note_annotation(kind="view_caption", view=pl.name, text=letter,
                           text_box=box, extent=box, attach=[])


def _emit_geometry(sh: PSheet, pl: Placed, style: Style, hidden=True):
    v = pl.ann.view
    for seg in v.segs + v.circles:
        if seg.kind == "hidden" and not (hidden and style.hidden_lines):
            continue
        layer = "HIDDEN" if seg.kind == "hidden" else "VISIBLE"
        if seg.is_circle and seg.full and seg.kind == "visible":
            sh.add(PCircle(pl.m2p(seg.center), seg.radius * pl.s, layer=layer))
        else:
            sh.add(PPoly([pl.m2p(p) for p in seg.pts], layer=layer))


#: Captions in the wording different offices use. "letters" is the
#: VIEW A / VIEW B convention that goes with an arrow on the parent view;
#: "building" is the plan/elevation wording used on fabrication drawings.
_VIEW_WORDS = {
    "building": {"top": "PLAN", "front": "ELEVATION", "right": "END VIEW",
                 "left": "END VIEW", "back": "REAR ELEVATION",
                 "bottom": "UNDERSIDE", "iso": "PICTORIAL"},
}


def _view_label(name: str, order, style: Style, proj=None) -> str:
    """Caption under a view, in this office's wording."""
    sec = getattr(proj, "section_id", "") if proj is not None else ""
    if sec:
        return f"SECTION {sec}"
    kind = getattr(style, "view_label_style", "ortho")
    if kind == "none":
        return ""
    if kind == "letters":
        if name == "iso":
            return "ISOMETRIC"
        idx = list(order).index(name) if name in order else 0
        return f"VIEW {chr(ord('A') + idx)}"
    if kind == "building":
        return _VIEW_WORDS["building"].get(name, name.upper())
    return VIEW_LABEL.get(name, name.upper())


def _draw_dim(sh: PSheet, pl: Placed, d, style: Style):
    """Draw one annotation object, whatever kind it is.

    One dispatcher, because the same list is now drawn in three places: on
    its own view, inside a detail view that took it off that view, and back
    on the parent when the detail could not fit it. Three copies of this
    if-chain is how a kind ends up drawn in one place and silently skipped in
    another -- the datum flags and control frames of 0000_00000413 could not
    go into a detail view at all for exactly that reason.
    """
    if isinstance(d, LinearDim):
        _linear(sh, pl, d, style)
    elif isinstance(d, RadiusDim):
        _diameter(sh, pl, d, style)
    elif isinstance(d, Callout):
        _note(sh, pl, d, style)
    elif isinstance(d, AngleDim):
        _angle(sh, pl, d, style)
    elif isinstance(d, FeatureFrame):
        _feature_frame(sh, pl, d, style)
    elif isinstance(d, DatumFlag):
        _datum_flag(sh, pl, d, style)
    elif isinstance(d, RoughnessMark):
        _roughness(sh, pl, d, style)
    elif isinstance(d, BoltCircle):
        sh.add(PCircle(pl.m2p(d.center), d.radius * pl.s, layer="CENTER"))


def _emit_view(sh: PSheet, pl: Placed, style: Style, label: str,
               skip: Optional[set] = None):
    # hatch first: it is background, and the outline is drawn over it
    _emit_hatch(sh, pl, style)
    _emit_geometry(sh, pl, style)
    ann = pl.ann
    for x0, y0, x1, y1, _lbl, _n in ann.feature_boxes:
        # The paper-space box is recorded whether or not it is drawn: it
        # describes recognised geometry, and --feature-boxes only controls
        # whether that rectangle is inked onto the sheet.
        a0 = pl.m2p((x0, y0))
        b0 = pl.m2p((x1, y1))
        sh.note_annotation(kind="feature", view=pl.name, text=_lbl, qty=_n,
                           text_box=None,
                           extent=(a0[0], a0[1], b0[0], b0[1]),
                           attach=[])
        if style.feature_boxes:
            # dashed rectangle around each recognised feature group
            m = style.feature_box_margin
            a = pl.m2p((x0 - m, y0 - m))
            b = pl.m2p((x1 + m, y1 + m))
            sh.add(PPoly([(a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1])],
                         layer="FEATURE", close=True))
    for m in ann.marks:
        _center_mark(sh, pl.m2p(m.center), max(m.radius * pl.s * 1.35, 2.0),
                     style)
    for d in ann.dims:
        # A dimension handed to a detail view is NOT drawn here: repeating it
        # on the parent at a scale where it does not fit is the crowding the
        # detail exists to relieve.
        if skip and id(d) in skip:
            continue
        _draw_dim(sh, pl, d, style)
    v = ann.view
    cx = pl.ox + (v.xmin + v.width / 2) * pl.s
    y_bottom = pl.oy + (v.ymin - ann.pad[2]) * pl.s
    if label:
        cap = PText((cx, y_bottom + 1.0), label, style.note_text_height,
                    ha="center", layer="TEXT")
        sh.add(cap)
        cbox = _text_box(cap, grow=0.0)
        sh.note_annotation(kind="view_caption", view=pl.name, text=label,
                           text_box=cbox, extent=cbox, attach=[])



def _union_box(boxes):
    """Union of paper-space boxes, ignoring empty ones."""
    bs = [b for b in boxes if b]
    if not bs:
        return None
    return (min(b[0] for b in bs), min(b[1] for b in bs),
            max(b[2] for b in bs), max(b[3] for b in bs))


#: Safety margin on an exported text box, as a fraction of the text height.
#:
#: The tight box is matplotlib's layout box intersected with the glyph
#: outlines, which is the right THING to measure but leaves nothing for how
#: the string is finally rasterised: outlines are filled with antialiasing,
#: the exporter's fontsize is rounded to points at the output dpi, and the
#: two disagree by a fraction of a percent per character. Measured on
#: 0000_00000126 at 200 dpi, ink escaped the recorded box by up to 0.12 mm on
#: the left of every string -- one pixel, and visibly clipping the first and
#: last glyph when zoomed in, which is what "the boxes do not contain the
#: whole text" looks like.
#:
#: 0.06 of the text height is 0.21 mm at h=3.5 -- comfortably more than the
#: worst measured escape, and small enough that the box still reads as tight.
_GLYPH_PAD = 0.06


def _text_box(t: PText, grow: float = 0.6, tight: bool = True):
    """Paper-space bounding box of a PText, matching how it is rendered.

    Two different jobs share this function.

    ``tight=False`` returns the nominal RESERVATION -- ``len * height *
    char_w``, anchored by treating ``ha``/``va`` as if they aligned the ink.
    That is what the occupancy solver and the witness-line breaker want:
    generous, and cheap to compute.

    ``tight=True`` returns the box a detection label needs, and asks
    matplotlib where it will actually draw the string
    (:mod:`autodraft.textmetrics`). This matters because ``ha``/``va`` do not
    align the ink: ``va="bottom"`` aligns the DESCENT LINE, so the nominal
    anchoring put every box a full descent (0.76 mm at h=3.5) BELOW its text
    -- visible as boxes sitting under their labels -- and ``ha="center"`` and
    ``"right"`` align the ADVANCE width, side bearings included. Together
    with ``char_w`` being an alphabet mean and the exporter drawing an em of
    ``0.9172 * h``, that left boxes both loose and offset.

    Falls back to the nominal estimate whenever the font cannot be measured,
    so a missing font cache costs tightness and never correctness.
    """
    cw = getattr(t, "char_w", 0.62)
    if not tight:
        # A RESERVATION is deliberately generous, and it must also cover what
        # an external checker measures: qa.py and any drawing-office rule of
        # thumb use ~0.62 em per character, so a narrow font must not shrink
        # the reserved corridor below that. A leader was left crossing a
        # callout label by exactly this margin.
        cw = max(cw, 0.62)
    if getattr(t, "bold", False):
        # Bold glyphs are wider; the box must follow or it under-reports the
        # extent of exactly the strings that are most prominent.
        cw *= 1.06
    fam = getattr(t, "font_family", None) or _DEFAULT_FAMILY

    if tight:
        from .textmetrics import drawn_box
        d = drawn_box(t.s, t.h, fam, ha=t.ha, va=t.va, rot=t.rot,
                      bold=getattr(t, "bold", False))
        if d is not None:
            x, y = t.p
            g = grow + _GLYPH_PAD * max(t.h, 1e-6)
            return (x + d[0] - g, y + d[1] - g,
                    x + d[2] + g, y + d[3] + g)

    # Nominal reservation (and the fallback when the font is unmeasurable).
    w = len(t.s) * t.h * cw
    x, y = t.p
    if round(t.rot) % 360 == 90:
        if t.ha == "center":  y0 = y - w / 2
        elif t.ha == "right": y0 = y - w
        else:                 y0 = y
        x0 = x - t.h
        if t.va == "center":  x0 = x - t.h / 2
        elif t.va == "top":   x0 = x
        box = (x0, y0, x0 + t.h, y0 + w)
    else:
        if t.ha == "center":  x -= w / 2
        elif t.ha == "right": x -= w
        y0 = y
        if t.va == "center":  y0 = y - t.h / 2
        elif t.va == "top":   y0 = y - t.h
        box = (x, y0, x + w, y0 + t.h)
    return (box[0] - grow, box[1] - grow, box[2] + grow, box[3] + grow)


def _break_lines_at_text(sh: PSheet):
    """Gap extension/dimension lines where they run through dimension text.

    Breaking a witness line at the text is the standard drafting fix for a
    number that has been shifted out over a neighbouring line; it keeps the
    drawing readable without moving the dimension.
    """
    boxes = sh.dim_texts
    if not boxes:
        return
    out = []
    for p in sh.prims:
        # CENTER carries the cutting-plane line, which runs the width of the
        # view and so can pass straight through a callout: it gets the same
        # break treatment as a witness line.
        if not isinstance(p, PLine) or p.layer not in ("EXT", "DIM", "CENTER",
                                                       "LEADER"):
            out.append(p)
            continue
        segs = [(p.a, p.b)]
        for bx in boxes:
            nxt = []
            for a, b in segs:
                nxt.extend(_clip_seg(a, b, bx))
            segs = nxt
            if not segs:
                break
        for a, b in segs:
            if math.dist(a, b) > 0.4:
                out.append(PLine(a, b, layer=p.layer))
    sh.prims = out


def _clip_seg(a, b, box):
    """Return the parts of segment a-b that lie outside an axis-aligned box."""
    x0, y0, x1, y1 = box
    inside = lambda pt: x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1
    if not inside(a) and not inside(b):
        # may still pass straight through: find entry/exit by sampling
        ts = []
        n = 64
        prev = inside(a)
        for i in range(1, n + 1):
            t = i / n
            pt = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            cur = inside(pt)
            if cur != prev:
                ts.append(t)
                prev = cur
        if len(ts) < 2:
            return [(a, b)]
        t0, t1 = ts[0], ts[-1]
        p0 = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
        p1 = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
        return [(a, p0), (p1, b)]
    if inside(a) and inside(b):
        return []
    # one endpoint inside: trim back to the boundary
    lo, hi = 0.0, 1.0
    a_in = inside(a)
    for _ in range(40):
        mid = (lo + hi) / 2
        pt = (a[0] + (b[0] - a[0]) * mid, a[1] + (b[1] - a[1]) * mid)
        if inside(pt) == a_in:
            lo = mid
        else:
            hi = mid
    cut = (a[0] + (b[0] - a[0]) * lo, a[1] + (b[1] - a[1]) * lo)
    return [(cut, b)] if a_in else [(a, cut)]


def _angle(sh: PSheet, pl: Placed, d: AngleDim, style: Style):
    """Arc with arrowheads between two edges, plus the angle text."""
    c = pl.m2p(d.vertex)
    r = d.radius * pl.s
    a0 = math.radians(d.start_ang)
    # sweep the short way round, which is the included angle
    delta = math.radians((d.end_ang - d.start_ang + 540.0) % 360.0 - 180.0)
    n = max(8, int(abs(math.degrees(delta)) / 4))
    pts = [(c[0] + math.cos(a0 + delta * i / n) * r,
            c[1] + math.sin(a0 + delta * i / n) * r) for i in range(n + 1)]
    sh.add(PPoly(pts, layer="DIM"))
    # extension lines from the corner out past the arc
    for ang in (d.start_ang, d.end_ang):
        a = math.radians(ang)
        sh.add(PLine((c[0] + math.cos(a) * r * 0.25,
                      c[1] + math.sin(a) * r * 0.25),
                     (c[0] + math.cos(a) * (r + style.arrow * 1.4),
                      c[1] + math.sin(a) * (r + style.arrow * 1.4)),
                     layer="EXT"))
    # arrowheads tangent to the arc at each end
    for p_end, sgn in ((pts[0], 1), (pts[-1], -1)):
        ang = math.atan2(p_end[1] - c[1], p_end[0] - c[0])
        tang = ang + sgn * math.copysign(math.pi / 2, delta)
        tail = (p_end[0] + math.cos(tang) * style.arrow * 2,
                p_end[1] + math.sin(tang) * style.arrow * 2)
        sh.add(PArrow(p_end, tail, style.arrow))
    mid = a0 + delta / 2
    th = style.dim_text_height
    tp = (c[0] + math.cos(mid) * (r + th * 0.9),
          c[1] + math.sin(mid) * (r + th * 0.9))
    te = PText(tp, d.text, th, ha="center", va="center", layer="DIM")
    sh.add(te)
    # Measured by the shared helper, not by a local copy of the same
    # arithmetic. The hand-rolled version here used the nominal
    # len*height*char_w estimate and so stayed loose (up to 1.52 mm of blank
    # paper around "139.4deg") after every other box was tightened -- the
    # duplicated-constant trap again.
    tb = _text_box(te, grow=0.0)
    inc = abs((d.end_ang - d.start_ang + 540.0) % 360.0 - 180.0)
    sh.note_annotation(
        kind="dimension", view=pl.name, text=d.text, value=round(inc, 4),
        detail={"form": "angular"},
        text_box=tb,
        extent=_union_box([tb, (min(p[0] for p in pts), min(p[1] for p in pts),
                                max(p[0] for p in pts), max(p[1] for p in pts))]),
        attach=[c])


def _center_mark(sh: PSheet, c, r, style: Style = None):
    """Centre mark on a bore.

    Offices differ: some draw a short cross inside the bore, others run full
    centre lines out past it, and a stripped-down CAD default draws neither.
    """
    kind = getattr(style, "center_style", "line") if style else "line"
    if kind == "none":
        return
    x, y = c
    reach = r if kind == "mark" else r * 1.6
    sh.add(PLine((x - reach, y), (x + reach, y), layer="CENTER"),
           PLine((x, y - reach), (x, y + reach), layer="CENTER"))


def _dim_line(sh: PSheet, a, b, gap_at=None, gap_len=0.0):
    """The dimension line, optionally broken where the number sits on it.

    A number written *in* the line rather than above it is standard US
    practice; the line has to open up for it or the text sits on the ink.
    """
    if gap_at is None or gap_len <= 0:
        sh.add(PLine(a, b, layer="DIM"))
        return
    L = math.dist(a, b) or 1.0
    ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    t = ((gap_at[0] - a[0]) * ux + (gap_at[1] - a[1]) * uy)
    h = gap_len / 2.0
    t0, t1 = max(0.0, t - h), min(L, t + h)
    if t0 > 0.02:
        sh.add(PLine(a, (a[0] + ux * t0, a[1] + uy * t0), layer="DIM"))
    if t1 < L - 0.02:
        sh.add(PLine((a[0] + ux * t1, a[1] + uy * t1), b, layer="DIM"))


def _aligned(sh: PSheet, pl: Placed, d: LinearDim, style: Style):
    """A dimension parallel to a sloping edge, stating its true length.

    Everything is built in the edge's own frame: the dimension line is the
    edge pushed out along its normal, the witness lines run from the edge's
    ends out to it, and the number sits above the line rotated to match --
    or horizontal, if the office reads every number that way.
    """
    import math as _m
    a3, b3 = pl.m2p(d.p1), pl.m2p(d.p2)
    L = _m.dist(a3, b3)
    if L <= 1e-9:
        return
    ux, uy = (b3[0] - a3[0]) / L, (b3[1] - a3[1]) / L
    nx, ny = -uy, ux
    # keep the dimension on the outside of the view, as the placer decided
    cx = pl.ox + (pl.ann.view.xmin + pl.ann.view.width / 2) * pl.s
    cy = pl.oy + (pl.ann.view.ymin + pl.ann.view.height / 2) * pl.s
    mid3 = ((a3[0] + b3[0]) / 2, (a3[1] + b3[1]) / 2)
    if (mid3[0] - cx) * nx + (mid3[1] - cy) * ny < 0:
        nx, ny = -nx, -ny
    off = abs(d.offset) * pl.s
    a = (a3[0] + nx * off, a3[1] + ny * off)
    b = (b3[0] + nx * off, b3[1] + ny * off)

    th = style.dim_text_height
    tw = len(d.text) * th * style.char_w
    ak = getattr(style, "arrow_style", "filled")
    gap_g, over = style.witness_gap, style.witness_overshoot
    for base, tip in ((a3, a), (b3, b)):
        sh.add(PLine((base[0] + nx * gap_g, base[1] + ny * gap_g),
                     (tip[0] + nx * over, tip[1] + ny * over), layer="EXT"))

    mode = getattr(style, "dim_text_mode", "above")
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    inline = mode in ("inline", "unidirectional") and L > tw * 1.8
    _dim_line(sh, a, b, mid if inline else None, tw + th * 0.8 + 1.6)
    sh.add(PArrow(a, b, style.arrow, kind=ak),
           PArrow(b, a, style.arrow, kind=ak))

    rot = _m.degrees(_m.atan2(uy, ux))
    if rot > 90 or rot <= -90:            # keep text readable, never upside down
        rot += 180
    if mode == "unidirectional":
        rot = 0
    if inline:
        tp = mid if rot == 0 else mid
        tp = (tp[0] - _m.cos(_m.radians(rot)) * 0, tp[1] - th * 0.5)
        t_ent = PText(tp, d.text, th, ha="center", va="bottom", rot=rot,
                      layer="DIM")
    else:
        rr = _m.radians(rot)
        # sit the number just clear of the line, on the outside
        tp = (mid[0] + nx * th * 0.45 - _m.cos(rr) * 0,
              mid[1] + ny * th * 0.45)
        t_ent = PText(tp, d.text, th, ha="center", va="bottom", rot=rot,
                      layer="DIM")
    sh.add(t_ent)
    _record_linear(sh, pl, d, t_ent, a, b, a3, b3, form="aligned")


def _linear(sh: PSheet, pl: Placed, d: LinearDim, style: Style):
    if d.direction == "a":
        _aligned(sh, pl, d, style)
        return
    p1, p2 = pl.m2p(d.p1), pl.m2p(d.p2)
    off = d.offset * pl.s
    th = style.dim_text_height
    tw = len(d.text) * th * style.char_w
    # Witness lines must start on the geometry they measure and overshoot the
    # dimension line slightly. Anchoring them at the view-box corner (the old
    # behaviour) left them visibly stopping short of the edge in question.
    gap_g = style.witness_gap
    over = style.witness_overshoot
    if d.direction == "h":
        yl = min(p1[1], p2[1]) + off
        a, b = (p1[0], yl), (p2[0], yl)
        for xp, anch in ((p1[0], d.anchor1), (p2[0], d.anchor2)):
            y_geom = pl.m2p((0.0, anch))[1] if anch is not None else p1[1]
            dirn = 1.0 if yl > y_geom else -1.0     # travel from geometry to dim
            start = y_geom + dirn * gap_g           # small visible break
            end = yl + dirn * over                  # slight overshoot past it
            if (end - start) * dirn > 0:
                sh.add(PLine((xp, start), (xp, end), layer="EXT"))
        rot = 0
    else:
        xl = min(p1[0], p2[0]) + off
        a, b = (xl, p1[1]), (xl, p2[1])
        for yp, anch in ((p1[1], d.anchor1), (p2[1], d.anchor2)):
            x_geom = pl.m2p((anch, 0.0))[0] if anch is not None else p1[0]
            dirn = 1.0 if xl > x_geom else -1.0
            start = x_geom + dirn * gap_g
            end = xl + dirn * over
            if (end - start) * dirn > 0:
                sh.add(PLine((start, yp), (end, yp), layer="EXT"))
        rot = 90

    L = math.dist(a, b)
    ak = getattr(style, "arrow_style", "filled")
    mode = getattr(style, "dim_text_mode", "above")
    # An inline number needs a gap in the line; an ISO number floats above an
    # unbroken one.
    inline = mode in ("inline", "unidirectional")
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    fits_inline = L > tw * 1.8
    if inline and fits_inline:
        # the gap clears the glyphs by more than the 0.6 mm pad a layout
        # checker grows the text box by, or the line stubs read as crossing
        # the number they were opened for
        _dim_line(sh, a, b, mid, tw + th * 0.8 + 1.6)
    else:
        _dim_line(sh, a, b)
    if L > style.arrow * 3:
        sh.add(PArrow(a, b, style.arrow, kind=ak),
               PArrow(b, a, style.arrow, kind=ak))
    else:  # outside arrows for tight dims
        ux = (b[0] - a[0]) / (L or 1); uy = (b[1] - a[1]) / (L or 1)
        oa = (a[0] - ux * style.arrow * 2, a[1] - uy * style.arrow * 2)
        ob = (b[0] + ux * style.arrow * 2, b[1] + uy * style.arrow * 2)
        sh.add(PLine(oa, a, layer="DIM"), PLine(b, ob, layer="DIM"),
               PArrow(a, oa, style.arrow, kind=ak),
               PArrow(b, ob, style.arrow, kind=ak))

    # Text placement. Centred text on a short dimension is straddled by its own
    # extension lines, so shift it clear of the arrows when it will not fit --
    # standard drafting practice, and it removes a whole class of overlaps.
    shift = getattr(d, "shift", None)
    if shift is None:
        shift = 0 if L > tw * 1.15 else 1
    if inline and fits_inline and shift == 0:
        # centred in the gap just opened in the line. "unidirectional" reads
        # every number horizontally whatever the line's direction; "inline"
        # keeps it aligned with the line.
        rot_t = 0 if mode == "unidirectional" else rot
        tp = (mid[0], mid[1] - th * 0.5) if rot_t == 0 else (mid[0] - th * 0.5, mid[1])
        t_ent = PText(tp, d.text, th, ha="center", va="bottom", rot=rot_t,
                      layer="DIM")
        sh.add(t_ent)
        _record_linear(sh, pl, d, t_ent, a, b, p1, p2)
        return
    if mode == "unidirectional":
        rot = 0
    if d.direction == "h":
        if shift == 0:
            tp, ha = ((a[0] + b[0]) / 2, a[1] + th * 0.5), "center"
        elif shift > 0:
            tp, ha = (max(a[0], b[0]) + style.arrow * 2.2, a[1] + th * 0.5), "left"
        else:
            tp, ha = (min(a[0], b[0]) - style.arrow * 2.2, a[1] - th * 1.4), "right"
    elif rot == 0:
        # vertical dimension, number read horizontally: park it clear of the
        # line rather than on top of it
        if shift == 0:
            tp, ha = (a[0] - th * 0.6, (a[1] + b[1]) / 2 - th * 0.5), "right"
        elif shift > 0:
            tp, ha = (a[0] - th * 0.6, max(a[1], b[1]) + style.arrow * 1.2), "right"
        else:
            tp, ha = (a[0] - th * 0.6, min(a[1], b[1]) - style.arrow * 2.2), "right"
    else:
        if shift == 0:
            tp, ha = (a[0] - th * 0.5, (a[1] + b[1]) / 2), "center"
        elif shift > 0:
            tp, ha = (a[0] - th * 0.5, max(a[1], b[1]) + style.arrow * 2.2), "left"
        else:
            tp, ha = (a[0] + th * 1.4, min(a[1], b[1]) - style.arrow * 2.2), "right"
    if shift:
        # ISO 129-1: a number that will not fit between its arrows is written
        # above an EXTENSION of the dimension line, not floated somewhere
        # near it. Without the extension the value reads as unattached -- on
        # 0000_00000386 "7.5 +0.300/-0.200" and "7.5 +0.100/-0.200" sat off
        # the ends of their chains with nothing joining them to the size they
        # state, which is exactly what a reader would call misaligned.
        _dim_shift_tail(sh, a, b, shift, tw, style, d.direction)
    t_ent = PText(tp, d.text, th, ha=ha, va="bottom", rot=rot, layer="DIM")
    sh.add(t_ent)
    _record_linear(sh, pl, d, t_ent, a, b, p1, p2)


def _dim_shift_tail(sh: PSheet, a, b, shift: int, tw: float, style: Style,
                    direction: str):
    """Extend the dimension line under a number that had to be shifted out.

    Runs from the arrow the text was pushed past to just beyond the end of
    the text, so the value sits on the line it belongs to.
    """
    reach = style.arrow * 2.2 + tw + style.dim_text_height * 0.4
    if direction == "h":
        y = a[1]
        x = max(a[0], b[0]) if shift > 0 else min(a[0], b[0])
        end = x + reach if shift > 0 else x - reach
        sh.add(PLine((x, y), (end, y), layer="DIM"))
    else:
        x = a[0]
        y = max(a[1], b[1]) if shift > 0 else min(a[1], b[1])
        end = y + reach if shift > 0 else y - reach
        sh.add(PLine((x, y), (x, end), layer="DIM"))


def _record_linear(sh, pl, d, t_ent, a, b, p1, p2, form="linear"):
    """Reserve the number's space and record it as a detection object."""
    # Loose box on purpose: this list drives _break_lines_at_text, which gaps
    # a witness line where it would run through a number. That break should
    # clear the glyphs with room to spare, so it uses the reservation estimate
    # rather than the tight ink extent.
    sh.dim_texts.append(_text_box(t_ent, tight=False))
    # measured value in model units, so a consumer does not have to parse the
    # formatted string back into a number
    if d.direction == "a":
        val = math.dist(d.p1, d.p2)          # true length along the edge
    elif d.direction == "h":
        val = abs(d.p2[0] - d.p1[0])
    else:
        val = abs(d.p2[1] - d.p1[1])
    sh.note_annotation(kind="dimension", view=pl.name, text=d.text,
                       detail={"form": form},
                       value=round(to_display(val), 4), direction=d.direction,
                       text_box=_text_box(t_ent, grow=0.0),
                       extent=_union_box([_text_box(t_ent, grow=0.0),
                                          (min(a[0], b[0]), min(a[1], b[1]),
                                           max(a[0], b[0]), max(a[1], b[1]))]),
                       attach=[p1, p2])


def _diameter(sh: PSheet, pl: Placed, d: RadiusDim, style: Style):
    c = pl.m2p(d.center)
    r = d.radius * pl.s
    L = d.leader_len * pl.s
    a = math.radians(d.angle)
    # use the snapped tip recorded at placement time so the arrowhead lands on
    # drawn geometry even when the hole appears as a partial arc
    start = pl.m2p(d.tip) if getattr(d, "tip", None) else \
        (c[0] + math.cos(a) * r, c[1] + math.sin(a) * r)
    knee = (c[0] + math.cos(a) * (r + L), c[1] + math.sin(a) * (r + L))
    sgn = 1 if math.cos(a) >= 0 else -1
    lines = d.text.split("\n")
    tw = max(len(l) for l in lines) * style.dim_text_height * style.char_w
    end = (knee[0] + sgn * tw, knee[1])
    sh.add(PLine(start, knee, layer="LEADER"), PLine(knee, end, layer="LEADER"),
           PArrow(start, knee, style.arrow))
    tx = knee[0] + (1.0 if sgn > 0 else -1.0)
    tboxes = []
    for i, ln in enumerate(lines):
        te = PText((tx, knee[1] + 0.8 + (len(lines) - 1 - i) * style.dim_text_height * 1.35),
                   ln, style.dim_text_height,
                   ha="left" if sgn > 0 else "right", layer="DIM")
        sh.add(te)
        tboxes.append(_text_box(te, grow=0.0))
        # reserve it, so no other leader is left running through this label
        sh.dim_texts.append(_text_box(te, tight=False))
    tb = _union_box(tboxes)
    sh.note_annotation(
        # Every printed diameter is a radial feature annotation, whether
        # it describes a hole or the overall diameter of a round view.
        kind=getattr(d, "cls", "radius"), view=pl.name, text=d.text,
        value=round(to_display(d.radius * 2), 4),
        detail={"form": "diameter"},
        text_box=tb,
        extent=_union_box(tboxes + [(min(start[0], knee[0], end[0]),
                                     min(start[1], knee[1], end[1]),
                                     max(start[0], knee[0], end[0]),
                                     max(start[1], knee[1], end[1]))]),
        attach=[start])


def _note(sh: PSheet, pl: Placed, d: Callout, style: Style):
    t = pl.m2p(d.tail)
    a = pl.m2p(d.anchor)
    sh.add(PLine(a, t, layer="LEADER"), PArrow(a, t, style.arrow))
    ha = "left" if t[0] >= a[0] else "right"
    te = PText((t[0] + (1 if ha == "left" else -1), t[1] + 0.8), d.text,
               style.note_text_height, ha=ha, layer="DIM")
    sh.add(te)
    tb = _text_box(te, grow=0.0)
    # Reserve the label so no OTHER leader is left running through it. The
    # per-view repair sweep cannot see this: two views are annotated
    # independently and only meet in paper space.
    sh.dim_texts.append(_text_box(te, tight=False))
    sh.note_annotation(
        # radius / chamfer / thread / dimension, decided where the note was
        # created rather than by pattern-matching the text here.
        # radius / chamfer / thread / bore keep their feature class; anything
        # that names no feature type is a LEADER note, which is a different
        # object from the paragraph block at the bottom of the sheet.
        kind=getattr(d, "cls", "leader_note"), view=pl.name, text=d.text,
        detail=({"synthetic": True} if getattr(d, "synthetic", False) else {}),
        text_box=tb,
        extent=_union_box([tb, (min(a[0], t[0]), min(a[1], t[1]),
                                max(a[0], t[0]), max(a[1], t[1]))]),
        attach=[a])


def _feature_frame(sh: PSheet, pl: Placed, d: FeatureFrame, style: Style):
    """Draw a GD&T feature control frame: |sym|value|A|B|C| on a leader.

    The frame is real boxed geometry with the characteristic drawn as vector
    strokes, because the font has no glyph for most characteristics (see
    autodraft/gdt_symbols.py). The bounding box reported for detection is the
    FRAME, not just its text: the box IS the object a reader recognises, and
    unlike a dimension the frame is compact, so boxing all of it is right.
    """
    from .gdt_symbols import symbol as _gsym
    a = pl.m2p(d.anchor)
    t = pl.m2p(d.tail)
    h = style.dim_text_height * 1.6          # compartment height
    pad = h * 0.22

    # Compartment widths: the symbol cell is square; text cells fit their text.
    cells = [("sym", h)]
    cells.append(("text:" + d.value_text,
                  len(d.value_text) * style.dim_text_height * style.char_w + 2 * pad))
    for dat in d.datums:
        cells.append(("text:" + dat, style.dim_text_height * style.char_w + 2 * pad))
    total = sum(w for _, w in cells)

    # Put the frame on whichever side of the anchor the placer chose.
    x0 = t[0] if t[0] >= a[0] else t[0] - total
    y0 = t[1] - h / 2

    sh.add(PLine(a, (x0 if t[0] >= a[0] else x0 + total, t[1]), layer="LEADER"),
           PArrow(a, (x0 if t[0] >= a[0] else x0 + total, t[1]), style.arrow))

    x = x0
    for name, w in cells:
        sh.add(PPoly([(x, y0), (x + w, y0), (x + w, y0 + h), (x, y0 + h)],
                     layer="DIM", close=True))
        if name == "sym":
            polys, circles = _gsym(d.characteristic, x + h * 0.16,
                                   y0 + h * 0.16, h * 0.68)
            for poly in polys:
                sh.add(PPoly(poly, layer="DIM"))
            for cx, cy, r in circles:
                sh.add(PCircle((cx, cy), r, layer="DIM"))
        else:
            sh.add(PText((x + w / 2, y0 + h * 0.5), name[5:],
                         style.dim_text_height, ha="center", va="center",
                         layer="DIM"))
        x += w

    box = (x0, y0, x0 + total, y0 + h)
    sh.note_annotation(
        kind="gdt", view=pl.name, text=d.text, value=d.value,
        text_box=box,
        extent=_union_box([box, (min(a[0], x0), min(a[1], y0),
                                 max(a[0], x0 + total), max(a[1], y0 + h))]),
        attach=[a],
        detail={"characteristic": d.characteristic,
                "datums": list(d.datums), "modifier": d.modifier,
                "synthetic": bool(d.synthetic)})


def _clip_to_box(a, b, box):
    """Walk from ``a`` toward ``b`` and stop at the boundary of ``box``.

    Returns the first point on the rectangle's edge, or ``b`` when the segment
    never enters it. Used so a leader terminates on a symbol's frame instead
    of running underneath its text.
    """
    x0, y0, x1, y1 = box
    if x0 <= a[0] <= x1 and y0 <= a[1] <= y1:
        return a                      # already inside: nothing to draw into
    prev = a
    steps = 96
    for i in range(1, steps + 1):
        t = i / steps
        p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
            return prev
        prev = p
    return b


def _datum_flag(sh: PSheet, pl: Placed, d: DatumFlag, style: Style):
    """Draw a datum feature symbol: a boxed letter with a filled triangle."""
    a = pl.m2p(d.anchor)
    t = pl.m2p(d.tail)
    h = style.dim_text_height * 1.6
    w = h
    x0, y0 = t[0] - w / 2, t[1] - h / 2

    # Leader from the surface to the flag, stopping ON the frame boundary.
    #
    # This used to pick the top or bottom edge purely from the vertical
    # direction of travel, which is wrong whenever the leader arrives mostly
    # sideways: it then ran past that edge and terminated inside the box, on
    # top of the datum letter. QA counted the letter as crossed on 12 of the
    # 18 parts. Clipping the segment against the frame rectangle puts the
    # endpoint on whichever edge the leader actually meets.
    stop = _clip_to_box(a, t, (x0, y0, x0 + w, y0 + h))
    sh.add(PLine(a, stop, layer="LEADER"))
    tri = h * 0.34
    sh.add(PPoly([(a[0] - tri * 0.6, a[1]), (a[0] + tri * 0.6, a[1]),
                  (a[0], a[1] + (tri if t[1] >= a[1] else -tri))],
                 layer="DIM", close=True))
    sh.add(PPoly([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)],
                 layer="DIM", close=True))
    sh.add(PText((x0 + w / 2, y0 + h * 0.5), d.letter, style.dim_text_height,
                 ha="center", va="center", layer="DIM"))

    box = (x0, y0, x0 + w, y0 + h)
    sh.note_annotation(
        kind="datum", view=pl.name, text=f"DATUM {d.letter}",
        text_box=box,
        extent=_union_box([box, (min(a[0], x0), min(a[1], y0),
                                 max(a[0], x0 + w), max(a[1], y0 + h))]),
        attach=[a], detail={"datum": d.letter,
                            "synthetic": bool(d.synthetic)})


def _roughness_glyph(sh: PSheet, x, y, size, machined: bool, layer="DIM",
                    commit: bool = True):
    """The ISO 1302 surface-texture tick, drawn at (x, y) as its vertex.

    Two legs meeting at the surface: a short one at 60 deg and a long one at
    60 deg the other way. A horizontal bar across the vee means material
    removal is required (the machined form).
    """
    import math as _m
    a = _m.radians(60)
    short = size * 0.62
    long_ = size * 1.24
    p_short = (x - _m.cos(a) * short, y + _m.sin(a) * short)
    p_long = (x + _m.cos(a) * long_, y + _m.sin(a) * long_)
    if commit:
        sh.add(PLine((x, y), p_short, layer=layer),
               PLine((x, y), p_long, layer=layer))
        if machined:
            sh.add(PLine(p_short, (p_long[0] - (p_long[0] - p_short[0]) * 0.0,
                                   p_short[1]), layer=layer))
    return p_short, p_long


def _roughness(sh: PSheet, pl: Placed, d: RoughnessMark, style: Style):
    """Draw a surface-texture symbol with its Ra value."""
    a = pl.m2p(d.anchor)
    t = pl.m2p(d.tail)
    size = style.dim_text_height * 1.5
    sh.add(PLine(a, t, layer="LEADER"), PArrow(a, t, style.arrow))
    p_short, p_long = _roughness_glyph(sh, t[0], t[1], size, d.machined)
    te = PText((p_long[0] + 0.8, p_long[1] - style.dim_text_height * 0.5),
               d.ra_text, style.dim_text_height, ha="left", layer="DIM")
    sh.add(te)
    box = _union_box([_text_box(te, grow=0.0),
                      (min(t[0], p_short[0]), t[1],
                       max(p_long[0], t[0]), p_long[1])])
    sh.note_annotation(
        kind="roughness", view=pl.name, text=d.ra_text,
        value=float(d.ra_text.split()[-1]),
        text_box=box,
        extent=_union_box([box, (min(a[0], box[0]), min(a[1], box[1]),
                                 max(a[0], box[2]), max(a[1], box[3]))]),
        attach=[a],
        # Flagged so a consumer can filter these out: unlike every other
        # annotation, the Ra value is assigned, not measured. See
        # autodraft/roughness.py.
        detail={"synthetic": True, "machined": d.machined})


def _draw_subparts(sh, style, subparts, x0, y0, x1, y1):
    """Draw the strip of per-item detail views along the bottom of the sheet.

    Each item is a real projection with its own dimensions, scaled to its
    cell; the caption comes from _emit_view, so a detail view is labelled and
    boxed exactly like a main view.
    """
    if not subparts:
        return
    n = len(subparts)
    cell_w = (x1 - x0) / n
    for i, (label, ann) in enumerate(subparts):
        proj = ann.view
        if proj is None or proj.width <= 0 or proj.height <= 0:
            continue
        pl_, pr_, pb_, pt_ = ann.pad
        need_w = proj.width + pl_ + pr_
        need_h = proj.height + pb_ + pt_
        pad = min(cell_w, y1 - y0) * 0.06
        avail_w, avail_h = cell_w - 2 * pad, (y1 - y0) - 2 * pad
        if avail_w <= 0 or avail_h <= 0 or need_w <= 0 or need_h <= 0:
            return
        sc = min(avail_w / need_w, avail_h / need_h)
        cx0 = x0 + i * cell_w + pad
        ox = cx0 + (avail_w - need_w * sc) / 2 + (pl_ - proj.xmin) * sc
        oy = y0 + pad + (avail_h - need_h * sc) / 2 + (pb_ - proj.ymin) * sc
        _emit_view(sh, Placed(ann, ox, oy, sc, name=proj.name), style, label,
                   skip=_illegible_dims(ann, sc, style))


def _illegible_dims(ann, sc: float, style: Style) -> set:
    """Dimensions that will not fit at scale ``sc``, as a set of ``id()``.

    A sub-part thumbnail is drawn at whatever scale its cell allows -- on
    0000_00000126 that is about a fifth of the sheet scale -- but the text is
    always ``dim_text_height`` millimetres tall, because text does not scale
    with the view. The chain that was legible on a full-size view therefore
    piles three labels into one column: on ITEM 1 the strings
    "1.68 +0.17/-0.10", "1.68 +0.10/-0.05" and "3.35" were printed on top of
    each other.

    Two measured rules, both in paper millimetres:

    * a dimension whose label is wider than the distance it measures cannot
      be written between its own arrows;
    * of the survivors, any whose label box lands on one already accepted is
      dropped -- longest measurement first, because that is the one a reader
      needs to size the part.

    Dropping is the honest outcome: an unreadable dimension is worse than an
    absent one, and the full-size view still carries the whole chain.
    """
    ch = style.dim_text_height
    cw = style.char_w
    # The part itself is the first thing on the paper: a SIZE printed across
    # its outline is an overlap whatever else it clears. On ITEM 1 of
    # 0000_00000126 the overall "4.47" landed on the bottom edge of the view,
    # because its dimension line sits one text height below the part. A
    # leader note is different -- its arrow is supposed to land on the
    # geometry -- so only linear dimensions are tested against this.
    v = ann.view
    part = (v.xmin * sc, v.ymin * sc, v.xmax * sc, v.ymax * sc)
    taken, skip = [], set()
    items = []
    for d in getattr(ann, "dims", []) or []:
        if isinstance(d, LinearDim):
            length = math.hypot(d.p2[0] - d.p1[0], d.p2[1] - d.p1[1]) * sc
            mid = ((d.p1[0] + d.p2[0]) / 2, (d.p1[1] + d.p2[1]) / 2)
            if d.direction == "v":
                pos = (mid[0] + d.offset, mid[1])
            else:
                pos = (mid[0], mid[1] + d.offset)
            lines = d.text.split("\n")
            tw = max(len(ln) for ln in lines) * ch * cw
            if d.direction == "v":
                w, h = ch * 1.4, tw          # rotated
            else:
                w, h = tw, ch * 1.4
            if length < tw * 0.95:
                skip.add(id(d))
                continue
            # Text is written ABOVE the dimension line (and, on a rotated
            # vertical dimension, to the left of it) unless the house style
            # breaks the line for it. Placing the box ON the line instead
            # under-reported the clash by half a text height, which is
            # exactly the amount by which "4.47" sat on the part outline.
            px, py = pos[0] * sc, pos[1] * sc
            if getattr(style, "dim_text_mode", "above") != "inline":
                if d.direction == "v":
                    px -= w / 2
                else:
                    py += h / 2
            box = (px - w / 2, py - h / 2, px + w / 2, py + h / 2)
            if not (box[2] < part[0] or box[0] > part[2] or
                    box[3] < part[1] or box[1] > part[3]):
                skip.add(id(d))          # printed across the part
                continue
            items.append((length, d, (px, py, w, h)))
        elif isinstance(d, (RadiusDim, Callout)):
            anchor = getattr(d, "center", None) or getattr(d, "anchor", None)
            tail = getattr(d, "tail", None)
            pos = tail if tail else anchor
            if pos is None:
                continue
            tw = max(len(ln) for ln in d.text.split("\n")) * ch * cw
            items.append((0.0, d, (pos[0] * sc, pos[1] * sc, tw, ch * 1.4)))
    items.sort(key=lambda it: -it[0])
    for _length, d, (x, y, w, h) in items:
        box = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        if any(not (box[2] < b[0] or box[0] > b[2] or
                    box[3] < b[1] or box[1] > b[3]) for b in taken):
            skip.add(id(d))
            continue
        taken.append(box)
    return skip


def _reserve_table_strip(style, table_rows, summary_rows, row_h, col_w,
                         H, margin, ax0, ax1, table_blocks_override):
    """Decide where the hole table goes and how much room it needs.

    A side table takes a full-height COLUMN, narrowing the view area; a
    top/bottom table takes a wide BAND, shortening it. Both are normal
    drafting practice. The reservation happens before the views are placed,
    which is what stops the table ever landing on drawn geometry.

    Returns ``(side, width, height, n_blocks)``.
    """
    # "tl"/"tr" are corner tables: a band across the top of the sheet like
    # "top", but pushed into a corner instead of centred, which is where a
    # hole chart or a revision block usually sits on a real drawing.
    side = getattr(style, "table_side", "right")
    if side not in ("right", "top", "bottom", "tl", "tr"):
        side = "right"
    # The title block owns the top of the sheet when it lives up there: its
    # parts list and note block hang below it, and a hole table is wide (two
    # or three blocks of five columns), so on a portrait sheet a top table
    # simply cannot clear it. The table goes to the bottom band instead.
    corner = getattr(style, "title_block_corner", "br")
    if corner in ("tl", "tr") and side in ("top", "tl", "tr"):
        side = "bottom"
    table_w = table_h = 0.0
    n_blocks = 1
    if table_rows:
        if side == "right":
            # let the table run the full usable height before adding a
            # column, so it stays narrow and leaves width for the views
            avail_h = (H - margin - 12) - (margin + 8)
            per_col = max(1, int(avail_h / row_h) - 1)
            min_blocks = int(math.ceil(len(table_rows) / per_col))
            n_blocks = min(style.table_max_cols, max(1, min_blocks))
            if table_blocks_override:
                n_blocks = min(style.table_max_cols,
                               max(min_blocks, table_blocks_override))
            table_w = n_blocks * sum(col_w) + (n_blocks - 1) * 4.0
        else:
            # A band is short and wide: use as many columns as the sheet
            # width allows so the block stays only a few rows deep.
            fit_cols = max(1, int((ax1 - ax0) / (sum(col_w) + 4.0)))
            n_blocks = min(style.table_max_cols, fit_cols,
                           max(1, len(table_rows)))
            per_col = int(math.ceil(len(table_rows) / n_blocks))
            table_w = n_blocks * sum(col_w) + (n_blocks - 1) * 4.0
            table_h = (per_col + 1) * row_h + row_h * 1.2   # +header +title
    elif summary_rows:
        table_w = 78.0
        if side != "right":
            table_h = (len(summary_rows) + 1) * row_h + row_h * 1.2
    return side, table_w, table_h, n_blocks


def _note_table(sh, x0, y0, x1, y1, kind, rows=None, title=""):
    """Record a ruled block of fields as one detection object.

    Whole-table boxes only: a reader recognises "the hole table" as one thing,
    and boxing every row would bury the eight annotation classes under a few
    hundred row objects per sheet. Which table it is stays in the detail, so
    a consumer can filter without a new class.
    """
    box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    sh.note_annotation(kind="table", view="", text=title or kind.upper(),
                       qty=(len(rows) if rows is not None else None),
                       text_box=box, extent=box, attach=[],
                       detail={"table_kind": kind,
                               "rows": (len(rows) if rows is not None else None)})


def _draw_hole_table(sh, style, side, table_rows, summary_rows, header,
                     col_w, row_h, table_w, table_h, n_blocks,
                     W, H, margin, tb_h):
    """Draw the hole table in the strip reserved by _reserve_table_strip."""
    if not (table_rows or summary_rows):
        return
    # How much paper the title block eats at the BOTTOM of the sheet. A
    # top-corner block eats none of it, and assuming it did drew the bottom
    # band a full block height (40 mm) above the strip that was reserved for
    # it -- which is how the HOLE TABLE ended up printed through the ITEM 1
    # sub-part view on 0000_00000126. _Page.ay0 makes exactly this
    # distinction; this function has to make the same one.
    bottom_block = getattr(style, "title_block_corner", "br") not in ("tl", "tr")
    tb_bottom = tb_h if bottom_block else 0.0
    if side == "right":
        tx, ty = W - margin - 4 - table_w, H - margin - 12
        max_h = (H - margin - 12) - (margin + tb_bottom + 8)
        max_cols = style.table_max_cols
    else:
        if side == "tr":
            tx = W - margin - 4 - table_w
        elif side == "tl":
            tx = margin + 4
        else:
            tx = margin + 6
        ty = ((margin + tb_bottom + 8 + table_h) if side == "bottom"
              else H - margin - 12)
        max_h = table_h
        max_cols = n_blocks
    before = len(sh.prims)
    if table_rows:
        draw_table_paginated(sh, tx, ty, header, table_rows, col_w, row_h,
                             style, title="HOLE TABLE", max_h=max_h,
                             max_cols=max_cols)
    else:
        draw_table(sh, tx, ty, ["QTY", "DESCRIPTION"], summary_rows,
                   [14, 62], style.table_row_h, style, title="HOLE SUMMARY")
    # measured from the ink the table actually put down, because pagination
    # decides how many columns it needed
    xs, ys = [], []
    for prim in sh.prims[before:]:
        for pt in getattr(prim, "pts", None) or (
                [prim.a, prim.b] if hasattr(prim, "a") else [getattr(prim, "p", None)]):
            if pt:
                xs.append(pt[0]); ys.append(pt[1])
    if xs:
        _note_table(sh, min(xs), min(ys), max(xs), max(ys),
                    "hole" if table_rows else "hole_summary",
                    table_rows or summary_rows,
                    "HOLE TABLE" if table_rows else "HOLE SUMMARY")


def _draw_sheet_furniture(sh: PSheet, info, meta, notes, style: Style,
                          W, margin, tb_w, tb_h):
    """General notes, title block, finish note and projection symbol.

    Extracted from build_sheet to keep it inside the complexity ratchet.
    Everything here hangs off the title block's corner, so it belongs
    together.
    """
    b = info.bbox
    from .pipeline import _scale_note
    for _n in _scale_note(info):
        notes.insert(0, _n)
    notes.insert(0, f"STOCK ENVELOPE {fmt(b.xlen)} X {fmt(b.ylen)} X "
                    f"{fmt(b.zlen)} {UNITS}")
    if getattr(info, "n_solids", 1) > 1:
        # The overall dimensions still span the envelope, but each body is now
        # sized individually by its own BODY note, so the old warning that
        # "DIMENSIONS SPAN THE OVERALL ENVELOPE" no longer describes the sheet.
        if getattr(info, "bodies", None):
            notes.insert(1, f"{info.n_solids} SEPARATE BODIES IN THIS FILE - "
                            "OVERALL DIMENSIONS ARE THE ENVELOPE; EACH BODY "
                            "IS SIZED SEPARATELY")
        else:
            notes.insert(1, f"{info.n_solids} SEPARATE BODIES IN THIS FILE - "
                            "DIMENSIONS SPAN THE OVERALL ENVELOPE")
    if info.is_plate and info.thickness:
        notes.insert(1, f"SHEET THICKNESS {fmt(info.thickness)} {UNITS}")
    # Title-block content: the real fields (title, part number, scale) come
    # from the model; the rest -- drafter, checker, revision, department --
    # are generated, because a CAD file does not record them. All flagged
    # synthetic. See autodraft/titleblock.py.
    if meta.get("tb_fields") is None and getattr(style, "vary_titleblock", True):
        from .titleblock import generate as _tb_generate
        from .variation import _Rng, _seed
        rng = _Rng(_seed("titleblock:" + str(meta.get("part_number", ""))))
        slots = 1 + (max(3, style.title_block_rows) - 1) * \
            max(2, style.title_block_cols)
        fields, order = _tb_generate(
            str(meta.get("part_number", "")), rng,
            {"TITLE": meta.get("title", ""),
             "PART NUMBER": meta.get("part_number", ""),
             "SCALE": meta.get("scale", ""),
             "UNITS": meta.get("units", "mm"),
             "SHEET": meta.get("sheet", "1/1")},
            slots,
            # WEIGHT is computed from this and the material, not invented.
            volume_mm3=float(getattr(info, "volume", 0.0) or 0.0))
        meta["tb_fields"] = fields
        meta["tb_order"] = order
        # One revision for the whole sheet: the block's REV cell, the
        # REVISIONS table and the change marks on the views all read it here.
        meta.setdefault("revision", fields.get("REV", "A"))
        meta.setdefault("drawn", fields.get("drawn", "AUTODRAFT"))
        meta.setdefault("date", fields.get("date", ""))
        if not meta.get("material"):
            meta["material"] = fields.get("material", "")
        fields.setdefault("MATERIAL", meta["material"])
        fields.setdefault("DRAWN", meta["drawn"])
        fields.setdefault("DATE", meta["date"])
    # What the note block says is a house convention. A numbered list states
    # facts about this part; an UNLESS OTHERWISE SPECIFIED block states the
    # general tolerances that govern every unmarked size -- which is exactly
    # what a sheet whose dimensions carry no individual tolerance needs.
    notes_style = getattr(style, "notes_style", "numbered")
    if notes_style == "none":
        notes = []
    elif notes_style == "uos":
        notes = _general_tolerance_notes(style) + notes
    if (getattr(style, "flag_notes", False) and notes
            and not any("BREAK SHARP" in str(n).upper() for n in notes)):
        # An office that flags notes onto the views needs a note worth
        # flagging. The edge-condition note is drafting boilerplate -- the
        # same category as the UNLESS OTHERWISE SPECIFIED block, and it is an
        # instruction rather than a measurement, so it states nothing about
        # the model that could be wrong.
        notes.append("BREAK SHARP EDGES 0.3 MAX; REMOVE ALL BURRS")
    meta["notes"] = notes

    # A parts list grows upward out of the title block, so it is drawn first
    # and the note block is pushed above it.
    from .sheet import title_block_origin
    tb_x0, tb_y0 = title_block_origin(sh, style, margin, tb_w, tb_h)
    # Left edge of the note text. The block's own rule sits on this x, so the
    # text is inset the same 1.5 mm the title-block cells use -- printed
    # flush it runs into the border line (visible on 0000_00000413, where the
    # numbers of the UOS block straddle the frame).
    #
    # ONE derivation, in meta, read by the drawing code and by the label
    # writer. _emit_general_notes used to test `corner == "bl"` and take the
    # right-hand formula for every other corner, so on a TOP-LEFT block it
    # recorded the notes 74 mm to the right of their ink.
    meta["notes_x"] = tb_x0 + 1.5
    parts_h = 0.0
    top_block = getattr(style, "title_block_corner", "br") in ("tl", "tr")
    if getattr(style, "parts_list", False):
        rows = _parts_rows(info, meta)
        # the list grows away from the block: up from a bottom block, down
        # from a top one
        py = (tb_y0 - _parts_height(style, rows) if top_block
              else tb_y0 + tb_h)
        parts_h = draw_parts_list(sh, tb_x0, py, tb_w, rows, style)
        if parts_h:
            _note_table(sh, tb_x0, py, tb_x0 + tb_w, py + parts_h,
                        "parts_list", rows, "PARTS LIST")
    # Where the general-note block sits. This is computed ONCE, here, and
    # both the drawing code (sheet.draw_title_block) and the label writer
    # (_emit_general_notes) read it from meta -- see the note in
    # draw_title_block about the 18.5 mm offset two derivations produced.
    #
    # notes_base is the baseline of the LOWEST line; the heading is
    # len(lines) pitches above it. From a bottom block the stack grows up out
    # of the title block (and its parts list); from a top block it hangs
    # below, so the base has to be dropped by the whole stack height.
    from .sheet import note_lines as _note_lines
    _t = style.note_text_height * getattr(style, "title_text_scale", 1.0)
    _n_lines = len(_note_lines(notes, (W - margin) - meta["notes_x"], _t * 0.9,
                               sh.char_w))
    if top_block:
        meta["notes_base"] = (tb_y0 - parts_h - 6
                              - (_n_lines + 1) * _t * 1.6)
    else:
        meta["notes_base"] = tb_y0 + tb_h + parts_h + 6

    # The revision table lives in the top-right corner, which is also where a
    # right-hand hole table runs. Only one of them can have that column.
    if getattr(style, "rev_table", False) and not meta.get("_right_table"):
        rev_w = min(tb_w, 92.0)
        rows = _rev_rows(meta)
        # under a top-corner title block, otherwise in the top-right corner
        rev_y = (tb_y0 - 2.0 if top_block else sh.h - margin - 2.0)
        h = draw_rev_table(sh, sh.w - margin - rev_w, rev_y, rev_w, rows, style)
        if h:
            _note_table(sh, sh.w - margin - rev_w, rev_y - h,
                        sh.w - margin, rev_y, "revision", rows, "REVISIONS")

    draw_title_block(sh, meta, margin, tb_w, tb_h, style)
    _note_table(sh, tb_x0, tb_y0, tb_x0 + tb_w, tb_y0 + tb_h, "title_block",
                None, "TITLE BLOCK")
    _emit_general_notes(sh, meta, margin, tb_w, tb_h, style)
    _emit_flag_notes(sh, meta, notes, style, margin, W)
    _emit_revision_marks(sh, meta, style, margin, W)
    # The finish note stacks clear of the notes block, so it must clear the
    # number of lines actually drawn, not the number of notes. A wrapped note
    # occupies several lines; passing len(notes) put the roughness symbol
    # back down on top of the text.
    _general_roughness(sh, info, style, W, margin, tb_w, tb_h, _n_lines,
                       base=meta.get("notes_base"), below=top_block,
                       x0=meta.get("notes_x"))
    # The projection symbol sits beside the title block, so it follows it --
    # but only if there is paper beside it. Two things have gone wrong here.
    # On a narrow sheet the block is nearly as wide as the page
    # (0000_00000386: 171 mm of block on 210 mm of paper) and the symbol was
    # clamped back into the left margin, i.e. on top of the DATE and SHEET
    # cells. Then, once it was allowed above the block, it landed on the
    # GENERAL NOTES instead (0000_00000093: the cone printed through "DO NOT
    # SCALE FROM THIS DRAWING" and "STOCK ENVELOPE ...").
    #
    # Testing against the title block alone can only ever fix one of those.
    # The test is now "is this rectangle of paper EMPTY", against everything
    # already drawn -- which by this point is the whole sheet bar the symbol
    # itself. Several candidate spots are tried in the order a draughtsman
    # would use them, and the symbol is left off entirely rather than printed
    # over something: a projection symbol is a convention, not information,
    # and every drawing here also states the convention in words.
    _sc = getattr(style, "title_text_scale", 1.0)
    _corner = getattr(style, "title_block_corner", "br")
    _half_w, _half_h = 13.0 * _sc, 9.0 * _sc
    _blk_lo = tb_y0 - (parts_h if top_block else 0.0)
    _blk_hi = tb_y0 + tb_h + (0.0 if top_block else parts_h)
    _beside_y = tb_y0 + (tb_h * 0.3 if not top_block else 4 * _sc)
    _clear = (_blk_hi + _half_h + 4 * _sc if not top_block
              else _blk_lo - _half_h - 4 * _sc)
    _cands = [
        # beside the block, on the side away from the sheet edge
        ((tb_x0 + tb_w + 22 * _sc) if _corner in ("bl", "tl")
         else (tb_x0 - 22 * _sc), _beside_y),
        # then just clear of the block, working across the sheet
        (tb_x0 + tb_w * 0.5, _clear),
        (tb_x0 + tb_w - _half_w - 2, _clear),
        (margin + _half_w + 2, _clear),
        # then the two bottom corners, which is where a lot of offices put it
        (margin + _half_w + 2, margin + _half_h + 2),
        (W - margin - _half_w - 2, margin + _half_h + 2),
        (margin + _half_w + 2, sh.h - margin - _half_h - 2),
        (W - margin - _half_w - 2, sh.h - margin - _half_h - 2),
    ]
    # ...and if none of those is free, anywhere on the sheet that is. A
    # drawing with no projection symbol at all looks unfinished; 0000_00000093
    # had 60 x 40 mm of blank paper doing nothing while the symbol was
    # dropped for want of a listed spot.
    for _r in (_free_rect(sh, style, margin, margin, W - margin,
                          sh.h - margin, margin, many=True) or []):
        if (_r[2] - _r[0]) < 2 * _half_w + 4 or (_r[3] - _r[1]) < 2 * _half_h + 4:
            continue
        _cands.append(((_r[0] + _r[2]) / 2, (_r[1] + _r[3]) / 2 - _half_h * 0.3))
    for _px, _py in _cands:
        _box = (_px - _half_w, _py - _half_h * 0.7,
                _px + _half_w, _py + _half_h)
        if _box[0] < margin or _box[2] > W - margin:
            continue
        if _box[1] < margin or _box[3] > sh.h - margin:
            continue
        if not _ink_free(sh, _box):
            continue
        _proj_symbol(sh, _px, _py, style.projection, scale=_sc)
        break
    _break_lines_at_text(sh)


def _emit_stock_outline(sh: PSheet, style: Style, info, placed):
    """The bar or plate the part is machined from, in phantom line.

    Standard practice on a machined-from-solid part: the stock envelope is
    drawn round the part in ISO 128 type K (long dash, two short dashes) so
    the shop can see what to buy and how much is coming off. It is the one
    piece of information on the sheet that is about the BLANK rather than the
    finished part, and the corpus had no phantom line of any kind.

    Drawn only where it says something: the envelope has to stand clear of
    the silhouette (a plate machined only on its edges gains nothing from a
    phantom line drawn on top of the outline), and only on the primary view.
    """
    if not getattr(style, "stock_outline", False) or not placed:
        return
    name, pl = next(iter(placed.items()))
    v = pl.ann.view
    # Round the envelope up to the next round size of stock, which is how a
    # bar or plate is actually bought. In millimetres AS DRAWN -- view units
    # already are the drawn size -- but the step has to suit the part: a
    # fixed 5 mm turned a 5 mm micro-part into "STOCK 10 X 10", a rectangle
    # twice the size of the view and, at 40:1, 516 mm of paper on a 420 mm
    # sheet. The box was recorded 520 mm off the top of the image.
    span = max(v.width, v.height)
    step = 5.0 if span > 40.0 else (2.0 if span > 12.0 else
                                    1.0 if span > 4.0 else 0.5)
    allow = step * 0.8
    import math as _m
    # At least a couple of millimetres of clean-up on every side, or the
    # phantom line lands on the outline it is meant to stand outside: on
    # 0000_00000061 (30 x 30) an exact multiple gave "STOCK 35 X 30", the top
    # and bottom edges drawn straight over the part.
    ow = _m.ceil((v.width + allow) / step) * step
    oh = _m.ceil((v.height + allow) / step) * step
    cx, cy = (v.xmin + v.xmax) / 2, (v.ymin + v.ymax) / 2
    a = pl.m2p((cx - ow / 2, cy - oh / 2))
    b = pl.m2p((cx + ow / 2, cy + oh / 2))
    # ...and it has to be ON the sheet. A stock outline that runs off the
    # paper is worse than none: the drawing loses nothing without it.
    m = style.sheet_margin
    if a[0] < m or a[1] < m or b[0] > sh.w - m or b[1] > sh.h - m:
        return
    sh.add(PPoly([(a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1])],
                 layer="PHANTOM", close=True))
    txt = PText(((a[0] + b[0]) / 2, b[1] + style.note_text_height * 0.6),
                f"STOCK {fmt(ow)} X {fmt(oh)}", style.note_text_height * 0.9,
                ha="center", layer="TEXT")
    sh.add(txt)
    box = _text_box(txt, grow=0.0)
    sh.note_annotation(kind="note", view=name, text=txt.s, text_box=box,
                       extent=_union_box([box, (a[0], a[1], b[0], b[1])]),
                       attach=[], detail={"placement": "stock_outline"})


def _emit_revision_marks(sh: PSheet, meta, style: Style, margin, W):
    """A triangle carrying the revision letter, beside a changed dimension.

    Every issued drawing above revision A carries these: the reader needs to
    know WHICH dimension moved at revision C without diffing two prints. The
    revision history on these sheets is generated (a CAD file records none),
    so the mark is flagged synthetic like the table it refers to -- but the
    convention, and the shape a detector has to learn, is exactly the real
    one.
    """
    if not getattr(style, "rev_marks", False):
        return
    # The revision letter lives in the title block's own field map when the
    # block was generated, which is the usual case.
    rev = str(meta.get("revision") or meta.get("rev")
              or (meta.get("tb_fields") or {}).get("REV") or "").strip()
    if not rev or rev in ("-", "A", "0", "01"):
        return                          # first issue changed nothing
    # "B" or "03" -- offices letter and offices number, and taking rev[0]
    # turned revision 03 into a triangle marked "0".
    letter = rev if len(rev) <= 2 else rev[0]
    # the dimensions this sheet drew, in the order they were emitted
    dims = [a for a in sh.annotations
            if a.get("kind") == "dimension" and a.get("text_box")]
    if not dims:
        return
    from .variation import _Rng, _seed
    rng = _Rng(_seed("revmark:" + str(meta.get("part_number", ""))))
    t = style.dim_text_height
    # A two-character revision ("03") needs a wider triangle than a letter.
    half = t * (1.15 if len(letter) == 1 else 1.55)
    placed = 0
    for k in range(len(dims)):
        if placed >= 2:
            break
        a = dims[(rng._next() + k * 7) % len(dims)]
        x0, y0, x1, y1 = a["text_box"]
        for dx, dy in ((x1 - x0, 0.0), (-(x1 - x0), 0.0),
                       (0.0, y1 - y0), (0.0, -(y1 - y0))):
            cx = (x0 + x1) / 2 + dx / 2 + (half * 1.6 if dx > 0 else
                                           -half * 1.6 if dx < 0 else 0.0)
            cy = (y0 + y1) / 2 + dy / 2 + (half * 1.6 if dy > 0 else
                                           -half * 1.6 if dy < 0 else 0.0)
            box = (cx - half * 1.3, cy - half * 1.1,
                   cx + half * 1.3, cy + half * 1.4)
            if box[0] < margin or box[2] > W - margin:
                continue
            if box[1] < margin or box[3] > sh.h - margin:
                continue
            if not _ink_free(sh, box):
                continue
            sh.add(PPoly([(cx, cy + half), (cx - half * 1.05, cy - half * 0.75),
                          (cx + half * 1.05, cy - half * 0.75)],
                         layer="DIM", close=True))
            txt = PText((cx, cy - half * 0.55), letter, t * 0.62,
                        ha="center", layer="DIM")
            sh.add(txt)
            sh.note_annotation(
                kind="leader_note", view=a.get("view", ""),
                text=f"REV {letter}",
                text_box=(cx - half * 1.05, cy - half * 0.75,
                          cx + half * 1.05, cy + half),
                extent=(cx - half * 1.05, cy - half * 0.75,
                        cx + half * 1.05, cy + half),
                attach=[((x0 + x1) / 2, (y0 + y1) / 2)],
                detail={"synthetic": True, "revision": letter,
                        "marks": a.get("text", "")})
            placed += 1
            break


def _emit_flag_notes(sh: PSheet, meta, notes, style: Style, margin, W):
    """Flag the note number on the feature the note is about.

    A flag note is a numbered triangle on a leader, pointing at the place a
    numbered general note applies: "3" on a corner means note 3 -- BREAK
    SHARP EDGES -- is about that corner. It is one of the commonest marks on
    an issued drawing and the corpus had none of them.

    Nothing is invented here: the flag repeats a note number that is already
    written out in the block below, and it is only drawn for notes that
    describe a FEATURE of the part rather than the drawing (an edge-break or
    burr note, not "DIMENSIONS IN MM"). The triangle is drawn as geometry,
    like the GD&T frames, because no font has the glyph.
    """
    if not getattr(style, "flag_notes", False) or not notes:
        return
    wanted = ("BREAK SHARP", "BURR", "EDGE", "DEBURR")
    idx = [i for i, n in enumerate(notes)
           if any(w in str(n).upper() for w in wanted)]
    if not idx:
        return
    number = idx[0] + 1
    t = style.dim_text_height
    half = t * 1.2          # the digit has to sit INSIDE the triangle
    placed = 0
    for name, pl in list(sh.placed_views.items()):
        if placed >= 2:
            break
        v = pl.ann.view
        # Corners and edge midpoints of the silhouette -- an edge-condition
        # note is about the outline, and the corners alone are exactly where
        # the witness lines run, so on a dimensioned view none of them were
        # ever free and no flag was drawn at all.
        x0, y0 = pl.m2p((v.xmin, v.ymin))
        x1, y1 = pl.m2p((v.xmax, v.ymax))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        anchors = [(x0, y0), (x1, y0), (x0, y1), (x1, y1),
                   (cx, y0), (cx, y1), (x0, cy), (x1, cy)]
        done = False
        for a in anchors:
            if done:
                break
            ux = 1.0 if a[0] > cx else (-1.0 if a[0] < cx else 0.0)
            uy = 1.0 if a[1] > cy else (-1.0 if a[1] < cy else 0.0)
            for mult in (3.2, 5.0, 7.0):
                tip = (a[0] + ux * t * mult, a[1] + uy * t * mult)
                # The anchor sits ON the outline by definition, so testing a
                # box that reaches back to it can never come out empty --
                # which is why the first version of this drew no flags at
                # all. Test the TRIANGLE, and the far half of the leader.
                box = (tip[0] - half * 1.4, tip[1] - half * 1.4,
                       tip[0] + half * 1.4, tip[1] + half * 1.6)
                # The whole leader bar a stub at the arrow, which is on the
                # outline by definition. Testing only its far half let a flag
                # leader cross a datum letter on the way out.
                near = (a[0] + (tip[0] - a[0]) * 0.18,
                        a[1] + (tip[1] - a[1]) * 0.18)
                lead = (min(near[0], tip[0]) - 0.4, min(near[1], tip[1]) - 0.4,
                        max(near[0], tip[0]) + 0.4, max(near[1], tip[1]) + 0.4)
                if box[0] < margin or box[2] > W - margin:
                    continue
                if box[1] < margin or box[3] > sh.h - margin:
                    continue
                if not (_ink_free(sh, box) and _ink_free(sh, lead)):
                    continue
                # Stop the leader ON the triangle, not at its centre: run
                # into the middle and the line crosses the digit it points
                # to, which is both wrong on paper and a self-inflicted
                # "text crossed by a line" in the layout tests.
                _L = math.hypot(tip[0] - a[0], tip[1] - a[1]) or 1.0
                stop = (tip[0] - (tip[0] - a[0]) / _L * half * 1.15,
                        tip[1] - (tip[1] - a[1]) / _L * half * 1.15)
                sh.add(PLine(a, stop, layer="LEADER"),
                       PArrow(a, stop, style.arrow, kind=style.arrow_style))
                tri = [(tip[0], tip[1] + half * 1.25),
                       (tip[0] - half * 1.1, tip[1] - half * 0.85),
                       (tip[0] + half * 1.1, tip[1] - half * 0.85)]
                sh.add(PPoly(tri, layer="DIM", close=True))
                num = PText((tip[0], tip[1] - half * 0.45), str(number),
                            t * 0.75, ha="center", layer="DIM")
                sh.add(num)
                fbox = (tip[0] - half * 1.1, tip[1] - half * 0.85,
                        tip[0] + half * 1.1, tip[1] + half * 1.25)
                sh.note_annotation(
                    kind="leader_note", view=name, text=f"FLAG NOTE {number}",
                    text_box=fbox,
                    extent=_union_box([fbox, (min(a[0], tip[0]),
                                              min(a[1], tip[1]),
                                              max(a[0], tip[0]),
                                              max(a[1], tip[1]))]),
                    attach=[a],
                    detail={"flag_note": number,
                            "note": str(notes[number - 1])})
                done = True
                break
        if done:
            placed += 1


def _general_tolerance_notes(style: Style):
    """The lines an UNLESS OTHERWISE SPECIFIED block states.

    These are real drafting boilerplate: the unit, the general tolerance that
    governs every unmarked size, and the edge-condition note. They are
    generic (not measured from this part), which is exactly what the heading
    says they are.

    Two conventions, because the corpus should contain both. An ISO shop
    quotes a general tolerance CLASS (ISO 2768-m) and lets the standard say
    what that means for each size band. An ASME shop writes the block out by
    DECIMAL PLACES -- X.X +/-0.2, X.XX +/-0.1 -- so the number of digits a
    dimension carries is itself the tolerance, which is why an inch drawing
    writes 2.500 and not 2.5.
    """
    grade = "ISO 2768-m" if getattr(style, "decimals", 2) < 3 else "ISO 2768-f"
    lin = "0.2" if getattr(style, "decimals", 2) < 3 else "0.05"
    common = ["BREAK SHARP EDGES 0.3 MAX; REMOVE ALL BURRS",
              "DO NOT SCALE FROM THIS DRAWING"]
    if getattr(style, "tolerance_block", "class") == "decimal":
        return [
            f"DIMENSIONS IN {UNITS.upper()}",
            "TOLERANCES UNLESS NOTED:  X.X \u00b10.5   X.XX \u00b10.25   "
            "X.XXX \u00b10.05",
            "ANGLES \u00b10\u00b0 30'",
        ] + common
    return [
        f"DIMENSIONS IN {UNITS.upper()}",
        f"GENERAL TOLERANCE {grade}: LINEAR \u00b1{lin}, ANGULAR \u00b10.5\u00b0",
    ] + common


def _parts_height(style, rows):
    """How tall the parts list will be, so a top block can hang it below."""
    t = style.note_text_height * getattr(style, "title_text_scale", 1.0) * 0.8
    return (len(rows) + 1) * max(t * 1.9, 4.2)


def _rev_rows(meta):
    """Revision history rows. Generated -- a CAD file records no history."""
    from .variation import _Rng, _seed
    rng = _Rng(_seed("rev:" + str(meta.get("part_number", ""))))
    rev = str(meta.get("revision") or meta.get("rev") or "A").strip() or "A"
    why = ("FIRST ISSUE", "TOLERANCES REVISED", "HOLE SIZE CHANGED",
           "MATERIAL CHANGED", "NOTES UPDATED", "GENERAL REVISION")
    date = str(meta.get("date", ""))
    who = str(meta.get("drawn", "AD"))[:2].upper()

    # The history has to reach the revision the block states. Offices letter
    # (A, B, C) and offices number (01, 02, 03); reading only rev[0] against
    # "ABCDEFG" gave every numbered drawing a single FIRST ISSUE row while
    # its title block said revision 03, and the change marks on the views
    # pointed at a revision the table had never heard of.
    letters = "ABCDEFG"
    if rev.isdigit():
        n = max(1, min(int(rev), 4))
        ids = [f"{i + 1:02d}" for i in range(n)]
    elif rev[0] in letters:
        ids = list(letters[:letters.index(rev[0]) + 1])
    else:
        ids = ["-"]
    rows = []
    for i, ident in enumerate(ids[-3:]):
        first = (i == 0 and ident in ("A", "01", "-"))
        rows.append([ident, why[0] if first else rng.pick(why[1:]),
                     date, who])
    return rows


def _parts_rows(info, meta):
    """Parts list rows: one per solid in the file, plus the part itself."""
    name = str(meta.get("title") or meta.get("part_number") or "PART").upper()
    pn = str(meta.get("part_number", ""))
    n_solids = int(getattr(info, "n_solids", 1) or 1)
    mat = str(meta.get("material") or "").upper()
    if n_solids > 1:
        return [[str(i + 1), "1", f"{name} BODY {i + 1}", f"{pn}-{i + 1:02d}"]
                for i in range(min(n_solids, 4))]
    return [["1", "1", f"{name}   {mat}".strip(), pn]]


def _emit_general_notes(sh: PSheet, meta, margin, tb_w, tb_h, style: Style):
    """Record the numbered NOTES: block as detection objects.

    These lines are drawn on every sheet by ``draw_title_block`` but were
    never emitted as annotations, so the ``notes`` class had no instances at
    all even though the text was right there on the page. The geometry here
    mirrors that function exactly -- same origin, same line pitch, same text
    heights -- because the box has to land on the ink actually drawn.

    The "NOTES:" heading is included: it is part of what a reader identifies
    as the notes block, and a detector asked to find the block should find its
    title too.
    """
    notes = meta.get("notes", []) or []
    from .sheet import notes_heading
    heading = notes_heading(style)
    if not notes or not heading:
        return
    t = style.note_text_height
    # The notes block sits directly above the title block, so it must follow
    # it to whichever corner the style put it in. Hardcoding bottom-right
    # left the recorded boxes 281 mm from the ink on a bottom-left sheet.
    # Where the ink is. Derived once in _draw_sheet_furniture and passed in
    # meta; the old local formula tested only for a BOTTOM-LEFT block and put
    # the recorded boxes 74 mm right of the text on a top-left one.
    from .sheet import title_block_origin
    x0 = float(meta.get("notes_x")
               or title_block_origin(sh, style, margin, tb_w, tb_h)[0])
    ny = float(meta.get("notes_base", margin + tb_h + 6))
    # These PTexts are MEASURED here rather than added through sh.add, so
    # they never get the sheet's font stamped on them and must carry it
    # explicitly -- otherwise the recorded box uses the default 0.62 while
    # the ink is drawn in the sheet's real font. That mislabelled the whole
    # notes block by up to 13.7 mm on a narrow font.
    # draw_title_block draws these notes at note_text_height * the title
    # block's own scale. Measuring them at the unscaled height reported a box
    # up to 26 mm wider than the ink -- the same duplicated-constant trap the
    # font factor had. One value, applied in both places.
    t = style.note_text_height * getattr(style, "title_text_scale", 1.0)
    cw = sh.char_w
    # A long note is word-wrapped so it cannot run off the sheet. The wrapped
    # lines come from sheet.note_lines -- the same call draw_title_block makes
    # -- because this function must measure the strings that were actually
    # drawn. Numbering the unwrapped notes here would record one box per note
    # while the ink shows several lines.
    from .sheet import note_lines
    # These PTexts never go through sh.add, so they carry neither the sheet's
    # char_w nor its font family; both must be passed explicitly or the tight
    # box is measured in the wrong face.
    _fam = getattr(sh, "font_family", _DEFAULT_FAMILY)
    lines = note_lines(notes, (sh.w - margin) - x0, t * 0.9, cw)
    entries = [(PText((x0, ny + len(lines) * t * 1.6), heading, t,
                      char_w=cw, font_family=_fam), heading)]
    for i, txt in enumerate(lines):
        entries.append(
            (PText((x0, ny + (len(lines) - 1 - i) * t * 1.6), txt, t * 0.9,
                   char_w=cw, font_family=_fam), txt))
    for pt, txt in entries:
        box = _text_box(pt, grow=0.0)
        sh.note_annotation(kind="note", view="", text=txt, text_box=box,
                           extent=box, attach=[],
                           detail={"placement": "general_notes"})


def _general_roughness(sh: PSheet, info, style: Style, W, margin, tb_w, tb_h,
                       n_notes, base=None, below=False, x0=None):
    """The drawing-wide surface-finish note, drawn above the general notes.

    Real drawings state a blanket finish once, near the title block, rather
    than repeating a symbol on every face. It is drawn as the ISO 1302 tick
    followed by the Ra value, which is what a reader looks for.

    The Ra value is assigned, not measured -- see autodraft/roughness.py --
    and the emitted annotation carries ``synthetic: true`` to say so.
    """
    if not getattr(style, "roughness_general_note", True):
        return
    if getattr(style, "notes_style", "numbered") == "none":
        return
    general = [r for r in getattr(info, "roughness", []) or [] if r.general]
    if not general:
        return
    r = general[0]
    # Sits with the notes block, so it uses the same scaled height; the
    # vertical stacking below is computed from it and would otherwise drift
    # away from the notes it sits above.
    t = style.note_text_height * getattr(style, "title_text_scale", 1.0)
    # Follows the title block's corner, as the notes block does -- and by the
    # same single derivation, not a second copy of the arithmetic.
    if x0 is None:
        from .sheet import title_block_origin
        x0 = title_block_origin(sh, style, margin, tb_w, tb_h)[0] + 1.5
    # Sit clear of the notes block. From a bottom-corner title block the note
    # stack grows upward, so the finish note caps it; from a top-corner block
    # the stack HANGS DOWN, and stacking upward here drove the tick straight
    # through the title block's bottom row (seen on 0000_00000126). ``below``
    # says which way the notes ran.
    nb = base if base is not None else margin + tb_h + 6
    size = t * 1.6
    glyph_h = size * 1.24 * 0.866        # long leg rise, see _roughness_glyph
    y = (nb - glyph_h - t * 1.2 if below
         else nb + (n_notes + 1) * t * 1.6 + t * 1.2)

    def _parts(origin_x, origin_y):
        """Build the note primitives and exact box without committing them."""
        gx = origin_x + size * 0.4
        p_short, p_long = _roughness_glyph(sh, gx, origin_y, size,
                                           r.machined, layer="TEXT",
                                           commit=False)
        te = PText((p_long[0] + 1.2, p_long[1] - t * 0.55), r.text(), t,
                   ha="left", layer="TEXT")
        tail = PText((p_long[0] + 1.2 + len(r.text()) * t * style.char_w + 2.0,
                      p_long[1] - t * 0.55), "ALL OVER", t * 0.85,
                     ha="left", layer="TEXT")
        # The box must cover every piece of ink this note draws: the vee, the
        # Ra text AND the "ALL OVER" tail. Measuring against x0 (the title
        # block edge) rather than the glyph vertex, and dropping the tail,
        # left the recorded box 3.3 mm off the ink on 0000_00000007.
        glyphs = [PLine((gx, origin_y), p_short, layer="TEXT"),
                  PLine((gx, origin_y), p_long, layer="TEXT")]
        if r.machined:
            glyphs.append(PLine(p_short, (p_long[0], p_short[1]), layer="TEXT"))
        box = _union_box([_text_box(te, grow=0.0), _text_box(tail, grow=0.0),
                          (min(gx, p_short[0]), origin_y,
                           max(p_long[0], gx), p_long[1])])
        return glyphs, te, tail, box

    base_parts = _parts(x0, y)
    base_box = base_parts[-1]
    # General finish notes are sheet furniture, but the view fitter can use
    # the last few millimetres of the furniture band for a large front view.
    # Search the actual paper for a clear placement before emitting the note;
    # this is especially important for a top-corner title block, where the
    # note stack hangs directly above the front view. Keep the note near its
    # furniture anchor, while allowing it to move to the free side of a view.
    view_boxes = []
    for placed in getattr(sh, "placed_views", {}).values():
        v = placed.ann.view
        pl_, pr_, pb_, pt_ = placed.ann.pad
        a = placed.m2p((v.xmin - pl_, v.ymin - pb_))
        b = placed.m2p((v.xmax + pr_, v.ymax + pt_))
        view_boxes.append((min(a[0], b[0]), min(a[1], b[1]),
                           max(a[0], b[0]), max(a[1], b[1])))

    def _overlaps(a, b, gap=1.5):
        return (a[0] < b[2] + gap and a[2] > b[0] - gap
                and a[1] < b[3] + gap and a[3] > b[1] - gap)

    note_w = base_box[2] - base_box[0]
    view_left = min((b[0] for b in view_boxes), default=margin)
    view_right = max((b[2] for b in view_boxes), default=W - margin)
    x_origins = [
        x0,
        x0 + (view_right + 3.0 - base_box[0]),
        x0 - (base_box[2] - (view_left - 3.0)),
        W - margin - note_w - 2.0 - (base_box[0] - x0),
        margin + 2.0 - (base_box[0] - x0),
    ]
    y_positions = [y]
    for delta in (-8.0, 8.0, -16.0, 16.0, -24.0, 24.0):
        y_positions.append(y + delta)

    chosen = None
    for origin_x in x_origins:
        for origin_y in y_positions:
            parts = _parts(origin_x, origin_y)
            box = parts[-1]
            if (box[0] < margin or box[1] < margin
                    or box[2] > W - margin or box[3] > sh.h - margin):
                continue
            if any(_overlaps(box, vb) for vb in view_boxes):
                continue
            if not _ink_free(sh, box):
                continue
            chosen = parts
            break
        if chosen is not None:
            break
    # Preserve the note even on an unusually saturated sheet; the normal path
    # above always wins for the uploaded reproduction and keeps the annotation
    # content instead of silently dropping a stated finish requirement.
    glyphs, te, tail, box = chosen or base_parts
    sh.add(*glyphs, te, tail)
    sh.note_annotation(
        kind="roughness", view="", text=r.text(),
        value=float(r.text().split()[-1]),
        text_box=box, extent=box, attach=[],
        detail={"synthetic": True, "machined": r.machined, "general": True})


def _proj_symbol(sh: PSheet, x, y, mode: str, scale: float = 1.0):
    """The truncated-cone projection symbol, with its convention named below.

    The label used to be dropped on top of the cone at a fixed 2 mm; it is now
    centred underneath and scaled with the symbol, which is how it is drawn on
    a real sheet.
    """
    r1, r2 = 3.0 * scale, 1.6 * scale
    cy = y + 4 * scale
    sh.add(PCircle((x, cy), r1, layer="FRAME"), PCircle((x, cy), r2, layer="FRAME"))
    d = 8 * scale if mode == "third" else -8 * scale
    tip = (5 * scale if d > 0 else -5 * scale)
    sh.add(PPoly([(x + d, cy + r1), (x + d + tip, cy + r2),
                  (x + d + tip, cy - r2), (x + d, cy - r1)],
                 layer="FRAME", close=True))
    sh.add(PText((x + d / 2, y - 2.6 * scale),
                 "THIRD ANGLE" if mode == "third" else "FIRST ANGLE",
                 2.0 * scale, ha="center", layer="TEXT"))
