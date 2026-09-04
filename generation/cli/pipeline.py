"""High-level API: STEP/IGES/BREP in -> drawing PNG + detection labels out."""
from __future__ import annotations

import os

from typing import List, Optional

from . import export
from .annotate import annotate, holes_in_view
from .geometry import UNITS, analyse, load
from .sheet import scale_text
from .projection import choose_views, project
from .render import build_sheet
from .sheet import Style
from .dataset import DEFAULT_TEST_FRAC, DEFAULT_VAL_FRAC


def _version() -> str:
    """Package version, read lazily to avoid a circular import."""
    from . import __version__
    return __version__


def _write_png(sh, out_base: str, dpi: int) -> dict:
    """Render the sheet. The PNG is the only image the labels describe."""
    path = f"{out_base}.png"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    export.to_png(sh, path, dpi=dpi)
    return {"png": path}


def _rounded_annotations(sh) -> list:
    """Annotations as plain dicts, with coordinates rounded for the record."""
    def box(b):
        return [round(v, 3) for v in b] if b else None

    def value(key, v):
        if key in ("text_box", "extent"):
            return box(v)
        if key == "attach":
            return [[round(c, 3) for c in p] for p in v]
        return v

    return [{k: value(k, v) for k, v in a.items()}
            for a in getattr(sh, "annotations", [])]


def _attach_record(out: dict, out_base: str, *, model_path, sheet, sh, style,
                   meta, write: bool):
    """Build the annotation record, and write it when asked.

    The record is always BUILT: the COCO writer is fed from it, so switching
    the file off must not leave it with nothing.
    """
    from .schema_adapter import annotation_record

    rec = annotation_record(
        out, source_path=model_path, sheet_size=sheet, sheet_wh=(sh.w, sh.h),
        scale=sh.scale, scale_text=scale_text(sh.scale),
        projection=style.projection, notes=list(meta.get("notes", [])),
        view_fill=getattr(sh, "fill", None),
        view_origins=out.get("view_origins"), generator_version=_version())
    out["annotation_record"] = rec
    if write:
        path = f"{out_base}.annotations.json"
        with open(path, "w") as f:
            f.write(rec.model_dump_json(indent=2))
        out["files"]["annotations"] = path


# Sheet sizes the layout may escalate through, smallest first.
SHEET_LADDER = ["A4", "A3", "A2", "A1", "A0"]
#: The same ladder, turned. A portrait drawing escalates to bigger portrait
#: sheets rather than flipping to landscape halfway up.
SHEET_LADDER_P = [s + "P" for s in SHEET_LADDER]


def _ladder_for(sheet: str):
    return SHEET_LADDER_P if str(sheet).endswith("P") else SHEET_LADDER


def _sheet_area(name: str) -> float:
    from .sheet import SHEETS
    w, h = SHEETS.get(name, (420.0, 297.0))
    return w * h


def _primary_round_diameter_mm(sh) -> Optional[float]:
    """Return the paper diameter of a circular primary view, if present."""
    placed = getattr(sh, "placed_views", {})
    primary = next(iter(placed.values()), None)
    if primary is None:
        return None
    from .annotate import outer_diameter
    if outer_diameter(primary.ann.view) is None:
        return None
    # Use the projected silhouette rather than the ideal CAD diameter. HLR
    # and edge tolerances can trim a fraction of a millimetre; the readable
    # size is the width that actually lands on the sheet.
    return primary.ann.view.width * sh.scale


def _choose_sheet(sh, sheet, best_on, style):
    """Pick the sheet size AND orientation the drawing actually fills.

    Two questions, and the old code got both wrong. It only ever grew the
    sheet, and it judged a candidate by whether the scale improved -- but a
    bigger sheet at a better scale can still leave two thirds of the paper
    blank, which is the most machine-made thing about a generated drawing.

    Real offices pick the smallest sheet the views fit on legibly, and turn
    the paper to suit the part. Both neighbours and both orientations are
    therefore laid out for real and scored: smallest sheet AREA first, fill as
    the tie-break, among candidates that draw at a legible scale. Circular
    primary views also have to clear the style's paper-space diameter minimum;
    a cluster of radial callouts around a thumbnail is not a readable drawing.
    Judging by fill alone cannot work -- view area grows with the square of the
    scale, so "most full" always answers A0.

    Returns the chosen (sheet, name).
    """
    ladder = _ladder_for(sheet)
    if sheet not in ladder:
        return sh, sheet
    i = ladder.index(sheet)
    names = [ladder[j] for j in (i - 1, i, i + 1, i + 2)
             if 0 <= j < len(ladder)]
    # ...and the turned form of each, because a tall part on landscape paper
    # is the commonest reason a sheet looks empty
    turned = []
    for n in names:
        t = n[:-1] if n.endswith("P") else n + "P"
        from .sheet import SHEETS
        if t in SHEETS:
            turned.append(t)
    cands = [(sheet, sh)]
    for name in names + turned:
        if name == sheet:
            continue
        try:
            cands.append((name, best_on(name)))
        except Exception:
            pass

    # ORIENTATION. The drawing was asked for on a particular way round (the
    # house style's sheet, turned or not by the per-drawing jitter in
    # variation.vary_style). A 15% fill bonus was too weak to defend that
    # choice: measured over the 24-part corpus it still returned 20 portrait
    # sheets to 4 landscape, because a two-view drawing stacks into a tall
    # block and the turned sheet therefore always filled better.
    #
    # So orientation is now a RANKING KEY, not a nudge -- with one escape
    # hatch: if turning the paper improves the fill by more than a third,
    # the part really does not suit the sheet and the drawing is turned.
    # That is the same judgement a draughtsman makes.
    pref_portrait = str(getattr(style, "sheet_pref", sheet)).endswith("P")

    def same_way(name):
        return name.endswith("P") == pref_portrait

    def rank(nc):
        return (round(_sheet_area(nc[0]), 1), 0 if same_way(nc[0]) else 1,
                -round(nc[1].fill, 3))

    def round_view_is_readable(nc):
        """Reject a sheet that makes a circular primary view thumbnail-sized."""
        minimum = float(getattr(style, "min_round_view_mm", 0.0) or 0.0)
        if minimum <= 0:
            return True
        diameter = _primary_round_diameter_mm(nc[1])
        return diameter is None or diameter >= minimum

    ok = [nc for nc in cands
          if nc[1].scale >= style.min_scale and nc[1].fill >= style.min_view_fill
          and round_view_is_readable(nc)]
    if ok:
        best_same = max((nc[1].fill for nc in ok if same_way(nc[0])),
                        default=0.0)
        best_other = max((nc[1].fill for nc in ok if not same_way(nc[0])),
                         default=0.0)
        if best_same > 0 and best_other <= best_same * 1.34:
            ok = [nc for nc in ok if same_way(nc[0])]
        name, best = min(ok, key=rank)
        # READABILITY. "The views fit" and "the drawing can be read" are
        # different questions and the bars above only ask the first. Measured
        # over the corpus, label DENSITY separates them cleanly: the median
        # sheet carries 0.15 labels per square centimetre of view and reads
        # easily, while 0000_00000093 carried 5.28 -- eleven callouts ringing
        # a 27 x 25 mm view, every one correctly placed and the whole thing
        # unreadable. (Collisions do not catch this: the placer avoids them,
        # so the same sheet had two.)
        #
        # A draughtsman answers this by drawing the part bigger. So does the
        # chooser: if the sheet it picked is over the office's density limit,
        # take the smallest sheet that comes in under it. Never the other
        # way -- a roomy drawing is never rejected for being roomy.
        limit = float(getattr(style, "max_label_density", 1.0) or 0.0)
        if limit > 0 and getattr(best, "label_density", 0.0) > limit:
            roomy = [nc for nc in ok
                     if getattr(nc[1], "label_density", 0.0) <= limit]
            if roomy:
                name, best = min(roomy, key=rank)
        return best, name
    # Nothing clears both bars. Take the smallest sheet that still draws the
    # part within a third of the best scale available: growing the paper for
    # the last few percent of scale is exactly what leaves a sheet empty.
    top = max(c.scale for _n, c in cands)
    near = [nc for nc in cands if nc[1].scale >= top * 0.7]
    name, best = min(near or cands, key=rank)
    return best, name


def _settle_scale(sh, anns, seed, reannotate, layout, *, valid=None):
    """Find a scale/annotation pair that agrees with itself."""
    from .sheet import PREFERRED_SCALES
    ceiling = max(sh.scale, seed) * 1.6
    for rung in sorted({n / d for n, d in PREFERRED_SCALES}, reverse=True):
        if rung > ceiling:
            continue
        candidate = reannotate(rung)
        laid_out = layout(candidate)
        if valid is not None and not valid(laid_out):
            continue
        if laid_out.scale >= rung - 1e-9:
            return laid_out, candidate

    candidate = reannotate(sh.scale)
    laid_out = layout(candidate)
    return laid_out, candidate

def _arrow_box(p):
    """Conservative paper-space bounds for a rendered ``PArrow``."""
    import math
    dx, dy = p.tail[0] - p.tip[0], p.tail[1] - p.tip[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    width = p.size * 0.16
    base = (p.tip[0] + ux * p.size, p.tip[1] + uy * p.size)
    pts = (p.tip, (base[0] + px * width, base[1] + py * width),
           (base[0] - px * width, base[1] - py * width), p.tail)
    return (min(q[0] for q in pts), min(q[1] for q in pts),
            max(q[0] for q in pts), max(q[1] for q in pts))


def _sheet_ink_overflow(sh, tolerance=0.01):
    """Return the largest finished-primitive overflow past the frame."""
    from .render import _prim_box
    from .sheet import PArrow, PPoly
    frame = [_prim_box(p) for p in sh.prims
             if getattr(p, "layer", "") == "FRAME"
             and isinstance(p, PPoly)]
    frame = [b for b in frame if b is not None]
    if frame:
        # The largest closed FRAME polyline is the drawing border. Centring
        # marks deliberately extend to the paper edge, and title/revision
        # boxes are FRAME geometry too; unioning all of them would make an
        # escaped leader look contained by the centring marks.
        bounds = max(frame, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    else:
        frame = [_prim_box(p) for p in sh.prims
                 if getattr(p, "layer", "") == "FRAME"]
        frame = [b for b in frame if b is not None]
        if not frame:
            return float("inf")
        bounds = (min(b[0] for b in frame), min(b[1] for b in frame),
                  max(b[2] for b in frame), max(b[3] for b in frame))
    worst = 0.0
    for p in sh.prims:
        if getattr(p, "layer", "") == "FRAME":
            continue
        box = _arrow_box(p) if isinstance(p, PArrow) else _prim_box(p)
        if box is None:
            continue
        worst = max(worst, bounds[0] - box[0], bounds[1] - box[1],
                    box[2] - bounds[2], box[3] - bounds[3])
    return max(0.0, float(worst))


def _drop_outside_ink(sh):
    """Drop an optional primitive that cannot be reflowed onto this sheet.

    This is a last-resort path for an explicitly fixed, undersized sheet (for
    example ``auto_sheet=False`` on a 600 mm part). Geometry is first fitted at
    the smallest ladder rung; if a paper-sized label still conflicts with the
    title-block stack, exporting it outside the frame is worse than omitting
    that optional annotation. Records are pruned along with the ink so labels
    never describe a primitive that was not emitted.
    """
    from .render import _prim_box
    from .sheet import PArrow, PPoly
    frames = [_prim_box(p) for p in sh.prims
              if getattr(p, "layer", "") == "FRAME"
              and isinstance(p, PPoly)]
    frames = [b for b in frames if b is not None]
    if not frames:
        return
    bounds = max(frames, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    def box(p):
        return _arrow_box(p) if isinstance(p, PArrow) else _prim_box(p)

    def inside(b):
        return (b is None or
                (b[0] >= bounds[0] and b[1] >= bounds[1]
                 and b[2] <= bounds[2] and b[3] <= bounds[3]))

    sh.prims = [p for p in sh.prims
                if getattr(p, "layer", "") == "FRAME" or inside(box(p))]
    kept = []
    for a in getattr(sh, "annotations", []):
        b = a.get("extent") or a.get("text_box")
        if inside(b):
            kept.append(a)
    sh.annotations = kept


def _sheet_is_contained(sh, tolerance=0.01):
    """Whether every emitted primitive is inside the drawn border."""
    return _sheet_ink_overflow(sh, tolerance) <= tolerance


def _table_views(info, projections, primary, style, hole_table):
    """Views whose holes go in a tagged table rather than leader callouts.

    Default is leaders: callouts are grouped per feature type ("8X o2.5 DP
    2.73"), so even a 66-hole part needs only a handful. A table is used only
    when the part is so dense that grouped notes still cannot be placed.
    """
    from .annotate import _cluster_by_diameter

    def circles(view):
        return holes_in_view(info, view, projections.get(view))

    if hole_table is None:
        groups_total = sum(len(_cluster_by_diameter(circles(v)))
                           for v in projections)
        hole_table = groups_total > style.max_callouts_per_sheet
        # A drawing office may tabulate holes as a matter of house style, not
        # only when leaders will not fit. Without this the density rule chose
        # a table on 1 of 24 sample parts -- most have 0-1 hole groups, so no
        # threshold could ever trigger one -- and the top/bottom/right table
        # placements were unreachable. Only parts that HAVE holes qualify: a
        # hole table on a part with no holes would be an empty box.
        if not hole_table and getattr(style, "prefer_hole_table", False):
            hole_table = any(len(circles(v)) >= 1 for v in projections)
    if not hole_table:
        return []
    return [v for v in projections if len(circles(v)) >= 3] or [primary]


def _seed_scale(projections, sheet: str) -> float:
    """A first guess at the drawing scale, before any layout has run.

    Annotation sizes are paper millimetres converted into model units, so
    assuming 1:1 on a 1.5 mm fastener makes the pads about ten times the view
    and every later decision is taken from nonsense geometry.
    """
    from .sheet import SHEETS
    w, h = SHEETS.get(sheet, (420.0, 297.0))
    # Usable area, less furniture. These were 0.62/0.55, which under-estimated
    # the paper badly enough that the ladder walk below never even probed the
    # rungs that fit: the seed caps the search, so a timid seed is a small
    # drawing. Measured over the corpus, 0.78/0.70 raises mean sheet fill
    # without pushing any part off its sheet (the layout still has to accept
    # each rung).
    draw_w, draw_h = w * 0.78, h * 0.70
    span_w = max(sum(p.width for p in projections.values()), 1e-9)
    span_h = max(max(p.height for p in projections.values()), 1e-9)
    return max(min(min(draw_w / span_w, draw_h / span_h), 1000.0), 1e-4)


def _isometric(shape, style):
    """The pictorial view, shaded if the style asks for it.

    Hidden-line removal is run even though the hidden edges are thrown away:
    without it HLR reports every sharp edge as visible, so a shaded isometric
    showed the far side of every bore straight through the solid it had just
    painted. Running the classification and keeping only the visible set is
    what makes the pictorial read as a solid.
    """
    proj = project(shape, "iso", hidden=style.iso_shaded)
    if not style.iso_shaded:
        return proj
    # shaded: keep only visible outlines, the facets carry the surfaces
    proj.segs = [g for g in proj.segs if g.kind == "visible"]
    proj.circles = [g for g in proj.circles if g.kind == "visible"]
    try:
        from .shading import shaded_facets
        proj.facets = shaded_facets(shape, "iso")
    except Exception:
        proj.facets = []          # shading is cosmetic: never fatal
    return proj


def _scale_note(info) -> list:
    """A note stating the model was scaled, when it was.

    The drawing is honest about it: a reader who compares a stated 1.5 mm
    against a 0.035 mm CAD file needs to know the geometry was lifted onto a
    drawable size, and by how much.
    """
    k = float(getattr(info, "model_scale", 1.0) or 1.0)
    if abs(k - 1.0) < 1e-9:
        return []
    txt = f"{int(k)}" if abs(k - int(k)) < 1e-9 else f"{k:g}"
    return [f"MODEL SCALED X{txt} FROM SOURCE FILE; DIMENSIONS AS DRAWN"]


def _warnings(info) -> list:
    """Non-fatal caveats about what the drawing can honestly state.

    Both are properties of the geometry against the drawing's conventions, so
    they belong on the record rather than in a log nobody reads.
    """
    out = []
    if getattr(info, "n_solids", 1) > 1:
        out.append(f"source contains {info.n_solids} disjoint bodies; "
                   "dimensions span the overall envelope, not a single part")
    # Dimensions print at two decimals of a millimetre, so a feature under
    # about 0.05 mm is stated to worse than 10% and one below 0.005 mm rounds
    # to "0" outright. Say so rather than let a consumer trust the number.
    b = info.bbox
    smallest = min([d for d in (b.xlen, b.ylen, b.zlen) if d > 0]
                   + [h.diameter for h in info.holes if h.diameter > 0]
                   + [f.radius for f in info.fillets if f.radius > 0],
                   default=None)
    if smallest is not None and smallest < 0.05:
        out.append(f"smallest feature is {smallest:.4f} mm; dimensions are "
                   "rounded to 0.01 mm, so values this size carry significant "
                   "rounding error")
    return out


def _hole_count(info) -> int:
    """Closed bores recognised on the solid, for the CLI's one-line summary."""
    return sum(1 for h in info.holes if getattr(h, "is_closed", True))


def _view_metrics(info, anns) -> dict:
    """Per-view silhouette size and padding, for the record.

    hole_count is a GEOMETRY count -- holes visible as circles in the
    view. Annotation counts are not duplicated here; the annotation
    list is authoritative for those.
    """
    return {
        name: {
            "width": round(a.view.width, 6),
            "height": round(a.view.height, 6),
            "xmin": round(a.view.xmin, 6),
            "ymin": round(a.view.ymin, 6),
            "pad_left": round(a.pad[0], 6),
            "pad_right": round(a.pad[1], 6),
            "pad_bottom": round(a.pad[2], 6),
            "pad_top": round(a.pad[3], 6),
            "hole_count": len(holes_in_view(info, name, a.view)),
        }
        for name, a in anns.items()
    }


def _load_and_style(model_path, style, vary):
    """Read the geometry, then pick this drawing's presentation.

    Order matters twice over. Styling must run AFTER the geometry, because
    the hole-table choice is only meaningful once we know whether the part
    has bores -- spending it on a part with none produced a style that could
    never take effect. It must run BEFORE anything measures text, because the
    font's character-width factor feeds both annotation placement and the
    exported bounding boxes.
    """
    from .variation import vary_style
    from .geometry import (normalising_factor, scale_shape, smallest_feature)
    style = style or Style()
    wp = load(model_path)
    info = analyse(wp)
    # Lift sub-millimetre models onto a drawable size. Half this corpus is
    # under 1 mm across, where two-decimal dimensions state a 0.035 mm feature
    # as "0.04" and a thread lookup has nothing real to match. The factor is a
    # round number and varies per part; it is recorded and stated on the sheet
    # so nobody mistakes the drawn size for the file's.
    stem = os.path.splitext(os.path.basename(model_path))[0]
    factor = normalising_factor(smallest_feature(info), stem)
    if factor != 1.0:
        wp = scale_shape(wp, factor)
        info = analyse(wp)
    info.model_scale = factor
    style = vary_style(
        style, stem,
        enabled=vary,
        has_holes=any(getattr(h, "is_closed", True) for h in info.holes))
    return wp, info, style


def _annotate_views(projections, info, *, style, ht_views, tagged_views,
                    scale):
    """Annotate every projection, sharing one thread registry across the sheet.

    A thread may be drawn on any view whose bore reads as a circle, so the
    registry stops the same designation being stated twice. It is created per
    call, because each annotation pass rebuilds the sheet from scratch.
    """
    out = {}
    thread_ids = set()
    frame_ids = set()
    for i, (name, proj) in enumerate(projections.items()):
        out[name] = annotate(proj, info, style=style, primary=(i == 0),
                             hole_table=(name in ht_views),
                             tagged=(name in tagged_views), scale=scale,
                             thread_ids=thread_ids, frame_ids=frame_ids)
    return out


def _attach_pmi(info, model_path: str, synth_pmi: bool = True) -> None:
    """Attach product & manufacturing information to a recognised part.

    Two different provenances, deliberately kept apart:

    * **PMI is read from the file** and is empty when the file carries none,
      so a plain STEP body simply gets no GD&T and no threads. Threads are
      then tied to the bores they belong to, which is what lets them be placed
      on a view; the pairing never invents a thread.
    * **Surface finish is not in any CAD file** and is not derivable from a
      B-rep, so it is assigned deterministically from the part name purely so
      the roughness class has examples. Every one is flagged synthetic
      downstream -- see autodraft/roughness.py for the full caveat.
    """
    from . import pmi as _pmi
    from . import synth_pmi as _synth
    from .roughness import assign as _assign_roughness
    stem = os.path.splitext(os.path.basename(model_path))[0]
    info.pmi = _pmi.read(model_path)
    if info.pmi:
        _pmi.attach_thread_anchors(info.pmi, info.holes)
    # Only one sample file carries PMI, which left gdnts/threads/chamfers with
    # instances on a single image out of 18 and none at all in valid/test --
    # unlearnable and unevaluatable. The generator fills the gap from REAL
    # geometry (a thread only on a genuine tap-drill-sized bore, and so on)
    # and marks everything it makes synthetic. A file that already carries
    # real PMI is left untouched. See autodraft/synth_pmi.py.
    if synth_pmi:
        info.pmi = _synth.generate(stem, info, info.pmi or _pmi.PMI())
        info.chamfers = (list(info.chamfers)
                         + _synth.generate_chamfers(stem, info))
    info.roughness = _assign_roughness(stem, info)


def _subpart_views(wp, info, style, scale):
    """One annotated orthographic view of each solid in a multi-body file.

    A STEP holding several disjoint solids is a drawing of several parts. The
    assembly views dimension the envelope, which describes none of them, so
    each body gets its own detail view -- a real projection with its own
    dimensions and callouts, captioned ITEM n, the way a multi-part sheet
    carries ITEM 1 BODY / ITEM 2 HANDLE. A pictorial was tried first and is
    not a drawing of the part: it states nothing.
    """
    if not getattr(style, "subpart_views", True):
        return []
    if int(getattr(info, "n_solids", 1) or 1) < 2:
        return []
    import cadquery as cq
    from .annotate import annotate as _annotate
    solids = sorted(wp.solids().vals(), key=lambda so: -abs(so.Volume()))
    out = []
    for i, solid in enumerate(solids[:4]):
        try:
            body_wp = cq.Workplane("XY").newObject([solid])
            body_info = analyse(body_wp)
            view = choose_views(body_info, max_views=1)[0]
            proj = project(solid, view, hidden=style.hidden_lines)
            if proj.width <= 0 or proj.height <= 0:
                continue
            proj.name = view
            ann = _annotate(proj, body_info, style=style, primary=False,
                            hole_table=False, tagged=False, scale=scale)
        except Exception:
            continue                      # a detail view is never fatal
        out.append((f"ITEM {i + 1}", ann))
    return out


def make_drawing(model_path: str, out_base: str, *,
                 sheet: str = "A3",
                 style: Optional[Style] = None,
                 views: Optional[List[str]] = None,
                 max_views: int = 3,
                 scale: Optional[float] = None,
                 meta: Optional[dict] = None,
                 iso: Optional[bool] = None,
                 hole_table: Optional[bool] = None,
                 write_record: bool = True,
                 dpi: int = 200,
                 coco: bool = True,
                 synth_pmi: bool = True,
                 vary: bool = True) -> dict:
    wp, info, style = _load_and_style(model_path, style, vary)
    # A house style has a sheet it reaches for -- including portrait. The
    # caller's choice wins; "A3" is the API default, so it counts as unset.
    sheet_auto = (sheet == "A3")
    if sheet_auto and getattr(style, "sheet_pref", ""):
        sheet = style.sheet_pref

    _attach_pmi(info, model_path, synth_pmi=synth_pmi)

    # Printed precision is a house-style convention, so it is set once here
    # and read by every fmt() call while this drawing is built.
    from .geometry import set_decimals
    set_decimals(getattr(style, "decimals", 2))
    shape = wp.val()

    vlist = views or choose_views(info, max_views=max_views)
    projections = {}
    for i, v in enumerate(vlist):
        p = project(shape, v, hidden=style.hidden_lines)
        if p.width <= 0 or p.height <= 0:
            continue
        projections[v] = p

    primary = list(projections)[0]

    # ---- section view ------------------------------------------------------
    # A part with internal features reads as a thicket of hidden lines in an
    # orthographic view and as solid material in a section, which is why real
    # drawings section them. One SECONDARY view is replaced -- never the
    # primary, which is the view a reader identifies the part by -- and the
    # parent view gets the cutting-plane line.
    # A busy part can carry a SECOND section, lettered B-B: two cuts through
    # different places is ordinary practice on anything with more than one
    # internal feature, and a corpus where every section is called A-A
    # teaches a detector that the letter is decoration. The second one is
    # only taken when there is a secondary view left to give up for it.
    meta_section = None
    meta_sections = []
    if getattr(style, "section_views", True) and len(projections) > 1:
        closed = [h for h in info.holes if getattr(h, "is_closed", True)]
        if closed or int(getattr(info, "n_solids", 1) or 1) > 1:
            from .section import section as _section
            letters = ["A-A", "B-B"]
            cuts = [0.5, 0.32]
            for name in list(projections)[1:]:
                if not letters:
                    break
                sec = _section(shape, name, at=cuts[len(meta_sections)],
                               hidden=False)
                if sec is not None and getattr(sec, "hatch", None):
                    sec.name = name
                    sec.section_id = letters.pop(0)
                    projections[name] = sec
                    meta_sections.append((primary, name, sec.section_id))
                    if meta_section is None:
                        meta_section = (primary, name)
                    if len(meta_sections) >= (2 if getattr(
                            style, "two_sections", False) else 1):
                        break

    # The sheet is NOT turned here any more. This used to compare the tallest
    # view against the widest and swap to portrait whenever the ratio passed
    # 1.25 -- but the view BLOCK, not the view, is what has to fit, and
    # _choose_sheet already lays the drawing out on both orientations for
    # real and measures the fill. The guess ran first and won, which is where
    # most of the corpus-wide portrait skew came from: 20 of 24 sheets
    # portrait, against 12 of 24 house preferences. Measured after removing
    # it: see CHANGES.md.

    ht_views = _table_views(info, projections, primary, style, hole_table)
    tagged_views = set(ht_views)
    _seed = _seed_scale(projections, sheet)

    def _reannotate(sc):
        """Annotate every view at scale `sc`.

        Run once to seed the layout and again once the true scale is known,
        so reserved text space matches what is actually drawn.
        """
        return _annotate_views(projections, info, style=style,
                               ht_views=ht_views, tagged_views=tagged_views,
                               scale=sc)

    anns = _reannotate(_seed)

    want_iso = style.iso_view if iso is None else iso
    iso_proj = _isometric(shape, style) if want_iso else None
    subparts = _subpart_views(wp, info, style, _seed)

    meta = dict(meta or {})
    meta.setdefault("title", os.path.splitext(os.path.basename(model_path))[0].upper())
    meta.setdefault("part_number", os.path.splitext(os.path.basename(model_path))[0])

    # ---- pick a sheet that gives the views enough room ---------------------
    # A dense part on A3 ends up as postage stamps next to a giant table, so
    # step the sheet up until the views occupy a reasonable share of the page.

    from .render import _layout_variants

    def _render(sheet_name, ann_set, blocks=0, slots=None):
        return build_sheet(info, ann_set, sheet=sheet_name, style=style,
                           meta=dict(meta), fixed_scale=scale,
                           section_of=meta_section,
                           sections=meta_sections,
                           iso_proj=iso_proj, hole_table_view=ht_views,
                           subparts=subparts,
                           table_blocks_override=blocks, slots_override=slots)

    def _score(sh_, strict):
        """Bigger drawing is better; nudge toward the strict projection grid."""
        return sh_.scale * (1.0 if strict else 0.94)

    def _best_on(sheet_name, ann_set):
        """Search table aspect x view arrangement; keep the largest drawing."""
        best = _render(sheet_name, ann_set)
        if scale:
            return best
        best_score = _score(best, True)
        variants = (_layout_variants(list(ann_set), style)
                    if style.auto_layout else [None])
        blocks_opts = (0, 2, 3) if ht_views else (0,)
        for vi, slots in enumerate(variants):
            for blocks in blocks_opts:
                if vi == 0 and blocks == 0:
                    continue
                cand = _render(sheet_name, ann_set, blocks, slots)
                sc = _score(cand, vi == 0)
                if sc > best_score * 1.001:
                    best, best_score = cand, sc
        return best

    sh = _best_on(sheet, anns)
    if style.auto_sheet and not scale and sheet in _ladder_for(sheet):
        sh, sheet = _choose_sheet(
            sh, sheet, lambda name: _best_on(name, anns), style)

    if scale:
        anns = _reannotate(scale)
        sh = _best_on(sheet, anns)
    else:
        sh, anns = _settle_scale(
            sh, anns, _seed, _reannotate,
            lambda a: _best_on(sheet, a),
            valid=_sheet_is_contained)

        # The first sheet search is necessarily based on a seed annotation pass;
        # a circular view can lose several scale rungs when its final leaders
        # are measured at the true scale. If that leaves the primary disc below
        # the paper-space minimum, escalate once and settle the larger sheet
        # with the same containment search instead of accepting a tiny, dense
        # front view.
        min_round = float(getattr(style, "min_round_view_mm", 0.0) or 0.0)
        if (getattr(style, "auto_sheet", True) and min_round > 0
                and (_primary_round_diameter_mm(sh) or float("inf")) < min_round):
            ladder = _ladder_for(sheet)
            start = ladder.index(sheet) if sheet in ladder else -1
            for larger in ladder[start + 1:]:
                next_anns = _reannotate(_seed)
                next_base = _best_on(larger, next_anns)
                next_sh, next_anns = _settle_scale(
                    next_base, next_anns, _seed, _reannotate,
                    lambda a, _name=larger: _best_on(_name, a),
                    valid=_sheet_is_contained)
                sh, anns, sheet = next_sh, next_anns, larger
                if ((_primary_round_diameter_mm(sh) or float("inf"))
                        >= min_round):
                    break

    # Explicit caller scales are also checked. Normally the forced layout above
    # is already contained; this final guard protects fixed-scale requests and
    # future renderers that add primitives after the scale search. An explicitly
    # undersized sheet gets a final drop of un-reflowable optional ink rather
    # than exporting a feature or label outside the drawing border.
    if not _sheet_is_contained(sh):
        _drop_outside_ink(sh)
    if not _sheet_is_contained(sh):
        raise RuntimeError(
            f"drawing primitives escaped the {sheet} frame by "
            f"{_sheet_ink_overflow(sh):.3f} mm")

    written = _write_png(sh, out_base, dpi)

    out = {
        "files": written,
        "views": list(projections),
        "hole_table": bool(ht_views),
        "hole_count": _hole_count(info),
        "model_scale": float(getattr(info, "model_scale", 1.0) or 1.0),
        "warnings": _warnings(info),
        "view_metrics": _view_metrics(info, anns),
        "sheet": sheet,
        "sheet_wh": (sh.w, sh.h),
        "scale": sh.scale,
        "scale_text": scale_text(sh.scale),
        "projection": style.projection,
        "units": UNITS,
        "notes": list(meta.get("notes", [])),
    }

    # paper-space origin of each placed view, so a consumer can map view
    # coordinates onto the sheet without guessing
    try:
        out["view_origins"] = {
            name: (round(pl.ox, 6), round(pl.oy, 6))
            for name, pl in getattr(sh, "placed_views", {}).items()
        }
    except Exception:
        out["view_origins"] = {}
    out["iso_box"] = tuple(getattr(sh, "iso_box", ()) or ())

    out["annotations"] = _rounded_annotations(sh)
    # The laid-out sheet itself, for in-process callers that want to inspect
    # what was drawn (the test suite reads it instead of parsing an export).
    # Dropped again before a batch worker ships the result to its parent.
    out["psheet"] = sh

    _attach_record(out, out_base, model_path=model_path, sheet=sheet, sh=sh,
                   style=style, meta=meta, write=write_record)
    _write_label_files(out, written, out_base, coco=coco)

    return out


def _write_label_files(out, written, out_base, *, coco):
    """Write the detection-label sidecars for the rendered image."""
    arec = out.get("annotation_record")
    if arec is None or "png" not in written:
        return
    if coco:
        from .coco import write_coco
        written["coco"] = write_coco(
            arec, written["png"], f"{out_base}.coco.json")


def _one(args):
    """Worker entry point for batch(): returns a result dict, never raises."""
    path, out_dir, kw = args
    stem = os.path.splitext(os.path.basename(path))[0]
    target = out_dir                    # a dataset renders into the staging dir
    base = os.path.join(target, stem)
    try:
        r = make_drawing(path, base, **kw)
        r.pop("psheet", None)          # not worth pickling back to the parent
        r["input"] = path
        r["output_dir"] = target
        r["ok"] = True
    except Exception as e:                     # keep the batch alive
        r = {"input": path, "ok": False, "error": f"{type(e).__name__}: {e}"}
    return r


class _Progress:
    """Minimal progress bar: no dependency, degrades to plain lines.

    Writes to stderr so piping stdout (``--json``) stays clean, and prints
    nothing fancy when the stream is not a TTY (logs, CI).
    """

    def __init__(self, total: int, stream=None, enabled: bool = True):
        import sys
        self.total = max(1, total)
        self.n = 0
        self.stream = stream or sys.stderr
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.enabled = enabled and total > 0
        self._t0 = __import__("time").monotonic()

    def update(self, label: str = "", ok: bool = True) -> None:
        if not self.enabled:
            return
        import time
        self.n += 1
        frac = self.n / self.total
        elapsed = time.monotonic() - self._t0
        eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
        if self.tty:
            width = 28
            filled = int(width * frac)
            bar = "#" * filled + "-" * (width - filled)
            mark = " " if ok else "!"
            msg = (f"\r[{bar}] {self.n}/{self.total} "
                   f"{frac*100:3.0f}% eta {eta:4.0f}s {mark}{label[:34]:<34}")
            self.stream.write(msg)
            self.stream.flush()
        else:
            self.stream.write(f"[{self.n}/{self.total}] "
                              f"{'ok' if ok else 'FAIL'} {label}\n")
            self.stream.flush()

    def close(self) -> None:
        if self.enabled and self.tty:
            self.stream.write("\n")
            self.stream.flush()


# A worker peaks around 0.7 GB while rasterising a sheet (OCCT tessellation
# plus a matplotlib figure at 200 dpi). Measured on the NIST part.
_WORKER_RSS_MB = 700


def _available_mb() -> int:
    """Free memory in MB, or 0 when it cannot be determined."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def _default_jobs(n_paths: int) -> int:
    """How many worker processes to use when the caller did not say.

    Bounded by BOTH cores and memory. A worker costs ~2.2 s to start (spawn
    re-imports cadquery/OCCT) and peaks near 0.7 GB rendering a sheet, so the
    limit is often RAM rather than CPU: on a 2-core box with ~1.2 GB free,
    two workers drove MemAvailable down to 123 MB and the 18-part sample
    batch took 46 s against 36 s serial -- 0.78x, occasionally 0.45x once the
    machine started thrashing. Counting cores alone produced exactly that.

    ``jobs=1`` forces serial, ``jobs=0`` uses every core regardless of memory.
    """
    if n_paths < 2:
        return 1
    cores = os.cpu_count() or 1
    limit = min(cores, n_paths)
    free_mb = _available_mb()
    if free_mb:
        # leave one worker's worth of headroom for the parent and the OS
        by_memory = int((free_mb - _WORKER_RSS_MB) // _WORKER_RSS_MB)
        limit = min(limit, max(1, by_memory))
    return max(1, limit)


def _clear_staging(render_dir: str) -> None:
    """Remove the staging directory once every render has been filed.

    Anything left behind is a sidecar the caller asked for (an annotation
    record); those are moved up to <out>/extra/ rather than deleted, so a
    requested file is never silently thrown away.
    """
    if not os.path.isdir(render_dir):
        return
    leftovers = sorted(os.listdir(render_dir))
    if leftovers:
        extra = os.path.join(os.path.dirname(render_dir), "extra")
        os.makedirs(extra, exist_ok=True)
        for name in leftovers:
            src = os.path.join(render_dir, name)
            try:
                os.replace(src, os.path.join(extra, name))
            except OSError:
                pass
    try:
        os.rmdir(render_dir)
    except OSError:
        pass


def batch(paths, out_dir: str, jobs=None, progress: bool = False,
          val_frac: float = DEFAULT_VAL_FRAC,
          test_frac: float = DEFAULT_TEST_FRAC, **kw) -> List[dict]:
    """Generate a COCO detection dataset from many models.

    Every sheet lands in ``<out_dir>/images/`` and one set of manifests in
    ``<out_dir>/annotations/`` covers all of them, with globally unique image
    and annotation ids. This is the shape a trainer expects, and it is what
    per-part COCO files cannot provide -- each of those numbers its only image
    ``1``, so they cannot be concatenated.

    The per-part sidecars (annotation record, per-part COCO) are off, because
    a dataset directory full of them is noise; the dataset manifest replaces
    them. The annotation record is still built in memory and feeds the split
    manifests.

    jobs > 1 runs parts in separate processes. The work is CPU-bound and each
    part is independent. jobs=0 uses every core.

    Parallelism is NOT free: the "spawn" start method makes each worker import
    cadquery/OCCT from scratch, which costs about 2.2 s. That is only repaid
    when a worker then does substantially more work than that, so on a small
    batch of small parts -- or on a machine with few real cores -- serial is
    faster. Measured on this 2-core box, the whole 18-part sample batch runs
    13.8 s serial against 14.9 s with jobs=0. Use -j on a folder of large
    parts and a machine with cores to spare; the default of 1 is the safe
    choice.

    progress draws a progress bar on stderr.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = list(paths)
    kw = dict(kw)
    # Sheets are rendered into a staging directory, then MOVED into their
    # split folder once the split is known. Rendering straight into train/
    # would mean guessing the split before the part had rendered.
    render_dir = os.path.join(out_dir, ".staging")
    os.makedirs(render_dir, exist_ok=True)
    # The manifests are built from each part's annotation record, which is
    # always built in memory; only the per-part sidecars are off, because a
    # dataset directory full of them is noise.
    kw.setdefault("write_record", False)
    kw["coco"] = False            # the dataset manifest replaces these
    if jobs is None:
        jobs = _default_jobs(len(paths))
    elif jobs == 0:
        jobs = os.cpu_count() or 1
    elif jobs < 0:
        raise ValueError(f"jobs must be >= 0, got {jobs} "
                         "(0 = one per core, 1 = serial)")

    bar = _Progress(len(paths), enabled=progress)
    results: List[dict] = []
    try:
        if jobs > 1 and len(paths) > 1:
            import multiprocessing as mp
            from concurrent.futures import (ProcessPoolExecutor,
                                            as_completed)
            work = [(p, render_dir, dict(kw)) for p in paths]
            # Each spawned worker pays ~2.2 s to import cadquery/OCCT before
            # it can do anything. Handing them one file at a time makes that
            # cost recur; with chunks, a worker imports once and then runs
            # several parts. Chunks are kept small enough that the slowest
            # part cannot leave a core idle at the end of the run.
            # NOTE: ex.map(chunksize=...) was tried and REVERTED. Handing
            # workers several files at once should amortise their ~2.2 s
            # cadquery/OCCT import, but measured 0.80-0.92x against serial for
            # every chunk size from 1 to 9 -- worse than no chunking at all.
            try:
                # "spawn", not the default fork: OCCT holds native state and
                # this process is multi-threaded, where fork() risks deadlocks.
                ctx = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=min(jobs, len(paths)),
                                         mp_context=ctx) as ex:
                    # as_completed, not ex.map: map yields in SUBMISSION
                    # order, so one slow part at the head of the queue holds
                    # back every result behind it and the bar sits at 0% and
                    # then jumps to 100%. Submitting individually and
                    # consuming as they finish makes the bar track real
                    # progress. Results are re-sorted below so the caller
                    # still sees a stable order.
                    futures = {ex.submit(_one, w): w[0] for w in work}
                    for fut in as_completed(futures):
                        r = fut.result()
                        results.append(r)
                        bar.update(os.path.basename(r.get("input", "")),
                                   ok=r.get("ok", False))
                # completion order is nondeterministic; the dataset builder
                # and the caller both want a stable order.
                order = {p: i for i, p in enumerate(paths)}
                results.sort(key=lambda r: order.get(r.get("input"), 0))
                return results
            except Exception:
                results = []          # pool failed: fall through to serial
        for p in paths:
            r = _one((p, render_dir, dict(kw)))
            results.append(r)
            bar.update(os.path.basename(p), ok=r.get("ok", False))
        return results
    finally:
        bar.close()
        from .dataset import build_dataset
        summary = build_dataset(results, out_dir, val_frac=val_frac,
                                test_frac=test_frac)
        _clear_staging(render_dir)
        for r in results:
            r["dataset"] = summary
