"""Detail views: a magnified circular window on a crowded region.

`DETAIL A (2:1)` is the other view type every real drawing has and a generated
one did not. Where a feature is too small to letter at the sheet scale -- a
counterbore on a 3 mm boss, a chamfer at the end of a slot -- a draughtsman
rings it on the parent view and redraws that circle enlarged elsewhere.

The window is chosen by density: the part of the view carrying the most
geometry in the smallest area is the part the reader cannot make out, which is
the part worth magnifying.
"""
from __future__ import annotations

import math
from typing import List, Tuple


def _clip_polyline_to_circle(pts, c, r):
    """Split a polyline into the runs that lie inside a circle."""
    out, run = [], []
    for a, b in zip(pts, pts[1:]):
        ina = math.hypot(a[0] - c[0], a[1] - c[1]) <= r
        inb = math.hypot(b[0] - c[0], b[1] - c[1]) <= r
        if ina:
            run.append(a)
        if ina != inb:
            hit = _circle_hit(a, b, c, r)
            if hit:
                run.append(hit)
            if run and len(run) > 1:
                out.append(run)
            run = [] if ina else [hit] if hit else []
    if pts:
        last = pts[-1]
        if math.hypot(last[0] - c[0], last[1] - c[1]) <= r:
            run.append(last)
    if len(run) > 1:
        out.append(run)
    return out


def _circle_hit(a, b, c, r):
    """Where segment ab crosses the circle (c, r); None if it does not."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    fx, fy = a[0] - c[0], a[1] - c[1]
    A = dx * dx + dy * dy
    if A < 1e-18:
        return None
    B = 2 * (fx * dx + fy * dy)
    C = fx * fx + fy * fy - r * r
    disc = B * B - 4 * A * C
    if disc < 0:
        return None
    disc = math.sqrt(disc)
    for t in ((-B - disc) / (2 * A), (-B + disc) / (2 * A)):
        if 0.0 <= t <= 1.0:
            return (a[0] + dx * t, a[1] + dy * t)
    return None


def _small_features(proj, scale: float, small_mm: float):
    """Features too small to letter at ``scale``, as (x, y, radius) in view units.

    "Too small" is measured ON PAPER, which is the only place the question
    means anything: a 1.1 mm bore drawn at 15:1 is 17 mm across and needs no
    help, while the same bore at 1:2 is half a millimetre and its callout
    text is thirty times the size of the thing it points at. ``small_mm`` is
    that paper threshold.

    Two kinds are collected:

    * circles and arcs whose drawn diameter is under the threshold -- bores,
      fillets, chamfer arcs, thread reliefs;
    * runs of short polyline edges -- a step, a groove or a chamfer flat,
      which have no radius but are just as unreadable.
    """
    small = small_mm / max(scale, 1e-9)          # view units
    out = []
    for seg in proj.segs + proj.circles:
        if seg.kind != "visible":
            continue
        if seg.is_circle and seg.center and seg.radius:
            if seg.radius * 2.0 <= small:
                out.append((seg.center[0], seg.center[1], seg.radius))
            continue
        pts = seg.pts
        run = []
        for a, b in zip(pts, pts[1:]):
            if math.hypot(b[0] - a[0], b[1] - a[1]) <= small * 0.6:
                if not run:
                    run.append(a)
                run.append(b)
            else:
                if len(run) >= 3:
                    xs = [q[0] for q in run]; ys = [q[1] for q in run]
                    out.append(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                                max(max(xs) - min(xs), max(ys) - min(ys)) / 2))
                run = []
        if len(run) >= 3:
            xs = [q[0] for q in run]; ys = [q[1] for q in run]
            out.append(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                        max(max(xs) - min(xs), max(ys) - min(ys)) / 2))
    return out


def _dim_anchor(d):
    """The point on the geometry a dimension object refers to, or None.

    A BoltCircle is deliberately anchorless. It is not a callout but a
    construction circle through a whole hole pattern, and a detail view that
    "carried" it drew that circle at the detail's own scale about the
    detail's own origin: on 0000_00000413 a 91 mm centre-line circle 34 mm
    off the edge of the paper.
    """
    if type(d).__name__ == "BoltCircle":
        return None
    for name in ("tip", "anchor", "center", "vertex"):
        p = getattr(d, name, None)
        if p and len(p) == 2:
            return (float(p[0]), float(p[1]))
    return None


def _ink_inside(proj, window) -> float:
    """Total drawn length inside the window -- how much there is to magnify."""
    cx, cy, r = window
    total = 0.0
    for seg in proj.segs + proj.circles:
        if seg.kind != "visible":
            continue
        if seg.is_circle and seg.full and seg.center and seg.radius:
            d = math.hypot(seg.center[0] - cx, seg.center[1] - cy)
            if d + seg.radius <= r * 1.05:
                total += 2 * math.pi * seg.radius
                continue
        for run in _clip_polyline_to_circle(seg.pts, (cx, cy), r):
            for a, b in zip(run, run[1:]):
                total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total


def _small_feature_windows(ann, scale: float, max_n: int = 2,
                           small_mm: float = 6.0, max_paper_r: float = 26.0):
    """Regions worth a magnified view, each with the dimensions it should carry.

    Rewritten after visual inspection. The previous selector scored ink
    density at polyline VERTICES, so the winner was always a silhouette
    CORNER -- three coincident vertices beat any real feature. On
    0000_00000061 that put the window exactly on the outside corner of a flat
    plate: ``centre_inset=(0.00, -0.00)``, 2 polylines and 0.12 mm of ink
    inside a 4:1 bubble. The detail view was an empty circle.

    A detail view answers one question: *this feature is too small to letter
    where it is*. So the selector now starts from features that are small ON
    PAPER (:func:`_small_features`), clusters them, and keeps a window only
    when there is real geometry inside it. The dimensions whose anchors fall
    in the window come back with it: the caller draws them in the magnified
    view instead of on the parent, which is what a draughtsman does and what
    makes the detail worth the paper.

    Returns a list of ``(cx, cy, r, dims)``, at most ``max_n``, largest
    cluster first. Empty when nothing on this view is cramped -- a plain
    plate still gets no detail view.
    """
    proj = ann.view
    span = max(proj.width, proj.height)
    if span <= 0 or scale <= 0:
        return []
    feats = _small_features(proj, scale, small_mm)
    if not feats:
        return []

    # Single-linkage clustering. Features within a couple of threshold widths
    # of each other share one bubble: two bores 3 mm apart on paper cannot be
    # given separate detail views without the same crowding all over again.
    link = (small_mm * 1.8) / scale
    clusters: List[List[Tuple[float, float, float]]] = []
    for f in feats:
        joined = None
        for cl in clusters:
            if any(math.hypot(f[0] - g[0], f[1] - g[1]) <= link for g in cl):
                if joined is None:
                    cl.append(f)
                    joined = cl
                else:                       # bridges two clusters -- merge
                    joined.extend(cl)
                    cl.clear()
        if joined is None:
            clusters.append([f])
    clusters = [cl for cl in clusters if cl]

    dims = list(getattr(ann, "dims", []) or [])
    out = []
    for cl in clusters:
        xs = [f[0] for f in cl]; ys = [f[1] for f in cl]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        reach = max(math.hypot(f[0] - cx, f[1] - cy) + f[2] for f in cl)
        r = max(reach * 1.45, (small_mm * 0.9) / scale)
        # A detail bubble that swallows a third of the parent view is not a
        # detail, it is a second copy of the drawing: on machined_block the
        # 0.32 cap put a ring right across the central bore.
        r = min(r, span * 0.22, max_paper_r / scale)
        if r <= 0:
            continue
        # Keep the bubble on the part. A window centred on a silhouette corner
        # is half empty by construction, and that emptiness is exactly the
        # defect this rewrite is for.
        cx = min(max(cx, proj.xmin + r * 0.55), proj.xmax - r * 0.55) \
            if proj.xmax - proj.xmin > r * 1.1 else cx
        cy = min(max(cy, proj.ymin + r * 0.55), proj.ymax - r * 0.55) \
            if proj.ymax - proj.ymin > r * 1.1 else cy
        win = (cx, cy, r)
        # There must be something to look at: at least the perimeter of the
        # smallest feature we are magnifying. 0.12 mm of ink in a 4:1 bubble
        # was the old failure.
        if _ink_inside(proj, win) < max(r * 0.8, 1e-6):
            continue
        mine = [d for d in dims
                if (lambda p: p is not None and
                    math.hypot(p[0] - cx, p[1] - cy) <= r)(_dim_anchor(d))]
        # A detail view exists to hold a dimension that will not fit where
        # the feature is. With nothing to carry it is just a magnifying glass
        # over some geometry: laser_panel grew two 10:1 bubbles, each a
        # quarter of the sheet, showing the 3 mm edge of the plate in its
        # ELEVATION. If no callout belongs in the window, no window.
        if not mine:
            continue
        # A detail is only useful if it makes something bigger: the parent
        # already shows a feature that fills a third of the view perfectly
        # well.
        if r * scale > max_paper_r:
            continue
        out.append((cx, cy, r, mine, len(cl) + 2 * len(mine)))

    out.sort(key=lambda w: -w[4])
    kept = []
    for cx, cy, r, mine, _score in out:
        if any(math.hypot(cx - k[0], cy - k[1]) < (r + k[2]) * 0.9
               for k in kept):
            continue                        # overlapping bubbles read as one
        kept.append((cx, cy, r, mine, False))
        if len(kept) >= max_n:
            break
    return kept


def detail_geometry(proj, window) -> List[List[Tuple[float, float]]]:
    """Visible geometry inside the window, as polylines in view coordinates."""
    cx, cy, r = window
    out = []
    for seg in proj.segs + proj.circles:
        if seg.kind != "visible" or len(seg.pts) < 2:
            continue
        out.extend(_clip_polyline_to_circle(seg.pts, (cx, cy), r))
    return out


def detail_circles(proj, window):
    """Full circles wholly inside the window, as (center, radius).

    Drawn as real circles in the magnified view rather than as their
    flattened polylines: at 4:1 the flattening chords of a small bore are
    visible as a polygon, which no drawing shows.
    """
    cx, cy, r = window
    out = []
    for seg in proj.segs + proj.circles:
        if seg.kind != "visible" or not (seg.is_circle and seg.full):
            continue
        if not seg.center or seg.radius <= 0:
            continue
        if math.hypot(seg.center[0] - cx, seg.center[1] - cy) + seg.radius \
                <= r * 1.02:
            out.append((seg.center, seg.radius))
    return out


def _has_feature_content(proj, window) -> bool:
    """True when the window holds something a reader would call a feature.

    Ink alone is not enough. On 0000_00000386 -- a plain washer -- the
    callouts all point at the same stretch of the outside diameter, so the
    crowd test found a window on the rim: 40 mm of arc inside it and nothing
    to look at. Magnifying a piece of a big circle tells a reader nothing,
    which is the "empty detail view" defect all over again.

    A feature is either the CENTRE of a circle or arc (a bore, a fillet, a
    boss) or a corner -- a vertex where the outline turns by more than 25
    degrees. A smooth flattened arc turns by a degree or two per segment and
    so contributes none.
    """
    cx, cy, r = window
    corners = 0
    for seg in proj.segs + proj.circles:
        if seg.kind != "visible":
            continue
        if seg.center and seg.radius > 0:
            if math.hypot(seg.center[0] - cx, seg.center[1] - cy) <= r:
                return True
        pts = seg.pts
        for a, b, c in zip(pts, pts[1:], pts[2:]):
            if math.hypot(b[0] - cx, b[1] - cy) > r:
                continue
            a1 = math.atan2(b[1] - a[1], b[0] - a[0])
            a2 = math.atan2(c[1] - b[1], c[0] - b[0])
            turn = abs((math.degrees(a2 - a1) + 180) % 360 - 180)
            if turn > 25.0:
                corners += 1
                if corners >= 2:
                    return True
    return False


def crowded_windows(ann, scale: float, max_n: int = 1,
                    paper_r: float = 19.0, min_items: int = 4,
                    max_paper_r: float = 15.0):
    """Regions where the ANNOTATION is too dense to read at sheet scale.

    The other selector asks "is this feature too small to letter?". This one
    asks the question a draughtsman actually asks first: *are there too many
    things to say about one small piece of the part?* On 0000_00000413 the
    centre of VIEW A carries a bore, three small holes, two datum flags, two
    feature control frames, a roughness tick and three angular dimensions
    inside a 20 mm circle -- every one of them correctly placed, and the
    result unreadable. Nothing there is a *small feature* by itself; the
    crowd is the defect.

    So: count annotation ANCHORS (the point on the geometry each one refers
    to, not the label, which the placer has already pushed away) inside a
    circle of ``paper_r`` millimetres of paper. ``min_items`` or more of them
    is a crowd, and the crowd goes into a detail view together -- which is
    also why the detail then shows the neighbouring holes rather than one
    hole in an empty bubble.

    Returns ``[(cx, cy, r, dims)]``, densest first.
    """
    if scale <= 0:
        return []
    items = []
    for d in getattr(ann, "dims", []) or []:
        p = _dim_anchor(d)
        if p is not None:
            items.append((p, d))
    if len(items) < min_items:
        return []
    r = paper_r / scale
    out = []
    used = set()
    for _ in range(max_n):
        best, best_n = None, 0
        for (cx, cy), _d in items:
            members = [i for i, ((x, y), dd) in enumerate(items)
                       if i not in used
                       and math.hypot(x - cx, y - cy) <= r]
            if len(members) > best_n:
                best, best_n = members, len(members)
        if best is None or best_n < min_items:
            break
        # Grow the cluster: re-centre on its members, then take in every
        # anchor the new circle reaches, twice. Without this the window was
        # whatever the first seed circle happened to catch -- on
        # 0000_00000413 it took the bore and two of the three small holes and
        # left C3 and both 106.8 deg angles behind on the parent, so the
        # crowd was split between two views instead of moved out of one.
        members = list(best)
        rr = 0.0
        for _round in range(3):
            pts = [items[i][0] for i in members]
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            reach = max(math.hypot(p[0] - cx, p[1] - cy) for p in pts)
            # room for the features themselves, not just their anchor points
            rr = min(max(reach * 1.35, (paper_r * 0.55) / scale),
                     max_paper_r / scale)
            grown = [i for i, ((x, y), _dd) in enumerate(items)
                     if i not in used and math.hypot(x - cx, y - cy) <= rr]
            if len(grown) <= len(members):
                break
            members = grown
        best = members
        # Take in the neighbouring small features. The crowd is defined by
        # where the ANNOTATION is, but a reader looking at the close-up wants
        # the holes it is talking about, whole: on 0000_00000413 the window
        # stopped 0.7 mm short of C3, so the detail showed C1 and C2 and cut
        # the third hole of the same pattern off its edge.
        near = _small_features(ann.view, scale, 8.0)
        for fx, fy, fr in near:
            d = math.hypot(fx - cx, fy - cy) + fr
            if rr < d <= rr * 1.25:
                rr = min(d * 1.06, max_paper_r / scale)
        if _ink_inside(ann.view, (cx, cy, rr)) < rr * 0.8:
            break
        if not _has_feature_content(ann.view, (cx, cy, rr)):
            break
        # A bubble that swallows the view is not a detail of it, it is a
        # second copy at another scale. On machined_block the crowd test
        # wanted a 60 mm circle over a 90 x 60 view -- which then had nowhere
        # to go, and the callouts it had taken off the parent came back on
        # top of each other.
        short = min(ann.view.width, ann.view.height)
        if short > 0 and 2.0 * rr > short * 0.6:
            break
        out.append((cx, cy, rr, [items[i][1] for i in best], True))
        used.update(best)
    return out


def choose_windows(ann, scale: float, max_n: int = 2,
                   small_mm: float = 6.0, max_paper_r: float = 26.0):
    """Every region of this view worth magnifying, densest crowd first.

    Two reasons to draw a detail view, and a real drawing has both:

    * the annotation is crowded (:func:`crowded_windows`) -- the general case
      the user asked for, "distribute complex geometries across views";
    * a single feature is too small to letter in place
      (:func:`_small_feature_windows`).

    A small-feature window that sits inside a crowd is dropped: the crowd's
    detail already magnifies it, and two bubbles over the same geometry read
    as two different places on the part.
    """
    # A single small feature with its own callout is the classic detail view
    # and the one that reads best, so it is offered first; the crowd window
    # is the fallback for a region where nothing is individually small but
    # there is too much to say about it. Taking them the other way round on
    # machined_block replaced two clean 2:1 counterbore details with one
    # crowd bubble that then had nowhere to go, and the C'BORE note it had
    # taken off the parent view came back on top of the roughness callout.
    # Both kinds are collected and then ranked by WHAT THEY CARRY. Ordering
    # by kind cannot be right for both parts that showed up: putting the
    # crowd first replaced two clean counterbore details on machined_block
    # with one bubble that had nowhere to go, and putting the small feature
    # first gave 0000_00000413 a 5:1 close-up of a single bore while the
    # crowded hole pattern it sits in -- the reason a detail was wanted --
    # stayed on the parent view. The detail worth drawing is the one that
    # takes the most annotation off the parent; a tie goes to the tighter
    # bubble.
    found = list(_small_feature_windows(ann, scale, max_n=max_n,
                                        small_mm=small_mm,
                                        max_paper_r=max_paper_r))
    found += list(crowded_windows(ann, scale))
    found.sort(key=lambda w: (-len(w[3]), w[2]))
    out = []
    for w in found:
        if any(math.hypot(w[0] - c[0], w[1] - c[1]) < (w[2] + c[2]) * 0.9
               for c in out):
            continue
        out.append(w)
    return out[:max_n + 1]
