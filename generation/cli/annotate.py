"""Automatic dimensioning and callout placement.

Dimensions are generated in *view coordinates* (model units) and placed by a
simple occupancy-grid solver that keeps annotation off the geometry and off
other annotation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .geometry import Hole, PartInfo, fmt, fmt_raw
from .projection import Projection, world_to_view


# --------------------------------------------------------------------------- #
# annotation records
# --------------------------------------------------------------------------- #
def _sized(style, value: float, key: str) -> str:
    """A length as this office writes it: nominal, toleranced, or as limits.

    Kept in one helper so the string a dimension reserves space for is the
    same string that is drawn and recorded -- a tolerance added at draw time
    would make every width estimate (and every exported box) too narrow.
    """
    from .styles import limit_text, tolerance_text
    house = getattr(style, "house_style", "")
    limits = limit_text(style, value, f"{house}:{key}", fmt)
    if limits:
        return limits
    return fmt(value) + tolerance_text(style, value, f"{house}:{key}")


@dataclass
class LinearDim:
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    offset: float                 # signed distance from the measured line
    #: 'h', 'v', or 'a' for a dimension aligned with an oblique edge: the
    #: dimension line runs parallel to the edge and states its true
    #: length, which is the only honest way to size a sloping face (a
    #: horizontal dimension across it states a projected length the part
    #: does not have).
    direction: str
    text: str
    kind: str = "linear"
    # 0 = text centred on the dim line, +1 = shifted past the far arrow,
    # -1 = shifted past the near arrow (used to stagger tight neighbours)
    shift: int = 0
    # Where the measured geometry actually lies on the cross axis. A witness
    # line must run from the feature to the dimension line; anchoring it at the
    # view-box corner instead leaves it visibly stopping short of the edge it
    # refers to ("lines ending nowhere").
    anchor1: Optional[float] = None
    anchor2: Optional[float] = None


@dataclass
class RadiusDim:
    center: Tuple[float, float]
    radius: float
    text: str
    angle: float = 45.0           # leader direction, degrees
    leader_len: float = 0.0
    # A diameter leader stores the physical radius used for its geometry.
    # ``cls`` supplies the semantic class: hole diameters are bores, while
    # an overall diameter is a size dimension.
    kind: str = "bore"
    cls: str = "bore"
    box: Optional[Tuple[float, float, float, float]] = None   # text+leader extent
    # the two segments actually drawn for this leader
    tip: Optional[Tuple[float, float]] = None   # arrowhead point, on geometry
    snap_gap: float = 0.0        # distance the tip moved to reach geometry
    seg_start: Optional[Tuple[float, float]] = None
    seg_knee: Optional[Tuple[float, float]] = None
    seg_shelf: Optional[Tuple[float, float]] = None


# Kept as a source-compatible alias for callers that imported the old class
# name. New code should use RadiusDim so the geometry and dataset taxonomy
# agree about what a ``⌀`` annotation is.
DiaDim = RadiusDim


@dataclass
class Callout:
    anchor: Tuple[float, float]   # point on the geometry
    text: str
    tail: Tuple[float, float] = (0, 0)   # text origin
    kind: str = "callout"
    # Detection class (schema.AnnotationKind). One leader-note primitive
    # carries radii, chamfers, threads, roughness and plain notes, so the
    # class is recorded when the note is created rather than guessed later
    # by pattern-matching the rendered text.
    cls: str = "dimension"
    #: True when the underlying specification was generated rather than read
    #: from the file or measured. Reaches attributes.detail.synthetic.
    synthetic: bool = False


@dataclass
class AngleDim:
    """An angular dimension between two straight edges of a view.

    Only emitted for genuinely oblique corners: a 90 deg corner is implied by
    the orthographic projection and dimensioning it is noise.
    """
    vertex: Tuple[float, float]      # corner where the two edges meet
    start_ang: float                 # bearing of the first leg, degrees
    end_ang: float                   # bearing of the second leg, degrees
    radius: float                    # arc radius, view units
    text: str                        # e.g. "40.6 deg"
    kind: str = "angle"


@dataclass
class FeatureFrame:
    """A GD&T feature control frame with a leader, e.g. |pos|o0.14|A|B|C|.

    The frame is a row of compartments: the characteristic symbol, the
    tolerance value, then one compartment per datum reference. It is drawn as
    real boxed geometry (see :mod:`autodraft.gdt_symbols` for why the symbol
    is vector rather than text) so it matches what a reader sees on a print.
    """

    anchor: Tuple[float, float]        # point on the geometry, view coords
    characteristic: str                # 'position', 'flatness', ...
    value_text: str                    # 'o0.14' -- already formatted
    datums: Tuple[str, ...] = ()
    modifier: Optional[str] = None     # 'M' / 'L' / 'S'
    tail: Tuple[float, float] = (0, 0)  # frame's lower-left corner
    text: str = ""                     # machine-readable equivalent
    value: Optional[float] = None      # the numeric tolerance, mm
    synthetic: bool = False            # generated, not read from the file
    kind: str = "frame"
    cls: str = "gdt"


@dataclass
class DatumFlag:
    """A datum feature symbol: a boxed letter on a short leader with a triangle."""

    anchor: Tuple[float, float]
    letter: str
    tail: Tuple[float, float] = (0, 0)
    synthetic: bool = False
    kind: str = "datum"
    cls: str = "gdt"


@dataclass
class RoughnessMark:
    """A surface-texture symbol -- the ISO 1302 tick with an Ra value.

    ``general`` marks the drawing-wide note, which is drawn near the title
    block rather than on a view and needs no leader.
    """

    anchor: Optional[Tuple[float, float]]
    ra_text: str                        # 'Ra 1.6'
    machined: bool = True               # barred symbol: material removal
    general: bool = False
    tail: Tuple[float, float] = (0, 0)
    kind: str = "roughness"
    cls: str = "roughness"


@dataclass
class BoltCircle:
    center: Tuple[float, float]
    radius: float
    kind: str = "boltcircle"


@dataclass
class CenterMark:
    center: Tuple[float, float]
    radius: float


@dataclass
class Annotated:
    view: Projection
    dims: List[object] = field(default_factory=list)
    marks: List[CenterMark] = field(default_factory=list)
    pad: Tuple[float, float, float, float] = (0, 0, 0, 0)  # l, r, b, t extra space
    # union of every annotation bounding box, in view coordinates. The sheet
    # uses this instead of a heuristic pad so nothing is ever clipped.
    ext: Optional[Tuple[float, float, float, float]] = None   # x0, y0, x1, y1
    # rendered text boxes of the linear dimensions, in view coordinates
    dim_boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    # per-feature bounding boxes: (x0, y0, x1, y1, label, qty) in view coords
    feature_boxes: List[Tuple[float, float, float, float, str, int]] = field(
        default_factory=list)
    # Thread designations already stated on this sheet. Threads may be drawn
    # on any view where their bore reads as a circle, so this stops the same
    # spec being called out twice. Shared across views by build_sheet.
    thread_ids: set = field(default_factory=set)
    # Same idea for GD&T frames: a frame may be drawn on any view with room,
    # so this stops one being repeated across views. Shared per sheet.
    frame_ids: set = field(default_factory=set)
    # every leader / dimension / witness segment already committed, so the
    # placement solver can reject a candidate that would cross one
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(
        default_factory=list)

    def add_seg(self, a, b):
        self.segments.append((a, b))

    def grow(self, x0, y0, x1, y1):
        if self.ext is None:
            self.ext = (x0, y0, x1, y1)
        else:
            a, b, c, d = self.ext
            self.ext = (min(a, x0), min(b, y0), max(c, x1), max(d, y1))


# --------------------------------------------------------------------------- #
# occupancy grid
# --------------------------------------------------------------------------- #
class Occupancy:
    """Coarse raster used to test whether a label/leader area is free."""

    def __init__(self, xmin, ymin, xmax, ymax, cell):
        m = max(xmax - xmin, ymax - ymin) * 0.6 + 1e-9
        self.x0, self.y0 = xmin - m, ymin - m
        self.x1, self.y1 = xmax + m, ymax + m
        self.cell = cell
        self.nx = max(1, int((self.x1 - self.x0) / cell) + 1)
        self.ny = max(1, int((self.y1 - self.y0) / cell) + 1)
        self.g = bytearray(self.nx * self.ny)

    def _idx(self, x, y):
        i = int((x - self.x0) / self.cell)
        j = int((y - self.y0) / self.cell)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return j * self.nx + i
        return None

    def mark_poly(self, pts: Sequence[Tuple[float, float]]):
        for a, b in zip(pts, pts[1:]):
            self.mark_line(a, b)

    def _samples(self, a, b):
        """Number of samples for a segment, capped to the grid's resolution.

        A segment longer than the grid cannot touch more cells than the grid
        has along its diagonal, so sampling further is wasted work -- and for
        a wildly out-of-range endpoint it is catastrophic: a (0,0)-(1e9,1e9)
        segment asked for 5.7 BILLION iterations and hung the process. Nothing
        in the drawing path passes such a segment today, but the placement
        solvers take arbitrary anchor points, so the loop is bounded here
        rather than trusting every caller.
        """
        n = int(math.dist(a, b) / (self.cell * 0.5)) + 1
        cap = 2 * (self.nx + self.ny) + 4
        if n > cap:
            n = cap
        return n if n >= 2 else 2

    def mark_line(self, a, b):
        n = self._samples(a, b)
        for k in range(n + 1):
            t = k / n
            k_ = self._idx(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if k_ is not None:
                self.g[k_] = 1

    def clear_rect(self, x0, y0, x1, y1):
        """Free a region (used when an annotation is re-placed)."""
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        if x1 < self.x0 or x0 > self.x1 or y1 < self.y0 or y0 > self.y1:
            return
        i0, j0, i1, j1 = self._bounds(x0, y0, x1, y1)
        if i1 < i0 or j1 < j0:
            return
        span = bytes(i1 - i0 + 1)
        nx = self.nx
        for j in range(j0, j1 + 1):
            base = j * nx
            self.g[base + i0: base + i1 + 1] = span

    def mark_rect(self, x0, y0, x1, y1):
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        if x1 < self.x0 or x0 > self.x1 or y1 < self.y0 or y0 > self.y1:
            return
        i0, j0, i1, j1 = self._bounds(x0, y0, x1, y1)
        if i1 < i0 or j1 < j0:
            return
        span = b"\x01" * (i1 - i0 + 1)
        nx = self.nx
        for j in range(j0, j1 + 1):
            base = j * nx
            self.g[base + i0: base + i1 + 1] = span

    def _bounds(self, x0, y0, x1, y1):
        """Clamped integer cell range covering a rectangle."""
        i0 = int((x0 - self.x0) / self.cell)
        i1 = int((x1 - self.x0) / self.cell)
        j0 = int((y0 - self.y0) / self.cell)
        j1 = int((y1 - self.y0) / self.cell)
        if i0 > i1:
            i0, i1 = i1, i0
        if j0 > j1:
            j0, j1 = j1, j0
        i0 = 0 if i0 < 0 else i0
        j0 = 0 if j0 < 0 else j0
        i1 = self.nx - 1 if i1 >= self.nx else i1
        j1 = self.ny - 1 if j1 >= self.ny else j1
        return i0, j0, i1, j1

    def rect_cost(self, x0, y0, x1, y1) -> int:
        """Occupied-cell count inside a rectangle.

        Scans whole rows as bytearray slices instead of sampling point by
        point: this is the hottest call in placement (millions of samples per
        drawing) and slicing moves the inner loop into C.
        """
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        if x1 < self.x0 or x0 > self.x1 or y1 < self.y0 or y0 > self.y1:
            return 0
        i0, j0, i1, j1 = self._bounds(x0, y0, x1, y1)
        if i1 < i0 or j1 < j0:
            return 0
        g = self.g
        nx = self.nx
        total = 0
        for j in range(j0, j1 + 1):
            base = j * nx
            row = g[base + i0: base + i1 + 1]
            if row:
                total += len(row) - row.count(0)
        return total

    def line_cost(self, a, b) -> int:
        """Occupied cells along a segment.

        Inlines the index arithmetic and hoists loop invariants: this runs tens
        of thousands of times per drawing inside the placement search, so the
        per-sample attribute lookups and the ``_idx`` call were measurable.
        """
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        cell = self.cell
        n = self._samples(a, b)
        gx0, gy0 = self.x0, self.y0
        nx, ny = self.nx, self.ny
        g = self.g
        c = 0
        inv = 1.0 / n
        # Step along the segment, but only touch the grid when the sample
        # moves to a new cell: consecutive samples land in the same cell about
        # 44% of the time (they are spaced half a cell apart), and the repeat
        # lookups cost more than remembering the last index. The RETURNED
        # VALUE is unchanged -- a cell that several samples fall into is still
        # counted once per sample, via `run`.
        step_x = dx * inv / cell
        step_y = dy * inv / cell
        fx = (ax - gx0) / cell
        fy = (ay - gy0) / cell
        last_i = last_j = -(1 << 30)
        hit = False
        for _k in range(n + 1):
            i = int(fx)
            j = int(fy)
            fx += step_x
            fy += step_y
            if i != last_i or j != last_j:
                last_i, last_j = i, j
                hit = (0 <= i < nx and 0 <= j < ny and g[j * nx + i] != 0)
            if hit:
                c += 1
        return c


def holes_in_view(info: PartInfo, view: str,
                  proj: "Projection" = None
                  ) -> List[Tuple[Hole, Tuple[float, float], float]]:
    """Holes that read as circles in this view.

    A hole qualifies when its axis is parallel to the viewing direction and
    its bore is a closed cylinder, so that it actually draws as a circle.

    Open (partial-span) cylinders are excluded. These are scallops, slot ends
    and radiused notches: a 65 deg arc has a nominal "centre" that sits in
    fresh air well outside the part, so it can never carry a diameter callout.
    They used to be filtered indirectly by testing the centre against the
    silhouette, which was only ever a proxy -- it dropped the far-outside ones
    and silently kept any whose phantom centre happened to land back on the
    part. Testing the span is the direct question and does not depend on where
    the arc's centre happens to fall.
    """
    from .projection import _axes
    dv, xv, yv = _axes(view)
    res = []
    for h in info.holes:
        if not getattr(h, "is_closed", True):
            continue
        a = (h.axis[0], h.axis[1], h.axis[2])
        dot = abs(a[0] * dv.x + a[1] * dv.y + a[2] * dv.z)
        if dot <= 0.999:
            continue
        p = world_to_view(view, h.pos)
        if proj is not None:
            # The centre itself must lie inside the silhouette. Allowing a
            # whole radius of slack let holes belonging to clipped features
            # through, and their callouts pointed at empty space.
            eps = max(proj.width, proj.height) * 1e-3
            if not (proj.xmin - eps <= p[0] <= proj.xmax + eps
                    and proj.ymin - eps <= p[1] <= proj.ymax + eps):
                continue
        res.append((h, p, h.diameter / 2))
    return res



def detect_bolt_circle(members, tol_frac=0.02):
    """If >=3 identical holes lie on a common circle, return (cx, cy, dia)."""
    if len(members) < 3:
        return None
    xs = [m[1][0] for m in members]
    ys = [m[1][1] for m in members]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    rs = [math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)]
    rm = sum(rs) / len(rs)
    if rm < 1e-9:
        return None
    if max(abs(r - rm) for r in rs) > max(rm * tol_frac, 1e-4):
        return None
    # require a roughly even angular spread
    angs = sorted(math.degrees(math.atan2(y - cy, x - cx)) % 360 for x, y in zip(xs, ys))
    gaps = [(angs[(i + 1) % len(angs)] - angs[i]) % 360 for i in range(len(angs))]
    even = max(gaps) - min(gaps) < 5.0
    return (cx, cy, 2 * rm, even)



def detect_polar_pattern(holes, proj, tol_frac=0.02, min_n=3):
    """Holes lying on a common circle, regardless of diameter.

    ``detect_bolt_circle`` only ever sees one diameter group and infers the
    centre by averaging the hole positions -- which is only correct when the
    holes happen to be symmetric about it. A three-lobe part with two hole
    sizes on one pitch circle therefore went undetected, and its angular
    spacing (the one dimension that actually locates the lobes) was never
    stated.

    Returns ``(cx, cy, pcd, step_deg, members)`` or None. ``step_deg`` is the
    angular pitch when the holes are evenly spaced, else None.
    """
    if len(holes) < min_n:
        return None
    # Candidate centres: the view centre, and any hole that could be the hub.
    cands = [((proj.xmin + proj.xmax) / 2.0, (proj.ymin + proj.ymax) / 2.0)]
    cands += [m[1] for m in holes]

    best = None
    for cx, cy in cands:
        ring = []
        for m in holes:
            r = math.hypot(m[1][0] - cx, m[1][1] - cy)
            if r > 1e-9:
                ring.append((r, m))
        if len(ring) < min_n:
            continue
        # group by radius: the pitch circle is the most populated one
        ring.sort(key=lambda t: t[0])
        i = 0
        while i < len(ring):
            r0 = ring[i][0]
            grp = [t for t in ring if abs(t[0] - r0) <= max(r0 * tol_frac, 1e-4)]
            if len(grp) >= min_n:
                rm = sum(t[0] for t in grp) / len(grp)
                if best is None or len(grp) > len(best[4]):
                    angs = sorted(math.degrees(
                        math.atan2(m[1][1] - cy, m[1][0] - cx)) % 360.0
                        for _r, m in grp)
                    gaps = [(angs[(k + 1) % len(angs)] - angs[k]) % 360.0
                            for k in range(len(angs))]
                    # Even spacing, allowing vacant stations: a 6-station
                    # 60 deg pattern with one hole missing still shows a
                    # 60 deg pitch (one gap is simply 120 deg).
                    step = None
                    if gaps:
                        if max(gaps) - min(gaps) < 2.0:
                            step = sum(gaps) / len(gaps)
                        else:
                            base = min(gaps)
                            if base > 1.0 and all(
                                    abs(g / base - round(g / base)) < 0.03
                                    for g in gaps):
                                step = base

                    # A rectangular grid trivially puts its corners on a
                    # circle, but it is located by X/Y, not by an angular
                    # pitch -- calling it polar would state a meaningless
                    # angle. Demand true rotational symmetry: the pitch must
                    # divide 360 a whole number of times, and every hole must
                    # sit on an exact multiple of it.
                    if step is not None and step > 0:
                        n_station = 360.0 / step
                        if abs(n_station - round(n_station)) > 0.02:
                            step = None
                    if step is not None:
                        a0 = angs[0]
                        for a in angs:
                            k = ((a - a0) % 360.0) / step
                            if abs(k - round(k)) > 0.02:
                                step = None
                                break
                    # four holes at 90 deg is a rectangle by another name
                    if step is not None and len(grp) == 4 and \
                            abs(step - 90.0) < 1.0:
                        step = None
                    best = (cx, cy, 2 * rm, step, [m for _r, m in grp])
            i += max(1, len(grp))
    return best


def _cluster_by_diameter(holes, tol=1e-4):
    groups = {}
    for h, p, r in holes:
        key = (round(h.diameter, 4), h.through, h.counterbore, h.countersink)
        groups.setdefault(key, []).append((h, p, r))
    return groups


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
class _Annotator:
    """Builds the annotation for one view, one concern at a time.

    This was a single 547-line function containing ten numbered comment
    blocks. Those blocks are now methods, applied in order by :meth:`run`.
    They share state through attributes instead of 28 function-level locals,
    so what each step reads and writes is visible in one place rather than
    having to be traced through the whole body.

    The order in :meth:`run` is load-bearing; it is documented there.
    """

    def __init__(self, proj, info, *, style, primary, hole_table, tagged,
                 scale):
        self.proj, self.info, self.style = proj, info, style
        self.name = getattr(proj, "name", "view")
        self.primary, self.hole_table, self.tagged = primary, hole_table, tagged
        self.ann = Annotated(view=proj)
        self.w, self.h = proj.width, proj.height

        # Paper mm -> model units. Text heights are given on paper, but every
        # placement decision here happens in model coordinates, so reserved
        # space must be divided by the scale to match what is actually drawn.
        sc = scale if (scale and scale > 0) else 1.0
        self.txt = style.text_height / sc
        self.gap = self._dimension_band_gap()

        self.occ = Occupancy(proj.xmin, proj.ymin, proj.xmax, proj.ymax,
                             cell=max(self.w, self.h) / 150)
        for s in proj.segs + proj.circles:
            self.occ.mark_poly(s.pts)

        # Shared state written by the steps below. Declared here so it is
        # discoverable without having to read every method.
        self.below = self.left = 0.0    # where the overall dimension lines sit
        self.lvl_b = self.lvl_l = 1     # dimension levels used, bottom / left
        self.tol = max(self.w, self.h) * 5e-3
        self.circles = []               # holes that read as circles in this view
        self.groups = {}                # those, clustered by identical geometry
        self.bolt_circles = {}
        self.linear_holes = []          # holes not already located by a B.C.
        self.placed_callouts = []
        self.od_radius = None           # set when the whole view is one circle
        self.od_centre = None           # its note is deferred; see run()

    # ------------------------------------------------------------------ #
    def _dimension_band_gap(self):
        """Spacing between stacked dimension lines, in model units.

        Clamped so a part with many distinct hole coordinates cannot end up
        with a dimension band taller than the view it describes.
        """
        style, span = self.style, max(self.w, self.h)
        gap = style.dim_gap * span if style.dim_gap < 1 else style.dim_gap
        gap = max(gap, self.txt * 2.0)
        hv = holes_in_view(self.info, self.proj.name, self.proj)
        levels = 1 + min(style.max_position_dims,
                         max(len({round(p[0], 3) for _, p, _ in hv}),
                             len({round(p[1], 3) for _, p, _ in hv})))
        gap = min(gap, span * 0.55 / max(2, levels))
        return max(gap, self.txt * 1.6)

    def run(self):
        """Apply every step in order, then size the annotation band.

        The order is not arbitrary:

        * sizes come before features, so the dimension band exists and later
          leaders can be routed around it;
        * the round-view diameter is deferred until after the position
          dimensions and hole callouts, because placing it first put the note
          exactly where an extension line was later drawn through it;
        * notes come last and are then swept, because a note can only avoid
          what has already been committed.
        """
        self._overall_dims()
        self._body_dims()
        self._position_dims()
        self._polar_pattern_dims()
        self._aligned_edge_dims()
        self._corner_angle_dims()
        self._hole_callouts()
        self._outer_diameter_dim()
        self._thickness_note()
        self._fillet_notes()
        self._chamfer_notes()
        self._thread_notes()
        self._gdt_frames()
        self._roughness_marks()
        self._repair_notes()
        self._mark_reference_dims()
        self._finish()
        return self.ann

    # -- steps, in run() order ------------------------------------------ #
    def _overall_dims(self):
        """Overall width and height -- or one diameter when the view is round."""
        self.below = self.proj.ymin - self.gap
        self.left = self.proj.xmin - self.gap
        self.tol = max(self.w, self.h) * 5e-3

        # A round view gets ONE diameter, not a width and a height. The diameter
        # is drawn across the circle rather than as an overall box dimension.
        self.od_radius = outer_diameter(self.proj) if getattr(self.style, "diameter_dims", True) else None
        self.od_centre = None
        if self.od_radius is not None:
            # The overall diameter is emitted LAST (see below) so that the
            # collision-aware placer can see the position dimensions and their
            # extension lines. Placing it here put the note in the best spot for
            # an empty sheet, which a later extension line then ran through.
            self.od_centre = ((self.proj.xmin + self.proj.xmax) / 2, (self.proj.ymin + self.proj.ymax) / 2)

        if self.od_radius is None:
            oh = LinearDim((self.proj.xmin, self.proj.ymin), (self.proj.xmax, self.proj.ymin),
                            offset=-self.gap, direction="h",
                            text=_sized(self.style, self.w, f"{self.name}:ow"))
            oh.anchor1 = _anchor_at(self.proj, self.proj.xmin, "h", self.proj.ymin, self.tol)
            oh.anchor2 = _anchor_at(self.proj, self.proj.xmax, "h", self.proj.ymin, self.tol)
            self.ann.dims.append(oh)
            self._overall_h = oh
            ov = LinearDim((self.proj.xmin, self.proj.ymin), (self.proj.xmin, self.proj.ymax),
                            offset=-self.gap, direction="v",
                            text=_sized(self.style, self.h, f"{self.name}:oh"))
            ov.anchor1 = _anchor_at(self.proj, self.proj.ymin, "v", self.proj.xmin, self.tol)
            ov.anchor2 = _anchor_at(self.proj, self.proj.ymax, "v", self.proj.xmin, self.tol)
            self.ann.dims.append(ov)
            self._overall_v = ov
        # Reserve the dimension line AND its centred text. render._linear puts
        # horizontal text at y = line + 0.5*txt (height txt) and vertical text at
        # x = line - 1.5*txt .. line - 0.5*txt, so the band must cover that or a
        # leader gets routed through the number.
        self.occ.mark_rect(self.proj.xmin, self.below - self.txt * 0.8, self.proj.xmax, self.below + self.txt * 1.8)
        self.occ.mark_rect(self.left - self.txt * 1.8, self.proj.ymin, self.left + self.txt * 0.8, self.proj.ymax)

        # witness lines of the overall dims run from the view corners out to the
        # dimension line and overshoot it slightly; mark them so callouts placed
        # later cannot be routed straight across them
        ow = self.txt * 1.8
        for x in (self.proj.xmin, self.proj.xmax):
            self.occ.mark_line((x, self.proj.ymin + ow), (x, self.below - ow * 0.6))
            self.ann.add_seg((x, self.proj.ymin), (x, self.below))
        for y in (self.proj.ymin, self.proj.ymax):
            self.occ.mark_line((self.proj.xmin + ow, y), (self.left - ow * 0.6, y))
            self.ann.add_seg((self.proj.xmin, y), (self.left, y))
        self.ann.add_seg((self.proj.xmin, self.below), (self.proj.xmax, self.below))
        self.ann.add_seg((self.left, self.proj.ymin), (self.left, self.proj.ymax))
        self.lvl_b, self.lvl_l = 1, 1

    def _mark_reference_dims(self):
        """Mark the overall size as REFERENCE when a chain already gives it.

        A chain of positions plus the overall length over-dimensions the
        part: the overall is arithmetic, not an instruction, and a shop that
        works to both can end up chasing a tolerance stack that no single
        dimension owns. ASME Y14.5 and ISO 129 both have a way of saying so
        -- "(34)" or "34 REF" -- and which one is a house convention
        (:func:`styles.reference_text`).

        Only fires when the chain really does span the same distance, to
        within a millimetre of model: marking an overall that nothing else
        adds up to would delete a size the part needs.
        """
        from .styles import reference_text
        if getattr(self.style, "ref_dim_style", "paren") == "none":
            return
        for attr, axis in (("_overall_h", "h"), ("_overall_v", "v")):
            od = getattr(self, attr, None)
            if od is None:
                continue
            span = abs((od.p2[0] - od.p1[0]) if axis == "h"
                       else (od.p2[1] - od.p1[1]))
            if span <= 0:
                continue
            total, n = 0.0, 0
            for d in self.ann.dims:
                if d is od or not isinstance(d, LinearDim):
                    continue
                if d.direction != axis:
                    continue
                total += abs((d.p2[0] - d.p1[0]) if axis == "h"
                             else (d.p2[1] - d.p1[1]))
                n += 1
            if n >= 2 and abs(total - span) <= max(span * 0.02, 1e-6):
                od.text = reference_text(self.style, od.text)

    def _body_dims(self):
        """Size each solid separately when the file holds several disjoint bodies."""
        # The overall dimensions above span the whole envelope, which on a
        # multi-body file describes none of the actual parts: 0000_00000073 is
        # three separate blocks and the print only said 1.5 x 0.8 x 0.65. Each
        # body gets its own width x height note so every block is manufacturable.
        # A note (rather than more dimension lines) keeps the already-crowded
        # dimension band from stacking three more levels.
        if (self.style.body_dims and self.primary and len(getattr(self.info, "bodies", [])) > 1
                and len(self.info.bodies) <= self.style.body_max_notes):
            for bd in self.info.bodies:
                c0 = world_to_view(self.proj.name, (bd.xmin, bd.ymin, bd.zmin))
                c1 = world_to_view(self.proj.name, (bd.xmax, bd.ymax, bd.zmax))
                bx0, bx1 = min(c0[0], c1[0]), max(c0[0], c1[0])
                by0, by1 = min(c0[1], c1[1]), max(c0[1], c1[1])
                bw, bh = bx1 - bx0, by1 - by0
                if bw <= self.tol or bh <= self.tol:
                    continue          # body is edge-on in this view
                label = f"BODY {bd.index + 1}: {fmt(bw)} X {fmt(bh)}"  # a size
                mid = ((bx0 + bx1) / 2, (by0 + by1) / 2)
                anc = _nearest_geometry(self.proj, mid, prefer_circle_edge=False) or mid
                tail = _place_note(self.occ, self.ann, anc, label, self.proj, self.style, self.txt, self.gap)
                # Free text describing one solid of a multi-body file; not a
                # dimension line, so it is classed as a note.
                self.ann.dims.append(Callout(anchor=anc, text=label, tail=tail,
                                             cls="leader_note"))
                self.ann.feature_boxes.append((bx0, by0, bx1, by1, label, 1))

        self.circles = holes_in_view(self.info, self.proj.name, self.proj)
        self.groups = _cluster_by_diameter(self.circles)

        # holes belonging to an even bolt circle get a BCD note, not X/Y baselines
        self.bolt_circles = {}
        patterned = set()
        for key, members in self.groups.items():
            bc = detect_bolt_circle(members)
            if bc and bc[3]:
                self.bolt_circles[key] = bc
                for m in members:
                    patterned.add((round(m[1][0], 6), round(m[1][1], 6)))
        self.linear_holes = [m for m in self.circles
                        if (round(m[1][0], 6), round(m[1][1], 6)) not in patterned]

    def _position_dims(self):
        """Locate the holes with chain dimensions, not a baseline forest."""
        # A concentric bore in a round primary view needs no X/Y chain: the
        # centre is already defined by the circular envelope, while the bolt
        # circle note locates the patterned holes. Four radial-looking ``6``
        # dimensions around a small disc only repeat that information and turn
        # the view into a nest of witness lines. Retain position dimensions only
        # for genuinely off-centre, unpatterned holes.
        if self.od_radius is not None:
            vcx = (self.proj.xmin + self.proj.xmax) / 2
            vcy = (self.proj.ymin + self.proj.ymax) / 2
            tol = max(self.tol, max(self.w, self.h) * 1e-3)
            if not any(math.hypot(m[1][0] - vcx, m[1][1] - vcy) > tol
                       for m in self.linear_holes):
                return

        # Essential-dimension selection. Dimensioning every distinct coordinate is
        # unreadable on a dense part (26 X values here), but dropping the chain
        # entirely leaves the holes unlocated -- which is worse, because the part
        # is then unmanufacturable from the print. So locate one representative
        # hole per feature group, prioritised by importance, and let the grouped
        # callout ("8X o2.5") carry the repetition.
        if self.linear_holes and not self.hole_table:
            tolp = max(self.w, self.h) * 2e-3
            by_group: Dict[object, list] = {}
            for m in self.linear_holes:
                key = (round(m[0].diameter, 4), m[0].through,
                       m[0].counterbore, m[0].countersink)
                by_group.setdefault(key, []).append(m)

            # importance: bigger holes and bigger patterns get located first
            def _rank(kv):
                key, members = kv
                return (-key[0] * (1.0 + 0.25 * len(members)), -len(members))

            # Collect candidate coordinates, most important groups first.
            #
            # A chain occupies ONE level regardless of how many features it spans,
            # so unlike baseline dimensioning there is no reason to keep just one
            # representative per group: a row of four identical holes needs all
            # four located, and the chain gives that for the same vertical space.
            # Readability is protected by the min-segment merge inside _emit_chain.
            cand_x, cand_y = [], []
            for key, members in sorted(by_group.items(), key=_rank):
                pts = [m[1] for m in members]
                spread_x = max(p[0] for p in pts) - min(p[0] for p in pts)
                spread_y = max(p[1] for p in pts) - min(p[1] for p in pts)
                # a pattern that runs along an axis gets every member on that axis;
                # otherwise just the member nearest the datum
                take_x = pts if spread_x > tolp * 4 else [
                    min(pts, key=lambda q: (q[0] - self.proj.xmin) ** 2 + (q[1] - self.proj.ymin) ** 2)]
                take_y = pts if spread_y > tolp * 4 else [
                    min(pts, key=lambda q: (q[0] - self.proj.xmin) ** 2 + (q[1] - self.proj.ymin) ** 2)]
                for q in take_x:
                    if all(abs(q[0] - v) > tolp for v in cand_x):
                        cand_x.append(q[0])
                for q in take_y:
                    if all(abs(q[1] - v) > tolp for v in cand_y):
                        cand_y.append(q[1])

            # Cap the number of chain segments so each still has room for its own
            # text; the chain is a single level, so this is a readability limit
            # rather than a vertical-space one.
            lim_x = max(1, min(self.style.max_chain_dims, int(self.w / (self.txt * 2.6)) or 1))
            lim_y = max(1, min(self.style.max_chain_dims, int(self.h / (self.txt * 2.6)) or 1))
            xs = sorted(cand_x[:lim_x])
            ys = sorted(cand_y[:lim_y])

            # Reserve the whole corridor each dimension chain will occupy *before*
            # any callout is placed. Callouts are positioned later in this function
            # and can only avoid what is already marked, so marking the band per
            # level as we go (the previous behaviour) left the last levels
            # unprotected and notes landed on the numbers.
            # a chain occupies a single level per axis (plus the overall dim)
            n_x = 1 if any(abs(x - self.proj.xmin) >= tolp for x in xs) else 0
            n_y = 1 if any(abs(y - self.proj.ymin) >= tolp for y in ys) else 0
            # Witness lines overshoot the dimension line by ~1.5 units and the
            # horizontal chain's lines rise above the view top, so the reserved
            # corridors must extend past the view box on both counts.
            over = self.txt * 1.8
            if n_x:
                self.occ.mark_rect(self.proj.xmin - self.txt, self.proj.ymin - self.gap * (self.lvl_b + n_x) - self.txt * 1.4,
                              self.proj.xmax + self.txt, self.proj.ymin + over)
            if n_y:
                self.occ.mark_rect(self.proj.xmin - self.gap * (self.lvl_l + n_y) - self.txt * 1.4, self.proj.ymin - self.txt,
                              self.proj.xmin + over, self.proj.ymax + self.txt)

            # Pre-mark the exact witness lines of both chains. The dims themselves
            # are emitted just below, but callouts are placed *after* that and the
            # coarse band reservation alone let notes land on these thin lines.
            wb, wl = self.lvl_b + n_x, self.lvl_l + n_y
            for _x in xs:
                self.occ.mark_line((_x, self.proj.ymin + self.txt * 1.8),
                              (_x, self.proj.ymin - self.gap * wb - self.txt))
            for _y in ys:
                self.occ.mark_line((self.proj.xmin + self.txt * 1.8, _y),
                              (self.proj.xmin - self.gap * wl - self.txt, _y))

            # ---- emit as a chain, not a baseline forest ----------------------
            # Baseline dimensioning puts every value on its own level measured
            # from one datum, so N features need N stacked levels and each long
            # dimension line runs across the witness lines of the ones below it
            # (this is what put a line through "31" in the TOP view).
            # A chain places consecutive gaps on a single level: 9 and 6 instead
            # of 9 and 15. Far fewer levels, far shorter lines, and it is what a
            # machinist wants for a row of features anyway. The overall size is
            # already dimensioned separately, so the chain stays fully determined.
            # Which edge a chain runs along is a house convention, not a rule:
            # a drawing office dimensions consistently from one datum corner,
            # and different offices pick different corners. Running every
            # chain along the bottom and the left made every sheet look the
            # same and left the top and right edges of the part bare.
            h_cross = (self.proj.ymax
                       if getattr(self.style, "dim_side_h", "bottom") == "top"
                       else self.proj.ymin)
            v_cross = (self.proj.xmax
                       if getattr(self.style, "dim_side_v", "left") == "right"
                       else self.proj.xmin)
            _emit_chain(self.ann, self.occ, self.proj, xs, "h", self.proj.xmin,
                        h_cross, self.gap, self.txt, self.lvl_b, self.style,
                        tolp)
            _emit_chain(self.ann, self.occ, self.proj, ys, "v", self.proj.ymin,
                        v_cross, self.gap, self.txt, self.lvl_l, self.style,
                        tolp)
            self.lvl_b += 1 if len([x for x in xs if abs(x - self.proj.xmin) >= tolp]) else 0
            self.lvl_l += 1 if len([y for y in ys if abs(y - self.proj.ymin) >= tolp]) else 0

    def _polar_pattern_dims(self):
        """Pitch-circle diameter and angular pitch of a ring of holes."""
        # A ring of holes is located by its pitch-circle diameter *and* its angular
        # pitch. Without the angle the lobes cannot be placed, and on an all-arc
        # outline (no straight edges) there is nothing else to dimension it from.
        if self.style.angle_dims and self.circles:
            pol = detect_polar_pattern(self.circles, self.proj)
            if pol is not None:
                pcx, pcy, pcd, pstep, pmem = pol
                if pstep is not None and len(pmem) >= 3:
                    self.ann.dims.append(BoltCircle(center=(pcx, pcy), radius=pcd / 2))
                    # dimension the pitch between two adjacent holes on the ring
                    bear = sorted(
                        (math.degrees(math.atan2(m[1][1] - pcy, m[1][0] - pcx))
                         % 360.0, m) for m in pmem)
                    pick = None
                    for i in range(len(bear)):
                        a0 = bear[i][0]
                        a1 = bear[(i + 1) % len(bear)][0]
                        if abs(((a1 - a0) % 360.0) - pstep) < 1.0:
                            pick = (a0, a1)
                            break
                    if pick:
                        r_arc = pcd / 2 * 0.62
                        self.ann.dims.append(AngleDim(
                            vertex=(pcx, pcy), start_ang=pick[0], end_ang=pick[1],
                            radius=r_arc, text=f"{fmt_raw(round(pstep, 1))}\u00b0"))
                        self.ann.grow(pcx - r_arc - self.txt, pcy - r_arc - self.txt,
                                 pcx + r_arc + self.txt, pcy + r_arc + self.txt)
                        _mark_arc(self.occ, self.ann, (pcx, pcy), r_arc, pick[0], pick[1], self.txt)
                    # the pitch circle itself is drawn: block it too
                    _mark_arc(self.occ, self.ann, (pcx, pcy), pcd / 2, 0.0, 359.9, self.txt,
                              seg_only=True)
                    # A "typical of" note, but only when the per-group callout
                    # will not already carry one: an even single-diameter ring
                    # gets "ON o110 B.C." appended to its own callout below, and
                    # repeating it here would double-dimension the pattern.
                    same_dia = len({round(m[0].diameter, 4) for m in pmem}) == 1
                    grouped = any(
                        detect_bolt_circle(mm) and detect_bolt_circle(mm)[3]
                        for mm in _cluster_by_diameter(pmem).values())
                    if not (same_dia and grouped):
                        anc = _nearest_geometry(self.proj, pmem[0][1]) or pmem[0][1]
                        label = f"{len(pmem)}X ON \u2300{fmt(pcd)} B.C."
                        tail = _place_note(self.occ, self.ann, anc, label, self.proj, self.style,
                                            self.txt, self.gap)
                        # A bolt-circle note states WHERE holes sit, which is
                        # a location dimension, not a description of the bore.
                        self.ann.dims.append(Callout(anchor=anc, text=label,
                                                tail=tail, cls="dimension"))

    def _aligned_edge_dims(self):
        """Size sloping edges along their own direction.

        Limited to the two longest per view: a chamfer flat gets its length
        stated, but a faceted curve is not turned into a fan of numbers.
        """
        if not getattr(self.style, "aligned_dims", True):
            return
        span = max(self.w, self.h)
        gap = self.gap * 0.8
        for (a, b), L in _oblique_runs(self.proj, self.style)[:3]:
            if L < span * 0.08:
                continue
            ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
            nx, ny = -uy, ux
            # push the dimension line to the side away from the view centre,
            # so it lands on paper rather than across the part
            cx = (self.proj.xmin + self.proj.xmax) / 2
            cy = (self.proj.ymin + self.proj.ymax) / 2
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            if (mid[0] - cx) * nx + (mid[1] - cy) * ny < 0:
                nx, ny = -nx, -ny
            text = _sized(self.style, L, self.name + ":aligned:" + str(round(L, 3)))
            tw = len(text) * self.txt * self.style.char_w
            p1 = (a[0] + nx * gap, a[1] + ny * gap)
            p2 = (b[0] + nx * gap, b[1] + ny * gap)
            # Test the corridor the dimension will actually occupy -- the
            # line itself plus the number at its midpoint. Testing the whole
            # axis-aligned bounding box of a diagonal rejects every sloping
            # edge, because that box always contains the part.
            mp = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            if (self.occ.line_cost(p1, p2)
                    or self.occ.rect_cost(mp[0] - tw / 2, mp[1] - self.txt * 0.8,
                                          mp[0] + tw / 2, mp[1] + self.txt * 0.8)):
                continue                       # no room: leave it to the angle
            self.ann.dims.append(LinearDim(p1=a, p2=b, offset=gap,
                                           direction="a", text=text))
            self.occ.mark_line(p1, p2)
            self.ann.add_seg(p1, p2)
            self.ann.grow(min(p1[0], p2[0]) - tw, min(p1[1], p2[1]) - self.txt,
                          max(p1[0], p2[0]) + tw, max(p1[1], p2[1]) + self.txt)

    def _corner_angle_dims(self):
        """Angular dimensions for genuinely oblique corners."""
        # An orthographic view already implies every 90 deg corner, but an oblique
        # face is otherwise undimensioned: the reader can scale the length but not
        # the angle, so the part is not manufacturable from the print.
        if self.style.angle_dims:
            span = max(self.w, self.h)
            found = _find_angles(self.proj, self.style, self.txt)
            found.sort(key=lambda t: -t[4])          # longest edges first
            for vertex, d1, d2, inc, leg in found[:self.style.angle_max_dims]:
                r_arc = min(leg * 0.55, span * 0.22)
                if r_arc < self.txt * 1.6:
                    continue
                # angles are rarely meaningful past 0.1 deg and long decimals
                # clutter the arc; snap to a whole degree when very close to one
                inc_r = round(inc, 1)
                if abs(inc_r - round(inc_r)) < 0.06:
                    inc_r = float(round(inc_r))
                label = f"{fmt_raw(inc_r)}\u00b0"
                # An angle is toleranced like any other size on offices that
                # tolerance at all -- the reference drawing states 79.6 +/-0.1
                if getattr(self.style, "tolerance_style", "none") in (
                        "symmetric", "bilateral"):
                    from .styles import tolerance_text as _tt
                    _a = _tt(self.style, 30.0,
                             f"{getattr(self.style, 'house_style', '')}:"
                             f"ang:{self.name}:{inc_r}")
                    if _a:
                        label += _a.replace(" ", " ", 1) + "\u00b0"
                # place the arc where it does not sit on the geometry
                mid = math.radians(d1 + ((d2 - d1 + 540.0) % 360.0 - 180.0) / 2.0)
                tx = vertex[0] + math.cos(mid) * (r_arc + self.txt * 0.9)
                ty = vertex[1] + math.sin(mid) * (r_arc + self.txt * 0.9)
                tw = len(label) * self.txt * self.style.char_w
                if self.occ.rect_cost(tx - tw / 2, ty - self.txt * 0.6,
                                 tx + tw / 2, ty + self.txt * 0.6) > 0:
                    r_arc *= 0.62
                    if r_arc < self.txt * 1.4:
                        continue
                    tx = vertex[0] + math.cos(mid) * (r_arc + self.txt * 0.9)
                    ty = vertex[1] + math.sin(mid) * (r_arc + self.txt * 0.9)
                self.ann.dims.append(AngleDim(vertex=vertex, start_ang=d1, end_ang=d2,
                                         radius=r_arc, text=label))
                self.occ.mark_rect(tx - tw / 2, ty - self.txt * 0.7,
                              tx + tw / 2, ty + self.txt * 0.7)
                self.ann.grow(min(tx - tw / 2, vertex[0]), min(ty - self.txt, vertex[1]),
                         max(tx + tw / 2, vertex[0]), max(ty + self.txt, vertex[1]))

    def _hole_callouts(self):
        """One leader note per identical hole group, then a repair sweep."""
        self.placed_callouts = []
        # Place callouts in angular order around the view centre. Leaders radiate
        # outward, so notes taken in a consistent sweep rarely cross; taking them
        # in arbitrary dict order is what tangles them.
        vcx = (self.proj.xmin + self.proj.xmax) / 2
        vcy = (self.proj.ymin + self.proj.ymax) / 2

        def _grp_angle(kv):
            k, mem = kv
            cxm = sum(m[1][0] for m in mem) / len(mem)
            cym = sum(m[1][1] for m in mem) / len(mem)
            return math.atan2(cym - vcy, cxm - vcx)

        for key, members in sorted(self.groups.items(), key=_grp_angle):
            cx, cy = (self.proj.xmin + self.proj.xmax) / 2, (self.proj.ymin + self.proj.ymax) / 2
            h0, p0, r0 = max(members, key=lambda m: (m[1][0] - cx) ** 2 + (m[1][1] - cy) ** 2)
            note = h0.note(symbols=getattr(self.style, "symbols", False))
            from .styles import counted as _counted
            note = _counted(self.style, len(members), note)
            if key in self.bolt_circles and not self.tagged:
                bx, by, bd, _ = self.bolt_circles[key]
                from .styles import bolt_circle_text as _bc
                # Equally spaced is a claim about the part: check it before
                # writing it. The angles round the circle must be uniform to
                # within half a degree, which is tighter than any pattern
                # this corpus draws by accident.
                _angs = sorted(math.degrees(math.atan2(m[1][1] - by,
                                                       m[1][0] - bx)) % 360
                               for m in members)
                _eq = False
                if len(_angs) > 2:
                    _steps = [(_angs[i + 1] - _angs[i]) % 360
                              for i in range(len(_angs) - 1)]
                    _steps.append((_angs[0] - _angs[-1]) % 360)
                    _eq = max(_steps) - min(_steps) < 0.5
                note += "\n" + _bc(self.style, f"\u2300{fmt(bd)}", _eq)
                self.ann.dims.append(BoltCircle(center=(bx, by), radius=bd / 2))

            # Per-feature bounding box: the extent of this feature group in the
            # view, useful for downstream tooling (nesting, inspection, CAM) and
            # drawable with --feature-boxes.
            #
            # This is recorded for every group that reads as circles in this view,
            # INCLUDING when the holes are tagged into a table instead of getting
            # leader callouts. The box describes recognised geometry, not the
            # annotation style chosen to present it, so a consumer doing nesting or
            # CAM must not lose it just because the drawing happened to be dense
            # enough to trigger a hole table. Previously the whole body of this
            # loop was skipped when `tagged`, so --hole-table parts recorded zero
            # feature boxes.
            bx0 = min(m[1][0] - m[2] for m in members)
            bx1 = max(m[1][0] + m[2] for m in members)
            by0 = min(m[1][1] - m[2] for m in members)
            by1 = max(m[1][1] + m[2] for m in members)
            self.ann.feature_boxes.append((bx0, by0, bx1, by1,
                                      note.replace("\n", " "), len(members)))

            if self.tagged:                      # the table carries the description
                for _, p, r in members:
                    self.ann.marks.append(CenterMark(center=p, radius=r))
                continue

            c = _place_callout(self.occ, p0, r0, note, self.proj, self.style, self.txt,
                                segs=self.ann.segments)
            # An arrow that cannot reach the feature points at nothing. This happens
            # when a hole is edge-on in this view, so no arc is drawn for it: the
            # note belongs on a view where the hole reads as a circle.
            if c.snap_gap > max(r0, self.txt) * 1.5:
                for _, p, r in members:
                    self.ann.marks.append(CenterMark(center=p, radius=r))
                continue
            self.ann.dims.append(c)
            _record_leader(self.ann, c, self.txt)
            self.placed_callouts.append((c, p0, r0, note))
            if c.box:
                self.ann.grow(*c.box)
            for _, p, r in members:
                self.ann.marks.append(CenterMark(center=p, radius=r))

    def _leader_is_bad(self, c, others_t):
        """Does this callout cross a committed segment, box or leader?"""
        own = _leader_segs(c, self.txt)
        rest = [sg for sg in self.ann.segments if sg not in own]
        if any(_seg_cross(a1, b1, a2, b2)
               for (a1, b1) in own for (a2, b2) in rest):
            return True, rest
        if _crosses_boxes(c, self.ann.dim_boxes):
            return True, rest
        if _leader_crosses(c, others_t, self.txt):
            return True, rest
        return False, rest

    def _repair_leaders(self):
        """Re-place callouts crossed by anything committed after them.

        A callout can only avoid what already exists when it is placed,
        so early notes get crossed by later leaders and by dimension
        lines emitted afterwards. Sweep until nothing moves: a single
        pass just relocates the problem, because fixing one note can
        introduce a crossing with another.
        """
        for _ in range(self.style.callout_repair_sweeps):
          changed = False
          for c, p0, r0, note in self.placed_callouts:
              if not c.box:
                  continue
              others = [t for t in self.placed_callouts if t[0] is not c]
              bad, rest = self._leader_is_bad(c, others)
              if not bad:
                  continue
              # vacate this note's cells so the retry can reuse them
              self.occ.clear_rect(*c.box)
              for s in self.proj.segs + self.proj.circles:
                  self.occ.mark_poly(s.pts)
              for tb in self.ann.dim_boxes:
                  self.occ.mark_rect(*tb)
              n = _place_callout(self.occ, p0, r0, note, self.proj, self.style, self.txt,
                                  avoid=self.ann.dim_boxes, segs=rest, commit=False)
              if not n.box:
                  continue
              probe = RadiusDim(center=p0, radius=r0, text=note,
                              angle=n.angle, leader_len=n.leader_len,
                              box=n.box, seg_start=n.seg_start,
                              seg_knee=n.seg_knee, seg_shelf=n.seg_shelf)
              new_segs = _leader_segs(probe, self.txt)
              still = any(_seg_cross(a1, b1, a2, b2)
                          for (a1, b1) in new_segs for (a2, b2) in rest)
              if (still or _crosses_boxes(probe, self.ann.dim_boxes)
                      or _leader_crosses(probe, others, self.txt)):
                  continue                      # no better option; keep as is
              for sg in _leader_segs(c, self.txt):
                  if sg in self.ann.segments:
                      self.ann.segments.remove(sg)
              c.angle, c.leader_len, c.box = n.angle, n.leader_len, n.box
              c.seg_start, c.seg_knee, c.seg_shelf = (n.seg_start, n.seg_knee,
                                                         n.seg_shelf)
              self.occ.mark_line(n.seg_start, n.seg_knee)
              self.occ.mark_line(n.seg_knee, n.seg_shelf)
              self.occ.mark_rect(n.box[0], n.box[1], n.box[2], n.box[3])
              _record_leader(self.ann, c, self.txt)
              self.ann.grow(*n.box)
              changed = True
          if not changed:
              break

    def _outer_diameter_dim(self):
        """The deferred diameter note for a round view."""
        # Emitted after the position dimensions and hole callouts so the placement
        # search can avoid their extension lines. Emitting it first (the obvious
        # order, since it is an "overall" dimension) put the note where an
        # extension line was later drawn straight through it.
        if self.od_centre is not None:
            ocx, ocy = self.od_centre
            dtext = "\u2300" + fmt(2 * self.od_radius)
            # The overall diameter of a round part is its SIZE. It uses the
            # same leader primitive as a hole callout but states something
            # different, which is why the class is carried explicitly.
            dd = _place_callout(self.occ, (ocx, ocy), self.od_radius, dtext, self.proj, self.style, self.txt,
                                 segs=self.ann.segments, cls="dimension")
            self.ann.dims.append(dd)
            _record_leader(self.ann, dd, self.txt)
            if dd.box:
                self.ann.grow(*dd.box)

    def _thickness_note(self):
        """Plate thickness note."""
        if self.primary and self.info.is_plate and self.info.thickness:
            tgt = (self.proj.xmax, self.proj.ymax)
            anc = _nearest_geometry(self.proj, tgt, prefer_circle_edge=False) or tgt
            label = f"THICKNESS {fmt(self.info.thickness)}"
            tail = _place_note(self.occ, self.ann, anc, label, self.proj, self.style, self.txt, self.gap)
            # A free-text leader, not a dimension: it has no witness lines and
            # names no feature type, so it reads as a note on the drawing.
            self.ann.dims.append(Callout(anchor=anc, text=label, tail=tail,
                                         cls="leader_note"))

    def _fillet_notes(self):
        """One note per distinct fillet radius, each anchored on such an arc."""
        if self.primary and self.info.fillets:
            # Every distinct radius gets its own note, not just the most common
            # one. 0000_00000093 carries R0.52 x6 AND R0.23 x3; noting only the
            # first left three visible arcs undimensioned, so the part could not
            # be made from the print. Largest count first (the dominant radius
            # reads as the "general" one), then largest radius, and capped so a
            # part with a dozen incidental radii does not bury the drawing.
            fl = sorted(self.info.fillets, key=lambda x: (-x.count, -x.radius))
            fl = [f for f in fl if f.count >= self.style.fillet_note_min_count]
            used = []
            for f in fl[:self.style.fillet_max_notes]:
                # Anchor on an arc that actually has THIS radius, and not on one
                # already used by another radius note. Every group used to snap to
                # the same guessed top-left corner, so three notes shared one
                # anchor and their leaders stacked on top of each other.
                arcs = _arc_anchor(self.proj, f.radius, used,
                                   want_centre="all")
                anc = arcs[0][0] if arcs else None
                if anc is None:
                    tgt = (self.proj.xmin + f.radius * 0.3,
                           self.proj.ymax - f.radius * 0.3)
                    anc = _nearest_geometry(self.proj, tgt,
                                            prefer_circle_edge=False) or tgt
                used.append(anc)
                label = (f"{f.count}X R{fmt(f.radius)}" if f.count > 1
                          else f"R{fmt(f.radius)}")
                # A radius is stated on whichever corner has room for it:
                # try every arc of this radius on the radial before giving up
                # and letting the generic solver put it anywhere.
                tail = None
                for cand_a, cand_c in (arcs or [])[:6]:
                    t = _radial_tail(self.occ, self.ann, cand_a, cand_c, label,
                                     self.proj, self.style, self.txt,
                                     self.gap, strict=True)
                    if t is not None:
                        anc, tail = cand_a, t
                        break
                if tail is None:
                    tail = _place_note(self.occ, self.ann, anc, label,
                                       self.proj, self.style, self.txt,
                                       self.gap)
                self.ann.dims.append(Callout(anchor=anc, text=label, tail=tail,
                                             cls="radius"))

        # The notes above are committed segments the hole callouts never saw, so
        # sweep once more. Without this a note leader placed late could cross a
        # callout leader with nothing left to move it.
        self._repair_leaders()

        # ...and the notes themselves need the same treatment.
        self._sweep_note_leaders()

    def _sweep_note_leaders(self):
        """Re-place any leader note whose leader crosses a committed segment.

        A note is placed against whatever exists at the time, so an earlier
        note can be crossed by a later one's leader. Each candidate is
        re-placed with its own segment temporarily withdrawn, so the solver
        does not score it against itself, and the original is restored when
        the retry is no better.

        Only ``Callout`` is swept.

        REVERTED: extending this to the GD&T frames, datum flags and roughness
        marks -- and running a second sweep after they are placed -- was tried
        to clear the 3 leader-leader crossings the new annotations introduced
        on the NIST sheet. It made things distinctly WORSE: lead-lead 3 -> 8,
        txt-line 2 -> 7, lead-dim 5 -> 19. The retry path re-places a frame
        using only the tail returned by ``_place_note``, which does not
        reproduce the geometry ``_feature_frame`` actually draws (the frame
        extends from the tail by its full compartment width, on whichever side
        the placer chose), so a "repaired" frame lands somewhere the solver
        never scored. Fixing this properly needs the frame placement to be
        solved as frame geometry, not approximated by a text-note tail.
        """
        kinds = (Callout,)
        for _ in range(self.style.callout_repair_sweeps):
            changed = False
            for d in self.ann.dims:
                if not isinstance(d, kinds):
                    continue
                own = (d.anchor, d.tail)
                rest = [sg for sg in self.ann.segments if sg != own]
                crossed = any(_seg_cross(d.anchor, d.tail, a2, b2)
                               for (a2, b2) in rest)
                if not crossed:
                    continue
                if own in self.ann.segments:
                    self.ann.segments.remove(own)
                new = _place_note(self.occ, self.ann, d.anchor,
                                   getattr(d, "text", "") or "",
                                   self.proj, self.style, self.txt, self.gap,
                                   size=self._note_footprint(d),
                                   centred=isinstance(d, (FeatureFrame,
                                                          DatumFlag)))
                if any(_seg_cross(d.anchor, new, a2, b2) for (a2, b2) in rest):
                    # no improvement: put the original back
                    if (d.anchor, new) in self.ann.segments:
                        self.ann.segments.remove((d.anchor, new))
                    self.ann.add_seg(*own)
                    continue
                d.tail = new
                changed = True
            if not changed:
                break

    def _note_footprint(self, d):
        """Reserved (w, h) for a leader annotation, or None for plain text.

        Mirrors what each renderer actually draws, so the retry reserves the
        same area the first placement did.
        """
        fh = self.txt * 1.6
        pad = fh * 0.22
        if isinstance(d, FeatureFrame):
            w = fh + len(d.value_text) * self.txt * self.style.char_w + 2 * pad
            w += sum(self.txt * self.style.char_w + 2 * pad for _ in d.datums)
            return (w, fh)
        if isinstance(d, DatumFlag):
            return (fh, fh)
        if isinstance(d, RoughnessMark):
            return (self.txt * 1.5 + len(d.ra_text) * self.txt * self.style.char_w
                    + self.txt, fh)
        return None

    def _chamfer_notes(self):
        """One note per distinct chamfer / countersink.

        Suppressed when this view's holes are tagged into a hole table: the
        table already describes every bore, and a "15X 0.5 X 30 deg" leader
        beside it is exactly the clutter the table exists to avoid.

        Only on the view where the break reads: a countersink is a pair of
        concentric circles face-on and an invisible sliver edge-on, so the
        note is placed where its axis points at the viewer, exactly as the
        hole callouts are.
        """
        if not (self.primary and getattr(self.info, "chamfers", [])):
            return
        if self.tagged:
            return
        from .projection import _axes
        dv, _, _ = _axes(self.proj.name)
        used = []
        # Same crowding argument as the GD&T budget: a dense view has no room
        # for several more leaders.
        budget = self._gdt_budget()
        # A synthetic chamfer is skipped entirely on a crowded view; a REAL
        # one (recognised from a conical face) is always worth stating.
        n_ch = min(self.style.chamfer_max_notes, max(1, budget))
        for ch in self.info.chamfers[:n_ch]:
            if budget == 0 and getattr(ch, "synthetic", False):
                continue
            a = ch.axis
            if abs(a[0] * dv.x + a[1] * dv.y + a[2] * dv.z) <= 0.999:
                continue                      # edge-on in this view
            p = world_to_view(self.proj.name, ch.position)
            eps = max(self.w, self.h) * 1e-3
            if not (self.proj.xmin - eps <= p[0] <= self.proj.xmax + eps
                    and self.proj.ymin - eps <= p[1] <= self.proj.ymax + eps):
                continue
            from .styles import counted as _counted
            label = _counted(self.style, ch.count, ch.note(self.style))
            anc = _arc_anchor(self.proj, ch.radius_max, used) \
                or _nearest_geometry(self.proj, p) or p
            used.append(anc)
            tail = _place_note(self.occ, self.ann, anc, label, self.proj,
                               self.style, self.txt, self.gap)
            self.ann.dims.append(Callout(
                anchor=anc, text=label, tail=tail, cls="chamfer",
                synthetic=getattr(ch, "synthetic", False)))

    def _thread_notes(self):
        """Thread specifications stated by the file, placed on their bore.

        Threads are never inferred: this draws only what the STEP's PMI
        actually declares, so a file without thread PMI gets no thread note
        even when it is full of tapped-looking holes.
        """
        pmi = getattr(self.info, "pmi", None)
        if not (pmi and getattr(pmi, "threads", None)):
            return
        # As for chamfers: the hole table already carries the bore data.
        if self.tagged:
            return
        from .projection import _axes
        dv, _, _ = _axes(self.proj.name)

        # Threads are NOT restricted to the primary view.
        #
        # They used to be, which was close to the worst possible rule: the
        # primary view is chosen because it shows the most features, so it is
        # always the most crowded one (14 and 19 annotations already placed on
        # bracket_plate and the NIST part, against a crowding threshold of
        # 11), while genuinely empty secondary views (2, 8, 6 placed) were
        # barred outright. A thread was therefore offered only the one view
        # with no room for it.
        #
        # A thread is drawn on any view where its bore actually reads as a
        # circle -- the axis test below -- which is the same rule the hole
        # callouts use. Each thread is claimed by the first view that can
        # take it, so the same spec is never called out twice on one sheet.
        # A thread is manufacturing-critical: a tapped hole drawn as a plain
        # bore gets MADE as a plain bore.
        room = self._has_room_for_note(self.style.thread_min_free_slots)
        n_th = self.style.thread_max_notes if room else 1
        drawn = self.ann.thread_ids
        for th in pmi.threads:
            if len(drawn) >= n_th:
                break
            key = th.designation
            if key in drawn:
                continue          # already stated on another view
            # On a view with no room for a full budget (``room`` False) the
            # budget is one note, but that one note may still be a SYNTHETIC
            # thread. The free-slot gate above is the quality protection -- it
            # guarantees at least some genuinely free label placements exist --
            # so the thread is not dropped outright; it is merely limited to a
            # single leader, which keeps a saturated drawing from taking many.
            # A tapped hole drawn as a plain bore gets made as a plain bore, so
            # a synthetic thread is the one callout worth the space on a busy
            # view. (This used to skip every synthetic thread here, which left
            # most eligible parts with no thread annotation at all.)
            if th.anchor is None:
                continue
            # The bore must read as a circle here, or the leader points at a
            # slot of empty paper where the hole is edge-on.
            a = getattr(th, "axis", None) or (0.0, 0.0, 1.0)
            if abs(a[0] * dv.x + a[1] * dv.y + a[2] * dv.z) <= 0.999:
                continue
            p = world_to_view(self.proj.name, th.anchor)
            eps = max(self.w, self.h) * 1e-3
            if not (self.proj.xmin - eps <= p[0] <= self.proj.xmax + eps
                    and self.proj.ymin - eps <= p[1] <= self.proj.ymax + eps):
                continue
            drawn.add(key)
            anc = _nearest_geometry(self.proj, p) or p
            label = th.text()
            tail = _place_note(self.occ, self.ann, anc, label, self.proj,
                               self.style, self.txt, self.gap)
            self.ann.dims.append(Callout(
                anchor=anc, text=label, tail=tail, cls="thread",
                synthetic=getattr(th, "synthetic", False)))

    def _has_room_for_note(self, need: int, width_chars: int = 14) -> bool:
        """Does this view still have `need` genuinely free places for a label?

        Counting annotations already placed is a poor proxy for congestion and
        was measured to conflate two very different views: bracket_plate's top
        view had 14 annotations but **26 of 96** candidate placements that hit
        no ink, crossed no committed segment and stayed inside the view, while
        the dense 160x90 fixture had 19 and its best candidate still landed on
        ink. A count threshold either admitted both (dropping the dense sheet
        from 1:1.25 to 1:2) or rejected both (losing a thread that had plenty
        of room). Probing the space directly separates them.

        The probe mirrors the geometry `_place_note` searches, so "free here"
        means the same thing as "free there".
        """
        base = max(self.proj.width, self.proj.height)
        txt = self.txt
        # Width of the label being placed. A thread note is ~14 characters;
        # a GD&T frame is a row of boxed compartments and needs far more, so
        # the caller states it -- probing at the wrong width answers a
        # different question than the one being asked.
        tw = width_chars * txt * self.style.char_w
        free = 0
        for angd in range(0, 360, 15):
            a = math.radians(angd)
            for mult in (0.18, 0.28, 0.40, 0.55):
                tail = (self.proj.xmin + (self.proj.xmax - self.proj.xmin) / 2
                        + math.cos(a) * base * mult,
                        self.proj.ymin + (self.proj.ymax - self.proj.ymin) / 2
                        + math.sin(a) * base * mult)
                x0 = tail[0] + txt * 0.35
                box = (x0, tail[1] + txt * 0.25, x0 + tw,
                       tail[1] + txt * 0.25 + txt)
                if self.occ.rect_cost(*box) > 0:
                    continue
                reach = (max(0.0, self.proj.xmin - min(box[0], box[2]))
                         + max(0.0, max(box[0], box[2]) - self.proj.xmax)
                         + max(0.0, self.proj.ymin - box[1])
                         + max(0.0, box[3] - self.proj.ymax))
                if reach > 0:
                    continue
                free += 1
                if free >= need:
                    return True
        return False

    def _gdt_budget(self) -> int:
        """How many feature control frames this view can take.

        A frame is a wide, tall object on a leader, so a crowded view cannot
        absorb the same number as an empty one. Budgeting by how much
        annotation is already committed keeps a dense part readable: on a
        160x90 plate with 22 holes, a flat budget of 6 frames drove
        leader-dimension crossings from 2 to 14 and cost a whole scale step,
        because every frame had to be pushed further out to find space.

        Counted against annotations already placed on this view, since those
        are what the frame has to avoid.
        """
        placed = sum(1 for d in self.ann.dims
                     if isinstance(d, (Callout, RadiusDim, LinearDim,
                                       FeatureFrame, DatumFlag,
                                       RoughnessMark)))
        cap = self.style.gdt_max_frames
        # Above this the view has no room left, and forcing annotation on
        # anyway does not merely crowd it -- the sheet fitter has to pad the
        # view to contain the overflow, which costs real scale steps. On the
        # dense 160x90 fixture (22 holes, 20 dimensions) the extra leaders
        # dropped the drawing from 1:1.25 to 1:2, two full steps, making every
        # number on the sheet smaller. A legible drawing with fewer synthetic
        # marks beats an illegible one with more.
        # Congestion is measured BOTH ways, because each catches a case the
        # other misses.
        #
        # The count alone was measuring the wrong thing: every sample part
        # places 9-15 annotations before this step runs, so a threshold of 10
        # suppressed GD&T on essentially every drawing -- the NIST part
        # carries 34 real tolerances from its AP242 PMI and stated none.
        #
        # The probe alone is not enough either: it samples outward from the
        # view centre, so it reports space that a frame anchored at a specific
        # feature cannot actually reach. On the dense 160x90 fixture it passed
        # while three frames still dropped the sheet from 1:1.25 to 1:2.5 and
        # took leader-dimension crossings from 2 to 8.
        if placed >= self.style.gdt_crowded_at:
            return 0
        if not self._has_room_for_note(self.style.gdt_min_free_slots,
                                       width_chars=self.style.gdt_frame_chars):
            return 0

        if placed >= 12:
            return min(cap, 2)
        if placed >= 8:
            return min(cap, 3)
        return cap

    def _gdt_frames(self):
        """Feature control frames and datum flags read from the file's PMI."""
        pmi = getattr(self.info, "pmi", None)
        if not (self.primary and pmi and getattr(pmi, "tolerances", None)):
            return
        # REVERTED: opening GD&T frames to secondary views, the way thread
        # callouts were opened, was tried and made the drawing worse -- one
        # 17 mm2 text overlap on the NIST sheet and a lost scale step
        # (1:1.5 -> 1:2). Measuring the free space at a realistic frame width
        # shows why: the NIST views offer 0, 1 and 0 placements that a frame
        # actually fits in, against 3, 1 and 0 for a short label. A frame is a
        # row of boxed compartments, several times wider than a note, so
        # "the secondary views look emptier" does not translate into room for
        # one. The count-based budget hid that by measuring the wrong thing.
        # A part can carry dozens of frames (the NIST sample has 34). Drawing
        # all of them on one view would bury the geometry, so the largest
        # tolerance zones are preferred -- those are the ones a reader checks
        # first -- and the rest stay in the JSON record.
        tols = sorted(pmi.tolerances, key=lambda t: -t.value)
        budget = self._gdt_budget()
        drawn = self.ann.frame_ids
        placed = 0
        for t in tols:
            key = (t.characteristic, t.value, tuple(t.datums), t.anchor)
            if key in drawn:
                continue
            if placed >= budget:
                break
            if t.anchor is None:
                continue
            p = world_to_view(self.proj.name, t.anchor)
            eps = max(self.w, self.h) * 1e-3
            if not (self.proj.xmin - eps <= p[0] <= self.proj.xmax + eps
                    and self.proj.ymin - eps <= p[1] <= self.proj.ymax + eps):
                continue
            anc = _nearest_geometry(self.proj, p, prefer_circle_edge=False) or p
            from .geometry import fmt as _fmt
            vtext = ("\u2300" if t.diametral else "") + _fmt(t.value)
            if t.modifier:
                vtext += f"({t.modifier})"
            # Reserve the frame's ACTUAL footprint, not a line of text. The
            # geometry here mirrors render._feature_frame: a square symbol
            # compartment, a value compartment, and one per datum, all
            # 1.6 text-heights tall and centred on the leader.
            fh = self.txt * 1.6
            pad = fh * 0.22
            fw = fh + len(vtext) * self.txt * self.style.char_w + 2 * pad
            fw += sum(self.txt * self.style.char_w + 2 * pad for _ in t.datums)
            tail = _place_note(self.occ, self.ann, anc, "", self.proj,
                               self.style, self.txt, self.gap,
                               size=(fw, fh), centred=True)
            self.ann.dims.append(FeatureFrame(
                anchor=anc, characteristic=t.characteristic, value_text=vtext,
                datums=tuple(t.datums), modifier=t.modifier, tail=tail,
                text=t.text(), value=t.value,
                synthetic=getattr(t, "synthetic", False)))
            drawn.add(key)
            placed += 1

        # Datum feature symbols for the letters the placed frames reference.
        cited = {d for f in self.ann.dims if isinstance(f, FeatureFrame)
                 for d in f.datums}
        # Datum flags are budgeted too. They are boxed symbols on their own
        # leaders, so on a crowded view they cost as much space as a frame;
        # leaving them uncapped meant a view budgeted down to 2 frames still
        # sprouted 2 more flags beside them.
        n_dat = min(self.style.gdt_max_datums, max(1, self._gdt_budget()))
        for dat in getattr(pmi, "datums", [])[:n_dat]:
            if dat.letter not in cited:
                continue
            # A datum feature symbol attaches to the surface that establishes
            # the datum. Without a resolved face, put it on the silhouette
            # nearest the view's lower-left, which is where a reader expects
            # the primary datums.
            tgt = (self.proj.xmin, self.proj.ymin)
            anc = _nearest_geometry(self.proj, tgt, prefer_circle_edge=False) \
                or tgt
            fh = self.txt * 1.6
            tail = _place_note(self.occ, self.ann, anc, "", self.proj,
                               self.style, self.txt, self.gap,
                               size=(fh, fh), centred=True)
            self.ann.dims.append(DatumFlag(
                anchor=anc, letter=dat.letter, tail=tail,
                synthetic=getattr(dat, "synthetic", False)))

    def _roughness_marks(self):
        """Surface-texture symbols.

        The general note is emitted by the sheet (it belongs beside the title
        block, not on a view); this places the per-feature ones only.
        """
        marks = [r for r in getattr(self.info, "roughness", []) or []
                 if not r.general and r.target is not None]
        if not (self.primary and marks):
            return
        for r in marks[:self.style.roughness_max_marks]:
            p = world_to_view(self.proj.name, r.target)
            eps = max(self.w, self.h) * 1e-3
            if not (self.proj.xmin - eps <= p[0] <= self.proj.xmax + eps
                    and self.proj.ymin - eps <= p[1] <= self.proj.ymax + eps):
                continue
            anc = _nearest_geometry(self.proj, p) or p
            # The tick rises ~1.1 text-heights above the leader tail and the
            # Ra text sits to its right, so reserve both rather than the text
            # alone.
            rw = self.txt * 1.5 + len(r.text()) * self.txt * self.style.char_w + self.txt
            tail = _place_note(self.occ, self.ann, anc, "", self.proj,
                               self.style, self.txt, self.gap,
                               size=(rw, self.txt * 1.6))
            self.ann.dims.append(RoughnessMark(
                anchor=anc, ra_text=r.text(), machined=r.machined, tail=tail))

    def _repair_notes(self):
        """Re-place any note leader that crosses another committed segment."""
        # NOTE: a final self._repair_leaders() here (letting a callout step aside for
        # a note, rather than only the reverse) was tried and REVERTED -- it moved
        # nothing on any of the 18 sample parts. The residual crossings are cases
        # where the solver has no better candidate, not cases where the wrong
        # object is being moved.

    def _finish(self):
        """Turn the accumulated annotation extent into view padding."""
        # dimension lines themselves
        self.ann.grow(self.proj.xmin - self.gap * self.lvl_l, self.proj.ymin - self.gap * self.lvl_b,
                 self.proj.xmax, self.proj.ymax)
        # ...and then every dimension actually emitted, on whichever side it
        # was drawn. The heuristic above assumes a bottom-left datum corner;
        # a chain running along the top or right edge, or an aligned dimension
        # standing off a sloping face, sits outside it. The view block is
        # sized from this padding, so anything missing here is ink the layout
        # never reserved room for -- which is how witness lines ended up
        # running off the top of the sheet and their boxes off the image.
        for d in self.ann.dims:
            if not isinstance(d, LinearDim):
                continue
            t = self.txt * 2.2
            if d.direction == "h":
                y = min(d.p1[1], d.p2[1]) + d.offset
                self.ann.grow(min(d.p1[0], d.p2[0]) - t, y - t,
                              max(d.p1[0], d.p2[0]) + t, y + t)
            elif d.direction == "v":
                x = min(d.p1[0], d.p2[0]) + d.offset
                self.ann.grow(x - t, min(d.p1[1], d.p2[1]) - t,
                              x + t, max(d.p1[1], d.p2[1]) + t)
            else:                      # aligned: offset along the edge normal
                off = abs(d.offset) + t
                self.ann.grow(min(d.p1[0], d.p2[0]) - off,
                              min(d.p1[1], d.p2[1]) - off,
                              max(d.p1[0], d.p2[0]) + off,
                              max(d.p1[1], d.p2[1]) + off)
        # the leader/notes placed above
        if self.primary and (self.info.is_plate or self.info.fillets):
            self.ann.grow(self.proj.xmin - self.gap * 0.9, self.proj.ymin, self.proj.xmax + self.gap * 0.9,
                     self.proj.ymax + self.gap * 0.9)

        x0, y0, x1, y1 = self.ann.ext
        m = self.txt * 1.2                                   # breathing room for glyphs
        label_strip = self.txt * 2.6                         # room for the view label
        pad_l = max(0.0, self.proj.xmin - x0) + m
        pad_b = max(0.0, self.proj.ymin - y0) + m + label_strip
        pad_r = max(0.0, x1 - self.proj.xmax) + m
        pad_t = max(0.0, y1 - self.proj.ymax) + m
        self.ann.pad = (pad_l, pad_r, pad_b, pad_t)
        return self.ann


def annotate(proj: Projection, info: PartInfo, *, style, primary: bool,
             hole_table: bool = False, tagged: bool = False,
             scale: Optional[float] = None,
             thread_ids: Optional[set] = None,
             frame_ids: Optional[set] = None) -> Annotated:
    """Dimension one view.

    scale -- drawing scale (paper mm per model unit). Passing None assumes
             1:1; the pipeline re-runs this once the real scale is known so
             that reserved space matches what is actually drawn.

    hole_table -- this view's hole positions live in a table, so skip the
                  X/Y baseline forest.
    tagged     -- this view's holes are tagged into a table, so skip the
                  per-group leader callouts too (center marks are kept).
    thread_ids -- set shared across the sheet's views, recording which thread
                  designations have already been called out. Threads may be
                  drawn on any view where their bore reads as a circle, so
                  without this the same spec appears on two views.
    """
    if proj.width <= 0 or proj.height <= 0:
        return Annotated(view=proj)
    ann = _Annotator(proj, info, style=style, primary=primary,
                     hole_table=hole_table, tagged=tagged, scale=scale)
    if thread_ids is not None:
        ann.ann.thread_ids = thread_ids
    if frame_ids is not None:
        ann.ann.frame_ids = frame_ids
    return ann.run()


def _sc_of(style, txt):
    """Recover the model-units-per-paper-mm factor from the converted text."""
    return style.text_height / txt if txt else 1.0



def _radial_tail(occ, ann, anchor, centre, label, proj, style, txt, gap,
                 strict=False):
    """Where a radius note's leader ends: out along the arc's own radius.

    ISO 129 draws a radius on the radius line -- leader collinear with the
    centre-to-arc direction, arrowhead landing on the arc. The generic note
    solver searches bearings for free space and never consults the centre,
    which put 15 of 16 radius notes on this corpus off the radial by a median
    of 54 degrees (one at 178, pointing back through the part).

    Only the LENGTH is searched here; the direction is fixed by the geometry.
    If nothing along the ray is free, the generic solver takes over rather
    than stacking the note on ink.
    """
    if centre is None or anchor is None:
        return None if strict else _place_note(occ, ann, anchor, label, proj,
                                               style, txt, gap)
    dx, dy = anchor[0] - centre[0], anchor[1] - centre[1]
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return None if strict else _place_note(occ, ann, anchor, label, proj,
                                               style, txt, gap)
    ux, uy = dx / L, dy / L
    tw = len(label) * txt * style.char_w
    span = max(proj.width, proj.height)
    for mult in (0.9, 1.3, 1.8, 2.4, 3.1, 4.0):
        t = gap * mult
        tail = (anchor[0] + ux * t, anchor[1] + uy * t)
        x0 = tail[0] if ux >= 0 else tail[0] - tw
        # The anchor sits ON the arc, which is ink: testing the leader from
        # there always fails and the note fell back to the generic solver
        # every single time. Test from just clear of the arc outward.
        start = (anchor[0] + ux * txt * 0.8, anchor[1] + uy * txt * 0.8)
        if occ.rect_cost(x0, tail[1] - txt * 0.7, x0 + tw,
                         tail[1] + txt * 0.9) == 0 \
                and occ.line_cost(start, tail) == 0 \
                and abs(tail[0] - proj.xmin) < span * 3:
            occ.mark_rect(x0, tail[1] - txt * 0.8, x0 + tw, tail[1] + txt * 1.0)
            occ.mark_line(anchor, tail)
            ann.add_seg(anchor, tail)
            ann.grow(min(x0, anchor[0]), min(tail[1] - txt, anchor[1]),
                     max(x0 + tw, anchor[0]), max(tail[1] + txt, anchor[1]))
            return tail
    return None if strict else _place_note(occ, ann, anchor, label, proj,
                                           style, txt, gap)


def _arc_anchor(proj, radius, used, tol_frac=0.06, want_centre=False):
    """A point on a drawn arc whose radius matches ``radius``.

    Prefers arcs far from the anchors already taken, so several radius notes
    on one view fan out instead of all pointing at the same corner.

    With ``want_centre`` the arc's centre comes back too: a radius note has to
    be drawn on the arc's own radius line (ISO 129), which needs the centre,
    not just a point on the curve.
    """
    cands = []
    for seg in proj.segs + proj.circles:
        if seg.kind != "visible" or not seg.pts:
            continue
        r = getattr(seg, "radius", 0.0)
        if r <= 0 or abs(r - radius) > max(radius * tol_frac, 1e-6):
            continue
        cands.append((seg.pts[len(seg.pts) // 2], getattr(seg, "center", None)))
    if not cands:
        return (None, None) if want_centre else None
    if not used:
        # start at the top-left-most arc, matching the old visual convention
        cands.sort(key=lambda pc: (-pc[0][1], pc[0][0]))
    else:
        cands.sort(key=lambda pc: -min(math.hypot(pc[0][0] - u[0],
                                                  pc[0][1] - u[1])
                                       for u in used))
    if want_centre == "all":
        return cands
    best = cands[0]
    return best if want_centre else best[0]


def _nearest_geometry(proj, target, prefer_circle_edge=True):
    """Closest point on drawn geometry to ``target``, in view coordinates.

    Note leaders (thickness, fillet) used to anchor at a guessed corner of the
    view box, so their arrowheads could land in empty space -- pointing at
    nothing. Snapping the anchor to real geometry guarantees the arrow lands
    on the part.
    """
    best = None
    best_d = None
    if prefer_circle_edge:
        for seg in proj.circles:
            if not seg.is_circle or seg.center is None or seg.radius <= 0:
                continue
            cx, cy = seg.center
            dx, dy = target[0] - cx, target[1] - cy
            L = math.hypot(dx, dy)
            if L < 1e-12:
                continue
            q = (cx + dx / L * seg.radius, cy + dy / L * seg.radius)
            dd = math.hypot(q[0] - target[0], q[1] - target[1])
            if best_d is None or dd < best_d:
                best, best_d = q, dd
    for seg in proj.segs + proj.circles:
        pts = seg.pts
        for i in range(len(pts) - 1):
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            vx, vy = bx - ax, by - ay
            L2 = vx * vx + vy * vy
            if L2 < 1e-18:
                q = (ax, ay)
            else:
                t = ((target[0] - ax) * vx + (target[1] - ay) * vy) / L2
                t = 0.0 if t < 0 else (1.0 if t > 1 else t)
                q = (ax + vx * t, ay + vy * t)
            dd = math.hypot(q[0] - target[0], q[1] - target[1])
            if best_d is None or dd < best_d:
                best, best_d = q, dd
    return best


def _arc_sweep_deg(seg, cx, cy):
    """Degrees of arc covered by a sampled polyline about (cx, cy).

    Summed step by step rather than taken end-to-end, so an arc of more than
    180 degrees is not mistaken for its complement.
    """
    pts = seg.pts
    if len(pts) < 2:
        return 0.0
    total = 0.0
    prev = math.atan2(pts[0][1] - cy, pts[0][0] - cx)
    for x, y in pts[1:]:
        a = math.atan2(y - cy, x - cx)
        d = (a - prev + math.pi) % (2 * math.pi) - math.pi
        total += abs(d)
        prev = a
    return math.degrees(total)


def outer_diameter(proj, tol_frac=0.02):
    """Radius of the silhouette when the view is a full circle, else None.

    A round part dimensioned as width x height is wrong twice over: it states
    two numbers where one suffices, and neither of them says the shape is
    round. A machinist reading "1.5 x 1.5" has to infer the part is a disc
    from the picture; "o1.5" states it.

    The test is deliberately strict -- a full 360 degree circle, concentric
    with the silhouette, whose diameter fills the bounding box in BOTH axes.
    A rounded-corner square would otherwise be mistaken for a disc.
    """
    w, h = proj.width, proj.height
    if w <= 0 or h <= 0 or abs(w - h) > max(w, h) * tol_frac:
        return None
    cx, cy = (proj.xmin + proj.xmax) / 2, (proj.ymin + proj.ymax) / 2
    # Accumulate the arc swept by every concentric full-size arc, not just
    # whole circles. A silhouette that happens to be split into two 180 deg
    # halves is still a circle -- a sphere projects that way in the front and
    # side views, and was dimensioned "20 x 20" while its top view (one whole
    # circle) correctly read "o20".
    best = None
    swept = 0.0
    for c in proj.circles:
        if not (c.is_circle and c.center) or c.kind != "visible":
            continue
        # must be concentric with the silhouette and fill it
        if math.hypot(c.center[0] - cx, c.center[1] - cy) > max(w, h) * tol_frac:
            continue
        if abs(2 * c.radius - w) > w * tol_frac:
            continue
        swept += 360.0 if c.full else _arc_sweep_deg(c, cx, cy)
        if best is None or c.radius > best:
            best = c.radius
    if best is None or swept < 350.0:
        return None
    # Guard against a circular *feature* on a square plate that happens to
    # touch the edges: require no straight silhouette edge running the full
    # width or height of the view.
    for s in proj.segs:
        if s.kind != "visible" or len(s.pts) < 2:
            continue
        xs = [p[0] for p in s.pts]
        ys = [p[1] for p in s.pts]
        if max(ys) - min(ys) < h * 1e-3 and max(xs) - min(xs) > w * 0.9:
            return None
        if max(xs) - min(xs) < w * 1e-3 and max(ys) - min(ys) > h * 0.9:
            return None
    return best


def _anchor_at(proj, pos, direction, fallback, tol, side="min"):
    """Where a witness line should start for a dimension at ``pos``.

    The line must begin just off the drawn outline and run outward to the
    dimension line. Anchoring at the view-box corner leaves it stopping short
    of the edge it refers to -- most visibly on a filleted corner, where the
    bounding-box corner has no material at all and the witness floats a whole
    fillet radius away from anything.

    Strategy: find geometry that lines up with ``pos`` on the measured axis and
    take the point closest to the dimension line. If nothing lines up (``pos``
    falls in a fillet's shadow, or off the part), widen the search until real
    geometry is found, so the line always terminates on something.
    """
    def _scan(width):
        found = []
        for seg in proj.segs + proj.circles:
            pts = seg.pts
            for i in range(len(pts) - 1):
                ax, ay = pts[i]
                bx, by = pts[i + 1]
                if direction == "h":
                    lo, hi = (ax, bx) if ax <= bx else (bx, ax)
                    if lo - width <= pos <= hi + width:
                        if abs(bx - ax) < 1e-12:
                            found.append(ay)
                            found.append(by)
                        else:
                            t = (pos - ax) / (bx - ax)
                            t = 0.0 if t < 0 else (1.0 if t > 1 else t)
                            found.append(ay + (by - ay) * t)
                else:
                    lo, hi = (ay, by) if ay <= by else (by, ay)
                    if lo - width <= pos <= hi + width:
                        if abs(by - ay) < 1e-12:
                            found.append(ax)
                            found.append(bx)
                        else:
                            t = (pos - ay) / (by - ay)
                            t = 0.0 if t < 0 else (1.0 if t > 1 else t)
                            found.append(ax + (bx - ax) * t)
        return found

    span = max(proj.width, proj.height)
    hits = _scan(tol)
    # Widen progressively: a dimension taken to a rounded corner has no
    # geometry exactly at the bounding-box extreme.
    for mult in (0.02, 0.06, 0.15, 0.35):
        if hits:
            break
        hits = _scan(span * mult)
    if not hits:
        return fallback
    return min(hits) if side == "min" else max(hits)



def _straight_runs(proj, min_len):
    """Straight segments of the visible outline, as ((a, b), bearing_deg)."""
    runs = []
    for seg in proj.segs:
        if seg.kind != "visible" or seg.is_circle:
            continue
        pts = seg.pts
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            if L < min_len:
                continue
            runs.append(((a, b), math.degrees(math.atan2(b[1] - a[1],
                                                         b[0] - a[0])) % 360.0,
                         L))
    return runs


def _oblique_runs(proj, style):
    """Sloping outline edges long enough to carry an aligned dimension.

    An orthographic view states horizontal and vertical sizes directly, but a
    sloping edge has no h/v size true to the part: its length has to be
    dimensioned along its own direction. Real drawings do exactly that, and an
    aligned dimension is the commonest non-orthogonal dimension on a print.
    """
    span = max(proj.width, proj.height)
    if span <= 0:
        return []
    runs = _straight_runs(proj, span * min(style.angle_min_edge, 0.08))
    tol = float(getattr(style, "angle_ortho_tol", 2.0))
    out = []
    for (a, b), bearing, L in runs:
        off_axis = min(bearing % 90.0, 90.0 - (bearing % 90.0))
        if off_axis <= tol:
            continue                     # square with the view: h/v covers it
        key = (round(bearing % 180.0, 1), round(L, 2))
        if any(k == key for k, _ab, _L in out):
            continue                     # same edge on the far side
        out.append((key, (a, b), L))
    out.sort(key=lambda t: -t[2])
    return [(ab, L) for _k, ab, L in out]


def _find_angles(proj, style, txt):
    """Corners between two straight edges whose included angle is not 90 deg.

    Returns (vertex, bearing_a, bearing_b, included_deg) for each corner worth
    dimensioning. Orthogonal and near-straight junctions are skipped: the first
    is implied by the projection, the second is not a corner.
    """
    span = max(proj.width, proj.height)
    if span <= 0:
        return []
    runs = _straight_runs(proj, span * style.angle_min_edge)
    tol = span * 1e-3
    out = []
    seen = []
    for i in range(len(runs)):
        (a1, b1), _ang1, L1 = runs[i]
        for j in range(i + 1, len(runs)):
            (a2, b2), _ang2, L2 = runs[j]
            # find the shared endpoint
            vertex = None
            for p in (a1, b1):
                for q in (a2, b2):
                    if math.hypot(p[0] - q[0], p[1] - q[1]) <= tol:
                        vertex = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
                        break
                if vertex:
                    break
            if vertex is None:
                continue
            # bearings measured outward from the vertex along each edge
            far1 = b1 if math.hypot(a1[0] - vertex[0], a1[1] - vertex[1]) <= tol else a1
            far2 = b2 if math.hypot(a2[0] - vertex[0], a2[1] - vertex[1]) <= tol else a2
            d1 = math.degrees(math.atan2(far1[1] - vertex[1],
                                         far1[0] - vertex[0])) % 360.0
            d2 = math.degrees(math.atan2(far2[1] - vertex[1],
                                         far2[0] - vertex[0])) % 360.0
            inc = abs((d2 - d1 + 180.0) % 360.0 - 180.0)
            if inc > 180.0:
                inc = 360.0 - inc
            # skip right angles, straight joins and razor slivers
            if abs(inc - 90.0) < style.angle_ortho_tol:
                continue
            if inc < style.angle_min_deg or inc > 180.0 - style.angle_min_deg:
                continue
            if any(math.hypot(vertex[0] - v[0], vertex[1] - v[1]) < span * 0.02
                   for v in seen):
                continue
            seen.append(vertex)
            out.append((vertex, d1, d2, inc, min(L1, L2)))
    return out



def _seg_hits_box(a, b, box) -> bool:
    """True when segment a-b passes through an axis-aligned box."""
    x0, y0, x1, y1 = box
    for i in range(17):
        t = i / 16
        px = a[0] + (b[0] - a[0]) * t
        py = a[1] + (b[1] - a[1]) * t
        if x0 < px < x1 and y0 < py < y1:
            return True
    return False


def _note_box_hits_geometry(box, proj):
    """Return whether a note footprint crosses projected part ink.

    Occupancy is intentionally a thin-line grid: it is excellent for routing
    leaders, but it can miss a label crossing a curved edge between cells. A
    note footprint is not allowed to cover visible or hidden projected edges;
    the leader may touch the anchor, while the label itself must clear the
    part. This is especially important for small circular views carrying a
    thread, roughness, or datum note.
    """
    for seg in proj.segs + proj.circles:
        pts = list(getattr(seg, "pts", ()) or ())
        if len(pts) < 2:
            continue
        if getattr(seg, "full", False) and pts[-1] != pts[0]:
            pts.append(pts[0])
        if any(_seg_hits_box(a, b, box) for a, b in zip(pts, pts[1:])):
            return True
    return False


def _place_note(occ, ann, anchor, text, proj, style, txt, gap,
                size=None, centred=False):
    """Pick a tail for a plain note leader that avoids ink and other leaders.

    The thickness/fillet notes used a hardcoded corner tail, so their leader
    could be drawn straight across a hole callout. This reuses the same idea as
    the callout solver: try directions and lengths, score against the occupancy
    grid, and reject any tail whose leader crosses a committed segment.

    ``size`` overrides the reserved footprint with an explicit ``(w, h)`` in
    model units, and ``centred`` reserves it centred on the tail rather than
    sitting above it. Both exist for annotations that are not a single line of
    text: a GD&T frame is a row of boxed compartments about 1.6 text-heights
    tall and centred on its leader. Reserving a one-line box for it understated
    the height by ~60% and the placer then dropped frames on top of their
    neighbours -- three overlapping pairs on the NIST sheet, up to 23 mm2.
    """
    tw = len(text) * txt * style.char_w
    th = txt
    if size is not None:
        tw, th = size
    base = max(proj.width, proj.height)
    best, best_cost = None, None
    clear_best, clear_cost = None, None
    for ang in range(0, 360, 15):
        a = math.radians(ang)
        for mult in (0.18, 0.28, 0.40, 0.55):
            L = base * mult
            tail = (anchor[0] + math.cos(a) * L, anchor[1] + math.sin(a) * L)
            sgn = 1 if tail[0] >= anchor[0] else -1
            pad_t = txt * 0.35
            if sgn > 0:
                x0 = tail[0] + pad_t
            else:
                x0 = tail[0] - pad_t - tw
            y0 = tail[1] - th / 2 if centred else tail[1] + txt * 0.25
            box = (x0, y0, x0 + tw, y0 + th)
            cost = occ.rect_cost(*box) * 40 + occ.line_cost(anchor, tail) * 4
            geometry_hit = _note_box_hits_geometry(box, proj)
            view_overlap = not (box[2] <= proj.xmin or box[0] >= proj.xmax
                                or box[3] <= proj.ymin or box[1] >= proj.ymax)
            # A note is paper-space furniture, not a watermark on the part.
            # Keep its complete footprint outside the projected view as well as
            # clear of individual edges. This matters for circular views: the
            # bounding rectangle has blank interior, so an occupancy-only test
            # accepted labels over other rings even though no single edge was
            # sampled in the crossed cell.
            if geometry_hit or view_overlap:
                cost += 10000
            # never cross a leader or dimension already committed
            for sa, sb in ann.segments:
                if _seg_cross(anchor, tail, sa, sb):
                    cost += 800
                # ...and no committed line may pass through the text box
                if _seg_hits_box(sa, sb, box):
                    cost += 800
            # nor may the text sit on a dimension number, or on the text of a
            # leader callout already placed. Only dim_boxes were checked, so
            # "4X R6" could land squarely on top of "4X o6.6 THRU".
            _blockers = list(ann.dim_boxes)
            for _d in ann.dims:
                _b = getattr(_d, "box", None)
                if _b:
                    _blockers.append(_b)
            for bx in _blockers:
                # Penalty scales with how much of THIS box is buried, on top
                # of a flat charge for touching at all. A flat 800 made "just
                # clipping a corner" and "sitting squarely on top of" score
                # identically, so on a crowded view -- where every candidate
                # collides with something -- the solver had no reason to
                # prefer the near miss and picked a total overlap. That is
                # what buried a feature control frame under a hole callout on
                # the NIST sheet (98 mm2 of a 102 mm2 frame).
                ox = min(box[2], bx[2]) - max(box[0], bx[0])
                oy = min(box[3], bx[3]) - max(box[1], bx[1])
                if ox > 0 and oy > 0:
                    frac = (ox * oy) / max((box[2] - box[0])
                                           * (box[3] - box[1]), 1e-9)
                    cost += 400 + 1600 * min(1.0, frac)
            # Keep the note near the view. Reach is how far the text strays
            # outside the view box, and it is expensive: the sheet is sized
            # from the union of view + annotation, so a note flung clear of
            # the geometry inflates that union and can cost a whole scale
            # step -- every number on the sheet then gets smaller to make room
            # for one label.
            reach = (max(0.0, proj.xmin - min(box[0], box[2]))
                     + max(0.0, max(box[0], box[2]) - proj.xmax)
                     + max(0.0, proj.ymin - box[1])
                     + max(0.0, box[3] - proj.ymax))
            cost += (reach / max(base, 1e-9)) * 40 + mult * 6
            # NOTE: a hard cap here (cost += 1200 past
            # style.max_callout_reach, mirroring _place_callout) was tried and
            # REVERTED -- it did not move the note, because on a saturated
            # view every alternative already carries a larger
            # segment-crossing penalty.
            # a note sitting over the part is unreadable; push it outside
            #
            # NOTE: waiving this charge for slots over blank paper inside the
            # view was tried, to stop a thread note being flung 60 mm clear of
            # a saturated view, and REVERTED -- it changed nothing. Measuring
            # the full cost surface showed every candidate on that view
            # crosses at least two committed segments (a flat 1600+ penalty
            # that dwarfs both this charge and the reach term), so the choice
            # was never between inside and outside. The view is simply full.
            cxm, cym = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if (proj.xmin <= cxm <= proj.xmax
                    and proj.ymin <= cym <= proj.ymax):
                cost += 300
            if best_cost is None or cost < best_cost:
                best_cost, best = cost, (tail, box)
            if (not geometry_hit and not view_overlap
                    and (clear_cost is None or cost < clear_cost)):
                clear_cost, clear_best = cost, (tail, box)
    tail, box = clear_best or best
    occ.mark_line(anchor, tail)
    occ.mark_rect(*box)
    ann.add_seg(anchor, tail)
    # Register the note's text as a protected box. The occupancy grid alone is
    # too coarse to stop a thin leader threading through a word, and the
    # repair sweep only reroutes around ann.dim_boxes -- so a note placed here
    # was invisible to it and got crossed by a later leader.
    ann.dim_boxes.append(tuple(box))
    ann.grow(min(box[0], anchor[0]), min(box[1], anchor[1]),
             max(box[2], anchor[0]), max(box[3], anchor[1]))
    return tail


def _mark_arc(occ, ann, centre, radius, a0, a1, txt, seg_only=False):
    """Register a drawn arc so later annotation routes around it."""
    if radius <= 0:
        return
    delta = (a1 - a0 + 540.0) % 360.0 - 180.0
    n = max(8, int(abs(delta) / 6))
    pts = []
    for i in range(n + 1):
        a = math.radians(a0 + delta * i / n)
        pts.append((centre[0] + math.cos(a) * radius,
                    centre[1] + math.sin(a) * radius))
    for p, q in zip(pts, pts[1:]):
        occ.mark_line(p, q)
        ann.add_seg(p, q)
    if seg_only:
        return
    # the two extension legs from the vertex out past the arc
    for ang in (a0, a1):
        a = math.radians(ang)
        p = (centre[0] + math.cos(a) * radius * 0.25,
             centre[1] + math.sin(a) * radius * 0.25)
        q = (centre[0] + math.cos(a) * (radius + txt * 1.2),
             centre[1] + math.sin(a) * (radius + txt * 1.2))
        occ.mark_line(p, q)
        ann.add_seg(p, q)


def _emit_chain(ann, occ, proj, coords, direction, datum, cross, gap, txt,
                level, style, tolp):
    """Emit a chain of consecutive gaps on one dimension level.

    ``coords`` are absolute positions along the dimensioned axis. The chain
    runs datum -> c1 -> c2 -> ... so each value is the *increment* from the
    previous feature (9 then 6) rather than an absolute offset from the datum
    (9 then 15). Segments too short to hold their own text are merged into the
    next one, so a chain never emits cramped, unreadable numbers.
    """
    # Close the chain on the far edge of the view. Without this the segments
    # stop at the last feature and no longer sum to the overall dimension, so
    # the far end of the part is never located -- the reader cannot derive the
    # missing distance.
    far = (proj.xmax if direction == "h" else proj.ymax)
    pts = [datum] + [c for c in sorted(coords) if abs(c - datum) >= tolp]
    if abs(far - pts[-1]) >= tolp:
        pts.append(far)
    if len(pts) < 2:
        return level

    # Merge runs that are too tight to letter. A shifted-out label needs room
    # beside its segment, so require space for the widest plausible number
    # (5 glyphs) plus clearance -- otherwise neighbouring shifted labels
    # collide with each other.
    min_seg = max(txt * 3.4, 5 * txt * style.char_w * 1.15)
    merged = [pts[0]]
    for c in pts[1:]:
        if c - merged[-1] < min_seg and c != pts[-1]:
            continue                      # skip: folded into the next segment
        merged.append(c)
    # a final segment can still end up cramped after the merge; fold it back
    if len(merged) > 2 and merged[-1] - merged[-2] < min_seg:
        merged.pop(-2)
    if len(merged) < 2:
        return level

    # A single segment spanning the whole view just repeats the overall
    # dimension drawn one level below it -- emit nothing rather than a
    # duplicate number.
    extent = merged[-1] - merged[0]
    view_extent = proj.width if direction == "h" else proj.height
    if len(merged) == 2 and extent > view_extent * 0.985:
        return level

    level += 1
    off = gap * level
    # `off` is subtracted from the cross coordinate below, which walks the
    # dimension line away from a bottom/left datum. On the far side it has to
    # walk the other way or the chain lands on top of the view.
    far_side = (cross > (proj.ymin + proj.ymax) / 2 if direction == "h"
                else cross > (proj.xmin + proj.xmax) / 2)
    if far_side:
        off = -off
    flip = False
    boxes = []
    for a, b in zip(merged, merged[1:]):
        val = _sized(style, b - a, f"{direction}:{round(a, 3)}:{round(b, 3)}")
        tw = len(val) * txt * style.char_w
        # A toleranced string is much wider than the bare size. On a short
        # chain segment it has to be shifted out past the arrows, where it
        # lands on the neighbouring segment's number -- so on a segment too
        # short to letter it, the tolerance is dropped and the general note
        # governs that size instead, which is what a draughtsman does.
        if tw > (b - a) * 1.6:
            plain = fmt(b - a)
            if plain != val:
                val = plain
                tw = len(val) * txt * style.char_w
        fits = (b - a) > tw * 1.15
        if not fits:
            flip = not flip          # stagger neighbouring shifted labels
        # Exact rendered text box, mirroring render._linear precisely.
        # (An approximate mirror leaves a systematic offset and leaders get
        # routed straight through the numbers.)
        ar = style.arrow / _sc_of(style, txt)
        if direction == "h":
            ty = cross - off
            if fits:
                cxm = (a + b) / 2
                tb = (cxm - tw / 2, ty + txt * 0.5, cxm + tw / 2, ty + txt * 1.5)
            elif flip:
                x0 = max(a, b) + ar * 2.2
                tb = (x0, ty + txt * 0.5, x0 + tw, ty + txt * 1.5)
            else:
                x1 = min(a, b) - ar * 2.2
                tb = (x1 - tw, ty - txt * 1.4, x1, ty - txt * 0.4)
        else:
            tx = cross - off
            if fits:
                cym = (a + b) / 2
                tb = (tx - txt * 1.5, cym - tw / 2, tx - txt * 0.5, cym + tw / 2)
            elif flip:
                y0 = max(a, b) + ar * 2.2
                tb = (tx - txt * 1.5, y0, tx - txt * 0.5, y0 + tw)
            else:
                y1 = min(a, b) - ar * 2.2
                tb = (tx + txt * 0.4, y1 - tw, tx + txt * 1.4, y1)
        boxes.append(tb)
        if direction == "h":
            _ld = LinearDim((a, cross), (b, cross), offset=-off,
                            direction="h", text=val)
            _tolp = max(proj.width, proj.height) * 5e-3
            _ld.anchor1 = _anchor_at(proj, a, "h", cross, _tolp)
            _ld.anchor2 = _anchor_at(proj, b, "h", cross, _tolp)
            _ld.shift = 0 if fits else (1 if flip else -1)
            ann.dims.append(_ld)
            occ.mark_line((a, cross), (a, cross - off))
            occ.mark_line((b, cross), (b, cross - off))
            ann.add_seg((a, cross - off), (b, cross - off))     # dim line
            ann.add_seg((a, cross), (a, cross - off))           # witness
            ann.add_seg((b, cross), (b, cross - off))
            if fits:
                occ.mark_rect(a, cross - off - txt * 0.9, b, cross - off + txt * 1.6)
            elif flip:  # shifted right of the segment
                occ.mark_rect(a, cross - off - txt * 0.6, b, cross - off + txt * 0.6)
                occ.mark_rect(b, cross - off - txt * 0.4,
                              b + tw + txt * 1.2, cross - off + txt * 1.6)
            else:       # shifted left and one line lower, to clear neighbours
                occ.mark_rect(a, cross - off - txt * 0.6, b, cross - off + txt * 0.6)
                occ.mark_rect(a - tw - txt * 1.6, cross - off - txt * 2.6,
                              a + txt * 0.5, cross - off + txt * 0.4)
        else:
            _ld = LinearDim((cross, a), (cross, b), offset=-off,
                            direction="v", text=val)
            _tolp = max(proj.width, proj.height) * 5e-3
            _ld.anchor1 = _anchor_at(proj, a, "v", cross, _tolp)
            _ld.anchor2 = _anchor_at(proj, b, "v", cross, _tolp)
            _ld.shift = 0 if fits else (1 if flip else -1)
            ann.dims.append(_ld)
            occ.mark_line((cross, a), (cross - off, a))
            occ.mark_line((cross, b), (cross - off, b))
            ann.add_seg((cross - off, a), (cross - off, b))     # dim line
            ann.add_seg((cross, a), (cross - off, a))           # witness
            ann.add_seg((cross, b), (cross - off, b))
            if fits:
                occ.mark_rect(cross - off - txt * 1.6, a, cross - off + txt * 0.9, b)
            elif flip:
                occ.mark_rect(cross - off - txt * 0.6, a, cross - off + txt * 0.6, b)
                occ.mark_rect(cross - off - txt * 1.6, b,
                              cross - off + txt * 0.9, b + tw + txt * 1.2)
            else:   # text sits below the lower arrow, offset toward the view
                occ.mark_rect(cross - off - txt * 0.6, a, cross - off + txt * 0.6, b)
                occ.mark_rect(cross - off - txt * 0.4, a - tw - txt * 1.6,
                              cross - off + txt * 2.2, a)
    # reserve the exact text boxes last, so nothing is placed over them
    for tb in boxes:
        occ.mark_rect(tb[0] - txt * 0.3, tb[1] - txt * 0.3,
                      tb[2] + txt * 0.3, tb[3] + txt * 0.3)
    ann.dim_boxes.extend(boxes)
    return level


def _leader_segs(c, txt):
    """The two drawn segments of a callout leader: diagonal and text shelf."""
    if c.seg_start is None:
        return []
    return [(c.seg_start, c.seg_knee), (c.seg_knee, c.seg_shelf)]


def _record_leader(ann, c, txt):
    for sg in _leader_segs(c, txt):
        ann.add_seg(*sg)


def _seg_cross(p1, p2, p3, p4) -> bool:
    """True when segments p1-p2 and p3-p4 properly intersect."""
    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1, d2 = cr(p3, p4, p1), cr(p3, p4, p2)
    d3, d4 = cr(p1, p2, p3), cr(p1, p2, p4)
    return (((d1 > 1e-9 and d2 < -1e-9) or (d1 < -1e-9 and d2 > 1e-9)) and
            ((d3 > 1e-9 and d4 < -1e-9) or (d3 < -1e-9 and d4 > 1e-9)))


def _crosses_boxes(c, boxes) -> bool:
    """True when this callout's leader runs through a dimension text box."""
    if not boxes:
        return False
    a = math.radians(c.angle)
    st = (c.center[0] + math.cos(a) * c.radius,
          c.center[1] + math.sin(a) * c.radius)
    kn = (c.center[0] + math.cos(a) * (c.radius + c.leader_len),
          c.center[1] + math.sin(a) * (c.radius + c.leader_len))
    for x0, y0, x1, y1 in boxes:
        for i in range(33):
            t = i / 32
            px = st[0] + (kn[0] - st[0]) * t
            py = st[1] + (kn[1] - st[1]) * t
            if x0 < px < x1 and y0 < py < y1:
                return True
    return False


def _leader_crosses(target, others, txt) -> bool:
    """True when another callout's leader passes through target's text box."""
    if not target.box:
        return False
    x0, y0, x1, y1 = target.box
    for other, p0, r0, _note in others:
        if other is target or not other.box:
            continue
        a = math.radians(other.angle)
        st = (other.center[0] + math.cos(a) * other.radius,
              other.center[1] + math.sin(a) * other.radius)
        kn = (other.center[0] + math.cos(a) * (other.radius + other.leader_len),
              other.center[1] + math.sin(a) * (other.radius + other.leader_len))
        for seg in ((st, kn),):
            (ax, ay), (bx, by) = seg
            for i in range(25):
                t = i / 24
                px, py = ax + (bx - ax) * t, ay + (by - ay) * t
                if x0 + 0.2 < px < x1 - 0.2 and y0 + 0.2 < py < y1 - 0.2:
                    return True
    return False


class _LeaderCandidate:
    """One trial placement of a leader callout, in view coordinates.

    Pure geometry -- no scoring, no grid. Splitting this out of the search
    loop means the cost function below reads as a list of independent
    penalties instead of being interleaved with trigonometry.
    """

    __slots__ = ("ang", "L", "start", "knee", "shelf_end",
                 "tx0", "ty0", "tx1", "ty1", "mult")

    def __init__(self, p, r, ang, mult, base, tw, th, txt):
        # The r*2.5 floor exists so a leader clears a small hole's own circle.
        # On a large feature -- an outer diameter especially -- it dwarfs every
        # candidate: a 70mm radius pinned all eight lengths to 179mm, so the
        # search had no shorter option and the note trailed right across the
        # sheet. Clear the circle by a text-height margin instead, which is
        # what the floor is actually for.
        self.ang = ang
        self.mult = mult
        # ...and the view-relative term needs a PAPER cap for the same reason
        # in reverse. ``base * mult`` is a fraction of the view span, so on a
        # drawing at 15:1 a 12 mm disc has a 180 mm view and the longest
        # candidate leader was 122 mm of paper -- the note for a hole ended
        # up a third of a sheet away from it, and the pad it demanded is what
        # left 0000_00000007 with two small views at opposite ends of an A2.
        # A leader is a short line on a real drawing whatever the scale: cap
        # it at roughly 12 text heights, which is about 35 mm at the usual
        # 2.8 mm text.
        # The floor is a CLEARANCE, not a radius. ``start`` is already on the
        # circle (p + r), so the knee sits L beyond the edge -- adding r again
        # made the leader as long as the feature is wide: on the 12 mm disc of
        # 0000_00000007 the floor alone was 7 model units = 105 mm of paper,
        # and the note for the outside diameter sat a third of a sheet from
        # it. Two text heights is what clears the arrowhead.
        self.L = max(min(base * mult, txt * 12.5), txt * 2.0)
        a = math.radians(ang)
        self.start = (p[0] + math.cos(a) * r, p[1] + math.sin(a) * r)
        self.knee = (p[0] + math.cos(a) * (r + self.L),
                     p[1] + math.sin(a) * (r + self.L))
        sgn = 1 if math.cos(a) >= 0 else -1
        # text block grows upward from the knee (first line on top)
        self.tx0, self.tx1 = self.knee[0], self.knee[0] + sgn * tw
        self.ty0, self.ty1 = self.knee[1] - txt * 0.4, self.knee[1] + th
        self.shelf_end = (self.tx1, self.knee[1])

    @property
    def text_box(self):
        return (min(self.tx0, self.tx1), self.ty0,
                max(self.tx0, self.tx1), self.ty1)


def _leader_cost(c, occ, p, proj, style, txt, avoid, segs, budget=None):
    """Total penalty for a candidate. Lower is better.

    ``budget`` is the best cost found so far. The terms are evaluated cheapest
    first, and the function bails out as soon as the running total exceeds the
    budget -- the segment-crossing and forbidden-region tests are the two
    expensive ones (32 segments and ~4 boxes on the NIST part, for each of 288
    candidates) and are almost always reached only by candidates that are
    already losing. Returning early is safe because every remaining term is
    non-negative except a single -6 bonus, which is folded in up front.

    Each term is one reason a placement is undesirable; they are summed and
    the cheapest candidate wins. The weights are tuned against the sample
    parts and are deliberately far apart in magnitude, so a hard defect
    (crossing a leader, covering a dimension number) always outranks a soft
    preference (a slightly longer leader).
    """
    base = max(proj.width, proj.height)
    tx0, ty0, tx1, ty1 = c.tx0, c.ty0, c.tx1, c.ty1

    # Ink already on the sheet: the diagonal, the shelf the text sits on, and
    # the text block itself. Omitting the shelf let leaders be drawn straight
    # through text that was already placed.
    cost = (occ.line_cost(c.start, c.knee) * 3
            + occ.line_cost(c.knee, c.shelf_end) * 3
            + occ.rect_cost(tx0, ty0, tx1, ty1) * 10)

    # text sitting over the part is unreadable
    corners = [(tx0, ty0), (tx1, ty0), (tx0, ty1), (tx1, ty1),
               ((tx0 + tx1) / 2, (ty0 + ty1) / 2)]
    cost += 40 * sum(1 for cxp, cyp in corners
                     if proj.xmin <= cxp <= proj.xmax
                     and proj.ymin <= cyp <= proj.ymax)

    # keep the text within a sane band around the view
    if tx1 < proj.xmin - base * 0.9 or tx1 > proj.xmax + base * 0.9:
        cost += 25
    if ty1 > proj.ymax + base * 0.9 or ty0 < proj.ymin - base * 0.9:
        cost += 25

    # reward a knee that has already escaped the silhouette
    if not (proj.xmin <= c.knee[0] <= proj.xmax
            and proj.ymin <= c.knee[1] <= proj.ymax):
        cost -= 6

    # Distance outside the view box is what inflates the sheet, so charge for
    # it directly rather than only for leader length.
    reach = (max(0.0, proj.xmin - min(tx0, tx1))
             + max(0.0, max(tx0, tx1) - proj.xmax)
             + max(0.0, proj.ymin - ty0) + max(0.0, ty1 - proj.ymax))
    cost += (reach / max(base, 1e-9)) * 55
    if reach > base * style.max_callout_reach:
        cost += 120

    # Charge for the leader itself, not only for how far the text escapes the
    # view box: on a large circle a note could otherwise trail a 271mm leader
    # across the sheet at no cost, having stayed inside the (equally large)
    # view box the whole way.
    cost += (c.L / max(base, 1e-9)) * 18

    # Everything above is cheap arithmetic. Everything below walks lists.
    # Fold in the remaining always-applicable terms now so the budget test
    # below compares like with like, then bail if this candidate cannot win.
    vcx, vcy = (proj.xmin + proj.xmax) / 2, (proj.ymin + proj.ymax) / 2
    rx, ry = p[0] - vcx, p[1] - vcy
    rn = math.hypot(rx, ry)
    if rn > 1e-9:
        a = math.radians(c.ang)
        dot = (math.cos(a) * rx + math.sin(a) * ry) / rn
        cost += (1.0 - dot) * style.leader_outward_weight
    # mild preferences: shorter leaders, and the four diagonal directions
    cost += c.mult * 8 + (0 if c.ang in (45, 135, 225, 315) else 2)
    if budget is not None and cost >= budget:
        return cost          # the remaining terms only ever add

    # Hard-forbidden regions: a leader or label over a dimension number is
    # never acceptable, so price it out of contention.
    for bx0, by0, bx1, by1 in avoid:
        if not (tx1 < bx0 or tx0 > bx1 or ty1 < by0 or ty0 > by1):
            cost += 400
        for i in range(17):
            t = i / 16
            lx = c.start[0] + (c.knee[0] - c.start[0]) * t
            ly = c.start[1] + (c.knee[1] - c.start[1]) * t
            if bx0 < lx < bx1 and by0 < ly < by1:
                cost += 400
                break

    # A leader crossing another leader or a dimension line is the most visible
    # defect on the sheet, and the occupancy grid barely sees it (two thin
    # diagonals share almost no cells), so test the actual segments.
    if segs:
        nx = 0
        for sa, sb in segs:
            if (_seg_cross(c.start, c.knee, sa, sb)
                    or _seg_cross(c.knee, c.shelf_end, sa, sb)):
                nx += 1
                if nx >= 3:
                    break
        cost += nx * 900

    return cost


def _place_callout(occ: Occupancy, p, r, note, proj, style, txt,
                   avoid=(), segs=(), boxes=(), commit: bool = True,
                   cls: str = "bore") -> RadiusDim:
    """Search leader angles/lengths for the cheapest (least colliding) placement."""
    lines = note.split("\n")
    tw = max(len(l) for l in lines) * txt * style.char_w
    th = len(lines) * txt * 1.35
    base = max(proj.width, proj.height)

    best, best_cost = None, None
    for ang in style.leader_angles:
        for mult in style.leader_lengths:
            cand = _LeaderCandidate(p, r, ang, mult, base, tw, th, txt)
            cost = _leader_cost(cand, occ, p, proj, style, txt, avoid, segs,
                                budget=best_cost)
            if best_cost is None or cost < best_cost:
                best_cost, best = cost, cand

    c = best
    start, knee = c.start, c.knee
    # The nominal circle edge may not be drawn: partial arcs occur where a hole
    # is clipped by another feature, and an arrow aimed there points at nothing.
    snapped = _nearest_geometry(proj, start)
    if snapped is None:
        snap_gap = float("inf")        # nothing to point at in this view
    else:
        snap_gap = math.hypot(snapped[0] - start[0], snapped[1] - start[1])
        if snap_gap > txt * 0.25:
            start = snapped

    if commit:
        # Only stamp the grid when the caller keeps this placement. A retry
        # that gets rejected must not leave its footprint behind, or the grid
        # fills with phantom obstacles and later searches find nothing.
        occ.mark_line(start, knee)
        occ.mark_line(knee, c.shelf_end)
        pad = txt * 0.35
        bx0, by0, bx1, by1 = c.text_box
        occ.mark_rect(bx0 - pad, by0 - pad, bx1 + pad, by1 + pad)

    d = RadiusDim(center=p, radius=r, text=note, angle=c.ang, leader_len=c.L,
               cls=cls)
    d.box = (min(c.tx0, c.tx1, start[0]), min(c.ty0, c.ty1, start[1]),
             max(c.tx0, c.tx1, start[0]), max(c.ty0, c.ty1, start[1]))
    # exact drawn geometry, so collision bookkeeping matches the render
    d.seg_start, d.seg_knee, d.seg_shelf = start, knee, c.shelf_end
    d.tip = start
    # how far the arrowhead had to move to reach real geometry; a large value
    # means the feature is not drawn as an arc in this view
    d.snap_gap = snap_gap
    return d


def build_hole_table(info: PartInfo, views, symbols: bool = False, projs=None):
    """Hole table across one or more views.

    Tag letters are assigned per distinct feature type (shared across views, so
    the same hole type keeps one letter everywhere) and numbered per type.
    Returns (rows, tags_by_view) where rows are
    [TAG, VIEW, X, Y, DESCRIPTION] and tags_by_view maps view -> [(tag, pt, r)].
    """
    if isinstance(views, str):
        views = [views]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # collect every hole per view, keyed by feature type
    projs = projs or {}
    per_view = {v: _cluster_by_diameter(holes_in_view(info, v, projs.get(v)))
                for v in views}

    # one letter per distinct feature type, largest diameter first
    all_keys = []
    for v in views:
        for k in per_view[v]:
            if k not in all_keys:
                all_keys.append(k)
    all_keys.sort(key=lambda k: -k[0])
    letter_of = {k: letters[i % 26] for i, k in enumerate(all_keys)}

    rows = []
    tags_by_view = {v: [] for v in views}
    counter = {k: 0 for k in all_keys}
    for v in views:
        for key in sorted(per_view[v], key=lambda k: -k[0]):
            members = per_view[v][key]
            letter = letter_of[key]
            for h, p, r in sorted(members, key=lambda m: (-m[1][1], m[1][0])):
                counter[key] += 1
                tag = f"{letter}{counter[key]}"
                tags_by_view[v].append((tag, p, r))
                rows.append([tag, v.upper()[:3], fmt(p[0]), fmt(p[1]),
                             h.note(symbols=symbols).replace("\n", " ")])
    return rows, tags_by_view


def summarise_hole_groups(info: PartInfo, views, symbols: bool = False, projs=None):
    """Compact QTY/DESCRIPTION summary used when a full table is too long."""
    if isinstance(views, str):
        views = [views]
    agg = {}
    for v in views:
        for key, members in _cluster_by_diameter(
                holes_in_view(info, v, (projs or {}).get(v))).items():
            if key not in agg:
                agg[key] = [0, members[0][0]]
            agg[key][0] += len(members)
    rows = []
    for key in sorted(agg, key=lambda k: -k[0]):
        n, h = agg[key]
        rows.append([f"{n}X", h.note(symbols=symbols).replace("\n", " ")])
    return rows
