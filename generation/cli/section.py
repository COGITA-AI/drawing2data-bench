"""Section views: cut the solid, project the remainder, hatch the cut faces.

A section is the commonest thing a real mechanical drawing has and a generated
one does not. Internal features -- a counterbore, a blind pocket, a stepped
bore -- read as a thicket of hidden lines in an orthographic view and as solid
material in a section, which is exactly why draughtsmen draw them.

What happens here:

1. the solid is cut by a half-space through the chosen plane, so the near half
   is removed and the reader looks at the cut;
2. the remainder is projected the normal way (:mod:`autodraft.projection`);
3. the faces that lie *in* the cutting plane are found and projected as
   polygons, and those are hatched at 45 degrees -- hatching is what tells a
   reader "this is cut material" rather than "this is a wall behind".

The hatch is clipped against the real cut outline (even-odd scanline), so a
bore through the cut face leaves a hole in the hatching, as it must.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cadquery as cq

from .projection import Projection, _axes, project


def _cut_halfspace(shape: cq.Shape, view: str, at: float = 0.5):
    """Remove the half of the solid between the viewer and the cutting plane.

    The cutting plane is PARALLEL to the picture plane -- its normal is the
    view direction. Cutting on the view's horizontal axis instead leaves the
    exposed face perpendicular to the paper, where it projects to a line and
    there is nothing to hatch.

    ``at`` is where the plane sits through the part's depth, as a fraction
    (0.5 = through the middle, which is what a symmetric part wants).
    """
    dv, _xv, _yv = _axes(view)
    bb = shape.BoundingBox()
    diag = bb.DiagonalLength or 1.0
    centre = cq.Vector((bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2,
                       (bb.zmin + bb.zmax) / 2)
    depths = [cq.Vector(*p).dot(dv) for p in _corners(bb)]
    cut_at = min(depths) + (max(depths) - min(depths)) * at

    try:
        rest = shape.cut(_halfspace_box(centre, dv, cut_at, diag))
    except Exception:
        return None, None
    if rest is None or not rest.Solids():
        return None, None
    return rest, cut_at


def _corners(bb):
    return [(x, y, z) for x in (bb.xmin, bb.xmax)
            for y in (bb.ymin, bb.ymax) for z in (bb.zmin, bb.zmax)]


def _halfspace_box(centre: cq.Vector, axis: cq.Vector, cut_at: float,
                   diag: float) -> cq.Shape:
    """A cube big enough to swallow the near half, one face on the plane."""
    size = diag * 3.0
    on_plane = axis.multiply(cut_at)
    perp = centre - axis.multiply(centre.dot(axis))
    box_centre = perp + on_plane - axis.multiply(size / 2.0)
    box = cq.Workplane("XY").box(size, size, size).val()
    if not _axis_is_cardinal(axis):
        # a non-cardinal view direction (the isometric): turn the cube onto it
        src = cq.Vector(1, 0, 0)
        cross = src.cross(axis)
        if cross.Length > 1e-9:
            ang = math.degrees(math.acos(max(-1.0, min(1.0, src.dot(axis)))))
            box = box.rotate((0, 0, 0), (cross.x, cross.y, cross.z), ang)
    return box.translate((box_centre.x, box_centre.y, box_centre.z))


def _axis_is_cardinal(axis: cq.Vector) -> bool:
    return sum(1 for c in (axis.x, axis.y, axis.z) if abs(c) > 1e-9) == 1


def _cut_face_polygons(rest: cq.Shape, view: str, cut_at: float,
                       tol: float) -> List[List[Tuple[float, float]]]:
    """Outlines, in view coordinates, of the faces lying in the cutting plane.

    These are the faces the hatching goes on. Inner wires come back too, as
    separate polygons: the even-odd fill rule then leaves a bore unhatched,
    which is what a section looks like.
    """
    dv, xv, yv = _axes(view)
    cut_normal = dv
    polys = []
    for face in rest.Faces():
        try:
            centre = face.Center()
            normal = face.normalAt()
        except Exception:
            continue
        if abs(abs(normal.dot(cut_normal)) - 1.0) > 1e-3:
            continue                      # not parallel to the cutting plane
        if abs(centre.dot(cut_normal) - cut_at) > tol:
            continue                      # parallel, but not ON it
        for wire in face.Wires():
            pts = _wire_polygon(wire, xv, yv)
            if len(pts) >= 3:
                polys.append(pts)
    return polys


def _wire_polygon(wire, xv, yv) -> List[Tuple[float, float]]:
    """One closed, correctly ORDERED loop of view-space points for a wire.

    ``wire.Edges()`` hands back the edges in no guaranteed order and each one
    carries its own parameterisation direction, so sampling them straight
    through (``for edge: for i in range(12): edge.positionAt(i/12)``) built a
    point list that jumped between opposite corners of the face. The polygon
    that came out self-intersected, and the even-odd scanline in
    ``hatch_lines`` then filled triangular wedges and left whole rectangles
    bare -- exactly what 0000_00000126 showed.

    Fix: sample each edge into its own polyline, then stitch the polylines
    end-to-end, reversing any that is traversed backwards. A straight edge
    only needs its two ends; a curve is sampled finely enough that a bore
    still reads as a circle at drawing scale.
    """
    segs: List[List[Tuple[float, float]]] = []
    for edge in wire.Edges():
        try:
            straight = edge.geomType() == "LINE"
        except Exception:
            straight = False
        n = 1 if straight else 24
        try:
            pts = [edge.positionAt(i / n) for i in range(n + 1)]
        except Exception:
            continue
        seg = [(q.dot(xv), q.dot(yv)) for q in pts]
        if len(seg) >= 2:
            segs.append(seg)
    if not segs:
        return []

    # Stitch. Tolerance is relative to the wire's own extent so it works on a
    # 2 mm micro-part and a 2 m weldment alike.
    xs = [p[0] for s in segs for p in s]
    ys = [p[1] for s in segs for p in s]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    tol = span * 1e-4

    chain = list(segs.pop(0))
    while segs:
        end = chain[-1]
        best_i, best_rev, best_d = None, False, None
        for i, s in enumerate(segs):
            for rev, q in ((False, s[0]), (True, s[-1])):
                d = (q[0] - end[0]) ** 2 + (q[1] - end[1]) ** 2
                if best_d is None or d < best_d:
                    best_i, best_rev, best_d = i, rev, d
        s = segs.pop(best_i)
        if best_rev:
            s = list(reversed(s))
        # Drop the duplicated joint point; if the nearest end is nowhere near
        # (a wire this face-finder should not have been given) keep it, so the
        # loop still closes rather than silently skipping geometry.
        chain.extend(s[1:] if best_d is not None and best_d <= tol * tol
                     else s)
    # close
    if (chain[0][0] - chain[-1][0]) ** 2 + \
       (chain[0][1] - chain[-1][1]) ** 2 <= tol * tol:
        chain.pop()
    return chain


def hatch_lines(polys, spacing: float, angle_deg: float = 45.0):
    """45-degree hatch clipped to ``polys`` by the even-odd rule."""
    if not polys:
        return []
    ang = math.radians(angle_deg)
    ux, uy = math.cos(ang), math.sin(ang)
    nx, ny = -uy, ux                       # hatch line normal
    xs = [p[0] for poly in polys for p in poly]
    ys = [p[1] for poly in polys for p in poly]
    if not xs:
        return []
    corners = [(min(xs), min(ys)), (min(xs), max(ys)),
               (max(xs), min(ys)), (max(xs), max(ys))]
    ds = [c[0] * nx + c[1] * ny for c in corners]
    d0, d1 = min(ds), max(ds)
    edges = []
    for poly in polys:
        for a, b in zip(poly, poly[1:] + poly[:1]):
            edges.append((a, b))
    out = []
    d = d0 + spacing * 0.5
    while d < d1:
        hits = []
        for a, b in edges:
            da = a[0] * nx + a[1] * ny - d
            db = b[0] * nx + b[1] * ny - d
            if (da > 0) == (db > 0):
                continue
            t = da / (da - db) if (da - db) else 0.0
            px = a[0] + (b[0] - a[0]) * t
            py = a[1] + (b[1] - a[1]) * t
            hits.append(px * ux + py * uy)
        hits.sort()
        for i in range(0, len(hits) - 1, 2):
            s0, s1 = hits[i], hits[i + 1]
            if s1 - s0 < spacing * 0.05:
                continue
            out.append(((d * nx + s0 * ux, d * ny + s0 * uy),
                        (d * nx + s1 * ux, d * ny + s1 * uy)))
        d += spacing
    return out


def section(shape: cq.Shape, view: str, at: float = 0.5,
            hidden: bool = False) -> Optional[Projection]:
    """Projection of ``shape`` cut through ``at``, carrying its hatch lines.

    Returns None when the cut produces nothing useful -- a part with no
    internal features gains nothing from a section, and a failed boolean must
    never take the drawing down with it.
    """
    rest, cut_at = _cut_halfspace(shape, view, at)
    if rest is None:
        return None
    try:
        proj = project(rest, view, hidden=hidden)
    except Exception:
        return None
    if proj.width <= 0 or proj.height <= 0:
        return None
    diag = shape.BoundingBox().DiagonalLength or 1.0
    polys = _cut_face_polygons(rest, view, cut_at, diag * 1e-4)
    proj.hatch = hatch_lines(polys, spacing=max(diag * 0.012, 1e-6))
    return proj
