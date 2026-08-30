"""Hidden-line-removal projection of a 3D shape to 2D edge sets."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cadquery as cq
from cadquery.occ_impl.shapes import Shape
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape

# view name -> (view direction (towards viewer), x-axis of the sheet)
VIEW_DIRS: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "front":  ((0, -1, 0), (1, 0, 0)),
    "back":   ((0, 1, 0), (-1, 0, 0)),
    "top":    ((0, 0, 1), (1, 0, 0)),
    "bottom": ((0, 0, -1), (1, 0, 0)),
    "right":  ((1, 0, 0), (0, 1, 0)),
    "left":   ((-1, 0, 0), (0, -1, 0)),
    "iso":    ((1, -1, 0.9), (1, 1, 0)),
}


@dataclass
class Seg:
    """A polyline (flattened curve) in view coordinates."""
    pts: List[Tuple[float, float]]
    kind: str = "visible"          # visible | hidden
    is_circle: bool = False
    center: Optional[Tuple[float, float]] = None
    radius: float = 0.0
    full: bool = False


@dataclass
class Projection:
    name: str
    segs: List[Seg] = field(default_factory=list)
    circles: List[Seg] = field(default_factory=list)
    xmin: float = 0.0
    xmax: float = 0.0
    ymin: float = 0.0
    ymax: float = 0.0
    # shaded facets for the pictorial view (populated only for the isometric)
    facets: list = field(default_factory=list)
    #: Hatch segments in view coordinates. Empty on an ordinary view; a
    #: section's cut faces are hatched at 45 degrees, which is what tells a
    #: reader the material is cut rather than standing behind.
    hatch: list = field(default_factory=list)
    #: Letters identifying the section, e.g. "A-A".
    section_id: str = ""

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin


def _axes(name: str):
    d, xa = VIEW_DIRS[name]
    dv = cq.Vector(*d).normalized()
    xv = cq.Vector(*xa)
    xv = (xv - dv.multiply(xv.dot(dv))).normalized()
    yv = dv.cross(xv).normalized()
    return dv, xv, yv


def world_to_view(name: str, p) -> Tuple[float, float]:
    dv, xv, yv = _axes(name)
    v = cq.Vector(*p)
    return (v.dot(xv), v.dot(yv))


def _diagonal(shape) -> float:
    """Bounding-box diagonal, cached on the shape object itself.

    ``Shape.BoundingBox()`` runs BRepBndLib.AddOptimal, which costs ~0.3 s on a
    complex solid, and the projection needs it once per view.

    The cache lives in a WeakKeyDictionary rather than a plain dict keyed on
    ``id()``: CPython reuses the address of a freed object, so an id-keyed
    cache silently hands one part's diagonal to another. That produced a
    genuinely wrong tessellation tolerance -- and only intermittently, when the
    garbage collector happened to recycle an address.
    """
    try:
        cached = getattr(shape, "_autodraft_diag", None)
        if cached is not None:
            return cached
    except Exception:
        pass
    try:
        from .geometry import bounding_box
        d = bounding_box(shape).DiagonalLength
    except Exception:
        d = 1.0
    try:
        object.__setattr__(shape, "_autodraft_diag", d)
    except Exception:
        pass          # some Shape subclasses use __slots__; recompute instead
    return d


def project(shape: cq.Shape, name: str, hidden: bool = True,
            deflection: Optional[float] = None,
            rel_deflection: float = 2.5e-4) -> Projection:
    """Run OCCT HLR and return flattened 2D geometry in view coordinates.

    deflection is the maximum chord error when discretising a curved edge, in
    model units. It must scale with the part: a fixed 0.05 mm is 0.04% of a
    112 mm block but 5% of a 0.35 mm connector pin, which visibly flattens
    arcs and even clips the silhouette. Passing None derives it from the
    bounding-box diagonal via ``rel_deflection`` so fidelity is identical at
    every size.
    """
    dv, xv, yv = _axes(name)
    if deflection is None:
        deflection = max(_diagonal(shape) * rel_deflection, 1e-9)
    ax = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(dv.x, dv.y, dv.z), gp_Dir(xv.x, xv.y, xv.z))

    algo = HLRBRep_Algo()
    algo.Add(shape.wrapped)
    algo.Projector(HLRAlgo_Projector(ax))
    algo.Update()
    if hidden:
        algo.Hide()

    hlr = HLRBRep_HLRToShape(algo)
    out = Projection(name=name)

    def grab(fn, kind):
        try:
            s = fn()
        except Exception:
            return []
        if s is None:
            return []
        try:
            return Shape.cast(s).Edges()
        except Exception:
            return []

    sets = [(grab(hlr.VCompound, "visible"), "visible"),
            (grab(hlr.Rg1LineVCompound, "visible"), "visible"),
            (grab(hlr.OutLineVCompound, "visible"), "visible")]
    if hidden:
        sets += [(grab(hlr.HCompound, "hidden"), "hidden"),
                 (grab(hlr.OutLineHCompound, "hidden"), "hidden")]

    seen = set()
    for edges, kind in sets:
        for e in edges:
            seg = _edge_to_seg(e, kind, deflection)
            if seg is None:
                continue
            key = (kind, round(seg.pts[0][0], 4), round(seg.pts[0][1], 4),
                   round(seg.pts[-1][0], 4), round(seg.pts[-1][1], 4), len(seg.pts))
            if key in seen:
                continue
            seen.add(key)
            (out.circles if seg.is_circle else out.segs).append(seg)

    allpts = [p for s in out.segs + out.circles for p in s.pts]
    if allpts:
        out.xmin = min(p[0] for p in allpts)
        out.xmax = max(p[0] for p in allpts)
        out.ymin = min(p[1] for p in allpts)
        out.ymax = max(p[1] for p in allpts)
    return out


def _edge_to_seg(edge: cq.Edge, kind: str, deflection: float) -> Optional[Seg]:
    """HLR output lies in the projection plane: keep (x, y), drop z."""
    try:
        gt = edge.geomType()
    except Exception:
        gt = "OTHER"

    if gt == "CIRCLE":
        try:
            c = edge._geomAdaptor().Circle()
            loc = c.Location()
            r = c.Radius()
            p0, p1 = edge.startPoint(), edge.endPoint()
            full = p0.sub(p1).Length < 1e-7
            pts = _tess(edge, deflection)
            if not pts:
                return None
            return Seg(pts=pts, kind=kind, is_circle=True,
                       center=(loc.X(), loc.Y()), radius=r, full=full)
        except Exception:
            pass

    pts = _tess(edge, deflection)
    if not pts or len(pts) < 2:
        return None
    return Seg(pts=pts, kind=kind)


def _tess(edge: cq.Edge, deflection: float) -> List[Tuple[float, float]]:
    """Discretise via the curve adaptor.

    HLR output edges carry no triangulation, so ``Edge.tessellate`` yields
    nothing; sampling the parametric curve directly always works.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection

    try:
        gt = edge.geomType()
    except Exception:
        gt = "OTHER"
    if gt == "LINE":
        a, b = edge.startPoint(), edge.endPoint()
        return [(a.x, a.y), (b.x, b.y)]

    try:
        ad = BRepAdaptor_Curve(edge.wrapped)
        d = GCPnts_QuasiUniformDeflection(ad, deflection, ad.FirstParameter(),
                                          ad.LastParameter())
        if d.IsDone() and d.NbPoints() >= 2:
            return [(d.Value(i).X(), d.Value(i).Y()) for i in range(1, d.NbPoints() + 1)]
        # Uniform fallback when the deflection sampler fails. Pick the count
        # from the curve length so the chord error still tracks ``deflection``
        # instead of being fixed at 48 samples regardless of arc size.
        f, l = ad.FirstParameter(), ad.LastParameter()
        try:
            from OCP.GCPnts import GCPnts_AbscissaPoint
            length = GCPnts_AbscissaPoint.Length_s(ad)
        except Exception:
            length = 0.0
        n = 48
        if length > 0 and deflection > 0:
            # chord error ~ L^2 / (8 n^2 R); using L as a proxy for R is
            # conservative and keeps the sampling bounded
            n = int(min(512, max(24, math.ceil(length / max(deflection, 1e-12)))))
        return [(ad.Value(f + (l - f) * i / n).X(), ad.Value(f + (l - f) * i / n).Y())
                for i in range(n + 1)]
    except Exception:
        pass
    try:
        a, b = edge.startPoint(), edge.endPoint()
        return [(a.x, a.y), (b.x, b.y)]
    except Exception:
        return []


def choose_views(info, max_views: int = 3) -> List[str]:
    """Pick a sensible view set from the part's shape and hole axes."""
    # Which view faces a part's thin (plate) or most-featured axis.
    face_view = {"z": "top", "y": "front", "x": "right"}
    bb = info.bbox
    dims = {"x": bb.xlen, "y": bb.ylen, "z": bb.zlen}

    if info.is_plate:
        thin = min(dims, key=dims.get)
        side_view = {"z": "front", "y": "top", "x": "front"}[thin]
        return [face_view[thin], side_view][:max_views]

    # count hole axes to bias the primary view
    from collections import Counter
    axc = Counter()
    for h in info.holes:
        a = max(range(3), key=lambda i: abs(h.axis[i]))
        axc["xyz"[a]] += 1
    if axc:
        primary = face_view[axc.most_common(1)[0][0]]
    else:
        primary = "front"

    order = {"front": ["front", "top", "right"],
             "top": ["top", "front", "right"],
             "right": ["right", "front", "top"]}[primary]
    return order[:max_views]
