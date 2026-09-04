"""Model loading, orientation and feature recognition (B-rep based)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cadquery as cq
from cadquery.occ_impl.shapes import Shape
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.gp import gp_Pnt, gp_Vec

TOL = 1e-6


class _PointClassifier:
    """Reusable OCCT solid point-classifier.

    ``Shape.isInside`` constructs a fresh BRepClass3d_SolidClassifier on every
    call, which dominates feature recognition (~24x slower than reusing one).
    Building it once per solid turns a 40 s analysis into a few seconds.
    """

    __slots__ = ("_cl",)

    def __init__(self, solid):
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        self._cl = BRepClass3d_SolidClassifier(solid.wrapped)

    def inside(self, x, y, z, tol=1e-6) -> bool:
        from OCP.TopAbs import TopAbs_State
        self._cl.Perform(gp_Pnt(x, y, z), tol)
        return self._cl.State() == TopAbs_State.TopAbs_IN


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load(path: str) -> cq.Workplane:
    """Load STEP / IGES / BREP via OCCT."""
    p = str(path).lower()
    if p.endswith((".step", ".stp")):
        wp = cq.importers.importStep(str(path))
    elif p.endswith((".igs", ".iges")):
        from OCP.IGESControl import IGESControl_Reader
        from OCP.IFSelect import IFSelect_RetDone

        r = IGESControl_Reader()
        if r.ReadFile(str(path)) != IFSelect_RetDone:
            raise IOError(f"cannot read {path}")
        r.TransferRoots()
        wp = cq.Workplane("XY").newObject([Shape.cast(r.OneShape())])
    elif p.endswith((".brep", ".brp")):
        wp = cq.Workplane("XY").newObject([cq.Shape.importBrep(str(path))])
    else:
        raise ValueError(
            f"unsupported format: {path} "
            "(expected .step/.stp, .igs/.iges or .brep/.brp)")
    return wp


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
@dataclass
class Hole:
    """A cylindrical feature, possibly a stack (counterbore / countersink)."""

    x: float
    y: float
    z: float
    axis: Tuple[float, float, float]
    diameter: float
    depth: float
    through: bool
    counterbore: Optional[Tuple[float, float]] = None  # (dia, depth)
    countersink: Optional[Tuple[float, float]] = None  # (dia, angle deg)
    # Total angular sweep of the cylindrical wall, in degrees. A drilled hole
    # is a full 360; anything materially less is an open feature -- a scallop,
    # a slot end, a radiused notch -- whose "centre" lies outside the material
    # and often outside the silhouette entirely. Kept so consumers can tell a
    # hole from a fillet-like cut instead of guessing from the position.
    span_deg: float = 360.0

    @property
    def pos(self):
        return (self.x, self.y, self.z)

    @property
    def is_closed(self) -> bool:
        """True for a real (fully enclosed) hole.

        A tolerance of 1 degree covers STEP files that stitch a bore out of
        two half-faces with a hair of numerical slop.
        """
        return self.span_deg >= 359.0

    def note(self, symbols: bool = False) -> str:
        """Feature-control note.

        symbols=True uses ISO drafting glyphs (needs a font that has them);
        the default is a plain-text form that renders correctly everywhere.
        """
        dia, deep = ("\u2300", "\u2193") if symbols else ("\u2300", "DP ")
        s = f"{dia}{fmt(self.diameter)}"
        s += " THRU" if self.through else f" {deep}{fmt(self.depth)}"
        if self.counterbore:
            cb = "\u2334 " if symbols else "C'BORE "
            s += f"\n{cb}{dia}{fmt(self.counterbore[0])} {deep}{fmt(self.counterbore[1])}"
        if self.countersink:
            cs = "\u2335 " if symbols else "C'SINK "
            s += (f"\n{cs}{dia}{fmt(self.countersink[0])} X "
                  f"{fmt_raw(self.countersink[1])}\u00b0")
        return s


@dataclass
class Fillet:
    radius: float
    count: int


@dataclass
class Chamfer:
    """A conical break between two faces -- a real chamfer or a countersink.

    Recognised from conical faces. The two are distinguished by what the cone
    runs into: a countersink is the mouth of a bore and so has a non-zero
    small radius, whereas a drill point tapers to nothing.

    ``angle`` is the *included* angle (twice the cone's semi-angle), which is
    how a countersink is called out (``82 deg``, ``90 deg``). ``width`` is the
    slant leg of the break, which is how a plain chamfer is called out
    (``1 x 45 deg``).
    """

    angle: float                  # included angle, degrees
    width: float                  # axial depth of the break, model units
    radius_max: float             # outer radius, model units
    radius_min: float             # inner radius (0 for a drill point)
    count: int = 1
    countersink: bool = False     # True when it opens a bore
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    #: True when generated rather than recognised from a conical face.
    synthetic: bool = False

    def note(self, style=None) -> str:
        """Drafting callout for this break, in the office's wording.

        A 45-degree break is written three ways on real drawings and all
        three are here: ``2 X 45\u00b0`` (the long form, and the only one
        allowed for a break that is not 45 degrees), ``C2`` (ISO 3040 short
        form, 45 degrees only) and ``2 X 45\u00b0 CHAM``. A break at any
        other angle always gets the long form, because ``C2`` would state a
        45-degree break the part does not have.
        """
        if self.countersink:
            return (f"\u2300{fmt(self.radius_max * 2)} X "
                    f"{fmt_raw(self.angle)}\u00b0 CSK")
        half = self.angle / 2
        long_form = f"{fmt(self.width)} X {fmt_raw(half)}\u00b0"
        kind = getattr(style, "chamfer_style", "long") if style else "long"
        if abs(half - 45.0) > 0.5:
            return long_form
        if kind == "c":
            return f"C{fmt(self.width)}"
        if kind == "cham":
            return long_form + " CHAM"
        return long_form


@dataclass
class Body:
    """One disjoint solid inside the source file.

    A STEP file holding several unconnected solids is dimensioned across the
    overall envelope, which describes none of the actual parts. Keeping the
    per-body boxes lets the drawing size each block individually.
    """
    index: int
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float
    volume: float = 0.0

    @property
    def size(self):
        return (self.xmax - self.xmin, self.ymax - self.ymin,
                self.zmax - self.zmin)


@dataclass
class PartInfo:
    solid: cq.Workplane
    bbox: cq.occ_impl.shapes.BoundBox
    holes: List[Hole] = field(default_factory=list)
    fillets: List[Fillet] = field(default_factory=list)
    chamfers: List["Chamfer"] = field(default_factory=list)
    thickness: Optional[float] = None      # set when the part is plate-like
    is_plate: bool = False
    volume: float = 0.0
    area: float = 0.0
    n_solids: int = 1        # >1 means the STEP holds several disjoint bodies
    #: Factor the source model was scaled by to make it drawable (1.0 = as
    #: supplied). Sub-millimetre models are lifted onto a real size; the
    #: drawing states the scaled geometry and says so.
    model_scale: float = 1.0
    bodies: List["Body"] = field(default_factory=list)
    # Product and Manufacturing Information read from the source file
    # (autodraft.pmi.PMI). Empty unless the file actually carries AP242 PMI;
    # nothing here is ever inferred from geometry.
    pmi: object = None
    # Surface-finish specifications to state on the drawing
    # (autodraft.roughness.Roughness). See that module: no sample file carries
    # measured finish data, so these are assigned deterministically.
    roughness: List[object] = field(default_factory=list)

    @property
    def size(self):
        b = self.bbox
        return (b.xlen, b.ylen, b.zlen)


# --------------------------------------------------------------------------- #
# number formatting
# --------------------------------------------------------------------------- #
# Drawings state real, full-size dimensions: the drawing scale changes how big
# the part is drawn, never the number printed next to it.
#
# Millimetres are the floor. The sample parts really are ~1.5 mm across -- the
# STEP files declare SI_UNIT(.MILLI.,.METRE.), carry no scaling transform, and
# their raw CARTESIAN_POINT coordinates span 1.5 -- so the awkward values were
# genuine millimetres, not a unit-conversion error. Switching such parts to
# micrometres was rejected: it makes the printed numbers tidier but states the
# part in a unit no mechanical drawing uses, and it moved the problem rather
# than fixing it.
#
# Everything is therefore printed in millimetres (or a LARGER unit for big
# parts) and rounded to two decimals. Two decimals is the normal precision on
# a mechanical drawing; a value that needs more than that is being quoted to a
# tolerance the recogniser cannot actually justify.
# Millimetres only, at two decimals. cm and m were offered and removed: at two
# decimals they silently destroy small features (on a 1.2 m weldment in metres,
# four distinct fillet radii all collapsed to "R0.02"), and a drawing that
# cannot state its own features is worse than no drawing. Micrometres for
# sub-millimetre parts were tried and reverted too -- they state the part in a
# unit no mechanical drawing uses. mm is both the floor and the ceiling, so
# there is no unit to select and nothing to convert.
UNITS = "mm"

# Printed precision. Offices differ -- a fabrication print states 0.5 where a
# jig-boring drawing states 0.500 -- so a house style sets this once per
# drawing through set_decimals(). Several call sites format numbers and they
# must all agree, hence one module-level value rather than a parameter.
_DECIMALS = 2


def set_decimals(nd: int) -> int:
    """Set the decimal places every :func:`fmt` and :func:`to_display` uses."""
    global _DECIMALS
    _DECIMALS = max(0, min(4, int(nd)))
    return _DECIMALS


def to_display(v_mm: float) -> float:
    """A length in model millimetres, rounded to the printed precision.

    Rounding here means a recorded value and the text drawn beside it are the
    same number, so the two can never disagree.
    """
    return round(v_mm, _DECIMALS)


def fmt(v: float, nd: Optional[int] = None) -> str:
    """Format a length for the drawing.

    ``v`` is in model millimetres, rounded to two decimals with trailing
    zeros stripped.
    """
    return _trim(v, _DECIMALS if nd is None else nd)


def fmt_raw(v: float, nd: int = 1) -> str:
    """Format a plain number with no unit conversion (angles, counts)."""
    return _trim(v, nd)


def _trim(x: float, nd: int) -> str:
    """Round to `nd` decimals and strip trailing zeros.

    A value that rounds to zero prints "0", never "-0" -- but a value that
    merely rounds to a small negative keeps its sign. An earlier version
    tested the formatted string for "-0" and so turned -0.04 into "0" at one
    decimal, silently dropping the sign.
    """
    s = f"{x:.{nd}f}".rstrip("0").rstrip(".")
    if not s or s in ("-", "-0"):
        return "0"
    return s


def bounding_box(shape):
    """Bounding box of a shape, computed once and cached on the shape.

    ``Shape.BoundingBox()`` runs ``BRepBndLib.AddOptimal``, which takes ~0.75 s
    on the 66-hole NIST part and is NOT cached by cadquery -- three separate
    call sites (analysis, projection, shading) each paid it in full, so the
    same box was computed three times per drawing.

    The faster ``BRepBndLib.Add_s`` is not a substitute: it bounds the control
    polygon of every curve rather than the curve itself, reporting 159.48 mm
    for a part that is 158.00 mm. The scale and sheet size are chosen from
    this number, so the loose value is not acceptable -- the fix is to compute
    the exact one once.

    Cached as an attribute rather than in an id()-keyed dict: CPython reuses
    the address of a freed object, which previously handed one part's value to
    another and produced intermittently wrong output.
    """
    cached = getattr(shape, "_autodraft_bbox", None)
    if cached is not None:
        return cached
    bb = shape.BoundingBox()
    try:
        object.__setattr__(shape, "_autodraft_bbox", bb)
    except Exception:
        pass            # some wrapped shapes forbid attributes; just recompute
    return bb


def _surface(face) -> BRepAdaptor_Surface:
    return BRepAdaptor_Surface(face.wrapped)


def _cyl_data(face):
    """Return (axis_dir, axis_point, radius, zmin, zmax along axis, angular span)."""
    s = _surface(face)
    cyl = s.Cylinder()
    ax = cyl.Axis()
    d = ax.Direction()
    p = ax.Location()
    r = cyl.Radius()
    umin, umax, vmin, vmax = s.FirstUParameter(), s.LastUParameter(), s.FirstVParameter(), s.LastVParameter()
    return (
        (d.X(), d.Y(), d.Z()),
        (p.X(), p.Y(), p.Z()),
        r,
        vmin,
        vmax,
        umax - umin,
    )


def _is_internal(face, solid, clf=None) -> bool:
    """True when a cylinder/cone face is a hole rather than a boss.

    Samples several (u, v) points on the face; at each one it steps a short
    distance along both surface normals and asks the solid which side holds
    material. A single midpoint sample is unreliable because a feature may
    break into another pocket exactly there, so the result is a majority vote.
    """
    from OCP.BRepGProp import BRepGProp_Face

    try:
        s = _surface(face)
        st = s.GetType()
        if st == GeomAbs_SurfaceType.GeomAbs_Cylinder:
            ax = s.Cylinder().Axis()
            ref = s.Cylinder().Radius()
        elif st == GeomAbs_SurfaceType.GeomAbs_Cone:
            ax = s.Cone().Axis()
            ref = max(s.Cone().RefRadius(), 1e-3)
        else:
            return False

        u0, u1 = s.FirstUParameter(), s.LastUParameter()
        v0, v1 = s.FirstVParameter(), s.LastVParameter()
        bf = BRepGProp_Face(face.wrapped)
        d = gp_Vec(ax.Direction())
        c = ax.Location()
        cpnt = gp_Pnt(c.X(), c.Y(), c.Z())

        eps = max(min(0.08 * ref, 0.25), 1e-4)
        votes_hole = votes_boss = 0
        _clf = clf if clf is not None else _PointClassifier(solid)

        for fu in (0.5, 0.2, 0.8, 0.35, 0.65):
            for fv in (0.5, 0.25, 0.75):
                u = u0 + (u1 - u0) * fu
                v = v0 + (v1 - v0) * fv
                pnt, nrm = gp_Pnt(), gp_Vec()
                bf.Normal(u, v, pnt, nrm)
                if nrm.Magnitude() < TOL:
                    continue
                nrm.Normalize()

                def inside(sign):
                    return _clf.inside(pnt.X() + sign * nrm.X() * eps,
                                       pnt.Y() + sign * nrm.Y() * eps,
                                       pnt.Z() + sign * nrm.Z() * eps,
                                       eps * 1e-3)

                pos, neg = inside(1), inside(-1)
                if pos == neg:
                    continue        # both void (feature intersection) or both solid
                rad = gp_Vec(cpnt, pnt)
                rad = rad - d * rad.Dot(d)
                if rad.Magnitude() < TOL:
                    continue
                rad.Normalize()
                material_dir = 1 if pos else -1
                if nrm.Dot(rad) * material_dir > 0:
                    votes_hole += 1
                else:
                    votes_boss += 1
                if votes_hole + votes_boss >= 6:
                    break
            if votes_hole + votes_boss >= 6:
                break
        return votes_hole > votes_boss
    except Exception:
        return False


def _norm_axis(a):
    """Canonical axis direction (avoid +/- duplicates)."""
    x, y, z = a
    for c in (z, y, x):
        if abs(c) > TOL:
            if c < 0:
                return (-x, -y, -z)
            break
    return (x, y, z)


def _hole_stacks(cyls, cones, solid, bbox, clf):
    """Group coaxial cylinders into holes.

    Several cylindrical faces can belong to one feature: a
    counterbored hole is a narrow bore plus a wide one on the same
    axis, and a bore split by a seam is two half-faces. They are
    collected here so the caller sees one Hole per physical feature.
    """
    holes = []
    groups: List[List[dict]] = []
    for c in cyls:
        a = _norm_axis(c["axis"])
        placed = False
        for g in groups:
            ga = _norm_axis(g[0]["axis"])
            if abs(abs(a[0] * ga[0] + a[1] * ga[1] + a[2] * ga[2]) - 1) < 1e-4:
                # same direction -> check the axes are collinear
                dp = tuple(c["pt"][i] - g[0]["pt"][i] for i in range(3))
                cross = (
                    dp[1] * ga[2] - dp[2] * ga[1],
                    dp[2] * ga[0] - dp[0] * ga[2],
                    dp[0] * ga[1] - dp[1] * ga[0],
                )
                if math.sqrt(sum(v * v for v in cross)) < 1e-3:
                    g.append(c)
                    placed = True
                    break
        if not placed:
            groups.append([c])

    for g in groups:
        axis = _norm_axis(g[0]["axis"])
        # extent of each cylinder along the axis
        segs = []
        for c in g:
            base = c["pt"]
            sgn = 1 if _norm_axis(c["axis"]) == c["axis"] else -1
            lo, hi = sorted((sgn * c["vmin"], sgn * c["vmax"]))
            off = base[0] * axis[0] + base[1] * axis[1] + base[2] * axis[2]
            segs.append((off + lo, off + hi, c["r"], c))
        segs.sort()
        smallest = min(segs, key=lambda s: s[2])
        r = smallest[2]
        depth = sum(s[1] - s[0] for s in segs if abs(s[2] - r) < 1e-6)

        # How much of the bore wall actually exists, as an angle.
        #
        # A bore is often split into several faces. Two kinds of split must be
        # handled differently: a *seam* split cuts the wall into arcs at the
        # same axial station (two 180 deg faces = one closed bore), whereas an
        # *axial* split stacks faces along the axis (two 360 deg faces = still
        # one closed bore, not 720). So the faces at the governing radius are
        # clustered by overlapping axial interval, the spans are summed inside
        # each cluster, and the widest cluster wins. The result is clamped to
        # 360 because overlapping duplicate faces can otherwise overshoot.
        at_r = [s for s in segs if abs(s[2] - r) < 1e-6]
        clusters = []                    # [lo, hi, summed_span_rad]
        for lo, hi, _rr, c in sorted(at_r):
            for cl in clusters:
                if lo <= cl[1] + 1e-9:   # axially overlapping / touching
                    cl[1] = max(cl[1], hi)
                    cl[2] += c["span"]
                    break
            else:
                clusters.append([lo, hi, c["span"]])
        span_deg = min(360.0, math.degrees(
            max((cl[2] for cl in clusters), default=0.0)))

        # centre point projected onto the mid of the stack
        p = g[0]["pt"]
        mid = 0.5 * (segs[0][0] + segs[-1][1])
        base_off = p[0] * axis[0] + p[1] * axis[1] + p[2] * axis[2]
        cx = p[0] + axis[0] * (mid - base_off)
        cy = p[1] + axis[1] * (mid - base_off)
        cz = p[2] + axis[2] * (mid - base_off)

        through = _is_through(solid, (cx, cy, cz), axis, bbox, clf)

        cbore = None
        larger = [s for s in segs if s[2] > r + 1e-6]
        if larger:
            big = max(larger, key=lambda s: s[2])
            cbore = (2 * big[2], big[1] - big[0])

        csk = None
        for cone in cones:
            ca = _norm_axis(cone["axis"])
            if abs(abs(ca[0] * axis[0] + ca[1] * axis[1] + ca[2] * axis[2]) - 1) < 1e-4:
                lp = cone["loc"]
                dp = (lp.X() - cx, lp.Y() - cy, lp.Z() - cz)
                cross = (
                    dp[1] * axis[2] - dp[2] * axis[1],
                    dp[2] * axis[0] - dp[0] * axis[2],
                    dp[0] * axis[1] - dp[1] * axis[0],
                )
                if math.sqrt(sum(v * v for v in cross)) < 1e-3:
                    bb = cone["face"].BoundingBox()
                    dia = max(bb.xlen, bb.ylen, bb.zlen)
                    csk = (dia, round(math.degrees(cone["half"]) * 2, 1))
        holes.append(
            Hole(
                x=cx, y=cy, z=cz, axis=axis,
                diameter=2 * r, depth=depth, through=through,
                counterbore=cbore, countersink=csk,
                span_deg=round(span_deg, 3),
            )
        )
    return holes


def _detect_plate(info, bbox, volume):
    """Mark constant-thickness plate stock, which selects sheet views."""
    dims = sorted([bbox.xlen, bbox.ylen, bbox.zlen])
    if dims[0] > TOL and dims[0] * 4 < dims[1]:
        # thin in one direction; confirm it really is a flat plate
        env = dims[0] * dims[1] * dims[2]
        if env > 0 and volume / env > 0.30:
            info.is_plate = True
            info.thickness = dims[0]


#: Nice scale factors. A drawing office scales a model by a round number, and
#: a corpus in which every tiny part was scaled by the same factor would teach
#: the factor rather than the part.
_NICE_FACTORS = (2, 2.5, 4, 5, 8, 10, 20, 25, 40, 50, 100, 200, 250, 500,
                 1000, 2000, 2500, 5000)


def smallest_feature(info) -> float:
    """The smallest length this part would state.

    Every source of a printed number counts: the envelope, each body of a
    multi-body file (those get their own BODY n note), bores and fillet radii.
    Leaving the per-body extents out let a file whose envelope was 1.3 mm
    still print "0.07" for a thin body.
    """
    b = info.bbox
    vals = [d for d in (b.xlen, b.ylen, b.zlen) if d > 0]
    vals += [h.diameter for h in info.holes if h.diameter > 0]
    vals += [f.radius for f in info.fillets if f.radius > 0]
    for body in getattr(info, "bodies", []) or []:
        vals += [d for d in (body.xmax - body.xmin, body.ymax - body.ymin,
                             body.zmax - body.zmin) if d > 0]
    return min(vals, default=0.0)


def normalising_factor(smallest: float, key: str, floor: float = 1.0,
                       ceiling: float = 3.0) -> float:
    """Round factor that lifts ``smallest`` to at least ``floor`` millimetres.

    Sub-millimetre geometry cannot be drawn honestly at the two decimals a
    mechanical drawing states: a 0.035 mm feature prints "0.04", four distinct
    radii collapse onto one number, and a thread lookup that works in real
    millimetres has nothing to match. Scaling the *model* -- and saying so on
    the record -- is what a drawing office does with a micro part.

    The factor is not always the smallest one that clears the floor: several
    rungs land the part in [floor, ceiling], and which one is used is chosen
    deterministically from the part name, so the corpus carries a spread of
    scales instead of one.
    """
    if smallest <= 0 or smallest >= floor:
        return 1.0
    import hashlib
    ok = [f for f in _NICE_FACTORS if floor <= smallest * f <= ceiling]
    if not ok:
        # nothing lands in the window: take the first rung above the floor
        ok = [f for f in _NICE_FACTORS if smallest * f >= floor][:1] \
            or [_NICE_FACTORS[-1]]
    n = int.from_bytes(
        hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big")
    return float(ok[n % len(ok)])


def scale_shape(wp: cq.Workplane, factor: float) -> cq.Workplane:
    """Uniformly scale a loaded model about the origin."""
    if factor == 1.0:
        return wp
    from OCP.gp import gp_Trsf
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    t = gp_Trsf()
    t.SetScaleFactor(factor)
    out = []
    for obj in wp.vals():
        built = BRepBuilderAPI_Transform(obj.wrapped, t, True)
        out.append(Shape.cast(built.Shape()))
    return cq.Workplane("XY").newObject(out)


def analyse(wp: cq.Workplane) -> PartInfo:
    solid = wp.val()
    bbox = bounding_box(solid)
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid.wrapped, props)
    volume = props.Mass()

    try:
        _nsol = len(solid.Solids())
    except Exception:
        _nsol = 1
    info = PartInfo(solid=wp, bbox=bbox, volume=volume, area=solid.Area(),
                    n_solids=max(1, _nsol))
    if _nsol > 1:
        try:
            for _i, _so in enumerate(solid.Solids()):
                _b = _so.BoundingBox()
                info.bodies.append(Body(
                    index=_i, xmin=_b.xmin, ymin=_b.ymin, zmin=_b.zmin,
                    xmax=_b.xmax, ymax=_b.ymax, zmax=_b.zmax,
                    volume=_so.Volume()))
            info.bodies.sort(key=lambda b: -b.volume)
            for _i, _b in enumerate(info.bodies):
                _b.index = _i
        except Exception:
            info.bodies = []
    clf = _PointClassifier(solid)      # built once, reused for every query

    cyls, cones, fillet_r = [], [], {}
    for f in wp.faces().vals():
        st = _surface(f).GetType()
        if st == GeomAbs_SurfaceType.GeomAbs_Cylinder:
            axis, pt, r, vmin, vmax, span = _cyl_data(f)
            internal = _is_internal(f, solid, clf)
            rec = dict(face=f, axis=axis, pt=pt, r=r, vmin=vmin, vmax=vmax, span=span, internal=internal)
            if internal:
                cyls.append(rec)
                # A partial internal cylinder is a CONCAVE corner radius, not
                # a bore: think of the rounded inside corner of an L-shaped
                # plate. It is a fillet to everyone who reads the drawing, and
                # it is the radius that caps the cutter, so it belongs in the
                # fillet list too. Only closed bores are excluded -- those are
                # drilled holes and are described by their diameter instead.
                #
                # 0000_00000061's only radius is exactly this: a 90 deg R0.075
                # internal arc. It was filed as an open-arc "hole", so the
                # drawing showed no radius at all.
                if span < 2 * math.pi - 1e-3:
                    fillet_r[round(r, 4)] = fillet_r.get(round(r, 4), 0) + 1
            elif span < 2 * math.pi - 1e-3:
                fillet_r[round(r, 4)] = fillet_r.get(round(r, 4), 0) + 1
        elif st == GeomAbs_SurfaceType.GeomAbs_Cone and _is_internal(f, solid, clf):
            s = _surface(f)
            cone = s.Cone()
            ax = cone.Axis()
            d = ax.Direction()
            cones.append(
                dict(
                    axis=(d.X(), d.Y(), d.Z()),
                    apex=cone.Apex(),
                    half=cone.SemiAngle(),
                    rref=cone.RefRadius(),
                    loc=ax.Location(),
                    face=f,
                )
            )

    info.fillets = [Fillet(radius=r, count=c) for r, c in sorted(fillet_r.items())]
    info.chamfers = _chamfers(cones)

    # ---- group coaxial cylinders into hole stacks --------------------------
    info.holes = _hole_stacks(cyls, cones, solid, bbox, clf)

    # ---- plate / sheet detection ------------------------------------------
    _detect_plate(info, bbox, volume)
    return info


def _chamfers(cones) -> List[Chamfer]:
    """Group conical faces into distinct chamfers / countersinks.

    Drill points are excluded. A twist drill leaves a cone that tapers to a
    point, so its inner radius is ~0; a countersink or a chamfer always opens
    onto a bore or an edge and keeps a finite inner radius. On the NIST part
    this rejects all 8 cones -- they are 118 deg drill points (59 deg
    semi-angle, inner radius 0.000) at the bottom of blind holes, which no
    drawing dimensions. On flange.step it keeps all 6, which are genuine 82
    deg countersinks (inner radius 4.500, outer 9.000).

    Identical breaks are collapsed into one entry with a count, the same way
    fillet radii are, so six countersinks on a bolt circle read as "6X".
    """
    out: Dict[tuple, Chamfer] = {}
    for c in cones:
        face = c["face"]
        try:
            s = _surface(face)
            half = abs(c["half"])
            if half <= 1e-6 or half >= math.pi / 2 - 1e-6:
                continue
            v0, v1 = s.FirstVParameter(), s.LastVParameter()
            r0 = abs(c["rref"] + v0 * math.sin(c["half"]))
            r1 = abs(c["rref"] + v1 * math.sin(c["half"]))
            rmin, rmax = min(r0, r1), max(r0, r1)
            axial = abs((v1 - v0) * math.cos(half))
        except Exception:
            continue
        if rmax <= 1e-9 or axial <= 1e-9:
            continue
        # Drill point: tapers to a point rather than opening a feature.
        if rmin < rmax * 0.05:
            continue
        key = (round(math.degrees(half) * 2, 1), round(axial, 4),
               round(rmax, 4), round(rmin, 4))
        hit = out.get(key)
        if hit is not None:
            hit.count += 1
            continue
        loc = c["loc"]
        ax = c["axis"]
        out[key] = Chamfer(
            angle=round(math.degrees(half) * 2, 4), width=round(axial, 6),
            radius_max=round(rmax, 6), radius_min=round(rmin, 6),
            count=1, countersink=rmin > 1e-6,
            position=(loc.X(), loc.Y(), loc.Z()),
            axis=(ax[0], ax[1], ax[2]))
    return sorted(out.values(), key=lambda x: (-x.count, -x.radius_max))


def _is_through(solid, center, axis, bbox, clf=None) -> bool:
    """A hole is 'through' if a ray along the axis leaves the solid on both sides."""
    L = bbox.DiagonalLength
    eps = max(L * 1e-4, 1e-4)
    _clf = clf if clf is not None else _PointClassifier(solid)
    for sgn in (1, -1):
        q = [center[i] + sgn * axis[i] * L for i in range(3)]
        if _clf.inside(q[0], q[1], q[2], eps):
            return False
    # additionally ensure the axis is not blocked: sample along it
    n = 40
    inside_hits = 0
    for i in range(n + 1):
        t = -L / 2 + L * i / n
        q = [center[j] + axis[j] * t for j in range(3)]
        if _clf.inside(q[0], q[1], q[2], eps):
            inside_hits += 1
            break                 # one hit already means "not through"
    return inside_hits == 0
