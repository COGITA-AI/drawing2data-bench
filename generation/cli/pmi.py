"""Product and Manufacturing Information read out of a STEP AP242 file.

This module extracts things a drawing *states* but a B-rep does not contain:
geometric tolerances (GD&T feature control frames), datum features, and
thread specifications. Nothing here is inferred from geometry -- if the file
does not carry the information, the reader returns nothing. That matters
because the output trains an extractor, so a plausible guess is worse than an
absent label.

Two readers are used together, because neither alone is sufficient:

* **OCCT's XCAF reader** (``STEPCAFControl_Reader`` with ``SetGDTMode``)
  resolves each tolerance to its *referenced faces*, which is what lets a
  frame be anchored to a point on the model and therefore placed on a view.
  Doing that from the raw file would mean re-implementing STEP's
  shape_aspect -> advanced_face resolution.
* **A raw text pass over the STEP entities** recovers the tolerance
  *magnitude* and the *thread designations*. This is not redundancy for its
  own sake: OCCT 7.9 returns ``0.0`` from ``GetValue()`` for every tolerance
  in the NIST AP242 sample, because the magnitude sits on a
  ``LENGTH_MEASURE_WITH_UNIT`` that its ``XCAFDimTolObjects`` translator does
  not follow. The real values (0.2, 0.65, 1.2, 1.5 ...) are plainly present
  in the file. The two sources are joined on the feature control frame's
  name, e.g. ``'Feature Control Frame (94)'``, which both expose.

Threads are read from ``DESCRIPTIVE_REPRESENTATION_ITEM`` PMI strings such as
``'DIM\\wM3 x 0.5-6g'``. No thread is ever inferred from a hole diameter: a
tap-drill-sized hole is not evidence of a thread, and labelling it as one
would put a fabricated specification in the training data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

#: ASME Y14.5 / ISO 1101 characteristic symbols, keyed by the OCCT enum tail.
#: The value is the drafting glyph name used by :mod:`autodraft.gdt_symbols`
#: to draw it as vector geometry. Most of these characters exist in Unicode
#: but NOT in DejaVu Sans (the renderer's font), so they are drawn rather than
#: typeset -- see gdt_symbols for the measurement behind that decision.
CHARACTERISTICS = {
    "Position": "position",
    "Concentricity": "concentricity",
    "Symmetry": "symmetry",
    "Parallelism": "parallelism",
    "Perpendicularity": "perpendicularity",
    "Angularity": "angularity",
    "Flatness": "flatness",
    "Straightness": "straightness",
    "Roundness": "circularity",
    "Circularity": "circularity",
    "Cylindricity": "cylindricity",
    "ProfileOfLine": "profile_line",
    "ProfileOfSurface": "profile_surface",
    "CircularRunout": "circular_runout",
    "TotalRunout": "total_runout",
}

#: Material-condition modifiers, as the letter drawn inside a circle.
MODIFIERS = {
    "MaximumMaterialRequirement": "M",
    "LeastMaterialRequirement": "L",
    "RegardlessOfFeatureSize": "S",
}


@dataclass
class GeomTolerance:
    """One feature control frame read from the file."""

    characteristic: str                     # 'position', 'flatness', ...
    value: float                            # tolerance zone magnitude, mm
    datums: Tuple[str, ...] = ()            # ('A', 'B', 'C')
    diametral: bool = False                 # zone is a diameter, prefix with o
    modifier: Optional[str] = None          # 'M' / 'L' / 'S'
    anchor: Optional[Tuple[float, float, float]] = None   # model-space point
    name: str = ""                          # 'Feature Control Frame (94)'
    #: True when this frame was generated rather than read from the file.
    #: Real PMI from a STEP AP242 file is always False. See
    #: autodraft/synth_pmi.py for what "generated" means and why.
    synthetic: bool = False

    def text(self) -> str:
        """Plain-text rendering, used for the label string and JSON.

        The drawn frame uses vector symbols; this is the machine-readable
        equivalent, e.g. ``POS |o0.14|A|B|C``.
        """
        from .geometry import fmt
        head = self.characteristic.upper()
        val = ("\u2300" if self.diametral else "") + fmt(self.value)
        if self.modifier:
            val += f"({self.modifier})"
        parts = [head, val, *self.datums]
        return "|".join(parts)


@dataclass
class DatumFeature:
    """A datum feature symbol (the boxed letter A, B, C ...)."""

    letter: str
    anchor: Optional[Tuple[float, float, float]] = None
    synthetic: bool = False


@dataclass
class Thread:
    """A thread specification stated by the file, e.g. ``M3 x 0.5-6g``."""

    designation: str                 # 'M3 x 0.5-6g'
    count: int = 1                   # leading '2X' multiplier
    nominal: Optional[float] = None  # major diameter in mm, parsed from 'M3'
    anchor: Optional[Tuple[float, float, float]] = None
    #: Axis of the bore this thread sits in, model coordinates. Needed to
    #: decide which views the hole reads as a circle in, so the callout is
    #: not placed on a view where the bore is edge-on.
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    synthetic: bool = False

    def text(self) -> str:
        return (f"{self.count}X {self.designation}" if self.count > 1
                else self.designation)


@dataclass
class PMI:
    """Everything read from one file. Empty when the file carries no PMI."""

    tolerances: List[GeomTolerance] = field(default_factory=list)
    datums: List[DatumFeature] = field(default_factory=list)
    threads: List[Thread] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.tolerances or self.datums or self.threads)


# --------------------------------------------------------------------------- #
# raw STEP pass: tolerance magnitudes and thread designations
# --------------------------------------------------------------------------- #
def _entities(text: str) -> Dict[str, str]:
    """``{'441': 'GEOMETRIC_TOLERANCE(...)'}`` with all whitespace stripped.

    STEP wraps entities across lines at arbitrary points -- mid-token and even
    mid-string-literal -- so every regex here runs against a whitespace-free
    copy of the file. Matching line by line silently misses roughly a third of
    the entities in the NIST sample.
    """
    flat = re.sub(r"\s+", "", text)
    return dict(re.findall(r"#(\d+)=(.*?);", flat, re.S))


def _key(name: str) -> str:
    """Normalise a frame name for joining the two readers.

    The raw pass matches against a whitespace-free copy of the file, so it
    sees ``'FeatureControlFrame(94)'`` where OCCT reports
    ``'Feature Control Frame (94)'``. Both sides are reduced to the same key.
    """
    return re.sub(r"\s+", "", name or "")


def _tolerance_values(text: str) -> Dict[str, List[float]]:
    """Feature-control-frame name -> its tolerance magnitudes, from raw STEP.

    Keys are normalised by :func:`_key`.

    A composite frame states two magnitudes under one name (an upper and a
    lower segment), so the value is a list and callers zip it against the
    tolerances OCCT reports for that name.
    """
    flat = re.sub(r"\s+", "", text)
    ents = _entities(text)
    out: Dict[str, List[float]] = {}
    # A frame is written one of two ways and both occur in the same file:
    #   * the complex/AND form,  #97=(GEOMETRIC_TOLERANCE(...) POSITION_TOLERANCE());
    #   * the simple subtype,    #83=CYLINDRICITY_TOLERANCE(name,desc,magnitude,shape);
    # Matching only the first missed 10 of 36 frames on the NIST sample --
    # every cylindricity, symmetry, flatness and most perpendicularity frames,
    # i.e. exactly the characteristics that take no datum-modified zone.
    pattern = (r"(?:GEOMETRIC_TOLERANCE|[A-Z_]*TOLERANCE)"
               r"\('([^']*)','[^']*',#(\d+),#(\d+)")
    for m in re.finditer(pattern, flat):
        name, mag_ref = m.group(1), m.group(2)
        mv = re.search(r"LENGTH_MEASURE\(([-0-9.eE+]+)\)", ents.get(mag_ref, ""))
        if mv:
            out.setdefault(_key(name), []).append(float(mv.group(1)))
    return out


# PMI presentation strings look like  'DIM\\w2 XM3 x 0.5-6g'  where \\w is a
# field separator. Escapes such as \X\A0 (non-breaking space) and \X2\2300\X0\
# (a UTF-16 run) are STEP's ISO 10303-21 control directives, not text.
_THREAD_RE = re.compile(
    r"(?:(\d+)\s*X\s*)?"                       # optional '2X' multiplier
    r"(M\d+(?:\.\d+)?)"                        # 'M3', 'M42', 'M1.6'
    r"\s*[xX\u00d7]\s*(\d+(?:\.\d+)?)"         # ' x 0.5' pitch
    r"\s*(-\s*\w+)?",                          # optional '-6g' class
)


def _decode_pmi_string(raw: str) -> str:
    """Strip ISO 10303-21 control directives from a PMI presentation string."""
    s = raw.replace("\\\\n", "\n").replace("\\\\w", "\t").replace("\\\\x", "")
    s = re.sub(r"\\X2\\([0-9A-Fa-f]+)\\X0\\",
               lambda m: "".join(
                   chr(int(m.group(1)[i:i + 4], 16))
                   for i in range(0, len(m.group(1)), 4)), s)
    s = re.sub(r"\\X\\A0", " ", s)             # non-breaking space
    s = re.sub(r"\\X\\B1", "\u00b1", s)        # plus/minus
    s = re.sub(r"<!SYM=[^>]*>", "", s)         # CAD-internal symbol references
    return s


def _threads(text: str) -> List[Thread]:
    """Thread designations stated in the file's PMI presentation strings."""
    found: Dict[str, Thread] = {}
    for raw in re.findall(
            r"DESCRIPTIVE_REPRESENTATION_ITEM\('[^']*',\s*'((?:[^']|'')*)'\)",
            text):
        s = _decode_pmi_string(raw)
        if not s.startswith("DIM"):
            continue
        for m in _THREAD_RE.finditer(s):
            n, major, pitch, cls = m.groups()
            desig = f"{major} x {pitch}"
            if cls:
                desig += cls.replace(" ", "")
            t = Thread(designation=desig, count=int(n) if n else 1,
                       nominal=float(major[1:]))
            # The same thread is presented once per annotated instance; keep
            # the largest stated multiplier rather than counting duplicates,
            # because '2X M3 x 0.5-6g' appearing three times is still 2 off.
            prev = found.get(desig)
            if prev is None or t.count > prev.count:
                found[desig] = t
    return list(found.values())


# --------------------------------------------------------------------------- #
# OCCT pass: characteristics, datums and anchors
# --------------------------------------------------------------------------- #
def _face_anchor(shape) -> Optional[Tuple[float, float, float]]:
    """Centre of mass of a face, in model coordinates.

    This is where a feature control frame's leader lands, which is the only
    reason the referenced face is resolved at all.
    """
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    g = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, g)
    c = g.CentreOfMass()
    return (c.X(), c.Y(), c.Z())


def read(path: str) -> PMI:
    """Read all PMI from a STEP file. Returns an empty :class:`PMI` otherwise.

    Never raises: a file with no PMI, a non-STEP file, or an OCCT reader
    failure all yield an empty result, because PMI is strictly additive to a
    drawing that is already complete without it.
    """
    if not re.search(r"\.stp$|\.step$", path, re.I):
        return PMI()
    try:
        return _read(path)
    except Exception:
        return PMI()


def _read(path: str) -> PMI:
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.XCAFDoc import (XCAFDoc_DocumentTool, XCAFDoc_GeomTolerance,
                             XCAFDoc_Datum, XCAFDoc_ShapeTool)
    from OCP.TDF import TDF_LabelSequence, TDF_AttributeIterator

    with open(path, "r", errors="ignore") as fh:
        text = fh.read()

    # Cheap reject: the XCAF reader costs ~0.5 s and every sample file except
    # the NIST part has no PMI at all.
    if "GEOMETRIC_TOLERANCE" not in text and "DIMENSIONAL_SIZE" not in text:
        return PMI()

    magnitudes = _tolerance_values(text)
    used: Dict[str, int] = {}

    doc = TDocStd_Document(TCollection_ExtendedString("pmi"))
    reader = STEPCAFControl_Reader()
    reader.SetGDTMode(True)
    reader.SetNameMode(True)
    reader.ReadFile(path)
    reader.Transfer(doc)
    tool = XCAFDoc_DocumentTool.DimTolTool_s(doc.Main())

    def _attr(label, cls):
        it = TDF_AttributeIterator(label)
        while it.More():
            if isinstance(it.Value(), cls):
                return it.Value()
            it.Next()
        return None

    pmi = PMI(threads=_threads(text))

    seq = TDF_LabelSequence()
    tool.GetGeomToleranceLabels(seq)
    for i in range(1, seq.Length() + 1):
        label = seq.Value(i)
        holder = _attr(label, XCAFDoc_GeomTolerance)
        if holder is None:
            continue
        obj = holder.GetObject()
        char = CHARACTERISTICS.get(str(obj.GetType()).split("_")[-1])
        if char is None:
            continue

        name_h = obj.GetSemanticName()
        name = _key(name_h.ToCString() if name_h is not None else "")

        # Magnitude: prefer the raw-file value, because OCCT reports 0.0 here.
        # A composite frame lists several under one name, consumed in order.
        vals = magnitudes.get(name) or []
        idx = used.get(name, 0)
        if idx < len(vals):
            value = vals[idx]
            used[name] = idx + 1
        else:
            value = obj.GetValue()
        if not value:
            continue          # no magnitude anywhere: not a statable frame

        datums: List[str] = []
        ds = TDF_LabelSequence()
        XCAFDoc_DocumentTool.DimTolTool_s(doc.Main()).GetDatumOfTolerLabels_s(
            label, ds)
        for k in range(1, ds.Length() + 1):
            da = _attr(ds.Value(k), XCAFDoc_Datum)
            if da is None:
                continue
            n = da.GetObject().GetName()
            letter = n.ToCString() if n is not None else ""
            if letter and letter not in datums:
                datums.append(letter)

        anchor = None
        shapes = TDF_LabelSequence()
        tool.GetRefShapeLabel_s(label, shapes, TDF_LabelSequence())
        for k in range(1, shapes.Length() + 1):
            s = XCAFDoc_ShapeTool.GetShape_s(shapes.Value(k))
            if s is not None and not s.IsNull() \
                    and str(s.ShapeType()).endswith("FACE"):
                anchor = _face_anchor(s)
                break

        pmi.tolerances.append(GeomTolerance(
            characteristic=char, value=round(float(value), 4),
            datums=tuple(datums),
            diametral=str(obj.GetTypeOfValue()).endswith("Diameter"),
            modifier=MODIFIERS.get(
                str(obj.GetMaterialRequirementModifier()).split("_")[-1]),
            anchor=anchor, name=name))

    seen: Dict[str, DatumFeature] = {}
    seq = TDF_LabelSequence()
    tool.GetDatumLabels(seq)
    for i in range(1, seq.Length() + 1):
        da = _attr(seq.Value(i), XCAFDoc_Datum)
        if da is None:
            continue
        n = da.GetObject().GetName()
        letter = n.ToCString() if n is not None else ""
        # A datum is referenced once per tolerance that cites it; the drawing
        # carries one symbol per letter.
        if letter and letter not in seen:
            seen[letter] = DatumFeature(letter=letter)
    pmi.datums = [seen[k] for k in sorted(seen)]

    # Give threads an anchor by pairing them with holes of the stated major
    # diameter. Done by the caller, which has the recognised holes.
    return pmi


def attach_thread_anchors(pmi: PMI, holes) -> None:
    """Anchor each thread on a recognised hole of matching nominal diameter.

    A tapped hole is modelled at its *minor* (tap-drill) diameter, so an M3
    thread appears as a bore somewhere between about 2.5 and 3.0 mm. The
    match therefore accepts a hole from 0.75x to 1.02x the nominal, and takes
    the closest. This only positions a thread the file already states -- it
    never creates one.
    """
    taken = set()
    for t in pmi.threads:
        if t.nominal is None:
            continue
        best, best_err = None, None
        for i, h in enumerate(holes):
            if i in taken or not getattr(h, "is_closed", True):
                continue
            if not (0.75 * t.nominal <= h.diameter <= 1.02 * t.nominal):
                continue
            err = abs(h.diameter - t.nominal)
            if best_err is None or err < best_err:
                best, best_err = i, err
        if best is not None:
            taken.add(best)
            h = holes[best]
            t.anchor = (h.x, h.y, h.z)
            t.axis = tuple(h.axis)
