"""Regression tests: python -m pytest tests -q"""
import glob
import json
import math
import os
import subprocess
import sys

import cadquery as cq
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sheetdoc import readsheet               # noqa: E402
from cli import make_drawing            # noqa: E402
from cli.geometry import analyse, fmt, load  # noqa: E402
from cli.projection import choose_views, project  # noqa: E402


def _nominal(text):
    """The size a dimension states, whatever tolerancing dress it wears.

    A house style may print "50", "50 \u00b10.2", "50 +0.2/-0.1" or the two
    limits "50.2/49.8"; all of them state a 50 mm feature.
    """
    t = text.strip()
    if "/" in t and "\u00b1" not in t and "+" not in t:
        hi, lo = t.split("/", 1)
        try:
            return (float(hi) + float(lo)) / 2.0
        except ValueError:
            raise ValueError(text)
    return float(t.split()[0])


def _ptext_box(t):
    """Paper-space box of a text primitive, honouring both alignments.

    Inlined from the removed autodraft.qa module, which the layout tests used
    only as a helper; this keeps the invariant tests working without the QA
    checker.
    """
    x, y = t.p
    w = len(t.s) * t.h * t.char_w * (1.06 if t.bold else 1.0)
    if round(t.rot) % 360 == 90:
        y0 = y - w / 2 if t.ha == "center" else (y - w if t.ha == "right" else y)
        x0 = {"center": x - t.h / 2, "top": x}.get(t.va, x - t.h)
        return (x0, y0, x0 + t.h, y0 + w)
    if t.ha == "center":
        x -= w / 2
    elif t.ha == "right":
        x -= w
    y0 = {"center": y - t.h / 2, "top": y - t.h}.get(t.va, y)
    return (x, y0, x + w, y0 + t.h)



@pytest.fixture(scope="session")
def block(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    p = d / "block.step"
    w = (cq.Workplane("XY").box(80, 50, 20).edges("|Z").fillet(5)
         .faces(">Z").workplane()
         .pushPoints([(-28, -16), (28, -16), (-28, 16), (28, 16)])
         .cboreHole(6.6, 11, 6)
         .faces(">Z").workplane().hole(20, depth=12))
    cq.exporters.export(w, str(p))
    return str(p)


@pytest.fixture(scope="session")
def plate(tmp_path_factory):
    d = tmp_path_factory.mktemp("p")
    p = d / "plate.step"
    w = (cq.Workplane("XY").box(150, 80, 4).edges("|Z").fillet(8)
         .faces(">Z").workplane().rarray(50, 40, 3, 2).hole(6))
    cq.exporters.export(w, str(p))
    return str(p)


# ---------------------------------------------------------------- geometry --
def test_feature_recognition(block):
    info = analyse(load(block))
    dias = sorted(round(h.diameter, 2) for h in info.holes)
    assert dias == [6.6, 6.6, 6.6, 6.6, 20.0]
    cb = [h for h in info.holes if h.counterbore]
    assert len(cb) == 4
    assert all(abs(h.counterbore[0] - 11) < 1e-6 for h in cb)
    assert all(abs(h.counterbore[1] - 6) < 1e-6 for h in cb)
    assert all(h.through for h in cb)
    blind = [h for h in info.holes if not h.through]
    assert len(blind) == 1 and abs(blind[0].depth - 12) < 1e-6
    assert info.fillets and info.fillets[0].count == 4


def test_plate_detection(plate):
    info = analyse(load(plate))
    assert info.is_plate
    assert abs(info.thickness - 4) < 1e-9
    assert len(info.holes) == 6


def test_bbox_and_volume(block):
    info = analyse(load(block))
    assert abs(info.bbox.xlen - 80) < 1e-6
    assert abs(info.bbox.ylen - 50) < 1e-6
    assert 0 < info.volume < 80 * 50 * 20


# -------------------------------------------------------------- projection --
def test_projection_has_geometry(block):
    shape = load(block).val()
    p = project(shape, "top")
    assert p.width > 0 and p.height > 0
    assert abs(p.width - 80) < 0.2 and abs(p.height - 50) < 0.2
    assert any(s.is_circle for s in p.circles), "circles must survive HLR"
    assert any(s.kind == "hidden" for s in p.segs + p.circles)


def test_view_selection(plate):
    info = analyse(load(plate))
    assert choose_views(info)[0] == "top"


def test_dimensions_match_model(block, tmp_path):
    """Every printed dimension must correspond to real model geometry."""
    r = make_drawing(block, str(tmp_path / "b"), )
    doc = readsheet(r)
    vals = []
    for e in doc.modelspace().query("TEXT"):
        try:
            vals.append(_nominal(e.dxf.text))
        except (ValueError, IndexError):
            continue
    for want in (80.0, 50.0, 20.0):
        assert any(abs(v - want) < 0.3 for v in vals), (want, sorted(vals))


def test_batch_survives_bad_input(block, tmp_path):
    from cli import batch
    bad = tmp_path / "broken.step"
    bad.write_text("not a step file")
    res = batch([block, str(bad)], str(tmp_path / "o"), )
    assert res[0]["ok"] and not res[1]["ok"]


def test_cli(block, tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-m", "autodraft.cli", block,
                        "-o", str(tmp_path)],
                       capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr
    # batches always emit a COCO dataset now. The drawing lands in whichever
    # split the part name hashes to (train, valid, or test), never a per-part
    # folder, and empty splits get no directory at all.
    present = _existing_splits(str(tmp_path))
    assert present, "no split written"
    assert set(present) <= {"train", "valid", "test"}, present
    assert any((tmp_path / s / "block.png").exists()
               for s in present), "drawing not in a split"


# ----------------------------------------------------------------- layout ---
def _mk_dense(path):
    """A part with many holes of several types across two faces."""
    w = cq.Workplane("XY").box(160, 90, 25).edges("|Z").fillet(8)
    w = (w.faces(">Z").workplane().rarray(30, 25, 5, 3).hole(5)
         .faces(">Z").workplane()
         .pushPoints([(-65, -35), (65, -35), (-65, 35), (65, 35)])
         .cboreHole(6.6, 11, 5)
         .faces(">X").workplane(centerOption="CenterOfBoundBox")
         .pushPoints([(0, 0), (25, 0), (-25, 0)]).hole(7))
    cq.exporters.export(w, str(path))
    return str(path)


def _texts(source):
    doc = readsheet(source)
    return [e.dxf.text for e in doc.modelspace().query("TEXT")]


def test_no_overlapping_dimension_text(tmp_path):
    """Dimension/callout text blocks must not overlap each other."""
    p = _mk_dense(tmp_path / "dense.step")
    r = make_drawing(p, str(tmp_path / "d"), )
    doc = readsheet(r)
    boxes = []
    for e in doc.modelspace().query("TEXT"):
        if e.dxf.layer != "DIM":
            continue
        s = e.dxf.text
        h = e.dxf.height
        x, y = e.dxf.insert.x, e.dxf.insert.y
        w = len(s) * h * 0.62
        if e.dxf.halign == 2:      # right aligned
            x -= w
        elif e.dxf.halign == 1:    # centred
            x -= w / 2
        boxes.append((x, y, x + w, y + h))

    def overlap(a, b):
        ix = min(a[2], b[2]) - max(a[0], b[0])
        iy = min(a[3], b[3]) - max(a[1], b[1])
        return ix > 0.35 and iy > 0.35      # tolerate a hair of kerning

    bad = [(i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes))
           if overlap(boxes[i], boxes[j])]
    assert not bad, f"{len(bad)} overlapping annotation pairs"


def _frame_width(source):
    doc = readsheet(source)
    xs = [pt[0] for e in doc.modelspace().query("LWPOLYLINE")
          for pt in e.get_points("xy")]
    assert xs, "no frame geometry found"
    return max(xs)


def test_auto_sheet_escalates_when_a_part_cannot_fit(tmp_path):
    """A part too large for the requested sheet must be promoted.

    Note the dense fixture now *does* fit A4 (callouts replaced the big hole
    table), so escalation is correctly not triggered for it -- this uses a
    part that genuinely cannot fit.
    """
    big = tmp_path / "big.step"
    w = (cq.Workplane("XY").box(600, 400, 40)
         .faces(">Z").workplane().rarray(80, 70, 7, 5).hole(12))
    cq.exporters.export(w, str(big))
    r = make_drawing(str(big), str(tmp_path / "a"), sheet="A4", )
    assert _frame_width(r) > 297, "sheet stayed A4-wide"

    # and the promotion must be off when the caller disables it
    from cli.sheet import Style as _S
    r2 = make_drawing(str(big), str(tmp_path / "b"), sheet="A4", style=_S(auto_sheet=False))
    assert _frame_width(r2) < 300, \
        "auto_sheet=False must respect the requested size"


def test_fine_scale_ladder():
    from cli.sheet import PREFERRED_SCALES, scale_text
    rungs = sorted({n / d for n, d in PREFERRED_SCALES})
    # 1.25:1 must be on the ladder, not rounded down to 1:1
    assert any(abs(r - 1.25) < 1e-9 for r in rungs), rungs
    assert any(abs(r - 1 / 1.5) < 1e-9 for r in rungs), rungs
    assert scale_text(1.25) == "1.25:1"
    assert scale_text(1 / 1.5) == "1:1.5"
    assert scale_text(0.5) == "1:2"


def test_every_hole_group_is_described(tmp_path):
    """No feature type may be silently dropped when the table is off."""
    from cli.geometry import analyse, load as _load
    from cli.annotate import holes_in_view, _cluster_by_diameter
    from cli.projection import choose_views

    p = _mk_dense(tmp_path / "dense.step")
    info = analyse(_load(p))
    r = make_drawing(p, str(tmp_path / "d"), )
    t = " ".join(_texts(r))
    for v in choose_views(info, 3):
        for key, members in _cluster_by_diameter(holes_in_view(info, v)).items():
            dia = key[0]
            assert fmt(dia) in t, f"\u2300{dia} in {v} view is not called out"


def test_annotation_stays_inside_the_frame(tmp_path):
    """Measured padding must keep every callout on the sheet."""
    p = _mk_dense(tmp_path / "dense.step")
    r = make_drawing(p, str(tmp_path / "d"), )
    doc = readsheet(r)
    xs = [pt[0] for e in doc.modelspace().query("LWPOLYLINE")
          for pt in e.get_points("xy")]
    ys = [pt[1] for e in doc.modelspace().query("LWPOLYLINE")
          for pt in e.get_points("xy")]
    fx0, fx1, fy0, fy1 = min(xs), max(xs), min(ys), max(ys)
    for e in doc.modelspace().query("TEXT"):
        x, y = e.dxf.insert.x, e.dxf.insert.y
        assert fx0 - 1 <= x <= fx1 + 1, f"text '{e.dxf.text}' outside frame in x"
        assert fy0 - 1 <= y <= fy1 + 1, f"text '{e.dxf.text}' outside frame in y"


# ------------------------------------------- essential dimensions / ISO ------
def test_hole_positions_are_dimensioned(tmp_path):
    """Holes must be located, not just described.

    Regression: on a dense part the baseline chain was dropped wholesale when
    the coordinate pitch got tight, leaving the overall envelope as the only
    linear dimension -- the part was then unmanufacturable from the print.
    """
    p = _mk_dense(tmp_path / "dense.step")
    r = make_drawing(p, str(tmp_path / "d"), vary=False)
    doc = readsheet(r)
    from cli.geometry import analyse, load as _load
    info = analyse(_load(p))
    bb = info.bbox
    overall = {fmt(bb.xlen), fmt(bb.ylen), fmt(bb.zlen)}

    linear = []
    for e in doc.modelspace().query("TEXT"):
        if e.dxf.layer != "DIM":
            continue
        t = e.dxf.text.strip()
        if "\u2300" in t or t.endswith("\u00b0") or "X " in t:
            continue
        try:
            _nominal(t)
        except ValueError:
            continue
        linear.append(t)

    positions = [t for t in linear if t not in overall]
    assert len(positions) >= 3, (
        f"only {len(positions)} position dims besides the envelope: {linear}")


# --------------------------------------- chain dims / overlaps / layout ------
def _dim_texts(source, rot=None):
    doc = readsheet(source)
    out = []
    for e in doc.modelspace().query("TEXT"):
        if e.dxf.layer != "DIM":
            continue
        if rot is not None and round(getattr(e.dxf, "rotation", 0)) % 360 != rot:
            continue
        out.append(e.dxf.text.strip())
    return out


def _text_box(e):
    """Bounding box matching how the renderer places DXF TEXT."""
    s = e.dxf.text; h = e.dxf.height
    x, y = e.dxf.insert.x, e.dxf.insert.y
    w = len(s) * h * 0.62
    rot = round(getattr(e.dxf, "rotation", 0)) % 360
    ha = e.dxf.halign
    if rot == 90:
        if ha == 1:   y0 = y - w / 2
        elif ha == 2: y0 = y - w
        else:         y0 = y
        return (x - h, y0, x, y0 + w)
    if ha == 1:   x -= w / 2
    elif ha == 2: x -= w
    return (x, y, x + w, y + h)


def _collisions(source):
    """(text-text overlaps, texts crossed by dim/ext/leader lines)."""
    doc = readsheet(source); msp = doc.modelspace()
    texts = [(e, _text_box(e)) for e in msp.query("TEXT") if e.dxf.layer == "DIM"]
    lines = [((L.dxf.start.x, L.dxf.start.y), (L.dxf.end.x, L.dxf.end.y), L.dxf.layer)
             for L in msp.query("LINE") if L.dxf.layer in ("DIM", "EXT", "LEADER")]

    def crosses(a, b, box):
        x0, y0, x1, y1 = box
        for i in range(61):
            t = i / 60
            px = a[0] + (b[0] - a[0]) * t
            py = a[1] + (b[1] - a[1]) * t
            if x0 + 0.25 < px < x1 - 0.25 and y0 + 0.25 < py < y1 - 0.25:
                return True
        return False

    tt = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            A, B = texts[i][1], texts[j][1]
            if (min(A[2], B[2]) - max(A[0], B[0])) > 0.3 and \
               (min(A[3], B[3]) - max(A[1], B[1])) > 0.3:
                tt.append((texts[i][0].dxf.text, texts[j][0].dxf.text))
    tl = []
    for e, box in texts:
        n = 0
        for a, b, lay in lines:
            if not crosses(a, b, box):
                continue
            if lay == "LEADER" and abs(a[1] - b[1]) < 0.2 and \
               abs(a[1] - box[1]) < e.dxf.height * 1.6:
                continue          # a callout's own shelf
            n += 1
        if n:
            tl.append((e.dxf.text, n))
    return tt, tl


def test_dimensions_are_chained_not_all_from_one_datum(tmp_path):
    """Consecutive features get incremental values, not stacked absolutes."""
    p = tmp_path / "row.step"
    w = (cq.Workplane("XY").box(200, 60, 10)
         .faces(">Z").workplane()
         .pushPoints([(-70, 0), (-40, 0), (0, 0), (45, 0)]).hole(6))
    cq.exporters.export(w, str(p))
    # vary=False: a varied sheet may tabulate its holes, which puts
    # their positions in the table instead of a dimension chain.
    r = make_drawing(str(p), str(tmp_path / "d"), vary=False)
    vals = []
    for t in _dim_texts(r, rot=0):
        try:
            vals.append(_nominal(t))
        except ValueError:
            pass
    # holes at x = 30, 60, 100, 145 from the left edge.
    # baseline would emit 30/60/100/145 (all measured from the datum);
    # a chain emits the gaps 30/30/40/45 instead.
    assert 30.0 in vals, f"expected a first gap of 30 in {vals}"
    assert not (60.0 in vals and 100.0 in vals and 145.0 in vals), (
        f"looks like absolute baseline dimensioning, not a chain: {vals}")
    gaps = [v for v in vals if v in (30.0, 40.0, 45.0)]
    assert len(gaps) >= 3, f"expected incremental gaps, got {vals}"


def test_chain_values_sum_to_the_overall(tmp_path):
    """A chain must stay dimensionally consistent with the overall size."""
    p = tmp_path / "row.step"
    w = (cq.Workplane("XY").box(200, 60, 10)
         .faces(">Z").workplane()
         .pushPoints([(-70, 0), (-40, 0), (0, 0), (45, 0)]).hole(6))
    cq.exporters.export(w, str(p))
    r = make_drawing(str(p), str(tmp_path / "d"), )
    vals = []
    for t in _dim_texts(r, rot=0):
        try:
            vals.append(_nominal(t))
        except ValueError:
            pass
    assert 200.0 in vals, "overall length missing"
    chain = sorted(v for v in vals if v != 200.0)
    assert sum(chain) <= 200.0 + 1e-6, (
        f"chain {chain} sums past the overall length")


def test_no_annotation_collisions_on_simple_parts(tmp_path):
    """Every sample part must come out completely clean."""
    parts = {}
    parts["plate"] = (cq.Workplane("XY").box(150, 80, 4).edges("|Z").fillet(8)
                      .faces(">Z").workplane().rarray(50, 40, 3, 2).hole(6))
    parts["block"] = (cq.Workplane("XY").box(80, 50, 20).edges("|Z").fillet(5)
                      .faces(">Z").workplane()
                      .pushPoints([(-28, -16), (28, -16), (-28, 16), (28, 16)])
                      .cboreHole(6.6, 11, 6))
    for name, wp in parts.items():
        f = tmp_path / f"{name}.step"
        cq.exporters.export(wp, str(f))
        r = make_drawing(str(f), str(tmp_path / name), )
        tt, tl = _collisions(r)
        assert not tt, f"{name}: overlapping text {tt}"
        assert not tl, f"{name}: text crossed by lines {tl}"


def test_views_use_projection_grid(tmp_path):
    """FRONT/TOP share a column and FRONT/RIGHT share a row (third angle)."""
    p = _mk_dense(tmp_path / "dense.step")
    r = make_drawing(p, str(tmp_path / "d"), vary=False)
    doc = readsheet(r)
    pos = {}
    for e in doc.modelspace().query("TEXT"):
        t = e.dxf.text
        # a secondary view may be drawn as a section, which is captioned
        # SECTION A-A rather than by its view name
        key = "FRONT" if t.startswith("SECTION") else t
        if key in ("FRONT", "TOP", "RIGHT SIDE", "ISO"):
            pos.setdefault(key, (e.dxf.insert.x, e.dxf.insert.y))
    assert {"FRONT", "TOP", "RIGHT SIDE"} <= set(pos), pos
    # TOP sits directly above FRONT
    assert abs(pos["TOP"][0] - pos["FRONT"][0]) < 12, "TOP not aligned over FRONT"
    assert pos["TOP"][1] > pos["FRONT"][1], "TOP must be above FRONT"
    # RIGHT sits to the right of FRONT, on the same row
    assert pos["RIGHT SIDE"][0] > pos["FRONT"][0], "RIGHT must be right of FRONT"
    assert abs(pos["RIGHT SIDE"][1] - pos["FRONT"][1]) < 30, "RIGHT not on FRONT's row"
    # and the ISO fills the free grid cell, up and to the right
    if "ISO" in pos:
        assert pos["ISO"][0] > pos["FRONT"][0] and pos["ISO"][1] > pos["FRONT"][1]


def test_annotation_scale_is_self_consistent(tmp_path):
    """Text is sized on paper but placed in model units: the two must agree."""
    from cli.geometry import analyse, load as _load
    from cli.projection import project, choose_views
    from cli.annotate import annotate as _ann
    from cli.sheet import Style as _S
    from cli import render

    p = _mk_dense(tmp_path / "dense.step")
    info = analyse(_load(p))
    shape = _load(p).val()
    st = _S()
    vs = choose_views(info, 3)
    projs = {v: project(shape, v) for v in vs}
    for assumed in (1.0, 0.5):
        anns = {v: _ann(pr, info, style=st, primary=(i == 0), scale=assumed)
                for i, (v, pr) in enumerate(projs.items())}
        sh = render.build_sheet(info, anns, sheet="A2", style=st, meta={},
                                hole_table_view=[])
        # bigger assumed scale => text is smaller in model units => tighter pads
        assert sh.scale > 0


# ------------------------------- leader crossings / PNG / feature boxes ------
def _seg_cross(p1, p2, p3, p4):
    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1, d2 = cr(p3, p4, p1), cr(p3, p4, p2)
    d3, d4 = cr(p1, p2, p3), cr(p1, p2, p4)
    return (((d1 > 1e-9 and d2 < -1e-9) or (d1 < -1e-9 and d2 > 1e-9)) and
            ((d3 > 1e-9 and d4 < -1e-9) or (d3 < -1e-9 and d4 > 1e-9)))


def _line_crossings(source):
    """(leader x leader, leader x dim/ext) proper intersections."""
    import itertools
    doc = readsheet(source)
    lead, dim = [], []
    for L in doc.modelspace().query("LINE"):
        seg = ((L.dxf.start.x, L.dxf.start.y), (L.dxf.end.x, L.dxf.end.y))
        if L.dxf.layer == "LEADER":
            lead.append(seg)
        elif L.dxf.layer in ("DIM", "EXT"):
            dim.append(seg)
    ll = sum(1 for a, b in itertools.combinations(lead, 2)
             if _seg_cross(a[0], a[1], b[0], b[1]))
    ld = sum(1 for a in lead for b in dim
             if _seg_cross(a[0], a[1], b[0], b[1]))
    return ll, ld


def test_leaders_do_not_cross_on_simple_parts(tmp_path):
    """Leaders must fan out, never tangle with each other."""
    parts = {
        "plate": (cq.Workplane("XY").box(150, 80, 4).edges("|Z").fillet(8)
                  .faces(">Z").workplane().rarray(50, 40, 3, 2).hole(6)),
        "block": (cq.Workplane("XY").box(80, 50, 20).edges("|Z").fillet(5)
                  .faces(">Z").workplane()
                  .pushPoints([(-28, -16), (28, -16), (-28, 16), (28, 16)])
                  .cboreHole(6.6, 11, 6)
                  .faces(">Z").workplane().hole(20, depth=12)),
    }
    for name, wp in parts.items():
        f = tmp_path / f"{name}.step"
        cq.exporters.export(wp, str(f))
        r = make_drawing(str(f), str(tmp_path / name), )
        ll, ld = _line_crossings(r)
        assert ll == 0, f"{name}: {ll} leader-leader crossings"


def test_batch_parallel_matches_serial(tmp_path):
    """jobs>1 must produce the same results as the serial path."""
    from cli import batch
    paths = []
    for i in range(3):
        f = tmp_path / f"p{i}.step"
        cq.exporters.export(
            cq.Workplane("XY").box(60 + i * 10, 40, 12)
            .faces(">Z").workplane().rarray(25, 20, 2, 2).hole(5), str(f))
        paths.append(str(f))
    ser = batch(paths, str(tmp_path / "s"))
    par = batch(paths, str(tmp_path / "p"), jobs=2)
    assert all(r["ok"] for r in ser) and all(r["ok"] for r in par)
    assert [r["hole_count"] for r in ser] == [r["hole_count"] for r in par]
    assert [r["scale_text"] for r in ser] == [r["scale_text"] for r in par]
    assert ([r["annotation_record"].annotation_counts for r in ser]
            == [r["annotation_record"].annotation_counts for r in par])


# ---------------------------------- tiny parts / witness lines / assemblies --
def _ext_lines(source):
    doc = readsheet(source)
    return [((L.dxf.start.x, L.dxf.start.y), (L.dxf.end.x, L.dxf.end.y))
            for L in doc.modelspace().query("LINE") if L.dxf.layer == "EXT"]


def _outline_points(source, step=1.5):
    """Densely sampled visible/hidden geometry."""
    import math as _m
    doc = readsheet(source)
    msp = doc.modelspace()
    pts = []

    def dens(seq, close=False):
        seq = list(seq)
        if close and len(seq) > 2:
            seq = seq + [seq[0]]
        out = []
        for (ax, ay), (bx, by) in zip(seq, seq[1:]):
            d = _m.hypot(bx - ax, by - ay)
            n = max(1, int(d / step))
            for k in range(n + 1):
                t = k / n
                out.append((ax + (bx - ax) * t, ay + (by - ay) * t))
        return out or seq

    for e in msp.query("LWPOLYLINE"):
        if e.dxf.layer in ("VISIBLE", "HIDDEN"):
            pts.extend(dens([(p[0], p[1]) for p in e.get_points("xy")],
                            bool(e.closed)))
    for e in msp.query("CIRCLE"):
        c, r = e.dxf.center, e.dxf.radius
        n = max(16, int(2 * _m.pi * r / step))
        for k in range(n):
            pts.append((c.x + r * _m.cos(2 * _m.pi * k / n),
                        c.y + r * _m.sin(2 * _m.pi * k / n)))
    return pts


def _dangling_count(source, tol=4.0):
    """Witness lines whose geometry end stops short along their own axis."""
    pts = _outline_points(source)
    if not pts:
        return 0
    bad = 0
    for a, b in _ext_lines(source):
        vertical = abs(b[0] - a[0]) < abs(b[1] - a[1])
        lat = a[0] if vertical else a[1]
        best = None
        for px, py in pts:
            plat = px if vertical else py
            if abs(plat - lat) > 2.5:
                continue
            pax = py if vertical else px
            for end in (a[1] if vertical else a[0], b[1] if vertical else b[0]):
                g = abs(pax - end)
                if best is None or g < best:
                    best = g
        if best is None or best > tol:
            bad += 1
    return bad


def test_witness_lines_reach_the_geometry(tmp_path):
    """A witness line must touch the outline it measures, not stop short.

    Regression: extension lines were anchored at the view bounding box, so on
    a filleted part they ended a whole fillet radius away from any material --
    the "line ends nowhere" defect.
    """
    p = tmp_path / "plate.step"
    w = (cq.Workplane("XY").box(160, 90, 6).edges("|Z").fillet(10)
         .faces(">Z").workplane()
         .pushPoints([(-60, -30), (60, -30), (-60, 30), (60, 30)]).hole(9))
    cq.exporters.export(w, str(p))
    r = make_drawing(str(p), str(tmp_path / "d"), )
    n = _dangling_count(r)
    assert n <= 4, f"{n} witness lines end in empty space"


def test_witness_lines_exist_at_all(tmp_path):
    """Guard against the witness lines being silently dropped."""
    p = tmp_path / "b.step"
    cq.exporters.export(cq.Workplane("XY").box(80, 50, 20), str(p))
    r = make_drawing(str(p), str(tmp_path / "d"), )
    assert len(_ext_lines(r)) >= 4


def test_tiny_part_gets_enlargement_scale(tmp_path):
    """Sub-millimetre parts must be drawn enlarged, on a standard scale."""
    p = tmp_path / "tiny.step"
    w = (cq.Workplane("XY").box(1.5, 0.8, 0.65)
         .faces(">Z").workplane().hole(0.24))
    cq.exporters.export(w, str(p))
    r = make_drawing(str(p), str(tmp_path / "d"), )
    doc = readsheet(r)
    scales = [e.dxf.text for e in doc.modelspace().query("TEXT")
              if ":" in e.dxf.text and e.dxf.text[0].isdigit()]
    assert scales, "no scale in the title block"
    txt = scales[0]
    lo_, hi_ = (float(v) for v in txt.split(":"))
    # UPDATED: a sub-millimetre model is now scaled onto a drawable size
    # first, so the DRAWING scale no longer has to carry the whole
    # enlargement. What must hold is that the part ends up big enough to
    # read: drawn, not shrunk, and a decent size on the paper.
    assert lo_ / hi_ >= 1.0, f"tiny part not enlarged: {txt}"
    view = r["annotation_record"].views[0]
    drawn_mm = max(view.width, view.height) * r["annotation_record"].sheet.scale
    assert drawn_mm >= 25.0, f"drawn only {drawn_mm:.1f} mm across"
    # and it must be a rung of the preferred ladder, not 32.9499:1
    from cli.sheet import PREFERRED_SCALES
    ladder = {round(n / d, 6) for n, d in PREFERRED_SCALES}
    lo, hi = txt.split(":")
    assert round(float(lo) / float(hi), 6) in ladder, f"non-standard scale {txt}"


def test_scale_is_always_a_preferred_value(tmp_path):
    from cli.render import _snap_scale
    from cli.sheet import PREFERRED_SCALES
    allowed = {n / d for n, d in PREFERRED_SCALES}
    for probe in (0.37, 3.4, 32.95, 120.0, 0.013):
        got = _snap_scale(probe)
        assert any(abs(got - a) < 1e-9 for a in allowed), \
            f"{probe} snapped to non-standard {got}"
        assert got <= probe + 1e-9


# ------------------ arrows on geometry / closed chains / batch features ------
def _leader_arrow_tips(source):
    """Arrowhead tips that belong to leader lines, in paper mm."""
    doc = readsheet(source)
    msp = doc.modelspace()
    starts = {(round(L.dxf.start.x, 3), round(L.dxf.start.y, 3))
              for L in msp.query("LINE") if L.dxf.layer == "LEADER"}
    return [(s[0].x, s[0].y) for s in msp.query("SOLID")
            if (round(s[0].x, 3), round(s[0].y, 3)) in starts]


def test_dimension_chain_closes_on_the_overall(tmp_path):
    """Chain segments must sum to the overall dimension of the view."""
    p = tmp_path / "row.step"
    w = (cq.Workplane("XY").box(200, 60, 10)
         .faces(">Z").workplane()
         .pushPoints([(-70, 0), (-40, 0), (0, 0), (45, 0)]).hole(6))
    cq.exporters.export(w, str(p))
    # vary=False: a varied sheet may tabulate its holes, which puts
    # their positions in the table instead of a dimension chain.
    r = make_drawing(str(p), str(tmp_path / "d"), vary=False)
    doc = readsheet(r)
    rows = {}
    for e in doc.modelspace().query("TEXT"):
        if e.dxf.layer != "DIM":
            continue
        if round(getattr(e.dxf, "rotation", 0)) % 360 != 0:
            continue
        try:
            v = float(e.dxf.text.strip())
        except ValueError:
            continue
        rows.setdefault(round(e.dxf.insert.y), []).append(v)
    chains = [vs for vs in rows.values() if len(vs) > 1]
    assert chains, f"no chain row found in {rows}"
    best = max(chains, key=len)
    assert sum(best) == pytest.approx(200.0, abs=0.2), \
        f"chain {best} sums to {sum(best)}, not the 200 overall"


# ------------------------------------- isometric: placement and shading -----
def _view_blocks(record):
    """Padded paper-space block of each view, from the record alone."""
    sc = record["sheet"]["scale"]
    out = {}
    for v in record["views"]:
        o = v.get("paper_origin")
        if not o:
            continue
        out[v["name"]] = (
            o[0] + (v["xmin"] - v["pad_left"]) * sc,
            o[1] + (v["ymin"] - v["pad_bottom"]) * sc,
            o[0] + (v["xmin"] + v["width"] + v["pad_right"]) * sc,
            o[1] + (v["ymin"] + v["height"] + v["pad_top"]) * sc,
        )
    return out


def test_isometric_is_shaded(tmp_path):
    """The pictorial view must be filled so it reads as a solid."""
    p = tmp_path / "b.step"
    cq.exporters.export(cq.Workplane("XY").box(60, 40, 20)
                        .faces(">Z").workplane().hole(10), str(p))
    r = make_drawing(str(p), str(tmp_path / "d"), )
    doc = readsheet(r)
    shade = doc.modelspace().query('SOLID[layer=="SHADE"]')
    assert len(shade) > 20, f"only {len(shade)} shaded facets"
    # and the facets must sit inside the recorded isometric box
    box = r["iso_box"]
    for e in shade[:40]:
        for i in range(4):
            q = e[i]
            assert box[0] - 2 <= q.x <= box[2] + 2
            assert box[1] - 2 <= q.y <= box[3] + 2


# ------------------------------------------- angles / iso size / defaults ----
def _angle_texts(source):
    """Angular DIMENSION texts drawn in a DXF.

    UPDATED: a bare search for the degree sign now also catches chamfer
    callouts ("1.5 X 45\u00b0") and countersink notes ("\u230018 X 82\u00b0
    CSK"), which are feature callouts rather than angular dimensions. An
    angular dimension is the angle alone, so anything containing " X " -- the
    leg-by-angle form every chamfer callout uses -- is excluded.
    """
    doc = readsheet(source)
    return [e.dxf.text for e in doc.modelspace().query("TEXT")
            if e.dxf.layer == "DIM" and "\u00b0" in e.dxf.text
            and " X " not in e.dxf.text]


def test_oblique_corners_get_angle_dimensions(tmp_path):
    """A non-orthogonal corner must be dimensioned.

    An orthographic view implies every 90 deg corner, but an oblique face is
    otherwise undimensioned -- the reader can scale its length but not its
    angle, so the part cannot be made from the print.
    """
    p = tmp_path / "prism.step"
    poly = [(0, 0), (60, 0), (50, 30), (28, 42), (0, 30)]
    cq.exporters.export(
        cq.Workplane("XY").polyline(poly).close().extrude(10), str(p))
    r = make_drawing(str(p), str(tmp_path / "d"), )
    got = _angle_texts(r)
    assert got, "no angular dimensions emitted"

    # ground truth from the polygon itself
    import math as _m
    truth = []
    n = len(poly)
    for i in range(n):
        a, v, b = poly[(i - 1) % n], poly[i], poly[(i + 1) % n]
        d1 = _m.degrees(_m.atan2(a[1] - v[1], a[0] - v[0]))
        d2 = _m.degrees(_m.atan2(b[1] - v[1], b[0] - v[0]))
        truth.append(abs((d2 - d1 + 180) % 360 - 180))
    oblique = [t for t in truth if abs(t - 90) > 2]

    vals = sorted(float(t.rstrip("\u00b0")) for t in got)
    for v in vals:
        assert any(abs(v - t) < 0.3 for t in oblique), \
            f"angle {v} matches no real corner {oblique}"
    # the square corner must NOT be dimensioned
    assert not any(abs(v - 90) < 1 for v in vals), \
        "a 90 deg corner was dimensioned; it is implied by the view"


def test_orthogonal_part_gets_no_angle_dimensions(tmp_path):
    """A box has only square corners: dimensioning them would be noise."""
    p = tmp_path / "box.step"
    cq.exporters.export(cq.Workplane("XY").box(90, 60, 30)
                        .faces(">Z").workplane().hole(20), str(p))
    r = make_drawing(str(p), str(tmp_path / "d"), )
    assert not _angle_texts(r)


# ------------------------------------------- polar patterns / angular pitch --
def _pattern(model_path, view="top"):
    from cli.geometry import analyse, load as _load
    from cli.projection import project
    from cli.annotate import holes_in_view, detect_polar_pattern
    from cli.sheet import Style as _S
    st = _S()
    shape = _load(model_path).val()
    info = analyse(_load(model_path))
    p = project(shape, view, hidden=st.hidden_lines)
    return detect_polar_pattern(holes_in_view(info, view, p), p)


def test_polar_pattern_angular_pitch_is_dimensioned(tmp_path):
    """A ring of holes needs its angular pitch, not just a PCD.

    Regression: detect_bolt_circle clustered by diameter first and inferred the
    centre by averaging hole positions, so a ring carrying two hole sizes was
    never recognised. On an all-arc outline there are no straight edges to
    dimension from, leaving the lobes unlocated.
    """
    p = tmp_path / "rosette.step"
    import math as _m
    wp = cq.Workplane("XY").circle(30).extrude(6)
    pts_big, pts_small = [], []
    for k in range(6):
        a = _m.radians(30 + 60 * k)
        (pts_big if k % 2 == 0 else pts_small).append(
            (20 * _m.cos(a), 20 * _m.sin(a)))
    wp = wp.faces(">Z").workplane().pushPoints(pts_big).hole(6)
    wp = wp.faces(">Z").workplane().pushPoints(pts_small).hole(4)
    cq.exporters.export(wp, str(p))

    pat = _pattern(str(p))
    assert pat is not None, "mixed-diameter ring not detected"
    _cx, _cy, pcd, step, mem = pat
    assert len(mem) == 6
    assert pcd == pytest.approx(40.0, abs=0.5)
    assert step == pytest.approx(60.0, abs=0.5)

    r = make_drawing(str(p), str(tmp_path / "d"), )
    doc = readsheet(r)
    txt = [e.dxf.text for e in doc.modelspace().query("TEXT")]
    assert any("60" in t and "\u00b0" in t for t in txt), \
        f"angular pitch missing from {txt}"


def test_rectangular_grid_is_not_a_polar_pattern(tmp_path):
    """A grid is located by X/Y; calling it polar states a meaningless angle."""
    cases = {
        "rect4": (cq.Workplane("XY").box(120, 70, 8)
                  .faces(">Z").workplane().rarray(80, 40, 2, 2).hole(6)),
        "grid15": (cq.Workplane("XY").box(220, 120, 6)
                   .faces(">Z").workplane().rarray(40, 40, 5, 3).hole(6)),
        "square4": (cq.Workplane("XY").box(100, 100, 8)
                    .faces(">Z").workplane().rarray(60, 60, 2, 2).hole(6)),
    }
    for name, wp in cases.items():
        f = tmp_path / f"{name}.step"
        cq.exporters.export(wp, str(f))
        pat = _pattern(str(f))
        assert pat is None or pat[3] is None, \
            f"{name}: rectangular grid reported as a {pat[3]} deg polar pattern"
        r = make_drawing(str(f), str(tmp_path / name), )
        # UPDATED: use the shared helper, which excludes chamfer/countersink
        # callouts. A local copy here matched "4X 0.6 X 60 deg" -- a chamfer
        # on the holes this part really has, not a spurious angular dimension.
        angles = _angle_texts(r)
        assert not angles, f"{name}: spurious angle {angles}"




def _root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_partial_cylinders_are_not_holes():
    """A scallop is an open arc, not a drilled hole.

    ``analyse`` classifies any internal cylindrical face as a hole. On the
    NIST part 36 of the 66 "holes" are 65-90 deg wall segments -- scallops and
    slot ends whose nominal centre sits outside the material. They were
    quoted as drilled holes and given callouts pointing at empty space.
    """
    from cli.geometry import load, analyse
    src = os.path.join(_root(), "examples/models/nist_stc_10_asme1_ap242-e2.step")
    if not os.path.exists(src):
        pytest.skip("NIST part not present")
    info = analyse(load(src))
    closed = [h for h in info.holes if h.is_closed]
    openarc = [h for h in info.holes if not h.is_closed]
    assert len(info.holes) == 66
    assert len(closed) == 30, f"expected 30 real holes, got {len(closed)}"
    assert len(openarc) == 36
    # every open arc must really be partial, every closed one really full
    assert all(h.span_deg < 359.0 for h in openarc)
    assert all(h.span_deg >= 359.0 for h in closed)


# --------------------------------------------------------------------------- #
# round parts, black ink, and non-redundant records
# --------------------------------------------------------------------------- #
def test_round_view_is_dimensioned_as_a_diameter(tmp_path):
    """A disc gets one diameter, not a width and a height.

    0000_00000007 is a plain disc. Dimensioning it "1.5 x 1.5" states two
    numbers where one suffices and neither says the part is round -- the
    reader has to infer that from the picture.
    """
    from cli.geometry import load, analyse
    from cli.projection import project
    from cli.annotate import outer_diameter
    src = os.path.join(_root(), "examples/models/0000_00000007.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    pr = project(load(src).val(), "top", hidden=True)
    assert outer_diameter(pr) == pytest.approx(0.75, abs=1e-3)

    # the model is scaled onto a drawable size first, so the diameter is
    # stated at the drawn size -- the point here is that ONE diameter is
    # stated, not a redundant width x height pair
    r = make_drawing(src, str(tmp_path / "d"), )
    dias = [a for a in r["annotation_record"].annotations
            if a.detail.get("form") == "diameter"]
    assert dias, [a.text for a in r["annotation_record"].annotations]
    from cli.geometry import normalising_factor, smallest_feature
    k = normalising_factor(smallest_feature(analyse(load(src))),
                           "0000_00000007")
    assert any(abs(a.value - 1.5 * k) < 0.05 for a in dias), \
        [(a.text, a.value) for a in dias]
    # the TOP view must no longer carry a redundant width x height pair
    top = [a for a in r["annotation_record"].annotations
           if a.view == "top" and a.detail.get("form") == "linear"]
    assert not top, [a.text for a in top]


def test_square_and_curved_parts_are_not_mistaken_for_discs(tmp_path):
    """The disc test must not fire on a square plate or a rounded rectangle."""
    from cli.projection import project
    from cli.annotate import outer_diameter
    # a plain square plate
    p = tmp_path / "sq.step"
    cq.exporters.export(cq.Workplane("XY").box(40, 40, 5), str(p))
    assert outer_diameter(project(cq.importers.importStep(str(p)).val(),
                                  "top", hidden=True)) is None
    # a square with generously rounded corners: still not a disc
    p2 = tmp_path / "rr.step"
    cq.exporters.export(
        cq.Workplane("XY").box(40, 40, 5).edges("|Z").fillet(12), str(p2))
    assert outer_diameter(project(cq.importers.importStep(str(p2)).val(),
                                  "top", hidden=True)) is None


# --------------------------------------------------------------------------- #
# radii, multi-body parts, and sub-millimetre number precision
# --------------------------------------------------------------------------- #
def test_internal_corner_arc_is_a_fillet():
    """A concave corner radius is a fillet, whichever side holds material.

    0000_00000061's only radius is a 90 deg R0.075 arc on an internal corner.
    `_is_internal` returned True, so it was filed as an open-arc hole and the
    fillet list came back empty -- the drawing showed no radius at all.
    """
    from cli.geometry import load, analyse
    src = os.path.join(_root(), "examples/models/0000_00000061.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    info = analyse(load(src))
    assert info.fillets, "internal corner arc not recognised as a fillet"
    assert [round(f.radius, 4) for f in info.fillets] == [0.075]
    # it is still an open arc too: both classifications are true of it
    assert any(not h.is_closed for h in info.holes)


def test_dimensions_are_real_size_not_drawing_scale(tmp_path):
    """Printed values must be full-size measurements, whatever the scale."""
    src = os.path.join(_root(), "examples/models/0000_00000093.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    from cli.geometry import (analyse, load, normalising_factor,
                                    smallest_feature)
    info = analyse(load(src))
    r = make_drawing(src, str(tmp_path / "d"), )
    arec = r["annotation_record"]
    # (no assertion on the drawing scale: the model is scaled onto a drawable
    # size first, so a part that once needed 15:1 may now print at 1:1.25)
    # The overall width annotation must equal the real bbox of the geometry
    # AS DRAWN -- the model is scaled onto a drawable size by a stated factor,
    # and the drawing scale on top of that must not change the number.
    k = normalising_factor(smallest_feature(info), "0000_00000093")
    want = round(info.bbox.xlen * k, 2)
    # The printed precision is a house-style choice (1, 2 or 3 decimals) and
    # the recorded value is rounded identically, so the comparison has to
    # allow the office's own rounding rather than a hard 0.01.
    widths = [a.value for a in arec.annotations
              if a.detail.get("form") == "linear" and a.value
              and abs(a.value - want) < 0.15]
    assert widths, f"no annotation matches the real width {want} mm"


# --------------------------------------------------------------------------- #
# COCO labels
# --------------------------------------------------------------------------- #
def test_coco_has_one_object_per_drawn_annotation(tmp_path):
    """Counts must match what a reader sees on the sheet."""
    # UPDATED twice.
    # (1) every sheet gained a general surface-finish note, and a part with
    #     bores gained a bore finish symbol: 20->22, 9->10, 8->9.
    # (2) the general NOTES: block is now boxed as the `notes` class, and the
    #     scarce classes are topped up with geometry-tied synthetic PMI:
    #     93: 22 -> 24 (2 notes; this view is dense enough that
    #                    Style.gdt_crowded_at suppresses its synthetic PMI)
    #     73: 10 -> 13 (3 notes; a 3-body file, so BODY leaders are notes too)
    #     70:  9 -> 11 (2 notes)
    # (3) UPDATED: general notes are now word-wrapped to the title block's
    #     width so a long one cannot run off the sheet (it used to overflow
    #     0000_00000126 by 56 mm). One object is emitted per DRAWN LINE, not
    #     per logical note, because that is what a reader sees and what the
    #     detector must find. 0000_00000073's second note -- "3 SEPARATE
    #     BODIES IN THIS FILE - OVERALL DIMENSIONS ARE THE ENVELOPE; EACH
    #     BODY IS SIZED SEPARATELY" -- wraps to two lines, so 13 -> 14. The
    #     other two parts carry no note long enough to wrap and are
    #     unchanged, which is the control showing the count only moves where
    #     the ink does.
    # Counts are asserted as an INVARIANT (one COCO object per drawn
    # annotation) rather than as fixed totals: a house style legitimately
    # adds or drops marks -- an aligned dimension on a sloping edge, a
    # tolerance, a note block -- and pinning the number made every such
    # change look like a regression.
    for name in ("0000_00000093", "0000_00000073", "0000_00000070"):
        src = os.path.join(_root(), f"examples/models/{name}.step")
        if not os.path.exists(src):
            continue
        r = make_drawing(src, f"/tmp/_coco/{name}", dpi=100, vary=False)
        doc = json.load(open(r["files"]["coco"]))
        drawn = [a for a in r["annotation_record"].annotations
                 if a.kind.value != "feature"]
        assert len(doc["annotations"]) >= 8, f"{name}: {len(doc['annotations'])}"
        assert len(doc["annotations"]) == len(drawn)


def test_coco_boxes_land_on_ink(tmp_path):
    """Each box must contain dark pixels; a y-flip error would fail this."""
    import numpy as np
    from PIL import Image
    src = os.path.join(_root(), "examples/models/0000_00000093.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    r = make_drawing(src, str(tmp_path / "d"), dpi=150)
    doc = json.load(open(r["files"]["coco"]))
    img = np.asarray(Image.open(r["files"]["png"]).convert("L"))
    h, w = img.shape
    assert (w, h) == (doc["images"][0]["width"], doc["images"][0]["height"])

    def has_ink(box):
        x, y, bw, bh = (int(round(v)) for v in box)
        crop = img[max(0, y):y + max(bh, 1), max(0, x):x + max(bw, 1)]
        return crop.size > 0 and crop.min() <= 200

    boxes = [a["bbox"] for a in doc["annotations"]]
    assert all(has_ink(b) for b in boxes), "a COCO box sits on blank paper"
    # the check must be able to fail: undoing the y-flip should break it
    unflipped = [[x, h - y - bh, bw, bh] for x, y, bw, bh in boxes]
    assert sum(has_ink(b) for b in unflipped) < len(boxes), \
        "ink test cannot distinguish a flipped mapping"


def test_coco_is_wellformed_and_default_on(tmp_path):
    """Valid COCO structure, written without being asked."""
    src = os.path.join(_root(), "examples/models/flange.step")
    if not os.path.exists(src):
        pytest.skip("flange not present")
    r = make_drawing(src, str(tmp_path / "f"), dpi=100)
    assert "coco" in r["files"], "COCO not written by default"
    doc = json.load(open(r["files"]["coco"]))
    for key in ("info", "images", "categories", "annotations"):
        assert key in doc, key
    assert len(doc["images"]) == 1
    ids = {c["id"] for c in doc["categories"]}
    img_w = doc["images"][0]["width"]
    img_h = doc["images"][0]["height"]
    seen = set()
    for a in doc["annotations"]:
        assert a["category_id"] in ids
        assert a["id"] not in seen, "duplicate annotation id"
        seen.add(a["id"])
        assert a["image_id"] == doc["images"][0]["id"]
        x, y, bw, bh = a["bbox"]
        assert bw > 0 and bh > 0, a
        assert -1 <= x and -1 <= y, a
        assert x + bw <= img_w + 1 and y + bh <= img_h + 1, a
        assert abs(a["area"] - bw * bh) < 0.5
    # units are recorded so a consumer knows what `value` means
    assert doc["info"]["units"] == r["annotation_record"].units


def test_degenerate_solids_do_not_crash(tmp_path):
    """Shapes with no flat faces or extreme aspect ratios must still render."""
    cases = {
        "sphere": cq.Workplane("XY").sphere(10),
        "wafer": cq.Workplane("XY").box(200, 200, 0.02),
        "long_bar": cq.Workplane("XY").box(500, 2, 2),
    }
    for name, wp in cases.items():
        p = tmp_path / f"{name}.step"
        cq.exporters.export(wp, str(p))
        r = make_drawing(str(p), str(tmp_path / name), )
        drawn = [a for a in r["annotation_record"].annotations
                 if a.kind.value != "feature"]
        assert drawn, f"{name}: produced no annotations at all"


# --------------------------------------------------------------------------- #
# COCO dataset output (the default batch layout)
# --------------------------------------------------------------------------- #
def _manifest_path(root, split="train"):
    """Path to one split's COCO manifest in the Roboflow layout."""
    return os.path.join(str(root), split, "_annotations.coco.json")


def _load_split(root, split="train"):
    with open(_manifest_path(root, split)) as fh:
        return json.load(fh)


def _existing_splits(root):
    """Splits that were actually written. Empty splits get no directory, so
    a tiny corpus may omit a split that hashed to zero images."""
    return [s for s in ("train", "valid", "test")
            if os.path.exists(_manifest_path(root, s))]


def _all_images(root):
    """Every image entry across all splits, as (split, entry) pairs."""
    out = []
    for split in _existing_splits(root):
        for im in json.load(open(_manifest_path(root, split)))["images"]:
            out.append((split, im))
    return out


def _dataset_files(root):
    import glob as _g
    return sorted(os.path.relpath(p, root)
                  for p in _g.glob(os.path.join(root, "**", "*"),
                                   recursive=True) if os.path.isfile(p))


def test_batch_writes_a_roboflow_coco_dataset_by_default(tmp_path):
    """train/ valid/ test/, each holding its own manifest and its images.

    UPDATED: the layout was a central images/ + annotations/ with
    instances_<split>.json. Roboflow's COCO importer expects a directory per
    split, each containing _annotations.coco.json alongside the images it
    lists, so that is what is produced now.
    """
    srcs = [os.path.join(_root(), f"examples/models/{n}.step")
            for n in ("flange", "0000_00000093", "machined_block")]
    srcs = [s for s in srcs if os.path.exists(s)]
    if not srcs:
        pytest.skip("samples not present")
    out = tmp_path / "ds"
    from cli.pipeline import batch
    batch(srcs, str(out))

    files = _dataset_files(str(out))
    assert "README.dataset.txt" not in files, files
    # Default split is 7:2:1. Empty splits are not written, so with only a
    # few parts a folder may be missing -- but every written split is one of
    # the three, and holds its own manifest.
    present = _existing_splits(str(out))
    assert present, "no split written"
    assert set(present) <= {"train", "valid", "test"}, present
    for split in present:
        assert os.path.isdir(str(out / split)), split
        assert os.path.join(split, "_annotations.coco.json") in files, files

    pngs = [f for f in files if f.endswith(".png")]
    assert len(pngs) == len(srcs), pngs
    # every image lives inside a split directory, never at the root
    for png in pngs:
        assert os.path.dirname(png) in ("train", "valid", "test"), png
    # the old central directories are gone
    assert not [f for f in files if f.startswith("images" + os.sep)], files
    assert not [f for f in files
                if f.startswith("annotations" + os.sep)], files
    # no per-part sidecars littering the dataset
    assert not [f for f in files if f.endswith(".annotations.json")], files


def test_dataset_manifest_is_valid_coco(tmp_path):
    """Unique ids, resolvable images, in-bounds boxes, known categories."""
    from PIL import Image
    from cli.pipeline import batch
    srcs = [os.path.join(_root(), f"examples/models/{n}.step")
            for n in ("flange", "0000_00000093", "bracket_plate")]
    srcs = [s for s in srcs if os.path.exists(s)]
    if not srcs:
        pytest.skip("samples not present")
    out = tmp_path / "ds"
    batch(srcs, str(out))

    total_images = total_anns = 0
    for split in _existing_splits(str(out)):
        doc = _load_split(out, split)
        ids = [i["id"] for i in doc["images"]]
        aids = [a["id"] for a in doc["annotations"]]
        assert len(set(ids)) == len(ids), f"{split}: duplicate image ids"
        assert len(set(aids)) == len(aids), f"{split}: duplicate ann ids"
        # each split's manifest stands alone, so its ids start at 1
        assert ids == list(range(1, len(ids) + 1)), f"{split}: ids not 1..n"
        total_images += len(ids)
        total_anns += len(aids)

        known = {c["id"] for c in doc["categories"]}
        by_id = {i["id"]: i for i in doc["images"]}
        for im in doc["images"]:
            # file_name is a bare name resolved against the split directory
            assert os.path.basename(im["file_name"]) == im["file_name"]
            path = out / split / im["file_name"]
            assert path.exists(), f"{split}/{im['file_name']}"
            assert Image.open(path).size == (im["width"], im["height"])
        for a in doc["annotations"]:
            assert a["image_id"] in by_id, "annotation points at no image"
            assert a["category_id"] in known
            im = by_id[a["image_id"]]
            x, y, w, h = a["bbox"]
            assert w > 0 and h > 0, a
            assert -1 <= x and -1 <= y
            assert x + w <= im["width"] + 1 and y + h <= im["height"] + 1
            assert a["attributes"]["units"] == "mm"

    assert total_images == len(srcs)
    assert total_anns, "no objects at all"


def test_train_val_split_partitions_the_dataset(tmp_path):
    """train + val == default, with no overlap and nothing lost."""
    from cli.pipeline import batch
    import glob as _g
    srcs = sorted(_g.glob(os.path.join(_root(), "examples/models/*.step")))
    if len(srcs) < 4:
        pytest.skip("need several samples")
    out = tmp_path / "ds"
    batch(srcs, str(out))

    # Splits partition by FILE NAME now: ids restart at 1 in each manifest,
    # because each one is a standalone COCO file. Empty splits are not
    # written, so only the non-empty splits are compared.
    splits = _existing_splits(str(out))
    names = {s: {i["file_name"] for i in _load_split(out, s)["images"]}
             for s in splits}
    assert names["train"], "training split is empty"
    pairs = list(names)
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            assert not (names[pairs[i]] & names[pairs[j]]), \
                f"image in {pairs[i]} and {pairs[j]}"

    everything = set().union(*names.values())
    assert len(everything) == len(srcs), "an input went missing"
    assert "valid" in names and names["valid"], "validation split is empty"
    # every listed image sits in its own split directory
    for split, listed in names.items():
        on_disk = {f for f in os.listdir(out / split) if f.endswith(".png")}
        assert on_disk == listed, f"{split}: {on_disk} != {listed}"


def test_split_is_stable_when_the_corpus_grows(tmp_path):
    """Adding a model must not move existing models between splits.

    Splitting by index or by a random seed would reshuffle the whole corpus
    whenever a part was added, quietly training on data that used to be held
    out.
    """
    from cli.dataset import DEFAULT_TEST_FRAC, DEFAULT_VAL_FRAC, _split_of
    names = [f"part_{i:03d}" for i in range(200)]
    first = {n: _split_of(n, 0.2, 0.0) for n in names}
    grown = {n: _split_of(n, 0.2, 0.0) for n in names + ["brand_new_part"]}
    assert all(first[n] == grown[n] for n in names)
    # and the split is roughly the requested size
    frac = sum(1 for v in first.values() if v == "valid") / len(first)
    assert 0.1 < frac < 0.3, frac
    # a three-way split honours both fractions -- this is the 7:2:1 default
    three = [_split_of(n, DEFAULT_VAL_FRAC, DEFAULT_TEST_FRAC) for n in names]
    assert 0.1 < three.count("valid") / len(three) < 0.3
    assert 0.05 < three.count("test") / len(three) < 0.2
    assert three.count("train") / len(three) > 0.6


def test_default_split_is_seven_two_one():
    """batch / CLI / build_dataset all default to a 7:2:1 train/valid/test."""
    import inspect
    from cli.cli import _build_parser
    from cli.dataset import (DEFAULT_TEST_FRAC, DEFAULT_VAL_FRAC,
                                   _split_of, build_dataset)
    from cli.pipeline import batch
    assert (DEFAULT_VAL_FRAC, DEFAULT_TEST_FRAC) == (0.2, 0.1)
    for fn in (build_dataset, batch):
        sig = inspect.signature(fn)
        assert sig.parameters["val_frac"].default == DEFAULT_VAL_FRAC
        assert sig.parameters["test_frac"].default == DEFAULT_TEST_FRAC
    ns = _build_parser().parse_args(["x.step"])
    assert ns.val_frac == DEFAULT_VAL_FRAC
    assert ns.test_frac == DEFAULT_TEST_FRAC
    names = [f"part_{i:04d}" for i in range(1000)]
    assigned = [_split_of(n, DEFAULT_VAL_FRAC, DEFAULT_TEST_FRAC) for n in names]
    n = len(assigned)
    assert 0.65 < assigned.count("train") / n < 0.75
    assert 0.15 < assigned.count("valid") / n < 0.25
    assert 0.05 < assigned.count("test") / n < 0.15


# --------------------------------------------------------------------------- #
# tools/coco_view.py -- dataset preview and consistency check
# --------------------------------------------------------------------------- #
def _coco_view():
    import importlib
    sys.path.insert(0, os.path.join(_root(), "tools"))
    mod = importlib.import_module("coco_view")
    importlib.reload(mod)
    return mod


def _mini_dataset(root, doc=None):
    """A tiny valid COCO dataset on disk, for the checker to chew on."""
    from PIL import Image
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "annotations"), exist_ok=True)
    Image.new("RGB", (200, 150), (255, 255, 255)).save(
        os.path.join(root, "images", "a.png"))
    doc = doc or {
        "images": [{"id": 1, "file_name": "a.png",
                    "width": 200, "height": 150}],
        "categories": [{"id": 1, "name": "x"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [10, 10, 50, 40]}],
    }
    with open(os.path.join(root, "annotations",
                           "instances_default.json"), "w") as fh:
        json.dump(doc, fh)
    return doc


def test_coco_view_detects_each_kind_of_corruption(tmp_path):
    """Every check must actually fire; a checker that never fails is useless."""
    cv = _coco_view()
    good = {
        "images": [{"id": 1, "file_name": "a.png",
                    "width": 200, "height": 150}],
        "categories": [{"id": 1, "name": "x"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [10, 10, 50, 40]}],
    }
    root = str(tmp_path / "ds")
    _mini_dataset(root)
    images_dir = os.path.join(root, "images")
    assert cv.check(good, images_dir) == [], "clean dataset flagged"

    def broken(**changes):
        d = json.loads(json.dumps(good))
        d.update(changes)
        return d

    cases = {
        "outside": broken(annotations=[dict(good["annotations"][0],
                                            bbox=[10, 10, 500, 40])]),
        "empty_box": broken(annotations=[dict(good["annotations"][0],
                                              bbox=[10, 10, 0, 40])]),
        "dangling": broken(annotations=[dict(good["annotations"][0],
                                             image_id=99)]),
        "dup_ann": broken(annotations=[good["annotations"][0],
                                       dict(good["annotations"][0],
                                            bbox=[1, 1, 5, 5])]),
        "unknown_cat": broken(annotations=[dict(good["annotations"][0],
                                                category_id=42)]),
        "missing_file": broken(images=[dict(good["images"][0],
                                            file_name="nope.png")]),
        "not_coco": {"hello": 1},
    }
    for name, doc in cases.items():
        assert cv.check(doc, images_dir), f"{name} was not detected"

    # UPDATED: a declared-but-unused category is NO LONGER a problem.
    # With seven feature-specific classes, most parts genuinely carry none of
    # several of them -- a turned disc has no thread and no GD&T -- so an
    # empty class is correct labelling, not corruption. Treating it as an
    # error made --check fail on valid data and would have pressured the
    # labeller into inventing examples to silence it.
    unused = broken(categories=[{"id": 1, "name": "x"},
                                {"id": 2, "name": "never"}])
    assert cv.check(unused, images_dir) == [], cv.check(unused, images_dir)


def test_coco_bbox_covers_the_text_only(tmp_path):
    """The detection target is the number, not the line pointing at it.

    A dimension line spans the whole feature it measures, so boxing text plus
    leader made the target mostly blank paper (the median text box is 7% of
    the full extent, one is 0.8%) and overlapped every neighbouring
    annotation on the same view.
    """
    import numpy as np
    from PIL import Image
    from cli.pipeline import batch
    src = os.path.join(_root(), "examples/models/0000_00000093.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    out = tmp_path / "ds"
    batch([src], str(out), dpi=150)
    present = _existing_splits(str(out))
    split = (s for s in present if _load_split(out, s)["images"])
    split = next(split, None)
    if split is None:
        pytest.skip("no non-empty split")
    doc = _load_split(out, split)
    anns = doc["annotations"]
    assert anns

    # the full extent is still available, and bbox is strictly inside it
    for a in anns:
        ext = a["attributes"]["extent_bbox"]
        assert ext is not None, "extent_bbox dropped"
        bx, by, bw, bh = a["bbox"]
        ex, ey, ew, eh = ext
        assert bx >= ex - 1 and by >= ey - 1
        assert bx + bw <= ex + ew + 1 and by + bh <= ey + eh + 1
        assert bw * bh <= ew * eh + 1

    # and it is genuinely tighter: a text box should be a small part of the
    # extent on a dimension whose line runs across the view
    dim_ids = {c["id"] for c in doc["categories"] if c["name"] == "dimensions"}
    ratios = [(a["bbox"][2] * a["bbox"][3]) /
              max(a["attributes"]["extent_bbox"][2] *
                  a["attributes"]["extent_bbox"][3], 1e-9)
              for a in anns if a["category_id"] in dim_ids]
    assert ratios, "no dimension objects to measure"
    assert sorted(ratios)[len(ratios) // 2] < 0.5, "boxes are not text-tight"

    # a text box must be mostly glyph, not mostly paper
    img = np.asarray(Image.open(out / split /
                                doc["images"][0]["file_name"]).convert("L"))
    density = []
    for a in anns:
        x, y, w, h = (int(round(v)) for v in a["bbox"])
        crop = img[max(0, y):y + max(h, 1), max(0, x):x + max(w, 1)]
        assert crop.size and crop.min() <= 200, f"empty box for {a['id']}"
        density.append((crop < 200).mean())
    assert sorted(density)[len(density) // 2] > 0.08, \
        "boxes contain too little ink to be text boxes"


def test_every_dataset_object_validates_as_feature(tmp_path):
    """Each annotation object must satisfy the public Feature schema.

    The objects are consumed by OpenRouter models through the exported
    :class:`Feature` contract, so any object the writer emits has to round-trip
    ``Feature.model_validate``: integer-pixel ``bbox``, a recognised string
    ``category``, ``text`` and a ``confidence`` in [0, 1]. Extra COCO fields
    are permitted, so the schema tolerates them rather than failing.
    """
    from cli import batch
    from cli.schema import Feature, FeatureList, Category
    src = os.path.join(_root(), "examples/models/0000_00000093.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    out = tmp_path / "ds"
    batch([src], str(out))
    all_anns = []
    for split in _existing_splits(str(out)):
        doc = _load_split(out, split)
        features = FeatureList.model_validate(doc)
        assert len(features) == len(doc["annotations"])
        for obj in doc["annotations"]:
            feature = Feature.model_validate(obj)   # raises if not valid
            assert feature.id == obj["id"]
            assert feature.text == obj["text"]
            assert feature.category == Category(obj["category"])
            assert 0.0 <= feature.confidence <= 1.0
            assert all(isinstance(v, int) and v >= 0 for v in feature.bbox)
            all_anns.append(feature)
    assert all_anns, "no annotations produced for the sample"
    # every exported kind must contribute a recognised category, so a corrupt
    # mapping cannot silently drop or mislabel a class
    assert {f.category for f in all_anns} <= set(Category)


def test_dataset_loads_with_pycocotools(tmp_path):
    """Each split must load with the reference COCO reader.

    Roboflow's importer and most trainers go through pycocotools; a manifest
    that only our own checker accepts is not good enough.
    """
    pycocotools = pytest.importorskip("pycocotools.coco")
    from cli.pipeline import batch
    srcs = [os.path.join(_root(), f"examples/models/{n}.step")
            for n in ("flange", "0000_00000093", "machined_block")]
    srcs = [s for s in srcs if os.path.exists(s)]
    if len(srcs) < 2:
        pytest.skip("samples not present")
    out = tmp_path / "ds"
    batch(srcs, str(out))

    seen = 0
    for split in _existing_splits(str(out)):
        coco = pycocotools.COCO(_manifest_path(out, split))
        for img_id in coco.getImgIds():
            info = coco.loadImgs(img_id)[0]
            # file_name must resolve against the split directory alone
            assert os.path.exists(out / split / info["file_name"])
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
                x, y, w, h = ann["bbox"]
                assert w > 0 and h > 0
            seen += 1
    assert seen == len(srcs)


# --------------------------------------------------------------------------- #
# detection classes: gdnts / roughnesses / radii / chamferes / bores /
# threads / dimensions
# --------------------------------------------------------------------------- #
NEW_CLASSES = {"gdnts", "roughnesses", "radii", "chamferes", "bores",
               "threads", "dimensions", "notes", "datums", "leader_notes",
               "tables", "view_captions"}


def test_coco_categories_are_the_twelve_classes():
    """The label set is exactly the twelve classes, plus Roboflow's root.

    UPDATED: split to twelve. `datums` left `gdnts` (a boxed letter is not a
    feature control frame), `leader_notes` left `notes` (a leader label is not
    a paragraph block), and `tables` and `view_captions` cover ink that was
    drawn on every sheet and never labelled at all.
    """
    from cli.coco import CATEGORIES
    names = [c["name"] for c in CATEGORIES]
    assert names[0] == "annotation" and CATEGORIES[0]["id"] == 0
    assert set(names[1:]) == NEW_CLASSES, names
    # ids are contiguous from 1 so a trainer's class vector has no holes
    assert [c["id"] for c in CATEGORIES] == list(range(len(CATEGORIES)))


def test_gdt_frames_come_from_the_file_and_carry_real_values(tmp_path):
    """GD&T is read from AP242 PMI, with the tolerance magnitudes intact.

    OCCT's XCAF reader returns 0.0 for every tolerance value on this file, so
    the magnitudes are recovered by parsing the raw STEP and joining on the
    feature control frame name. This test fails if that join breaks: a frame
    with no magnitude is not a statable requirement and must not be drawn.
    """
    from cli import pmi as _pmi
    src = os.path.join(_root(), "examples/models/nist_stc_10_asme1_ap242-e2.step")
    if not os.path.exists(src):
        pytest.skip("NIST sample not present")
    p = _pmi.read(src)
    assert len(p.tolerances) >= 30, len(p.tolerances)
    # every frame has a real, non-zero magnitude and an anchor on the solid
    assert all(t.value > 0 for t in p.tolerances)
    assert all(t.anchor is not None for t in p.tolerances)
    # the values really are the ones in the file
    assert {0.2, 0.65, 1.2, 1.5}.issubset({t.value for t in p.tolerances})
    # datum letters are resolved, not invented
    assert {"A", "B", "C"}.issubset({d for t in p.tolerances for d in t.datums})
    chars = {t.characteristic for t in p.tolerances}
    assert {"position", "flatness", "cylindricity"}.issubset(chars), chars


def test_threads_are_never_inferred_from_hole_size():
    """A tap-drill-sized hole is not a thread unless the file says so.

    machined_block has plenty of holes near metric nominal sizes and states
    no thread anywhere; labelling any of them would put a fabricated
    specification into the training data.
    """
    from cli import pmi as _pmi
    from cli.geometry import analyse, load
    src = os.path.join(_root(), "examples/models/machined_block.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    info = analyse(load(src))
    p = _pmi.read(src)
    _pmi.attach_thread_anchors(p, info.holes)
    assert p.threads == []


def test_chamfer_detection_rejects_drill_points():
    """A drill point is a cone but is not a chamfer and is never called out.

    The NIST part's 8 cones are all 118 deg drill points at the bottom of
    blind holes (inner radius 0.000); flange.step's 6 cones are real 82 deg
    countersinks (inner radius 4.500). Dimensioning a drill point would be
    wrong on a drawing, so the two must not be confused.
    """
    from cli.geometry import analyse, load
    nist = os.path.join(_root(), "examples/models/nist_stc_10_asme1_ap242-e2.step")
    if os.path.exists(nist):
        assert analyse(load(nist)).chamfers == []
    src = os.path.join(_root(), "examples/models/flange.step")
    if not os.path.exists(src):
        pytest.skip("flange not present")
    ch = analyse(load(src)).chamfers
    assert len(ch) == 1 and ch[0].count == 6
    assert ch[0].countersink and abs(ch[0].angle - 82.0) < 0.5
    assert "CSK" in ch[0].note()


def test_roughness_is_deterministic_and_flagged_synthetic(tmp_path):
    """Finish is assigned, not measured, so it must be stable AND declared.

    No input file carries surface-texture data, so the Ra value is derived
    from a hash of the part name. Two things matter: the same part must give
    the same drawing on every run (otherwise a dataset cannot be regenerated),
    and every such annotation must be machine-detectable as synthetic so a
    consumer can filter it.
    """
    from cli.roughness import assign, PREFERRED_RA

    class _Info:
        holes = []

    a = assign("flange", _Info())
    b = assign("flange", _Info())
    assert [x.ra for x in a] == [x.ra for x in b]
    assert assign("bracket", _Info())[0].ra in PREFERRED_RA
    # a different part generally gets a different finish
    names = {assign(n, _Info())[0].ra for n in
             ("a", "b", "c", "d", "e", "f", "g", "h")}
    assert len(names) > 1
    assert all(x.synthetic for x in a)

    src = os.path.join(_root(), "examples/models/flange.step")
    if not os.path.exists(src):
        pytest.skip("flange not present")
    r = make_drawing(src, str(tmp_path / "f"), )
    rough = [x for x in r["annotation_record"].annotations
             if x.kind.value == "roughness"]
    assert rough, "no roughness annotation emitted"
    for x in rough:
        assert x.detail.get("synthetic") is True, x.detail


def test_general_roughness_clears_the_views_on_round_plate(tmp_path):
    """The drawing-wide Ra note must not sit on a view footprint."""
    src = os.path.join(_root(), "examples/models/0000_00000007.step")
    if not os.path.exists(src):
        pytest.skip("disc sample not present")
    r = make_drawing(src, str(tmp_path / "disc"), dpi=100)
    rec = json.load(open(r["files"]["annotations"]))
    rough = [a for a in rec["annotations"]
             if a["kind"] == "roughness" and a["detail"].get("general")]
    assert rough
    view_boxes = _view_blocks(rec)

    def overlap(a, b):
        return (min(a[2], b[2]) - max(a[0], b[0]) > 0.25
                and min(a[3], b[3]) - max(a[1], b[1]) > 0.25)

    for note in rough:
        tb = note["text_box"]
        box = (tb["x0"], tb["y0"], tb["x1"], tb["y1"])
        assert not any(overlap(box, vb) for vb in view_boxes.values()), (
            note["text"], box, view_boxes)


def test_annotation_text_boxes_do_not_overlap(tmp_path):
    """Two annotations must not be drawn on top of each other.

    Regression for the GD&T frames: a frame is ~1.6 text-heights tall and
    centred on its leader, but the placer reserved a single line of text
    sitting above it. The footprint was understated by ~60% and frames landed
    on their neighbours -- measured at 2 overlapping pairs totalling 103 mm2
    on the NIST sheet before the fix, 0 after.
    """
    src = os.path.join(_root(), "examples/models/nist_stc_10_asme1_ap242-e2.step")
    if not os.path.exists(src):
        pytest.skip("NIST sample not present")
    r = make_drawing(src, str(tmp_path / "n"), )
    anns = [a for a in r["annotation_record"].annotations if a.text_box]
    bad = []
    for i in range(len(anns)):
        for j in range(i + 1, len(anns)):
            p, q = anns[i].text_box, anns[j].text_box
            ox = min(p.x1, q.x1) - max(p.x0, q.x0)
            oy = min(p.y1, q.y1) - max(p.y0, q.y0)
            if ox > 0 and oy > 0 and ox * oy > 0.5:
                bad.append((anns[i].text[:20], anns[j].text[:20],
                            round(ox * oy, 1)))
    assert not bad, f"overlapping annotation text: {bad}"


def test_gdt_symbols_are_vector_not_font_glyphs():
    """Characteristics are drawn, because the render font lacks the glyphs.

    U+2316 (position) and most other characteristic glyphs are absent from
    DejaVu Sans, so typesetting them would render tofu boxes. Every
    characteristic the PMI reader can return must have vector geometry.
    """
    from cli.gdt_symbols import SYMBOLS, symbol
    from cli.pmi import CHARACTERISTICS
    for char in set(CHARACTERISTICS.values()):
        assert char in SYMBOLS, char
        polys, circles = symbol(char, 0.0, 0.0, 10.0)
        assert polys or circles, char
        for poly in polys:
            for x, y in poly:
                assert -0.1 <= x <= 10.1 and -0.1 <= y <= 10.1, (char, x, y)


def test_synthetic_pmi_never_overrides_real_pmi():
    """The NIST part's genuine PMI is the only ground truth; keep it intact."""
    from cli import pmi as _pmi, synth_pmi as _synth
    from cli.geometry import analyse, load
    src = os.path.join(_root(), "examples/models/nist_stc_10_asme1_ap242-e2.step")
    if not os.path.exists(src):
        pytest.skip("NIST sample not present")
    info = analyse(load(src))
    real = _pmi.read(src)
    n_real = len(real.tolerances)
    out = _synth.generate("nist_stc_10_asme1_ap242-e2", info, real)
    assert len(out.tolerances) == n_real
    assert not any(t.synthetic for t in out.tolerances)


def test_synthetic_items_are_all_flagged(tmp_path):
    """Anything invented must be machine-detectable as invented."""
    src = os.path.join(_root(), "examples/models/machined_block.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    r = make_drawing(src, str(tmp_path / "m"), )
    for a in r["annotation_record"].annotations:
        if a.kind.value in ("gdt", "thread", "chamfer", "roughness"):
            assert a.detail.get("synthetic") is True, (a.kind.value, a.text)


def test_threads_only_on_genuine_tap_drill_bores():
    """A thread is only defensible where the bore really is a tap drill.

    Also excludes counterbored and countersunk holes: those are CLEARANCE
    holes -- the recess takes a screw head that passes through and threads
    into something else -- so tapping one would be a real drafting error.
    machined_block's 6.6 mm holes are counterbored M6 clearance and must not
    be threaded even though 6.6 is close to the M8 tap drill of 6.75.
    """
    from cli import synth_pmi as _synth
    from cli.pmi import PMI
    from cli.geometry import analyse, load
    src = os.path.join(_root(), "examples/models/machined_block.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    info = analyse(load(src))
    out = _synth.generate("machined_block", info, PMI())
    # UPDATED: this part now yields exactly one thread -- its 25 mm BLIND
    # bore, which the through-only size cap no longer rejects. What must stay
    # absent is any thread on its 6.6 mm COUNTERBORED holes, which are M6
    # clearance and are the point of this test.
    assert [t.designation for t in out.threads] == ["M27 x 2-6g"], \
        [t.designation for t in out.threads]
    for t in out.threads:
        assert t.nominal != 8.0, "counterbored 6.6 mm clearance hole tapped"

    # UPDATED: laser_panel is no longer threaded, and that is the correct
    # answer. Its 6.5 mm holes pass through 3 mm of sheet -- 0.46 x diameter
    # of engagement, too few turns to hold. Thin sheet takes a clinch nut or
    # a rivnut, not a tapped thread, so calling it M8 would be wrong on the
    # shop floor. See _MIN_ENGAGEMENT.
    src = os.path.join(_root(), "examples/models/laser_panel.step")
    if os.path.exists(src):
        info = analyse(load(src))
        assert _synth.generate("laser_panel", info, PMI()).threads == []

    # ...but a bore with real depth behind it DOES get one. angled_bracket's
    # 6 mm holes run through 12 mm of material: 2.0 x diameter, ample.
    src = os.path.join(_root(), "examples/models/angled_bracket.step")
    if os.path.exists(src):
        info = analyse(load(src))
        out = _synth.generate("angled_bracket", info, PMI())
        assert out.threads, "no thread on a deep, tap-drill-sized bore"
        t = out.threads[0]
        # the stated nominal must match the bore it sits on: a tap drill is
        # nominal - pitch, so this 6.0 mm bore is an M7 x 1
        assert abs((t.nominal - 1.0) - 6.0) <= 0.3, t.designation


def _thread_eligible_groups():
    """(eligible, total) hole groups across the sample corpus.

    A "group" is a set of identical-diameter closed bores on one part, which
    is the unit a drawing states once ("4X M10 x 1-6H").

    REWRITTEN: this used to re-implement the four eligibility tests inline,
    so it measured its own copy of the rules and PASSED with the real ones
    reverted. It now asks the generator, which is the thing under test.
    """
    from cli.geometry import analyse, load
    from cli.pmi import PMI
    from cli import synth_pmi as _synth
    total = eligible = 0
    for f in sorted(glob.glob(os.path.join(_root(), "examples/models/*.step"))):
        info = analyse(load(f))
        groups = {}
        for h in info.holes:
            if h.is_closed:
                groups.setdefault(round(h.diameter, 3), []).append(h)
        total += len(groups)
        # one spec per distinct thread size the generator finds on this part
        stem = os.path.splitext(os.path.basename(f))[0]
        eligible += len(_synth.generate(stem, info, PMI()).threads)
    return eligible, total


# --------------------------------------------------------------------------- #
# per-drawing style variation
# --------------------------------------------------------------------------- #
def test_variation_is_deterministic_and_actually_varies():
    """Same part -> same style; different parts -> different styles.

    Determinism matters because a dataset must regenerate byte-for-byte;
    variation matters because that is the whole point of the feature.
    """
    from cli.variation import vary_style
    from cli.sheet import Style
    a = vary_style(Style(), "flange")
    b = vary_style(Style(), "flange")
    assert (a.font_family, a.char_w, a.stroke_scale, a.title_block_w) == \
           (b.font_family, b.char_w, b.stroke_scale, b.title_block_w)

    names = ["flange", "machined_block", "bracket_plate", "laser_panel",
             "angled_bracket", "oblique_prism", "0000_00000093",
             "0000_00000559"]
    styles = [vary_style(Style(), n) for n in names]
    assert len({s.font_family for s in styles}) > 1
    assert len({round(s.stroke_scale, 2) for s in styles}) > 1
    assert len({round(s.title_block_w, 1) for s in styles}) > 1

    # and it can be switched off completely
    off = vary_style(Style(), "flange", enabled=False)
    assert off == Style()


def test_char_w_travels_with_the_font():
    """The width factor must match the font, or every box is the wrong size.

    Text width is estimated as len * height * char_w in ~20 places, and that
    estimate is what the exported bounding boxes are built from. Measured
    across the installed families the ratio spans 0.49-0.60, so a fixed
    factor would mislabel any drawing not in the default font.
    """
    from cli.variation import fonts, vary_style
    FONTS = fonts()
    from cli.sheet import Style
    assert len(FONTS) >= 2, FONTS
    for family, ratio in FONTS.items():
        assert 0.35 < ratio < 0.95, (family, ratio)
    for name in ("flange", "machined_block", "laser_panel", "bracket_plate"):
        st = vary_style(Style(), name)
        assert st.char_w == FONTS[st.font_family], st.font_family


def test_variation_never_changes_what_the_drawing_says(tmp_path):
    """Style may vary; values, geometry and view choice may not.

    Variation is augmentation, not corruption -- if it moved a dimension it
    would be falsifying the labels rather than diversifying the pixels.
    """
    src = os.path.join(_root(), "examples/models/machined_block.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    plain = make_drawing(src, str(tmp_path / "a"), vary=False)
    varied = make_drawing(src, str(tmp_path / "b"), )

    # Not an equality test on the annotation SET. A varied sheet has a
    # different amount of free space, so the placer legitimately fits a
    # different number of dimensions -- machined_block gains "10" and "20"
    # when its text is smaller. What must not change is what any given
    # annotation SAYS, so every text that appears in both must carry the
    # same value, and no value may be invented.
    def by_text(rec):
        # Keyed on the VALUE rounded to 0.1 mm, not on the string: a house
        # style states 63.5 where another states 63.456, which is the same
        # size written to that office's precision. What must never differ is
        # the size itself.
        out = {}
        for a in rec["annotation_record"].annotations:
            if a.value is None:
                continue
            out[(a.kind.value, round(a.value, 1))] = (a.kind.value, a.value)
        return out

    pa, va = by_text(plain), by_text(varied)
    shared = set(pa) & set(va)
    assert len(shared) > 5, "the two renderings share almost nothing"
    for text in shared:
        # `feature` entries are internal bookkeeping boxes, not drawn marks.
        # A varied sheet may present its holes as a TABLE, which turns a
        # drawn `bore` callout into a table row plus a feature box -- a
        # presentation change, not a content one. Compare the VALUE, which
        # is what must never differ.
        if "feature" in (pa[text][0], va[text][0]):
            continue
        assert abs(pa[text][1] - va[text][1]) <= 0.05, (text, pa[text], va[text])
    # the same views are chosen: variation restyles, it does not re-plan
    assert [v.name.value for v in plain["annotation_record"].views] == \
           [v.name.value for v in varied["annotation_record"].views]
    # and the part itself is identical
    assert plain["annotation_record"].model_sha256 == \
           varied["annotation_record"].model_sha256


def test_rendering_emits_no_missing_glyph_warnings(tmp_path):
    """End to end: no part may render text its font cannot draw."""
    import warnings
    for name in ("machined_block", "flange", "laser_panel", "bracket_plate"):
        src = os.path.join(_root(), f"examples/models/{name}.step")
        if not os.path.exists(src):
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            make_drawing(src, str(tmp_path / name), dpi=80)
        missing = [str(w.message) for w in caught
                   if "missing from font" in str(w.message)]
        assert not missing, (name, missing[:2])


def test_table_never_collides_with_the_views(tmp_path):
    """Whichever edge it sits on, the table must not land on drawn geometry."""
    import dataclasses
    from cli.sheet import Style
    src = os.path.join(_root(), "examples/models/nist_stc_10_asme1_ap242-e2.step")
    if not os.path.exists(src):
        pytest.skip("NIST sample not present")
    for side in ("right", "top", "bottom"):
        style = dataclasses.replace(Style(), table_side=side)
        r = make_drawing(src, str(tmp_path / side), style=style, vary=False, hole_table=True)
        doc = readsheet(r)
        pts = [(p.x, p.y) for l in doc.modelspace().query("LINE")
               if l.dxf.layer == "TABLE"
               for p in (l.dxf.start, l.dxf.end)]
        assert pts, side
        x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
        x1, y1 = max(p[0] for p in pts), max(p[1] for p in pts)
        for e in doc.modelspace().query("TEXT"):
            if e.dxf.layer != "DIM":
                continue
            b = _ptext_box(e.prim)
            assert (b[2] < x0 or b[0] > x1 or b[3] < y0 or b[1] > y1), \
                f"{side}: dimension {e.dxf.text!r} sits inside the table"


def test_title_block_content_is_generated_and_flagged(tmp_path):
    """Drafter, checker, revision etc. are invented -- and say so.

    A CAD file records geometry, not who signed the drawing. The fields are
    generated so the block carries realistic variety, and the whole invented
    subset must stay identifiable.
    """
    from cli.titleblock import generate, _NAMES
    from cli.variation import _Rng, _seed
    fields, order = generate("demo", _Rng(_seed("titleblock:demo")),
                             {"TITLE": "DEMO", "PART NUMBER": "demo"}, 12)
    assert fields["drawn"] in _NAMES
    # a checker is never the same person as the drafter
    if "CHECKED" in fields:
        assert fields["CHECKED"] != fields["drawn"]
    # the real fields are passed through untouched
    assert fields["TITLE"] == "DEMO" and fields["PART NUMBER"] == "demo"
    # more slots means more fields, not the same ones spaced out
    few, few_order = generate("demo", _Rng(_seed("titleblock:demo")),
                              {"TITLE": "DEMO"}, 8)
    assert len(few_order) < len(order)


def test_thin_strokes_stay_legible(tmp_path):
    """A light drawing must still render solid ink, not pale grey.

    Stroke variation goes down to x0.71, which put the thinnest layer at
    0.284 pt -- under one device pixel, so it anti-aliased to grey. Measured
    on the lightest sheet, 44% of ink pixels were faint rather than solid
    (0.78 faint per solid, against 0.17 on the heaviest). A per-layer floor
    fixes it without flattening the weight hierarchy.
    """
    import dataclasses
    import numpy as np
    from PIL import Image
    from cli.sheet import Style
    src = os.path.join(_root(), "examples/models/machined_block.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    ratios = {}
    for scale in (0.71, 1.81):
        style = dataclasses.replace(Style(), stroke_scale=scale)
        r = make_drawing(src, str(tmp_path / f"s{scale}"), dpi=150, style=style, vary=False)
        a = np.asarray(Image.open(r["files"]["png"]).convert("L"))
        faint = int(((a >= 80) & (a < 200)).sum())
        solid = int((a < 80).sum())
        assert solid > 0
        ratios[scale] = faint / solid
    # the light sheet must be comparable to the heavy one, not 4x worse
    assert ratios[0.71] < 0.45, ratios
    assert ratios[0.71] < ratios[1.81] * 2.5, ratios


def test_uploaded_model_callout_stays_inside_the_sheet_frame(tmp_path):
    """The varied portrait reproduction keeps its long bolt-circle callout in frame.

    This is the uploaded STEP regression: its annotation pass and final view
    scale used to disagree, so the second line of the callout was emitted past
    the right border. Check the finished paper primitives, not only the model
    coordinates or the serialized label boxes.
    """
    from cli.pipeline import _sheet_ink_overflow

    src = os.path.join(_root(), "examples/models/0002_00022071.step")
    if not os.path.exists(src):
        pytest.skip("uploaded sample not present")
    r = make_drawing(src, str(tmp_path / "uploaded"),
                     vary=True, write_record=False, coco=False, dpi=50)
    assert _sheet_ink_overflow(r["psheet"]) <= 0.01


def test_uploaded_model_feature_notes_clear_the_front_view(tmp_path):
    """Thread and finish notes must not be printed on top of the small view."""
    src = os.path.join(_root(), "examples/models/0002_00022071.step")
    if not os.path.exists(src):
        pytest.skip("uploaded sample not present")
    r = make_drawing(src, str(tmp_path / "uploaded_notes"),
                     vary=True, write_record=False, coco=False, dpi=50)
    sh = r["psheet"]
    pl = sh.placed_views["front"]
    v = pl.ann.view
    view_box = (pl.m2p((v.xmin, v.ymin))[0], pl.m2p((v.xmin, v.ymin))[1],
                pl.m2p((v.xmax, v.ymax))[0], pl.m2p((v.xmax, v.ymax))[1])
    for a in sh.annotations:
        if a.get("view") != "front" or a.get("kind") not in ("thread", "roughness"):
            continue
        b = a.get("text_box")
        assert b is not None
        assert (b[2] <= view_box[0] or b[0] >= view_box[2]
                or b[3] <= view_box[1] or b[1] >= view_box[3]), a


def test_uploaded_model_diameters_use_bore_class_and_iso_caption(tmp_path):
    """Hole diameters use the bore class; overall size remains a dimension."""
    src = os.path.join(_root(), "examples/models/0002_00022071.step")
    if not os.path.exists(src):
        pytest.skip("uploaded sample not present")
    r = make_drawing(src, str(tmp_path / "uploaded_taxonomy"),
                     vary=True, write_record=False, coco=True, dpi=50)
    anns = r["annotation_record"].annotations
    diameters = [a for a in anns if a.detail.get("form") == "diameter"]
    assert diameters
    assert any(a.kind.value == "bore" for a in diameters), diameters
    assert all(a.kind.value in {"bore", "dimension"} for a in diameters), diameters
    iso = [a for a in anns if a.view == "iso" and a.text == "ISO"]
    assert iso and all(a.kind.value == "view_caption" for a in iso)
    # The concentric circular view is already located by its diameter and bolt
    # circle. It must not grow the rejected stack of redundant 6 mm X/Y dims;
    # its single overall diameter is intentionally a size dimension.
    assert not [a for a in anns if a.view == "front"
                and a.kind.value == "dimension"
                and a.detail.get("form") != "diameter"], [a.text for a in anns]
    front_view = next(v for v in r["annotation_record"].views
                      if v.name.value == "front")
    assert front_view.width * r["annotation_record"].sheet.scale >= 24.0
    coco = json.load(open(r["files"]["coco"]))
    dia_objects = [o for o in coco["annotations"]
                   if o["attributes"].get("detail", {}).get("form") == "diameter"]
    assert dia_objects
    assert any(o["category_id"] == 5 and o["category"] == "bores"
               for o in dia_objects)
    assert all(o["category_id"] in {5, 7}
               and o["category"] in {"bores", "dimensions"}
               for o in dia_objects)
    assert next(c for c in coco["categories"] if c["id"] == 3)["name"] == "radii"
    from cli.schema import FeatureList
    parsed = FeatureList.model_validate(coco)
    assert len(parsed) == len(coco["annotations"])
    assert all(isinstance(parsed[i].is_artificial, bool)
               for i in range(len(parsed)))


def test_model_0000_00000061_views_are_clear_of_frame_and_furniture(tmp_path):
    """The rolled-back placement keeps both views on the drawing paper."""
    from cli.pipeline import _sheet_ink_overflow
    from cli.render import _prim_box
    from cli.sheet import PPoly

    src = os.path.join(_root(), "examples/models/0000_00000061.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    r = make_drawing(src, str(tmp_path / "model_61"),
                     vary=True, write_record=False, coco=False, dpi=50)
    sh = r["psheet"]
    frames = [_prim_box(p) for p in sh.prims
              if getattr(p, "layer", "") == "FRAME" and isinstance(p, PPoly)]
    border = max((b for b in frames if b),
                 key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    def inside(box):
        return (box[0] >= border[0] and box[1] >= border[1]
                and box[2] <= border[2] and box[3] <= border[3])

    view_boxes = {}
    for name, placed in sh.placed_views.items():
        view = placed.ann.view
        pl, pr, pb, pt = placed.ann.pad
        a = placed.m2p((view.xmin - pl, view.ymin - pb))
        b = placed.m2p((view.xmax + pr, view.ymax + pt))
        view_boxes[name] = (min(a[0], b[0]), min(a[1], b[1]),
                            max(a[0], b[0]), max(a[1], b[1]))

    assert {"top", "front"} <= view_boxes.keys()
    assert all(inside(box) for box in view_boxes.values())
    assert _sheet_ink_overflow(sh) <= 0.01

    def overlaps(a, b):
        return (a[0] < b[2] and a[2] > b[0]
                and a[1] < b[3] and a[3] > b[1])

    furniture = [a.get("extent") or a.get("text_box")
                 for a in sh.annotations
                 if not a.get("view")
                 and a.get("kind") in {"table", "note", "roughness"}]
    assert not [(name, box, item) for name, box in view_boxes.items()
                for item in furniture if item and overlaps(box, item)]


def test_no_ink_runs_outside_the_sheet_frame(tmp_path):
    """Nothing drawn may cross the border frame, on any sample part.

    User report: "for many drawings, bounding boxes are outside of their
    bounds", naming 0000_00000126. The recorded boxes were faithful -- the
    INK was off the page. A general note is free text of arbitrary length and
    was drawn as one unwrapped line from the title block's left edge, so
    0000_00000126's 93-character two-bodies note ran 56.15 mm past the right
    border of its A3 sheet and off the image entirely, taking its detection
    box with it.

    Measured over all 24 sample parts, worst overflow past the frame:
    56.15 mm before, 0.00 mm after. This asserts the property directly on
    every drawn primitive, not just on the one part that happened to fail.
    """
    import glob as _glob
    srcs = sorted(_glob.glob(os.path.join(_root(), "examples/models/*.step")))
    if len(srcs) < 8:
        pytest.skip("samples not present")

    worst, where = 0.0, ""
    for src in srcs:
        name = os.path.splitext(os.path.basename(src))[0]
        r = make_drawing(src, str(tmp_path / name), )
        msp = readsheet(r).modelspace()
        xs, ys = [], []
        for e in msp.query('LWPOLYLINE[layer=="FRAME"]'):
            for q in e.get_points("xy"):
                xs.append(q[0]); ys.append(q[1])
        assert xs, f"{name}: no border frame drawn"
        fx0, fy0, fx1, fy1 = min(xs), min(ys), max(xs), max(ys)

        boxes = []
        for e in msp:
            if e.dxf.layer == "FRAME":
                continue
            t = e.dxftype()
            if t == "TEXT":
                boxes.append((_ptext_box(e.prim), e.dxf.text))
            elif t == "LINE":
                a, b = e.dxf.start, e.dxf.end
                boxes.append(((min(a.x, b.x), min(a.y, b.y),
                               max(a.x, b.x), max(a.y, b.y)), "LINE"))
            elif t == "LWPOLYLINE":
                p = [(q[0], q[1]) for q in e.get_points("xy")]
                boxes.append(((min(q[0] for q in p), min(q[1] for q in p),
                               max(q[0] for q in p), max(q[1] for q in p)),
                              "POLY"))
            elif t == "CIRCLE":
                c, rr = e.dxf.center, e.dxf.radius
                boxes.append(((c.x - rr, c.y - rr, c.x + rr, c.y + rr),
                              "CIRCLE"))
        for (x0, y0, x1, y1), what in boxes:
            ov = max(max(0.0, fx0 - x0), max(0.0, fy0 - y0),
                     max(0.0, x1 - fx1), max(0.0, y1 - fy1))
            if ov > worst:
                worst, where = ov, f"{name}: {what[:60]!r}"
    assert worst < 0.5, f"ink {worst:.2f} mm outside the frame -- {where}"


def test_long_general_notes_are_wrapped_not_run_off_the_page():
    """A note longer than the paper is wrapped, and wrapping stays on paper.

    Unit-level guard on sheet.note_lines so the property is pinned even if no
    sample part happens to carry a long note. Continuation lines are indented
    past the "N. " numbering.
    """
    from cli.sheet import note_lines
    long_note = ("2 SEPARATE BODIES IN THIS FILE - OVERALL DIMENSIONS ARE "
                 "THE ENVELOPE; EACH BODY IS SIZED SEPARATELY")
    # 105 mm is the narrowest title block autodraft.variation produces, and
    # the note is 100 characters, so this must wrap.
    width, t, cw = 105.0, 2.8 * 0.9, 0.5697
    lines = note_lines([long_note], width, t, cw)
    assert len(lines) > 1, "a 100-char note must wrap on a 105 mm block"
    for ln in lines:
        assert len(ln) * t * cw <= width + 1e-6, f"line too wide: {ln!r}"
    assert lines[0].startswith("1. ")
    assert lines[1].startswith("   "), "continuation must clear the number"
    # every word survives the wrap, in order
    assert " ".join(" ".join(lines).split())[3:] == long_note

    # a short note is left exactly as it was
    assert note_lines(["FINISH ALL OVER"], width, t, cw) == \
        ["1. FINISH ALL OVER"]


def test_exported_boxes_are_tight_around_their_ink(tmp_path):
    """A detection box must contain the glyphs and little else.

    User report: the box for "BODY <n>: <w> X <h>" is not tight, on
    0000_00000126. Two errors compounded, each worth about half:

    1. `Style.char_w` is the MEAN width per character over a sample alphabet,
       plus 4% deliberate headroom, because its real job is to RESERVE space
       during placement. `BODY 1: 0.89 X 0.67` is almost entirely narrow
       glyphs -- digits, dots, spaces, a colon -- so the mean over-predicted
       it by 2.96 mm.
    2. The exporter draws at `fontsize = h * 2.6` points, an em of
       `h * 2.6 * 25.4 / 72 = 0.9172 * h` mm, while char_w is measured per em
       and the box maths assumed an em of exactly `h`. Every string was drawn
       8.3% smaller than its box believed: another 2.90 mm.

    Exported boxes now measure the string's real ink (autodraft.textmetrics);
    placement keeps the loose estimate, since shrinking a reservation makes
    labels collide. Measured over the 24 sample parts, horizontal slack:
    mean 1.15 -> 0.29 mm, max 12.07 -> 1.52 mm.

    This reads the rendered pixels, so it fails if the box drifts off the ink
    in either direction.
    """
    import numpy as np
    from PIL import Image
    src = os.path.join(_root(), "examples/models/0000_00000126.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    r = make_drawing(src, str(tmp_path / "d"), dpi=200)
    doc = json.load(open(r["files"]["coco"]))
    g = np.asarray(Image.open(r["files"]["png"]).convert("L")) < 128
    H, W = g.shape
    mm_per_px = 420.0 / W               # this part lands on A3 landscape

    worst, where = 0.0, ""
    checked = 0
    for a in doc["annotations"]:
        txt = a.get("attributes", {}).get("text", "")
        if not txt.startswith("BODY "):
            continue
        x, y, w, h = a["bbox"]
        x0, y0 = int(max(0, x)), int(max(0, y))
        x1, y1 = int(min(W, x + w)), int(min(H, y + h))
        sub = g[y0:y1, x0:x1]
        assert sub.any(), f"{txt!r}: box contains no ink at all"
        checked += 1
        ys, xs = np.where(sub)
        slack = max(xs.min(), sub.shape[1] - 1 - xs.max()) * mm_per_px
        if slack > worst:
            worst, where = slack, txt
    assert checked >= 2, "expected the two BODY notes on this part"
    assert worst < 1.0, f"{where!r}: {worst:.2f} mm of blank paper in its box"


def test_boxes_contain_their_glyphs_for_every_alignment():
    """A recorded box must CONTAIN its string, whatever ha/va it is drawn with.

    User report, with a screenshot: the boxes sit below and short of the text
    on 0000_00000126, 0000_00000102, 0000_00000073 "and almost all other
    drawings".

    Root cause: `ha`/`va` do not align the ink, and _text_box assumed they
    did. matplotlib's `va="bottom"` aligns the DESCENT LINE, so every box sat
    a full font descent -- 0.76 mm at h=3.5 -- below its glyphs; `ha="center"`
    and `"right"` align the ADVANCE width, side bearings included, not the
    ink. Deriving the correction from FT2Font ascender/descender was tried and
    left a 0.5-0.6 mm residual, because matplotlib's laid-out line is not the
    face's ascent-to-descent span (0.9282 em vs ~1.06 em for DejaVu Sans).
    The box now comes from matplotlib's own layout box, with the baseline
    located empirically per face.

    This renders each string in isolation at high dpi and checks the pixels,
    across every ha x va combination and all three fonts the corpus uses, so
    it cannot be satisfied by a box that merely overlaps its text.
    """
    import numpy as np
    from PIL import Image
    from cli.sheet import PSheet, PText
    from cli.render import _text_box
    from cli.export import to_png
    from cli.variation import fonts

    fams = [f for f in ("DejaVu Sans", "DejaVu Sans Mono", "STIXGeneral")
            if f in fonts()]
    if not fams:
        pytest.skip("no measurable font available")

    worst, where = 0.0, ""
    for fam in fams:
        for s, h in (("Hxq", 4.0), ("BODY 1: 0.89 X 0.67", 3.5),
                     ("gjpqy", 3.0), ("NOTES:", 3.2)):
            for ha in ("left", "center", "right"):
                for va in ("bottom", "center", "top"):
                    sh = PSheet(100.0, 34.0)
                    sh.char_w = 0.57
                    sh.font_family = fam
                    t = PText((50.0, 17.0), s, h, ha=ha, va=va, layer="DIM")
                    sh.add(t)
                    p = "/tmp/_align_probe.png"
                    to_png(sh, p, dpi=600)
                    g = np.asarray(Image.open(p).convert("L")) < 128
                    assert g.any(), f"{fam} {s!r}: nothing rendered"
                    ys, xs = np.where(g)
                    mmpp = 100.0 / g.shape[1]
                    ink = (xs.min() * mmpp, 34.0 - (ys.max() + 1) * mmpp,
                           (xs.max() + 1) * mmpp, 34.0 - ys.min() * mmpp)
                    b = _text_box(t, grow=0.0, tight=True)
                    out = max(b[0] - ink[0], b[1] - ink[1],
                              ink[2] - b[2], ink[3] - b[3])
                    if out > worst:
                        worst, where = out, f"{fam} {s!r} ha={ha} va={va}"
    # 0.15 mm is sub-pixel at drawing scale: it is the antialiasing threshold,
    # not a placement error. The defect being guarded was 0.76 mm and
    # systematic.
    assert worst < 0.15, f"{where}: ink {worst:.3f} mm outside its box"


# --------------------------------------------------------------------------- #
# outputs, records and the isometric  (rewritten for the PNG/COCO-only pipeline)
# --------------------------------------------------------------------------- #
def test_make_drawing_outputs(block, tmp_path):
    """A drawing is a PNG plus its labels -- nothing else is written."""
    r = make_drawing(block, str(tmp_path / "block"))
    assert os.path.getsize(r["files"]["png"]) > 1000
    assert set(r["files"]) == {"png", "annotations", "coco"}
    assert r["hole_count"] == 5
    doc = json.load(open(r["files"]["coco"]))
    assert doc["annotations"] and doc["images"][0]["file_name"] == "block.png"


def test_annotation_record_round_trips_json(block, tmp_path):
    """The record must survive a JSON round-trip, computed fields included."""
    from cli import AnnotationRecord
    r = make_drawing(block, str(tmp_path / "block"))
    rec = r["annotation_record"]
    back = AnnotationRecord.model_validate_json(rec.model_dump_json())
    assert back.annotations == rec.annotations
    assert back.annotation_counts == rec.annotation_counts
    assert back.total_annotations == len(rec.annotations)
    assert back.model_sha256 and len(back.model_sha256) == 64
    on_disk = AnnotationRecord.model_validate_json(
        open(r["files"]["annotations"]).read())
    assert on_disk.total_annotations == rec.total_annotations


def test_view_paper_origin_is_recorded(tmp_path):
    """Each view must record the transform needed to map boxes to the sheet."""
    p = tmp_path / "b.step"
    w = (cq.Workplane("XY").box(90, 60, 30).faces(">Z").workplane()
         .pushPoints([(-30, -18), (30, -18)]).hole(6))
    cq.exporters.export(w, str(p))
    r = make_drawing(str(p), str(tmp_path / "d"))
    arec = r["annotation_record"]
    # feature boxes live in the annotation list, not on ViewRecord
    feats = [a for a in arec.annotations if a.kind.value == "feature"]
    assert feats, "no feature annotations recorded"
    named = {a.view for a in feats}
    for v in arec.views:
        if v.name.value in named:
            assert v.paper_origin is not None, f"{v.name} has no paper origin"
        # the model origin is inside the part, so xmin must be negative
        assert v.xmin < 0, f"xmin looks unset: {v.xmin}"


def test_isometric_never_overlaps_a_view(tmp_path):
    """The pictorial view must not be drawn on top of a dimensioned view.

    Regression: with only two views in one column the projection grid has no
    spare cell, and the old code fell back to "use the top-right corner"
    without checking whether that corner was free.
    """
    cases = {
        "plate2v": (cq.Workplane("XY").box(40, 40, 2)
                    .faces(">Z").workplane().hole(4)),
        "block3v": (cq.Workplane("XY").box(90, 60, 30)
                    .faces(">Z").workplane()
                    .pushPoints([(-30, -18), (30, 18)]).hole(6)
                    .faces(">X").workplane(centerOption="CenterOfBoundBox")
                    .hole(8)),
        "wide":    (cq.Workplane("XY").box(200, 30, 8)
                    .faces(">Z").workplane().rarray(40, 1, 4, 1).hole(5)),
    }
    for name, wp in cases.items():
        f = tmp_path / f"{name}.step"
        cq.exporters.export(wp, str(f))
        r = make_drawing(str(f), str(tmp_path / name))
        box = r.get("iso_box")
        if not box:
            continue                     # skipped on purpose: nowhere clear
        rec = json.loads(r["annotation_record"].model_dump_json())
        for vname, (x0, y0, x1, y1) in _view_blocks(rec).items():
            ox = min(box[2], x1) - max(box[0], x0)
            oy = min(box[3], y1) - max(box[1], y0)
            assert not (ox > 0.5 and oy > 0.5), (
                f"{name}: ISO overlaps the {vname} view by "
                f"{ox:.1f}x{oy:.1f} mm")


def test_text_carries_the_font_width_factor(tmp_path):
    """Every string must record the character width of the font it is drawn in.

    The width factor is what lets a checker recover the true text extent
    instead of assuming the default ratio -- and the exported label boxes are
    measured with it.
    """
    from cli.variation import vary_style
    from cli.sheet import Style
    src = os.path.join(_root(), "examples/models/laser_panel.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    st = vary_style(Style(), "laser_panel")
    r = make_drawing(src, str(tmp_path / "d"))
    texts = list(readsheet(r).modelspace().query("TEXT"))
    assert texts
    expected = round(st.char_w / 0.62, 4)
    assert abs(expected - 1.0) > 1e-6, \
        "this part must pick a non-default font for the test to mean anything"
    # a bold string is legitimately wider, so it encodes char_w * the bold
    # factor; both values are correct, anything else is not
    bold_expected = round(st.char_w * 1.06 / 0.62, 4)
    for e in texts:
        got = float(e.dxf.width)
        assert (abs(got - expected) < 1e-3
                or abs(got - bold_expected) < 1e-3), \
            f"width {got} encodes neither char_w {st.char_w} nor its bold form"


def test_every_layer_inks_black(tmp_path):
    """Layer separation must be carried by weight and dash, never by colour."""
    from cli.export import LAYER_STYLE
    for name, (col, _lw, _dash) in LAYER_STYLE.items():
        assert col == "#000000", f"layer {name} is {col}"
    assert LAYER_STYLE["HIDDEN"][2] and LAYER_STYLE["CENTER"][2], \
        "hidden and centre lines must stay distinguishable by dash pattern"
    # the only non-black ink on a sheet is the isometric's grey shading
    from cli.sheet import PFace
    src = os.path.join(_root(), "examples/models/flange.step")
    if not os.path.exists(src):
        pytest.skip("flange not present")
    r = make_drawing(src, str(tmp_path / "f"))
    for p in r["psheet"].prims:
        if isinstance(p, PFace):
            assert 0.0 <= p.shade <= 1.0, p.shade


# --------------------------------------------------------------------------- #
# house styles: whole drawing-office conventions, not just the font
# --------------------------------------------------------------------------- #
def _styled(name):
    from cli.sheet import Style
    from cli.variation import vary_style
    return vary_style(Style(), name)


def test_house_styles_spread_across_the_corpus():
    """The corpus must use several offices' conventions, not one template."""
    from cli.styles import HOUSE_STYLES
    names = [os.path.splitext(os.path.basename(f))[0]
             for f in sorted(glob.glob(os.path.join(
                 _root(), "examples/models/*.step")))]
    if len(names) < 8:
        pytest.skip("sample corpus not present")
    picked = {_styled(n).house_style for n in names}
    assert len(picked) >= 4, picked
    assert picked <= set(HOUSE_STYLES), picked
    # and a style is a BUNDLE: the conventions really differ between them
    axes = {(s.frame_style, s.arrow_style, s.dim_text_mode, s.tolerance_style,
             s.notes_style, s.qty_prefix, s.view_label_style)
            for s in (_styled(n) for n in names)}
    assert len(axes) >= 4, axes


def test_style_is_deterministic_from_the_part_name():
    a, b = _styled("flange"), _styled("flange")
    assert a.house_style == b.house_style
    assert a.arrow_style == b.arrow_style and a.decimals == b.decimals


def test_zoned_border_carries_zone_labels(tmp_path):
    """A zoned sheet gets the ruled band with numbers and letters."""
    from dataclasses import replace
    from cli.sheet import Style
    from cli.styles import HOUSE_STYLES
    p = tmp_path / "b.step"
    cq.exporters.export(cq.Workplane("XY").box(60, 40, 10), str(p))
    st = replace(Style(), **HOUSE_STYLES["iso_office"])
    r = make_drawing(str(p), str(tmp_path / "d"), style=st, vary=False)
    texts = [e.dxf.text for e in readsheet(r).modelspace().query("TEXT")]
    assert texts.count("1") >= 2, "zone numbers missing along top and bottom"
    assert texts.count("A") >= 2, "zone letters missing down the sides"
    # ...and a plain border has none of them
    st2 = replace(Style(), **HOUSE_STYLES["minimal_cad"])
    r2 = make_drawing(str(p), str(tmp_path / "e"), style=st2, vary=False)
    t2 = [e.dxf.text for e in readsheet(r2).modelspace().query("TEXT")]
    assert t2.count("1") == 0 and t2.count("A") == 0, t2


def test_nothing_is_drawn_over_the_zone_band(tmp_path):
    """The title block and notes sit inside the zone strip, not across it."""
    from dataclasses import replace
    from cli.sheet import Style
    from cli.styles import HOUSE_STYLES
    p = tmp_path / "b.step"
    cq.exporters.export(cq.Workplane("XY").box(80, 50, 12)
                        .faces(">Z").workplane().hole(10), str(p))
    st = replace(Style(), **HOUSE_STYLES["iso_office"])
    r = make_drawing(str(p), str(tmp_path / "d"), style=st, vary=False)
    sh = r["psheet"]
    margin = st.sheet_margin
    band = min(margin * 0.62, 6.0)
    inner = (margin + band, margin + band, sh.w - margin - band,
             sh.h - margin - band)
    from cli.sheet import PText
    for t in sh.prims:
        if not isinstance(t, PText) or t.layer != "TEXT":
            continue
        if len(t.s) <= 1:
            continue                       # the zone labels themselves
        assert inner[0] - 0.5 <= t.p[0] <= inner[2] + 0.5, (t.s, t.p)


def test_tolerances_are_stated_and_plausible(tmp_path):
    """A toleranced office prints tolerances, and they are real magnitudes."""
    import re as _re
    from dataclasses import replace
    from cli.sheet import Style
    from cli.styles import HOUSE_STYLES
    p = tmp_path / "b.step"
    cq.exporters.export(cq.Workplane("XY").box(90, 60, 20)
                        .faces(">Z").workplane()
                        .pushPoints([(-30, -18), (30, 18)]).hole(8), str(p))
    st = replace(Style(), **dict(HOUSE_STYLES["iso_office"],
                                 tolerance_fraction=1.0))
    r = make_drawing(str(p), str(tmp_path / "d"), style=st, vary=False)
    dims = [a for a in r["annotation_record"].annotations
            if a.kind.value == "dimension"]
    toleranced = [a for a in dims if "\u00b1" in a.text]
    assert toleranced, [a.text for a in dims]
    for a in toleranced:
        mag = float(_re.search(r"\u00b1([0-9.]+)", a.text).group(1))
        assert 0 < mag <= max(0.5, (a.value or 1) * 0.1 + 1e-9), a.text
        # the recorded value is still the nominal, not the tolerance. A size
        # that repeats a chain is marked as reference -- "(90)" or "90 REF"
        # (styles.reference_text) -- so the marking comes off before the
        # number is read.
        nominal = a.text.split()[0].strip("()")
        assert a.value and abs(a.value - float(nominal)) < 1e-6


def test_limit_dimensions_are_a_max_over_a_min(tmp_path):
    """Limits print as high/low, and a limit is never negative."""
    from dataclasses import replace
    from cli.sheet import Style
    from cli.styles import HOUSE_STYLES
    p = tmp_path / "b.step"
    cq.exporters.export(cq.Workplane("XY").box(40, 25, 6), str(p))
    st = replace(Style(), **dict(HOUSE_STYLES["vintage_blueprint"],
                                 tolerance_fraction=1.0))
    r = make_drawing(str(p), str(tmp_path / "d"), style=st, vary=False)
    limits = [a.text for a in r["annotation_record"].annotations
              if a.kind.value == "dimension" and "/" in a.text]
    assert limits, "no limit dimensions on a limits-style sheet"
    for text in limits:
        hi, lo = (float(x) for x in text.split("/"))
        assert hi > lo > 0, text


def test_sloping_edges_are_dimensioned_along_themselves():
    """An oblique edge gets an aligned dimension stating its true length."""
    src = os.path.join(_root(), "examples/models/oblique_prism.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    r = make_drawing(src, "/tmp/_aligned/oblique")
    aligned = [a for a in r["annotation_record"].annotations
               if (a.detail or {}).get("form") == "aligned"]
    assert aligned, "no aligned dimension on a part made of sloping faces"
    for a in aligned:
        # the number states the edge's true length, not its projection
        assert a.value and a.value > 0
        assert abs(_nominal(a.text) - a.value) < 0.05


def test_arrowheads_follow_the_house_style(tmp_path):
    """filled / open / tick terminators all reach the sheet."""
    from dataclasses import replace
    from cli.sheet import PArrow, Style
    from cli.styles import HOUSE_STYLES
    p = tmp_path / "b.step"
    cq.exporters.export(cq.Workplane("XY").box(60, 40, 10), str(p))
    seen = {}
    for house in ("iso_office", "minimal_cad", "vintage_blueprint"):
        st = replace(Style(), **HOUSE_STYLES[house])
        r = make_drawing(str(p), str(tmp_path / house), style=st, vary=False)
        kinds = {a.kind for a in r["psheet"].prims if isinstance(a, PArrow)}
        assert kinds, house
        seen[house] = kinds
    assert "filled" in seen["iso_office"]
    assert "open" in seen["minimal_cad"]
    assert "tick" in seen["vintage_blueprint"]


def test_inline_numbers_break_their_dimension_line(tmp_path):
    """A number written in the line needs a gap in it, not ink behind it."""
    from dataclasses import replace
    from cli.sheet import PLine, PText, Style
    from cli.styles import HOUSE_STYLES
    p = tmp_path / "b.step"
    cq.exporters.export(cq.Workplane("XY").box(90, 50, 10), str(p))
    st = replace(Style(), **HOUSE_STYLES["asme_inch"])
    r = make_drawing(str(p), str(tmp_path / "d"), style=st, vary=False)
    sh = r["psheet"]
    dim_texts = [t for t in sh.prims
                 if isinstance(t, PText) and t.layer == "DIM"]
    assert dim_texts
    lines = [l for l in sh.prims if isinstance(l, PLine) and l.layer == "DIM"]
    for t in dim_texts:
        cx, cy = t.p[0], t.p[1] + t.h * 0.5
        for L in lines:
            if abs(L.a[1] - L.b[1]) > 1e-6:
                continue                      # only horizontal dimension lines
            if abs(L.a[1] - cy) > t.h * 0.6:
                continue
            x0, x1 = sorted((L.a[0], L.b[0]))
            assert not (x0 + 0.2 < cx < x1 - 0.2), \
                f"dimension line runs through {t.s!r}"


def test_unless_otherwise_specified_block_states_the_general_tolerance(tmp_path):
    from dataclasses import replace
    from cli.sheet import Style
    from cli.styles import HOUSE_STYLES
    p = tmp_path / "b.step"
    cq.exporters.export(cq.Workplane("XY").box(50, 30, 8), str(p))
    st = replace(Style(), **HOUSE_STYLES["asme_inch"])
    r = make_drawing(str(p), str(tmp_path / "d"), style=st, vary=False)
    texts = " ".join(e.dxf.text for e in readsheet(r).modelspace().query("TEXT"))
    assert "UNLESS OTHERWISE SPECIFIED" in texts
    # Two conventions for the block: an ISO general-tolerance CLASS, or an
    # ASME table by DECIMAL PLACES. asme_inch writes the second one.
    assert ("GENERAL TOLERANCE" in texts and "ANGULAR" in texts) or \
           ("TOLERANCES UNLESS NOTED" in texts and "ANGLES" in texts), texts
    # a minimal office prints no note block at all
    st2 = replace(Style(), **HOUSE_STYLES["minimal_cad"])
    r2 = make_drawing(str(p), str(tmp_path / "e"), style=st2, vary=False)
    t2 = " ".join(e.dxf.text for e in readsheet(r2).modelspace().query("TEXT"))
    assert "NOTES:" not in t2 and "UNLESS OTHERWISE" not in t2


def test_view_captions_and_count_offs_follow_the_house_style(tmp_path):
    from dataclasses import replace
    from cli.sheet import Style
    from cli.styles import HOUSE_STYLES
    p = tmp_path / "b.step"
    cq.exporters.export(cq.Workplane("XY").box(90, 60, 20)
                        .faces(">Z").workplane()
                        .rarray(40, 30, 2, 2).hole(6), str(p))
    got = {}
    for house in ("iso_office", "asme_inch", "workshop_metric"):
        st = replace(Style(), **HOUSE_STYLES[house])
        r = make_drawing(str(p), str(tmp_path / house), style=st, vary=False)
        got[house] = " ".join(e.dxf.text
                              for e in readsheet(r).modelspace().query("TEXT"))
    assert "TOP" in got["iso_office"]
    assert "VIEW A" in got["asme_inch"]
    assert "PLAN" in got["workshop_metric"]
    assert "4X" in got["iso_office"] and "4 OFF" in got["workshop_metric"]


def test_parts_list_and_revision_table_are_drawn(tmp_path):
    from dataclasses import replace
    from cli.sheet import Style
    from cli.styles import HOUSE_STYLES
    p = tmp_path / "b.step"
    cq.exporters.export(cq.Workplane("XY").box(70, 40, 12), str(p))
    st = replace(Style(), **HOUSE_STYLES["iso_office"])
    r = make_drawing(str(p), str(tmp_path / "d"), style=st, vary=False)
    texts = [e.dxf.text for e in readsheet(r).modelspace().query("TEXT")]
    assert "ITEM" in texts and "PART NO" in texts, "no parts list"
    assert "REV" in texts and "DESCRIPTION" in texts, "no revision table"


def test_printed_precision_follows_the_house_style(tmp_path):
    """One office states 0.5, another 0.500 -- both are the same size."""
    from dataclasses import replace
    from cli.sheet import Style
    from cli.styles import HOUSE_STYLES
    p = tmp_path / "b.step"
    cq.exporters.export(cq.Workplane("XY").box(63.456, 40.2, 10), str(p))
    seen = {}
    for house in ("workshop_metric", "asme_inch"):
        st = replace(Style(), **HOUSE_STYLES[house])
        r = make_drawing(str(p), str(tmp_path / house), style=st, vary=False)
        seen[house] = {a.text for a in r["annotation_record"].annotations
                       if a.kind.value == "dimension"}
    assert any(t.startswith("63.5") for t in seen["workshop_metric"]), seen
    assert any(t.startswith("63.456") for t in seen["asme_inch"]), seen


# --------------------------------------------------------------------------- #
# drawable sizes, the wider class set, tables and subpart views
# --------------------------------------------------------------------------- #
def test_no_model_is_drawn_below_a_millimetre():
    """Sub-millimetre models are scaled onto a drawable size before drawing.

    At two decimals a 0.035 mm feature prints "0.04", four distinct radii
    collapse onto one number and a thread lookup has nothing real to match.
    """
    from cli.geometry import (analyse, load, normalising_factor,
                                    smallest_feature)
    files = sorted(glob.glob(os.path.join(_root(), "examples/models/*.step")))
    if not files:
        pytest.skip("sample corpus not present")
    factors = set()
    for f in files[:8]:
        stem = os.path.splitext(os.path.basename(f))[0]
        info = analyse(load(f))
        sm = smallest_feature(info)
        k = normalising_factor(sm, stem)
        factors.add(k)
        assert sm * k >= 1.0 - 1e-9, f"{stem}: smallest {sm * k}"
    # ...and not everything is scaled to exactly the same place
    assert len(factors) >= 2, factors


def test_scaling_states_real_sizes_and_is_recorded(tmp_path):
    src = os.path.join(_root(), "examples/models/0000_00000559.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    r = make_drawing(src, str(tmp_path / "d"))
    dims = [a.value for a in r["annotation_record"].annotations
            if a.kind.value == "dimension" and a.value]
    assert dims and min(dims) >= 1.0, sorted(dims)[:4]


def test_every_sheet_carries_table_objects(tmp_path):
    """Each ruled block of fields is one detection object, kind in the detail."""
    src = os.path.join(_root(), "examples/models/flange.step")
    if not os.path.exists(src):
        pytest.skip("flange not present")
    r = make_drawing(src, str(tmp_path / "d"))
    tables = [a for a in r["annotation_record"].annotations
              if a.kind.value == "table"]
    assert tables, "no table objects"
    kinds = {a.detail.get("table_kind") for a in tables}
    assert "title_block" in kinds, kinds
    # whole-table boxes: one object per table, never one per row
    for a in tables:
        assert a.text_box and a.text_box.x1 > a.text_box.x0
        assert (a.text_box.y1 - a.text_box.y0) > 4.0, a.text


def test_class_split_separates_datums_leaders_and_captions(tmp_path):
    """The twelve classes must actually be populated by the renderer."""
    src = os.path.join(_root(), "examples/models/machined_block.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    r = make_drawing(src, str(tmp_path / "d"))
    kinds = {a.kind.value for a in r["annotation_record"].annotations}
    assert "view_caption" in kinds and "table" in kinds, kinds
    # a datum flag is no longer filed as a feature control frame
    for a in r["annotation_record"].annotations:
        if a.text.startswith("DATUM "):
            assert a.kind.value == "datum", a
        if a.text.startswith("THICKNESS") or a.text.startswith("BODY "):
            assert a.kind.value == "leader_note", a


def test_multi_body_file_gets_a_detail_view_per_item(tmp_path):
    """A file of several solids is a drawing of several parts."""
    src = os.path.join(_root(), "examples/models/handle_assembly.step")
    if not os.path.exists(src):
        pytest.skip("assembly sample not present")
    r = make_drawing(src, str(tmp_path / "d"))
    arec = r["annotation_record"]
    caps = [a.text for a in arec.annotations
            if a.kind.value == "view_caption" and a.text.startswith("ITEM")]
    assert len(caps) >= 3, caps
    # each item view is a DRAWING of that part: a real projection carrying its
    # own dimensions, not a pictorial that states nothing
    item_views = {a.view for a in arec.annotations
                  if a.kind.value == "view_caption" and a.text.startswith("ITEM")}
    dims = [a for a in arec.annotations
            if a.kind.value == "dimension" and a.view in item_views]
    assert dims, "item views carry no dimensions"


def test_dimension_chains_use_both_datum_corners():
    """Chains do not all hug the bottom-left corner."""
    from cli.sheet import Style
    from cli.variation import vary_style
    names = [os.path.splitext(os.path.basename(f))[0]
             for f in sorted(glob.glob(os.path.join(
                 _root(), "examples/models/*.step")))]
    if len(names) < 8:
        pytest.skip("sample corpus not present")
    sides = {(vary_style(Style(), n).dim_side_h, vary_style(Style(), n).dim_side_v)
             for n in names}
    assert len(sides) >= 2, sides


def test_tables_can_sit_in_a_top_corner():
    from cli.sheet import Style
    from cli.variation import vary_style
    names = [os.path.splitext(os.path.basename(f))[0]
             for f in sorted(glob.glob(os.path.join(
                 _root(), "examples/models/*.step")))]
    if len(names) < 8:
        pytest.skip("sample corpus not present")
    sides = {vary_style(Style(), n).table_side for n in names}
    assert sides & {"tl", "tr"}, sides


def test_sheet_turns_portrait_for_a_tall_part(tmp_path):
    """A tall part gets a tall sheet instead of two columns of blank paper."""
    p = tmp_path / "tall.step"
    cq.exporters.export(cq.Workplane("XY").box(20, 150, 10)
                        .faces(">Z").workplane()
                        .pushPoints([(0, 60), (0, -60)]).hole(8), str(p))
    r = make_drawing(str(p), str(tmp_path / "d"))
    sheet = r["annotation_record"].sheet
    assert sheet.height_mm > sheet.width_mm, (sheet.size, sheet.width_mm,
                                              sheet.height_mm)


def test_title_block_can_occupy_any_corner(tmp_path):
    """All four corners are reachable, and the furniture follows the block."""
    from dataclasses import replace
    from cli.sheet import Style, title_block_origin
    p = tmp_path / "b.step"
    cq.exporters.export(cq.Workplane("XY").box(60, 40, 10), str(p))
    for corner in ("br", "bl", "tr", "tl"):
        st = replace(Style(), title_block_corner=corner, notes_style="numbered")
        r = make_drawing(str(p), str(tmp_path / corner), style=st, vary=False)
        sh = r["psheet"]
        x0, y0 = title_block_origin(sh, st, st.sheet_margin,
                                    st.title_block_w, st.title_block_h)
        blocks = [a for a in r["annotation_record"].annotations
                  if a.kind.value == "table"
                  and a.detail.get("table_kind") == "title_block"]
        assert blocks, corner
        assert abs(blocks[0].text_box.x0 - x0) < 1.0, (corner, blocks[0])
        assert abs(blocks[0].text_box.y0 - y0) < 1.0, (corner, blocks[0])


def test_no_ink_escapes_the_frame_from_a_far_side_datum(tmp_path):
    """Chains drawn along the top or right edge must be reserved for.

    Regression: `_finish` padded the view block from the bottom-left only, so
    a top/right chain's witness lines ran off the sheet and their recorded
    boxes landed outside the image.
    """
    from dataclasses import replace
    from cli.sheet import Style, PText, PLine, PPoly, PCircle
    p = tmp_path / "rod.step"
    cq.exporters.export(cq.Workplane("XY").box(18, 120, 10)
                        .faces(">Z").workplane()
                        .pushPoints([(0, 50), (0, -50)]).hole(9), str(p))
    for sh_side, sv_side in (("top", "right"), ("top", "left"),
                             ("bottom", "right")):
        st = replace(Style(), dim_side_h=sh_side, dim_side_v=sv_side)
        r = make_drawing(str(p), str(tmp_path / f"{sh_side}{sv_side}"),
                         style=st, vary=False)
        sh = r["psheet"]
        frames = [q for pr in sh.prims
                  if isinstance(pr, PPoly) and pr.layer == "FRAME"
                  for q in pr.pts]
        fx0 = min(q[0] for q in frames); fx1 = max(q[0] for q in frames)
        fy0 = min(q[1] for q in frames); fy1 = max(q[1] for q in frames)
        for pr in sh.prims:
            if getattr(pr, "layer", "") == "FRAME":
                continue          # centering marks cross the border by design
            pts = []
            if isinstance(pr, PLine):
                pts = [pr.a, pr.b]
            elif isinstance(pr, PPoly):
                pts = pr.pts
            elif isinstance(pr, PCircle):
                pts = [(pr.c[0] - pr.r, pr.c[1]), (pr.c[0] + pr.r, pr.c[1])]
            elif isinstance(pr, PText):
                pts = [pr.p]
            for q in pts:
                assert fx0 - 0.5 <= q[0] <= fx1 + 0.5, (sh_side, sv_side, pr)
                assert fy0 - 0.5 <= q[1] <= fy1 + 0.5, (sh_side, sv_side, pr)
        # ...and no recorded box outside the sheet either
        for a in r["annotation_record"].annotations:
            if a.text_box:
                assert -0.5 <= a.text_box.x0 and a.text_box.x1 <= sh.w + 0.5, a
                assert -0.5 <= a.text_box.y0 and a.text_box.y1 <= sh.h + 0.5, a


def test_viewer_gives_every_class_its_own_colour():
    sys.path.insert(0, os.path.join(_root(), "tools"))
    import importlib
    cv = importlib.import_module("coco_view")
    from cli.coco import CATEGORIES
    names = [c["name"] for c in CATEGORIES if c["id"]]
    cols = [cv.colour_for(n) for n in names]
    assert len(set(cols)) == len(cols), dict(zip(names, cols))
    # the colour follows the class NAME, so two previews are comparable
    assert cv.colour_for("bores") == cv.colour_for("bores")


# --------------------------------------------------------------------------- #
# sections, detail views, radial radius leaders, sheet fill
# --------------------------------------------------------------------------- #
def test_section_view_is_cut_and_hatched():
    """A part with internal features gets a hatched section, not hidden lines."""
    from cli.geometry import load
    from cli.section import section
    src = os.path.join(_root(), "examples/models/machined_block.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    shape = load(src).val()
    sec = section(shape, "front")
    assert sec is not None, "no section produced"
    assert sec.hatch, "cut faces are not hatched"
    # the cut really removed material: the section is no wider than the part
    from cli.projection import project
    plain = project(shape, "front", hidden=True)
    assert sec.width <= plain.width + 1e-6
    # hatch lines lie inside the view, and run at a constant angle
    angs = set()
    for a, b in sec.hatch[:40]:
        assert min(sec.xmin, sec.xmax) - 1 <= a[0] <= max(sec.xmin, sec.xmax) + 1
        angs.add(round(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180))
    assert len(angs) == 1, angs


def test_section_is_drawn_with_a_cutting_plane_and_caption(tmp_path):
    # The style is pinned: whether an office sections at all is a house
    # convention (a sheet-metal shop never does), and which office a part
    # draws in moves as styles are added. This test is about the DRAWING of a
    # section, not about who asks for one.
    from dataclasses import replace
    from cli.sheet import Style
    from cli.styles import HOUSE_STYLES
    src = os.path.join(_root(), "examples/models/machined_block.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    st = replace(Style(), **dict(HOUSE_STYLES["iso_office"],
                                 section_views=True))
    r = make_drawing(src, str(tmp_path / "d"), style=st, vary=False)
    caps = [a.text for a in r["annotation_record"].annotations
            if a.kind.value == "view_caption"]
    assert any(c.startswith("SECTION") for c in caps), caps
    # the parent view carries the cutting-plane line, lettered at both ends
    assert caps.count("A") >= 2, caps
    from cli.sheet import PLine
    assert any(isinstance(p, PLine) and p.layer == "HATCH"
               for p in r["psheet"].prims), "no hatching drawn"


def test_detail_view_magnifies_a_crowded_region(tmp_path):
    """A detail view goes where the feature is too small to letter in place.

    Retargeted from flange.step to machined_block.step. The selector no
    longer scores ink density at polyline vertices -- that always picked a
    silhouette CORNER, and on 0000_00000061 it produced an empty bubble
    tangent to the edge of the part (2 polylines, 0.12 mm of ink). It now
    starts from features that are small ON PAPER, which the flange's 13.5 mm
    bores at 1.5:1 are not and the block's counterbores are.
    """
    src = os.path.join(_root(), "examples/models/machined_block.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    r = make_drawing(src, str(tmp_path / "d"))
    anns = r["annotation_record"].annotations
    caps = [a.text for a in anns if a.kind.value == "view_caption"]
    detail = [c for c in caps if c.startswith("DETAIL")]
    assert detail, caps
    # the caption states the magnification, or the view is unreadable
    assert ":" in detail[0], detail
    # ...and it carries the dimensions it was created to make room for
    carried = [a for a in anns if str(a.view or "").startswith("detail_")
               and a.kind.value != "view_caption"]
    assert carried, [(a.kind.value, a.view, a.text) for a in anns]


def test_detail_view_is_not_drawn_on_a_plain_plate(tmp_path):
    """No feature small enough to need magnifying -> no detail view.

    0000_00000061 is a 30 x 30 x 1.5 plate with one R3 corner. The old
    density selector gave it a 4:1 bubble on the outside corner containing
    nothing at all.
    """
    src = os.path.join(_root(), "examples/models/0000_00000061.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    r = make_drawing(src, str(tmp_path / "d"))
    caps = [a.text for a in r["annotation_record"].annotations
            if a.kind.value == "view_caption"]
    assert not [c for c in caps if c.startswith("DETAIL")], caps


def test_radius_notes_sit_on_the_radial():
    """A radius is stated on the arc's own radius line (ISO 129)."""
    from cli.annotate import Callout, annotate
    from cli.geometry import analyse, load
    from cli.projection import choose_views, project
    from cli.sheet import Style
    from cli.variation import vary_style
    devs = []
    for name in ("laser_panel", "bracket_plate", "0000_00000093",
                 "machined_block"):
        src = os.path.join(_root(), f"examples/models/{name}.step")
        if not os.path.exists(src):
            continue
        info = analyse(load(src))
        view = choose_views(info, max_views=1)[0]
        proj = project(load(src).val(), view, hidden=True)
        ann = annotate(proj, info, style=vary_style(Style(), name),
                       primary=True, hole_table=False, tagged=False, scale=1.0)
        arcs = [(s.center, s.radius) for s in proj.segs + proj.circles
                if getattr(s, "center", None) and getattr(s, "radius", 0) > 0]
        for d in ann.dims:
            if not isinstance(d, Callout) or getattr(d, "cls", "") != "radius":
                continue
            if not arcs:
                continue
            c, _r = min(arcs, key=lambda cr: abs(
                math.hypot(d.anchor[0] - cr[0][0], d.anchor[1] - cr[0][1])
                - cr[1]))
            rad = math.degrees(math.atan2(d.anchor[1] - c[1],
                                          d.anchor[0] - c[0]))
            lead = math.degrees(math.atan2(d.tail[1] - d.anchor[1],
                                           d.tail[0] - d.anchor[0]))
            devs.append(abs((lead - rad + 180) % 360 - 180))
    assert devs, "no radius notes found to measure"
    devs.sort()
    median = devs[len(devs) // 2]
    # was 53.6 deg median, 15 of 16 non-radial, before the radial placer
    assert median <= 5.0, devs
    assert sum(1 for d in devs if d > 5) <= len(devs) // 2, devs


def test_sheet_is_reasonably_full():
    """The drawing must use its paper: empty sheets are the tell of a robot."""
    files = sorted(glob.glob(os.path.join(_root(), "examples/models/*.step")))
    if len(files) < 8:
        pytest.skip("sample corpus not present")
    fills = []
    for f in files[:10]:
        stem = os.path.splitext(os.path.basename(f))[0]
        r = make_drawing(f, f"/tmp/_fill/{stem}", dpi=60)
        fills.append(r["annotation_record"].sheet.view_fill or 0.0)
    mean = sum(fills) / len(fills)
    # 0.394 before the sheet-choice and view-spread work, 0.475 after.
    #
    # The bar is 0.40, not the 0.44 that measured: view_fill now reports the
    # PACKED view block, not the block after the leftover paper has been
    # shared out between the columns. Spreading the views apart added no
    # drawing but inflated the number by ~0.05, and the sheet chooser was
    # using it -- 0000_00000007 passed the 0.34 legibility bar on an A2 whose
    # two small views sat at opposite ends of the paper. Same drawings,
    # honest measurement: 0.423 over this sample.
    assert mean >= 0.40, f"mean fill {mean:.3f}: {fills}"


def test_ten_house_styles_are_reachable():
    from cli.styles import HOUSE_STYLES, STYLE_WEIGHTS
    assert len(HOUSE_STYLES) >= 10, sorted(HOUSE_STYLES)
    assert set(STYLE_WEIGHTS) == set(HOUSE_STYLES)
    # the bundles really differ, on the axes a reader would notice
    axes = {(v.get("frame_style"), v.get("arrow_style"), v.get("dim_text_mode"),
             v.get("tolerance_style"), v.get("notes_style"),
             v.get("title_block_corner"), str(v.get("sheet_pref")).endswith("P"))
            for v in HOUSE_STYLES.values()}
    assert len(axes) >= 8, axes


def test_sheet_furniture_stays_inside_the_border(tmp_path):
    """Nothing but the frame's own trim lines may cross the border.

    The title block is a HOUSE dimension (up to 170 mm, plus jitter) and the
    paper can be a 210 mm A4 portrait with a 13.4 mm margin and a 6 mm zone
    strip. 181.5 mm of block into 171.2 mm of paper put the left edge of
    0000_00000386's block at x = 9.1 -- 4.3 mm outside the border -- and took
    the general notes, the finish note and the projection symbol with it,
    because all three hang off that corner.
    """
    from dataclasses import replace
    from cli.render import _prim_box
    from cli.sheet import Style
    from cli.styles import HOUSE_STYLES
    src = os.path.join(_root(), "examples/models/0000_00000061.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    # the worst case: the widest house block on the narrowest sheet
    st = replace(Style(), **dict(HOUSE_STYLES["aerospace_iso"],
                                 sheet_pref="A4P", frame_style="zoned",
                                 title_block_w=170.0))
    r = make_drawing(src, str(tmp_path / "d"), sheet="A4P", style=st,
                     vary=False)
    sh = r["psheet"]
    m = st.sheet_margin
    x0, y0, x1, y1 = m, m, sh.w - m, sh.h - m
    worst, who = 0.0, None
    for p in sh.prims:
        # the double border's trim line and the centering marks are drawn
        # outside the frame on purpose (ISO 5457)
        if getattr(p, "layer", "") == "FRAME":
            continue
        b = _prim_box(p)
        if b is None:
            continue
        d = max(x0 - b[0], y0 - b[1], b[2] - x1, b[3] - y1)
        if d > worst:
            worst, who = d, (type(p).__name__, getattr(p, "s", ""))
    assert worst <= 0.01, f"{worst:.2f} mm outside the border: {who}"


def test_notes_box_lands_on_the_notes_for_every_block_corner(tmp_path):
    """The recorded notes box must sit on the ink, in all four corners.

    Two derivations of the same origin have now caused this twice: 18.5 mm
    vertically (draw_title_block recomputing ``ny``) and 74 mm horizontally
    (_emit_general_notes testing only for a BOTTOM-left block and taking the
    right-hand formula for a top-left one). Both are single-sourced through
    ``meta["notes_x"]`` / ``meta["notes_base"]`` now, and this measures it.
    """
    from dataclasses import replace
    from cli.sheet import PText, Style
    src = os.path.join(_root(), "examples/models/0000_00000061.step")
    if not os.path.exists(src):
        pytest.skip("sample not present")
    for corner in ("bl", "br", "tl", "tr"):
        st = replace(Style(), title_block_corner=corner, notes_style="numbered")
        r = make_drawing(src, str(tmp_path / f"d_{corner}"), style=st,
                         vary=False)
        sh = r["psheet"]
        texts = [p for p in sh.prims if isinstance(p, PText)]
        for a in r["annotation_record"].annotations:
            if a.kind.value != "note" or not a.text_box:
                continue
            match = [t for t in texts if t.s == a.text]
            assert match, (corner, a.text)
            bx, by = a.text_box.x0, a.text_box.y0
            drawn = min((_ptext_box(t) for t in match),
                        key=lambda b: abs(b[0] - bx) + abs(b[1] - by))
            dx, dy = abs(drawn[0] - bx), abs(drawn[1] - by)
            assert dx < 1.0 and dy < 1.0, (corner, a.text, dx, dy)


def test_exported_text_boxes_contain_their_glyphs():
    """A detection box must hold the whole string, with a hair to spare.

    The tight box is matplotlib's layout box intersected with the glyph
    outlines. Measured against the rasterised PNG at 200 dpi it was up to
    0.12 mm too small on the LEFT of every string -- one pixel, and visibly
    clipping the first and last glyph. _GLYPH_PAD is the fix; this checks it
    still covers the ink extent the renderer will draw.
    """
    from cli.render import _text_box
    from cli.sheet import PText
    from cli.textmetrics import drawn_box
    for s in ("1.68", "\u23004.47 THRU", "SECTION A-A", "Ra 1.6"):
        for h in (2.2, 3.5):
            t = PText((10.0, 20.0), s, h, char_w=0.62,
                      font_family="DejaVu Sans")
            box = _text_box(t, grow=0.0)
            ink = drawn_box(s, h, "DejaVu Sans", ha=t.ha, va=t.va, rot=t.rot)
            if ink is None:
                pytest.skip("font not measurable here")
            assert box[0] <= 10.0 + ink[0] and box[1] <= 20.0 + ink[1]
            assert box[2] >= 10.0 + ink[2] and box[3] >= 20.0 + ink[3]
            # ...and still tight: no more than a third of a text height
            assert (box[2] - box[0]) - (ink[2] - ink[0]) <= h * 0.34


def test_weight_is_computed_from_the_solid():
    """The WEIGHT field states a fact about the part, or it is left out.

    It used to be ``(rng % 90000) / 1000`` kilograms: 0000_00000386 is a
    15 mm washer and its title block read "55.661 kg".
    """
    from cli.titleblock import weight_text
    # 15 x 1.04 washer, minus its bore, in aluminium: well under a gram
    assert weight_text(179.0, "AL 7075-T6").endswith(" g")
    assert float(weight_text(179.0, "AL 7075-T6").split()[0]) < 1.0
    # a 150 cm3 steel block is over a kilogram
    assert weight_text(150000.0, "S355JR").endswith(" kg")
    # nothing to compute from -> no field at all
    assert weight_text(0.0, "AL 6061-T6") == ""
    assert weight_text(1000.0, "UNOBTAINIUM") == ""


def test_shifted_dimension_text_sits_on_an_extension_of_its_line():
    """ISO 129-1: a number that will not fit between its arrows is written

    above an extension of the dimension line. Without the extension it reads
    as unattached -- "7.5 +0.300/-0.200" floated off the end of its chain on
    0000_00000386.
    """
    from cli.annotate import Annotated, LinearDim
    from cli.projection import Projection
    from cli.render import Placed, _linear
    from cli.sheet import PLine, PSheet, Style
    st = Style()
    sh = PSheet(420.0, 297.0)
    proj = Projection("top", xmin=0, xmax=10, ymin=0, ymax=10)
    pl = Placed(Annotated(view=proj), 50.0, 50.0, 1.0, name="top")
    d = LinearDim((0.0, 0.0), (4.0, 0.0), -8.0, "h",
                  "7.5 +0.300/-0.200", shift=1)
    before = len(sh.prims)
    _linear(sh, pl, d, st)
    tail = [p for p in sh.prims[before:]
            if isinstance(p, PLine) and p.layer == "DIM"
            and abs(p.a[1] - p.b[1]) < 1e-9 and p.a[0] >= 54.0 - 1e-6]
    assert tail, "no extension drawn under the shifted number"
    reach = max(max(p.a[0], p.b[0]) for p in tail) - 54.0
    tw = len(d.text) * st.dim_text_height * st.char_w
    assert reach >= tw * 0.8, (reach, tw)
