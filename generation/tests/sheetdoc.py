"""Read a rendered sheet's primitives the way the DXF tests used to.

The pipeline writes one output format (the dataset PNG), so the drawing-quality
tests can no longer inspect a DXF. They inspect the paper-space primitive list
instead -- the very thing the exporter used to serialise -- through this thin
adapter, which keeps the entity/attribute vocabulary the assertions were
written against.

    r = make_drawing(part, base)
    msp = readsheet(r).modelspace()
    [e.dxf.text for e in msp.query("TEXT")]

Coordinates are paper millimetres with the origin bottom-left, exactly as they
were in the DXF.
"""
from __future__ import annotations

import re

from cli.export import _arrow_pts
from cli.sheet import PArrow, PCircle, PFace, PLine, PPoly, PSheet, PText


class _Attrs:
    """Stand-in for ezdxf's ``entity.dxf`` namespace."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def hasattr(self, name: str) -> bool:
        return name in self.__dict__


class _Point(tuple):
    """A 2D point that also answers to ``.x`` / ``.y``."""

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]


class _Entity:
    def __init__(self, kind: str, points=(), prim=None, closed=False, **attrs):
        self._kind = kind
        self._points = [tuple(p) for p in points]
        self.closed = closed
        #: the sheet primitive this entity was built from
        self.prim = prim
        self.dxf = _Attrs(**attrs)

    def dxftype(self) -> str:
        return self._kind

    def get_points(self, fmt: str = "xy"):
        return list(self._points)

    def __getitem__(self, i):
        """Corner access, as ezdxf offers on a SOLID."""
        return _Point(self._points[i])

    def __len__(self):
        return len(self._points)


_HALIGN = {"left": 0, "center": 1, "right": 2}
#: DXF vertical alignment codes: 0 baseline, 1 bottom, 2 middle, 3 top.
_VALIGN = {"bottom": 0, "center": 2, "top": 3}


def _entities(sh: PSheet):
    """Every primitive on the sheet, as DXF-shaped entities."""
    out = []
    for p in sh.prims:
        if isinstance(p, PLine):
            out.append(_Entity("LINE", prim=p, layer=p.layer,
                               start=_Point(p.a), end=_Point(p.b)))
        elif isinstance(p, PPoly):
            out.append(_Entity("LWPOLYLINE", points=p.pts, prim=p,
                               closed=bool(p.close), layer=p.layer))
        elif isinstance(p, PCircle):
            out.append(_Entity("CIRCLE", prim=p, layer=p.layer,
                               center=_Point(p.c), radius=p.r))
        elif isinstance(p, PFace):
            q = p.pts[:4]
            if len(q) == 3:
                q = q + [q[2]]
            grey = max(1, min(255, int(250 - p.shade * 80)))
            out.append(_Entity("SOLID", points=q, prim=p, layer=p.layer,
                               true_color=grey * 0x010101))
        elif isinstance(p, PArrow):
            pts = _arrow_pts(p.tip, p.tail, p.size) + [p.tip]
            out.append(_Entity("SOLID", points=pts, prim=p, layer=p.layer))
        elif isinstance(p, PText):
            # `width` mirrors the DXF glyph width factor the exporter wrote:
            # the sheet's real character-width ratio over the 0.62 a default
            # text style assumes, so a checker can recover the true extent.
            out.append(_Entity(
                "TEXT", prim=p, layer=p.layer, text=p.s, height=p.h,
                insert=_Point(p.p), rotation=p.rot,
                halign=_HALIGN.get(p.ha, 0), valign=_VALIGN.get(p.va, 0),
                width=round(p.char_w * (1.06 if p.bold else 1.0) / 0.62, 4),
                char_w=p.char_w, bold=p.bold, font_family=p.font_family))
    return out


class _Modelspace:
    def __init__(self, sh: PSheet):
        self._entities = _entities(sh)

    def __iter__(self):
        return iter(self._entities)

    def __len__(self):
        return len(self._entities)

    def query(self, spec: str):
        """Entities by type, with ezdxf's ``TYPE[layer=="NAME"]`` filter."""
        layer = None
        m = re.fullmatch(r'([A-Z ]+)\[layer=="([^"]+)"\]', spec.strip())
        if m:
            spec, layer = m.group(1), m.group(2)
        wanted = set(spec.split())
        return [e for e in self._entities if e.dxftype() in wanted
                and (layer is None or e.dxf.layer == layer)]


class _Doc:
    def __init__(self, sh: PSheet):
        self.sheet = sh
        self._msp = _Modelspace(sh)

    def modelspace(self):
        return self._msp


def readsheet(source) -> _Doc:
    """Adapt a ``make_drawing`` result (or a raw PSheet) for inspection."""
    sh = source["psheet"] if isinstance(source, dict) else source
    return _Doc(sh)
