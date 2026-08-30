"""Convert the pipeline's plain dicts into a validated :mod:`schema` record.

Kept separate from ``schema.py`` so the models stay a pure, dependency-light
description of the data and can be imported by consumers that never run the
pipeline.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Optional

from .schema import (Annotation, AnnotationKind, AnnotationRecord,
                     BoundingBox2D, ProjectionAngle, SheetRecord, TitleBlock,
                     ViewName, ViewRecord)


def _sha256(path: str, limit: int = 64 * 1024 * 1024) -> Optional[str]:
    """Content hash of the source model, so a record names a CAD revision."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            read = 0
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                read += len(chunk)
                if read > limit:          # don't hash enormous files
                    return None
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _views(result: Dict[str, Any], view_origins: Optional[dict]) -> list:
    """One :class:`ViewRecord` per view actually placed on the sheet."""
    metrics = result.get("view_metrics", {}) or {}
    views = []
    for i, vname in enumerate(result.get("views", []) or []):
        try:
            vn = ViewName(vname)
        except ValueError:
            continue
        m = metrics.get(vname, {})
        origin = (view_origins or {}).get(vname)
        views.append(ViewRecord(
            name=vn, is_primary=(i == 0),
            width=m.get("width", 0.0), height=m.get("height", 0.0),
            xmin=m.get("xmin", 0.0), ymin=m.get("ymin", 0.0),
            pad_left=m.get("pad_left", 0.0), pad_right=m.get("pad_right", 0.0),
            pad_bottom=m.get("pad_bottom", 0.0), pad_top=m.get("pad_top", 0.0),
            hole_count=m.get("hole_count", 0),
            paper_origin=tuple(origin) if origin else None))
    return views


def _annotations(result: Dict[str, Any]) -> list:
    """Every mark drawn on the sheet, as validated records."""
    anns = []
    for a in result.get("annotations", []) or []:
        def _bb(key):
            b = a.get(key)
            return (BoundingBox2D(x0=b[0], y0=b[1], x1=b[2], y1=b[3])
                    if b else None)
        try:
            kind = AnnotationKind(a.get("kind", "note"))
        except ValueError:
            continue
        anns.append(Annotation(
            kind=kind, view=a.get("view", "") or "",
            text=a.get("text", ""), value=a.get("value"),
            direction=a.get("direction"), qty=a.get("qty"),
            text_box=_bb("text_box"), extent=_bb("extent"),
            attach=[tuple(p) for p in (a.get("attach") or [])],
            detail=a.get("detail") or {},
        ))
    return anns


def annotation_record(result: Dict[str, Any], *,
                      source_path: Optional[str] = None,
                      sheet_size: str = "A3",
                      sheet_wh: Optional[tuple] = None,
                      scale: Optional[float] = None,
                      scale_text: Optional[str] = None,
                      projection: str = "third",
                      title: Optional[str] = None,
                      notes: Optional[list] = None,
                      view_fill: Optional[float] = None,
                      view_origins: Optional[dict] = None,
                      generator_version: Optional[str] = None,
                      **_ignored: Any) -> AnnotationRecord:
    """Build an :class:`AnnotationRecord` from ``make_drawing()`` output."""
    path = str(source_path or result.get("input") or "unknown")
    stem = os.path.splitext(os.path.basename(path))[0]

    from .sheet import SHEETS
    w, h = sheet_wh or SHEETS.get(sheet_size, (420.0, 297.0))

    return AnnotationRecord(
        generator_version=generator_version,
        model_path=path,
        model_sha256=(_sha256(path) if os.path.exists(path) else None),
        units=result.get("units", "mm"),
        sheet=SheetRecord(
            size=sheet_size, width_mm=w, height_mm=h,
            scale=scale or 1.0, scale_text=scale_text or "1:1",
            projection=ProjectionAngle(projection), view_fill=view_fill),
        title_block=TitleBlock(title=title or stem.upper(), part_number=stem,
                               notes=list(notes or [])),
        views=_views(result, view_origins),
        annotations=_annotations(result),
        hole_table=bool(result.get("hole_table", False)),
        model_scale=float(result.get("model_scale", 1.0) or 1.0),
        warnings=list(result.get("warnings") or []),
        files=dict(result.get("files") or {}),
    )
