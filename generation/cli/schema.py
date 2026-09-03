"""Typed record of what one drawing says.

:class:`AnnotationRecord` is the contract between the renderer and the label
writer (:mod:`autodraft.coco`): every mark drawn on
the sheet, with its exact text, its measured value and its paper-space box.

Design notes
------------
* **Units are explicit.** Model-space lengths are in the drawing's ``units``;
  paper-space lengths are always millimetres and named ``*_mm``. The two are
  never mixed in one field.
* **Nothing is silently invented.** Values the pipeline cannot determine are
  ``None`` rather than a plausible-looking default.
* **Round-trips JSON.** ``AnnotationRecord.model_validate_json(
  rec.model_dump_json())`` reproduces the record, computed fields included.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

__all__ = [
    "ProjectionAngle", "ViewName", "BoundingBox2D", "ViewRecord",
    "SheetRecord", "TitleBlock", "AnnotationKind", "Annotation",
    "AnnotationRecord", "Coordinate", "Category", "Feature", "FeatureList",
]


# --------------------------------------------------------------------------- #
# enumerations
# --------------------------------------------------------------------------- #
class ProjectionAngle(str, Enum):
    """Multiview projection convention used to arrange the views."""

    FIRST = "first"
    THIRD = "third"


class ViewName(str, Enum):
    """Canonical orthographic/pictorial view identifiers."""

    FRONT = "front"
    BACK = "back"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    ISO = "iso"


class AnnotationKind(str, Enum):
    """What sort of thing an annotation *states*, not how it is drawn.

    These are the detection classes. They are semantic: a plate-thickness
    leader note and a witness-line dimension both state a size, but they are
    told apart by what they name, not by the ink used to draw them.

    The split follows what a reader recognises as one object:

    * ``GDT`` is a feature control frame, ``DATUM`` the boxed datum letter --
      different glyphs, different jobs, so different classes.
    * ``RADIUS`` covers R-value fillet/corner callouts; ``BORE`` covers hole
      and bore-diameter callouts, including the ``⌀`` symbol.
    * ``NOTE`` is the paragraph block (``NOTES:`` / ``UNLESS OTHERWISE
      SPECIFIED``); ``LEADER_NOTE`` is a short string on a leader
      (``THICKNESS 12``). A block of prose and a leader label are not one
      class.
    * ``TABLE`` is any ruled grid of fields -- title block, hole table, parts
      list, revision table -- with which one in ``detail.table_kind``.
    * ``VIEW_CAPTION`` is the label under a view (``TOP``, ``VIEW A``).
    * ``FEATURE`` is internal: the recognised-feature box, recorded whether or
      not it is inked, and never exported as a detection object.
    """

    GDT = "gdt"
    DATUM = "datum"
    ROUGHNESS = "roughness"
    RADIUS = "radius"
    CHAMFER = "chamfer"
    BORE = "bore"
    THREAD = "thread"
    DIMENSION = "dimension"
    NOTE = "note"
    LEADER_NOTE = "leader_note"
    TABLE = "table"
    VIEW_CAPTION = "view_caption"
    FEATURE = "feature"


class BoundingBox2D(BaseModel):
    """Axis-aligned rectangle in a view's 2D coordinate system.

    Coordinates are *view space*: the projection of model space onto the
    viewing plane, still in model units, with the origin inherited from the
    model rather than from the sheet.
    """

    model_config = ConfigDict(extra="ignore")

    x0: float = Field(..., description="Lower-left X in view coordinates, model units.")
    y0: float = Field(..., description="Lower-left Y in view coordinates, model units.")
    x1: float = Field(..., description="Upper-right X in view coordinates, model units.")
    y1: float = Field(..., description="Upper-right Y in view coordinates, model units.")

    @field_validator("x1")
    @classmethod
    def _x_ordered(cls, v: float, info) -> float:
        x0 = info.data.get("x0")
        if x0 is not None and v < x0:
            raise ValueError("x1 must be >= x0 (box corners must be ordered)")
        return v

    @field_validator("y1")
    @classmethod
    def _y_ordered(cls, v: float, info) -> float:
        y0 = info.data.get("y0")
        if y0 is not None and v < y0:
            raise ValueError("y1 must be >= y0 (box corners must be ordered)")
        return v

    @computed_field(description="Box width (x1-x0), in model units.")
    @property
    def w(self) -> float:
        return round(self.x1 - self.x0, 6)

    @computed_field(description="Box height (y1-y0), in model units.")
    @property
    def h(self) -> float:
        return round(self.y1 - self.y0, 6)


# --------------------------------------------------------------------------- #
# views and sheet
# --------------------------------------------------------------------------- #
class ViewRecord(BaseModel):
    """One projected view placed on the sheet."""

    model_config = ConfigDict(extra="ignore")

    name: ViewName = Field(..., description="Which orthographic/pictorial view this is.")
    is_primary: bool = Field(
        False,
        description="True for the view chosen as the drawing's principal view; "
                    "general notes (thickness, fillet callouts) attach to it.")
    width: float = Field(
        ..., ge=0,
        description="Silhouette width in view coordinates, model units "
                    "(excludes dimensions and callouts).")
    height: float = Field(
        ..., ge=0,
        description="Silhouette height in view coordinates, model units.")
    pad_left: float = Field(
        0.0, ge=0,
        description="Space reserved left of the silhouette for annotation, "
                    "model units. Measured from the real annotation extents, "
                    "not estimated.")
    pad_right: float = Field(0.0, ge=0, description="Annotation space to the right, model units.")
    pad_bottom: float = Field(
        0.0, ge=0,
        description="Annotation space below, model units. Includes the strip "
                    "reserved for the view label.")
    pad_top: float = Field(0.0, ge=0, description="Annotation space above, model units.")
    hole_count: int = Field(
        0, ge=0,
        description="Holes visible as circles in this view, i.e. whose axis is "
                    "parallel to the viewing direction. This counts GEOMETRY, "
                    "not annotations, so it is not derivable from the "
                    "annotation list and is kept here.")
    xmin: float = Field(
        0.0,
        description="Lowest X of the silhouette in view coordinates. The view "
                    "origin is the model origin, so this is rarely 0 -- a "
                    "consumer mapping boxes to paper needs it together with "
                    "paper_origin.")
    ymin: float = Field(
        0.0, description="Lowest Y of the silhouette in view coordinates.")
    paper_origin: Optional[Tuple[float, float]] = Field(
        None,
        description="Paper-space position (mm) of this view's coordinate "
                    "origin. A view-space point maps to the sheet as "
                    "paper = origin + view_coord * sheet.scale. Lets a "
                    "consumer place feature boxes on the drawing exactly, "
                    "rather than inferring the transform.")


class SheetRecord(BaseModel):
    """Paper the drawing was laid out on, and how densely it is used."""

    model_config = ConfigDict(extra="forbid")

    size: str = Field(
        ...,
        description="Sheet designation: ISO A4-A0 or ANSI A-D. May differ from "
                    "the requested size when auto-escalation found the part "
                    "would otherwise be drawn too small.")
    width_mm: float = Field(..., gt=0, description="Sheet width in PAPER millimetres.")
    height_mm: float = Field(..., gt=0, description="Sheet height in PAPER millimetres.")
    scale: float = Field(
        ..., gt=0,
        description="Paper millimetres per model unit. >1 is an enlargement "
                    "(0.35 mm fastener at 150:1), <1 a reduction. Always a "
                    "value from the preferred ladder, never an arbitrary fit.")
    scale_text: str = Field(
        ...,
        description="Scale as printed in the title block, e.g. '1:2', '2:1', "
                    "'1.25:1'.")
    projection: ProjectionAngle = Field(
        ProjectionAngle.THIRD,
        description="Projection convention; also determines which grid cell is "
                    "free for the isometric view.")
    view_fill: Optional[float] = Field(
        None, ge=0, le=1,
        description="Fraction of the usable drawing area occupied by the view "
                    "block. Low values indicate a wasteful layout.")
    auto_escalated: bool = Field(
        False,
        description="True when the renderer moved to a larger sheet than "
                    "requested because the part could not be drawn legibly.")


class TitleBlock(BaseModel):
    """Text fields printed in the drawing's title block."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., description="Drawing title, defaults to the source file stem.")
    part_number: Optional[str] = Field(None, description="Part number as printed.")
    material: Optional[str] = Field(None, description="Material spec as printed.")
    revision: Optional[str] = Field(None, description="Revision identifier, if supplied.")
    drawn_by: str = Field("AUTODRAFT", description="Value of the DRAWN field.")
    date: _dt.date = Field(
        default_factory=_dt.date.today,
        description="Date printed on the sheet (ISO-8601 in JSON).")
    sheet: str = Field("1/1", description="Sheet number within the drawing set.")
    notes: List[str] = Field(
        default_factory=list,
        description="General notes printed above the title block, in order. "
                    "Includes the auto-generated stock envelope note and any "
                    "multi-body warning.")


# --------------------------------------------------------------------------- #
# annotations
# --------------------------------------------------------------------------- #
class Annotation(BaseModel):
    """One annotation actually drawn on the sheet.

    Covers dimensions, angular dimensions, leader callouts, notes and feature
    bounding boxes under a single shape, because a consumer laying out or
    auditing a drawing wants them in one ordered list with one geometry
    convention -- not split across three differently-shaped collections.

    All boxes are in PAPER millimetres with origin at the bottom-left of the
    sheet. Feature detail carried in ``detail`` may be expressed in view/model
    units; see ``ViewRecord.paper_origin`` for the mapping between the two
    spaces.
    """

    model_config = ConfigDict(extra="forbid")

    kind: AnnotationKind = Field(..., description="Type of annotation.")
    view: str = Field(
        "", description="Name of the view this annotation belongs to.")
    text: str = Field(
        ...,
        description="Exact string as rendered, including any prefix/suffix "
                    "and embedded newlines for multi-line callouts. Compare "
                    "this against the drawing verbatim.")
    value: Optional[float] = Field(
        None,
        description="Measured quantity behind the text, as a number. Lengths "
                    "and diameters are in the record's `units` -- the same "
                    "unit the text is printed in, so value and text always "
                    "agree. Angles are degrees. None for notes and feature "
                    "boxes, which measure nothing. Saves the consumer parsing "
                    "the formatted string.")
    direction: Optional[str] = Field(
        None, description="'h' or 'v' for linear dimensions, else None.")
    qty: Optional[int] = Field(
        None, ge=1,
        description="Number of features a feature box encloses, else None.")
    text_box: Optional[BoundingBox2D] = Field(
        None,
        description="Paper-space box of the rendered text alone, in mm. None "
                    "for a feature box, which has no text of its own.")
    extent: Optional[BoundingBox2D] = Field(
        None,
        description="Paper-space box of the whole annotation in mm: text plus "
                    "its dimension line, leader or arc. This is what a layout "
                    "checker should test for collisions.")
    attach: List[Tuple[float, float]] = Field(
        default_factory=list,
        description="Paper-space points the annotation attaches to: the two "
                    "measured points of a dimension, the arrowhead tip of a "
                    "leader, or the vertex of an angle.")
    detail: Dict[str, Any] = Field(
        default_factory=dict,
        description="Class-specific extras that do not apply to every "
                    "annotation, so they are not promoted to columns that "
                    "would be null for most rows. Keys in use: "
                    "`form` ('linear' / 'angular' / 'diameter') on a "
                    "dimension; `characteristic`, `datums` and `modifier` on "
                    "a GD&T frame; `datum` on a datum flag; `machined` and "
                    "`synthetic` on a roughness symbol. `synthetic: true` "
                    "marks a value that was assigned rather than measured -- "
                    "currently only surface finish, which no input file "
                    "carries; see autodraft/roughness.py.")


class AnnotationRecord(BaseModel):
    """Everything that describes the drawn sheet.

    An extractor training on drawing->annotation pairs needs exactly this
    plus the rendered image.

    Nothing derivable from ``annotations`` is stored twice. In particular
    there are no per-view dimension/callout counters and no separate
    feature-box list -- feature boxes are annotations with kind='feature'.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    # -- identity ----------------------------------------------------------
    schema_version: str = Field("2.0", description="Record layout version.")
    generated_at: _dt.datetime = Field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc),
        description="UTC timestamp when the sheet was rendered.")
    generator_version: Optional[str] = Field(
        None, description="AutoDraft version that produced this record.")
    model_path: str = Field(
        "", description="Source CAD path, for human orientation only.")
    model_sha256: Optional[str] = Field(
        None,
        description="sha256 of the source model, so a record can be tied to "
                    "the exact CAD revision it was rendered from.")

    # -- paper -------------------------------------------------------------
    units: str = Field(
        "mm",
        description="Unit the drawing states its dimensions in, and the unit "
                    "of every Annotation.value. Always millimetres, whatever "
                    "the part size: cm and m destroy small features at two "
                    "decimals. Paper boxes (text_box, extent) are millimetres "
                    "of sheet -- they measure the page, not the part.")
    sheet: SheetRecord = Field(..., description="Sheet size, scale and layout.")
    title_block: TitleBlock = Field(..., description="Title block text.")
    views: List[ViewRecord] = Field(
        default_factory=list,
        description="Views placed on the sheet, with the transform needed to "
                    "map view coordinates onto paper.")

    # -- content -----------------------------------------------------------
    annotations: List[Annotation] = Field(
        default_factory=list,
        description="Dimensions, angles, callouts, notes and feature boxes "
                    "actually drawn, in the order they were emitted. This is "
                    "the authoritative list; nothing counts them separately.")
    hole_table: bool = Field(
        False, description="True when hole data is given as a tagged table.")
    model_scale: float = Field(
        1.0,
        description="Factor the source model was scaled by before drawing "
                    "(1.0 = as supplied). Sub-millimetre models are lifted "
                    "onto a drawable size, because two decimals cannot state "
                    "scaled geometry, so divide by this to recover the "
                    "size in the file.")
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal caveats about what the sheet can state: a "
                    "multi-body source whose overall dimensions are an "
                    "envelope, or features too small to state at the "
                    "drawing's precision.")

    # -- output ------------------------------------------------------------
    files: Dict[str, str] = Field(
        default_factory=dict,
        description="Files written for this drawing, keyed by kind "
                    "('png', 'coco').")

    # ------------------------------------------------------------------ #
    @computed_field(description="Total annotations of every kind on the sheet.")
    @property
    def total_annotations(self) -> int:
        return len(self.annotations)

    @computed_field(description="Count of annotations by kind, so a consumer "
                                "can spot an under-dimensioned drawing "
                                "without walking the list.")
    @property
    def annotation_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for a in self.annotations:
            out[a.kind.value] = out.get(a.kind.value, 0) + 1
        return out


# --------------------------------------------------------------------------- #
# detection output
# --------------------------------------------------------------------------- #
# OpenRouter detection output
Coordinate = Annotated[int, Field(ge=0)]


class Category(str, Enum):
    GDNT = "gdnts"
    ROUGHNESS = "roughnesses"
    RADIUS = "radii"
    CHAMFER = "chamferes"
    BORE = "bores"
    THREAD = "threads"
    DIMENSION = "dimensions"
    NOTE = "notes"
    DATUM = "datums"
    LEADER_NOTE = "leader_notes"
    TABLE = "tables"
    VIEW_CAPTION = "view_captions"


class Feature(BaseModel):
    """One OpenRouter/COCO feature box."""

    model_config = ConfigDict(extra="ignore")

    id: int = Field(ge=0)
    bbox: Tuple[Coordinate, Coordinate, Coordinate, Coordinate]
    text: str = ""
    category: Category
    confidence: float = Field(ge=0, le=1)
    is_artificial: bool


class FeatureList(BaseModel):
    """Structured list returned by an OpenRouter model."""

    feature_list: List[Feature]

    def __getitem__(self, key):
        return self.feature_list[key]

    def __len__(self):
        return len(self.feature_list)
