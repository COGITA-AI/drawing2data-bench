"""COCO-format detection labels for the rendered drawing.

One COCO *object* per annotation actually drawn on the sheet: every dimension,
angular dimension, leader callout and note. These are the things a detector
would be asked to find in the drawing image, so the file is written against
the PNG and its boxes are in image pixels.

``bbox`` covers the annotation's TEXT only -- the number or note itself, not
the dimension line, witness lines or leader that point at it. A dimension
line spans the whole feature it measures, so boxing it would make the target
mostly empty paper and would overlap every other annotation on that view. The
full extent is kept in ``attributes.extent_bbox`` for anyone who wants it.

Two conversions matter and are easy to get wrong:

* **Origin.** Sheet coordinates are millimetres with y increasing upward from
  the bottom-left. COCO boxes are pixels with y increasing downward from the
  top-left, so y is flipped, not merely scaled.
* **Scale.** Pixels per millimetre is taken from the rendered image size
  rather than from ``dpi / 25.4``. matplotlib rounds the figure to a whole
  number of pixels, so the two differ slightly (7.8738 vs 7.8740 px/mm on
  A3 at 200 dpi) and the axes differ from each other. Using the nominal
  value puts a box up to a pixel out at the right-hand edge of the sheet.

Internal ``feature`` annotations are excluded: they are bookkeeping boxes
around recognised geometry that duplicate the callout describing the same
holes, and nothing is drawn for them unless ``--feature-boxes`` is set.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import List

# Category ids are fixed and explicit rather than derived from a set, so a
# label file stays comparable across runs and across versions of this tool.
CATEGORIES = [
    # Roboflow's COCO exports carry a root category at id 0 with
    # supercategory "none", and its importer expects the real classes to be
    # its children. Nothing references id 0, so plain COCO readers and
    # pycocotools simply see one unused category.
    {"id": 0, "name": "annotation", "supercategory": "none"},
    {"id": 1, "name": "gdnts", "supercategory": "annotation"},
    {"id": 2, "name": "roughnesses", "supercategory": "annotation"},
    {"id": 3, "name": "radii", "supercategory": "annotation"},  # R and ⌀ callouts
    {"id": 4, "name": "chamferes", "supercategory": "annotation"},
    {"id": 5, "name": "bores", "supercategory": "annotation"},
    {"id": 6, "name": "threads", "supercategory": "annotation"},
    {"id": 7, "name": "dimensions", "supercategory": "annotation"},
    {"id": 8, "name": "notes", "supercategory": "annotation"},
    {"id": 9, "name": "datums", "supercategory": "annotation"},
    {"id": 10, "name": "leader_notes", "supercategory": "annotation"},
    {"id": 11, "name": "tables", "supercategory": "annotation"},
    {"id": 12, "name": "view_captions", "supercategory": "annotation"},
]

_CATEGORY_OF_KIND = {
    "gdt": 1,
    "roughness": 2,
    "radius": 3,
    "chamfer": 4,
    "bore": 5,
    "thread": 6,
    "dimension": 7,
    "note": 8,
    "datum": 9,
    "leader_note": 10,
    "table": 11,
    "view_caption": 12,
}

# The detection output is validated against the public :class:`Feature`
# schema, whose ``category`` is a :class:`Category` enum. Every internal
# kind names in AnnotationRecord are singular; the OpenRouter/Feature
# contract and the COCO category names are the fixed plural labels below.
_FEATURE_CATEGORY_OF_KIND = {
    "gdt": "gdnts",
    "roughness": "roughnesses",
    "radius": "radii",
    "chamfer": "chamferes",
    "bore": "bores",
    "thread": "threads",
    "dimension": "dimensions",
    "note": "notes",
    "datum": "datums",
    "leader_note": "leader_notes",
    "table": "tables",
    "view_caption": "view_captions",
}
_FEATURE_FIELDS = ("id", "bbox", "text", "category", "confidence", "is_artificial")


def _feature_list(objects):
    return [{key: obj[key] for key in _FEATURE_FIELDS} for obj in objects]


def _boxes_to_pixels(x0: float, y0: float, x1: float, y1: float,
                     px_per_mm_x: float, px_per_mm_y: float,
                     sheet_h_mm: float):
    """Sheet mm (y-up, bottom-left) -> COCO [x, y, w, h] in px (y-down).

    Returns whole-number pixel boxes (``int``). The dataset objects are
    validated by the :class:`~autodraft.schema.Feature` model, whose ``bbox``
    is a tuple of four non-negative ``int``; sub-pixel floats are rejected, so
    the box is rounded here rather than at the consumer.
    """
    left = x0 * px_per_mm_x
    right = x1 * px_per_mm_x
    # flip: the sheet's top edge (y1) becomes the box's top in image space
    top = (sheet_h_mm - y1) * px_per_mm_y
    bottom = (sheet_h_mm - y0) * px_per_mm_y
    return [round(left), round(top),
            round(right - left), round(bottom - top)]


def build_coco(annotation_record, image_path: str, image_size=None,
               image_id: int = 1, start_annotation_id: int = 1) -> dict:
    """Build a COCO dict for one drawing.

    ``annotation_record`` is the :class:`~autodraft.schema.AnnotationRecord`
    for the sheet; ``image_path`` is the PNG the boxes refer to. Pass
    ``image_size`` as ``(width_px, height_px)`` to avoid re-opening the image.
    """
    sheet = annotation_record.sheet
    if image_size is None:
        image_size = _image_size(image_path)
    width_px, height_px = image_size
    px_x = width_px / sheet.width_mm
    px_y = height_px / sheet.height_mm

    objects: List[dict] = []
    next_id = start_annotation_id
    for a in annotation_record.annotations:
        category = _CATEGORY_OF_KIND.get(a.kind.value)
        if category is None:        # 'feature' boxes are not drawn objects
            continue
        # The box is the TEXT ONLY -- the number or note a reader looks at,
        # not the dimension line and leader that point to it. Those extend
        # right across the view (the median text box is 7% of the area of the
        # full extent, and one is 0.8%), so boxing them makes the target
        # mostly blank paper and overlaps every neighbouring annotation.
        box_mm = a.text_box or a.extent
        if box_mm is None:
            continue
        bbox = _boxes_to_pixels(box_mm.x0, box_mm.y0, box_mm.x1, box_mm.y1,
                                px_x, px_y, sheet.height_mm)
        # The full extent stays available as an extra, for anyone who wants
        # the leader as well.
        extent_box = None
        if a.extent is not None:
            extent_box = _boxes_to_pixels(a.extent.x0, a.extent.y0,
                                          a.extent.x1, a.extent.y1,
                                          px_x, px_y, sheet.height_mm)
        feature_category = _FEATURE_CATEGORY_OF_KIND[a.kind.value]
        objects.append({
            "id": next_id,
            "image_id": image_id,
            # COCO's integer category_id, kept for plain-COCO consumers.
            "category_id": category,
            "bbox": bbox,
            "area": round(bbox[2] * bbox[3], 2),
            "iscrowd": 0,
            "segmentation": [],
            # -- the :class:`Feature` validation contract ------------------
            # Every object must validate against the public Feature schema, so
            # the class label, rendered text and certainty are carried at the
            # top level (not only inside attributes), using the plural
            # OpenRouter/COCO label rather than the internal singular kind.
            "category": feature_category,
            "text": a.text,
            "confidence": 1.0,
            # This is intentionally top-level so Feature validation can use it
            # without opening the richer COCO attributes block.
            "is_artificial": bool((a.detail or {}).get("synthetic", False)),
            # -- extras beyond the COCO spec, for training on the semantics --
            # Consumers that only understand plain COCO ignore these; they are
            # here so a model can be trained to read the value, not just to
            # box it.
            "attributes": {
                "text": a.text,
                "value": a.value,
                "units": annotation_record.units,
                "view": a.view,
                "kind": a.kind.value,
                # Class-specific extras. This carries `synthetic: true`, which
                # is the only way a consumer can tell a generated
                # specification from one measured off the model or read from
                # the file -- so it must reach the COCO file, not stop at the
                # .annotations.json.
                "detail": dict(a.detail or {}),
                "extent_bbox": extent_box,
            },
        })
        next_id += 1

    return {
        "info": {
            "description": "AutoDraft drawing annotations",
            "version": "1.0",
            "date_created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "part": annotation_record.title_block.part_number,
            "model_sha256": annotation_record.model_sha256,
            # The drawing scale is recorded but does NOT affect any value:
            # dimensions are real, full-size measurements regardless of how
            # large the part is drawn.
            "drawing_scale": sheet.scale_text,
            "units": annotation_record.units,
        },
        "licenses": [],
        "images": [{
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": width_px,
            "height": height_px,
        }],
        "categories": CATEGORIES,
        "annotations": objects,
        "feature_list": _feature_list(objects),
    }


def _image_size(path: str):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def write_coco(annotation_record, image_path: str, out_path: str,
               image_size=None) -> str:
    """Write ``<part>.coco.json`` next to the drawing. Returns the path."""
    doc = build_coco(annotation_record, image_path, image_size=image_size)
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return out_path
