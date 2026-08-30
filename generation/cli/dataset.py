"""Assemble rendered drawings into a Roboflow-style COCO dataset.

The per-part COCO files that :mod:`autodraft.coco` writes are self-contained
but not a *dataset*: every one of them numbers its single image ``1`` and
restarts annotation ids from ``1``, so they cannot simply be concatenated.

Layout produced -- the one Roboflow's COCO importer expects::

    <out>/
      train/
        _annotations.coco.json
        <part>.png
        ...
      valid/
        _annotations.coco.json
        ...
      test/
        _annotations.coco.json
        ...
      README.dataset.txt

The default split is 7:2:1 (train / valid / test). A split is written only
when it holds images, so an empty ``test/`` never appears. Each split directory is self-contained: its manifest lists only the
images sitting beside it, with ``file_name`` a bare name and no directory
part. That is what lets a trainer point at ``train/`` and ``valid/``
independently.

Image and annotation ids restart at 1 within each split, which is what the
importer expects of separate manifests, and every image referenced by a split
manifest is present in that split's folder.

The split itself is decided by a hash of the part name rather than by
position or a random seed: adding a model to the corpus leaves every existing
model in the split it was already in, which keeps a validation set honest
across regenerations.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
from typing import Dict, List, Sequence

from .coco import CATEGORIES, _feature_list, build_coco

# Roboflow's conventional names. "valid", not "val".
SPLITS = ("train", "valid", "test")
MANIFEST_NAME = "_annotations.coco.json"
README_NAME = "README.dataset.txt"

#: Default dataset split, 7 : 2 : 1 train / valid / test.
DEFAULT_VAL_FRAC = 0.2
DEFAULT_TEST_FRAC = 0.1


def _split_of(stem: str, val_frac: float, test_frac: float) -> str:
    """Assign one part to a split, deterministically from its name.

    A hash rather than an index: the corpus grows over time, and splitting by
    position would silently move previously-trained-on parts into validation
    the moment a new model was added.
    """
    val_frac = max(0.0, val_frac)
    test_frac = max(0.0, test_frac)
    if val_frac + test_frac >= 1.0:
        # degenerate request; keep at least something to train on
        val_frac = min(val_frac, 1.0)
        test_frac = max(0.0, 1.0 - val_frac)
    digest = hashlib.sha1(stem.encode("utf-8")).digest()
    frac = int.from_bytes(digest[:4], "big") / 2 ** 32
    if frac < val_frac:
        return "valid"
    if frac < val_frac + test_frac:
        return "test"
    return "train"


def _manifest(images: List[dict], annotations: List[dict],
              description: str) -> dict:
    return {
        "info": {
            "description": description,
            "version": "1.0",
            "year": _dt.datetime.now(_dt.timezone.utc).year,
            "date_created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            # Dimensions are real, full-size millimetres regardless of the
            # scale each sheet was drawn at; see Annotation.value.
            "units": "mm",
        },
        "licenses": [{"id": 1, "name": "Unspecified", "url": ""}],
        "images": images,
        "categories": CATEGORIES,
        "annotations": annotations,
        "feature_list": _feature_list(annotations),
    }


README = """AutoDraft COCO dataset
======================

Object detection labels for automatically generated mechanical drawings.
One image per part; one object per annotation actually drawn on the sheet.

Layout (COCO, as read by Roboflow and by pycocotools):

    train/_annotations.coco.json   plus the training images
    valid/_annotations.coco.json   plus the validation images
    test/_annotations.coco.json    plus the test images

A split is written only when it holds images. The default is a 7:2:1
train / valid / test split (``val_frac=0.2``, ``test_frac=0.1``); an empty
split is not a directory sitting empty on disk.

Each split is self-contained: file_name in a manifest is a bare name, and the
file sits in the same directory as the manifest.

Categories
----------

    1  gdnts         GD&T feature control frames and datum feature symbols
    2  roughnesses   surface-texture symbols, e.g. "Ra 1.6"
    3  radii         fillet / corner radius (R) callouts
    4  chamferes     chamfer and countersink callouts
    5  bores         bore and hole-diameter (⌀) callouts
    6  threads       thread specifications, e.g. "2X M3 x 0.5-6g"
    7  dimensions    linear, aligned and angular size or location
    8  notes         free text: the general NOTES: block, and leader notes
                     that state no specific feature type (THICKNESS, BODY n)

Category id 0 is Roboflow's conventional root and is never used by an
annotation.

The classes are semantic -- what a mark STATES, not how it is drawn. A
plate-thickness leader and a witness-line dimension are both "dimensions"
because both state a size; R callouts are "radii", while bore and diameter
callouts are "bores". The finer distinction is in attributes.detail.form.

An empty class in a split is normal, not an error: most parts carry no thread
and no GD&T.

bbox is [x, y, width, height] in pixels, origin top-left, and covers the
annotation TEXT only -- the number a reader looks at, not the dimension line
or leader pointing at it. A GD&T frame is the exception: its box is the whole
frame, because the boxed compartments ARE the object a reader recognises. A
roughness box likewise includes the ISO 1302 tick as well as the Ra value.

Non-standard fields
-------------------

Each annotation carries an "attributes" block beyond the COCO spec. Plain
COCO readers, including Roboflow's importer, ignore it. It is there so a
model can be trained to read the value, not merely to box it:

    text          the exact string drawn, e.g. "1.36" or "4X D0.34 THRU"
    value         the measured quantity as a number, in millimetres
                  (degrees for angles)
    units         always "mm"
    view          which orthographic view the annotation belongs to
    kind          the internal annotation kind behind category_id
    detail        class-specific extras: "form" (linear/angular/diameter) on
                  a dimension; "characteristic", "datums" and "modifier" on a
                  GD&T frame; "synthetic" on a roughness symbol
    extent_bbox   text plus its dimension line, witness lines or leader

Values are real full-size dimensions: the drawing scale changes how large the
part is drawn, never the number printed beside it.

Provenance, and one caveat
--------------------------

Dimensions, radii, chamfers and bores are measured from the 3D model. GD&T and
threads are read from the file's STEP AP242 PMI and are absent when the file
carries none; no thread is ever inferred from a hole diameter.

Some annotations are SYNTHETIC, and all of them say so: every generated item
carries "synthetic": true in attributes.detail, so the whole set can be
filtered or down-weighted with one predicate.

  roughness   Always synthetic. No CAD file carries finish data and it cannot
              be derived from geometry, so an Ra value is assigned from a hash
              of the part name, drawn from the ISO 1302 preferred series.

  gdnts,      Synthetic ONLY where the source file carried no PMI. One sample
  threads,    file (the NIST AP242 part) has real PMI and is never overridden.
  chamfers    Elsewhere these are generated, because otherwise the three
              classes had instances on a single image out of 18 and none at
              all in valid/test -- unlearnable and unevaluatable.

The GEOMETRY behind a synthetic callout is always real. A thread only ever
goes on a genuine blind or plain-through bore whose diameter is a standard
metric tap drill (counterbored and countersunk holes are clearance holes and
are excluded); a chamfer only on a genuine bore mouth that has no counterbore
already; flatness only on a genuine planar face; position only on a genuine
hole. What is invented is the SPECIFICATION -- the tolerance magnitude, the
thread class, the chamfer angle, the datum letters -- chosen from real
standard values (ISO 965 pitches, ASME Y14.5 characteristics, R10 tolerance
magnitudes) and kept internally consistent: a datum letter is never cited
before it is defined, a form tolerance never carries a diameter symbol, and a
positional zone is never looser than the feature it locates.

So the marks and their relationships are realistic and are what a detector
should learn, but the NUMBERS do not describe the real part. Do not quote them
and do not use them as regression targets.
"""


def build_dataset(results: Sequence[dict], out_dir: str,
                  val_frac: float = DEFAULT_VAL_FRAC,
                  test_frac: float = DEFAULT_TEST_FRAC,
                  description: str = "AutoDraft drawing annotations") -> dict:
    """Write a Roboflow-style COCO dataset from rendered parts.

    ``results`` is what :func:`autodraft.pipeline.batch` returns. Parts that
    failed to render, or that produced no PNG, are skipped and counted in the
    returned summary rather than aborting the dataset.

    Images are *moved* into their split directory, not copied: the renderer
    already wrote them once and a dataset of large PNGs should not be
    duplicated on disk.

    Returns a summary dict: counts per split, the files written, and any
    parts that were skipped.
    """
    # -- collect the renders, deciding each one's split ------------------- #
    # Sorted by input path so ids are stable between runs: a parallel batch
    # completes out of order and would otherwise renumber everything.
    ok = [r for r in results if r.get("ok", True)]
    ok.sort(key=lambda r: str(r.get("input", "")))

    per_split: Dict[str, list] = {s: [] for s in SPLITS}
    skipped: List[str] = []
    seen: set = set()

    for r in ok:
        png = (r.get("files") or {}).get("png")
        arec = r.get("annotation_record")
        if not png or arec is None or not os.path.exists(png):
            skipped.append(str(r.get("input", "?")))
            continue

        # Two inputs can render to the same PNG -- the same file passed twice,
        # or two models with the same stem in different folders. The second
        # render overwrites the first, so listing both would put one image in
        # a manifest twice and duplicate every one of its annotations.
        name = os.path.basename(png)
        if name in seen:
            skipped.append(str(r.get("input", "?")))
            continue
        seen.add(name)

        stem = os.path.splitext(name)[0]
        per_split[_split_of(stem, val_frac, test_frac)].append((png, arec, r))

    # -- write each split as a self-contained COCO directory -------------- #
    written: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    total_images = total_annotations = 0

    for split in SPLITS:
        entries = per_split[split]
        counts[split] = len(entries)
        # An empty split gets no directory and no manifest: a tiny corpus
        # that happens to hash nothing into test/ would otherwise leave an
        # empty directory cluttering the output. A split only appears once it
        # has images, so a trainer expecting it gets a directory, not a ghost.
        if not entries:
            continue
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        images: List[dict] = []
        annotations: List[dict] = []
        # Ids restart per split, because each manifest stands alone.
        next_image_id = 1
        next_ann_id = 1

        for png, arec, result in entries:
            dest = os.path.join(split_dir, os.path.basename(png))
            if os.path.abspath(dest) != os.path.abspath(png):
                shutil.move(png, dest)
                files = result.get("files")
                if isinstance(files, dict):
                    files["png"] = dest      # keep the result honest

            doc = build_coco(arec, dest, image_id=next_image_id,
                             start_annotation_id=next_ann_id)
            images.append(doc["images"][0])
            annotations.extend(doc["annotations"])
            next_image_id += 1
            next_ann_id += len(doc["annotations"])

        path = os.path.join(split_dir, MANIFEST_NAME)
        with open(path, "w") as fh:
            json.dump(_manifest(images, annotations,
                                f"{description} ({split})"), fh, indent=2)
        written[f"{split}/{MANIFEST_NAME}"] = path
        total_images += len(images)
        total_annotations += len(annotations)

    return {
        "images": total_images,
        "annotations": total_annotations,
        "train_images": counts["train"],
        "val_images": counts["valid"],
        "test_images": counts["test"],
        "skipped": skipped,
        "files": written,
    }
