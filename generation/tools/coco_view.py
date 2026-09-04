"""Draw COCO boxes onto their images, so a dataset can be checked by eye.

    python tools/coco_view.py out                       # whole dataset -> out/preview/
    python tools/coco_view.py out --image flange.png    # just one image
    python tools/coco_view.py out --split val           # only the val manifest
    python tools/coco_view.py out --check               # numbers only, no images

Deliberately plain: it reads a COCO JSON, draws a labelled rectangle per
annotation, and writes a PNG beside it. Pillow is the only dependency, and it
is already required for PNG output.

Only the standard COCO fields are relied upon -- ``images``, ``categories``
and ``annotations[].bbox`` -- so this works on any COCO file, not just the
ones AutoDraft writes. The non-standard ``attributes.text`` is shown when it
happens to be there and ignored when it is not.

``--check`` runs the same consistency tests without rendering anything: it is
the part worth putting in CI, because a dataset can be perfectly valid JSON
and still be unusable (boxes off the edge of the image, annotations pointing
at images that were never written, categories nothing refers to).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# One colour per class, keyed by NAME so a class always gets the same colour
# whichever split or dataset it appears in -- cycling by position meant
# `bores` was blue on one sheet and green on the next, which makes two
# previews impossible to compare. Twelve distinct hues, kept apart in
# lightness as well as hue so they survive a greyscale print.
CLASS_COLOURS = {
    "gdnts":         (200, 30, 30),      # red
    "roughnesses":   (30, 140, 60),      # green
    "radii":         (20, 110, 200),     # blue
    "chamferes":     (215, 120, 20),     # orange
    "bores":         (140, 60, 170),     # purple
    "threads":       (0, 140, 150),      # teal
    "dimensions":    (25, 60, 190),      # indigo
    "notes":         (120, 100, 20),     # olive
    "datums":        (220, 40, 140),     # magenta
    "leader_notes":  (90, 160, 40),      # lime
    "tables":        (0, 90, 110),       # dark cyan
    "view_captions": (170, 70, 40),      # brick
}

#: Fallback for a dataset whose classes we do not know: distinct, and stable
#: for a given name rather than for a position in the file.
PALETTE = [
    (200, 30, 30), (20, 110, 200), (30, 140, 60), (215, 120, 20),
    (140, 60, 170), (0, 140, 150), (220, 40, 140), (120, 100, 20),
    (90, 160, 40), (0, 90, 110), (170, 70, 40), (60, 60, 60),
]


def colour_for(name: str, index: int = 0):
    """Colour of a class, by name where we know it and by hash otherwise."""
    if name in CLASS_COLOURS:
        return CLASS_COLOURS[name]
    import hashlib
    h = int.from_bytes(hashlib.blake2b(str(name).encode(), digest_size=4
                                       ).digest(), "big")
    return PALETTE[h % len(PALETTE)]


def load_manifest(path: str) -> tuple:
    """Return (manifest_dict, images_dir).

    ``path`` may be the dataset root, the annotations directory, or a JSON
    file. Accepting all three means the tool does the obvious thing whichever
    one a user happens to type.
    """
    if not os.path.exists(path):
        raise SystemExit(f"no such dataset or file: {path}")
    if os.path.isdir(path):
        # A Roboflow-style split directory holds its own manifest; a dataset
        # root holds train/ valid/ test/. Older layouts kept everything in
        # annotations/, so those names are still accepted.
        for candidate in (os.path.join(path, "_annotations.coco.json"),
                          os.path.join(path, "train",
                                       "_annotations.coco.json"),
                          os.path.join(path, "annotations",
                                       "instances_default.json"),
                          os.path.join(path, "instances_default.json")):
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            raise SystemExit(
                f"no COCO manifest under {path} (looked for "
                "_annotations.coco.json, train/_annotations.coco.json)")
    with open(path) as fh:
        doc = json.load(fh)

    # In the Roboflow layout the images sit beside the manifest. Older
    # layouts put them in a sibling images/ directory.
    ann_dir = os.path.dirname(os.path.abspath(path))
    root = os.path.dirname(ann_dir)
    if os.path.basename(path) == "_annotations.coco.json":
        images_dir = ann_dir
    else:
        for candidate in (os.path.join(root, "images"), ann_dir, root):
            if os.path.isdir(candidate):
                images_dir = candidate
                break
        else:
            images_dir = ann_dir
    return doc, images_dir


def check(doc: dict, images_dir: str) -> list:
    """Consistency problems that make a dataset unusable. Empty == fine."""
    problems = []
    if "images" not in doc or "annotations" not in doc:
        # A non-COCO file would otherwise pass every check below and report
        # "no problems found", which is worse than an error. An empty split
        # is different: `images: []` is a perfectly valid manifest, and a
        # corpus that puts nothing in test/ should not be called broken.
        problems.append("no 'images'/'annotations' array: this does not look "
                        "like a COCO manifest")
        return problems
    image_ids = [im["id"] for im in doc.get("images", [])]
    if len(set(image_ids)) != len(image_ids):
        problems.append("duplicate image ids")
    ann_ids = [a["id"] for a in doc.get("annotations", [])]
    if len(set(ann_ids)) != len(ann_ids):
        problems.append("duplicate annotation ids")

    by_id = {im["id"]: im for im in doc.get("images", [])}
    known_cats = {c["id"] for c in doc.get("categories", [])}

    for im in doc.get("images", []):
        path = os.path.join(images_dir, im["file_name"])
        if not os.path.exists(path):
            problems.append(f"missing image file: {im['file_name']}")

    for a in doc.get("annotations", []):
        image = by_id.get(a["image_id"])
        if image is None:
            problems.append(f"annotation {a['id']} refers to unknown "
                            f"image {a['image_id']}")
            continue
        if a.get("category_id") not in known_cats:
            problems.append(f"annotation {a['id']} has unlisted "
                            f"category {a.get('category_id')}")
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            problems.append(f"annotation {a['id']} has empty bbox {a['bbox']}")
        # a box wholly outside the image is a coordinate-system bug; one
        # pixel over the edge is just rounding, so allow a little slack
        if (x < -1 or y < -1 or x + w > image["width"] + 1
                or y + h > image["height"] + 1):
            problems.append(f"annotation {a['id']} bbox {a['bbox']} outside "
                            f"{image['file_name']} "
                            f"({image['width']}x{image['height']})")

    # An unused category is NOT reported as a problem.
    #
    # This used to be an error, on the reasoning that a declared class with no
    # examples means the labelling has gone wrong. That held when the classes
    # were four generic drawing marks that nearly every part carries. It is
    # wrong now: the classes are specific features (gdnts, threads, chamferes
    # ...) and most parts genuinely have none of them. A plain turned disc has
    # no thread, and a dataset of such parts is correctly labelled with zero
    # thread objects. Flagging that as a defect made `--check` fail on valid
    # data and, worse, would pressure the labeller into inventing examples.
    #
    # The class distribution is still reported by `summarise`, where a zero is
    # visible as information rather than as an error.
    return problems


def summarise(doc: dict) -> str:
    """One-paragraph description of what is in the file."""
    names = {c["id"]: c["name"] for c in doc.get("categories", [])}
    counts = {}
    for a in doc.get("annotations", []):
        key = names.get(a.get("category_id"), f"id {a.get('category_id')}")
        counts[key] = counts.get(key, 0) + 1
    lines = [f"{len(doc.get('images', []))} images, "
             f"{len(doc.get('annotations', []))} annotations"]
    for name in sorted(counts, key=lambda k: -counts[k]):
        lines.append(f"    {counts[name]:>5}  {name}")
    return "\n".join(lines)


def _font(size: int):
    """A truetype font if one is installed, else Pillow's builtin."""
    from PIL import ImageFont
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_image(doc: dict, images_dir: str, image: dict, out_path: str,
               show_text: bool = True, width: int = 0,
               show_extent: bool = False) -> str:
    """Draw one image's boxes and save it. Returns the output path."""
    from PIL import Image, ImageDraw

    names = {c["id"]: c["name"] for c in doc.get("categories", [])}
    order = {c["id"]: i for i, c in enumerate(doc.get("categories", []))}
    anns = [a for a in doc.get("annotations", [])
            if a["image_id"] == image["id"]]

    src = os.path.join(images_dir, image["file_name"])
    img = Image.open(src).convert("RGB")

    # Line and text sizes scale with the image so the overlay is legible on a
    # 3307px sheet and on a thumbnail alike.
    scale = max(img.width, img.height) / 1000.0
    line = max(2, int(round(2 * scale)))
    text_px = max(11, int(round(11 * scale)))
    font = _font(text_px)
    draw = ImageDraw.Draw(img)

    for a in anns:
        colour = colour_for(names.get(a.get("category_id"), ""),
                            order.get(a.get("category_id"), 0))
        if show_extent:
            # the leader/dimension line the text belongs to, drawn thin so it
            # reads as context rather than as the label
            ext = (a.get("attributes") or {}).get("extent_bbox")
            if ext:
                ex, ey, ew, eh = ext
                draw.rectangle([ex, ey, ex + ew, ey + eh],
                               outline=colour, width=max(1, line // 2))
        x, y, w, h = a["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=colour, width=line)

        if not show_text:
            continue
        # The colour already says which category this is (see the legend), so
        # the tag carries the annotation's own text instead. Repeating
        # "linear_dimension" 150 times buried the drawing under labels.
        text = (a.get("attributes") or {}).get("text")
        label = (text.replace("\n", " / ") if text
                 else names.get(a.get("category_id"),
                                str(a.get("category_id"))))
        # The chip is sized to ITS OWN box, not to the sheet: a fixed 11 px
        # tag drawn beside a 9 px view-caption box sat above and to the left
        # of the ink and covered the neighbouring geometry, which read as a
        # misplaced bounding box when the boxes were in fact correct
        # (measured 0.50 mm off on 0000_00000061). See _label.
        _label(draw, label, x, y, w, h, colour, img.size, line, text_px)

    if show_text:
        _legend(draw, font, doc, anns, img.size, line)

    if width and img.width > width:
        height = round(img.height * width / img.width)
        img = img.resize((width, height), Image.LANCZOS)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    img.save(out_path)
    return out_path


def _legend(draw, font, doc, anns, size, pad):
    """Colour key in the top-left, listing only the categories in use."""
    names = {c["id"]: c["name"] for c in doc.get("categories", [])}
    order = {c["id"]: i for i, c in enumerate(doc.get("categories", []))}
    used = {}
    for a in anns:
        used[a.get("category_id")] = used.get(a.get("category_id"), 0) + 1
    if not used:
        return
    try:
        _l, _t, _r, _b = draw.textbbox((0, 0), "Ag", font=font)
        row = (_b - _t) + 2 * pad
    except AttributeError:
        row = draw.textsize("Ag", font=font)[1] + 2 * pad

    x0, y0 = 2 * pad, 2 * pad
    for i, cat in enumerate(sorted(used)):
        colour = colour_for(names.get(cat, ""), order.get(cat, 0))
        y = y0 + i * row
        draw.rectangle([x0, y, x0 + row - pad, y + row - pad], fill=colour)
        draw.text((x0 + row + pad, y),
                  f"{names.get(cat, cat)}  ({used[cat]})",
                  fill=(0, 0, 0), font=font)


_FONT_CACHE = {}


def _sized_font(px: int):
    px = max(7, int(px))
    if px not in _FONT_CACHE:
        _FONT_CACHE[px] = _font(px)
    return _FONT_CACHE[px]


def _label(draw, text, x, y, w, h, colour, size, pad, max_px):
    """Draw `text` in a filled tag on the box, sized to fit that box.

    The tag used to be drawn at one size for the whole sheet. On a drawing
    the objects differ by an order of magnitude -- a title block is 130 mm
    wide and a view caption 9 mm -- so the small boxes disappeared under
    their own labels and the label, sitting up and to the left, looked like a
    misaligned box. The chip now takes its height from the box and shrinks
    further to fit the box's width, and it is never taller than the box it
    annotates plus a little.
    """
    px = int(max(7, min(max_px, h * 0.85 if h > 0 else max_px)))
    font = _sized_font(px)

    def measure(f):
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=f)
            return right - left, bottom - top
        except AttributeError:                # very old Pillow
            return draw.textsize(text, font=f)

    tw, th = measure(font)
    room = max(w, 24)
    if tw > room and tw > 0:
        px = int(max(7, px * room / tw))
        font = _sized_font(px)
        tw, th = measure(font)

    pad = max(1, min(pad, px // 3))
    tx, ty = x, y - th - 2 * pad
    if ty < 0:                                # no room above: put it inside
        ty = y + pad
    tx = max(0, min(tx, size[0] - tw - 2 * pad))
    draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad], fill=colour)
    draw.text((tx + pad, ty + pad), text, fill=(255, 255, 255), font=font)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="coco_view",
        description="Draw COCO boxes on their images, or check a dataset.")
    ap.add_argument("dataset",
                    help="dataset directory, annotations directory, or a "
                         "COCO json file")
    ap.add_argument("-o", "--out", default=None,
                    help="output directory (default: <dataset>/preview)")
    ap.add_argument("--split", default=None,
                    help="which split directory to read, e.g. --split valid "
                         "(default: train)")
    ap.add_argument("--image", default=None,
                    help="only this file_name")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N images (0 = all)")
    ap.add_argument("--width", type=int, default=0,
                    help="downscale output to this width; 0 (the default) "
                         "keeps the drawing's own resolution, which is what "
                         "you need to read a dimension under its box")
    ap.add_argument("--no-text", action="store_true",
                    help="draw boxes only, without labels")
    ap.add_argument("--extent", action="store_true",
                    help="also outline attributes.extent_bbox (the dimension "
                         "line or leader), in a thin dashed-looking box")
    ap.add_argument("--check", action="store_true",
                    help="report consistency problems and exit; writes no "
                         "images. Exit status is non-zero if any are found.")
    a = ap.parse_args(argv)

    target = a.dataset
    if a.split and os.path.isdir(target):
        for candidate in (os.path.join(target, a.split,
                                       "_annotations.coco.json"),
                          os.path.join(target, "annotations",
                                       f"instances_{a.split}.json"),
                          os.path.join(target, f"instances_{a.split}.json")):
            if os.path.exists(candidate):
                target = candidate
                break
        else:
            raise SystemExit(f"no {a.split} split under {a.dataset}")

    doc, images_dir = load_manifest(target)
    print(summarise(doc))

    problems = check(doc, images_dir)
    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for p in problems[:20]:
            print(f"  - {p}", file=sys.stderr)
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more", file=sys.stderr)
    elif a.check:
        print("\nno problems found")
    if a.check:
        return 1 if problems else 0

    out_dir = a.out or os.path.join(
        a.dataset if os.path.isdir(a.dataset)
        else os.path.dirname(os.path.abspath(a.dataset)), "preview")

    images = doc.get("images", [])
    if a.image:
        images = [im for im in images if im["file_name"] == a.image]
        if not images:
            raise SystemExit(f"no image named {a.image} in the manifest")
    if a.limit:
        images = images[:a.limit]

    written = 0
    for im in images:
        src = os.path.join(images_dir, im["file_name"])
        if not os.path.exists(src):
            print(f"  skip {im['file_name']}: file missing", file=sys.stderr)
            continue
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        out_path = os.path.join(out_dir, f"{stem}.boxes.png")
        draw_image(doc, images_dir, im, out_path,
                   show_text=not a.no_text, width=a.width,
                   show_extent=a.extent)
        written += 1

    print(f"\nwrote {written} preview image(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
