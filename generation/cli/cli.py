"""AutoDraft CLI.

Run from the repository root with::

    python -m generation.cli "generation/examples/models/*.step" -o dataset

After ``pip install -e generation``, the shorter ``autodraft`` command is
available from any working directory.
"""
from __future__ import annotations
import argparse, glob, os, sys
from .pipeline import batch
from .sheet import SHEETS, Style


def _report_results(res) -> None:
    """Print one line per part, then the dataset summary."""
    for r in res:
        if not r["ok"]:
            print(f"[FAIL] {r['input']}: {r['error']}", file=sys.stderr)
            continue
        print(f"[ok] {r['input']}  views={','.join(r['views'])} "
              f"holes={r['hole_count']} "
              f"-> {', '.join(r['files'].values())}")
    ds = next((r.get("dataset") for r in res if r.get("dataset")), None)
    if ds:
        skipped = f", {len(ds['skipped'])} skipped" if ds["skipped"] else ""
        print(f"[dataset] {ds['images']} images, {ds['annotations']} objects "
              f"({ds['train_images']} train / {ds['val_images']} valid / "
              f"{ds.get('test_images', 0)} test){skipped}")


def _drawing_options(a, style, meta) -> dict:
    """Per-drawing keyword arguments for batch().

    Only the switches the user actually set are forwarded. The sidecar
    switches (record, per-part COCO) belong to per-part mode and no longer
    exist: a dataset build turns them off itself.
    """
    opts = dict(sheet=a.sheet, style=style, hole_table=a.hole_table, dpi=a.dpi)
    if a.no_vary:
        opts["vary"] = False
    return opts


def _build_parser():
    """Construct the CLI parser.

    Split out of main() to keep that function within the complexity
    ratchet: this is a flat list of add_argument calls and grows with every
    new flag, which says nothing about main's own logic.
    """
    ap = argparse.ArgumentParser(prog="autodraft",
        description="Generate labelled drawing images (PNG + COCO) from 3D "
                    "CAD models (STEP/IGES/BREP).")
    ap.add_argument("inputs", nargs="+", help="model files or globs")
    ap.add_argument("-o", "--out", default="dataset", help="output directory "
                                                           "(default: dataset/)")
    ap.add_argument("--sheet", default="A3", choices=sorted(SHEETS))
    ap.add_argument("--progress", dest="progress", action="store_true",
                    default=True,
                    help="show a progress bar on stderr (default)")
    ap.add_argument("--no-progress", dest="progress", action="store_false",
                    help="suppress the progress bar")
    ap.add_argument("-j", "--jobs", type=int, default=None,
                    help="parallel worker processes. Default picks one per "
                         "core when building a dataset (1.5x on the sample "
                         "batch) and serial otherwise, because a worker costs "
                         "~2s to start and only heavier per-part work repays "
                         "that. Use 1 to force serial, 0 for one per core.")
    ap.add_argument("--dpi", type=int, default=200,
                    help="raster resolution of the drawing PNG (default 200)")
    ap.add_argument("--hole-table", dest="hole_table", action="store_true", default=None)
    ap.add_argument("--no-hole-table", dest="hole_table", action="store_false")
    from .dataset import DEFAULT_TEST_FRAC, DEFAULT_VAL_FRAC
    ap.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                    help="fraction of parts held out for validation, split "
                         "deterministically by name (default 0.2; with "
                         "--test-frac this is a 7:2:1 train/valid/test split)")
    ap.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC,
                    help="fraction held out for test/ (default 0.1). Together "
                         "with --val-frac the default is 7:2:1 "
                         "train/valid/test. An empty split is not written.")
    ap.add_argument("--no-vary", action="store_true",
                    help="disable per-drawing style variation (font, stroke "
                         "weights, text sizes, title-block and table layout). "
                         "Variation is ON by default and is deterministic "
                         "from the part name; values and geometry never vary.")
    return ap


def main(argv=None):
    ap = _build_parser()
    a = ap.parse_args(argv)

    files = []
    for pat in a.inputs:
        files.extend(sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else []))
    if not files:
        print("no input files found", file=sys.stderr); return 2

    style = Style()
    meta = {}

    if a.jobs is not None and a.jobs < 0:
        ap.error("--jobs must be >= 0 (0 = one per core, 1 = serial)")
    res = batch(files, a.out, jobs=a.jobs, progress=a.progress,
                val_frac=a.val_frac, test_frac=a.test_frac,
                **_drawing_options(a, style, meta))
    _report_results(res)
    return 0 if all(r["ok"] for r in res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
