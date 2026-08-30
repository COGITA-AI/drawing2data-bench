"""Title-block content and structure, varied per drawing.

**Read this before trusting anything in a title block.**

Every value here is INVENTED. A CAD file records the geometry of a part; it
does not record who drew it, who approved it, which revision this is, or
which department owns it. Those come from a PDM system the pipeline has no
access to. They are generated so that the block a detector sees carries the
same variety of fields, names, codes and dates that real drawings do --
rather than the identical eight cells with ``AUTODRAFT`` in every one, which
is what a model would otherwise learn to expect.

The consequences, stated plainly:

* the **field set, layout, labels and formatting are realistic** and are what
  a detector should learn;
* the **content is fiction** -- do not read a name, a revision or a date off
  one of these drawings and believe it;
* everything generated here is reported under ``synthetic: true``, the same
  flag :mod:`autodraft.roughness` and :mod:`autodraft.synth_pmi` use, so the
  whole invented subset is one predicate away from being filtered.

The one exception is the fields that ARE derived from the model: the title,
part number, scale, units and sheet size. Those are real and are passed in by
the caller rather than generated here.

Determinism is by BLAKE2b of the part name, so a drawing is reproducible and
a dataset regenerates byte-for-byte.
"""
from __future__ import annotations

import datetime as _dt
from typing import Dict, List, Tuple

#: Drafter / checker names. Deliberately mundane and international, because a
#: title block full of one nationality is its own bias.
_NAMES = (
    "J. MORAN", "K. TANAKA", "A. SCHMIDT", "L. ROSSI", "P. NOVAK",
    "R. OKAFOR", "S. DUBOIS", "M. HALVORSEN", "T. WRIGHT", "D. KOWALSKI",
    "H. ANDERSSON", "N. PATEL", "C. FERREIRA", "E. LINDQVIST", "B. HORVATH",
    "G. MARTINEZ", "V. PETROV", "W. CHEN", "F. BAUER", "O. LEHTINEN",
)

#: Department / owner strings that appear on real prints.
_DEPARTMENTS = (
    "MECHANICAL DESIGN", "TOOLING", "PRODUCT ENG.", "R&D", "MANUFACTURING",
    "JIG & FIXTURE", "PROTOTYPE SHOP", "QUALITY",
)

#: Materials, with the specification a drawing actually quotes.
_MATERIALS = (
    "AL 6061-T6", "AL 7075-T6", "S355JR", "1.4301 (304)", "1.4404 (316L)",
    "C45E", "42CrMo4", "PA6-GF30", "POM-C", "TI-6AL-4V", "BRASS CuZn39Pb3",
    "MILD STEEL S275", "AL 5083", "PEEK", "ABS",
)

#: Finish / treatment callouts.
_FINISHES = (
    "ANODISE CLEAR", "BLACK OXIDE", "ZINC PLATE", "POWDER COAT RAL 7016",
    "PASSIVATE", "AS MACHINED", "BEAD BLAST", "NONE",
)

#: Density in g/cm3 for each material above, so the WEIGHT field can be
#: COMPUTED from the solid instead of invented. Handbook values, and the only
#: reason they are here: a title block that states a mass is stating a fact
#: about the part, and a random one contradicts the drawing it sits on --
#: 0000_00000386 is a 15 mm washer and its block read "55.661 kg".
_DENSITY = {
    "AL 6061-T6": 2.70, "AL 7075-T6": 2.81, "AL 5083": 2.66,
    "S355JR": 7.85, "MILD STEEL S275": 7.85, "C45E": 7.85, "42CrMo4": 7.85,
    "1.4301 (304)": 7.90, "1.4404 (316L)": 8.00,
    "TI-6AL-4V": 4.43, "BRASS CuZn39Pb3": 8.47,
    "PA6-GF30": 1.36, "POM-C": 1.41, "PEEK": 1.30, "ABS": 1.04,
}


def weight_text(volume_mm3: float, material: str) -> str:
    """Mass of the part AS DRAWN, from its own volume and its material.

    Returned in grams under a kilogram, because that is how a drawing writes
    it -- "0.6 g" and not "0.001 kg". Empty when the volume is unknown, so
    the field is simply left out rather than filled with a guess.
    """
    rho = _DENSITY.get(str(material).strip())
    if not volume_mm3 or volume_mm3 <= 0 or not rho:
        return ""
    grams = volume_mm3 * rho / 1000.0
    if grams < 1000.0:
        return f"{grams:.3g} g" if grams < 10 else f"{grams:.1f} g"
    return f"{grams / 1000.0:.3f} kg"


#: General-tolerance standards, which is a real field on a real block.
_TOLERANCE_STDS = (
    "ISO 2768-mK", "ISO 2768-fH", "ISO 2768-mH", "DIN 7168-m",
    "ASME Y14.5-2018",
)

_REVISIONS = ("-", "A", "B", "C", "D", "01", "02", "03")

#: Optional extra fields, drawn only when the block has rows to spare. This
#: is what makes one block structurally different from the next rather than
#: the same eight cells every time.
_OPTIONAL_FIELDS = (
    "CHECKED", "APPROVED", "REV", "DEPT", "PROJECT", "FINISH",
    "GEN. TOL.", "WEIGHT", "DRAWING NO.",
)


def _iso_date(seed: int) -> str:
    """A plausible recent date, deterministic from the seed."""
    base = _dt.date(2021, 1, 1)
    return (base + _dt.timedelta(days=seed % 1800)).isoformat()


def generate(part_name: str, rng, base: Dict[str, str],
             n_slots: int, volume_mm3: float = 0.0
             ) -> Tuple[Dict[str, str], List[str]]:
    """Build the title-block field map and the order fields appear in.

    ``base`` carries the values that are REAL -- title, part number, scale,
    units, sheet -- and is never overwritten. ``n_slots`` is how many cells
    the block's geometry can hold, so a taller or wider block genuinely shows
    more fields rather than the same ones spaced out.

    Returns ``(fields, order)``.
    """
    fields = dict(base)
    drawn_by = rng.pick(_NAMES)
    # Stored under BOTH the lowercase pipeline key and the upper-case label
    # the grid looks up. Writing only the lowercase form left MATERIAL, DRAWN
    # and DATE reading as None in the block: the renderer's own fallback
    # covered most of them, so the gap only showed as an empty cell once a
    # 5-row block put SHEET and DEPT on the same row.
    for lower, upper, value in (
            ("drawn", "DRAWN", drawn_by),
            ("material", "MATERIAL", rng.pick(_MATERIALS)),
            ("date", "DATE", _iso_date(rng._next()))):
        fields[lower] = fields.get(lower) or value
        fields[upper] = fields[lower]

    # Values for the optional fields. A checker is never the same person as
    # the drafter -- that is the point of checking -- so the pool is drawn
    # from without replacement.
    others = [n for n in _NAMES if n != drawn_by]
    checker = others[rng._next() % len(others)]
    approver = [n for n in others if n != checker][
        rng._next() % max(1, len(others) - 1)]

    pool = {
        "CHECKED": checker,
        "APPROVED": approver,
        "REV": rng.pick(_REVISIONS),
        "DEPT": rng.pick(_DEPARTMENTS),
        "PROJECT": f"{rng.pick('ABCDEFGHJKLMNPRSTUVWXYZ')}"
                   f"{rng.pick('ABCDEFGHJKLMNPRSTUVWXYZ')}"
                   f"-{1000 + rng._next() % 9000}",
        "FINISH": rng.pick(_FINISHES),
        "GEN. TOL.": rng.pick(_TOLERANCE_STDS),
        "WEIGHT": weight_text(volume_mm3, fields.get("MATERIAL", "")),
        "DRAWING NO.": f"{100000 + rng._next() % 900000}",
    }

    # The revision is decided whether or not the block has a cell for it: the
    # REVISIONS table and the revision triangles on the views have to agree
    # with the block, and a drawing at revision C with an empty history is
    # the kind of inconsistency a reader notices immediately.
    fields.setdefault("REV", pool["REV"])

    # The mandatory spine of any title block, in the order a reader expects.
    order = ["TITLE", "PART NUMBER", "MATERIAL", "SCALE", "UNITS",
             "DRAWN", "DATE", "SHEET"]
    fields["CHECKED"] = pool["CHECKED"]

    # Fill the remaining slots with a rotating subset, so blocks differ in
    # WHICH extra fields they carry, not merely how many.
    extras = list(_OPTIONAL_FIELDS)
    start = rng._next() % len(extras)
    extras = extras[start:] + extras[:start]
    for name in extras:
        if len(order) >= n_slots:
            break
        if name in order:
            continue
        if not str(pool.get(name, "")).strip():
            # WEIGHT is computed from the solid and its material; when either
            # is unknown there is no honest value, and a labelled cell with
            # nothing in it is worse than one field fewer.
            continue
        order.append(name)
        fields[name] = pool[name]

    return fields, order
