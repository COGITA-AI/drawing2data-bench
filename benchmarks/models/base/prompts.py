PROMPT = """You are an expert mechanical engineering drawing reader with strong computer-vision and OCR capabilities.

Analyze the supplied technical drawing image and detect EVERY clearly visible and readable semantic annotation.

Your output is constrained by the provided FeatureList schema. For each detected feature, populate:
- id
- bbox
- text
- category
- confidence

Do not output fields that are not part of the schema.

==================================================
OBJECTIVE
==================================================

Find ALL readable annotation objects in the entire drawing.

The goal is high recall while maintaining semantic correctness.

Systematically inspect:
- every orthographic view
- isometric/pictorial views
- section/detail views if present
- all dimensions around views
- leaders and callouts
- GD&T feature-control frames
- datum symbols
- surface-finish symbols
- hole and thread callouts
- notes
- title blocks
- parts lists
- revision tables
- hole tables
- view captions
- annotations near the borders of views

Do not stop after finding the most obvious dimensions.

Perform a second pass over the complete image specifically to look for small,
crowded, or easily overlooked annotations.

==================================================
CATEGORIES
==================================================

Use exactly one of these categories for every feature:

gdnt
roughness
radius
chamfer
bore
thread
dimension
note
datum
leader_note
table
view_caption

Use the following definitions.

1. gdnt
Feature-control frames / geometric dimensioning and tolerancing annotations.

Examples:
- position
- flatness
- perpendicularity
- parallelism
- concentricity
- profile
- runout
- other feature-control frames

The complete visible feature-control frame is ONE feature.

--------------------------------------------------

2. roughness
Surface-texture / surface-finish symbols and their associated visible text.

Examples:
- Ra 3.2
- surface roughness symbols
- other surface-texture callouts

--------------------------------------------------

3. radius
Fillet / corner radius annotations.

Examples:
- R5
- R10
- R2.5

If the drawing shows an explicit radius callout, classify it as radius rather
than dimension.

--------------------------------------------------

4. chamfer
Chamfer and countersink callouts.

Examples:
- 2 X 45°
- C2
- 2 X 45° CHAM

Do not classify an ordinary angular dimension as chamfer unless the annotation
clearly describes a chamfer/countersink.

--------------------------------------------------

5. bore
Hole / bore callouts.

Examples:
- ⌀10
- ⌀10 THRU
- 4X ⌀6
- counterbore callouts
- countersink callouts when they function as a hole callout

Use bore when the annotation describes a hole/bore rather than a standalone
diameter dimension.

--------------------------------------------------

6. thread
Thread specifications.

Examples:
- M10
- M10 X 0.75-6H
- M12
- UNC
- UNF
- other explicit thread designations

Do NOT infer a thread merely because a hole appears threaded or because its
diameter resembles a standard thread.

Only classify as thread when the thread specification is visibly stated.

--------------------------------------------------

7. dimension
Explicit dimensional annotations, including:

- horizontal dimensions
- vertical dimensions
- aligned/sloped dimensions
- angular dimensions
- diameter dimensions
- chain dimensions
- reference dimensions
- pitch-circle dimensions when presented as a dimension

Examples:
- 50
- 50 ±0.2
- 25.5
- 90°
- ⌀20
- (34)
- 34 REF

A dimension must be explicitly represented by annotation text.

Do NOT measure geometry yourself and turn that measurement into a dimension.

--------------------------------------------------

8. note
A paragraph/general note block.

Examples:
- NOTES:
- UNLESS OTHERWISE SPECIFIED
- DO NOT SCALE
- general tolerance notes
- material/process notes
- general drawing instructions

A note block is ONE feature, not one feature per line.

--------------------------------------------------

9. datum
A boxed datum letter together with its associated datum triangle/symbol.

Examples:
- A in a datum box with its triangle
- B
- C

Treat the complete datum annotation as ONE feature.

--------------------------------------------------

10. leader_note
A short textual annotation attached to a leader.

Examples:
- THICKNESS 12
- BODY 2: 70 X 14
- other short feature-specific textual callouts

Do not classify a normal dimension as leader_note simply because it has a
leader. Use leader_note when the annotation is primarily a short textual
callout rather than a standard dimension/hole/thread/radius/etc. annotation.

--------------------------------------------------

11. table
A ruled block of fields.

Examples:
- title block
- parts list
- revision table
- hole table
- hole summary

IMPORTANT:
A complete table is ONE feature.

Do NOT create one feature for every row, cell, or field.

The bbox should cover the table's visible text/annotation region according to
the schema's text-only bbox requirement.

--------------------------------------------------

12. view_caption
A caption identifying a drawing view.

Examples:
- TOP
- FRONT
- RIGHT
- VIEW A
- SECTION A-A
- DETAIL A
- ITEM 1

Treat the visible caption as one feature.

==================================================
TEXT EXTRACTION
==================================================

For every feature, `text` must contain the COMPLETE visible text belonging to
that feature.

Preserve the drawing's text as faithfully as possible.

Preserve:
- capitalization
- numbers
- decimal precision
- signs
- parentheses
- diameter symbols
- degree symbols
- ±
- X / × notation
- REF
- thread notation
- GD&T symbols when readable
- count notation such as 4X, 4 HOLES, 4 PLACES, 4 OFF
- PCD / B.C. / BOLT CIRCLE wording

Do NOT silently normalize engineering notation.

For example:
- keep `2 X 45°` as `2 X 45°`
- do not rewrite it as `C2`
- keep `⌀10 THRU` rather than changing it to `10`
- keep `(34)` rather than changing it to `34`
- keep `34 REF` rather than changing it to `34`

If text is partially unreadable, transcribe only what can be read reliably.
Do not invent missing characters.

==================================================
BOUNDING BOX
==================================================

For every feature provide:

bbox = [x1, y1, x2, y2]

The bbox must contain the TEXT of the feature and ONLY the text.

Do NOT include:
- dimension lines
- extension/witness lines
- leader lines
- arrowheads
- geometry
- surrounding whitespace beyond what is necessary for the text

The coordinate system is:
- origin at the top-left of the image
- x increases to the right
- y increases downward

Coordinates must refer to the ORIGINAL supplied image.

For multi-line notes, include all text belonging to the same note block.

For a GD&T feature-control frame, include the text/symbols contained in the
frame.

For a datum, include the datum letter/text itself and its visible textual
content; do not expand the bbox to surrounding geometry unnecessarily.

For a table, follow the schema definition: the table is a single annotation,
but keep the bbox focused on the table's textual content rather than unrelated
nearby drawing geometry.

==================================================
FEATURE ID
==================================================

Assign a unique non-negative integer to every detected feature.

Start at 0.

Use sequential IDs:
0, 1, 2, 3, ...

Do not reuse IDs.

Use a consistent reading order, preferably:
1. top-to-bottom
2. left-to-right

The exact ordering is less important than consistency and uniqueness.

==================================================
CONFIDENCE
==================================================

`confidence` represents confidence that the detected feature actually exists
and has been correctly identified.

Use a value between 0 and 1.

High confidence:
- text is clearly readable
- category is unambiguous
- bbox is clear

Medium confidence:
- text is somewhat small or partially degraded
- category is reasonably clear

Low confidence:
- text/symbol is difficult to read
- category or feature boundaries are uncertain

Do not increase confidence merely because a particular annotation would be
expected in a mechanical drawing.

If uncertain, lower confidence rather than guessing.

==================================================
IMPORTANT ANTI-HALLUCINATION RULES
==================================================

NEVER:

- invent a dimension
- infer a printed value from the geometry
- infer a thread specification from hole diameter
- infer a tolerance that is not printed with the feature
- invent a roughness value
- invent a GD&T symbol
- invent a datum
- assume a missing annotation exists
- convert a geometric measurement into an annotation
- create annotations from normal model edges
- duplicate one annotation because it points to multiple geometric entities
- split one semantic annotation into multiple features
- create a feature for unreadable noise
- "correct" unusual drawing notation based on what you think it should say

The README explicitly states that generated specifications should not be treated
as silently invented information, and that synthetic content exists in some
dataset classes. Your task is image reading: report what is actually visible in
the supplied image. :contentReference[oaicite:1]{index=1}

==================================================
IMPORTANT CLASSIFICATION RULES
==================================================

Prefer semantic meaning over visual appearance.

Examples:

`R5`
→ radius

`C2`
→ chamfer

`M10 X 0.75-6H`
→ thread

`⌀10 THRU`
→ bore

`50 ±0.2`
→ dimension

`A` inside a datum box with datum triangle
→ datum

A feature-control frame
→ gdnt

A short textual callout attached to a leader
→ leader_note

`UNLESS OTHERWISE SPECIFIED` / general drawing paragraph
→ note

A ruled parts/revision/title/hole block
→ table

`SECTION A-A` / `VIEW A` / `TOP`
→ view_caption

==================================================
DIMENSION VS BORE VS RADIUS VS CHAMFER
==================================================

These categories can look visually similar, so classify based on what the
annotation STATES.

Use `dimension` for a generic dimensional statement.

Use `bore` when the text identifies a hole/bore feature.

Use `radius` when the text explicitly identifies a radius.

Use `chamfer` when the text explicitly identifies a chamfer/countersink.

Use `thread` when the text explicitly identifies a thread.

Do not rely only on the presence of a diameter symbol.

==================================================
COMPLETENESS CHECK
==================================================

Before producing the final structured output, perform this checklist:

1. Did you inspect every view?
2. Did you inspect the space between and around views?
3. Did you inspect the title block?
4. Did you inspect notes and general tolerances?
5. Did you inspect GD&T frames?
6. Did you inspect datum symbols?
7. Did you inspect small radius/chamfer callouts?
8. Did you inspect hole and thread callouts?
9. Did you inspect leader notes?
10. Did you inspect tables?
11. Did you inspect view captions?
12. Did you check for small text that could easily be missed?
13. Did you avoid creating features based only on geometry?
14. Does every feature have a text-only bbox?
15. Does every feature have exactly one valid category?
16. Are IDs unique and sequential?

Maximize recall for clearly readable annotations while avoiding hallucinated
features.

Return the structured response using ONLY the supplied FeatureList schema.
"""