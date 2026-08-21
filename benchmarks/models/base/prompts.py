PROMPTS = {
    "META_DATA": """You are a technical drawing reader.

Extract structured metadata from the engineering drawing using the provided schema.

Extract:
- All identifiers visible on the drawing (drawing number, part number, revision).
  Use identifier_type values: DRAWING_NUMBER, PART_NUMBER, REVISION, ORDER_NUMBER, CUSTOMER_NUMBER, SUPPLIER_NUMBER, INTERNAL_NUMBER.
- Designation / title of the part.
- All material specifications. For each material set raw_ocr, standard, designation, and material_category.
- Weight if visible (value + unit).
- General tolerances if visible (standard, class, principle).
- General roughness if visible.
- Projection method: FIRST_ANGLE or THIRD_ANGLE.
- Unit system: METRIC or IMPERIAL.
- Language(s) of the drawing text.
- Notes visible on the drawing.
- Bill of materials if present.

Assign a sequential reference_id (starting at 0) to each object.
Only extract what is clearly visible. Do not invent values.
""",

    "FEATURES_POSITIONED": """You are a technical drawing reader with computer vision capability.

Extract ALL labeled dimensions and annotations from the drawing.

For each feature report:
1. Semantic data: feature_type, label (exact text), nominal_value, unit, tolerance, confidence.
2. Pixel coordinates in the image: 
   - label_center: [x, y] — pixel position of the text label centre.
   - dimension_line: [[x1,y1],[x2,y2]] — pixel endpoints of the dimension/leader line.
     Horizontal → [left arrow tip, right arrow tip].
     Vertical   → [top arrow tip, bottom arrow tip].
     Leave empty [] if there is no visible line.

Feature types: dimension, thread, bore, chamfer, radius, roughness, gdt.
Assign reference_id sequentially from 0. Report ALL clearly readable features.
""",

    "INSIGHTS": """You are a technical drawing reader and manufacturing expert.

Analyze the engineering drawing and extract manufacturing insights using the provided schema.

Extract:
- dimensions_after_processing: overall bounding box of the finished part (width, height, depth with unit).
- dimensions_before_processing: raw stock dimensions if indicated, otherwise omit.
- primary_process_options: most likely primary manufacturing process(es).
  Values: CUTTING, TURNING, MILLING, DRILLING, CASTING, FORGING, SHEET_METAL, ADDITIVE, INJECTION_MOULDING, EXTRUSION.
- secondary_processes: finishing operations visible on the drawing (grinding, heat treatment, deburring, etc.).
- volume_estimate: omit unless directly calculable.

Base inferences only on what is visible. Do not invent processes.
""",
}
