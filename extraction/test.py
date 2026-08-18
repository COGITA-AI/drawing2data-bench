from rfdetr import RFDETRLarge
import supervision as sv
from PIL import Image
import numpy as np

image = Image.open("image.png")

model = RFDETRLarge(num_classes=9, resolution=704, pretrain_weights="output/checkpoint_best_total.pth", device="cuda")
model.optimize_for_inference()

detections = model.predict(image, threshold=0.5)

color = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66"
])
thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)

annotated_image = image.copy()
annotated_image = bbox_annotator.annotate(annotated_image, detections)
annotated_image.save("annotated.png")