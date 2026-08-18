import numpy as np
import supervision as sv

from PIL import Image

from rfdetr import RFDETRLarge

image = Image.open("image.png")

model = RFDETRLarge(num_classes=9, resolution=704, pretrain_weights="output_10000/checkpoint_best_total.pth", device="cuda")
model.optimize_for_inference()

detections = model.predict(image, threshold=0.5)

# xyxy = np.array(detections.xyxy)
# print(xyxy)
# print((xyxy[:, 0] + xyxy[:, 2])/2)
# print((xyxy[:, 0] + xyxy[:, 2])/2)
# print(np.concat(((xyxy[:, 0:1] + xyxy[:, 2:3])/2, (xyxy[:, 1:2] + xyxy[:, 3:4])/2), axis=1))
# for pt in xyxy:
#     print(pt)
# print(np.concat(((xyxy[:, 0:1] + xyxy[:, 2:3])/2, (xyxy[:, 1:2] + xyxy[:, 3:4])/2), axis=1))

color = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])
thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)

annotated_image = image.copy()
annotated_image = bbox_annotator.annotate(annotated_image, detections)
# annotated_image.thumbnail((800, 800))
annotated_image.save("annotated.png")