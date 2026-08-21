from models.base import ModelClassBase
from rfdetr import RFDETRLarge
from io import BytesIO
from PIL import Image
import numpy as np
import base64
import os

class ModelClass(ModelClassBase):
    def __init__(self):
        super().__init__()

        self.detector = RFDETRLarge(num_classes=9, resolution=704, pretrain_weights=f"{os.path.dirname(os.path.abspath(__file__))}/detector.pth", device="cuda")
        self.detector.optimize_for_inference()

    def forward(self, image):
        image = Image.open(BytesIO(base64.b64decode(image)))

        xyxy = np.array(self.detector.predict(image, threshold=0.5).xyxy)
        balloons_points = np.concat(((xyxy[:, 0:1] + xyxy[:, 2:3])/2, (xyxy[:, 1:2] + xyxy[:, 3:4])/2), axis=1)

        balloons = {"balloons": {
            "ask_version": "v2",
            "ask_type": "BALLOONS",
            "balloons": []
        }}


        for idx, pt in enumerate(balloons_points):
            balloons["balloons"]["balloons"].append({
                "reference_id": idx,
                "center": pt.tolist()
            })

        return balloons