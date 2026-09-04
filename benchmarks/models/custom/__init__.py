from .model_download import download_model
from datasets.base import FeatureList, Feature, Category
from models.base import ModelClassBase
from rfdetr import RFDETRLarge
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
import base64
import os

class ModelClass(ModelClassBase):
    def __init__(self):
        super().__init__()

        path = (Path(os.path.dirname(os.path.abspath(__file__))) / "detector.pth")

        if not path.exists():
            download_model()

        self.detector = RFDETRLarge(num_classes=9, resolution=704, pretrain_weights=str(path), device="cuda")
        self.detector.optimize_for_inference()

    def forward(self, image):
        image = Image.open(BytesIO(base64.b64decode(image)))
        detections = self.detector.predict(image, threshold=0.5)
        class_name = detections.data["class_name"]

        lst = []
        for i in range(len(detections.xyxy)):
            try:
                category = Category(class_name[i])
            except:
                continue
            obj = {
                "id": i,
                "bbox": tuple(detections.xyxy[i]),
                "confidence": detections.confidence[i],
                "category": category,
                "text": ""
            }
            lst.append(Feature.model_validate(obj))

        feature_lst = FeatureList.model_validate({"feature_list": lst})

        return feature_lst