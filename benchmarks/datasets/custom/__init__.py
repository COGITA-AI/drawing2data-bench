from ..base import DatasetClassBase, SampleClassBase, FeatureList
from pycocotools.coco import COCO
from pathlib import Path
import base64
import json
import os

class SampleClass(SampleClassBase):
    def __init__(self, id, coco, dirname):
        super().__init__(id)
        categories = ["gdnts", "roughnesses", "radii", "chamferes", "bores", "threads", "dimensions", "notes", "datums", "leader_notes", "tables", "view_captions"]
        self.img_info = coco.loadImgs(coco.getImgIds()[int(id)])[0]
        ann_ids = coco.getAnnIds(imgIds=self.img_info["id"])
        annotations = coco.loadAnns(ann_ids)
        for annotation in annotations:
            x, y, width, height = annotation["bbox"]
            annotation["bbox"] = (x, y, x + width, y + height)
            annotation["confidence"] = 1
            if "category" not in annotation:
                annotation["category"] = categories[annotation["category_id"]]
            
        self.anns = FeatureList.model_validate({"feature_list": annotations})
        self.path = dirname / self.img_info["file_name"]

    def image(self):
        image = base64.b64encode(self.path.read_bytes()).decode()
        return image
    
    def ground_truth(self):
        return self.anns

class DatasetClass(DatasetClassBase):
    def __init__(self) -> None:
        super().__init__()
        # dirname = Path(f"{os.path.dirname(os.path.abspath(__file__))}")
        # coco = COCO(dirname / 'data' / '_annotations.coco.json')
        # img_ids = coco.getImgIds()
        # self.samples = [SampleClass(i, coco, dirname) for i in range(len(img_ids))]

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, key : int) -> SampleClassBase:
        return self.samples[key]