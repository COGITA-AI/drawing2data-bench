from ..base import DatasetClassBase, SampleClassBase, ExtractionResult, FeatureList
from pycocotools.coco import COCO
from pathlib import Path
import base64
import json
import os

class SampleClass(SampleClassBase):
    def __init__(self, id, coco, dirname):
        self.img_info = coco.loadImgs(self.coco.getImgIds()[int(id)])[0]
        ann_ids = self.coco.getAnnIds(imgIds=self.img_info['id'])
        self.anns = FeatureList.model_validate({"feature_list": coco.loadAnns(ann_ids)})
        self.path = dirname / self.img_info["file_name"]

    def image(self):
        image = base64.b64encode(self.path.read_bytes()).decode()
        return image
    
    def ground_truth(self):
        return self.anns

class DatasetClass(DatasetClassBase):
    def __init__(self) -> None:
        super().__init__()
        dirname = Path(f"{os.path.dirname(os.path.abspath(__file__))}")
        coco = COCO(dirname / '_annotations.coco.json')
        img_ids = coco.getImgIds()
        self.samples = [SampleClass(i, coco, dirname) for i in range(len(img_ids))]

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, key : int) -> SampleClassBase:
        return self.samples[key]