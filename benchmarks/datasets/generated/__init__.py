from ..custom import DatasetClass as DatasetClassBase
from ..custom import SampleClass 
from pycocotools.coco import COCO
from pathlib import Path
import os

class DatasetClass(DatasetClassBase):
    def __init__(self) -> None:
        super().__init__()
        dirname = Path(f"{os.path.dirname(os.path.abspath(__file__))}")
        coco = COCO(dirname / 'data' / '_annotations.coco.json')
        img_ids = coco.getImgIds()
        self.samples = [SampleClass(i, coco, dirname / 'data') for i in range(len(img_ids))]