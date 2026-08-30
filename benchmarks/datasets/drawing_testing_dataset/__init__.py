"""A dataset with a single sample, used to test the drawing metrics.

Follows the same COCO layout as ``datasets.custom``: a single
``_annotations.coco.json`` plus one image in this package's ``data/``
folder. The ground-truth features are the ``FeatureList`` of that image's
annotations.
"""
import os
from pathlib import Path

from ..custom import DatasetClass as CustomDatasetClass
from ..custom import SampleClass as CustomSampleClass


class DatasetClass(CustomDatasetClass):
    def __init__(self) -> None:
        dirname = Path(f"{os.path.dirname(os.path.abspath(__file__))}/data")
        from pycocotools.coco import COCO

        coco = COCO(dirname / "_annotations.coco.json")
        img_ids = coco.getImgIds()
        self.samples = [
            CustomSampleClass(i, coco, dirname) for i in range(len(img_ids))
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, key: int):
        return self.samples[key]
