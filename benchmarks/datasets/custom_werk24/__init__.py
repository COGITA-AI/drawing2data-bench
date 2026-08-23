from ..base import DatasetClassBase, SampleClassBase, ExtractionResult
from pathlib import Path
import base64
import json
import os

class SampleClass(SampleClassBase):
    def __init__(self, id : str):
        self.id : str = id

    def image(self):
        image = base64.b64encode(Path(f"{self.id}/image.png").read_bytes()).decode()
        return image
    
    def ground_truth(self):
        files = {
            "BALLOONS.json": "balloons",
            "META_DATA.json": "metadata",
            "INSIGHTS.json": "insights",
            "FEATURES.json": "features",
            "REFERENCE_POSITIONS.json": "reference_positions",
        }
        json_ground_truth = {}
        for key, name in files.items():
            json_obj = json.loads(Path(f"{self.id}/{key}").read_text())[0]
            json_ground_truth[name] = json_obj
        ground_truth = ExtractionResult.model_validate(json_ground_truth)
        return ground_truth

class DatasetClass(DatasetClassBase):
    def __init__(self) -> None:
        super().__init__()
        dirname = f"{os.path.dirname(os.path.abspath(__file__))}/data"
        sample_paths = os.listdir(dirname)
        self.samples = [SampleClass(f"{dirname}/{sample_path}") for sample_path in sample_paths]

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, key : int) -> SampleClassBase:
        return self.samples[key]