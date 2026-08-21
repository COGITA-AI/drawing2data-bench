from ..custom import DatasetClass as DatasetClassBase
from ..custom import SampleClass as SampleClassBase
from ..base.tables import Table
import os
from pathlib import Path

class SampleClass(SampleClassBase):
    def __init__(self, id):
        super().__init__(id)

    def ground_truth(self):
        return Table.model_validate_json((Path(self.id) / "table.json").read_text())

class DatasetClass(DatasetClassBase):
    def __init__(self) -> None:
        super().__init__()
        dirname = f"{os.path.dirname(os.path.abspath(__file__))}/data"
        sample_paths = os.listdir(dirname)
        self.samples = [SampleClass(f"{dirname}/{sample_path}") for sample_path in sample_paths]