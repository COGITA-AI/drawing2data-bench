from ..custom_werk24 import DatasetClass as DatasetClassBase
from ..custom_werk24 import SampleClass, SampleClassBase
import os

class DatasetClass(DatasetClassBase):
    def __init__(self) -> None:
        super().__init__()
        dirname = f"{os.path.dirname(os.path.abspath(__file__))}/data"
        sample_paths = os.listdir(dirname)
        self.samples = [SampleClass(f"{dirname}/{sample_path}") for sample_path in sample_paths]