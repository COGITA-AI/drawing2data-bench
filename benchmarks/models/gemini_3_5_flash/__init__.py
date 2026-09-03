from models.base.openrouter import ModelClass as ModelClassBase
import numpy as np

class ModelClass(ModelClassBase):
    def __init__(self, MAX_TOKENS = 100000):
        super().__init__(MAX_TOKENS)
        self.name:str = "google/gemini-3.5-flash"