from abc import ABC, abstractmethod
from datasets.base import FeatureList

class ModelClassBase(ABC):
    @abstractmethod
    def forward(self, image:str) -> FeatureList:
        ...
