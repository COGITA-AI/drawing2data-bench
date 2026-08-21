from abc import ABC, abstractmethod
from pydantic import BaseModel
from werk24.models.v2.responses import (
    ResponseInsightsComponentDrawing,
    ResponseMetaDataComponentDrawing,
    ResponseBalloons,
    ResponseReferencePositions,
    ResponseFeaturesComponentDrawing,
    )

class ExtractionResult(BaseModel):
    balloons: ResponseBalloons
    metadata: ResponseMetaDataComponentDrawing
    insights: ResponseInsightsComponentDrawing
    features: ResponseFeaturesComponentDrawing
    reference_positions: ResponseReferencePositions

class SampleClassBase(ABC):
    def __init__(self, id : str):
        self.id : str = id

    @abstractmethod
    def image(self):
        ...

    @abstractmethod
    def ground_truth(self):
        ...

class DatasetClassBase(ABC):
    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, key: int) -> SampleClassBase:
        ...