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

class Feature(BaseModel):
    id: int
    bbox: tuple[int, int, int, int]
    text: str = ""
    category_id: int

class FeatureList(BaseModel):
    feature_list: list[Feature]

    def __getitem__(self, key):
        return self.feature_list[key]

    def __len__(self):
        return len(self.feature_list)

class SampleClassBase(ABC):
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