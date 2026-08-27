from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Annotated
from enum import Enum

Coordinate = Annotated[int, Field(ge=0)]

class Category(str, Enum):
    GDNT = "gdnt"
    ROUGHNESS = "roughness"
    RADIUS = "radius"
    CHAMFER = "chamfer"
    BORE = "bore"
    THREAD = "thread"
    DIMENSION = "dimension"
    NOTE = "note"
    DATUM = "datum"
    LEADER_NOTE = "leader_note"
    TABLE = "table"
    VIEW_CAPTION = "view_caption"

class Feature(BaseModel):
    id: int = Field(ge=0, description="The unique id of the feature.")
    bbox: tuple[Coordinate, Coordinate, Coordinate, Coordinate] = Field(description="The bounding box containing the text (and only text) of a feature")
    text: str = Field(default="", description="The whole text displayed by the feature")
    category: Category = Field(description="The category of the feature. List of what every category marks:" \
    "1. gdnt - feature control frames "\
    "2. roughness - surface-texture symbols" \
    "3. radius - fillet / corner radius notes" \
    "4. chamfer - chamfer and countersink callouts" \
    "5. bore - hole callouts" \
    "6. thread - thread specifications" \
    "7. dimension - linear, aligned, angular, diameter" \
    "8. note - the paragraph block (e. g. `NOTES:`, `UNLESS OTHERWISE SPECIFIED`)" \
    "9. datum - boxed datum letters with their triangle" \
    "10. leader_note - a short string on a leader (e. g. `THICKNESS 12`, `BODY 2: 70 X 14`)" \
    "11. table - any ruled block of fields" \
    "12. view_captions - the caption under a view (e. g. `TOP`, `VIEW A`, `ITEM 3`)")
    confidence: float = Field(ge=0, le=1, description="Confindence of feature existence")

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
    def ground_truth(self) -> FeatureList:
        ...

class DatasetClassBase(ABC):
    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, key: int) -> SampleClassBase:
        ...