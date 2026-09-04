from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Annotated, Optional
from enum import Enum

Coordinate = Annotated[float, Field(ge=0)]

class Category(str, Enum):
    GDNT = "gdnts"
    ROUGHNESS = "roughnesses"
    RADIUS = "radii"
    CHAMFER = "chamfers"
    BORE = "bores"
    THREAD = "threads"
    DIMENSION = "dimensions"
    NOTE = "notes"
    DATUM = "datums"
    LEADER_NOTE = "leader_notes"
    TABLE = "tables"
    VIEW_CAPTION = "view_captions"

class Feature(BaseModel):
    id: int = Field(ge=0, description="The unique id of the feature.")
    bbox: tuple[Coordinate, Coordinate, Coordinate, Coordinate] = Field(description="The bounding box containing the text (and only text) of a feature")
    text: str = Field(default="", description="The whole text displayed by the feature")
    category: Category = Field(description="The category of the feature. List of what every category marks:" \
    "1. gdnts - feature control frames "\
    "2. roughnesses - surface-texture symbols" \
    "3. radii - fillet / corner radius notes" \
    "4. chamferes - chamfer and countersink callouts" \
    "5. bores - hole callouts" \
    "6. threads - thread specifications" \
    "7. dimensions - linear, aligned, angular, diameter" \
    "8. notes - the paragraph block (e. g. `NOTES:`, `UNLESS OTHERWISE SPECIFIED`)" \
    "9. datums - boxed datum letters with their triangle" \
    "10. leader_notes - a short string on a leader (e. g. `THICKNESS 12`, `BODY 2: 70 X 14`)" \
    "11. tables - any ruled block of fields" \
    "12. view_captions - the caption under a view (e. g. `TOP`, `VIEW A`, `ITEM 3`)")
    confidence: float = Field(ge=0, le=1, description="Confindence of feature existence")

class FeatureList(BaseModel):
    feature_list: list[Feature]
    cost: Optional[float] = 0.0

    def __getitem__(self, key):
        return self.feature_list[key]

    def __len__(self):
        return len(self.feature_list)

class SampleClassBase(ABC):
    def __init__(self, id):
        self.id = id

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