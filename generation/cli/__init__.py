from .pipeline import make_drawing, batch
from .schema import (Annotation, AnnotationKind, AnnotationRecord,
                     BoundingBox2D, Category, Coordinate, Feature, FeatureList,
                     ProjectionAngle, SheetRecord, TitleBlock, ViewName,
                     ViewRecord)
from .sheet import Style
from .geometry import analyse, load
__all__ = [
    "make_drawing", "batch", "Style", "analyse", "load",
    # structured record of what the sheet says
    "AnnotationRecord", "Annotation", "AnnotationKind", "BoundingBox2D",
    "SheetRecord", "TitleBlock", "ViewRecord", "ViewName", "ProjectionAngle",
    # detection-output schema
    "Coordinate", "Category", "Feature", "FeatureList",
]
__version__ = "0.2.0"
