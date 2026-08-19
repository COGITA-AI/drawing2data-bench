from typing import Literal, cast, override
import json
import instructor
from models.base import ModelClassBase
from .prompts import PROMPTS
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from typing import Literal, Optional
from pydantic import BaseModel, Field
from werk24.models.v2.responses import (
    ResponseInsightsComponentDrawing,
    ResponseMetaDataComponentDrawing,
    ResponseBalloons,
    ResponseFeaturesComponentDrawing,
    ResponseReferencePositions
)

from pathlib import Path
from datasets.base import ExtractionResult
import dotenv
import numpy as np
import base64
from io import BytesIO
from PIL import Image

class PositionedFeature(BaseModel):
    reference_id: int
    feature_type: Literal["dimension", "thread", "bore", "chamfer", "radius", "roughness", "gdt"]
    label: str
    nominal_value: Optional[str] = None
    unit: Optional[str] = None
    tolerance: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    label_center: list[int] = Field(..., min_length=2, max_length=2,
        description="[x, y] pixel coordinates of the text label centre.")
    dimension_line: list[list[int]] = Field(default_factory=list,
        description="[[x1,y1],[x2,y2]] pixel endpoints of the dimension/leader line. Empty [] if none.")


class LLMExtractionResult(BaseModel):
    features: list[PositionedFeature]

#główna klasa do wywołania modelu na danym zdjęciu
class ModelClass(ModelClassBase):
    def __init__(self, MAX_TOKENS:int=13000):
        self.MAX_TOKENS:int = MAX_TOKENS
        self.prompts:dict[str,str] = PROMPTS 

        config = dotenv.dotenv_values(".env")

        openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config["OPENROUTER_API_KEY"],
        default_headers={
            "HTTP-Referer": "https://cogita.ai",
            "X-Title":      "Cogita Evaluation Framework",
            },
        )
        self.client = instructor.from_openai(openrouter_client, mode=instructor.Mode.JSON)

    def get_dimensions(self, image: str):
        img_bytes = base64.b64decode(image)
        with Image.open(BytesIO(img_bytes)) as img:
            return img.size

    # wyciąga dane z całej strony według schematu : meta data -> features positioned -> insights
    def forward(self, image:str) -> ExtractionResult: #dostaje base64 zdjęcie, i wyciąga z niego dane

        ASK_SCHEMAS = {"META_DATA":ResponseMetaDataComponentDrawing,"FEATURES_POSITIONED":LLMExtractionResult,"INSIGHTS":ResponseInsightsComponentDrawing}
        # ASK_SCHEMAS = {"INSIGHTS":ResponseInsightsComponentDrawing}
        results:dict[str,tuple[str,str]] = {}
        for ask, schema in ASK_SCHEMAS.items():
            try:
                res = self.extract_image_with_prompt(image, self.prompts[ask].strip(), schema)
                results[ask] = res
            except Exception as exc:
                print(str(exc))
                results[ask] = json.loads("{}")
        extraction_json = {}
        Path("features_positioned.json").write_text(LLMExtractionResult.model_validate(results["FEATURES_POSITIONED"]).model_dump_json())
        Path("metadata.json").write_text(results["META_DATA"].model_dump_json())
        Path("insights.json").write_text(results["INSIGHTS"].model_dump_json())

        width, height = self.get_dimensions(image)
        factor = self.factor(width, height)
        extraction_json = self.to_werk24_results(LLMExtractionResult.model_validate(results["FEATURES_POSITIONED"]), factor)
        extraction_json["metadata"] = json.loads(results["META_DATA"].model_dump_json())
        extraction_json["insights"] = json.loads(results["INSIGHTS"].model_dump_json())
        return extraction_json
        
    #wyciąga dane ze zdjęcia na podstawie promptu i schematu 
    def extract_image_with_prompt(self, image:str, prompt:str, schema:None) -> str:
         
        image_parts = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
        messages:list[ChatCompletionMessageParam] = [
            {"role": "system", "content": prompt},
            {"role": "user",   "content": image_parts}
        ]

        assert self.client is not None, "The model needs to be loaded first."

        res = cast(str, self.client.chat.completions.create(
            model=self.name,
            response_model=schema,
            messages=messages,
            temperature=0,
            max_tokens=self.MAX_TOKENS,
        ))
        return res
    
    # zamiana LLMExtractionResult na poszczególne modele w stylu werk24
    def to_werk24_results(self, extraction: LLMExtractionResult, factor):
        feats = extraction.features

        balloons_data = {"ask_version": "v2", "ask_type": "BALLOONS", "balloons": [
            {"reference_id": f.reference_id, "center": self.scale(f.label_center, factor)}
            for f in feats
        ]}
        
        ref_positions_data = {"ask_version": "v2", "ask_type": "REFERENCE_POSITIONS", "reference_positions": [
            {
                "reference_id": f.reference_id,
                "polygon": {
                    "coordinate_space": "PIXEL_SPACE",
                    "coordinates": self.scale(f.dimension_line if f.dimension_line
                                else [f.label_center, f.label_center], factor),
                },
            }
            for f in feats
        ]}
        

        type_map = {
            "dimension": "dimensions", "thread": "threads", "bore": "bores",
            "chamfer": "chamfers", "radius": "radii", "roughness": "roughnesses", "gdt": "gdnts",
        }
        groups: dict[str, list] = {v: [] for v in type_map.values()}
        for f in feats:
            groups[type_map.get(f.feature_type, "dimensions")].append({
                "reference_id": f.reference_id,
                "label": f.label,
                "confidence": {"score": str(round(f.confidence, 2))},
                **({"nominal_value": f.nominal_value} if f.nominal_value else {}),
                **({"unit": f.unit} if f.unit else {}),
                **({"tolerance": f.tolerance} if f.tolerance else {}),
            })

        features_data = {"ask_version": "v2", "ask_type": "FEATURES",
                        "page_type": "COMPONENT_DRAWING", **groups}
        
        dct = {
            "balloons": balloons_data,
            "reference_positions": ref_positions_data,
            "features": features_data
        }
        return dct
    
    def factor(self, width, height):
        return np.array([width, height]) / 1000

    def scale(self, points, factor):
        points = np.array(points)
        points = points * factor
        return points.tolist()