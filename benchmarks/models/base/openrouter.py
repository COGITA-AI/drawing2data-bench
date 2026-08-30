from typing import Literal, cast, override
import instructor
from models.base import ModelClassBase
from .prompts import PROMPT
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from datasets.base import FeatureList
import dotenv
import numpy as np
import base64
from io import BytesIO
from PIL import Image

class ModelClass(ModelClassBase):
    def __init__(self, MAX_TOKENS:int=13000):
        self.MAX_TOKENS:int = MAX_TOKENS
        self.prompts:dict[str,str] = PROMPT 

        config = dotenv.dotenv_values(".env")

        openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config["OPENROUTER_API_KEY"],
        )
        self.client = instructor.from_openai(openrouter_client, mode=instructor.Mode.JSON)

    def get_dimensions(self, image: str):
        img_bytes = base64.b64decode(image)
        with Image.open(BytesIO(img_bytes)) as img:
            return img.size

    def forward(self, image:str) -> FeatureList: 
        result_str = self.extract_image_with_prompt(image, self.prompt.strip(), FeatureList)
        result = FeatureList.model_validate_json(result_str)
        width, height = self.get_dimensions(image)
        factor = self.factor(width, height)
        for feature in result.feature_list:
            feature.bbox = self.scale(feature.bbox, factor)

        return result
        
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
    
    def factor(self, width, height):
        return np.array([width, height]) / 1000

    def scale(self, points, factor):
        points = np.array(points)
        points = points * factor
        return points.tolist()