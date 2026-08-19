from models.base.openrouter import ModelClass as ModelClassBase
import math

class ModelClass(ModelClassBase):
    def __init__(self, MAX_TOKENS = 13000):
        super().__init__(MAX_TOKENS)
        self.name:str = "openai/gpt-5.4-mini"
        self.max_chunks = 1536
        self.max_dim = 2048

    def factor(self, width: int, height: int):
        maxDim = max(width, height)
        scaled_width = width
        scaled_height = height
        factor_1 = 1
        if maxDim > self.max_dim:
            factor_1 = self.max_dim / maxDim
            scaled_height = round(factor_1 * height)
            scaled_width = round(factor_1 * width)

        low = 0.0
        high = 1.0

        def works(scale: float) -> bool:
            nw = round(scaled_width * scale)
            nh = round(scaled_height * scale)

            chunks_width = math.ceil(nw / 32)
            chunks_height = math.ceil(nh / 32)
            
            return chunks_height * chunks_width <= self.max_chunks

        if works(1.0):
            return factor_1

        while abs(high - low) < 1e-6:
            mid = (low + high) / 2
            if works(mid):
                low = mid
            else:
                high = mid

        factor_2 = low
        
        return factor_1 * factor_2