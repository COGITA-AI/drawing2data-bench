from models.openai_gpt5_4_mini import ModelClass as ModelClassBase

class ModelClass(ModelClassBase):
    def __init__(self, MAX_TOKENS = 13_000):
        super().__init__(MAX_TOKENS)
        self.name:str = "openai/gpt-5.6-sol"
        # some big numbers
        self.max_chunks = 1_000_000_000
        self.max_dim = 1_000_000_000