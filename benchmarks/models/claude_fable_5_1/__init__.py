from models.base.openrouter import ModelClass as ModelClassBase

class ModelClass(ModelClassBase):
    def __init__(self, MAX_TOKENS=100000):
        super().__init__(MAX_TOKENS)
        self.name: str = "anthropic/claude-fable-5.1"
