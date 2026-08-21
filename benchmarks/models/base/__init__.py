from abc import ABC, abstractmethod

class ModelClassBase(ABC):
    @abstractmethod
    def forward(self, image:str) -> dict[str,tuple[str,str]]:
        ...
