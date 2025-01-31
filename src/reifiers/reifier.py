from abc import ABC, abstractmethod
from typing import Any


class Reifier(ABC):
    @abstractmethod
    def export(self) -> Any:
        pass
