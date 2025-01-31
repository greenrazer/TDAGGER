from abc import ABC, abstractmethod
from typing import Any


class Exporter(ABC):
    @abstractmethod
    def export(self) -> Any:
        pass
