from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Any

class Lowerer(ABC):
    def __init__(self, passes: List[Tuple[str, Callable[[Any], Any]]] = []):
        self.passes = passes if len(passes) > 0 else self.default_passes

    @property
    @abstractmethod
    def default_passes(self) -> List[Tuple[str, Callable[[Any], Any]]]:
        pass

    def lower(self, graph: Any) -> Any:
        last_graph = graph
        for name, p in self.passes:
            last_graph = p(last_graph)
        return last_graph
