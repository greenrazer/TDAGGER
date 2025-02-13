from typing import Callable, List, Tuple, Any

from ..lowerer import Lowerer


class TorchLowerer(Lowerer):
    @property
    def default_passes(self) -> List[Tuple[str, Callable[[Any], Any]]]:
        return []
