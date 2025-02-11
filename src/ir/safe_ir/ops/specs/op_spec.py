from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Type, Union

from ..inputs.op_input import OpInput


class OpSpec(ABC):
    @property
    @abstractmethod
    def type(self) -> str:
        pass

    @property
    @abstractmethod
    def input_type(self) -> Type[OpInput]:
        pass

    @abstractmethod
    def format_input(self, input: OpInput) -> str:
        pass
