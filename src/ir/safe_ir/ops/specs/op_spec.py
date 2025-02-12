from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Type, Union

from ..inputs.op_input import OpInput
from ...safe_ir import SpecType
from ....compute_stats import ComputeStats


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
    
    @abstractmethod
    def output_spec(self, inputs: List[SpecType]) -> SpecType:
        pass

    @abstractmethod
    def compute_stats(self, inputs: List[SpecType]) -> ComputeStats:
        pass