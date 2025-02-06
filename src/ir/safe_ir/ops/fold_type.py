from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple, Union, Set

from .op_type import OpType


@dataclass
class FoldSpec:
    fold: Dict[int, Tuple[int, int, int]]

    # TODO: remove and propagate shape through network
    _output_shape_sidecar: List[int]

    @property
    def type(self) -> str:
        return "fold"


@dataclass
class UnfoldSpec:
    unfold: Dict[int, Tuple[int, int, int]]

    @property
    def type(self) -> str:
        return "unfold"


class FoldType(OpType):
    spec: Union[FoldSpec, UnfoldSpec]

    def __init__(
        self, name, inputs, spec: Union[FoldSpec, UnfoldSpec], debug_sources=[]
    ):
        super().__init__(name, inputs, debug_sources=debug_sources)
        self.spec = spec

    def __str__(self) -> str:
        inp_name = self.inputs["input"]
        out = f"%{self.name}: %{inp_name}[{self.spec}]{self.debug_sources_to_str()}"
        return out

    @property
    def type(self) -> str:
        return self.spec.type

    @property
    def required_input_keys(self) -> List[str]:
        return ["input"]
