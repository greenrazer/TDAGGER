from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple, Union, Set

from .op_type import OpType
    

@dataclass
class GroupSpec:
    group: List[Union[int, List[int]]]

class GroupType(OpType):
    spec: GroupSpec

    def __init__(self, name, inputs, spec: GroupSpec, debug_sources=[]):
        super().__init__(name, inputs, debug_sources=debug_sources)
        self.spec = spec

    def __str__(self) -> str:
        inp_name = self.inputs["input"]
        out = f"%{self.name}: %{inp_name}[{self.spec}]{self.debug_sources_to_str()}"
        return out

    @property
    def type(self) -> str:
        return "group"

    @property
    def required_input_keys(self) -> List[str]:
        return ["input"]