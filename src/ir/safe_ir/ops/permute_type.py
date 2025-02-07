from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from .op_type import OpType


@dataclass
class PermuteSpec:
    new_permutation: List[int]

    def __str__(self):
        return f"{' '.join(map(str, self.new_permutation))}"


class PermuteType(OpType):
    spec: PermuteSpec

    def __init__(self, name, inputs, spec: PermuteSpec, debug_sources=[]):
        super().__init__(name, inputs, debug_sources=debug_sources)
        self.spec = spec

    def __str__(self) -> str:
        inp_name = self.inputs["input"]
        out = f"%{self.name}: {self.type}[{self.spec}](%{inp_name}){self.debug_sources_to_str()}"
        return out

    @property
    def type(self) -> str:
        return "permute"

    @property
    def required_input_keys(self) -> List[str]:
        return ["input"]
