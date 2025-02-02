from enum import Enum, auto
from typing import Dict, List, Tuple, Union
from .op_type import OpType


class BinaryElementwiseSpec(Enum):
    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()

    def to_infix(self) -> str:
        match self:
            case self.ADD:
                return "+"
            case self.SUBTRACT:
                return "-"
            case self.MULTIPLY:
                return "*"
            case self.DIVIDE:
                return "/"

    def __str__(self):
        match self:
            case self.ADD:
                return "add"
            case self.SUBTRACT:
                return "subtract"
            case self.MULTIPLY:
                return "multiply"
            case self.DIVIDE:
                return "divide"


class BinaryElementwiseType(OpType):
    spec: BinaryElementwiseSpec

    def __init__(self, name, inputs, spec: BinaryElementwiseSpec, debug_sources=[]):
        super().__init__(name, inputs, debug_sources=debug_sources)
        self.spec = spec

    def __str__(self) -> str:
        inp_0_name = self.inputs["input_0"]
        inp_1_name = self.inputs["input_1"]
        out = f"%{self.name}: %{inp_0_name} {self.spec.to_infix()} %{inp_1_name}"
        if len(self.debug_sources) > 0:
            out += f' #{self.debug_sources[0][2]}( "{self.debug_sources[0][0]}", line {self.debug_sources[0][1]} )'
        return out
    
    @property
    def type(self) -> str:
        return f"{self.spec}"

    @property
    def required_input_keys(self) -> List[str]:
        return ["input_0", "input_1"]
