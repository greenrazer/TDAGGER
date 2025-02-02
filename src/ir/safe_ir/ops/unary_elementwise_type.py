from enum import Enum, auto
from typing import Dict, List, Tuple, Union

from .op_type import OpType


class UnaryElementwiseSpec(Enum):
    ABSOLUTE_VALUE = auto()
    NEGATIVE = auto()
    INVERSE = auto()
    SQUARE_ROOT = auto()
    EXP = auto()
    LOG = auto()

    SIN = auto()
    COS = auto()
    TAN = auto()
    ARCSIN = auto()
    ARCCOS = auto()
    ARCTAN = auto()
    SINH = auto()
    COSH = auto()
    TANH = auto()
    ARCSINH = auto()
    ARCCOSH = auto()
    ARCTANH = auto()

    def __str__(self):
        match self:
            case self.ABSOLUTE_VALUE:
                return "abs"
            case self.NEGATIVE:
                return "negative"
            case self.INVERSE:
                return "inverse"
            case self.SQUARE_ROOT:
                return "square_root"
            case self.EXP:
                return "exp"
            case self.LOG:
                return "log"

            case self.SIN:
                return "sin"
            case self.COS:
                return "cos"
            case self.TAN:
                return "tan"
            case self.ARCSIN:
                return "arcsin"
            case self.ARCCOS:
                return "arccos"
            case self.ARCTAN:
                return "arctan"
            case self.SINH:
                return "sinh"
            case self.COSH:
                return "cosh"
            case self.TANH:
                return "tanh"
            case self.ARCSINH:
                return "arcsinh"
            case self.ARCCOSH:
                return "arccosh"
            case self.ARCTANH:
                return "arctanh"


class UnaryElementwiseType(OpType):
    spec: UnaryElementwiseSpec

    def __init__(self, name, inputs, spec: UnaryElementwiseSpec, debug_sources=[]):
        super().__init__(name, inputs, debug_sources=debug_sources)
        self.spec = spec

    def __str__(self) -> str:
        inp_name = self.inputs["input"]
        out = f"%{self.name}: {self.type}(%{inp_name})"
        if len(self.debug_sources) > 0:
            out += f' #{self.debug_sources[0][2]}( "{self.debug_sources[0][0]}", line {self.debug_sources[0][1]} )'
        return out

    @property
    def type(self) -> str:
        return f"{self.spec}"

    @property
    def required_input_keys(self) -> List[str]:
        return ["input"]
