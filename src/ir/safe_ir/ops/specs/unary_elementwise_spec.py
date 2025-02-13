from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Type, Union

from ....compute_stats import ComputeStats
from ...safe_ir import ScalarSpec, SpecType, TensorSpec
from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec


@dataclass
class UnaryElementwiseSpec(OpSpec):
    class UnaryElementwiseType(Enum):
        SIGN = auto()

        NEGATIVE = auto()
        RECIPROCAL = auto()
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
                case UnaryElementwiseSpec.UnaryElementwiseType.SIGN:
                    return "sign"
                case UnaryElementwiseSpec.UnaryElementwiseType.NEGATIVE:
                    return "negative"
                case UnaryElementwiseSpec.UnaryElementwiseType.RECIPROCAL:
                    return "reciprocal"
                case UnaryElementwiseSpec.UnaryElementwiseType.EXP:
                    return "exp"
                case UnaryElementwiseSpec.UnaryElementwiseType.LOG:
                    return "log"

                case UnaryElementwiseSpec.UnaryElementwiseType.SIN:
                    return "sin"
                case UnaryElementwiseSpec.UnaryElementwiseType.COS:
                    return "cos"
                case UnaryElementwiseSpec.UnaryElementwiseType.TAN:
                    return "tan"
                case UnaryElementwiseSpec.UnaryElementwiseType.ARCSIN:
                    return "arcsin"
                case UnaryElementwiseSpec.UnaryElementwiseType.ARCCOS:
                    return "arccos"
                case UnaryElementwiseSpec.UnaryElementwiseType.ARCTAN:
                    return "arctan"
                case UnaryElementwiseSpec.UnaryElementwiseType.SINH:
                    return "sinh"
                case UnaryElementwiseSpec.UnaryElementwiseType.COSH:
                    return "cosh"
                case UnaryElementwiseSpec.UnaryElementwiseType.TANH:
                    return "tanh"
                case UnaryElementwiseSpec.UnaryElementwiseType.ARCSINH:
                    return "arcsinh"
                case UnaryElementwiseSpec.UnaryElementwiseType.ARCCOSH:
                    return "arccosh"
                case UnaryElementwiseSpec.UnaryElementwiseType.ARCTANH:
                    return "arctanh"

    op_type: UnaryElementwiseType

    @property
    def type(self) -> str:
        return f"{self.op_type}"

    @property
    def input_type(self) -> Type[OpInput]:
        return UnaryTensorInput

    def format_input(self, input: UnaryTensorInput) -> str:
        return f"{self.type}(%{input.input})"

    def output_spec(self, inputs: List[SpecType]) -> SpecType:
        return inputs[0]

    def compute_stats(self, inputs: List[SpecType]) -> ComputeStats:
        input_0_size = inputs[0].size()
        output_size = self.output_spec(inputs).size()
        return ComputeStats(
            flops=0,  # one addition per element pair
            reads=input_0_size,  # one read per element
            writes=output_size,  # one write to the output tensor
        )
