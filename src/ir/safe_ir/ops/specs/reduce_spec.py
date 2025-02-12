from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Set, Tuple, Type, Union

from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec
from ...safe_ir import SpecType, TensorSpec, ScalarSpec
from ....compute_stats import ComputeStats


@dataclass
class ReduceSpec(OpSpec):
    class ReductionType(Enum):
        SUM = auto()
        PRODUCT = auto()
        MINIMUM = auto()
        MAXIMUM = auto()
        MEAN = auto()

        def __str__(self):
            match self:
                case ReduceSpec.ReductionType.SUM:
                    return "sum"
                case ReduceSpec.ReductionType.PRODUCT:
                    return "prod"
                case ReduceSpec.ReductionType.MAXIMUM:
                    return "max"
                case ReduceSpec.ReductionType.MINIMUM:
                    return "min"
                case ReduceSpec.ReductionType.MEAN:
                    return "mean"

    dimensions: Set[int]
    reduction_type: ReductionType

    @property
    def type(self) -> str:
        return f"reduce_{self.reduction_type}"

    @property
    def input_type(self) -> Type[OpInput]:
        return UnaryTensorInput

    def format_input(self, input: UnaryTensorInput) -> str:
        return f"{self.type}[{self._op_string()}](%{input.input})"

    def _op_string(self) -> str:
        out = []
        last_dim = -1
        for d in sorted([d for d in self.dimensions if d >= 0]):
            if d != last_dim + 1:
                out.append("...")
            out.append(f"{d}")
            last_dim = d
        out.append("...")
        neg_dims = sorted([d for d in self.dimensions if d < 0])
        if len(neg_dims) > 0:
            last_dim = neg_dims[0] - 1
            for d in neg_dims:
                if d != last_dim + 1:
                    out.append("...")
                out.append(f"{d}")
                last_dim = d
            if last_dim != -1:
                out.append("...")

        return f"{' '.join(out)} | reduction={self.type}"

    def output_spec(self, inputs: List[SpecType]) -> SpecType:
        pass
    
    def compute_stats(self, inputs: List[SpecType]) -> ComputeStats:
        pass