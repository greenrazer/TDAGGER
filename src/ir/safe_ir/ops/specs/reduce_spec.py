import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Set, Tuple, Type, Union

from ....compute_stats import ComputeStats
from ...safe_ir import ScalarSpec, SpecType, TensorSpec
from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec


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
        real_indices = [
            (idx if idx >= 0 else len(inputs[0].shape) + idx)
            for idx in self.dimensions
        ]

        if len(real_indices) != len(set(real_indices)):
            raise Exception(f"Concrete pad dimensions must be unique: {real_indices}.")
        
        real_indices = {
            (idx if idx >= 0 else len(inputs[0].shape) + idx) for idx in self.dimensions
        }
        seen = {idx: False for idx in real_indices}

        out_shape = []
        for i, size in enumerate(inputs[0].shape):
            if i in real_indices:
                out_shape.append(1)
                seen[i] = True
            else:
                out_shape.append(size)

        if not all(seen.values()):
            raise Exception(
                f"shape not sufficient for reduction spec: {inputs[0].shape}."
            )

        return TensorSpec(shape=out_shape, data_type=inputs[0].data_type)

    def compute_stats(self, inputs: List[SpecType]) -> ComputeStats:
        real_indices = {
            (idx if idx >= 0 else len(inputs[0].shape) + idx) for idx in self.dimensions
        }

        out_spec = self.output_spec(inputs)
        out_size = out_spec.size()

        reduce_shape = [s for i, s in enumerate(inputs[0].shape) if i in real_indices]
        reduce_size = math.prod(reduce_shape)
        match self.reduction_type:
            case ReduceSpec.ReductionType.SUM:
                return ComputeStats(
                    flops=reduce_size,  # an addition for every reduced element
                    reads=inputs[0].size(),  # must read every input once
                    # a write for every out size and one for every addition
                    writes=out_size * reduce_size,
                )
            case ReduceSpec.ReductionType.PRODUCT:
                return ComputeStats(
                    flops=reduce_size,  # a multiplication for every reduced element
                    reads=inputs[0].size(),  # must read every input once
                    # a write for every out size and one for every multiplication
                    writes=out_size * reduce_size,
                )
            case ReduceSpec.ReductionType.MINIMUM:
                return ComputeStats(
                    flops=reduce_size,  # a comparison for every reduced element
                    reads=inputs[0].size(),  # must read every input once
                    # a write for every out size and one for the minimum
                    writes=out_size + 1,
                )
            case ReduceSpec.ReductionType.MAXIMUM:
                return ComputeStats(
                    flops=reduce_size,  # a comparison for every reduced element
                    reads=inputs[0].size(),  # must read every input once
                    # a write for every out size and one for the maximum
                    writes=out_size + 1,
                )
            case ReduceSpec.ReductionType.MEAN:
                return ComputeStats(
                    flops=reduce_size
                    + 1,  # an addition for every reduced element and one division
                    reads=inputs[0].size(),  # must read every input once
                    # a write for every out size and one for every addition and one for the division
                    writes=out_size * reduce_size + 1,
                )
            case _:
                assert False, "ReductionType not defined"
