from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Type, Union

import sympy
from sympy import Eq, Expr, Piecewise

from ....compute_stats import ComputeStats
from ...safe_ir import ScalarSpec, SpecType, SymbolicTensorSpec, TensorSpec
from ..inputs.binary_tensor_input import BinaryTensorInput
from ..inputs.op_input import OpInput
from .op_spec import OpSpec


@dataclass
class BinaryBroadcastElementwiseSpec(OpSpec):
    class BinaryElementwiseType(Enum):
        ADD = auto()
        MULTIPLY = auto()

        def to_infix(self) -> str:
            match self:
                case self.ADD:
                    return "+"
                case self.MULTIPLY:
                    return "*"

        def __str__(self):
            match self:
                case BinaryBroadcastElementwiseSpec.BinaryElementwiseType.ADD:
                    return "add"
                case BinaryBroadcastElementwiseSpec.BinaryElementwiseType.MULTIPLY:
                    return "multiply"

    op_type: BinaryElementwiseType

    @property
    def type(self) -> str:
        return f"binary_{self.op_type}"

    @property
    def input_type(self) -> Type[OpInput]:
        return BinaryTensorInput

    def format_input(self, input: BinaryTensorInput) -> str:
        return f"%{input.input_0} {self.op_type.to_infix()} %{input.input_1}"

    def output_spec(self, inputs: List[SpecType]) -> SpecType:
        if isinstance(inputs[0], (TensorSpec, SymbolicTensorSpec)) and isinstance(
            inputs[1], (TensorSpec, SymbolicTensorSpec)
        ):
            output_shape = self._broadcast_shape(inputs[0].shape, inputs[1].shape)
            out_cls = (
                TensorSpec
                if isinstance(inputs[0], TensorSpec)
                and isinstance(inputs[1], TensorSpec)
                else SymbolicTensorSpec
            )
            return out_cls(shape=output_shape, data_type=inputs[1].data_type)

        if isinstance(inputs[1], (TensorSpec, SymbolicTensorSpec)):
            return inputs[1]

        return inputs[0]

    def compute_stats(self, inputs: List[SpecType]) -> ComputeStats:
        output_size = self.output_spec(inputs).size()
        return ComputeStats(
            flops=output_size,  # one addition per output element
            reads=2 * output_size,  # one read from each tensor for each output element
            writes=output_size,  # one write to the output tensor
        )

    def with_removed_dimensions(
        self, _dimensions: List[int]
    ) -> "BinaryBroadcastElementwiseSpec":
        return self

    def _broadcast_shape(
        self, input_a_shape: List[int], input_b_shape: List[int]
    ) -> List[int]:
        s0, s1 = (
            (input_a_shape, input_b_shape)
            if len(input_a_shape) < len(input_b_shape)
            else (input_b_shape, input_a_shape)
        )
        s0 = list(reversed(s0))
        s1 = list(reversed(s1))
        s0.extend([1] * (len(s1) - len(s0)))

        output_shape = []
        for dim, (dim0, dim1) in enumerate(zip(s0, s1)):
            if isinstance(dim0, int) and isinstance(dim1, int):
                if dim0 == dim1:
                    output_shape.append(dim0)
                elif dim0 == 1:
                    output_shape.append(dim1)
                elif dim1 == 1:
                    output_shape.append(dim0)
                else:
                    raise Exception(
                        "Shapes not broadcastable: {input_a_shape} {input_b_shape}"
                    )
            else:
                output_shape.append(
                    Piecewise(
                        (dim0, Eq(dim0, dim1)),
                        (dim1, Eq(dim0, 1)),
                        (dim0, Eq(dim1, 1)),
                        (sympy.nan, True),  # Else
                    )
                )

        return list(reversed(output_shape))
