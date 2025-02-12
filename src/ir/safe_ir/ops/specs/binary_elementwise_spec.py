from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Type

from ....compute_stats import ComputeStats
from ...safe_ir import ScalarSpec, SpecType, TensorSpec
from ..inputs.binary_tensor_input import BinaryTensorInput
from ..inputs.op_input import OpInput
from .op_spec import OpSpec


@dataclass
class BinaryElementwiseSpec(OpSpec):
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
                case BinaryElementwiseSpec.BinaryElementwiseType.ADD:
                    return "add"
                case BinaryElementwiseSpec.BinaryElementwiseType.MULTIPLY:
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
        if isinstance(inputs[0], TensorSpec) and isinstance(inputs[1], TensorSpec):
            if inputs[0].shape != inputs[1].shape:
                raise Exception(
                    f"All input shapes must be the same: {inputs[0].shape} != {inputs[1].shape}"
                )
            return inputs[0]

        if isinstance(inputs[1], TensorSpec):
            return inputs[1]

        return inputs[0]

    def compute_stats(self, inputs: List[SpecType]) -> ComputeStats:
        # these should be the same
        input_0_size = inputs[0].size()
        input_1_size = inputs[1].size()
        output_size = self.output_spec(inputs).size()
        match self.op_type:
            case BinaryElementwiseSpec.BinaryElementwiseType.ADD:
                return ComputeStats(
                    flops=output_size,  # one addition per element pair
                    reads=input_0_size + input_1_size,  # one read from each tensor
                    writes=output_size,  # one write to the output tensor
                )
            case BinaryElementwiseSpec.BinaryElementwiseType.MULTIPLY:
                return ComputeStats(
                    flops=output_size,  # one multiplication per element pair
                    reads=input_0_size + input_1_size,  # one read from each tensor
                    writes=output_size,  # one write to the output tensor
                )
