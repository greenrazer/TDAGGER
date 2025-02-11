from dataclasses import dataclass
from enum import Enum, auto
from typing import Type

from ..inputs.binary_tensor_input import BinaryTensorInput
from ..inputs.op_input import OpInput
from .op_spec import OpSpec


@dataclass
class BinaryElementwiseSpec(OpSpec):
    class BinaryElementwiseType(Enum):
        ADD = auto()
        MULTIPLY = auto()
        EXPONENTIATE = auto()

        def to_infix(self) -> str:
            match self:
                case self.ADD:
                    return "+"
                case self.MULTIPLY:
                    return "*"
                case self.EXPONENTIATE:
                    return "^"

        def __str__(self):
            match self:
                case BinaryElementwiseSpec.BinaryElementwiseType.ADD:
                    return "add"
                case BinaryElementwiseSpec.BinaryElementwiseType.MULTIPLY:
                    return "multiply"
                case BinaryElementwiseSpec.BinaryElementwiseType.EXPONENTIATE:
                    return "exponentiate"

    op_type: BinaryElementwiseType

    @property
    def type(self) -> str:
        return f"binary_{self.op_type}"

    @property
    def input_type(self) -> Type[OpInput]:
        return BinaryTensorInput

    def format_input(self, input: BinaryTensorInput) -> str:
        return f"%{input.input_0} {self.op_type.to_infix()} %{input.input_1}"
