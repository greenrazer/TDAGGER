from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Set, Tuple, Type, Union

from ....compute_stats import ComputeStats
from ...safe_ir import ScalarSpec, SpecType, TensorSpec
from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec


@dataclass
class SqueezeSpec(OpSpec):
    dimensions: Set[int]

    @property
    def type(self) -> str:
        return "squeeze"

    @property
    def input_type(self) -> Type[OpInput]:
        return UnaryTensorInput

    def format_input(self, input: UnaryTensorInput) -> str:
        return f"{self.type}[{self._op_string()}](%{input.input})"

    def __str__(self):
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

        return " ".join(out)

    def output_spec(self, inputs: List[SpecType]) -> SpecType:
        real_indices = [
            (idx if idx >= 0 else len(inputs[0].shape) + idx) for idx in self.dimensions
        ]

        if len(real_indices) != len(set(real_indices)):
            raise Exception(
                f"Concrete repeat dimensions must be unique: {real_indices}."
            )

        real_indices = {
            (idx if idx >= 0 else len(inputs[0].shape) + idx) for idx in self.dimensions
        }
        seen = {idx: False for idx in real_indices}

        out_shape = []
        for i, size in enumerate(inputs[0].shape):
            if i in real_indices:
                if size != 1:
                    raise Exception(
                        f"cannot squeeze dimension with size greater than 1: dimension={i} size={size}."
                    )
                seen[i] = True
            else:
                out_shape.append(size)

        if not all(seen.values()):
            raise Exception(
                f"shape not sufficient for squeeze spec: {inputs[0].shape}."
            )

        return TensorSpec(shape=out_shape, data_type=inputs[0].data_type)

    def compute_stats(self, inputs: List[SpecType]) -> ComputeStats:
        out_spec = self.output_spec(inputs)
        return ComputeStats(
            flops=0,
            reads=out_spec.size(),
            writes=out_spec.size(),
        )
