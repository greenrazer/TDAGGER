from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Set, Tuple, Type, Union

from ....compute_stats import ComputeStats
from ...safe_ir import ScalarSpec, SpecType, TensorSpec
from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec


@dataclass
class PermuteSpec(OpSpec):
    permutation: Dict[int, int]  # dimension -> new dimension

    @property
    def type(self) -> str:
        return "permute"

    @property
    def input_type(self) -> Type[OpInput]:
        return UnaryTensorInput

    def __post_init__(self):
        if set(self.permutation.keys()) != set(self.permutation.values()):
            raise Exception(
                "Invalid permutation: all keys must appear in values, and all values must appear in keys."
            )

    def format_input(self, input: UnaryTensorInput) -> str:
        return f"{self.type}[{self._op_string()}](%{input.input})"

    def _op_string(self) -> str:
        out = []
        last_dim = -1
        for d in sorted([d for d in self.permutation.keys() if d >= 0]):
            if d != last_dim + 1:
                out.append("...")
            p = self.permutation[d]
            out.append(f"{d} -> {p}")
            last_dim = d
        out.append("...")
        neg_dims = sorted([d for d in self.permutation.keys() if d < 0])
        if len(neg_dims) > 0:
            last_dim = neg_dims[0] - 1
            for d in neg_dims:
                if d != last_dim + 1:
                    out.append("...")
                p = self.permutation[d]
                out.append(f"{d} -> {p}")
                last_dim = d
            if last_dim != -1:
                out.append("...")

        return " ".join(out)

    def output_spec(self, inputs: List[SpecType]) -> SpecType:
        real_indices = [
            (idx if idx >= 0 else len(inputs[0].shape) + idx)
            for idx in self.permutation
        ]

        if len(real_indices) != len(set(real_indices)):
            raise Exception(f"Concrete pad dimensions must be unique: {real_indices}.")

        real_indices_dict = {
            (idx if idx >= 0 else len(inputs[0].shape) + idx): (
                idx2 if idx2 >= 0 else len(inputs[0].shape) + idx2
            )
            for idx, idx2 in self.permutation.items()
        }
        seen = {idx: False for idx in real_indices}

        out_shape = []
        for i, size in enumerate(inputs[0].shape):
            if i in real_indices_dict:
                out_shape.append(inputs[0].shape[real_indices_dict[i]])
                seen[i] = True
            else:
                out_shape.append(size)

        if not all(seen.values()):
            raise Exception(
                f"shape not sufficient for permute spec: {inputs[0].shape}."
            )

        return TensorSpec(shape=out_shape, data_type=inputs[0].data_type)

    def compute_stats(self, inputs: List[SpecType]) -> ComputeStats:
        out_spec = self.output_spec(inputs)
        return ComputeStats(
            flops=0,  # just a reshaping
            reads=inputs[0].size(),
            writes=out_spec.size(),
        )
