from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Set, Tuple, Type, Union

from ....compute_stats import ComputeStats
from ...safe_ir import ScalarSpec, SpecType, TensorSpec
from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec


@dataclass
class UnsqueezeSpec(OpSpec):
    dimensions: Set[int]

    @property
    def type(self) -> str:
        return "unsqueeze"

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
        positive_real_inds = sorted([idx for idx in self.dimensions if idx >= 0])
        positive_real_inds_set = set(positive_real_inds)

        out_shape = list(inputs[0].shape)
        for i in positive_real_inds:
            if i > len(out_shape):
                raise Exception(
                    f"Cannot unsqueeze more than one past the end of the shape: shape={out_shape} unsqueeze_dim={i}."
                )
            out_shape.insert(i, 1)

        for i in sorted([idx for idx in self.dimensions if idx < 0]):
            real_idx = len(out_shape) + i + 1
            if real_idx in positive_real_inds_set:
                raise Exception(
                    f"Cannot unsqueeze same dimension twice: shape={out_shape} unsqueeze_dim={real_idx} orignal_dim={i}."
                )
            if real_idx < 0:
                raise Exception(
                    f"Cannot unsqueeze past the beginning of the shape: shape={out_shape} unsqueeze_dim={i}."
                )
            out_shape.insert(real_idx, 1)

        return TensorSpec(shape=out_shape, data_type=inputs[0].data_type)

    def compute_stats(self, inputs: List[SpecType]) -> ComputeStats:
        out_spec = self.output_spec(inputs)
        return ComputeStats(
            flops=0,
            reads=out_spec.size(),
            writes=out_spec.size(),
        )
