from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Set, Tuple, Type, Union

from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec
from ...safe_ir import SpecType, TensorSpec, ScalarSpec
from ....compute_stats import ComputeStats


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
        pass
    
    def compute_stats(self, inputs: List[SpecType]) -> ComputeStats:
        pass