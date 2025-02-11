from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Set, Tuple, Type, Union

from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec


@dataclass
class InterleaveSpec(OpSpec):
    interleave_dimensions: Dict[int, int]  # dim -> inbetween dims
    interleave_mode: Any

    @property
    def type(self) -> str:
        return "interleave"

    @property
    def input_type(self) -> Type[OpInput]:
        return UnaryTensorInput

    def format_input(self, input: UnaryTensorInput) -> str:
        return f"{self.type}[{self._op_string()}](%{input.input})"

    def _op_string(self) -> str:
        out = []
        last_dim = -1
        for d in sorted([d for d in self.interleave_dimensions.keys() if d >= 0]):
            if d != last_dim + 1:
                out.append("...")
            i = self.interleave_dimensions[d]
            out.append(f"{{{d}: step={i}}}")
            last_dim = d
        out.append("...")
        neg_dims = sorted([d for d in self.interleave_dimensions.keys() if d < 0])
        if len(neg_dims) > 0:
            last_dim = neg_dims[0] - 1
            for d in neg_dims:
                if d != last_dim + 1:
                    out.append("...")
                i = self.interleave_dimensions[d]
                out.append(f"{{{d}: step={i}}}")
                last_dim = d
            if last_dim != -1:
                out.append("...")

        return f"{' '.join(out)} | mode={self.interleave_mode}"
