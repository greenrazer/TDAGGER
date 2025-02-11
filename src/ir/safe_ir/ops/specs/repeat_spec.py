from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Set, Tuple, Type, Union

from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec


@dataclass
class RepeatSpec:
    repeat: Dict[int, int]  # dim -> num_repeats

    # TODO: remove and propagate shape through network
    _output_dims_sidecar: int

    @property
    def type(self) -> str:
        return "repeat"

    @property
    def input_type(self) -> Type[OpInput]:
        return UnaryTensorInput

    def format_input(self, input: UnaryTensorInput) -> str:
        return f"{self.type}[{self._op_string()}](%{input.input})"

    def _op_string(self) -> str:
        out = []
        last_dim = -1
        for d in sorted([d for d in self.reduce_dimensions if d >= 0]):
            if d != last_dim + 1:
                out.append("...")
            n = self.repeat[d]
            out.append(f"{{{d}: times={n}}}")
            last_dim = d
        out.append("...")
        neg_dims = sorted([d for d in self.reduce_dimensions if d < 0])
        if len(neg_dims) > 0:
            last_dim = neg_dims[0] - 1
            for d in neg_dims:
                if d != last_dim + 1:
                    out.append("...")
                n = self.repeat[d]
                out.append(f"{{{d}: times={n}}}")
                last_dim = d
            if last_dim != -1:
                out.append("...")

        return " ".join(out)
