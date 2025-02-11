from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Set, Tuple, Type, Union

from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec


@dataclass
class SliceSpec(OpSpec):
    slice: Dict[int, Tuple[int, int]]  # dim -> (slice_begin, slice_end)

    @property
    def type(self) -> str:
        return "slice"

    @property
    def input_type(self) -> Type[OpInput]:
        return UnaryTensorInput

    def format_input(self, input: UnaryTensorInput) -> str:
        return f"{self.type}[{self._op_string()}](%{input.input})"

    def _op_string(self) -> str:
        out = []
        last_dim = -1
        for d in sorted([d for d in self.slice.keys() if d >= 0]):
            if d != last_dim + 1:
                out.append("...")
            match self.slice[d]:
                case (0, 0):
                    continue
                case (0, r):
                    out.append(f"{{{d}: end={r}}}")
                case (l, 0):
                    out.append(f"{{{d}: begin={l}}}")
                case (l, r):
                    out.append(f"{{{d}: begin={l} end={r}}}")
            last_dim = d
        out.append("...")
        neg_dims = sorted([d for d in self.slice.keys() if d < 0])
        if len(neg_dims) > 0:
            last_dim = neg_dims[0] - 1
            for d in neg_dims:
                if d != last_dim + 1:
                    out.append("...")
                match self.slice[d]:
                    case (0, 0):
                        continue
                    case (0, r):
                        out.append(f"{{{d}: end={r}}}")
                    case (l, 0):
                        out.append(f"{{{d}: begin={l}}}")
                    case (l, r):
                        out.append(f"{{{d}: begin={l} end={r}}}")
                last_dim = d
            if last_dim != -1:
                out.append("...")

        return " ".join(out)
