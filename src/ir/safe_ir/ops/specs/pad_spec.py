from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Set, Tuple, Type, Union

from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec


@dataclass
class PadSpec(OpSpec):
    class PadMode(Enum):
        REFLECT = auto()
        CIRCULAR = auto()
        CLOSEST_EDGE = auto()

        def from_str(pad_str: str) -> "PadSpec.PadMode":
            match pad_str:
                case "reflect":
                    return PadSpec.PadMode.REFLECT
                case "circular":
                    return PadSpec.PadMode.CIRCULAR
                case "replicate":
                    return PadSpec.PadMode.CLOSEST_EDGE
                case _:
                    raise Exception(f"Pad mode unknown: {pad_str}.")

        def to_str(self) -> str:
            match self:
                case PadSpec.PadMode.REFLECT:
                    return "reflect"
                case PadSpec.PadMode.CIRCULAR:
                    return "circular"
                case PadSpec.PadMode.CLOSEST_EDGE:
                    return "replicate"

        def __str__(self):
            return self.to_str()

    pad: Dict[int, Tuple[int, int]]  # dim -> (pad_before, pad_after)
    pad_mode: Union[Any, PadMode]

    # TODO: remove and propagate dims through network
    _ouptut_dims_sidecar: int

    @property
    def type(self) -> str:
        return "pad"

    @property
    def input_type(self) -> Type[OpInput]:
        return UnaryTensorInput

    def format_input(self, input: UnaryTensorInput) -> str:
        return f"{self.type}[{self._op_string()}](%{input.input})"

    def _op_string(self) -> str:
        out = []
        last_dim = -1
        for d in sorted([d for d in self.pad.keys() if d >= 0]):
            if d != last_dim + 1:
                out.append("...")
            match self.pad[d]:
                case (0, 0):
                    continue
                case (0, r):
                    out.append(f"{{{d}: after={r}}}")
                case (l, 0):
                    out.append(f"{{{d}: before={l}}}")
                case (l, r):
                    out.append(f"{{{d}: before={l} after={r}}}")
            last_dim = d
        out.append("...")
        neg_dims = sorted([d for d in self.pad.keys() if d < 0])
        if len(neg_dims) > 0:
            last_dim = neg_dims[0] - 1
            for d in neg_dims:
                if d != last_dim + 1:
                    out.append("...")
                match self.pad[d]:
                    case (0, 0):
                        continue
                    case (0, r):
                        out.append(f"{{{d}: after={r}}}")
                    case (l, 0):
                        out.append(f"{{{d}: before={l}}}")
                    case (l, r):
                        out.append(f"{{{d}: before={l} after={r}}}")
                last_dim = d
            if last_dim != -1:
                out.append("...")

        return f"{' '.join(out)} | mode={self.pad_mode}"
