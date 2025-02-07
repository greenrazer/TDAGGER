from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Union

from .op_type import OpType


@dataclass
class SqueezeSpec:
    dimensions: List[int]

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

    @property
    def type(self):
        return "squeeze"


@dataclass
class UnsqueezeSpec:
    dimensions: List[int]

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

    @property
    def type(self):
        return "unsqueeze"


class SqueezeType(OpType):
    spec: Union[SqueezeSpec, UnsqueezeSpec]

    def __init__(
        self, name, inputs, spec: Union[SqueezeSpec, UnsqueezeSpec], debug_sources=[]
    ):
        super().__init__(name, inputs, debug_sources=debug_sources)
        self.spec = spec

    def __str__(self) -> str:
        inp_name = self.inputs["input"]
        out = f"%{self.name}: {self.type}[{self.spec}](%{inp_name}){self.debug_sources_to_str()}"
        return out

    @property
    def type(self) -> str:
        return self.spec.type

    @property
    def required_input_keys(self) -> List[str]:
        return ["input"]
