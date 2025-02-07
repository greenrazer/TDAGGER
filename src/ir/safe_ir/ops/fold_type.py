from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Union

from .op_type import OpType


@dataclass
class FoldSpec:
    fold: Dict[int, Tuple[int, int, int]]  # dim -> (kernel_size, stride, dialation)

    # TODO: remove and propagate shape through network
    _output_shape_sidecar: List[int]

    def __str__(self):
        out = []
        last_dim = -1
        for d in sorted([d for d in self.fold.keys() if d >= 0]):
            if d != last_dim + 1:
                out.append("...")
            k, s, di = self.fold[d]
            out.append(f"{{{d}: kernel={k} stride={s} dialation={di}}}")
            last_dim = d
        out.append("...")
        neg_dims = sorted([d for d in self.fold.keys() if d < 0])
        if len(neg_dims) > 0:
            last_dim = neg_dims[0] - 1
            for d in neg_dims:
                if d != last_dim + 1:
                    out.append("...")
                k, s, di = self.fold[d]
                out.append(f"{{{d}: kernel={k} stride={s} dialation={di}}}")
                last_dim = d
            if last_dim != -1:
                out.append("...")

        return " ".join(out)

    @property
    def type(self) -> str:
        return "fold"


@dataclass
class UnfoldSpec:
    unfold: Dict[int, Tuple[int, int, int]]  # dim -> (kernel_size, stride, dialation)

    # TODO: remove and propagate shape through network
    _output_shape_sidecar: List[int]

    def __str__(self):
        out = []
        last_dim = -1
        for d in sorted([d for d in self.unfold.keys() if d >= 0]):
            if d != last_dim + 1:
                out.append("...")
            k, s, di = self.unfold[d]
            out.append(f"{{{d}: kernel={k} stride={s} dialation={di}}}")
            last_dim = d
        out.append("...")
        neg_dims = sorted([d for d in self.unfold.keys() if d < 0])
        if len(neg_dims) > 0:
            last_dim = neg_dims[0] - 1
            for d in neg_dims:
                if d != last_dim + 1:
                    out.append("...")
                k, s, di = self.unfold[d]
                out.append(f"{{{d}: kernel={k} stride={s} dialation={di}}}")
                last_dim = d
            if -1 not in self.unfold:
                out.append("...")

        return " ".join(out)

    @property
    def type(self) -> str:
        return "unfold"


class FoldType(OpType):
    spec: Union[FoldSpec, UnfoldSpec]

    def __init__(
        self, name, inputs, spec: Union[FoldSpec, UnfoldSpec], debug_sources=[]
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
