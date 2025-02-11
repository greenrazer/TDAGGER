from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Type, Union

from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec

@dataclass
class GroupSpec(OpSpec):
    groups: List[List[int]]

    # TODO: remove and propagate shape through network
    _output_shape_sidecar: List[int]

    @property
    def type(self) -> str:
        return "group"

    @property
    def input_type(self) -> Type[OpInput]:
        return UnaryTensorInput

    def __post_init__(self):
        flattened = []

        for sublist in self.groups:
            if not sublist:
                continue
            flattened.extend(sublist)
            is_positive = all(x >= 0 for x in sublist)
            is_negative = all(x < 0 for x in sublist)
            if not (is_positive or is_negative):
                raise Exception(
                    f"Group dimensions must be entirely positive or entirely negitive: {sublist}."
                )

        if len(flattened) != len(set(flattened)):
            raise Exception("Group dimensions must be unique.")

    def format_input(self, input: UnaryTensorInput) -> str:
        return f"{self.type}[{self._op_string()}](%{input.input})"

    def _op_string(self) -> str:
        sorted_groups = [sorted(sublist) for sublist in self.groups]

        out = []
        last_dim = -1
        for group in sorted(
            [lst for lst in sorted_groups if lst and lst[0] >= 0], key=lambda x: x[0]
        ):
            if group[0] != last_dim + 1:
                out.append("...")
            out.append(f"({' '.join(group)})")
            last_dim = group[-1]
        out.append("...")

        negative_groups = sorted(
            [lst for lst in sorted_groups if lst and lst[0] < 0], key=lambda x: x[0]
        )
        if len(negative_groups) > 0:
            last_dim = negative_groups[0][0] - 1
            for group in negative_groups:
                if group[0] != last_dim + 1:
                    out.append("...")
                out.append(f"({' '.join(group)})")
                last_dim = group[-1]
            if last_dim != -1:
                out.append("...")

        return " ".join(out)