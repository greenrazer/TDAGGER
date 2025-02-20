import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Type, Union

from ....compute_stats import ComputeStats
from ...safe_ir import ScalarSpec, SpecType, TensorSpec
from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec


@dataclass
class GroupSpec(OpSpec):
    groups: List[List[int]]

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
            is_consecutive = sublist == list(range(min(sublist), max(sublist) + 1))
            if not is_consecutive:
                raise Exception(f"Group dimensions must be consecutive: {sublist}.")

        if len(flattened) != len(set(flattened)):
            raise Exception("Group dimensions must be unique.")

        sorted_groups = [sorted(sublist) for sublist in self.groups]
        sorted_positive_groups = sorted(
            [lst for lst in sorted_groups if lst and lst[0] >= 0], key=lambda x: x[0]
        )
        sorted_negative_groups = sorted(
            [lst for lst in sorted_groups if lst and lst[0] < 0], key=lambda x: x[0]
        )

        self.groups = sorted_positive_groups + sorted_negative_groups

    def format_input(self, input: UnaryTensorInput) -> str:
        return f"{self.type}[{self._op_string()}](%{input.input})"

    def _op_string(self) -> str:
        out = []
        last_dim = -1
        for group in [lst for lst in self.groups if lst and lst[0] >= 0]:
            if group[0] != last_dim + 1:
                out.append("...")
            out.append(f"({' '.join([str(g) for g in group])})")
            last_dim = group[-1]
        out.append("...")

        negative_groups = [lst for lst in self.groups if lst and lst[0] < 0]
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

    def output_spec(self, inputs: List[SpecType]) -> SpecType:
        groups_real_indexes = sorted(
            [
                [(idx if idx >= 0 else len(inputs[0].shape) + idx) for idx in group]
                for group in self.groups
            ],
            key=lambda x: x[0],
        )

        flattened = []
        for sublist in groups_real_indexes:
            if not sublist:
                continue
            flattened.extend(sublist)

        if len(flattened) != len(set(flattened)):
            raise Exception(
                f"Concrete group dimensions must be unique: {groups_real_indexes}."
            )

        seen = {idx: False for idx in flattened}

        out_shape = []
        curr_group = 0
        for i, size in enumerate(inputs[0].shape):
            if curr_group >= len(groups_real_indexes):
                out_shape.append(size)
            elif i == groups_real_indexes[curr_group][-1]:
                if len(out_shape) == 0:
                    out_shape.append(1)
                out_shape[-1] *= size
                curr_group += 1
                seen[i] = True
            elif i == groups_real_indexes[curr_group][0]:
                out_shape.append(size)
                seen[i] = True
            elif i in groups_real_indexes[curr_group]:
                out_shape[-1] *= size
                seen[i] = True
            else:
                out_shape.append(size)

        if not all(seen.values()):
            raise Exception(f"shape not sufficient for group spec: {inputs[0].shape}.")

        return TensorSpec(shape=out_shape, data_type=inputs[0].data_type)

    def compute_stats(self, inputs: List[SpecType]) -> ComputeStats:
        out_spec = self.output_spec(inputs)
        return ComputeStats(
            flops=0,  # just a reshaping
            reads=out_spec.size(),
            writes=out_spec.size(),
        )
