import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Type, Union

from ....compute_stats import ComputeStats
from ...safe_ir import ScalarSpec, SpecType, SymbolicTensorSpec, TensorSpec
from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec


@dataclass
class UngroupSpec(OpSpec):
    ungroups: Dict[int, List[int]]

    @property
    def type(self) -> str:
        return "ungroup"

    @property
    def input_type(self) -> Type[OpInput]:
        return UnaryTensorInput

    def __post_init__(self):
        for ungroup in self.ungroups.values():
            if ungroup.count(-1) > 1:
                raise Exception("Each ungroup can only have one dynamic dimension.")

    def format_input(self, input: UnaryTensorInput) -> str:
        return f"{self.type}[{self._op_string()}](%{input.input})"

    def _op_string(self):
        out = []
        last_dim = -1
        for d in sorted([d for d in self.ungroups.keys() if d >= 0]):
            if d != last_dim + 1:
                out.append("...")
            ungroup = [str(i) for i in self.ungroups[d]]
            out.append(f"{d}({' '.join(ungroup)})")
            last_dim = d
        out.append("...")
        neg_dims = sorted([d for d in self.ungroups.keys() if d < 0])
        if len(neg_dims) > 0:
            last_dim = neg_dims[0] - 1
            for d in neg_dims:
                if d != last_dim + 1:
                    out.append("...")
                ungroup = [str(i) for i in self.ungroups[d]]
                out.append(f"{d}({' '.join(ungroup)})")
                last_dim = d
            if last_dim != -1:
                out.append("...")

        return f"{' '.join(out)}"

    def output_spec(self, inputs: List[SpecType]) -> SpecType:
        real_indices = [
            (idx if idx >= 0 else len(inputs[0].shape) + idx) for idx in self.ungroups
        ]

        if len(real_indices) != len(set(real_indices)):
            raise Exception(f"Concrete pad dimensions must be unique: {real_indices}.")

        real_indices_dict = {
            (idx if idx >= 0 else len(inputs[0].shape) + idx): ug
            for idx, ug in self.ungroups.items()
        }

        seen = {idx: False for idx in real_indices_dict}

        out_shape = []
        for i, size in enumerate(inputs[0].shape):
            if i in real_indices_dict:
                ungroup = real_indices_dict[i].copy()
                if -1 in ungroup:
                    divisor = math.prod([idx for idx in ungroup if idx != -1])
                    if isinstance(size, int):
                        if size % divisor != 0:
                            raise Exception(f"Cannot reshape: {size} into {ungroup}.")
                    missing_index = ungroup.index(-1)
                    ungroup[missing_index] = size // divisor
                else:
                    if isinstance(size, int):
                        divisor = math.prod([idx for idx in ungroup])
                        if size != divisor:
                            raise Exception(f"Cannot reshape: {size} into {ungroup}.")
                    else:
                        divisor = math.prod(ungroup[1:])
                        ungroup[0] = size // divisor

                out_shape.extend(ungroup)
                seen[i] = True
            else:
                out_shape.append(size)

        if not all(seen.values()):
            raise Exception(
                f"shape not sufficient for ungroup spec: {inputs[0].shape}."
            )

        out_cls = (
            TensorSpec if isinstance(inputs[0], TensorSpec) else SymbolicTensorSpec
        )
        return out_cls(shape=out_shape, data_type=inputs[0].data_type)

    def compute_stats(self, inputs: List[SpecType]) -> ComputeStats:
        out_spec = self.output_spec(inputs)
        return ComputeStats(
            flops=0,  # just a reshaping
            reads=out_spec.size(),
            writes=out_spec.size(),
        )

    def with_removed_dimensions(self, dimensions: List[int]) -> "UngroupSpec":
        new_ungroups_dict = {}
        for ungroups_dim, ungroups_info in self.ungroups.items():
            if ungroups_dim not in dimensions:
                num_before = sum(1 for dim in dimensions if dim < ungroups_dim)
                new_ungroups_dict[ungroups_dim - num_before] = ungroups_info
        return UngroupSpec(ungroups=new_ungroups_dict)
