from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Set, Tuple, Type, Union

from ....compute_stats import ComputeStats
from ...safe_ir import ScalarSpec, SpecType, SymbolicTensorSpec, TensorSpec
from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec, ReducedDimension


@dataclass
class SelectSpec(OpSpec):
    select: Dict[int, int]  # dimension -> index

    @property
    def type(self) -> str:
        return "select"

    @property
    def input_type(self) -> Type[OpInput]:
        return UnaryTensorInput

    def format_input(self, input: UnaryTensorInput) -> str:
        return f"{self.type}[{self._op_string()}](%{input.input})"

    def _op_string(self) -> str:
        out = []
        last_dim = -1
        for d in sorted([d for d in self.select if d >= 0]):
            if d != last_dim + 1:
                out.append("...")
            ind = self.select[d]
            out.append(f"{d}[{ind}]")
            last_dim = d
        out.append("...")
        neg_dims = sorted([d for d in self.select if d < 0])
        if len(neg_dims) > 0:
            last_dim = neg_dims[0] - 1
            for d in neg_dims:
                if d != last_dim + 1:
                    out.append("...")
                ind = self.select[d]
                out.append(f"{d}[{ind}]")
                last_dim = d
            if last_dim != -1:
                out.append("...")

        return f"{' '.join(out)}"

    def output_spec(self, inputs: List[SpecType]) -> SpecType:
        real_indices = [
            (idx if idx >= 0 else len(inputs[0].shape) + idx) for idx in self.select
        ]

        if len(real_indices) != len(set(real_indices)):
            raise Exception(
                f"Concrete select dimensions must be unique: {real_indices}."
            )

        real_indices_dict = {
            (idx if idx >= 0 else len(inputs[0].shape) + idx): idx2
            for idx, idx2 in self.select.items()
        }

        seen = {idx: False for idx in real_indices}

        out_shape = []
        for i, size in enumerate(inputs[0].shape):
            if i in real_indices_dict:
                real_select = (
                    real_indices_dict[i]
                    if real_indices_dict[i] >= 0
                    else size + real_indices_dict[i]
                )
                if isinstance(size, int):
                    if real_select >= size:
                        raise Exception(
                            f"Select out of bounds: dimension={i} size={size} index={real_select}"
                        )
                    out_shape.append(1)
                else:
                    out_shape.append(ReducedDimension(size))
                seen[i] = True
            else:
                out_shape.append(size)

        if not all(seen.values()):
            raise Exception(f"shape not sufficient for select spec: {inputs[0].shape}.")

        out_cls = (
            TensorSpec if isinstance(inputs[0], TensorSpec) else SymbolicTensorSpec
        )
        return out_cls(shape=out_shape, data_type=inputs[0].data_type)

    def compute_stats(self, inputs: List[SpecType]) -> ComputeStats:
        out_spec = self.output_spec(inputs)
        return ComputeStats(
            flops=0,
            reads=out_spec.size(),
            writes=out_spec.size(),
        )

    def with_removed_dimensions(self, dimensions: List[int]) -> "SelectSpec":
        new_select_dict = {}
        for select_dim, select_info in self.select.items():
            if select_dim not in dimensions:
                num_before = sum(1 for dim in dimensions if dim < select_dim)
                new_select_dict[select_dim - num_before] = select_info
        return SelectSpec(select=new_select_dict)
