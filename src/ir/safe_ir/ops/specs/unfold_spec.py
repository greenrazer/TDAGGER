from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Type, Union

from ....compute_stats import ComputeStats
from ...safe_ir import ScalarSpec, SpecType, SymbolicTensorSpec, TensorSpec
from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec


@dataclass
class UnfoldSpec(OpSpec):
    unfold: Dict[int, Tuple[int, int]]  # dim -> (kernel_size, stride)

    @property
    def type(self) -> str:
        return "unfold"

    @property
    def input_type(self) -> Type[OpInput]:
        return UnaryTensorInput

    def format_input(self, input: UnaryTensorInput) -> str:
        return f"{self.type}[{self._op_string()}](%{input.input})"

    def _op_string(self) -> str:
        out = []
        last_dim = -1
        for d in sorted([d for d in self.unfold.keys() if d >= 0]):
            if d != last_dim + 1:
                out.append("...")
            k, s = self.unfold[d]
            out.append(f"{{{d}: kernel={k} stride={s}}}")
            last_dim = d
        out.append("...")
        neg_dims = sorted([d for d in self.unfold.keys() if d < 0])
        if len(neg_dims) > 0:
            last_dim = neg_dims[0] - 1
            for d in neg_dims:
                if d != last_dim + 1:
                    out.append("...")
                k, s = self.unfold[d]
                out.append(f"{{{d}: kernel={k} stride={s}}}")
                last_dim = d
            if last_dim != -1:
                out.append("...")

        return " ".join(out)

    def output_spec(self, inputs: List[SpecType]) -> SpecType:
        real_indices = [
            (idx if idx >= 0 else len(inputs[0].shape) + idx) for idx in self.unfold
        ]

        if len(real_indices) != len(set(real_indices)):
            raise Exception(
                f"Concrete unfold dimensions must be unique: {real_indices}."
            )

        real_indices_dict = {
            (idx if idx >= 0 else len(inputs[0].shape) + idx): fld
            for idx, fld in self.unfold.items()
        }

        seen = {idx: False for idx in real_indices_dict}

        # L[d] = kernel_size*((x.shape[d] - kernel_size[d]) / stride[d] + 1)

        out_shape = []
        for i, size in enumerate(inputs[0].shape):
            if i in real_indices_dict:
                kernel_size, stride = real_indices_dict[i]
                out_shape.append(kernel_size * ((size - kernel_size) // stride + 1))
                seen[i] = True
            else:
                out_shape.append(size)

        if not all(seen.values()):
            raise Exception(f"shape not sufficient for unfold spec: {inputs[0].shape}.")

        out_cls = (
            TensorSpec if isinstance(inputs[0], TensorSpec) else SymbolicTensorSpec
        )
        return out_cls(shape=out_shape, data_type=inputs[0].data_type)

    def compute_stats(self, inputs: List[SpecType]) -> ComputeStats:
        return ComputeStats(
            flops=0,
            reads=0,
            writes=0,
        )

    def with_removed_dimensions(self, dimensions: List[int]) -> "UnfoldSpec":
        new_unfold_dict = {}
        for unfold_dim, unfold_info in self.unfold.items():
            if unfold_dim not in dimensions:
                num_before = sum(1 for dim in dimensions if dim < unfold_dim)
                new_unfold_dict[unfold_dim - num_before] = unfold_info
        return UnfoldSpec(unfold=new_unfold_dict)
