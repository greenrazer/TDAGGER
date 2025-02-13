from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Type, Union

from ....compute_stats import ComputeStats
from ...safe_ir import ScalarSpec, SpecType, TensorSpec
from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec


@dataclass
class FoldSpec(OpSpec):
    fold: Dict[int, Tuple[int, int]]  # dim -> (kernel_size, stride)

    # TODO: remove and propagate shape through network
    _output_shape_sidecar: List[int]

    @property
    def type(self) -> str:
        return "fold"

    @property
    def input_type(self) -> Type[OpInput]:
        return UnaryTensorInput

    def format_input(self, input: UnaryTensorInput) -> str:
        return f"{self.type}[{self._op_string()}](%{input.input})"

    def _op_string(self) -> str:
        out = []
        last_dim = -1
        for d in sorted([d for d in self.fold.keys() if d >= 0]):
            if d != last_dim + 1:
                out.append("...")
            k, s = self.fold[d]
            out.append(f"{{{d}: kernel={k} stride={s}}}")
            last_dim = d
        out.append("...")
        neg_dims = sorted([d for d in self.fold.keys() if d < 0])
        if len(neg_dims) > 0:
            last_dim = neg_dims[0] - 1
            for d in neg_dims:
                if d != last_dim + 1:
                    out.append("...")
                k, s = self.fold[d]
                out.append(f"{{{d}: kernel={k} stride={s}}}")
                last_dim = d
            if last_dim != -1:
                out.append("...")

        return " ".join(out)

    def output_spec(self, inputs: List[SpecType]) -> SpecType:
        real_indices = [
            (idx if idx >= 0 else len(inputs[0].shape) + idx) for idx in self.fold
        ]

        if len(real_indices) != len(set(real_indices)):
            raise Exception(f"Concrete pad dimensions must be unique: {real_indices}.")

        real_indices_dict = {
            (idx if idx >= 0 else len(inputs[0].shape) + idx): fld
            for idx, fld in self.fold.items()
        }

        seen = {idx: False for idx in real_indices_dict}

        # L_inv[d] = (L[d]//kernel_size - 1) * stride[d] + kernel_size[d]
        out_shape = []
        for i, size in enumerate(inputs[0].shape):
            if i in real_indices_dict:
                kernel_size, stride = real_indices_dict[i]
                out_shape.append((size // kernel_size - 1) * stride + kernel_size)
                seen[i] = True
            else:
                out_shape.append(size)

        if not all(seen.values()):
            raise Exception(f"shape not sufficient for fold spec: {inputs[0].shape}.")

        return TensorSpec(shape=out_shape, data_type=inputs[0].data_type)

    def compute_stats(self, inputs: List[SpecType]) -> ComputeStats:
        return ComputeStats(
            flops=0,
            reads=0,
            writes=0,
        )
