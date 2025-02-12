from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Type, Union

from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec
from ...safe_ir import SpecType, TensorSpec, ScalarSpec
from ....compute_stats import ComputeStats

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
         # L_inv[d] = (L[d] - 1) * stride[d] + (kernel_size[d] - 1) + 1
        out_shape = []
        for i, size in enumerate(inputs[0].shape):
            if i not in self.fold:
                out_shape.append(size)
            else:
                kernel_size, stride = self.fold[i]
                out_shape.append((size - 1) * stride + (kernel_size - 1) + 1)

        return TensorSpec(
            shape=out_shape,
            data_type=inputs[0].data_type
        )

    def compute_stats(self, inputs: List[SpecType]) -> ComputeStats:
        # out_spec = self.output_spec(inputs)
        # input_size = inputs[0].size()
        # output_size = out_spec.size()
        # non-folded-size = prod(non-folded-dims)
        # overlap size per dim = (kernel_size[i] - stride[i]) 
        # number of overlaps per dim = ceil(shape[i] - kernel_size[i] + stride[i]) / stride[i]) - 1
        # flops = non-folded-size * prod (overlap size per dim * number of overlaps per dim)
        # reads = non-folded-size * prod ( ceil(kernel_size[i]/stride[i])*shape[i] )
        # writes = non-folded-size * prod(ceil((N_i - k_i + s_i)/s_i) * k_i)

        ComputeStats(
            flops=0,
            reads=0,
            writes=0
        )