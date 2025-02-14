from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Set, Tuple, Type, Union

from ....compute_stats import ComputeStats
from ...safe_ir import ScalarSpec, SpecType, TensorSpec
from ..inputs.op_input import OpInput
from ..inputs.unary_tensor_input import UnaryTensorInput
from .op_spec import OpSpec


@dataclass
class SliceSpec(OpSpec):
    slice: Dict[int, Tuple[int, int]]  # dim -> (slice_begin, slice_end)

    @property
    def type(self) -> str:
        return "slice"

    @property
    def input_type(self) -> Type[OpInput]:
        return UnaryTensorInput

    def format_input(self, input: UnaryTensorInput) -> str:
        return f"{self.type}[{self._op_string()}](%{input.input})"

    def _op_string(self) -> str:
        out = []
        last_dim = -1
        for d in sorted([d for d in self.slice.keys() if d >= 0]):
            if d != last_dim + 1:
                out.append("...")
            match self.slice[d]:
                case (0, 0):
                    continue
                case (0, r):
                    out.append(f"{{{d}: end={r}}}")
                case (l, 0):
                    out.append(f"{{{d}: begin={l}}}")
                case (l, r):
                    out.append(f"{{{d}: begin={l} end={r}}}")
            last_dim = d
        out.append("...")
        neg_dims = sorted([d for d in self.slice.keys() if d < 0])
        if len(neg_dims) > 0:
            last_dim = neg_dims[0] - 1
            for d in neg_dims:
                if d != last_dim + 1:
                    out.append("...")
                match self.slice[d]:
                    case (0, 0):
                        continue
                    case (0, r):
                        out.append(f"{{{d}: end={r}}}")
                    case (l, 0):
                        out.append(f"{{{d}: begin={l}}}")
                    case (l, r):
                        out.append(f"{{{d}: begin={l} end={r}}}")
                last_dim = d
            if last_dim != -1:
                out.append("...")

        return " ".join(out)

    def output_spec(self, inputs: List[SpecType]) -> SpecType:
        real_indices = [
            (idx if idx >= 0 else len(inputs[0].shape) + idx) for idx in self.slice
        ]

        if len(real_indices) != len(set(real_indices)):
            raise Exception(
                f"Concrete slice dimensions must be unique: {real_indices}."
            )

        real_indices_dict = {
            (idx if idx >= 0 else len(inputs[0].shape) + idx): sl
            for idx, sl in self.slice.items()
        }
        seen = {idx: False for idx in real_indices}

        out_shape = []
        for i, size in enumerate(inputs[0].shape):
            if i in real_indices_dict:
                raw_begin, raw_end = real_indices_dict[i]
                begin = raw_begin if raw_begin >= 0 else size + raw_begin
                end = raw_end if raw_end >= 0 else size + raw_end
                if begin >= size:
                    raise Exception(
                        f"Begin index must smaller than the size of the dimension: dimension={i} size={size} begin={begin}"
                    )
                if end >= size:
                    raise Exception(
                        f"End index must smaller than the size of the dimension: dimension={i} size={size} end={end}"
                    )
                if begin > end:
                    raise Exception(
                        f"Begin index must be before end index: dimension={i} begin={begin} end={end}"
                    )
                out_shape.append(end - begin + 1)
                seen[i] = True
            else:
                out_shape.append(size)

        if not all(seen.values()):
            raise Exception(f"shape not sufficient for slice spec: {inputs[0].shape}.")

        return TensorSpec(shape=out_shape, data_type=inputs[0].data_type)

    def compute_stats(self, inputs: List[SpecType]) -> ComputeStats:
        out_spec = self.output_spec(inputs)
        return ComputeStats(
            flops=0,
            reads=out_spec.size(),
            writes=out_spec.size(),
        )
