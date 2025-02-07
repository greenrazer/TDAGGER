from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Union

from .op_type import OpType


@dataclass
class ReduceSpec:
    class ReductionType(Enum):
        SUM = auto()
        MEAN = auto()
        MIN = auto()
        MAX = auto()
        PROD = auto()

        def __str__(self):
            match self:
                case ReduceSpec.ReductionType.SUM:
                    return "sum"
                case ReduceSpec.ReductionType.MEAN:
                    return "mean"
                case ReduceSpec.ReductionType.MAX:
                    return "max"
                case ReduceSpec.ReductionType.MIN:
                    return "min"
                case ReduceSpec.ReductionType.PROD:
                    return "prod"

    reduce_dimensions: List[int]
    reduction_type: ReductionType

    def __post_init__(self):
        if not self.reduce_dimensions:
            raise Exception("Must specify at least one reduction dimension")

        if len(set(self.reduce_dimensions)) != len(self.reduce_dimensions):
            raise Exception("Duplicate dimensions in reduce_dimensions")

    def __str__(self):
        # Separate positive and negative numbers and sort them independently
        pos_dims = sorted([d for d in self.reduce_dimensions if d >= 0])
        neg_dims = sorted([d for d in self.reduce_dimensions if d < 0])

        parts = []
        prev_included = False
        prev_num = None

        # Process positive dimensions first
        if pos_dims:
            for i in range(pos_dims[0], pos_dims[-1] + 1):
                if i in pos_dims:
                    if prev_included and parts[-1] != "..." and i != prev_num + 1:
                        parts.append("...")
                    parts.append(str(i))
                    prev_included = True
                    prev_num = i
                elif prev_included:
                    parts.append("...")
                    prev_included = False

        # Then process negative dimensions
        if neg_dims:
            prev_included = False
            prev_num = None
            if parts and parts[-1] != "...":
                parts.append("...")

            for i in range(neg_dims[0], neg_dims[-1] - 1, -1):
                if i in neg_dims:
                    if prev_included and parts[-1] != "..." and i != prev_num - 1:
                        parts.append("...")
                    parts.append(str(i))
                    prev_included = True
                    prev_num = i
                elif prev_included:
                    parts.append("...")
                    prev_included = False

        # Add leading ellipsis if not starting at 0
        if not pos_dims or pos_dims[0] != 0:
            parts.insert(0, "...")

        # Add trailing ellipsis if not ending in -1
        if not neg_dims or neg_dims[-1] != -1:
            parts.append("...")

        return f"{' '.join(parts)} | {self.reduction_type}"


class ReduceType(OpType):
    spec: ReduceSpec

    def __init__(self, name, inputs, spec: ReduceSpec, debug_sources=[]):
        super().__init__(name, inputs, debug_sources=debug_sources)
        self.spec = spec

    def __str__(self) -> str:
        inp_name = self.inputs["input"]
        out = f"%{self.name}: reduce[{self.spec}](%{inp_name}){self.debug_sources_to_str()}"
        return out

    @property
    def type(self) -> str:
        return f"reduce_{self.spec.reduction_type}"

    @property
    def required_input_keys(self) -> List[str]:
        return ["input"]
