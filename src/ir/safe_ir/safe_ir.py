from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Tuple, Union

import numpy as np


class DataType(Enum):
    FLOAT32 = auto()
    FLOAT64 = auto()
    INT32 = auto()
    INT64 = auto()
    BOOL = auto()
    NONE = auto()
    STRING = auto()


@dataclass
class ScalarSpec:
    type: DataType

    def size(self):
        return 1


@dataclass
class ScalarType:
    spec: ScalarSpec
    data: Any


@dataclass
class TensorSpec:
    shape: Tuple[int, ...]
    data_type: DataType

    def size(self):
        out = 1
        for s in self.shape:
            out *= s
        return out

@dataclass
class TensorType:
    spec: TensorSpec
    data: np.array


SpecType = Union[ScalarSpec, TensorSpec]
DataHolderType = Union[ScalarSpec, TensorSpec]
