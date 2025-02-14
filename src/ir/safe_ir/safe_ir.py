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

    def size(self):
        match self:
            case DataType.FLOAT32:
                return 4
            case DataType.FLOAT64:
                return 8
            case DataType.INT32:
                return 4
            case DataType.INT64:
                return 8
            case DataType.BOOL:
                return 1
            case DataType.NONE:
                return 1



@dataclass
class ScalarSpec:
    type: DataType

    def size(self) -> int:
        return 1
    
    def size_bytes(self) -> int:
        return self.size() * self.type.size()


@dataclass
class ScalarType:
    spec: ScalarSpec
    data: Any


@dataclass
class TensorSpec:
    shape: Tuple[int, ...]
    data_type: DataType

    def size(self) -> int:
        out = 1
        for s in self.shape:
            out *= s
        return out
    
    def size_bytes(self) -> int:
        return self.size() * self.data_type.size()

@dataclass
class TensorType:
    spec: TensorSpec
    data: np.array


SpecType = Union[ScalarSpec, TensorSpec]
DataHolderType = Union[ScalarSpec, TensorSpec]
