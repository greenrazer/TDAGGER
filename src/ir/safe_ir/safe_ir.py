from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Union

import numpy as np
from sympy import Expr, Symbol


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
    shape: List[int]
    data_type: DataType

    def size(self) -> int:
        out = 1
        for s in self.shape:
            out *= s
        return out

    def size_bytes(self) -> int:
        return self.size() * self.data_type.size()

    def to_symbolic(self, name: str) -> "SymbolicTensorSpec":
        return SymbolicTensorSpec(
            shape=[Symbol(f"{name}_{dim}") for dim, _ in enumerate(self.shape)],
            data_type=self.data_type,
        )

    def symbolic_values(self, name: str) -> Dict[str, int]:
        out = {}
        for dim, size in enumerate(self.shape):
            out[f"{name}_{dim}"] = size
        return out


@dataclass
class SymbolicTensorSpec:
    shape: List[Expr]
    data_type: DataType

    def size(self) -> Expr:
        out = 1
        for s in self.shape:
            out *= s
        return out

    def size_bytes(self) -> Expr:
        return self.size() * self.data_type.size()

    def to_concrete(self, symbol_values: Dict[str, int]) -> "TensorSpec": ...


@dataclass
class TensorType:
    spec: TensorSpec
    data: np.array


SpecType = Union[ScalarSpec, TensorSpec, SymbolicTensorSpec]
DataHolderType = Union[ScalarType, TensorType]
