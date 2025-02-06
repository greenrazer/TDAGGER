import torch

from .safe_ir import DataType


@staticmethod
def datatype_from_torch(torch_dtype: torch.dtype) -> "DataType":
    mapping = {
        torch.float32: DataType.FLOAT32,
        torch.float64: DataType.FLOAT64,
        torch.int32: DataType.INT32,
        torch.int64: DataType.INT64,
        torch.bool: DataType.BOOL,
    }
    if torch_dtype not in mapping:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return mapping[torch_dtype]


def datatype_to_torch(self) -> torch.dtype:
    mapping = {
        DataType.FLOAT32: torch.float32,
        DataType.FLOAT64: torch.float64,
        DataType.INT32: torch.int32,
        DataType.INT64: torch.int64,
        DataType.BOOL: torch.bool,
    }
    return mapping[self]


DataType.from_torch = staticmethod(datatype_from_torch)
DataType.to_torch = datatype_to_torch
