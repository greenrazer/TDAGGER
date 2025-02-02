from enum import Enum, auto
from typing import Dict, List

from ..ir.safe_ir import (
    SpecType,
    DataHolderType,
    OpType,
)


class DataLocation(Enum):
    INPUT = auto()
    CONSTANT = auto()
    PARAMETER = auto()
    BUFFER = auto()
    OP = auto()


class DAGGraph:
    def __init__(
        self,
        inputs: Dict[str, SpecType],
        constants: Dict[str, DataHolderType],
        parameters: Dict[str, DataHolderType],
        buffers: Dict[str, DataHolderType],
        ops: Dict[str, OpType],
        outputs: List[str],
    ):
        self.inputs = inputs
        self.constants = constants
        self.parameters = parameters
        self.buffers = buffers
        self.ops = ops
        self.outputs = outputs

        self.name_registry: Dict[str, DataLocation] = self._populate_name_registry()

    def _populate_name_registry(self):
        name_registry = {}
        for name in self.inputs:
            name_registry[name] = DataLocation.INPUT

        for name in self.constants:
            name_registry[name] = DataLocation.CONSTANT

        for name in self.parameters:
            name_registry[name] = DataLocation.PARAMETER

        for name in self.buffers:
            name_registry[name] = DataLocation.BUFFER

        for name in self.ops:
            name_registry[name] = DataLocation.OP

        return name_registry

    def __index__(self, name):
        if name not in self.name_registry:
            raise Exception(f"Name not found in graph: {name}.")
        data_location = self.name_registry[name]
        match data_location:
            case DataLocation.INPUT:
                return self.constant_tensors[name]
            case DataLocation.CONSTANT:
                return self.constant_tensors[name]
            case DataLocation.PARAMETER:
                return self.parameter_tensors[name]
            case DataLocation.BUFFER:
                return self.buffer_tensors[name]
            case DataLocation.OP:
                return self.ops[name]
            case _:
                raise Exception("Data location not found.")

    def __str__(self):
        return "\n".join([op.__str__() for op in self.ops.values()])
