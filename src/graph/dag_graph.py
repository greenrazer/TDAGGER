from enum import Enum, auto
from typing import Dict, List, Tuple

from ..ir.safe_ir import (
    DataHolderType,
    OpType,
    SpecType,
)
from ..ir.compute_stats import ComputeStats


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
        self.op_output_shapes, self.op_compute_stats = self._propagate_input_specs()

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
    
    def _propagate_input_specs(self) -> Tuple[Dict[str, List[int]], Dict[str, ComputeStats]]:
        return {}, {}

    def swap_input_specs(self, inputs: Dict[str, SpecType]):
        self.inputs = inputs
        self.op_output_shapes, self.op_compute_stats = self._propagate_input_specs()

    def parameter_bytes(self) -> int:
        total = 0
        for p in self.parameters.values():
            total += p.spec.size_bytes()
        return total

    def buffer_bytes(self) -> int:
        total = 0
        for b in self.buffers.values():
            total += b.spec.size_bytes()
        return total

    def constant_bytes(self) -> int:
        total = 0
        for c in self.constants.values():
            total += c.spec.size_bytes()
        return total

    def input_bytes(self) -> int:
        total = 0
        for i in self.inputs.values():
            total += i.size_bytes()
        return total
    
    def total_flops(self) -> int:
        return 0

    def total_memory_reads(self) -> int:
        return 0

    def total_memory_writes(self) -> int:
        return 0
    
    def layers(self) -> List[List[OpType]]:
        return []