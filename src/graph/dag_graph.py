from enum import Enum, auto
from typing import Dict, List, Tuple

from ..ir.safe_ir import (
    DataHolderType,
    ScalarType,
    TensorType,
    OpType,
    SpecType,
    ScalarSpec,
    TensorSpec,
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
        self.layers = self._seperate_layers()
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

    def __getitem__(self, name):
        if name not in self.name_registry:
            raise Exception(f"Name not found in graph: {name}.")
        data_location = self.name_registry[name]
        match data_location:
            case DataLocation.INPUT:
                return self.inputs[name]
            case DataLocation.CONSTANT:
                return self.constants[name]
            case DataLocation.PARAMETER:
                return self.parameters[name]
            case DataLocation.BUFFER:
                return self.buffers[name]
            case DataLocation.OP:
                return self.ops[name]
            case _:
                raise Exception("Data location not found.")

    def __str__(self):
        return "\n".join([op.__str__() for op in self.ops.values()])

    def _seperate_layers(self) -> List[List[str]]:
        def unprocessed_ops(ops, processed):
            unprocessed = []
            for name, op in ops.items():
                if op.name not in processed:
                    unprocessed.append(op)

            return unprocessed

        def processable_ops(unprocessed_ops, processed):
            processable = []
            for op in unprocessed_ops:
                if all([op_input in processed for op_input in op.input.unique_indices]):
                    processable.append(op)

            return processable

        def layer_index_for_op(op, name_to_layer):
            max_input_layer = 0
            for op_input in op.input.unique_indices:
                max_input_layer = max(max_input_layer, name_to_layer[op_input])
            return max_input_layer + 1

        name_to_layer = {
            name: -1
            for name, location in self.name_registry.items()
            if location != DataLocation.OP
        }
        processed = {name for name in name_to_layer}
        while unprocessed := unprocessed_ops(self.ops, processed):
            processable = processable_ops(unprocessed, processed)
            if len(processable) == 0:
                raise Exception("Graph contains cycle or is missing node.")
            for p in processable:
                layer_index = layer_index_for_op(p, name_to_layer)
                name_to_layer[p.name] = layer_index
                processed.add(p.name)

        layer_to_names = {}
        for name, layer in name_to_layer.items():
            if layer not in layer_to_names:
                layer_to_names[layer] = []
            layer_to_names[layer].append(name)

        out = []
        for layer in sorted(layer_to_names.keys()):
            if layer != -1:
                out.append(layer_to_names[layer])

        return out

    def _propagate_input_specs(
        self,
    ) -> Tuple[Dict[str, SpecType], Dict[str, ComputeStats]]:
        output_specs = {}
        compute_stats = {}
        for layer in self.layers:
            for op_name in layer:
                op = self.ops[op_name]
                input_specs = []
                for inp in op.input.flattened_indices:
                    op_input = self[inp]
                    if isinstance(op_input, OpType):
                        input_specs.append(output_specs[inp])
                    elif isinstance(op_input, (ScalarType, TensorType)):
                        input_specs.append(op_input.spec)
                    else:
                        input_specs.append(op_input)

                try:
                    output_specs[op_name] = op.spec.output_spec(input_specs)
                    compute_stats[op_name] = op.spec.compute_stats(input_specs)
                except Exception as e:
                    print(input_specs)
                    print(op)
                    raise Exception(e)
        return output_specs, compute_stats

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

    def total_bytes(self) -> int:
        return self.parameter_bytes() + self.buffer_bytes() + self.constant_bytes()

    def total_flops(self) -> int:
        total = 0
        for cs in self.op_compute_stats.values():
            total += cs.flops
        return total

    def total_memory_reads(self) -> int:
        total = 0
        for cs in self.op_compute_stats.values():
            total += cs.reads
        return total

    def total_memory_writes(self) -> int:
        total = 0
        for cs in self.op_compute_stats.values():
            total += cs.writes
        return total
