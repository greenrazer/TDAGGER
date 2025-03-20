from enum import Enum, auto
from typing import Dict, List, Tuple, Union

from sympy import Expr, Symbol

from ..ir.compute_stats import ComputeStats
from ..ir.safe_ir import (
    DataHolderType,
    OpType,
    ScalarSpec,
    ScalarType,
    SpecType,
    TensorSpec,
    TensorType,
    UnaryTensorInput,
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
        self.layers = self._seperate_layers()

        self.symbolic_input_specs, self.symbolic_input_shape_values = (
            self._create_symbolic_inputs()
        )
        self.symbolic_op_output_specs = self._propagate_symbolic_input_specs()

        self.op_output_specs, self.op_compute_stats = self._propagate_input_specs()

        for i, layer in enumerate(self.layers):
            for z in layer:
                if not isinstance(self.op_output_specs[z], ScalarSpec):
                    a = [
                        int(ex.subs(self.symbolic_input_shape_values))
                        if isinstance(ex, Expr)
                        else ex
                        for ex in self.symbolic_op_output_specs[z].shape
                    ]
                    b = self.op_output_specs[z].shape
                    if a != b:
                        raise Exception(
                            f"Symbolic shapes differs from concrete shapes: symbolic={a} concrete={b}"
                        )

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

    def _create_symbolic_inputs(self):
        symbolic_input_specs = {}
        symbolic_input_shape_values = {}
        for name, spec in self.inputs.items():
            if isinstance(spec, TensorSpec):
                symbolic_input_specs[name] = spec.to_symbolic(name)
                symbolic_input_shape_values |= spec.symbolic_values(name)
            elif isinstance(spec, ScalarSpec):
                symbolic_input_specs[name] = spec
            else:
                raise Exception("Input spec not supported.")

        return symbolic_input_specs, symbolic_input_shape_values

    def _propagate_symbolic_input_specs(self):
        output_symbolic_specs = {}
        for layer in self.layers:
            for op_name in layer:
                op = self.ops[op_name]
                input_specs = []
                for inp in op.input.flattened_indices:
                    if inp in self.symbolic_input_specs:
                        op_input = self.symbolic_input_specs[inp]
                    else:
                        op_input = self[inp]

                    if isinstance(op_input, OpType):
                        input_specs.append(output_symbolic_specs[inp])
                    elif isinstance(op_input, (ScalarType, TensorType)):
                        input_specs.append(op_input.spec)
                    else:
                        input_specs.append(op_input)

                try:
                    output_symbolic_specs[op_name] = op.spec.output_spec(input_specs)
                except Exception as e:
                    print(input_specs)
                    print(op)
                    raise Exception(e)

        return output_symbolic_specs

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

    def rename_input_dimensions(self, new_name: str, input_dimensions: List[str]):
        if new_name in self.symbolic_input_shape_values:
            raise Exception(f"New name not unique: {new_name}.")

        sizes = []
        for symbol_name in input_dimensions:
            if symbol_name not in self.symbolic_input_shape_values:
                raise Exception(f"Symbol name not found in inputs: {symbol_name}")
            sizes.append(self.symbolic_input_shape_values[symbol_name])

        if not all([size == sizes[0] for size in sizes]):
            raise Exception("Not all dimensions are of equal size.")

        for symbol_name in input_dimensions:
            del self.symbolic_input_shape_values[symbol_name]

        self.symbolic_input_shape_values[new_name] = sizes[0]

        for symbol_name in input_dimensions:
            old_symbol = Symbol(symbol_name)
            new_symbol = Symbol(new_name)
            for sym_op_spec in self.symbolic_op_output_specs:
                if isinstance(sym_op_spec, ScalarSpec):
                    continue
                for sym_size in sym_op_spec.shape:
                    sym_size.subs(old_symbol, new_symbol)

    def with_removed_dimension(self, dimension_name: str) -> "DAGGraph":
        if dimension_name not in self.symbolic_input_shape_values:
            raise Exception(
                f"Dimension not found: {dimension_name} valid={self.symbolic_input_shape_values.keys()}."
            )

        dimension_symbol = Symbol(dimension_name)

        new_ops = {}
        for layer in self.layers:
            for op_name in layer:
                op = self.ops[op_name]
                new_ops[op_name] = op.with_removed_dimension(
                    dimension_symbol,
                    self.symbolic_input_specs,
                    self.symbolic_op_output_specs,
                )

        new_inputs = {}
        for input_name in self.inputs:
            spec = self.inputs[input_name]

            if isinstance(spec, ScalarSpec):
                new_inputs[input_name] = spec

            symbolic_spec = self.symbolic_input_specs[input_name]
            new_spec_shape = []
            for dim, expr in enumerate(symbolic_spec.shape[:]):
                if not expr.has(dimension_symbol):
                    new_spec_shape.append(spec.shape[dim])

            new_inputs[input_name] = TensorSpec(
                shape=new_spec_shape, data_type=spec.data_type
            )

        return DAGGraph(
            inputs=new_inputs,
            constants=self.constants,
            parameters=self.parameters,
            buffers=self.buffers,
            ops=new_ops,
            outputs=self.outputs,
        )

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
