from typing import Dict

import torch

from ....ir.safe_ir import (
    DataType,
    ScalarSpec,
    ScalarType,
    TensorSpec,
    TensorType,
)
from ..reifier import Reifier
from .ir_to_torch_op_converter import IRToTorchOpConverter


class TorchReifier(Reifier):
    def __init__(self, graph):
        super().__init__()

        self.ir_graph = graph
        self.torch_graph = torch._C.Graph()

        module = torch.nn.Module()
        for name, param in self._create_torch_parameters().items():
            module.register_parameter(name, param)
        for name, buffer in self._create_torch_buffers().items():
            module.register_buffer(name, buffer)
        self.script_module = torch.jit.script(module)

        self.name_to_output_value: Dict[str, torch._C.Value] = {}
        self.op_converter = IRToTorchOpConverter()
        self.processed_ops = set()

        self.class_input = self._create_class_input()

    def _create_class_input(self):
        class_input = self.torch_graph.addInput("class_input")
        module_type = self.script_module._c._type()
        class_input.setType(module_type)
        self.name_to_output_value["class_input"] = class_input
        return class_input

    def _create_inputs(self):
        for name, input in self.ir_graph.inputs.items():
            if isinstance(input, TensorSpec):
                input_value = self.torch_graph.addInput(name)
                tensor_type = torch._C.TensorType.get().with_dtype(
                    input.data_type.to_torch()
                )
                if input.shape:
                    tensor_type = tensor_type.with_sizes(input.shape)
                input_value.setType(tensor_type)
            else:
                raise Exception(f"Input type unknown: {type(input)}")

            self.name_to_output_value[name] = input_value
            self.processed_ops.add(name)

    def _create_constants(self):
        for name, const in self.ir_graph.constants.items():
            const_node = self.torch_graph.create("prim::Constant")
            self.torch_graph.insertNode(const_node)
            if isinstance(const, ScalarType):
                match const.spec.type:
                    case DataType.INT32:
                        const_node.output().setType(torch._C.IntType.get())
                        const_node.i_("value", const.data)
                    case DataType.INT64:
                        const_node.output().setType(torch._C.IntType.get())
                        const_node.i_("value", const.data)
                    case DataType.FLOAT32:
                        const_node.output().setType(torch._C.FloatType.get())
                        const_node.f_("value", const.data)
                    case DataType.FLOAT64:
                        const_node.output().setType(torch._C.FloatType.get())
                        const_node.f_("value", const.data)
                    case DataType.STRING:
                        const_node.output().setType(torch._C.StringType.get())
                        const_node.s_("value", const.data)
                    case DataType.BOOL:
                        const_node.output().setType(torch._C.BoolType.get())
                        const_node.i_("value", int(const.data))
                    case DataType.NONE:
                        const_node.output().setType(torch._C.NoneType.get())
                    case _:
                        raise Exception(f"Unknown Constant DataType: {const.spec.type}")
            elif isinstance(const, TensorType):
                tensor_type = torch._C.TensorType.get().with_dtype(
                    const.spec.data_type.to_torch()
                )
                tensor_type = tensor_type.with_sizes(const.spec.shape)
                const_node.output().setType(tensor_type)
                const_node.t_("value", torch.tensor(const.data))
            else:
                raise Exception(f"Constant type unknown: {type(const)}")

            self.processed_ops.add(name)
            self.name_to_output_value[name] = const_node.output()

    def _create_parameters(self):
        for name, param in self.ir_graph.parameters.items():
            param_node = self.torch_graph.create("prim::GetAttr")
            self.torch_graph.insertNode(param_node)

            if isinstance(param, TensorType):
                param_node.addInput(self.class_input)
                param_node.s_("name", name)

                tensor_type = torch._C.TensorType.get().with_dtype(
                    param.spec.data_type.to_torch()
                )
                if param.spec.shape:
                    tensor_type = tensor_type.with_sizes(param.spec.shape)
                param_node.output().setType(tensor_type)
            else:
                raise Exception(f"Parameter type unknown: {type(param)}")

            self.processed_ops.add(name)
            self.name_to_output_value[name] = param_node.output()

    def _create_buffers(self):
        for name, buffer in self.ir_graph.buffers.items():
            buffer_node = self.torch_graph.create("prim::GetAttr")
            self.torch_graph.insertNode(buffer_node)

            if isinstance(buffer, TensorType):
                buffer_node.addInput(self.class_input)
                buffer_node.s_("name", name)

                tensor_type = torch._C.TensorType.get().with_dtype(
                    buffer.spec.data_type.to_torch()
                )

                if buffer.spec.shape:
                    tensor_type = tensor_type.with_sizes(buffer.spec.shape)
                buffer_node.output().setType(tensor_type)
            else:
                raise Exception(f"Buffer type unknown: {type(buffer)}")

            self.processed_ops.add(name)
            self.name_to_output_value[name] = buffer_node.output()

    def _create_outputs(self):
        # register the output if only 1 tensor
        # else finish the torch graph with a tuple construct
        if len(self.ir_graph.outputs) == 1:
            name = self.ir_graph.outputs[0]
            if name in self.name_to_output_value:
                out_value = self.name_to_output_value[name]
                self.torch_graph.registerOutput(out_value)
            else:
                raise Exception(f"Graph output not found in torch graph: {name}.")
        else:
            node = self.torch_graph.create("prim::TupleConstruct")

            input_types = []
            for name in self.ir_graph.outputs:
                if name not in self.name_to_output_value:
                    raise Exception(f"Graph output not found in torch graph: {name}.")
                input_value = self.name_to_output_value[name]

                input_types.append(input_value.type())
                node.addInput(input_value)

            tuple_type = torch._C.TupleType(input_types)
            node.output().setType(tuple_type)
            self.torch_graph.insertNode(node)

            self.torch_graph.registerOutput(node.outputsAt(0))

    def _create_torch_parameters(self) -> Dict[str, torch.nn.Parameter]:
        parameters = {}
        for name, param in self.ir_graph.parameters.items():
            if isinstance(param, TensorType):
                parameters[name] = torch.nn.Parameter(
                    torch.tensor(param.data, dtype=param.spec.data_type.to_torch())
                )
        return parameters

    def _create_torch_buffers(self) -> Dict[str, torch.tensor]:
        buffers = {}
        for name, buffer in self.ir_graph.buffers.items():
            if isinstance(buffer, TensorType):
                buffers[name] = torch.tensor(
                    buffer.data, dtype=buffer.spec.data_type.to_torch()
                )
        return buffers

    def _all_inputs_processed(self, op):
        return all(
            [input_name in self.processed_ops for input_name in op.unique_indices]
        )

    def export(self) -> torch.nn.Module:
        self._create_inputs()
        self._create_constants()
        self._create_parameters()
        self._create_buffers()

        ops_to_process = {
            op_name
            for op_name in self.ir_graph.ops.keys()
            if op_name not in self.processed_ops
        }
        while ops_to_process:
            ready_ops = {
                self.ir_graph.ops[op_name]
                for op_name in ops_to_process
                if self._all_inputs_processed(self.ir_graph.ops[op_name])
            }

            if not ready_ops:
                remaining_ops = ops_to_process - self.processed_ops
                if remaining_ops:
                    for r in remaining_ops:
                        op_debug = self.ir_graph.ops[r]
                        print(op_debug)
                        for input_val in op_debug.inputs:
                            print("     ", input_val)
                    raise Exception(
                        "Graph creation failed: All remaining unprocessed ops have at least one unprocessed input."
                    )
                else:
                    break

            for op in ready_ops:
                for torch_op_node in self.op_converter.convert_op(
                    op, self.ir_graph, self.torch_graph, self.name_to_output_value
                ):
                    self.torch_graph.insertNode(torch_op_node)
                    self.name_to_output_value[op.name] = torch_op_node.output()

                self.processed_ops.add(op.name)
                ops_to_process.remove(op.name)

        self._create_outputs()

        script_function = torch._C._create_function_from_graph(
            "forward", self.torch_graph
        )
        self.script_module._forward_function = script_function
        self.script_module.forward = lambda *x: script_function(self.script_module, *x)
        return self.script_module
