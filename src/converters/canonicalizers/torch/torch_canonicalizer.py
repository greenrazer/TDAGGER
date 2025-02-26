import re
from typing import Dict, List, Union

import torch

from ....graph.dag_graph_builder import DAGGraphBuilder
from ....ir.safe_ir import DataType, ScalarSpec, ScalarType, TensorSpec, TensorType
from ..canonicalizer import Canonicalizer
from .torch_to_ir_op_converter import TorchToIROpConverter


class TorchCanonicalizer(Canonicalizer):
    def __init__(self, model: torch.jit.ScriptModule):
        super().__init__()

        self.state_dict = model.state_dict(keep_vars=True)
        self.named_buffers = dict(model.named_buffers())
        self.forward_graph = self._retrieve_graph(model)

        self.output_value_to_node = self._create_value_to_node_map()
        self.output_value_to_name = self._create_value_to_name_map()
        self.processed_nodes = set()
        self.op_converter = TorchToIROpConverter()

    def _retrieve_graph(self, model) -> torch._C.Graph:
        forward_graph = model.forward.graph.copy()
        # Inline all function calls in the forward graph to simplify the graph structure.
        torch._C._jit_pass_inline(forward_graph)
        torch._C._jit_pass_lint(forward_graph)
        # Inline the fork-wait operations in the graph to optimize asynchronous execution.
        torch._C._jit_pass_inline_fork_wait(forward_graph)
        torch._C._jit_pass_lint(forward_graph)
        # Lower Tuples
        torch._C._jit_pass_lower_all_tuples(forward_graph)
        torch._C._jit_pass_lint(forward_graph)
        # Propagate constants
        torch._C._jit_pass_constant_propagation(forward_graph)
        torch._C._jit_pass_lint(forward_graph)
        # Perform Dead Code Elimination (DCE) to remove unused nodes and operations from the graph.
        torch._C._jit_pass_dce(forward_graph)
        torch._C._jit_pass_lint(forward_graph)
        # Canonicalize operations in the graph to ensure they are in a standard form for the fuser.
        torch._C._jit_pass_canonicalize_graph_fuser_ops(forward_graph)
        torch._C._jit_pass_lint(forward_graph)
        # Canonicalize the graph to ensure all operations are in a standard, simplified form.
        torch._C._jit_pass_canonicalize(forward_graph)
        torch._C._jit_pass_lint(forward_graph)
        # Replace obvious patterns (e.g. x*0 -> 0, x*1 -> x, x+0 -> x,--x = x, x.view(a).view(b) = x.view(b))
        torch._C._jit_pass_peephole(forward_graph)
        torch._C._jit_pass_lint(forward_graph)
        return forward_graph

    def _is_get_attr_parent(self, node):
        if node.kind() == "prim::GetAttr":
            for other_node in self.forward_graph.nodes():
                if other_node.kind() == "prim::GetAttr" and other_node != node:
                    for input in other_node.inputs():
                        if input.node() == node:
                            return True
        return False

    def _get_full_attr_name(self, node):
        names = [node.s("name")]
        current = node

        while True:
            inputs = list(current.inputs())
            if not inputs:
                break

            input_node = inputs[0].node()
            if input_node.kind() != "prim::GetAttr":
                break

            current = input_node
            names.append(current.s("name"))

        return ".".join(reversed(names))

    def _create_value_to_node_map(self):
        output_value_to_node: Dict[torch._C.Value, torch._C.Node] = {}
        for input_value in self.forward_graph.inputs():
            output_value_to_node[input_value] = None

        for node in self.forward_graph.nodes():
            for output in node.outputs():
                output_value_to_node[output] = node
        return output_value_to_node

    def _create_value_to_name_map(self):
        semantic_names = {}
        for node in self.forward_graph.nodes():
            if node.kind() == "prim::GetAttr":
                output_value = node.output()
                full_name = self._get_full_attr_name(node).replace(".", "_")
                semantic_names[output_value] = full_name
        return semantic_names

    def _process_debug_source(self, debug_source_str):
        pattern = r"(?P<file_path>.+)\((?P<line_number>\d+)\): (?P<function_name>.+)"
        source_lines = debug_source_str.strip().split("\n")

        ranges = []

        for line in source_lines:
            match = re.match(pattern, line)
            if match:
                file_path = match.group("file_path")
                line_number = match.group("line_number")
                function_name = match.group("function_name")

                ranges.append((file_path, line_number, function_name))

        return ranges

    def _retrieve_inputs(self) -> Dict[str, TensorSpec]:
        inputs = {}
        for input_value in self.forward_graph.inputs():
            input_type = input_value.type()
            name = input_value.debugName().replace(".", "_")
            match input_type.kind():
                case "ClassType":
                    # Ignore classes
                    pass
                case "TensorType":
                    inputs[name] = TensorSpec(
                        shape=list(input_type.sizes()),
                        data_type=DataType.from_torch(input_type.dtype()),
                    )
                case _:
                    raise Exception("Input Type Unknown.")
            self.processed_nodes.add(input_value)
        return inputs

    def _retrieve_constants(self) -> Dict[str, Union[TensorType, ScalarType]]:
        constants = {}
        for node in self.forward_graph.nodes():
            if node.kind() == "prim::Constant":
                # Constants can only have 1 output
                name = node.output().debugName().replace(".", "_")
                match node.output().type().kind():
                    case "TensorType":
                        constants[name] = TensorType(
                            spec=TensorSpec(
                                shape=list(node.output().type().sizes()),
                                data_type=DataType.from_torch(
                                    node.output().type().dtype()
                                ),
                            ),
                            data=node.t("value").numpy(),
                        )
                    case "IntType":
                        constants[name] = ScalarType(
                            spec=ScalarSpec(type=DataType.INT32),
                            data=node.i("value"),
                        )
                    case "FloatType":
                        constants[name] = ScalarType(
                            spec=ScalarSpec(type=DataType.FLOAT32),
                            data=node.f("value"),
                        )
                    case "StringType":
                        pass  # skipping for now
                    case "BoolType":
                        constants[name] = ScalarType(
                            spec=ScalarSpec(type=DataType.BOOL),
                            data=bool(node.i("value")),
                        )
                    case "NoneType":
                        constants[name] = ScalarType(
                            spec=ScalarSpec(type=DataType.NONE),
                            data=None,
                        )
                    case "DeviceObjType":
                        # we don't care about devices
                        constants[name] = ScalarType(
                            spec=ScalarSpec(type=DataType.NONE),
                            data=None,
                        )
                        pass
                    case "ListType":
                        # ignore list types for now
                        pass
                    case _:
                        raise Exception(
                            f"Constant Type Unknown: {node.output().type().kind()}."
                        )
                self.processed_nodes.add(node)
        return constants

    def _retrieve_parameters(self):
        parameters = {}
        for node in self.forward_graph.nodes():
            if node.kind() == "prim::GetAttr":
                if self._is_get_attr_parent(node):
                    self.processed_nodes.add(node)
                    continue

                attr_name = self._get_full_attr_name(node)

                if attr_name not in self.state_dict or attr_name in self.named_buffers:
                    continue

                clean_name = attr_name.replace(".", "_")

                # GetAttr can only have 1 output
                match node.output().type().kind():
                    case "TensorType":
                        tensor = self.state_dict[attr_name].data
                        parameters[clean_name] = TensorType(
                            spec=TensorSpec(
                                shape=list(tensor.shape),
                                data_type=DataType.from_torch(tensor.dtype),
                            ),
                            data=tensor.numpy(),
                        )
                    case _:
                        raise Exception("Parameter Type Unknown.")
                self.processed_nodes.add(node)
        return parameters

    def _retrieve_buffers(self):
        buffers = {}
        for node in self.forward_graph.nodes():
            if node.kind() == "prim::GetAttr":
                if self._is_get_attr_parent(node):
                    self.processed_nodes.add(node)
                    continue

                attr_name = self._get_full_attr_name(node)

                if attr_name not in self.named_buffers:
                    continue

                clean_name = attr_name.replace(".", "_")

                # GetAttr can only have 1 output
                match node.output().type().kind():
                    case "TensorType":
                        tensor = self.state_dict[attr_name].data
                        buffers[clean_name] = TensorType(
                            spec=TensorSpec(
                                shape=list(tensor.shape),
                                data_type=DataType.from_torch(tensor.dtype),
                            ),
                            data=tensor.numpy(),
                        )
                    case _:
                        raise Exception("Buffer Type Unknown.")
                self.processed_nodes.add(node)
        return buffers

    def _retrieve_outputs(self) -> List[str]:
        outputs = []
        for output_value in self.forward_graph.outputs():
            output_type = output_value.type()
            name = output_value.debugName().replace(".", "_")
            match output_type.kind():
                case "TensorType":
                    outputs.append(name)
                case _:
                    raise Exception(f"Output Type Unknown: {output_type.kind()}")
        return outputs

    def _all_inputs_processed(self, node):
        return all(
            [
                self.output_value_to_node[input_val] is None
                or self.output_value_to_node[input_val] in self.processed_nodes
                for input_val in node.inputs()
            ]
        )

    def build_graph(self, graph_builder: DAGGraphBuilder):
        name_to_spec = {}
        for name, value in self._retrieve_inputs().items():
            name_to_spec[name] = value
            graph_builder.add_input(name, value)

        for name, value in self._retrieve_constants().items():
            name_to_spec[name] = value.spec
            graph_builder.add_constant(name, value)

        for name, value in self._retrieve_parameters().items():
            name_to_spec[name] = value.spec
            graph_builder.add_parameter(name, value)

        for name, value in self._retrieve_buffers().items():
            name_to_spec[name] = value.spec
            graph_builder.add_buffer(name, value)

        nodes_to_process = {
            node
            for node in self.forward_graph.nodes()
            if node not in self.processed_nodes
        }
        while nodes_to_process:
            ready_nodes = {
                node for node in nodes_to_process if self._all_inputs_processed(node)
            }

            if not ready_nodes:
                remaining_nodes = nodes_to_process - self.processed_nodes
                if remaining_nodes:
                    for r in remaining_nodes:
                        print(r)
                        for input_val in r.inputs():
                            print(
                                "     ",
                                input_val,
                                self.output_value_to_node[input_val],
                                self.output_value_to_node[input_val]
                                in self.processed_nodes,
                            )
                    raise Exception(
                        "Graph creation failed: All remaining unprocessed nodes have at least one unprocessed input."
                    )
                else:
                    break

            for node in ready_nodes:
                debug_source = self._process_debug_source(node.sourceRange())
                safe_ops, consts = self.op_converter.convert_op(
                    node,
                    self.forward_graph,
                    self.output_value_to_node,
                    self.output_value_to_name,
                    name_to_spec,
                    debug_source,
                )
                for const_name, const in consts.items():
                    graph_builder.add_constant(const_name, const)

                for safe_op in safe_ops:
                    graph_builder.add_op(safe_op.name, safe_op)

                self.processed_nodes.add(node)
                nodes_to_process.remove(node)

        for name in self._retrieve_outputs():
            graph_builder.add_output(name)
