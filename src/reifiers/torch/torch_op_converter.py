from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

import torch

from ...graph.dag_graph import DAGGraph
from ...ir.safe_ir import BinaryElementwiseSpec, OpType, UnaryElementwiseSpec

UNARY_ELEMENTWISE_SPEC_TO_ATEN = {
    str(UnaryElementwiseSpec.ABSOLUTE_VALUE): "aten::abs",
    str(UnaryElementwiseSpec.NEGATIVE): "aten::neg",
    str(UnaryElementwiseSpec.SQUARE_ROOT): "aten::sqrt",
    str(UnaryElementwiseSpec.EXP): "aten::exp",
    str(UnaryElementwiseSpec.LOG): "aten::log",
    str(UnaryElementwiseSpec.SIN): "aten::sin",
    str(UnaryElementwiseSpec.COS): "aten::cos",
    str(UnaryElementwiseSpec.TAN): "aten::tan",
    str(UnaryElementwiseSpec.ARCSIN): "aten::asin",
    str(UnaryElementwiseSpec.ARCCOS): "aten::acos",
    str(UnaryElementwiseSpec.ARCTAN): "aten::atan",
    str(UnaryElementwiseSpec.SINH): "aten::sinh",
    str(UnaryElementwiseSpec.COSH): "aten::cosh",
    str(UnaryElementwiseSpec.TANH): "aten::tanh",
    str(UnaryElementwiseSpec.ARCSINH): "aten::asinh",
    str(UnaryElementwiseSpec.ARCCOSH): "aten::acosh",
    str(UnaryElementwiseSpec.ARCTANH): "aten::atanh",
}

BINARY_ELEMENTWISE_SPEC_TO_ATEN = {
    str(BinaryElementwiseSpec.ADD): "aten::add",
    str(BinaryElementwiseSpec.SUBTRACT): "aten::sub",
    str(BinaryElementwiseSpec.MULTIPLY): "aten::multiply",
    str(BinaryElementwiseSpec.DIVIDE): "aten::div",
}


@dataclass
class ConversionContext:
    op: OpType
    graph: DAGGraph
    torch_graph: torch._C.Graph
    name_to_output_value: Dict[str, torch._C.Value]


class TorchOpConverter:
    def __init__(self):
        self._converters: Dict[
            str, Callable[[ConversionContext], List[Tuple[str, OpType]]]
        ] = {}
        self._register_converters()

    def _register_converters(self):
        self._converters.update(
            {
                key: self._convert_binary_elementwise
                for key in BINARY_ELEMENTWISE_SPEC_TO_ATEN.keys()
            }
        )

        self._converters.update(
            {
                key: self._convert_unary_elementwise
                for key in UNARY_ELEMENTWISE_SPEC_TO_ATEN.keys()
            }
        )

    def convert_op(
        self,
        op: OpType,
        graph: DAGGraph,
        torch_graph: torch._C.Graph,
        name_to_output_value: Dict[str, torch._C.Value],
    ) -> List[torch._C.Node]:
        ctx = ConversionContext(op, graph, torch_graph, name_to_output_value)

        if op.type not in self._converters:
            raise Exception(f"Unsupported operation type: {op.type}")

        return self._converters[op.type](ctx)

    def _convert_binary_elementwise(
        self, ctx: ConversionContext
    ) -> List[torch._C.Node]:
        input_0_val = ctx.name_to_output_value[ctx.op.inputs["input_0"]]
        input_1_val = ctx.name_to_output_value[ctx.op.inputs["input_1"]]

        match (input_0_val.type().kind(), input_1_val.type().kind(), ctx.op.spec):
            case (_, _, BinaryElementwiseSpec.MULTIPLY):
                node = ctx.torch_graph.create("aten::mul")
            case _:
                node = ctx.torch_graph.create(
                    BINARY_ELEMENTWISE_SPEC_TO_ATEN[ctx.op.type]
                )

        match (input_0_val.type().kind(), input_1_val.type().kind(), ctx.op.spec):
            case ("TensorType", _, BinaryElementwiseSpec.ADD):
                node.addInput(input_0_val)
                node.addInput(input_1_val)
                node.addInput(ctx.torch_graph.insertConstant(1))
            case (_, "TensorType", BinaryElementwiseSpec.ADD):
                node.addInput(input_1_val)
                node.addInput(input_0_val)
                node.addInput(ctx.torch_graph.insertConstant(1))
            case ("TensorType", _, BinaryElementwiseSpec.MULTIPLY):
                node.addInput(input_0_val)
                node.addInput(input_1_val)
            case (_, "TensorType", BinaryElementwiseSpec.MULTIPLY):
                node.addInput(input_1_val)
                node.addInput(input_0_val)
            case ("TensorType", "TensorType", _):
                node.addInput(input_0_val)
                node.addInput(input_1_val)
            case (_, _, _):
                node.addInput(input_0_val)
                node.addInput(input_1_val)
                node.output().setType(input_0_val.type())

        return [node]

    def _convert_unary_elementwise(self, ctx: ConversionContext) -> List[torch._C.Node]:
        node = ctx.torch_graph.create(UNARY_ELEMENTWISE_SPEC_TO_ATEN[ctx.op.type])

        input_val = ctx.name_to_output_value[ctx.op.inputs["input"]]

        if input_val.type().kind() != "TensorType":
            # in -> in_tensor
            num_to_tensor_node = ctx.torch_graph.create("aten::scalar_tensor")
            num_to_tensor_node.addInput(input_val)
            num_to_tensor_node.addInput(ctx.torch_graph.insertConstant(2))
            num_to_tensor_node.addInput(ctx.torch_graph.insertConstant(None))
            num_to_tensor_node.addInput(ctx.torch_graph.insertConstant(None))
            num_to_tensor_node.addInput(ctx.torch_graph.insertConstant(None))

            # in_tensor -> op_tensor
            node.addInput(num_to_tensor_node.output())

            # op_tensor -> op
            tensor_to_num_node = ctx.torch_graph.create("aten::item")
            tensor_to_num_node.addInput(node.output())
            tensor_to_num_node.output().setType(torch._C.FloatType.get())

            out = [num_to_tensor_node, node, tensor_to_num_node]
        else:
            node.addInput(input_val)
            node.output().setType(torch._C.TensorType.get())
            out = [node]

        return out
