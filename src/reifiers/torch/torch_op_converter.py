from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union
import torch

from ...ir.safe_ir import OpType, UnaryElementwiseSpec
from ...graph.dag_graph import DAGGraph

UNARY_ELEMENTWISE_SPEC_TO_ATEN = {
   str(UnaryElementwiseSpec.ABS): "aten::abs",
   str(UnaryElementwiseSpec.NEGATIVE): "aten::neg",
   str(UnaryElementwiseSpec.SQUARE_ROOT): "aten::sqrt",
   str(UnaryElementwiseSpec.SQUARE): "aten::square", 
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

   str(UnaryElementwiseSpec.RELU): "aten::relu",
   str(UnaryElementwiseSpec.SELU): "aten::selu",
}

@dataclass
class ConversionContext:
    op: OpType
    graph: DAGGraph
    torch_graph: torch._C.Graph
    name_to_output_value: Dict[str, torch._C.Value]


class TorchOpConverter:
    """Handles conversion of torch operations to safe operations."""

    def __init__(self):
        # Dictionary mapping operation types to their conversion functions
        self._converters: Dict[
            str, Callable[[ConversionContext], List[Tuple[str, OpType]]]
        ] = {}
        self._register_converters()

    def _register_converters(self):
        """Register all conversion functions."""
        self._converters.update({
            "binary_add": self._convert_add,
            "binary_multiply": self._convert_multiply,
            "leaky_relu": self._convert_leaky_relu,
            "elu": self._convert_elu,
        })

        self._converters.update({
            key: self._convert_unary_elementwise
            for key in UNARY_ELEMENTWISE_SPEC_TO_ATEN.keys()
        })

    def convert_op(
        self,
        op: OpType,
        graph: DAGGraph,
        torch_graph: torch._C.Graph,
        name_to_output_value: Dict[str, torch._C.Value]
    ) -> List[torch._C.Node]:
        ctx = ConversionContext(op, graph, torch_graph, name_to_output_value)

        if op.type not in self._converters:
            raise Exception(f"Unsupported operation type: {op.type}")

        return self._converters[op.type](ctx)

    def _convert_add(self, ctx: ConversionContext) -> List[torch._C.Node]:
        node = ctx.torch_graph.create("aten::add")

        input_0_val = ctx.name_to_output_value[ctx.op.inputs["input_0"]]
        input_1_val = ctx.name_to_output_value[ctx.op.inputs["input_1"]]

        node.addInput(input_0_val)
        node.addInput(input_1_val)
        # add has a weird alpha parameter for a + alpha*b
        node.addInput(ctx.torch_graph.insertConstant(1))
        
        node.output().setType(torch._C.TensorType.get())
        return [node]

    def _convert_multiply(self, ctx: ConversionContext) -> List[torch._C.Node]:
        node = ctx.torch_graph.create("aten::multiply")

        input_0_val = ctx.name_to_output_value[ctx.op.inputs["input_0"]]
        input_1_val = ctx.name_to_output_value[ctx.op.inputs["input_1"]]

        node.addInput(input_0_val)
        node.addInput(input_1_val)

        node.output().setType(torch._C.TensorType.get())
        return [node]
    
    def _convert_unary_elementwise(self, ctx: ConversionContext) -> List[torch._C.Node]:
        node = ctx.torch_graph.create(UNARY_ELEMENTWISE_SPEC_TO_ATEN[ctx.op.type])

        input_val = ctx.name_to_output_value[ctx.op.inputs["input"]]

        node.addInput(input_val)

        node.output().setType(torch._C.TensorType.get())

        return [node]

    def _convert_leaky_relu(self, ctx: ConversionContext) -> List[OpType]:
        node = ctx.torch_graph.create("aten::leaky_relu")

        input_val = ctx.name_to_output_value[ctx.op.inputs["input"]]
        negative_slope_val = ctx.name_to_output_value[ctx.op.inputs["negative_slope"]]

        node.addInput(input_val)
        node.addInput(negative_slope_val)

        node.output().setType(torch._C.TensorType.get())

        return [node]
    
    def _convert_elu(self, ctx: ConversionContext) -> List[OpType]:
        node = ctx.torch_graph.create("aten::elu")

        input_val = ctx.name_to_output_value[ctx.op.inputs["input"]]
        alpha_val = ctx.name_to_output_value[ctx.op.inputs["alpha"]]

        node.addInput(input_val)
        node.addInput(alpha_val)
        # aten::elu has two weird extra parameters scale, and input_scale, which similar to add are completely unnessisary
        node.addInput(ctx.torch_graph.insertConstant(1))
        node.addInput(ctx.torch_graph.insertConstant(1))

        return [node]