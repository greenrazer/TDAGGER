from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union
import torch

from ...ir.safe_ir import OpType, BinaryArithmeticType
from ...graph.dag_graph import DAGGraph

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
            "binary_multiply": self._convert_multiply
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