from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union
import torch

from ...ir.safe_ir import OpType, BinaryArithmeticType, BinaryArithmeticSpec


@dataclass
class ConversionContext:
    torch_op: torch._C.Node
    forward_graph: torch._C.Graph
    output_value_to_node: Dict[torch._C.Value, torch._C.Node]
    output_value_to_name: Dict[torch._C.Value, str]
    debug_sources: Union[None, List[Tuple[str, str, str]]]


class TorchOpConverter:
    def __init__(self):
        self._converters: Dict[
            str, Callable[[ConversionContext], List[Tuple[str, OpType]]]
        ] = {}
        self._register_converters()

    def _register_converters(self):
        self._converters.update(
            {
                "aten::add": self._convert_add,
            }
        )

    def convert_op(
        self,
        torch_op: torch._C.Node,
        forward_graph: torch._C.Graph,
        output_value_to_node: Dict[torch._C.Value, torch._C.Node],
        output_value_to_name: Dict[torch._C.Value, str],
        debug_sources: Union[None, List[Tuple[str, str, str]]] = [],
    ) -> List[Tuple[str, OpType]]:
        ctx = ConversionContext(
            torch_op,
            forward_graph,
            output_value_to_node,
            output_value_to_name,
            debug_sources,
        )

        if torch_op.kind() not in self._converters:
            raise Exception(f"Unsupported operation type: {torch_op.kind()}")

        return self._converters[torch_op.kind()](ctx)

    def _convert_add(self, ctx: ConversionContext) -> List[OpType]:
        # torch add unintuitively has 3 inputs a, b, and alpha for a + alpha*b

        input_names = []
        for input_value in ctx.torch_op.inputs():
            if input_value in ctx.output_value_to_name:
                in_name = ctx.output_value_to_name[input_value]
            else:
                in_name = input_value.debugName().replace(".", "_")

            input_names.append(in_name)

        out_name = ctx.torch_op.output().debugName().replace(".", "_")
        multiply_op = BinaryArithmeticType(
            name=f"{out_name}_multiply",
            inputs={
                "input_0": input_names[1],
                "input_1": input_names[2]
            },
            spec=BinaryArithmeticSpec.MULTIPLY,
            debug_sources=ctx.debug_sources
        )
        add_op = BinaryArithmeticType(
            name=out_name,
            inputs={
                "input_0": input_names[0],
                "input_1": f"{out_name}_multiply"
            },
            spec=BinaryArithmeticSpec.ADD,
            debug_sources=ctx.debug_sources
        )

        return [multiply_op, add_op]
