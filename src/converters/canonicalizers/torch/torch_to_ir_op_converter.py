from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

import torch

from ....ir.safe_ir import (
    BinaryElementwiseSpec,
    BinaryElementwiseType,
    DataType,
    OpType,
    ScalarType,
    TensorSpec,
    TensorType,
    UnaryElementwiseSpec,
    UnaryElementwiseType,
    PermuteSpec,
    PermuteType,
    ReduceSpec,
    ReduceType,
)
from ..canon_op_converter import CanonOpConverter

ATEN_TO_UNARY_ELEMENTWISE_SPEC = {
    "aten::abs": UnaryElementwiseSpec.ABSOLUTE_VALUE,
    "aten::neg": UnaryElementwiseSpec.NEGATIVE,
    "aten::sqrt": UnaryElementwiseSpec.SQUARE_ROOT,
    "aten::exp": UnaryElementwiseSpec.EXP,
    "aten::log": UnaryElementwiseSpec.LOG,
    "aten::sin": UnaryElementwiseSpec.SIN,
    "aten::cos": UnaryElementwiseSpec.COS,
    "aten::tan": UnaryElementwiseSpec.TAN,
    "aten::asin": UnaryElementwiseSpec.ARCSIN,
    "aten::acos": UnaryElementwiseSpec.ARCCOS,
    "aten::atan": UnaryElementwiseSpec.ARCTAN,
    "aten::sinh": UnaryElementwiseSpec.SINH,
    "aten::cosh": UnaryElementwiseSpec.COSH,
    "aten::tanh": UnaryElementwiseSpec.TANH,
    "aten::asinh": UnaryElementwiseSpec.ARCSINH,
    "aten::acosh": UnaryElementwiseSpec.ARCCOSH,
    "aten::atanh": UnaryElementwiseSpec.ARCTANH,
}

ATEN_TO_BINARY_ELEMENTWISE_SPEC = {
    "aten::add": BinaryElementwiseSpec.ADD,
    "aten::sub": BinaryElementwiseSpec.SUBTRACT,
    "aten::mul": BinaryElementwiseSpec.MULTIPLY,
    "aten::div": BinaryElementwiseSpec.DIVIDE,
}

ATEN_TO_REDUCE_SPEC_TYPE = {
    "aten::sum": ReduceSpec.ReductionType.SUM,
    "aten::mean": ReduceSpec.ReductionType.MEAN,
    "aten::amax": ReduceSpec.ReductionType.MAX,
    "aten::amin": ReduceSpec.ReductionType.MIN,
    "aten::prod": ReduceSpec.ReductionType.PROD,
}


@dataclass
class ConversionContext:
    torch_op: torch._C.Node
    forward_graph: torch._C.Graph
    output_value_to_node: Dict[torch._C.Value, torch._C.Node]
    output_value_to_name: Dict[torch._C.Value, str]
    debug_sources: Union[None, List[Tuple[str, str, str]]]


class TorchToIROpConverter(
    CanonOpConverter[ConversionContext, Callable, torch._C.Node, Tuple]
):
    def _register_converters(self):
        self._converters.update(
            {
                "aten::relu": self._convert_relu,
                "aten::leaky_relu": self._convert_leaky_relu,
                "aten::softplus": self._convert_softplus,
                "aten::permute": self._convert_permute,
            }
        )

        self._converters.update(
            {
                key: self._convert_binary_elementwise
                for key in ATEN_TO_BINARY_ELEMENTWISE_SPEC.keys()
            }
        )

        self._converters.update(
            {
                key: self._convert_unary_elementwise
                for key in ATEN_TO_UNARY_ELEMENTWISE_SPEC.keys()
            }
        )

        self._converters.update(
            {key: self._convert_reduce for key in ATEN_TO_REDUCE_SPEC_TYPE.keys()}
        )

    def _create_context(
        self,
        torch_op: torch._C.Node,
        forward_graph: torch._C.Graph,
        output_value_to_node: Dict[torch._C.Value, torch._C.Node],
        output_value_to_name: Dict[torch._C.Value, str],
        debug_sources: Union[None, List[Tuple[str, str, str]]] = [],
    ) -> ConversionContext:
        return ConversionContext(
            torch_op,
            forward_graph,
            output_value_to_node,
            output_value_to_name,
            debug_sources,
        )

    def _get_operation_key(self, torch_op: "torch._C.Node") -> str:
        return torch_op.kind()

    def _inputs_to_names(self, ctx: ConversionContext) -> List[str]:
        input_names = []
        for input_value in ctx.torch_op.inputs():
            if input_value in ctx.output_value_to_name:
                in_name = ctx.output_value_to_name[input_value]
            else:
                in_name = input_value.debugName().replace(".", "_")

            input_names.append(in_name)
        return input_names

    def _inputs_to_nodes(self, ctx: ConversionContext) -> List[torch._C.Node]:
        return [ctx.output_value_to_node[i] for i in ctx.torch_op.inputs()]

    def _inputs_to_torch_types(self, ctx):
        return [i.type().kind() for i in ctx.torch_op.inputs()]

    def _inputs_constants_to_values(self, ctx):
        input_scalar_values = []
        for input_value in ctx.torch_op.inputs():
            node = ctx.output_value_to_node[input_value]
            # node is none if it is a placeholder
            if node is None or node.kind() not in ["prim::Constant"]:
                input_scalar_values.append(None)
                continue
            inp_type = input_value.type()
            match inp_type.kind():
                case "IntType":
                    input_scalar_values.append(node.i("value"))
                case "FloatType":
                    input_scalar_values.append(node.f("value"))
                case "StringType":
                    input_scalar_values.append(node.s("value"))
                case "BoolType":
                    input_scalar_values.append(bool(node.i("value")))
                case "ListType":
                    input_scalar_values.append(node.output().toIValue())
                case _:
                    input_scalar_values.append(None)

        return input_scalar_values
    

    def _convert_relu(
        self, ctx: ConversionContext
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = ctx.torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(ctx)
        return self._create_relu(
            out_name, input_names[0], debug_sources=ctx.debug_sources
        )

    def _convert_leaky_relu(
        self, ctx: ConversionContext
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = ctx.torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(ctx)
        return self._create_leaky_relu(
            out_name,
            input_names[0],
            input_names[1],
            debug_sources=ctx.debug_sources,
        )

    def _convert_softplus(
        self, ctx: ConversionContext
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = ctx.torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(ctx)
        return self._create_softplus(
            out_name,
            input_names[0],
            input_names[1],
            debug_sources=ctx.debug_sources,
        )

    def _convert_permute(
        self, ctx: ConversionContext
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = ctx.torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(ctx)
        input_constant_values = self._inputs_constants_to_values(ctx)

        permute_op = PermuteType(
            name=out_name,
            inputs={"input": input_names[0]},
            spec=PermuteSpec(new_permutation=input_constant_values[1]),
            debug_sources=ctx.debug_sources,
        )

        return [permute_op], {}

    def _convert_binary_elementwise(
        self, ctx: ConversionContext
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = ctx.torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(ctx)

        # torch add unintuitively has 3 inputs a, b, and alpha for a + alpha*b
        # I will ignore the alpha parameter beacuse it is not nessisary,
        # and if someone is using it they are doing something wrong
        bin_op = BinaryElementwiseType(
            name=out_name,
            inputs={"input_0": input_names[0], "input_1": input_names[1]},
            spec=ATEN_TO_BINARY_ELEMENTWISE_SPEC[ctx.torch_op.kind()],
            debug_sources=ctx.debug_sources,
        )

        return [bin_op], {}

    def _convert_unary_elementwise(
        self, ctx: ConversionContext
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = ctx.torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(ctx)

        unary_op = UnaryElementwiseType(
            name=out_name,
            inputs={"input": input_names[0]},
            spec=ATEN_TO_UNARY_ELEMENTWISE_SPEC[ctx.torch_op.kind()],
            debug_sources=ctx.debug_sources,
        )

        return [unary_op], {}

    def _convert_reduce(
        self, ctx: ConversionContext
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = ctx.torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(ctx)
        input_types = self._inputs_to_torch_types(ctx)
        input_constant_values = self._inputs_constants_to_values(ctx)

        reduce_dimensions = input_constant_values[1] if input_types[1] == "ListType" else [input_constant_values[1]]
        squeeze_dimensions = [] if input_constant_values[2] else reduce_dimensions

        reduction_spec = ReduceSpec(
            reduce_dimensions=reduce_dimensions,
            squeeze_dimensions=squeeze_dimensions,
            reduction_type=ATEN_TO_REDUCE_SPEC_TYPE[ctx.torch_op.kind()],
        )

        reduction_op = ReduceType(
            out_name,
            {"input": input_names[0]},
            reduction_spec,
            debug_sources=ctx.debug_sources,
        )

        return [reduction_op], {}

