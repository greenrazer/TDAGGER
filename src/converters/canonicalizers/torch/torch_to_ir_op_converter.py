import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union

import torch

from ....ir.safe_ir import (
    BinaryElementwiseSpec,
    BinaryTensorInput,
    DataHolderType,
    DataType,
    FoldSpec,
    GroupSpec,
    InterleaveSpec,
    OpInput,
    OpSpec,
    OpType,
    PadSpec,
    PermuteSpec,
    ReduceSpec,
    RepeatSpec,
    ScalarSpec,
    ScalarType,
    SliceSpec,
    SpecType,
    SqueezeSpec,
    StrideSpec,
    TensorSpec,
    TensorType,
    UnaryElementwiseSpec,
    UnaryTensorInput,
    UnfoldSpec,
    UngroupSpec,
    UnsqueezeSpec,
)
from ..canon_op_converter import CanonOpConverter
from .group_helpers import specs_from_reshape

ATEN_TO_UNARY_ELEMENTWISE_SPEC = {
    "aten::sign": UnaryElementwiseSpec.UnaryElementwiseType.SIGN,
    "aten::neg": UnaryElementwiseSpec.UnaryElementwiseType.NEGATIVE,
    "aten::reciprocal": UnaryElementwiseSpec.UnaryElementwiseType.RECIPROCAL,
    "aten::exp": UnaryElementwiseSpec.UnaryElementwiseType.EXP,
    "aten::log": UnaryElementwiseSpec.UnaryElementwiseType.LOG,
    "aten::sin": UnaryElementwiseSpec.UnaryElementwiseType.SIN,
    "aten::cos": UnaryElementwiseSpec.UnaryElementwiseType.COS,
    "aten::tan": UnaryElementwiseSpec.UnaryElementwiseType.TAN,
    "aten::asin": UnaryElementwiseSpec.UnaryElementwiseType.ARCSIN,
    "aten::acos": UnaryElementwiseSpec.UnaryElementwiseType.ARCCOS,
    "aten::atan": UnaryElementwiseSpec.UnaryElementwiseType.ARCTAN,
    "aten::sinh": UnaryElementwiseSpec.UnaryElementwiseType.SINH,
    "aten::cosh": UnaryElementwiseSpec.UnaryElementwiseType.COSH,
    "aten::tanh": UnaryElementwiseSpec.UnaryElementwiseType.TANH,
    "aten::asinh": UnaryElementwiseSpec.UnaryElementwiseType.ARCSINH,
    "aten::acosh": UnaryElementwiseSpec.UnaryElementwiseType.ARCCOSH,
    "aten::atanh": UnaryElementwiseSpec.UnaryElementwiseType.ARCTANH,
}

ATEN_TO_BINARY_ELEMENTWISE_SPEC = {
    "aten::add": BinaryElementwiseSpec.BinaryElementwiseType.ADD,
    "aten::mul": BinaryElementwiseSpec.BinaryElementwiseType.MULTIPLY,
    "aten::pow": BinaryElementwiseSpec.BinaryElementwiseType.EXPONENTIATE,
}

ATEN_TO_REDUCE_SPEC_TYPE = {
    "aten::sum": ReduceSpec.ReductionType.SUM,
    "aten::prod": ReduceSpec.ReductionType.PRODUCT,
    "aten::amax": ReduceSpec.ReductionType.MAXIMUM,
    "aten::amin": ReduceSpec.ReductionType.MINIMUM,
    "aten::mean": ReduceSpec.ReductionType.MEAN,
}


@dataclass
class ConversionContext:
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

        self._converters.update(
            {
                "aten::sub": self._convert_subtract,
                "aten::div": self._convert_divide,
                "aten::abs": self._convert_abs,
                "aten::relu": self._convert_relu,
                "aten::leaky_relu": self._convert_leaky_relu,
                "aten::softplus": self._convert_softplus,
                "aten::permute": self._convert_permute,
                "aten::slice": self._convert_index,
                "aten::select": self._convert_index,
                "aten::reshape": self._convert_reshape,
                "aten::pad": self._convert_pad,
                "aten::im2col": self._convert_fold,
                "aten::col2im": self._convert_fold,
                "aten::squeeze": self._convert_squeeze,
                "aten::unsqueeze": self._convert_squeeze,
            }
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
            forward_graph,
            output_value_to_node,
            output_value_to_name,
            debug_sources,
        )

    def _get_operation_key(self, torch_op: "torch._C.Node") -> str:
        return torch_op.kind()

    def _inputs_to_names(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> List[str]:
        input_names = []
        for input_value in torch_op.inputs():
            if input_value in context.output_value_to_name:
                in_name = context.output_value_to_name[input_value]
            else:
                in_name = input_value.debugName().replace(".", "_")

            input_names.append(in_name)
        return input_names

    def _inputs_to_nodes(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> List[torch._C.Node]:
        return [context.output_value_to_node[i] for i in torch_op.inputs()]

    def _inputs_to_torch_types(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> List[str]:
        return [i.type().kind() for i in torch_op.inputs()]

    def _inputs_constants_to_values(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> List[Any]:
        input_scalar_values = []
        for input_value in torch_op.inputs():
            node = context.output_value_to_node[input_value]
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

    def _convert_binary_elementwise(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(context, torch_op)

        # torch add unintuitively has 3 inputs a, b, and alpha for a + alpha*b
        # I will ignore the alpha parameter beacuse it is not nessisary,
        # and if someone is using it they are doing something wrong
        bin_op = OpType(
            name=out_name,
            input=BinaryTensorInput(input_names[0], input_names[1]),
            spec=BinaryElementwiseSpec(
                ATEN_TO_BINARY_ELEMENTWISE_SPEC[torch_op.kind()]
            ),
            debug_sources=context.debug_sources,
        )

        return [bin_op], {}

    def _convert_unary_elementwise(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(context, torch_op)

        unary_op = OpType(
            name=out_name,
            input=UnaryTensorInput(input_names[0]),
            spec=UnaryElementwiseSpec(ATEN_TO_UNARY_ELEMENTWISE_SPEC[torch_op.kind()]),
            debug_sources=context.debug_sources,
        )

        return [unary_op], {}

    def _convert_reduce(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(context, torch_op)
        input_types = self._inputs_to_torch_types(context, torch_op)
        input_constant_values = self._inputs_constants_to_values(context, torch_op)

        reduce_dimensions = set(
            input_constant_values[1]
            if input_types[1] == "ListType"
            else [input_constant_values[1]]
        )

        reduction_op = OpType(
            name=out_name if input_constant_values[2] else f"{out_name}_reduction",
            input=UnaryTensorInput(input_names[0]),
            spec=ReduceSpec(
                dimensions=reduce_dimensions,
                reduction_type=ATEN_TO_REDUCE_SPEC_TYPE[torch_op.kind()],
            ),
            debug_sources=context.debug_sources,
        )

        output = [reduction_op]

        if not input_constant_values[2]:
            squeeze_op = OpType(
                name=out_name,
                input=UnaryTensorInput(reduction_op.name),
                spec=SqueezeSpec(dimensions=reduce_dimensions),
                debug_sources=context.debug_sources,
            )

            output.append(squeeze_op)

        return output, {}

    def _convert_subtract(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(context, torch_op)
        return self._create_subtract(
            out_name,
            input_names[0],
            input_names[1],
            debug_sources=context.debug_sources,
        )

    def _convert_divide(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(context, torch_op)
        return self._create_divide(
            out_name,
            input_names[0],
            input_names[1],
            debug_sources=context.debug_sources,
        )

    def _convert_abs(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(context, torch_op)
        return self._create_elementwise_abs(
            out_name, input_names[0], debug_sources=context.debug_sources
        )

    def _convert_relu(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(context, torch_op)
        return self._create_relu(
            out_name, input_names[0], debug_sources=context.debug_sources
        )

    def _convert_leaky_relu(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(context, torch_op)
        return self._create_leaky_relu(
            out_name,
            input_names[0],
            input_names[1],
            debug_sources=context.debug_sources,
        )

    def _convert_softplus(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(context, torch_op)
        return self._create_softplus(
            out_name,
            input_names[0],
            input_names[1],
            debug_sources=context.debug_sources,
        )

    def _convert_permute(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(context, torch_op)
        input_constant_values = self._inputs_constants_to_values(context, torch_op)

        permute_op = OpType(
            name=out_name,
            input=UnaryTensorInput(input_names[0]),
            spec=PermuteSpec(
                permutation={d: p for d, p in enumerate(input_constant_values[1])}
            ),
            debug_sources=context.debug_sources,
        )

        return [permute_op], {}

    def _convert_index(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(context, torch_op)
        input_constant_values = self._inputs_constants_to_values(context, torch_op)

        if torch_op.kind() == "aten::select":
            index_obj = input_constant_values[2]
        elif torch_op.kind() == "aten::slice":
            # step can never be negitive in pytorch slices
            match (
                input_constant_values[2],
                input_constant_values[3],
                input_constant_values[4],
            ):
                case (begin, 9223372036854775807, step):
                    index_obj = (begin, -1, step)
                case (begin, end, step):
                    # end - 1 because inclusive indexing
                    index_obj = (begin, end - 1, step)

        index_op = OpType(
            name=out_name,
            input=UnaryTensorInput(input_names[0]),
            spec=SliceSpec(slice={input_constant_values[1]: index_obj}),
            debug_sources=context.debug_sources,
        )

        return [index_op], {}

    def _convert_reshape(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = torch_op.output().debugName().replace(".", "_")
        input_names = self._inputs_to_names(context, torch_op)
        input_constant_values = self._inputs_constants_to_values(context, torch_op)

        input_shape = torch_op.inputsAt(0).type().sizes()
        output_shape = input_constant_values[1]

        spec_list = specs_from_reshape(input_shape, output_shape)

        out = []
        for s in spec_list[:-1]:
            out.append(
                OpType(
                    f"{out_name}_{s.type}",
                    input=UnaryTensorInput(input_names[0]),
                    spec=s,
                    debug_sources=context.debug_sources,
                )
            )

        out.append(
            OpType(
                f"{out_name}",
                input=UnaryTensorInput(
                    input_names[0] if len(out) == 0 else out[-1].name
                ),
                spec=spec_list[-1],
                debug_sources=context.debug_sources,
            )
        )

        return out, {}

    def _convert_pad(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = torch_op.output().debugName().replace(".", "_")
        input_names = self._inputs_to_names(context, torch_op)
        input_constant_values = self._inputs_constants_to_values(context, torch_op)

        input_shape = torch_op.inputsAt(0).type().sizes()
        input_pad = input_constant_values[1]
        pad_constant = (
            0 if input_constant_values[3] is None else input_constant_values[3]
        )
        pad_mode = (
            pad_constant
            if input_constant_values[2] == "constant"
            else PadSpec.PadMode.from_str(input_constant_values[2])
        )

        # padding is stored in reverse last dim -> first dim
        pad_dict = {}
        for pad_idx, dim_idx in zip(
            range(len(input_pad) - 1, -1, -2),
            range(len(input_shape) - len(input_pad) // 2, len(input_shape)),
        ):
            pad_tup = (input_pad[pad_idx - 1], input_pad[pad_idx])
            if pad_tup != (0, 0):
                pad_dict[dim_idx] = pad_tup

        pad_op = OpType(
            name=out_name,
            input=UnaryTensorInput(input_names[0]),
            spec=PadSpec(
                pad=pad_dict, pad_mode=pad_mode, _ouptut_dims_sidecar=len(input_shape)
            ),
            debug_sources=context.debug_sources,
        )

        return [pad_op], {}

    def _convert_fold(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = torch_op.output().debugName().replace(".", "_")
        input_names = self._inputs_to_names(context, torch_op)
        input_constant_values = self._inputs_constants_to_values(context, torch_op)

        input_shape = torch_op.inputsAt(0).type().sizes()
        output = []

        # currently kernel must be 2d and only affects the last 2 dimensions
        if torch_op.kind() == "aten::im2col":
            kernel_shape = input_constant_values[1]
            dilation = input_constant_values[2]
            padding = input_constant_values[3]
            has_padding = any([p > 0 for p in padding])
            stride = input_constant_values[4]

            if has_padding:
                pad_dict = {
                    (i + len(input_shape) - 2): (p, p)
                    for i, p in enumerate(padding)
                    if p > 0
                }
                pad_op = OpType(
                    name=f"{out_name}_pad",
                    input=UnaryTensorInput(input_names[0]),
                    spec=PadSpec(
                        pad=pad_dict, pad_mode=0, _ouptut_dims_sidecar=len(input_shape)
                    ),
                    debug_sources=context.debug_sources,
                )
                output.append(pad_op)

            unfold_dict = {
                (i + len(input_shape) - 2): (k, s, d)
                for i, (k, s, d) in enumerate(zip(kernel_shape, stride, dilation))
                if k != 0
            }

            def out_size(d):
                return math.floor(
                    (
                        input_shape[d]
                        + 2 * padding[d]
                        - dilation[d] * (kernel_shape[d] - 1)
                        - 1
                    )
                    / stride[d]
                    + 1
                )

            out_shape = []
            if len(input_shape) == 4:
                out_shape.append(input_shape[-4])
            out_shape.extend([input_shape[-3], out_size(-2), out_size(-1)])
            spec = UnfoldSpec(unfold=unfold_dict, _output_shape_sidecar=out_shape)
            unfold_op = OpType(
                name=out_name,
                input=UnaryTensorInput(
                    input_names[0] if len(output) == 0 else output[-1].name
                ),
                spec=spec,
                debug_sources=context.debug_sources,
            )

            output.append(unfold_op)

            return output, {}
        elif torch_op.kind() == "aten::col2im":
            input_shape = torch_op.inputsAt(0).type().sizes()
            h, w = input_constant_values[1]
            kernel_shape = input_constant_values[2]
            kernel_size = kernel_shape[0] * kernel_shape[1]
            dilation = input_constant_values[3]
            padding = input_constant_values[4]
            has_padding = any([p > 0 for p in padding])
            stride = input_constant_values[5]

            unfold_dict = {
                (i + len(input_shape) - 1): (k, s, d)
                for i, (k, s, d) in enumerate(zip(kernel_shape, stride, dilation))
                if k != 0
            }
            out_shape = []
            if len(input_shape) == 3:
                out_shape.append(input_shape[-3])
            out_shape.extend(
                [
                    input_shape[-2] // kernel_size,
                    h + 2 * padding[-2],
                    w + 2 * padding[-1],
                ]
            )
            # the outside equation should be (input_shape[d] - 1) * stride[d] - 2 * padding[d] + dilation[d] * (kernel_size[d] - 1) + 1
            spec = FoldSpec(
                fold=unfold_dict,
                _output_shape_sidecar=out_shape,
            )
            fold_op = OpType(
                name=f"{out_name}_fold" if has_padding else out_name,
                input=UnaryTensorInput(input_names[0]),
                spec=spec,
                debug_sources=context.debug_sources,
            )

            output.append(fold_op)

            if has_padding:
                index_dict = {
                    (i + len(input_shape) - 1): (p, -p - 1)
                    for i, p in enumerate(padding)
                    if p > 0
                }
                index_op = OpType(
                    name=out_name,
                    input=UnaryTensorInput(fold_op.name),
                    spec=SliceSpec(slice=index_dict),
                    debug_sources=context.debug_sources,
                )
                output.append(index_op)

        else:
            raise Exception(f"Unknown torch fold op: {torch_op.kind()}.")

        return output, {}

    def _convert_squeeze(
        self, context: ConversionContext, torch_op: torch._C.Node
    ) -> Tuple[List[OpType], Dict[str, Union[ScalarType, TensorType]]]:
        out_name = torch_op.output().debugName().replace(".", "_")

        input_names = self._inputs_to_names(context, torch_op)
        input_constant_values = self._inputs_constants_to_values(context, torch_op)
        input_types = self._inputs_to_torch_types(context, torch_op)
        input_shape = torch_op.inputsAt(0).type().sizes()
        match torch_op.kind():
            case "aten::squeeze":
                if len(input_constant_values) == 2:
                    dims = (
                        input_constant_values[1]
                        if input_types[1] == "ListType"
                        else [input_constant_values[1]]
                    )
                else:
                    dims = [i for i, d in enumerate(input_shape) if d == 1]
                squeeze_spec = SqueezeSpec(dimensions=dims)
            case "aten::unsqueeze":
                squeeze_spec = UnsqueezeSpec(dimensions=[input_constant_values[1]])

        sq_op = OpType(
            name=out_name,
            input=UnaryTensorInput(input_names[0]),
            spec=squeeze_spec,
            debug_sources=context.debug_sources,
        )

        return [sq_op], {}
