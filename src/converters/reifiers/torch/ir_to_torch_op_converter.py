from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

import torch

from ....graph.dag_graph import DAGGraph
from ....ir.safe_ir import (
    BinaryElementwiseSpec,
    OpType,
    PadSpec,
    ReduceSpec,
    UnaryElementwiseSpec,
)
from ...op_converter import OpConverter

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
    f"binary_{BinaryElementwiseSpec.ADD}": "aten::add",
    f"binary_{BinaryElementwiseSpec.SUBTRACT}": "aten::sub",
    f"binary_{BinaryElementwiseSpec.MULTIPLY}": "aten::multiply",
    f"binary_{BinaryElementwiseSpec.DIVIDE}": "aten::div",
}

REDUCE_SPEC_TO_ATEN = {
    f"reduce_{ReduceSpec.ReductionType.SUM}": "aten::sum",
    f"reduce_{ReduceSpec.ReductionType.MEAN}": "aten::mean",
    f"reduce_{ReduceSpec.ReductionType.MAX}": "aten::amax",
    f"reduce_{ReduceSpec.ReductionType.MIN}": "aten::amin",
    f"reduce_{ReduceSpec.ReductionType.PROD}": "aten::prod",
}


@dataclass
class ConversionContext:
    op: OpType
    graph: DAGGraph
    torch_graph: torch._C.Graph
    name_to_output_value: Dict[str, torch._C.Value]


class IRToTorchOpConverter(OpConverter[ConversionContext, Callable, OpType, List]):
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

        self._converters.update(
            {key: self._convert_reduce for key in REDUCE_SPEC_TO_ATEN.keys()}
        )

        self._converters.update(
            {
                "permute": self._convert_permute,
                "index": self._convert_index,
                "group": self._convert_group,
                "ungroup": self._convert_group,
                "pad": self._convert_pad,
                "fold": self._convert_fold,
                "unfold": self._convert_fold,
            }
        )

    def _create_context(
        self,
        op: OpType,
        graph: DAGGraph,
        torch_graph: torch._C.Graph,
        name_to_output_value: Dict[str, torch._C.Value],
    ) -> ConversionContext:
        return ConversionContext(op, graph, torch_graph, name_to_output_value)

    def _get_operation_key(self, op: OpType) -> str:
        return op.type

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
            case (
                "TensorType",
                _,
                BinaryElementwiseSpec.ADD | BinaryElementwiseSpec.SUBTRACT,
            ):
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

    def _convert_reduce(self, ctx: ConversionContext) -> List[torch._C.Node]:
        input_val = ctx.name_to_output_value[ctx.op.inputs["input"]]
        reduction_dims = ctx.op.spec.reduce_dimensions
        squeeze_dims = ctx.op.spec.squeeze_dimensions
        if len(squeeze_dims) > 0 and reduction_dims != squeeze_dims:
            raise Exception(
                "Pytorch cannot keep only some reduced dims only all or nothing."
            )

        const_true = ctx.torch_graph.insertConstant(True)
        const_none = ctx.torch_graph.insertConstant(None)

        if ctx.op.type == "reduce_prod":
            out_nodes = []
            for r in reduction_dims:
                node = ctx.torch_graph.create(REDUCE_SPEC_TO_ATEN[ctx.op.type])
                node.addInput(input_val)
                node.addInput(ctx.torch_graph.insertConstant(r))
                node.addInput(const_true)
                node.addInput(const_none)
                input_val = node.output()
                out_nodes.append(node)

            if len(squeeze_dims) > 0:
                node = ctx.torch_graph.create("aten::squeeze")
                node.addInput(input_val)
                out_nodes.append(node)
            return out_nodes
        else:
            keep_dims = ctx.torch_graph.insertConstant(
                True if len(squeeze_dims) == 0 else False
            )
            node = ctx.torch_graph.create(REDUCE_SPEC_TO_ATEN[ctx.op.type])

            node.addInput(input_val)
            node.addInput(ctx.torch_graph.insertConstant(reduction_dims))
            node.addInput(keep_dims)
            if ctx.op.type in ["reduce_sum", "reduce_mean"]:
                node.addInput(const_none)  # for some reason sum has an output dtype

            return [node]

    def _convert_permute(self, ctx: ConversionContext) -> List[torch._C.Node]:
        node = ctx.torch_graph.create("aten::permute")

        input_val = ctx.name_to_output_value[ctx.op.inputs["input"]]

        node.addInput(input_val)
        node.addInput(ctx.torch_graph.insertConstant(ctx.op.spec.new_permutation))
        node.output().setType(torch._C.TensorType.get())

        return [node]

    def _convert_index(self, ctx: ConversionContext) -> List[torch._C.Node]:
        sorted_inds = sorted(ctx.op.spec.index.keys())

        input_val = ctx.name_to_output_value[ctx.op.inputs["input"]]

        num_removed = 0
        nodes = []

        for i in sorted_inds:
            curr_val = ctx.op.spec.index[i]
            if isinstance(curr_val, tuple):
                node = ctx.torch_graph.create("aten::slice")

                node.addInput(input_val)
                node.addInput(ctx.torch_graph.insertConstant(i - num_removed))
                node.addInput(ctx.torch_graph.insertConstant(curr_val[0]))
                # from inclusive to exclusive end
                if curr_val[1] == -1:  # if to end of list replace with 2^63 - 1
                    node.addInput(ctx.torch_graph.insertConstant(9223372036854775807))
                else:
                    node.addInput(ctx.torch_graph.insertConstant(curr_val[1] + 1))
                node.addInput(ctx.torch_graph.insertConstant(curr_val[2]))
            else:
                node = ctx.torch_graph.create("aten::select")

                node.addInput(input_val)
                node.addInput(ctx.torch_graph.insertConstant(i - num_removed))
                node.addInput(ctx.torch_graph.insertConstant(curr_val))

                num_removed += 1

            node.output().setType(torch._C.TensorType.get())

            nodes.append(node)

        return nodes

    def _convert_group(self, ctx: ConversionContext) -> List[torch._C.Node]:
        input_val = ctx.name_to_output_value[ctx.op.inputs["input"]]
        node = ctx.torch_graph.create("aten::reshape")
        node.addInput(input_val)
        node.addInput(ctx.torch_graph.insertConstant(ctx.op.spec._output_shape_sidecar))
        return [node]

    def _convert_pad(self, ctx: ConversionContext) -> List[torch._C.Node]:
        input_val = ctx.name_to_output_value[ctx.op.inputs["input"]]
        node = ctx.torch_graph.create("aten::pad")
        node.addInput(input_val)

        # padding is stored in reverse last dim -> first dim
        pad_arr = []
        for i in range(
            ctx.op.spec._ouptut_dims_sidecar - 1, min(ctx.op.spec.pad.keys()) - 1, -1
        ):
            if i in ctx.op.spec.pad:
                pad_arr.extend(list(ctx.op.spec.pad[i]))
            else:
                pad_arr.extend([0, 0])

        if isinstance(ctx.op.spec.pad_mode, PadSpec.PadMode):
            # non-constant padding only works with 2-5D tensors
            if len(pad_arr) < 4:
                pad_arr += [0, 0]
            node.addInput(ctx.torch_graph.insertConstant(pad_arr))
            node.addInput(ctx.torch_graph.insertConstant(ctx.op.spec.pad_mode.to_str()))
            node.addInput(ctx.torch_graph.insertConstant(None))
        else:
            node.addInput(ctx.torch_graph.insertConstant(pad_arr))
            node.addInput(ctx.torch_graph.insertConstant("constant"))
            node.addInput(ctx.torch_graph.insertConstant(float(ctx.op.spec.pad_mode)))

        return [node]

    def _convert_fold(self, ctx: ConversionContext) -> List[torch._C.Node]:
        input_val = ctx.name_to_output_value[ctx.op.inputs["input"]]

        # fold and unfold can only really work on 3D or 4D tensors in torch
        match ctx.op.spec.type:
            case "fold":
                node = ctx.torch_graph.create("aten::col2im")
                node.addInput(input_val)
                node.addInput(
                    ctx.torch_graph.insertConstant(
                        ctx.op.spec._output_shape_sidecar[-2:]
                    )
                )
                fold_dict = ctx.op.spec.fold
            case "unfold":
                node = ctx.torch_graph.create("aten::im2col")
                node.addInput(input_val)
                fold_dict = ctx.op.spec.unfold
            case _:
                raise Exception(f"Fold type Unknown: {ctx.op.spec.type}")

        kernel_h, stride_h, dilation_h = fold_dict[2] if 2 in fold_dict else (0, 0, 0)
        kernel_w, stride_w, dilation_w = fold_dict[3] if 3 in fold_dict else (0, 0, 0)
        kernel = [kernel_h, kernel_w]
        stride = [stride_h, stride_w]
        dilation = [dilation_h, dilation_w]

        node.addInput(ctx.torch_graph.insertConstant(kernel))
        node.addInput(ctx.torch_graph.insertConstant(dilation))
        node.addInput(ctx.torch_graph.insertConstant([0, 0]))
        node.addInput(ctx.torch_graph.insertConstant(stride))

        return [node]
