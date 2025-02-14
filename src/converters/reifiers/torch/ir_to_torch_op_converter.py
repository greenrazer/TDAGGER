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
    str(UnaryElementwiseSpec.UnaryElementwiseType.SIGN): "aten::sign",
    str(UnaryElementwiseSpec.UnaryElementwiseType.NEGATIVE): "aten::neg",
    str(UnaryElementwiseSpec.UnaryElementwiseType.RECIPROCAL): "aten::reciprocal",
    str(UnaryElementwiseSpec.UnaryElementwiseType.EXP): "aten::exp",
    str(UnaryElementwiseSpec.UnaryElementwiseType.LOG): "aten::log",
    str(UnaryElementwiseSpec.UnaryElementwiseType.SIN): "aten::sin",
    str(UnaryElementwiseSpec.UnaryElementwiseType.COS): "aten::cos",
    str(UnaryElementwiseSpec.UnaryElementwiseType.TAN): "aten::tan",
    str(UnaryElementwiseSpec.UnaryElementwiseType.ARCSIN): "aten::asin",
    str(UnaryElementwiseSpec.UnaryElementwiseType.ARCCOS): "aten::acos",
    str(UnaryElementwiseSpec.UnaryElementwiseType.ARCTAN): "aten::atan",
    str(UnaryElementwiseSpec.UnaryElementwiseType.SINH): "aten::sinh",
    str(UnaryElementwiseSpec.UnaryElementwiseType.COSH): "aten::cosh",
    str(UnaryElementwiseSpec.UnaryElementwiseType.TANH): "aten::tanh",
    str(UnaryElementwiseSpec.UnaryElementwiseType.ARCSINH): "aten::asinh",
    str(UnaryElementwiseSpec.UnaryElementwiseType.ARCCOSH): "aten::acosh",
    str(UnaryElementwiseSpec.UnaryElementwiseType.ARCTANH): "aten::atanh",
}

BINARY_ELEMENTWISE_SPEC_TO_ATEN = {
    f"binary_{BinaryElementwiseSpec.BinaryElementwiseType.ADD}": "aten::add",
    f"binary_{BinaryElementwiseSpec.BinaryElementwiseType.MULTIPLY}": "aten::mul"
}

REDUCE_SPEC_TO_ATEN = {
    f"reduce_{ReduceSpec.ReductionType.SUM}": "aten::sum",
    f"reduce_{ReduceSpec.ReductionType.PRODUCT}": "aten::prod",
    f"reduce_{ReduceSpec.ReductionType.MAXIMUM}": "aten::amax",
    f"reduce_{ReduceSpec.ReductionType.MINIMUM}": "aten::amin",
    f"reduce_{ReduceSpec.ReductionType.MEAN}": "aten::mean",
}


@dataclass
class ConversionContext:
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
                "group": self._convert_group,
                "ungroup": self._convert_group,
                "slice": self._convert_slice,
                "pad": self._convert_pad,
                "select": self._convert_select,
                "fold": self._convert_fold,
                "unfold": self._convert_fold,
                "squeeze": self._convert_squeeze,
                "unsqueeze": self._convert_squeeze,
                "repeat": self._convert_repeat
            }
        )

    def _create_context(
        self,
        op: OpType,
        graph: DAGGraph,
        torch_graph: torch._C.Graph,
        name_to_output_value: Dict[str, torch._C.Value],
    ) -> ConversionContext:
        return ConversionContext(graph, torch_graph, name_to_output_value)

    def _get_operation_key(self, op: OpType) -> str:
        return op.spec.type

    def _convert_binary_elementwise(
        self, context: ConversionContext, op: OpType
    ) -> List[torch._C.Node]:
        input_0_val = context.name_to_output_value[op.input.input_0]
        input_1_val = context.name_to_output_value[op.input.input_1]

        node = context.torch_graph.create(BINARY_ELEMENTWISE_SPEC_TO_ATEN[op.spec.type])

        match (input_0_val.type().kind(), input_1_val.type().kind(), op.spec.op_type):
            case ("TensorType", _, BinaryElementwiseSpec.BinaryElementwiseType.ADD):
                node.addInput(input_0_val)
                node.addInput(input_1_val)
                node.addInput(context.torch_graph.insertConstant(1))
            case (_, "TensorType", BinaryElementwiseSpec.BinaryElementwiseType.ADD):
                node.addInput(input_1_val)
                node.addInput(input_0_val)
                node.addInput(context.torch_graph.insertConstant(1))
            case (
                "TensorType",
                _,
                BinaryElementwiseSpec.BinaryElementwiseType.MULTIPLY,
            ):
                node.addInput(input_0_val)
                node.addInput(input_1_val)
            case (
                _,
                "TensorType",
                BinaryElementwiseSpec.BinaryElementwiseType.MULTIPLY,
            ):
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

    def _convert_unary_elementwise(
        self, context: ConversionContext, op: OpType
    ) -> List[torch._C.Node]:
        node = context.torch_graph.create(UNARY_ELEMENTWISE_SPEC_TO_ATEN[op.spec.type])

        input_val = context.name_to_output_value[op.input.input]

        if input_val.type().kind() != "TensorType":
            # in -> in_tensor
            num_to_tensor_node = context.torch_graph.create("aten::scalar_tensor")
            num_to_tensor_node.addInput(input_val)
            # look at https://github.com/pytorch/pytorch/blob/main/c10/core/ScalarType.h
            num_to_tensor_node.addInput(context.torch_graph.insertConstant(6)) # 6 = float32
            num_to_tensor_node.addInput(context.torch_graph.insertConstant(None))
            num_to_tensor_node.addInput(context.torch_graph.insertConstant(None))
            num_to_tensor_node.addInput(context.torch_graph.insertConstant(None))

            # in_tensor -> op_tensor
            node.addInput(num_to_tensor_node.output())

            # op_tensor -> op
            tensor_to_num_node = context.torch_graph.create("aten::item")
            tensor_to_num_node.addInput(node.output())
            tensor_to_num_node.output().setType(torch._C.FloatType.get())

            out = [num_to_tensor_node, node, tensor_to_num_node]
        else:
            node.addInput(input_val)
            node.output().setType(torch._C.TensorType.get())
            out = [node]
        return out

    def _convert_reduce(
        self, context: ConversionContext, op: OpType
    ) -> List[torch._C.Node]:
        input_val = context.name_to_output_value[op.input.input]
        reduction_dims = op.spec.dimensions

        const_true = context.torch_graph.insertConstant(True)
        const_none = context.torch_graph.insertConstant(None)

        if op.spec.type == "reduce_prod":
            out_nodes = []
            for r in reduction_dims:
                node = context.torch_graph.create(REDUCE_SPEC_TO_ATEN[op.spec.type])
                node.addInput(input_val)
                node.addInput(context.torch_graph.insertConstant(r))
                node.addInput(const_true)
                node.addInput(const_none)
                input_val = node.output()
                out_nodes.append(node)
            return out_nodes
        else:
            node = context.torch_graph.create(REDUCE_SPEC_TO_ATEN[op.spec.type])

            node.addInput(input_val)
            node.addInput(context.torch_graph.insertConstant(list(reduction_dims)))
            node.addInput(context.torch_graph.insertConstant(True))
            if op.spec.type in ["reduce_sum", "reduce_mean"]:
                node.addInput(const_none)  # for some reason sum has an output dtype

            return [node]

    def _convert_permute(
        self, context: ConversionContext, op: OpType
    ) -> List[torch._C.Node]:
        node = context.torch_graph.create("aten::permute")

        input_val = context.name_to_output_value[op.input.input]

        max_dim = max(op.spec.permutation.keys())
        torch_perm = []
        for d in range(max_dim + 1):
            if d in op.spec.permutation:
                torch_perm.append(op.spec.permutation[d])
            else:
                torch_perm.append(d)

        node.addInput(input_val)
        node.addInput(context.torch_graph.insertConstant(torch_perm))
        node.output().setType(torch._C.TensorType.get())

        return [node]

    def _convert_slice(
        self, context: ConversionContext, op: OpType
    ) -> List[torch._C.Node]:
        sorted_inds = sorted(op.spec.slice.keys())

        input_val = context.name_to_output_value[op.input.input]

        nodes = []

        last_inp = input_val
        for i in sorted_inds:
            curr_val = op.spec.slice[i]

            node = context.torch_graph.create("aten::slice")

            node.addInput(last_inp)
            node.addInput(context.torch_graph.insertConstant(i))
            node.addInput(context.torch_graph.insertConstant(curr_val[0]))
            # from inclusive to exclusive end
            if curr_val[1] == -1:  # if to end of list replace with 2^63 - 1
                node.addInput(
                    context.torch_graph.insertConstant(9223372036854775807)
                )
            else:
                node.addInput(context.torch_graph.insertConstant(curr_val[1] + 1))
            node.addInput(context.torch_graph.insertConstant(1))

            node.output().setType(torch._C.TensorType.get())

            nodes.append(node)
            last_inp = node.output()

        return nodes

    def _convert_select(
        self, context: ConversionContext, op: OpType
    ) -> List[torch._C.Node]:
        sorted_inds = sorted(op.spec.select.keys())

        input_val = context.name_to_output_value[op.input.input]

        nodes = []

        last_inp = input_val
        for i in sorted_inds:
            index = op.spec.select[i]
            select_node = context.torch_graph.create("aten::select")

            select_node.addInput(last_inp)
            select_node.addInput(context.torch_graph.insertConstant(i))
            select_node.addInput(context.torch_graph.insertConstant(index))
            select_node.output().setType(torch._C.TensorType.get())

            nodes.append(select_node)

            unsqueeze_node = context.torch_graph.create("aten::unsqueeze")
            unsqueeze_node.addInput(select_node.output())
            unsqueeze_node.addInput(context.torch_graph.insertConstant(i))
            unsqueeze_node.output().setType(torch._C.TensorType.get())

            nodes.append(unsqueeze_node)

            last_inp = unsqueeze_node.output()

        return nodes
    
    def _convert_group(
        self, context: ConversionContext, op: OpType
    ) -> List[torch._C.Node]:
        input_val = context.name_to_output_value[op.input.input]
        node = context.torch_graph.create("aten::reshape")
        node.addInput(input_val)
        node.addInput(context.torch_graph.insertConstant(op.spec._output_shape_sidecar))
        return [node]

    def _convert_pad(
        self, context: ConversionContext, op: OpType
    ) -> List[torch._C.Node]:
        input_val = context.name_to_output_value[op.input.input]
        node = context.torch_graph.create("aten::pad")
        node.addInput(input_val)

        # padding is stored in reverse last dim -> first dim
        pad_arr = []
        for i in range(
            op.spec._output_dims_sidecar - 1, min(op.spec.pad.keys()) - 1, -1
        ):
            if i in op.spec.pad:
                pad_arr.extend(list(op.spec.pad[i]))
            else:
                pad_arr.extend([0, 0])

        if isinstance(op.spec.pad_mode, PadSpec.PadMode):
            # non-constant padding only works with 2-5D tensors
            if len(pad_arr) < 4:
                pad_arr += [0, 0]
            node.addInput(context.torch_graph.insertConstant(pad_arr))
            node.addInput(context.torch_graph.insertConstant(op.spec.pad_mode.to_str()))
            node.addInput(context.torch_graph.insertConstant(None))
        else:
            node.addInput(context.torch_graph.insertConstant(pad_arr))
            node.addInput(context.torch_graph.insertConstant("constant"))
            node.addInput(context.torch_graph.insertConstant(float(op.spec.pad_mode)))

        return [node]

    def _convert_fold(
        self, context: ConversionContext, op: OpType
    ) -> List[torch._C.Node]:
        input_val = context.name_to_output_value[op.input.input]

        # fold and unfold can only really work on 3D or 4D tensors in torch
        match op.spec.type:
            case "fold":
                node = context.torch_graph.create("aten::col2im")
                node.addInput(input_val)
                node.addInput(
                    context.torch_graph.insertConstant(
                        op.spec._output_shape_sidecar[-2:]
                    )
                )
                fold_dict = op.spec.fold
            case "unfold":
                node = context.torch_graph.create("aten::im2col")
                node.addInput(input_val)
                fold_dict = op.spec.unfold
            case _:
                raise Exception(f"Fold type Unknown: {op.spec.type}")

        key_0 = 0 + len(op.spec._output_shape_sidecar) - 2
        key_1 = 1 + len(op.spec._output_shape_sidecar) - 2
        kernel_h, stride_h = (
            fold_dict[key_0] if key_0 in fold_dict else (0, 0)
        )
        kernel_w, stride_w = (
            fold_dict[key_1] if key_1 in fold_dict else (0, 0)
        )
        kernel = [kernel_h, kernel_w]
        stride = [stride_h, stride_w]
        dilation = [1, 1]

        node.addInput(context.torch_graph.insertConstant(kernel))
        node.addInput(context.torch_graph.insertConstant(dilation))
        node.addInput(context.torch_graph.insertConstant([0, 0]))
        node.addInput(context.torch_graph.insertConstant(stride))

        return [node]

    def _convert_squeeze(
        self, context: ConversionContext, op: OpType
    ) -> List[torch._C.Node]:
        input_val = context.name_to_output_value[op.input.input]

        output = []
        match op.spec.type:
            case "squeeze":
                node = context.torch_graph.create("aten::squeeze")
                node.addInput(input_val)
                node.addInput(
                    context.torch_graph.insertConstant(list(op.spec.dimensions))
                )
                output.append(node)
            case "unsqueeze":
                last_node = input_val
                for d in sorted([i for i in op.spec.dimensions if i >= 0]):
                    node = context.torch_graph.create("aten::unsqueeze")
                    node.addInput(last_node)
                    node.addInput(context.torch_graph.insertConstant(d))
                    output.append(node)
                    last_node = node.output()
                for d in sorted([i for i in op.spec.dimensions if i < 0]):
                    node = context.torch_graph.create("aten::unsqueeze")
                    node.addInput(last_node)
                    node.addInput(context.torch_graph.insertConstant(d))
                    output.append(node)
                    last_node = node.output()
            case _:
                raise Exception(f"Squeeze type Unknown: {op.spec.type}")
        return output

    def _convert_repeat(
        self, context: ConversionContext, op: OpType
    ) -> List[torch._C.Node]:
        input_val = context.name_to_output_value[op.input.input]
    
        node = context.torch_graph.create("aten::repeat")

        repeat_list = []
        for d in range(op.spec._output_dims_sidecar):
            if d in op.spec.repeat:
                repeat_list.append(op.spec.repeat[d])
            else:
                repeat_list.append(1)

        node.addInput(input_val)
        node.addInput(
            context.torch_graph.insertConstant(repeat_list)
        )

        return [node]