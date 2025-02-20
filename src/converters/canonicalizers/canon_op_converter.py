from typing import Callable, Dict, Generic, List, Tuple, TypeVar, Union

from ...ir.safe_ir import (
    BinaryElementwiseSpec,
    BinaryTensorInput,
    DataType,
    OpType,
    ScalarSpec,
    ScalarType,
    TensorType,
    UnaryElementwiseSpec,
    UnaryTensorInput,
    RepeatSpec,
    UnsqueezeSpec,
)
from ..op_converter import OpConverter

ContextT = TypeVar("ContextT")
ConverterT = TypeVar("ConverterT")
InputT = TypeVar("InputT")  # Type of the primary input (torch_op or op)
OutputT = TypeVar("OutputT")  # Type of the conversion output


class CanonOpConverter(
    OpConverter[ContextT, ConverterT, InputT, OutputT],
    Generic[ContextT, ConverterT, InputT, OutputT],
):
    def _create_broadcast(
        self,
        name: str,
        input_a: str,
        input_a_shape: List[int],
        input_b: str,
        input_b_shape: List[int],
        debug_sources=[],
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]], List[str]]:
        s0 = list(reversed(input_a_shape))
        s1 = list(reversed(input_b_shape))

        max_len = max(len(s0), len(s1))

        unsqueeze_left_amount = abs(len(s0) - len(s1))
        unsqueeze_idx, unsqueeze_shape_reversed = (
            (1, s1) if len(s0) == max_len else (0, s0)
        )

        unsqueeze_shape_reversed.extend([1] * (unsqueeze_left_amount))

        # Calculate resulting shape
        repeats_0 = {}
        repeats_1 = {}

        s0 = list(reversed(s0))
        s1 = list(reversed(s1))

        # Check compatibility of each dimension
        for dim, (dim0, dim1) in enumerate(zip(s0, s1)):
            if dim0 == dim1:
                pass
            elif dim0 == 1:
                repeats_0[dim] = dim1
            elif dim1 == 1:
                repeats_1[dim] = dim0
            else:
                raise Exception(
                    f"two inputs cannot be broadcast: {input_a_shape} {input_b_shape}."
                )

        in_names = [input_a, input_b]
        out_ops = []
        if unsqueeze_left_amount > 0:
            unsqueeze_op = OpType(
                name=f"{name}_unsqueeze",
                input=UnaryTensorInput(in_names[unsqueeze_idx]),
                spec=UnsqueezeSpec(
                    dimensions={i for i in range(unsqueeze_left_amount)}
                ),
            )
            in_names[unsqueeze_idx] = unsqueeze_op.name
            out_ops.append(unsqueeze_op)

        if len(repeats_0) > 0:
            repeat_0_op = OpType(
                name=f"{name}_repeat_a",
                input=UnaryTensorInput(in_names[0]),
                spec=RepeatSpec(repeat=repeats_0),
                debug_sources=debug_sources,
            )
            in_names[0] = repeat_0_op.name
            out_ops.append(repeat_0_op)

        if len(repeats_1) > 0:
            repeat_1_op = OpType(
                name=f"{name}_repeat_b",
                input=UnaryTensorInput(in_names[1]),
                spec=RepeatSpec(repeat=repeats_1),
                debug_sources=debug_sources,
            )
            in_names[1] = repeat_1_op.name
            out_ops.append(repeat_1_op)

        return out_ops, {}, in_names

    def _create_subtract(
        self,
        name: str,
        input_a: str,
        input_b: str,
        input_a_shape: List[int] = [],
        input_b_shape: List[int] = [],
        debug_sources=[],
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]]]:
        neg_b = OpType(
            name=f"{name}_neg",
            input=UnaryTensorInput(input_b),
            spec=UnaryElementwiseSpec(
                UnaryElementwiseSpec.UnaryElementwiseType.NEGATIVE
            ),
            debug_sources=debug_sources,
        )

        if len(input_a_shape) == 0 or len(input_b_shape) == 0:
            op = OpType(
                name=f"{name}",
                input=BinaryTensorInput(input_a, input_b),
                spec=BinaryElementwiseSpec(
                    BinaryElementwiseSpec.BinaryElementwiseType.ADD
                ),
                debug_sources=debug_sources,
            )

            return [op], {}

        out_ops, _brodcast_consts, in_names = self._create_broadcast(
            f"{name}_broadcast",
            input_a,
            input_a_shape,
            neg_b.name,
            input_b_shape,
            debug_sources=debug_sources,
        )

        op = OpType(
            name=f"{name}",
            input=BinaryTensorInput(in_names[0], in_names[1]),
            spec=BinaryElementwiseSpec(BinaryElementwiseSpec.BinaryElementwiseType.ADD),
            debug_sources=debug_sources,
        )

        return [neg_b] + out_ops + [op], {}

    def _create_divide(
        self,
        name: str,
        input_a: str,
        input_b: str,
        input_a_shape: List[int] = [],
        input_b_shape: List[int] = [],
        debug_sources=[],
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]]]:
        reciprocal_b = OpType(
            name=f"{name}_reciprocal",
            input=UnaryTensorInput(input_b),
            spec=UnaryElementwiseSpec(
                UnaryElementwiseSpec.UnaryElementwiseType.RECIPROCAL
            ),
            debug_sources=debug_sources,
        )

        if len(input_a_shape) == 0 or len(input_b_shape) == 0:
            op = OpType(
                name=f"{name}",
                input=BinaryTensorInput(input_a, input_b),
                spec=BinaryElementwiseSpec(
                    BinaryElementwiseSpec.BinaryElementwiseType.MULTIPLY
                ),
                debug_sources=debug_sources,
            )

            return [op], {}

        out_ops, _brodcast_consts, in_names = self._create_broadcast(
            f"{name}_broadcast",
            input_a,
            input_a_shape,
            reciprocal_b.name,
            input_b_shape,
            debug_sources=debug_sources,
        )

        op = OpType(
            name=f"{name}",
            input=BinaryTensorInput(in_names[0], in_names[1]),
            spec=BinaryElementwiseSpec(
                BinaryElementwiseSpec.BinaryElementwiseType.MULTIPLY
            ),
            debug_sources=debug_sources,
        )

        return [reciprocal_b] + out_ops + [op], {}

    def _create_elementwise_abs(
        self, name: str, input: str, debug_sources=[]
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]]]:
        sign_op = OpType(
            name=f"{name}_sign",
            input=UnaryTensorInput(input),
            spec=UnaryElementwiseSpec(UnaryElementwiseSpec.UnaryElementwiseType.SIGN),
            debug_sources=debug_sources,
        )

        op = OpType(
            name=f"{name}",
            input=BinaryTensorInput(input, sign_op.name),
            spec=BinaryElementwiseSpec(
                BinaryElementwiseSpec.BinaryElementwiseType.MULTIPLY
            ),
            debug_sources=debug_sources,
        )

        return [sign_op, op], {}

    def _create_add_one(
        self, name: str, input: str, debug_sources=[]
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]]]:
        one_name = f"{name}_constant_one"
        one = ScalarType(ScalarSpec(DataType.FLOAT32), data=1)

        op = OpType(
            name=f"{name}",
            input=BinaryTensorInput(input, one_name),
            spec=BinaryElementwiseSpec(BinaryElementwiseSpec.BinaryElementwiseType.ADD),
            debug_sources=debug_sources,
        )

        return [op], {one_name: one}

    def _create_div_two(
        self, name: str, input: str, debug_sources=[]
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]]]:
        half_name = f"{name}_constant_half"
        half = ScalarType(ScalarSpec(DataType.FLOAT32), data=0.5)

        div_by_2 = OpType(
            name=f"{name}",
            input=BinaryTensorInput(input, half_name),
            spec=BinaryElementwiseSpec(
                BinaryElementwiseSpec.BinaryElementwiseType.MULTIPLY
            ),
            debug_sources=debug_sources,
        )

        return [div_by_2], {half_name: half}

    def _create_elementwise_min_max(
        self, name: str, input_a: str, input_b: str, debug_sources=[], min=True
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]]]:
        add_op = OpType(
            name=f"{name}_add",
            input=BinaryTensorInput(input_a, input_b),
            spec=BinaryElementwiseSpec(BinaryElementwiseSpec.BinaryElementwiseType.ADD),
            debug_sources=debug_sources,
        )
        sub_ops, sub_consts = self._create_subtract(
            name=f"{name}_sub",
            input_a=input_a,
            input_b=input_b,
            debug_sources=debug_sources,
        )

        abs_ops, abs_consts = self._create_elementwise_abs(
            f"{name}_abs",
            sub_ops[-1].name,
            debug_sources=debug_sources,
        )

        if min:
            abs_ops.append(
                OpType(
                    name=f"{name}_pre_sub_neg_2",
                    input=UnaryTensorInput(abs_ops[-1].name),
                    spec=UnaryElementwiseSpec(
                        UnaryElementwiseSpec.UnaryElementwiseType.NEGATIVE
                    ),
                    debug_sources=debug_sources,
                )
            )

        middle_op = OpType(
            name=f"{name}_middle_op",
            input=BinaryTensorInput(add_op.name, abs_ops[-1].name),
            spec=BinaryElementwiseSpec(BinaryElementwiseSpec.BinaryElementwiseType.ADD),
            debug_sources=debug_sources,
        )

        div_by_two_ops, div_by_2_consts = self._create_div_two(
            name,
            middle_op.name,
            debug_sources=debug_sources,
        )

        return [add_op] + sub_ops + abs_ops + [
            middle_op
        ] + div_by_two_ops, sub_consts | abs_consts | div_by_2_consts

    def _create_elementwise_max(
        self, name: str, input_a: str, input_b: str, debug_sources=[]
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]]]:
        return self._create_elementwise_min_max(
            name, input_a, input_b, debug_sources=debug_sources, min=False
        )

    def _create_elementwise_min(
        self, name: str, input_a: str, input_b: str, debug_sources=[]
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]]]:
        return self._create_elementwise_min_max(
            name, input_a, input_b, debug_sources=debug_sources, min=True
        )

    def _create_relu(
        self, name: str, input: str, debug_sources=[]
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]]]:
        abs_ops, abs_consts = self._create_elementwise_abs(
            f"{name}_abs",
            input,
            debug_sources=debug_sources,
        )

        middle_op = OpType(
            name=f"{name}_middle_op",
            input=BinaryTensorInput(input, abs_ops[-1].name),
            spec=BinaryElementwiseSpec(BinaryElementwiseSpec.BinaryElementwiseType.ADD),
            debug_sources=debug_sources,
        )

        div_by_two_ops, div_by_2_consts = self._create_div_two(
            name,
            middle_op.name,
            debug_sources=debug_sources,
        )

        return abs_ops + [middle_op] + div_by_two_ops, abs_consts | div_by_2_consts

    def _create_leaky_relu(
        self, name: str, input: str, negative_slope: str, debug_sources=[]
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]]]:
        # (1 + negative_slope)/2 * x + (1 -  negative_slope)/2 * torch.abs(x)`

        # (negative_slope + 1)
        add_one_ops, add_one_consts = self._create_add_one(
            f"{name}_add_one",
            negative_slope,
            debug_sources=debug_sources,
        )

        # (negative_slope + 1)/2
        x_coeff_ops, x_coeff_consts = self._create_div_two(
            f"{name}_x_coeff",
            add_one_ops[-1].name,
            debug_sources=debug_sources,
        )

        # x_part = (1 + negative_slope)/2 * x
        x_part = OpType(
            name=f"{name}_x_part",
            input=BinaryTensorInput(x_coeff_ops[-1].name, input),
            spec=BinaryElementwiseSpec(
                BinaryElementwiseSpec.BinaryElementwiseType.MULTIPLY
            ),
            debug_sources=debug_sources,
        )

        # -negitive_slope
        positive_slope = OpType(
            name=f"{name}_positive_slope",
            input=UnaryTensorInput(negative_slope),
            spec=UnaryElementwiseSpec(
                UnaryElementwiseSpec.UnaryElementwiseType.NEGATIVE
            ),
            debug_sources=debug_sources,
        )

        # -negitive_slope + 1
        sub_one_ops, sub_one_consts = self._create_add_one(
            f"{name}_sub_one",
            positive_slope.name,
            debug_sources=debug_sources,
        )

        # (-negitive_slope + 1)/2
        x_abs_coeff_ops, x_abs_coeff_consts = self._create_div_two(
            f"{name}_x_abs_coeff",
            sub_one_ops[-1].name,
            debug_sources=debug_sources,
        )

        # abs(x)
        abs_ops, abs_consts = self._create_elementwise_abs(
            f"{name}_x_abs",
            input,
            debug_sources=debug_sources,
        )

        # x_abs_part = (-negative_slope + 1)/2 * abs(x)
        x_abs_part = OpType(
            name=f"{name}_x_abs_part",
            input=BinaryTensorInput(x_abs_coeff_ops[-1].name, abs_ops[-1].name),
            spec=BinaryElementwiseSpec(
                BinaryElementwiseSpec.BinaryElementwiseType.MULTIPLY
            ),
            debug_sources=debug_sources,
        )

        # x_part + x_abs_part
        add_op = OpType(
            name=f"{name}",
            input=BinaryTensorInput(x_part.name, x_abs_part.name),
            spec=BinaryElementwiseSpec(BinaryElementwiseSpec.BinaryElementwiseType.ADD),
            debug_sources=debug_sources,
        )

        return (
            add_one_ops
            + x_coeff_ops
            + [x_part]
            + [positive_slope]
            + sub_one_ops
            + x_abs_coeff_ops
            + abs_ops
            + [x_abs_part, add_op],
            add_one_consts
            | x_coeff_consts
            | sub_one_consts
            | x_abs_coeff_consts
            | abs_consts,
        )

    def _create_softplus(
        self, name: str, input: str, beta: str, debug_sources=[]
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]]]:
        # log(1 + exp(beta * x))/beta
        # n0 = beta * x
        # n1 = exp(n0)
        # n2 = 1 + n1
        # n3 = log(n2)
        # n4 = n3 / beta

        mul_op = OpType(
            name=f"{name}_mul",
            input=BinaryTensorInput(input, beta),
            spec=BinaryElementwiseSpec(
                BinaryElementwiseSpec.BinaryElementwiseType.MULTIPLY
            ),
            debug_sources=debug_sources,
        )

        exp_op = OpType(
            name=f"{name}_exp",
            input=UnaryTensorInput(mul_op.name),
            spec=UnaryElementwiseSpec(UnaryElementwiseSpec.UnaryElementwiseType.EXP),
            debug_sources=debug_sources,
        )

        add_one_ops, add_one_consts = self._create_add_one(
            f"{name}_add_one",
            exp_op.name,
            debug_sources=debug_sources,
        )

        log_op = OpType(
            name=f"{name}_log",
            input=UnaryTensorInput(add_one_ops[-1].name),
            spec=UnaryElementwiseSpec(UnaryElementwiseSpec.UnaryElementwiseType.LOG),
            debug_sources=debug_sources,
        )

        div_ops, div_consts = self._create_divide(
            name,
            input_a=log_op.name,
            input_b=beta,
            debug_sources=debug_sources,
        )

        return (
            [mul_op, exp_op] + add_one_ops + [log_op] + div_ops,
            add_one_consts | div_consts,
        )
