from abc import ABC
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

import torch

from ..ir.safe_ir import (
    BinaryElementwiseSpec,
    BinaryElementwiseType,
    DataType,
    OpType,
    ScalarSpec,
    ScalarType,
    TensorType,
    UnaryElementwiseSpec,
    UnaryElementwiseType,
)


class OpConverter(ABC):
    def _create_add_one(
        self, name: str, input: str, debug_sources=[]
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]]]:
        one_name = f"{name}_constant_one"
        one = ScalarType(ScalarSpec(DataType.FLOAT32), data=1)

        op = BinaryElementwiseType(
            name=f"{name}",
            inputs={"input_0": input, "input_1": one_name},
            spec=BinaryElementwiseSpec.ADD,
            debug_sources=debug_sources,
        )

        return [op], {one_name: one}

    def _create_subtract_one(
        self, name: str, input: str, debug_sources=[], first=True
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]]]:
        one_name = f"{name}_constant_one"
        one = ScalarType(ScalarSpec(DataType.FLOAT32), data=1)

        op = BinaryElementwiseType(
            name=f"{name}",
            inputs={"input_0": input, "input_1": one_name}
            if first
            else {"input_0": one_name, "input_1": input},
            spec=BinaryElementwiseSpec.SUBTRACT,
            debug_sources=debug_sources,
        )

        return [op], {one_name: one}

    def _create_div_two(
        self, name: str, input: str, debug_sources=[]
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]]]:
        two_name = f"{name}_constant_two"
        two = ScalarType(ScalarSpec(DataType.FLOAT32), data=2)

        div_by_2 = BinaryElementwiseType(
            name=f"{name}",
            inputs={"input_0": input, "input_1": two_name},
            spec=BinaryElementwiseSpec.DIVIDE,
            debug_sources=debug_sources,
        )

        return [div_by_2], {two_name: two}

    def _create_elementwise_min_max(
        self, name: str, input_a: str, input_b: str, debug_sources=[], min=True
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]]]:
        add_op = BinaryElementwiseType(
            name=f"{name}_add",
            inputs={"input_0": input_a, "input_1": input_b},
            spec=BinaryElementwiseSpec.ADD,
            debug_sources=debug_sources,
        )
        sub_op = BinaryElementwiseType(
            name=f"{name}_sub",
            inputs={"input_0": input_a, "input_1": input_b},
            spec=BinaryElementwiseSpec.SUBTRACT,
            debug_sources=debug_sources,
        )
        abs_op = UnaryElementwiseType(
            name=f"{name}_abs",
            inputs={"input": sub_op.name},
            spec=UnaryElementwiseSpec.ABSOLUTE_VALUE,
            debug_sources=debug_sources,
        )

        middle_op = BinaryElementwiseType(
            name=f"{name}_middle_op",
            inputs={"input_0": add_op.name, "input_1": abs_op.name},
            spec=BinaryElementwiseSpec.SUBTRACT if min else BinaryElementwiseSpec.ADD,
            debug_sources=debug_sources,
        )

        div_by_two_ops, div_by_2_consts = self._create_div_two(
            name,
            middle_op.name,
            debug_sources=debug_sources,
        )

        return [add_op, sub_op, abs_op, middle_op] + div_by_two_ops, div_by_2_consts

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
        self, name: str, input: str, debug_sources=[], flipped=False
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]]]:
        abs_op = UnaryElementwiseType(
            name=f"{name}_abs",
            inputs={"input": input},
            spec=UnaryElementwiseSpec.ABSOLUTE_VALUE,
            debug_sources=debug_sources,
        )

        middle_op = BinaryElementwiseType(
            name=f"{name}_middle_op",
            inputs={"input_0": input, "input_1": abs_op.name},
            spec=BinaryElementwiseSpec.SUBTRACT
            if flipped
            else BinaryElementwiseSpec.ADD,
            debug_sources=debug_sources,
        )

        div_by_two_ops, div_by_2_consts = self._create_div_two(
            name,
            middle_op.name,
            debug_sources=debug_sources,
        )

        return [abs_op, middle_op] + div_by_two_ops, div_by_2_consts

    def _create_leaky_relu(
        self, name: str, input: str, negitive_slope: str, debug_sources=[]
    ) -> Tuple[List[OpType], Dict[str, Union[TensorType, ScalarType]]]:
        # (1 + negative_slope)/2 * x + (1 -  negative_slope)/2 * torch.abs(x)`
        add_one_name = f"{name}_add_one"
        add_one_ops, add_one_consts = self._create_add_one(
            add_one_name,
            negitive_slope,
            debug_sources=debug_sources,
        )

        x_coeff_name = f"{name}_x_coeff"
        x_coeff_ops, x_coeff_consts = self._create_div_two(
            x_coeff_name,
            add_one_name,
            debug_sources=debug_sources,
        )

        x_part = BinaryElementwiseType(
            name=f"{name}_x_part",
            inputs={"input_0": x_coeff_name, "input_1": input},
            spec=BinaryElementwiseSpec.MULTIPLY,
            debug_sources=debug_sources,
        )

        sub_one_name = f"{name}_sub_one"
        sub_one_ops, sub_one_consts = self._create_subtract_one(
            sub_one_name,
            negitive_slope,
            debug_sources=debug_sources,
            first=False,
        )

        x_abs_coeff_name = f"{name}_x_abs_coeff"
        x_abs_coeff_ops, x_abs_coeff_consts = self._create_div_two(
            x_abs_coeff_name,
            sub_one_name,
            debug_sources=debug_sources,
        )

        x_abs = UnaryElementwiseType(
            name=f"{name}_x_abs",
            inputs={"input": input},
            spec=UnaryElementwiseSpec.ABSOLUTE_VALUE,
            debug_sources=debug_sources,
        )

        x_abs_part = BinaryElementwiseType(
            name=f"{name}_x_abs_part",
            inputs={"input_0": x_abs_coeff_name, "input_1": x_abs.name},
            spec=BinaryElementwiseSpec.MULTIPLY,
            debug_sources=debug_sources,
        )

        add_op = BinaryElementwiseType(
            name=f"{name}",
            inputs={"input_0": x_abs_part.name, "input_1": x_part.name},
            spec=BinaryElementwiseSpec.ADD,
            debug_sources=debug_sources,
        )

        return (
            [
                x_part,
                x_abs,
                x_abs_part,
                add_op,
            ]
            + add_one_ops
            + sub_one_ops
            + x_coeff_ops
            + x_abs_coeff_ops,
            add_one_consts | x_coeff_consts | sub_one_consts | x_abs_coeff_consts,
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

        mul_op = BinaryElementwiseType(
            name=f"{name}_mul",
            inputs={"input_0": input, "input_1": beta},
            spec=BinaryElementwiseSpec.MULTIPLY,
            debug_sources=debug_sources,
        )

        exp_op = UnaryElementwiseType(
            name=f"{name}_exp",
            inputs={"input": mul_op.name},
            spec=UnaryElementwiseSpec.EXP,
            debug_sources=debug_sources,
        )

        one_name = f"{name}_constant_one"
        one = ScalarType(ScalarSpec(DataType.FLOAT32), data=1)

        add_op = BinaryElementwiseType(
            name=f"{name}_add",
            inputs={"input_0": exp_op.name, "input_1": one_name},
            spec=BinaryElementwiseSpec.ADD,
            debug_sources=debug_sources,
        )

        log_op = UnaryElementwiseType(
            name=f"{name}_log",
            inputs={"input": add_op.name},
            spec=UnaryElementwiseSpec.LOG,
            debug_sources=debug_sources,
        )

        div_op = BinaryElementwiseType(
            name=f"{name}",
            inputs={"input_0": log_op.name, "input_1": beta},
            spec=BinaryElementwiseSpec.MULTIPLY,
            debug_sources=debug_sources,
        )

        return [mul_op, exp_op, add_op, log_op, div_op], {one_name: one}
