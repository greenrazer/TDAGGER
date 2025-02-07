from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Union

from .op_type import OpType


def product_of_noninferred_dimensions(iter) -> int:
    out = 1
    for a in iter:
        if a != -1:
            out *= a
    return out


def prefix_products(shape):
    result = [1]
    for dim in shape:
        result.append(result[-1] * dim)
    return result


def prime_factors(n: int):
    factors = []
    i = 2
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n = n // i
        i += 1
    if n > 1:
        factors.append(n)
    return factors


def flatten_unpacked_shape(arr):
    out = []
    for g in arr:
        if isinstance(g, list):
            out.extend([i for i in g])
        else:
            out.append(g)
    return out


def random_permutation_unpacked_shape(groups):
    import random

    result = []
    for group in groups:
        if isinstance(group, list):
            group_list = list(group)
            random.shuffle(group_list)
            result.append(group_list)
        else:
            result.append(group)
    return result


@dataclass
class GroupSpec:
    groups: List[List[int]]

    # TODO: remove and propagate shape through network
    _output_shape_sidecar: List[int]

    def __post_init__(self):
        flattened = []

        for sublist in self.groups:
            if not sublist:
                continue
            flattened.extend(sublist)
            is_positive = all(x >= 0 for x in sublist)
            is_negative = all(x < 0 for x in sublist)
            if not (is_positive or is_negative):
                raise Exception(
                    f"Group dimensions must be entirely positive or entirely negitive: {sublist}."
                )

        if len(flattened) != len(set(flattened)):
            raise Exception("Group dimensions must be unique.")

    def __str__(self):
        sorted_groups = [sorted(sublist) for sublist in self.groups]
        positive_groups = sorted(
            [lst for lst in sorted_groups if lst and lst[0] >= 0], key=lambda x: x[0]
        )
        negative_groups = sorted(
            [lst for lst in sorted_groups if lst and lst[0] < 0], key=lambda x: x[0]
        )

        out = []
        last_dim = -1
        for group in positive_groups:
            if group[0] != last_dim + 1:
                out.append("...")
            out.append(f"({' '.join(group)})")
            last_dim = group[-1]
        out.append("...")
        if len(negative_groups) > 0:
            last_dim = negative_groups[0][0] - 1
            for group in negative_groups:
                if group[0] != last_dim + 1:
                    out.append("...")
                out.append(f"({' '.join(group)})")
                last_dim = group[-1]
            if last_dim != -1:
                out.append("...")

        return " ".join(out)

    @property
    def type(self):
        return "group"


@dataclass
class UngroupSpec:
    ungroups: Dict[int, List[int]]

    # TODO: remove and propagate shape through network
    _output_shape_sidecar: List[int]

    @property
    def type(self):
        return "ungroup"

    def __str__(self):
        out = []
        last_dim = -1
        for d in sorted([d for d in self.ungroups.keys() if d >= 0]):
            if d != last_dim + 1:
                out.append("...")
            ungroup = [str(i) for i in self.ungroups[d]]
            ungroup[0] = "-1"
            out.append(f"{d}({' '.join(ungroup)})")
            last_dim = d
        out.append("...")
        neg_dims = sorted([d for d in self.ungroups.keys() if d < 0])
        if len(neg_dims) > 0:
            last_dim = neg_dims[0] - 1
            for d in neg_dims:
                if d != last_dim + 1:
                    out.append("...")
                ungroup = [str(i) for i in self.ungroups[d]]
                ungroup[0] = "-1"
                out.append(f"{d}({' '.join(ungroup)})")
                last_dim = d
            if last_dim != -1:
                out.append("...")

        return f"{' '.join(out)}"


class GroupType(OpType):
    spec: Union[GroupSpec, UngroupSpec]

    def __init__(
        self, name, inputs, spec: Union[GroupSpec, UngroupSpec], debug_sources=[]
    ):
        super().__init__(name, inputs, debug_sources=debug_sources)
        self.spec = spec

    def __str__(self) -> str:
        inp_name = self.inputs["input"]
        out = f"%{self.name}: {self.type}[{self.spec}](%{inp_name}){self.debug_sources_to_str()}"
        return out

    @property
    def type(self) -> str:
        return self.spec.type

    @property
    def required_input_keys(self) -> List[str]:
        return ["input"]

    @staticmethod
    def specs_from_reshape(
        input_shape: List[int], output_shape: List[int]
    ) -> List[Union[GroupSpec, UngroupSpec]]:
        # Handle dynamic shapes with -1
        input_total_elems = product_of_noninferred_dimensions(input_shape)
        output_total_elems = product_of_noninferred_dimensions(output_shape)
        if input_total_elems != output_total_elems:
            try:
                missing_dim_index = output_shape.index(-1)
                output_shape[missing_dim_index] = (
                    input_total_elems // output_total_elems
                )
            except ValueError:
                raise Exception(f"Cannot reshape: {input_shape} into {output_shape}")

        # Calculate prefix products
        input_prefix_prods = prefix_products(input_shape)
        output_prefix_prods = prefix_products(output_shape)
        input_prefix_prods_set = set(input_prefix_prods)
        output_prefix_prods_set = set(output_prefix_prods)

        # Case 1: Output is grouping of input dimensions
        if output_prefix_prods_set.issubset(input_prefix_prods_set):
            mapping = []

            input_idx = 0
            curr_prod = input_prefix_prods[input_idx]
            for out_idx, out_size in enumerate(output_prefix_prods[:]):
                group = []
                while curr_prod != out_size:
                    group.append(input_idx)
                    input_idx += 1
                    curr_prod = input_prefix_prods[input_idx]
                if len(group) > 1:
                    mapping.append(group)

            return [GroupSpec(groups=mapping, _output_shape_sidecar=output_shape)]

        # Case 2: Output is ungrouping of input dimensions
        elif input_prefix_prods_set.issubset(output_prefix_prods_set):
            mapping = {}
            _side_car_mapping = []

            output_idx = 0
            curr_prod = output_prefix_prods[output_idx]
            for input_idx, input_size in enumerate(input_prefix_prods[:]):
                factors = []
                while curr_prod != input_size:
                    factors.append(output_shape[output_idx])
                    _side_car_mapping.append(output_shape[output_idx])
                    output_idx += 1
                    curr_prod = output_prefix_prods[output_idx]
                if len(factors) > 1:
                    mapping[input_idx - 1] = factors

            return [
                UngroupSpec(ungroups=mapping, _output_shape_sidecar=_side_car_mapping)
            ]

        # Case 3: Complex reshaping
        else:
            max_attempts = 100000

            input_groups = [prime_factors(d) for d in input_shape]
            output_groups = [prime_factors(d) for d in output_shape]

            # Identify and replace matching dimensions' prime factors
            for in_idx, (in_size, in_prefix_prod) in enumerate(
                zip(input_shape, input_prefix_prods)
            ):
                for out_idx, (out_size, out_prefix_prod) in enumerate(
                    zip(output_shape, output_prefix_prods)
                ):
                    if in_size == out_size and in_prefix_prod == out_prefix_prod:
                        input_groups[in_idx] = in_size
                        output_groups[out_idx] = out_size

            # Since this is probably an NP complete problem, try random permutations
            for _ in range(max_attempts):
                input_perm_combo = random_permutation_unpacked_shape(input_groups)
                output_perm_combo = random_permutation_unpacked_shape(output_groups)

                if flatten_unpacked_shape(input_perm_combo) == flatten_unpacked_shape(
                    output_perm_combo
                ):
                    ungroup_map = {
                        dim: shape
                        for dim, shape in enumerate(input_perm_combo)
                        if isinstance(shape, list)
                    }

                    group_list = []
                    i = 0
                    for g in output_perm_combo:
                        if isinstance(g, list):
                            group_list.append([i + j for j in range(len(g))])
                            i += len(g)
                        else:
                            i += 1

                    return [
                        UngroupSpec(
                            ungroups=ungroup_map,
                            _output_shape_sidecar=flatten_unpacked_shape(
                                input_perm_combo
                            ),
                        ),
                        GroupSpec(
                            groups=group_list, _output_shape_sidecar=output_shape
                        ),
                    ]

            # Fallback case: use simple group-ungroup
            return [
                GroupSpec(
                    groups=[list(range(len(input_shape)))],
                    _output_shape_sidecar=[
                        product_of_noninferred_dimensions(input_shape)
                    ],
                ),
                UngroupSpec(
                    ungroups={0: output_shape}, _output_shape_sidecar=output_shape
                ),
            ]
