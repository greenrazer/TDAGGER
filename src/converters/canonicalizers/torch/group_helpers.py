from typing import Any, Callable, Dict, List, Tuple, Union

from ....ir.safe_ir import GroupSpec, UngroupSpec


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


def specs_from_reshape(
    input_shape: List[int], output_shape: List[int]
) -> List[Union[GroupSpec, UngroupSpec]]:
    # Handle dynamic shapes with -1
    input_total_elems = product_of_noninferred_dimensions(input_shape)
    output_total_elems = product_of_noninferred_dimensions(output_shape)
    if input_total_elems != output_total_elems:
        try:
            missing_dim_index = output_shape.index(-1)
            output_shape[missing_dim_index] = input_total_elems // output_total_elems
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

        return [GroupSpec(groups=mapping)]

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

        return [UngroupSpec(ungroups=mapping)]

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
                    ),
                    GroupSpec(groups=group_list),
                ]

        # Fallback case: use simple group-ungroup
        return [
            GroupSpec(groups=[list(range(len(input_shape)))]),
            UngroupSpec(ungroups={0: output_shape}),
        ]
