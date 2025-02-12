import unittest
import math

import torch
import torch.nn as nn

from src.ir.safe_ir import PermuteSpec, TensorSpec, DataType


class TestPermuteSpec(unittest.TestCase):
    def test_basic(self):
        # doesn't raise
        # nhwc -> nchw
        PermuteSpec(
            permutation={
                1: 2,
                2: 3,
                3: 1
            }
        )

    def test_raises_on_incomplete_cycle(self):
        with self.assertRaises(Exception):
            # nhwc -> nchw
            PermuteSpec(
                permutation={
                    1: 2,
                    2: 3,
                    3: 0
                }
            )

    def test_raises_on_overlapping_values(self):
        with self.assertRaises(Exception):
            # nhwc -> nchw
            PermuteSpec(
                permutation={
                    1: 2,
                    2: 3,
                    3: 2
                }
            )


class TestPermuteOutputSpec(unittest.TestCase):
    def test_basic(self):
        # nhwc -> nchw
        spec = PermuteSpec(
            permutation={
                1: 2,
                2: 3,
                3: 1
            }
        )

        input_specs = [
            TensorSpec(shape=(1, 2, 3, 4), data_type=DataType.FLOAT32)
        ]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [1, 4, 2, 3])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_negitive_dimensions(self):
        spec = PermuteSpec(
            permutation={
                1: -2,
                -2: -1,
                -1: 1
            }
        )

        input_specs = [
            TensorSpec(shape=(1, 2, 3, 4), data_type=DataType.FLOAT32)
        ]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [1, 4, 2, 3])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_raises_on_negitive_overlap(self):
        spec = PermuteSpec(
            permutation={
                2: -2,
                -2: -1,
                -1: 2
            }
        )

        input_specs = [
            TensorSpec(shape=(1, 2, 3, 4), data_type=DataType.FLOAT32)
        ]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_raises_on_insufficient_shape(self):
        spec = PermuteSpec(
            permutation={
                1: 2,
                2: 3,
                3: 1
            }
        )

        input_specs = [
            TensorSpec(shape=(1, 2, 3), data_type=DataType.FLOAT32)
        ]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)


class TestPermuteComputeStats(unittest.TestCase):
    def test_basic(self):
        spec = PermuteSpec(
            permutation={
                1: 2,
                2: 3,
                3: 1
            }
        )

        input_specs = [
            TensorSpec(shape=(1, 2, 3, 4), data_type=DataType.FLOAT32)
        ]

        out_spec = spec.compute_stats(input_specs)

        self.assertEqual(out_spec.flops, 0)
        self.assertEqual(out_spec.reads, math.prod(input_specs[0].shape))
        self.assertEqual(out_spec.writes, math.prod(input_specs[0].shape))
