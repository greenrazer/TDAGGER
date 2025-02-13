import math
import unittest

import torch
import torch.nn as nn

from src.ir.safe_ir import DataType, ReduceSpec, TensorSpec


class TestReduceOutputSpec(unittest.TestCase):
    def test_basic(self):
        spec = ReduceSpec(
            dimensions={1, 2}, reduction_type=ReduceSpec.ReductionType.SUM
        )

        input_specs = [TensorSpec(shape=(5, 2, 3, 4), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [5, 1, 1, 4])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_negitive_dimensions(self):
        spec = ReduceSpec(
            dimensions={-3, -2}, reduction_type=ReduceSpec.ReductionType.SUM
        )

        input_specs = [TensorSpec(shape=(5, 2, 3, 4), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [5, 1, 1, 4])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_raises_on_overlapping_dimensions(self):
        spec = ReduceSpec(
            dimensions={3, -1}, reduction_type=ReduceSpec.ReductionType.SUM
        )

        input_specs = [TensorSpec(shape=(5, 2, 3, 4), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_raises_on_insufficient_shape(self):
        spec = ReduceSpec(
            dimensions={3, 4}, reduction_type=ReduceSpec.ReductionType.SUM
        )

        input_specs = [TensorSpec(shape=(5, 2, 3, 4), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)


class TestReduceComputeStats(unittest.TestCase):
    def test_sum(self):
        spec = ReduceSpec(
            dimensions={1, 2}, reduction_type=ReduceSpec.ReductionType.SUM
        )

        input_specs = [TensorSpec(shape=(5, 2, 3, 4), data_type=DataType.FLOAT32)]

        out_spec = spec.compute_stats(input_specs)

        self.assertEqual(out_spec.flops, 6)
        self.assertEqual(out_spec.reads, math.prod(input_specs[0].shape))
        self.assertEqual(out_spec.writes, math.prod(input_specs[0].shape))

    def test_prod(self):
        spec = ReduceSpec(
            dimensions={1, 2}, reduction_type=ReduceSpec.ReductionType.PRODUCT
        )

        input_specs = [TensorSpec(shape=(5, 2, 3, 4), data_type=DataType.FLOAT32)]

        out_spec = spec.compute_stats(input_specs)

        self.assertEqual(out_spec.flops, 6)
        self.assertEqual(out_spec.reads, math.prod(input_specs[0].shape))
        self.assertEqual(out_spec.writes, math.prod(input_specs[0].shape))

    def test_min(self):
        spec = ReduceSpec(
            dimensions={1, 2}, reduction_type=ReduceSpec.ReductionType.MINIMUM
        )

        input_specs = [TensorSpec(shape=(5, 2, 3, 4), data_type=DataType.FLOAT32)]

        out_spec = spec.compute_stats(input_specs)

        self.assertEqual(out_spec.flops, 6)
        self.assertEqual(out_spec.reads, math.prod(input_specs[0].shape))
        self.assertEqual(out_spec.writes, 21)

    def test_max(self):
        spec = ReduceSpec(
            dimensions={1, 2}, reduction_type=ReduceSpec.ReductionType.MAXIMUM
        )

        input_specs = [TensorSpec(shape=(5, 2, 3, 4), data_type=DataType.FLOAT32)]

        out_spec = spec.compute_stats(input_specs)

        self.assertEqual(out_spec.flops, 6)
        self.assertEqual(out_spec.reads, math.prod(input_specs[0].shape))
        self.assertEqual(out_spec.writes, 21)

    def test_mean(self):
        spec = ReduceSpec(
            dimensions={1, 2}, reduction_type=ReduceSpec.ReductionType.MEAN
        )

        input_specs = [TensorSpec(shape=(5, 2, 3, 4), data_type=DataType.FLOAT32)]

        out_spec = spec.compute_stats(input_specs)

        self.assertEqual(out_spec.flops, 6 + 1)
        self.assertEqual(out_spec.reads, math.prod(input_specs[0].shape))
        self.assertEqual(out_spec.writes, math.prod(input_specs[0].shape) + 1)
