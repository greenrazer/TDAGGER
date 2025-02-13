import math
import unittest

import torch
import torch.nn as nn

from src.ir.safe_ir import DataType, TensorSpec, UnsqueezeSpec


class TestUnsqueezeOutputSpec(unittest.TestCase):
    def test_basic(self):
        spec = UnsqueezeSpec(
            dimensions={1, 2},
        )

        input_specs = [TensorSpec(shape=(6, 8, 4, 2), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [6, 1, 1, 8, 4, 2])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_basic2(self):
        spec = UnsqueezeSpec(
            dimensions={1, 3},
        )

        input_specs = [TensorSpec(shape=(6, 8, 4, 2), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [6, 1, 8, 1, 4, 2])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_negitive_dimensions(self):
        spec = UnsqueezeSpec(
            dimensions={1, -2},
        )

        input_specs = [TensorSpec(shape=(6, 8, 4, 2), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [6, 1, 8, 4, 1, 2])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_raises_out_of_bounds(self):
        spec = UnsqueezeSpec(
            dimensions={1, 10},
        )

        input_specs = [TensorSpec(shape=(6, 8, 4, 2), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_raises_out_of_bounds_before(self):
        spec = UnsqueezeSpec(
            dimensions={1, -10},
        )

        input_specs = [TensorSpec(shape=(6, 8, 4, 2), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_raises_negitive_overlap(self):
        spec = UnsqueezeSpec(
            dimensions={1, -5},
        )

        input_specs = [TensorSpec(shape=(6, 8, 4, 2), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)


class TestUnsqueezeComputeStats(unittest.TestCase):
    def test_basic(self):
        spec = UnsqueezeSpec(
            dimensions={1, 2},
        )

        input_specs = [TensorSpec(shape=(6, 8, 4, 2), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)
        out_stats = spec.compute_stats(input_specs)

        self.assertEqual(out_stats.flops, 0)
        self.assertEqual(out_stats.reads, out_spec.size())
        self.assertEqual(out_stats.writes, out_spec.size())
