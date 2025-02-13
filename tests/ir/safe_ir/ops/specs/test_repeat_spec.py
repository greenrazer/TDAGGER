import math
import unittest

import torch
import torch.nn as nn

from src.ir.safe_ir import DataType, RepeatSpec, TensorSpec


class TestRepeatOutputSpec(unittest.TestCase):
    def test_basic(self):
        spec = RepeatSpec(
            repeat={1: 3, 2: 4},
            _output_dims_sidecar=0,  # shouldn't need this
        )

        input_specs = [TensorSpec(shape=(5, 1, 1, 4), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [5, 3, 4, 4])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_negitive_dimensions(self):
        spec = RepeatSpec(
            repeat={1: 3, -2: 4},
            _output_dims_sidecar=0,  # shouldn't need this
        )

        input_specs = [TensorSpec(shape=(5, 1, 1, 4), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [5, 3, 4, 4])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_raises_on_overlapping_dimensions(self):
        spec = RepeatSpec(
            repeat={3: 3, -1: 4},
            _output_dims_sidecar=0,  # shouldn't need this
        )

        input_specs = [TensorSpec(shape=(1, 1, 1, 1), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_raises_on_insufficient_shape(self):
        spec = RepeatSpec(
            repeat={1: 3, 4: 4},
            _output_dims_sidecar=0,  # shouldn't need this
        )

        input_specs = [TensorSpec(shape=(5, 1, 1, 4), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_raises_on_size_greater_than_one(self):
        spec = RepeatSpec(
            repeat={0: 3, 1: 4},
            _output_dims_sidecar=0,  # shouldn't need this
        )

        input_specs = [TensorSpec(shape=(5, 1, 1, 4), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)


class TestRepeatComputeStats(unittest.TestCase):
    def test_basic(self):
        spec = RepeatSpec(
            repeat={1: 3, 2: 4},
            _output_dims_sidecar=0,  # shouldn't need this
        )

        input_specs = [TensorSpec(shape=(5, 1, 1, 4), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)
        out_stats = spec.compute_stats(input_specs)

        self.assertEqual(out_stats.flops, 0)
        self.assertEqual(out_stats.reads, input_specs[0].size())
        self.assertEqual(out_stats.writes, out_spec.size())
