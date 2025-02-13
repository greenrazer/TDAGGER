import math
import unittest

import torch
import torch.nn as nn

from src.ir.safe_ir import DataType, UngroupSpec, TensorSpec

class TestUngroupSpec(unittest.TestCase):
    def test_basic(self):
        # doesn't raise
        UngroupSpec(
            ungroups={
                1: [2,4],
                2: [2,2]
            },
            _output_shape_sidecar=0 # doesn't matter
        )

    def test_raises_on_multiple_missing_ungroups_per_group(self):
        with self.assertRaises(Exception):
            UngroupSpec(
                ungroups={
                    1: [-1, -1, 1],
                    2: [2,2]
                },
                _output_shape_sidecar=0 # doesn't matter
            )

class TestUngroupOutputSpec(unittest.TestCase):
    def test_basic(self):
        spec = UngroupSpec(
            ungroups={
                1: [2,4],
                2: [2,2]
            },
            _output_shape_sidecar=0 # doesn't matter
        )

        input_specs = [TensorSpec(shape=(6, 8, 4, 2), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [6, 2, 4, 2, 2, 2])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_negitive_dimensions(self):
        spec = UngroupSpec(
            ungroups={
                1: [2,4],
                -2: [2,2]
            },
            _output_shape_sidecar=0 # doesn't matter
        )

        input_specs = [TensorSpec(shape=(6, 8, 4, 2), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [6, 2, 4, 2, 2, 2])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_missing_dimensions(self):
        spec = UngroupSpec(
            ungroups={
                1: [-1,4],
                -2: [-1,2]
            },
            _output_shape_sidecar=0 # doesn't matter
        )

        input_specs = [TensorSpec(shape=(6, 8, 4, 2), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [6, 2, 4, 2, 2, 2])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_raises_on_insufficient_shape(self):
        spec = UngroupSpec(
            ungroups={
                1: [2,4],
                4: [2,2]
            },
            _output_shape_sidecar=0 # doesn't matter
        )

        input_specs = [TensorSpec(shape=(6, 8, 4, 2), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_raises_on_cannot_reshape(self):
        spec = UngroupSpec(
            ungroups={
                1: [3,4],
                4: [2,2]
            },
            _output_shape_sidecar=0 # doesn't matter
        )

        input_specs = [TensorSpec(shape=(6, 8, 4, 2), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_raises_on_cannot_reshape_missing_dimension(self):
        spec = UngroupSpec(
            ungroups={
                1: [-1,3],
                4: [2,2]
            },
            _output_shape_sidecar=0 # doesn't matter
        )

        input_specs = [TensorSpec(shape=(6, 8, 4, 2), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

class TestUngroupComputeStats(unittest.TestCase):
    def test_basic(self):
        spec = UngroupSpec(
            ungroups={
                1: [2,4],
                2: [2,2]
            },
            _output_shape_sidecar=0 # doesn't matter
        )

        input_specs = [TensorSpec(shape=(6, 8, 4, 2), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)
        out_stats = spec.compute_stats(input_specs)

        self.assertEqual(out_stats.flops, 0)
        self.assertEqual(out_stats.reads, out_spec.size())
        self.assertEqual(out_stats.writes, out_spec.size())