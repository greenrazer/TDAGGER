import unittest
import math

import torch
import torch.nn as nn

from src.ir.safe_ir import PadSpec, TensorSpec, DataType

class TestPadOutputSpec(unittest.TestCase):
    def test_basic(self):
        spec = PadSpec(
            pad={
                0: (2,2),
                2: (3,4),
                3: (0,5),
                5: (1,0)
            },
            pad_mode=0,
            _output_dims_sidecar=0 # shouldn't need this
        )

        input_specs = [
            TensorSpec(
                shape=(1,2,3,4,5,6,7,8),
                data_type=DataType.FLOAT32
            )
        ]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [5,2,10,9,5,7,7,8])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_negitive_dimensions(self):
        spec = PadSpec(
            pad={
                0: (2,2),
                2: (3,4),
                -2: (0,5),
                -1: (1,0)
            },
            pad_mode=0,
            _output_dims_sidecar=0 # shouldn't need this
        )

        input_specs = [
            TensorSpec(
                shape=(1,2,3,4,5,6,7,8),
                data_type=DataType.FLOAT32
            )
        ]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [5,2,10,4,5,6,12,9])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_raises_on_negitive_overlap(self):
        spec = PadSpec(
            pad={
                2: (3,4),
                -2: (1,0)
            },
            pad_mode=0,
            _output_dims_sidecar=0 # shouldn't need this
        )

        input_specs = [
            TensorSpec(
                shape=(1,2,3,4),
                data_type=DataType.FLOAT32
            )
        ]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_raises_on_insufficient_shape(self):
        spec = PadSpec(
            pad={
                3: (3,4),
                4: (1,0)
            },
            pad_mode=0,
            _output_dims_sidecar=0 # shouldn't need this
        )

        input_specs = [
            TensorSpec(
                shape=(1,2,3,4),
                data_type=DataType.FLOAT32
            )
        ]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

class TestPadComputeStats(unittest.TestCase):
    def test_basic(self):
        spec = PadSpec(
            pad={
                0: (2,2),
                2: (3,4),
                3: (0,5),
                5: (1,0)
            },
            pad_mode=0,
            _output_dims_sidecar=0 # shouldn't need this
        )

        input_specs = [
            TensorSpec(
                shape=(1,2,3,4,5,6,7,8),
                data_type=DataType.FLOAT32
            )
        ]

        out_spec = spec.output_spec(input_specs)
        out_stats = spec.compute_stats(input_specs)

        self.assertEqual(out_stats.flops, 0)
        self.assertEqual(out_stats.reads, math.prod(input_specs[0].shape))
        self.assertEqual(out_stats.writes, math.prod(out_spec.shape))

    def test_pad_mode(self):
        spec = PadSpec(
            pad={
                0: (2,2),
                2: (3,4),
                3: (0,5),
                5: (1,0)
            },
            pad_mode=PadSpec.PadMode.CIRCULAR,
            _output_dims_sidecar=0 # shouldn't need this
        )

        input_specs = [
            TensorSpec(
                shape=(1,2,3,4,5,6,7,8),
                data_type=DataType.FLOAT32
            )
        ]

        out_spec = spec.output_spec(input_specs)
        out_stats = spec.compute_stats(input_specs)

        self.assertEqual(out_stats.flops, 0)
        self.assertEqual(out_stats.reads, math.prod(input_specs[0].shape) + (4*7*5*1))
        self.assertEqual(out_stats.writes, math.prod(out_spec.shape))