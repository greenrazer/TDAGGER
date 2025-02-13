import math
import unittest

import torch
import torch.nn as nn

from src.ir.safe_ir import DataType, SelectSpec, TensorSpec


class TestSelectOutputSpec(unittest.TestCase):
    def test_basic(self):
        spec = SelectSpec(
            select= {
                1: 0,
                2: 1
            }
        )

        input_specs = [TensorSpec(shape=(5, 2, 3, 4), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [5, 1, 1, 4])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_negitive_dimensions(self):
        spec = SelectSpec(
            select= {
                1: 0,
                -2: 1
            }
        )

        input_specs = [TensorSpec(shape=(5, 2, 3, 4), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [5, 1, 1, 4])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_negitive_select(self):
        spec = SelectSpec(
            select= {
                1: 0,
                -2: -1
            }
        )

        input_specs = [TensorSpec(shape=(5, 2, 3, 4), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [5, 1, 1, 4])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_raises_on_overlapping_dimensions(self):
        spec = SelectSpec(
            select= {
                2: 0,
                -2: 1
            }
        )

        input_specs = [TensorSpec(shape=(5, 2, 3, 4), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_raises_on_insufficient_shape(self):
        spec = SelectSpec(
            select= {
                2: 0,
                4: 1
            }
        )

        input_specs = [TensorSpec(shape=(5, 2, 3, 4), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)


class TestSelectComputeStats(unittest.TestCase):
    def test_basic(self):
        spec = SelectSpec(
            select= {
                1: 0,
                2: 1
            }
        )

        input_specs = [TensorSpec(shape=(5, 2, 3, 4), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)
        out_stats = spec.compute_stats(input_specs)

        self.assertEqual(out_stats.flops, 0)
        self.assertEqual(out_stats.reads, out_spec.size())
        self.assertEqual(out_stats.writes, out_spec.size())