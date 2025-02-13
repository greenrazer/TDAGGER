import math
import unittest

from src.ir.safe_ir import DataType, SqueezeSpec, TensorSpec


class TestSqueezeOutputSpec(unittest.TestCase):
    def test_basic(self):
        spec = SqueezeSpec(dimensions={1, 2})

        input_specs = [TensorSpec(shape=(5, 1, 1, 5), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [5, 5])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_negitive_dimensions(self):
        spec = SqueezeSpec(dimensions={1, -2})

        input_specs = [TensorSpec(shape=(5, 1, 1, 5), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [5, 5])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_raises_on_insufficient_shape(self):
        spec = SqueezeSpec(dimensions={1, 4})

        input_specs = [TensorSpec(shape=(5, 1, 1, 5), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_squeeze_on_non_unit_dimension(self):
        spec = SqueezeSpec(dimensions={0})

        input_specs = [TensorSpec(shape=(5, 1, 1, 5), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)


class TestSqueezeComputeStats(unittest.TestCase):
    def test_basic(self):
        spec = SqueezeSpec(dimensions={1, 2})

        input_specs = [TensorSpec(shape=(5, 1, 1, 5), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)
        out_stats = spec.compute_stats(input_specs)

        self.assertEqual(out_stats.flops, 0)
        self.assertEqual(out_stats.reads, out_spec.size())
        self.assertEqual(out_stats.writes, out_spec.size())
