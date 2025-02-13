import math
import unittest

from src.ir.safe_ir import DataType, SliceSpec, TensorSpec


class TestSliceOutputSpec(unittest.TestCase):
    def test_basic(self):
        spec = SliceSpec(slice={1: (0, 1), 2: (1, 3)})

        input_specs = [TensorSpec(shape=(5, 5, 5, 5), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [5, 2, 3, 5])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_negitive_dimensions(self):
        spec = SliceSpec(slice={1: (0, 1), -2: (1, 3)})

        input_specs = [TensorSpec(shape=(5, 5, 5, 5), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [5, 2, 3, 5])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_negitive_slices(self):
        spec = SliceSpec(slice={1: (1, -2), -2: (-3, -2)})

        input_specs = [TensorSpec(shape=(5, 5, 5, 5), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [5, 3, 2, 5])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_raises_on_overlapping_dimensions(self):
        spec = SliceSpec(slice={2: (0, 1), -2: (1, 3)})

        input_specs = [TensorSpec(shape=(5, 5, 5, 5), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_raises_on_insufficient_shape(self):
        spec = SliceSpec(slice={1: (0, 1), 4: (1, 3)})

        input_specs = [TensorSpec(shape=(5, 5, 5, 5), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_raises_on_begin_out_of_bounds(self):
        spec = SliceSpec(slice={1: (30, 40), 2: (1, 3)})

        input_specs = [TensorSpec(shape=(5, 5, 5, 5), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_raises_on_end_out_of_bounds(self):
        spec = SliceSpec(slice={1: (0, 10), 2: (1, 3)})

        input_specs = [TensorSpec(shape=(5, 5, 5, 5), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_raises_on_end_before_begin(self):
        spec = SliceSpec(slice={2: (-1, -2)})

        input_specs = [TensorSpec(shape=(5, 5, 5, 5), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)


class TestSliceComputeStats(unittest.TestCase):
    def test_basic(self):
        spec = SliceSpec(slice={1: (0, 1), 2: (1, 3)})

        input_specs = [TensorSpec(shape=(5, 5, 5, 5), data_type=DataType.FLOAT32)]

        out_spec = spec.output_spec(input_specs)
        out_stats = spec.compute_stats(input_specs)

        self.assertEqual(out_stats.flops, 0)
        self.assertEqual(out_stats.reads, out_spec.size())
        self.assertEqual(out_stats.writes, out_spec.size())
