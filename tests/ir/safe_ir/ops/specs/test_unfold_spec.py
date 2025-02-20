import math
import unittest

from src.ir.safe_ir import DataType, TensorSpec, UnfoldSpec


class TestUnfoldOutputSpec(unittest.TestCase):
    def test_basic(self):
        spec = UnfoldSpec(unfold={2: (3, 2), 3: (5, 2)})

        input_specs = [
            TensorSpec(shape=[2, 3, 33, 31], data_type=DataType.FLOAT32),
        ]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [2, 3, 3 * 16, 5 * 14])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_negative_dimensions(self):
        spec = UnfoldSpec(unfold={-2: (3, 2), 3: (5, 2)})

        input_specs = [
            TensorSpec(shape=[2, 3, 33, 31], data_type=DataType.FLOAT32),
        ]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [2, 3, 3 * 16, 5 * 14])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_raises_on_overlap(self):
        spec = UnfoldSpec(unfold={-1: (3, 2), 3: (5, 2)})

        input_specs = [
            TensorSpec(shape=[2, 3, 33, 30], data_type=DataType.FLOAT32),
        ]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_raises_on_insufficient_shape(self):
        spec = UnfoldSpec(unfold={2: (3, 2), 4: (5, 2)})

        input_specs = [
            TensorSpec(shape=[2, 3, 33, 30], data_type=DataType.FLOAT32),
        ]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)
