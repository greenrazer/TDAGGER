import math
import unittest

from src.ir.safe_ir import DataType, GroupSpec, TensorSpec


class TestGroupSpec(unittest.TestCase):
    def test_basic(self):
        # doesn't raise
        GroupSpec(
            groups=[[1, 2, 3], [6, 7]],
            _output_shape_sidecar=[],  # shouldn't need this
        )

    def test_raises_on_overlap(self):
        with self.assertRaises(Exception):
            GroupSpec(
                groups=[[1, 2, 3], [3, 4]],
                _output_shape_sidecar=[],  # shouldn't need this
            )

    def test_raises_on_non_consecutive(self):
        with self.assertRaises(Exception):
            GroupSpec(
                groups=[[1, 2, 5]],
                _output_shape_sidecar=[],  # shouldn't need this
            )

        with self.assertRaises(Exception):
            GroupSpec(
                groups=[[-3, -1]],
                _output_shape_sidecar=[],  # shouldn't need this
            )

    def test_raises_on_mixed_group_dims(self):
        with self.assertRaises(Exception):
            GroupSpec(
                groups=[[1, 2, -3]],
                _output_shape_sidecar=[],  # shouldn't need this
            )


class TestGroupOutputSpec(unittest.TestCase):
    def test_basic(self):
        spec = GroupSpec(
            groups=[[1, 2, 3], [6, 7]],
            _output_shape_sidecar=[],  # shouldn't need this
        )

        input_specs = [
            TensorSpec(shape=(1, 2, 3, 4, 5, 6, 7, 8), data_type=DataType.FLOAT32)
        ]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [1, 24, 5, 6, 56])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_negitive_dimensions(self):
        spec = GroupSpec(
            groups=[[1, 2, 3], [-2, -1]],
            _output_shape_sidecar=[],  # shouldn't need this
        )

        input_specs = [
            TensorSpec(shape=(1, 2, 3, 4, 5, 6, 7, 8), data_type=DataType.FLOAT32)
        ]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [1, 24, 5, 6, 56])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_concrete_negitive_dimensions_appear_before(self):
        spec = GroupSpec(
            groups=[[6, 7], [-7, -6, -5]],
            _output_shape_sidecar=[],  # shouldn't need this
        )

        input_specs = [
            TensorSpec(shape=(1, 2, 3, 4, 5, 6, 7, 8), data_type=DataType.FLOAT32)
        ]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [1, 24, 5, 6, 56])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_raises_on_negitive_overlap(self):
        spec = GroupSpec(
            groups=[[1, 2, 3], [-2, -1]],
            _output_shape_sidecar=[],  # shouldn't need this
        )

        input_specs = [TensorSpec(shape=(1, 2, 3, 4), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_raises_on_insufficient_shape(self):
        spec = GroupSpec(
            groups=[[2, 3, 4]],
            _output_shape_sidecar=[],  # shouldn't need this
        )

        input_specs = [TensorSpec(shape=(1, 2, 3, 4), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)


class TestGroupComputeStats(unittest.TestCase):
    def test_basic(self):
        spec = GroupSpec(
            groups=[[1, 2, 3], [6, 7]],
            _output_shape_sidecar=[],  # shouldn't need this
        )

        input_specs = [
            TensorSpec(shape=(1, 2, 3, 4, 5, 6, 7, 8), data_type=DataType.FLOAT32)
        ]

        out_spec = spec.compute_stats(input_specs)

        self.assertEqual(out_spec.flops, 0)
        self.assertEqual(out_spec.reads, math.prod(input_specs[0].shape))
        self.assertEqual(out_spec.writes, math.prod(input_specs[0].shape))

    def test_raises_on_negitive_overlap(self):
        spec = GroupSpec(
            groups=[[1, 2, 3], [-2, -1]],
            _output_shape_sidecar=[],  # shouldn't need this
        )

        input_specs = [TensorSpec(shape=(1, 2, 3, 4), data_type=DataType.FLOAT32)]

        with self.assertRaises(Exception):
            spec.compute_stats(input_specs)
