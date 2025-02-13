import math
import unittest

from src.ir.safe_ir import BinaryElementwiseSpec, DataType, TensorSpec


class TestBinaryElementwiseOutputSpec(unittest.TestCase):
    def test_basic(self):
        spec = BinaryElementwiseSpec(
            op_type=BinaryElementwiseSpec.BinaryElementwiseType.ADD
        )

        input_specs = [
            TensorSpec(shape=[1, 2, 3, 4, 5, 6, 7, 8], data_type=DataType.FLOAT32),
            TensorSpec(shape=[1, 2, 3, 4, 5, 6, 7, 8], data_type=DataType.FLOAT32),
        ]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_raises_on_incorrect_sizes(self):
        spec = BinaryElementwiseSpec(
            op_type=BinaryElementwiseSpec.BinaryElementwiseType.ADD
        )

        input_specs = [
            TensorSpec(shape=[1, 2, 3, 4, 5, 6, 7, 8], data_type=DataType.FLOAT32),
            TensorSpec(shape=[1, 2, 3, 4, 5, 6, 7], data_type=DataType.FLOAT32),
        ]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)


class TestGroupComputeStats(unittest.TestCase):
    def test_basic(self):
        spec = BinaryElementwiseSpec(
            op_type=BinaryElementwiseSpec.BinaryElementwiseType.ADD
        )

        input_specs = [
            TensorSpec(shape=[1, 2, 3, 4, 5, 6, 7, 8], data_type=DataType.FLOAT32),
            TensorSpec(shape=[1, 2, 3, 4, 5, 6, 7, 8], data_type=DataType.FLOAT32),
        ]

        out_spec = spec.compute_stats(input_specs)

        inp_size = math.prod(input_specs[0].shape)
        self.assertEqual(out_spec.flops, inp_size)
        self.assertEqual(out_spec.reads, 2 * inp_size)
        self.assertEqual(out_spec.writes, inp_size)
