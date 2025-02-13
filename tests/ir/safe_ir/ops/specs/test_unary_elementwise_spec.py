import math
import unittest

import torch
import torch.nn as nn

from src.ir.safe_ir import DataType, TensorSpec, UnaryElementwiseSpec


class TestUnaryElementwiseOutputSpec(unittest.TestCase):
    def test_basic(self):
        spec = UnaryElementwiseSpec(
            op_type=UnaryElementwiseSpec.UnaryElementwiseType.NEGATIVE
        )

        input_specs = [
            TensorSpec(shape=[1, 2, 3, 4, 5, 6, 7, 8], data_type=DataType.FLOAT32),
        ]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)
