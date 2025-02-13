import math
import unittest

import torch
import torch.nn as nn

from src.ir.safe_ir import FoldSpec, DataType, TensorSpec


class TestFoldOutputSpec(unittest.TestCase):
    def test_basic(self):
        spec = FoldSpec(
            fold={
                2: (3, 2),
                3: (5, 2)
            },
            _output_shape_sidecar=[] # doesn't matter
        )

        input_specs = [
            TensorSpec(shape=[2, 3, 3*16, 5*14], data_type=DataType.FLOAT32),
        ]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [2, 3, 33, 31])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_negative_dimensions(self):
        spec = FoldSpec(
            fold={
                -2: (3, 2),
                3: (5, 2)
            },
            _output_shape_sidecar=[] # doesn't matter
        )

        input_specs = [
            TensorSpec(shape=[2, 3, 3*16, 5*14], data_type=DataType.FLOAT32),
        ]

        out_spec = spec.output_spec(input_specs)

        self.assertEqual(out_spec.shape, [2, 3, 33, 31])
        self.assertEqual(out_spec.data_type, DataType.FLOAT32)

    def test_raises_on_overlap(self):
        spec = FoldSpec(
            fold={
                -1: (3, 2),
                3: (5, 2)
            },
            _output_shape_sidecar=[] # doesn't matter
        )

        input_specs = [
            TensorSpec(shape=[2, 3, 32, 32], data_type=DataType.FLOAT32),
        ]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)

    def test_raises_on_insufficient_shape(self):
        spec = FoldSpec(
            fold={
                2: (3, 2),
                4: (5, 2)
            },
            _output_shape_sidecar=[] # doesn't matter
        )

        input_specs = [
            TensorSpec(shape=[2, 3, 32, 32], data_type=DataType.FLOAT32),
        ]

        with self.assertRaises(Exception):
            spec.output_spec(input_specs)