import unittest

import torch
import torch.nn as nn

from src.tensor_dag import TensorDAG


class BinaryElementwise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x + y
        x = x - y
        x = x * y
        x = x / y
        return x, y


class TestTorchBinaryElementwise(unittest.TestCase):
    def setUp(self):
        self.model = BinaryElementwise()
        self.example_input = (
            torch.rand((10, 20, 30, 40)),
            torch.rand((10, 20, 30, 40)) + 0.01,
        )
        self.traced_model = torch.jit.trace(self.model, self.example_input)
        self.tdag = TensorDAG.from_torchscript(self.traced_model)

    def test_reconstructed_output(self):
        out_model = self.tdag.to_torchscript()
        out_ref = self.traced_model(*self.example_input)
        out_test = out_model(*self.example_input)
        self.assertEqual(len(out_ref), len(out_test))
        for ref, test in zip(out_ref, out_test):
            self.assertTrue(torch.allclose(ref, test))
