import unittest

import torch
import torch.nn as nn

from src.tensor_dag import TensorDAG


class Sign(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sign(x)

        return x


class TestTorchUnary(unittest.TestCase):
    def setUp(self):
        self.model = Sign()
        self.example_input = torch.rand((10, 20, 30, 40)) - 0.5

        self.traced_model = torch.jit.trace(self.model, self.example_input)
        self.tdag = TensorDAG.from_torchscript(self.traced_model)

    def test_reconstructed_output(self):
        out_model = self.tdag.to_torchscript()

        self.assertTrue(
            torch.allclose(
                self.traced_model(self.example_input), out_model(self.example_input)
            )
        )
