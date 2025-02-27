import unittest

import torch
import torch.nn as nn

from src.tensor_dag import TensorDAG


class Unary(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = -x
        x = torch.reciprocal(x)
        x = torch.reciprocal(x)
        x = torch.exp(x)
        x = torch.log(x)

        x = torch.sin(x)
        x = torch.cos(x)
        x = torch.tan(x)
        x = torch.arctan(x)
        x = torch.arccos(x)
        x = torch.arcsin(x)
        x = torch.sinh(x)
        x = torch.cosh(x)
        x = torch.tanh(x)
        x = torch.arctanh(x)
        x = torch.arccosh(x)
        x = torch.arcsinh(x)

        x = torch.abs(x)
        x = torch.relu(x)
        x = torch.nn.functional.softplus(x)
        x = torch.nn.functional.leaky_relu(x)

        return x


class TestTorchUnary(unittest.TestCase):
    def setUp(self):
        self.model = Unary()
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
