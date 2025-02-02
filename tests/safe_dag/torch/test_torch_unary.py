import unittest
import torch
import torch.nn as nn

from src.safe_dag import SafeDAG


class Unary(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.neg(x)
        x = torch.abs(x)
        x = torch.sqrt(x)
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

        x = torch.relu(x)
        x = torch.nn.functional.leaky_relu(x)
        x = torch.nn.functional.softplus(x)

        return x


class TestTorchUnary(unittest.TestCase):
    def setUp(self):
        self.model = Unary()
        self.example_input = torch.rand((10, 20, 30, 40))

        self.traced_model = torch.jit.trace(self.model, self.example_input)
        self.safe_dag = SafeDAG.from_torchscript(self.traced_model)

    def test_reconstructed_output(self):
        print(self.safe_dag.graph)
        # print(self.safe_dag.graph.name_registry.keys())
        out_model = self.safe_dag.to_torchscript()
        # print(type(out_model))
        # print(out_model.graph)
        # print(out_model._c.graph)

        # print(type(out_model))  # Let's see what type we're dealing with
        # print(dir(out_model._c))

        print(out_model._forward_function.graph)

        # Try getting all methods
        # print(out_model._c._method_names())  # This will show all available methods

        # If there's a different method name, use that instead
        # graph = out_model._c._get_method("forward").graph
        # print(self.traced_model(self.example_input)[0, 0, 0, :10], out_model(self.example_input)[0, 0, 0, :10])
        self.assertTrue(
            torch.allclose(
                self.traced_model(self.example_input), out_model(self.example_input)
            )
        )
