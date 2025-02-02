import unittest

import torch
import torch.nn as nn

from src.safe_dag import SafeDAG


class Basic(nn.Module):
    def __init__(self):
        super().__init__()

        self.to_add = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.register_buffer("my_buffer", torch.ones(40) * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.to_add + self.my_buffer


class TestTorchBasic(unittest.TestCase):
    def setUp(self):
        self.model = Basic()
        self.example_input = torch.rand((10, 20, 30, 40))

        self.traced_model = torch.jit.trace(self.model, self.example_input)
        self.safe_dag = SafeDAG.from_torchscript(self.traced_model)

    def test_reconstructed_output(self):
        out_model = self.safe_dag.to_torchscript()
        self.assertTrue(
            torch.allclose(
                self.traced_model(self.example_input), out_model(self.example_input)
            )
        )
