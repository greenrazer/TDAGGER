import unittest

import torch
import torch.nn as nn

from src.tensor_dag import TensorDAG


class Select(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[0, :, 1, ...]
        x = x[..., 0]
        return x


class TestTorchSelect(unittest.TestCase):
    def setUp(self):
        self.model = Select()
        self.example_input = torch.rand((8, 32, 32, 4, 5, 6, 7))

        self.traced_model = torch.jit.trace(self.model, self.example_input)
        self.tdag = TensorDAG.from_torchscript(self.traced_model)

    def test_reconstructed_output(self):
        out_model = self.tdag.to_torchscript()
        original_output = self.traced_model(self.example_input)
        reconstructed_output = out_model(self.example_input)

        self.assertTrue(
            torch.allclose(original_output, reconstructed_output),
            f"Outputs differ: original shape {original_output.shape}, "
            f"reconstructed shape {reconstructed_output.shape}",
        )
