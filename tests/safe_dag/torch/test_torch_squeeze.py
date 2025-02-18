import unittest

import torch
import torch.nn as nn

from src.tensor_dag import TensorDAG


class SqueezeModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze((0,2))
        x = x.unsqueeze(0)
        x = x.unsqueeze(2)
        x = x.squeeze(2)
        x = x.squeeze(0)
        x = x.unsqueeze(0)
        x = x.unsqueeze(2)
        x = x.squeeze()
        return x


class TestTorchSqueeze(unittest.TestCase):
    def setUp(self):
        self.model = SqueezeModule()
        self.example_input = torch.rand((1, 4, 1, 32))

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
