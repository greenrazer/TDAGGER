import unittest

import torch
import torch.nn as nn

from src.safe_dag import SafeDAG


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
        self.example_input = torch.rand((4, 1, 32, 1))

        self.traced_model = torch.jit.trace(self.model, self.example_input)
        self.safe_dag = SafeDAG.from_torchscript(self.traced_model)

    def test_reconstructed_output(self):
        out_model = self.safe_dag.to_torchscript()
        original_output = self.traced_model(self.example_input)
        reconstructed_output = out_model(self.example_input)

        self.assertTrue(
            torch.allclose(original_output, reconstructed_output),
            f"Outputs differ: original shape {original_output.shape}, "
            f"reconstructed shape {reconstructed_output.shape}",
        )
