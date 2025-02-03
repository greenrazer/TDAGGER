import unittest

import torch
import torch.nn as nn

from src.safe_dag import SafeDAG


class SimplePermute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = x.permute(0, 2, 3, 1)
        x = x.permute(1, 2, 3, 0)
        x = x.permute(2, 3, 0, 1)
        x = x.permute(3, 0, 1, 2)
        x = x.permute(0, 1, 2, 3)
        return x


class TestTorchPermute(unittest.TestCase):
    def setUp(self):
        self.model = SimplePermute()
        # Create example input: batch_size=2, height=32, width=32, channels=3
        self.example_input = torch.rand((2, 32, 32, 3))

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


class Permute7D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 2, 3, 4, 5, 6, 0)
        x = x.permute(2, 3, 4, 5, 6, 0, 1)
        x = x.permute(3, 4, 5, 6, 0, 1, 2)
        x = x.permute(4, 5, 6, 0, 1, 2, 3)
        x = x.permute(5, 6, 0, 1, 2, 3, 4)
        x = x.permute(6, 0, 1, 2, 3, 4, 5)
        x = x.permute(0, 1, 2, 3, 4, 5, 6)
        return x


class TestTorchPermute7D(unittest.TestCase):
    def setUp(self):
        self.model = Permute7D()
        self.example_input = torch.rand((2, 32, 32, 3, 4, 5, 6))

        self.traced_model = torch.jit.trace(self.model, self.example_input)
        self.safe_dag = SafeDAG.from_torchscript(self.traced_model)

    def test_reconstructed_output(self):
        out_model = self.safe_dag.to_torchscript()
        original_output = self.traced_model(self.example_input)
        reconstructed_output = out_model(self.example_input)

        self.assertTrue(
            torch.allclose(original_output, reconstructed_output),
            f"Outputs differ: original shape {original_output.shape}, ",
        )
