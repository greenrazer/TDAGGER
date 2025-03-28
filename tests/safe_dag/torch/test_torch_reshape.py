import unittest

import torch
import torch.nn as nn

from src.tensor_dag import TensorDAG


class ReshapeModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(4, 16, 2, 32, 4)
        x = x.reshape(4, 32, 32, 4)
        x = x.reshape(4, 32, 32, 2, 2)
        x = x.reshape(4, 32, 32, 4)
        x = x.reshape(2, 2, 32, 32, 4)
        x = x.reshape(4, 32, 32, 4)
        x = x.reshape(4, -1, 8, 16)
        x = x.reshape(8, 16, -1)
        x = x.reshape(2, 4, 8, -1)
        x = x.reshape(-1, 32)
        return x


class TestTorchReshape(unittest.TestCase):
    def setUp(self):
        self.model = ReshapeModule()
        self.example_input = torch.rand((4, 32, 32, 4))

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
