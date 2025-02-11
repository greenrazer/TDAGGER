import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.safe_dag import SafeDAG


class All(nn.Module):
    def __init__(self):
        super().__init__()

        self.to_add = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.register_buffer("my_buffer", torch.ones(40) * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.to_add + self.my_buffer
        x = x - self.my_buffer
        x = x * self.my_buffer
        x = x / self.my_buffer
        x = F.unfold(x, kernel_size=(2, 2), stride=3, dilation=2, padding=1)
        x = F.fold(
            x, output_size=(30, 40), kernel_size=(2, 2), stride=3, dilation=2, padding=1
        )
        x = x[:, :10, 10:, 10:-10]
        x = torch.nn.functional.pad(x, (1, 1, 2, 0, 0, 2, 1, 1))
        x = x.permute(1, 2, 3, 0)
        x = torch.sum(x, dim=(0, -1))
        x = x.reshape(2, 11, 11, 2)
        x = x.unsqueeze(0)
        x = x.squeeze(0)
        x = -x
        x = torch.abs(x)
        x = torch.sin(x)
        x = torch.nn.functional.leaky_relu(x)

        return x


class TestTorchAll(unittest.TestCase):
    def setUp(self):
        self.model = All()
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
