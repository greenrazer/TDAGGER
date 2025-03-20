import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tensor_dag import TensorDAG


class WithBatch(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + x[0, ...]
        x = x[:, :10, 10:, 10:-10]
        x = torch.nn.functional.pad(x, (1, 1, 2, 0, 0, 2, 1, 1))
        x = x.permute(0, 3, 1, 2)
        x = torch.sum(x, dim=(1, -1))
        x = x.unsqueeze(0)
        x = x.squeeze(0)
        x = -x
        x = torch.abs(x)
        x = torch.sin(x)
        x = torch.nn.functional.leaky_relu(x)

        return x


class WithoutBatch(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + x
        x = x[:10, 10:, 10:-10]
        x = torch.nn.functional.pad(x, (1, 1, 2, 0, 0, 2))
        x = x.permute(2, 0, 1)
        x = torch.sum(x, dim=(0, -1))
        x = x.unsqueeze(0)
        x = x.squeeze(0)
        x = -x
        x = torch.abs(x)
        x = torch.sin(x)
        x = torch.nn.functional.leaky_relu(x)

        return x


class TestTorchBatch(unittest.TestCase):
    def setUp(self):
        self.with_batch_model = WithBatch()
        self.without_batch_model = WithoutBatch()

        self.with_batch_example_input = torch.rand((1, 20, 30, 40))
        self.without_batch_example_input = self.with_batch_example_input.squeeze(0)

        self.with_batch_traced_model = torch.jit.trace(
            self.with_batch_model, self.with_batch_example_input
        )
        self.without_batch_traced_model = torch.jit.trace(
            self.without_batch_model, self.without_batch_example_input
        )

    def test_with_batch_reconstructed_output(self):
        with_batch_tdag = TensorDAG.from_torchscript(self.with_batch_traced_model)
        out_model = with_batch_tdag.to_torchscript()
        self.assertTrue(
            torch.allclose(
                self.with_batch_traced_model(self.with_batch_example_input),
                out_model(self.with_batch_example_input),
            )
        )

    def test_without_batch_reconstructed_output(self):
        with_batch_tdag = TensorDAG.from_torchscript(self.with_batch_traced_model)
        without_batch_tdag = TensorDAG(
            with_batch_tdag.graph.with_removed_dimension("x_1_0")
        )

        out_model = without_batch_tdag.to_torchscript()

        self.assertTrue(
            torch.allclose(
                self.without_batch_traced_model(self.without_batch_example_input),
                out_model(self.without_batch_example_input),
            )
        )
