import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.safe_dag import SafeDAG


class UnfoldFoldModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.unfold(x, kernel_size=(2, 2), stride=3, dilation=2, padding=1)
        # L[d] = (x.shape[d] + 2 * padding[d] - dilation[d] * (kernel_size[d] - 1) - 1) / stride[d] + 1
        # batch, channels * kernel_elements, prod over all d dimensions L[d]
        x = F.fold(
            x, output_size=(32, 32), kernel_size=(2, 2), stride=3, dilation=2, padding=1
        )
        # L_inv[d] = (L[d] - 1) * stride[d] - 2 * padding[d] + dilation[d] * (kernel_size[d] - 1) + 1
        # batch, channels, L_inv[0], L_inv[1]
        return x


class TestTorchUnfoldFold3D(unittest.TestCase):
    def setUp(self):
        self.model = UnfoldFoldModule()
        self.example_input = torch.rand((3, 32, 32))

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


class TestTorchUnfoldFold4D(unittest.TestCase):
    def setUp(self):
        self.model = UnfoldFoldModule()
        self.example_input = torch.rand((4, 3, 32, 32))

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
