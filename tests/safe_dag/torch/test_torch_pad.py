import unittest

import torch
import torch.nn as nn

from src.safe_dag import SafeDAG


class PadModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(x, (1, 2))
        x = torch.nn.functional.pad(x, (1, 2, 3, 4))
        x = torch.nn.functional.pad(x, (4, 3, 2, 1, 2, 3))
        x = torch.nn.functional.pad(x, (4, 3, 2, 1, 2, 3, 4, 5))
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1))
        return x


class TestTorchPadding(unittest.TestCase):
    def setUp(self):
        self.model = PadModule()
        self.example_input = torch.rand((2, 4, 6, 8, 10, 12, 14))

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


class PadNonConstant(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(x, (1, 1, 0, 0), mode="reflect")
        x = torch.nn.functional.pad(x, (0, 0, 2, 2), mode="reflect")
        x = torch.nn.functional.pad(x, (0, 0, 2, 2), mode="replicate")
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 1, 1), mode="circular")
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 3, 3), mode="circular")
        x = torch.nn.functional.pad(x, (1, 1, 3, 0, 0, 3), mode="replicate")
        x = torch.nn.functional.pad(x, (1, 1, 3, 0, 0, 3), mode="reflect")
        return x


class TestTorchPaddingNonConstant(unittest.TestCase):
    def setUp(self):
        self.model = PadNonConstant()
        self.example_input = torch.rand((2, 4, 6, 8))

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
