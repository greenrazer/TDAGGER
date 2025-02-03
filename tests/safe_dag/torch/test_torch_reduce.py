import unittest

import torch
import torch.nn as nn

from src.safe_dag import SafeDAG


class SimpleReduce(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # a b c d e f g -> c d e f g | sum
        x = torch.sum(x, dim=(0, -1))
        # c d e f g -> c e f g| mean
        x = torch.mean(x, dim=(1, -2))
        # c e f g -> c e g| max
        x = torch.amax(x, dim=2)
        # c e g -> c e| min
        x = torch.amin(x, dim=-2, keepdim=True)
        # c e-> c| prod
        x = torch.prod(x, dim=-1, keepdim=True)

        return x


class TestTorchReduce(unittest.TestCase):
    def setUp(self):
        self.model = SimpleReduce()
        self.example_input = torch.rand((2, 32, 32, 3, 4, 5, 6, 7))

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


if __name__ == "__main__":
    unittest.main()
