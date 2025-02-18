# TDAGGER

Transforming neural networks into portable, analyzable tensor graphs.

> [!WARNING]
> This is an early work in progress.

## Why This Exists

Working with neural network architectures across different frameworks and platforms presents several challenges:
- Analyzing model characteristics (FLOPs, memory usage) is often difficult and framework-specific.
- Extracting subgraphs for reuse is often challenging due to complicated model architectures and lack of standardized tools.
- Removing batch dimensions and other model transformations can be hard, and usually involves changing the base model code directly.
- Existing intermediate representations (IRs) typically prioritize performance over portability.

This framework takes a different approach by decomposing neural networks into their fundamental building blocks: Tensor Directed Acyclic Graphs (TDAGs) with clean, top-level control flow.

## Key Features

- **Framework Agnostic**: Design focuses on portability rather than framework-specific optimizations.
- **Simple Core Operations**: Reduces complex neural operations to a minimal set of portable tensor operations.
- **Easy Analysis**: Clear graph structure makes it simpler to analyze model characteristics.
- **Modular Architecture**: Split models into independent Tensor Directed Acyclic Graphs(TDAGs) that can be easily reused and modified.
- **Clean Control Flow**: Handles branching and dynamic behavior at the top level, keeping tensor operations pure.

## Example

```
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.to_add = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.register_buffer("my_buffer", torch.ones(48) * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.to_add + self.my_buffer
        x = x - self.my_buffer
        x = x * self.my_buffer
        x = x / self.my_buffer
        x = F.unfold(x, kernel_size=(2, 2), stride=3, dilation=1, padding=1)
        x = F.fold(
            x, output_size=(30, 48), kernel_size=(2, 2), stride=3, dilation=1, padding=1
        )
        x = x[:, :10, 10:, 10:-10]
        x = torch.nn.functional.pad(x, (1, 1, 2, 0, 0, 2, 1, 1))
        x = x.permute(1, 2, 3, 0)
        x = torch.sum(x, dim=(0, -1))
        x = x.reshape(2, 11, 15, 2)
        x = x.unsqueeze(0)
        x = x.squeeze(0)
        x = -x
        x = torch.abs(x)
        x = torch.sin(x)
        x = torch.nn.functional.leaky_relu(x)

        return x

model = Model()
example_input = torch.rand((10, 20, 30, 48))

traced_model = torch.jit.trace(model, example_input)
safe_dag = SafeDAG.from_torchscript(traced_model)

print(safe_dag.graph.total_bytes())
print(safe_dag.graph.total_flops())
print(safe_dag.graph.total_memory_reads())
print(safe_dag.graph.total_memory_writes())

reconstructed_model = safe_dag.to_torchscript()
assert torch.allclose(traced_model(example_input), reconstructed_model(example_input))
```

## Deinop Overview

[Deinops](docs/DEINOPS.md)

## Architecture

[Architecture](docs/ARCHITECTURE.md)

## Progress

| Part | Status |
|-------------|-----------|
| torch -> IR     | 🟨        |
| IR -> torch    | 🟨        |
| flop counts      | 🟨        |
| memory read counts      | 🟨        |
| memory write counts      | 🟨        |
| Shape Propagation   | 🟨        |
| Serialization/Deserialization    | ❌        |
| batch dimension removal     | ❌        |
| CoreML -> IR     | ❌        |
| IR -> CoreML    | ❌        |
| Control Flow     | ❌        |

### Tensor Ops

| Op | Status |
|-------------|-----------|
| permute     | ✅        |
| group       | ✅        |
| ├── squeeze | ✅        |
| ungroup     | ✅        |
| ├── unsqueeze | ✅      |
| slice       | ✅        |
| pad         | ✅        |
| reduce      | 🟨        |
| ├── select  | ✅        |
| repeat      | ✅        |
| fold        | ✅        |
| unfold      | ✅        |

### Unary Ops

| Op   | Supported |
|-------------|-----------|
| neg         | ✅        |
| reciprocal  | ✅        |
| exp         | ✅        |
| log         | ✅        |
| sin         | ✅        |
| arcsin      | ✅        |
| cos         | ✅        |
| arccos      | ✅        |
| tan         | ✅        |
| arctan      | ✅        |
| sinh        | ✅        |
| arcsinh     | ✅        |
| cosh        | ✅        |
| arccosh     | ✅        |
| tanh        | ✅        |
| arctanh     | ✅        |
| gaussian    | ❌        |
| inv_gaussian | ❌       |
| sign        | ✅        |

### Binary Ops

| Op  | Supported |
|-------------|-----------|
| add         | ✅        |
| multiply    | ✅        |

### Derived Unary Ops

|  Op  | Supported |
|-------------|-----------|
| Stride      | ✅        |
| Reshape     | ✅        |
| torch repeat | ✅       |
| repeat interleave | ❌  |
| relu        | ✅        |
| leakyrelu   | ✅        |
| softplus    | ✅        |
| sigmoid     | ❌        |
| normalize     | ❌        |
| softmax     | ❌        |

### Derived Binary Ops

|  Op  | Supported |
|-------------|-----------|
| Exponentiate | ❌       |
| Einsum      | ❌        |
| Convolution | ❌        |
| Transpose Convolution | ❌ |
| Concat | ❌ |
| Stack | ❌ |