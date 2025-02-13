# SafeGraph

> [!WARNING]
> This is a work in progress

## Example

```
self.model = Model()
self.example_input = torch.rand((10, 20, 30, 40))

self.traced_model = torch.jit.trace(self.model, self.example_input)
self.safe_dag = SafeDAG.from_torchscript(self.traced_model)
reconstructed_model = self.safe_dag.to_torchscript()
assert torch.allclose(self.traced_model(self.example_input), reconstructed_model(self.example_input))
```

## Progress

| Op | Status |
|-------------|-----------|
| torch -> IR     | 🟨        |
| IR -> torch    | 🟨        |
| flop counts      | 🟨        |
| memory read counts      | 🟨        |
| memory write counts      | 🟨        |
| batch dimension removal     | ❌        |

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
| gaussian    | ✅        |
| inv_gaussian | ✅       |
| sign        | ✅        |

### Binary Ops

| Op  | Supported |
|-------------|-----------|
| add         | ✅        |
| multiply    | ✅        |
| exponentiate | ❌       |

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
| Einsum      | ❌        |
| Convolution | ❌        |
| Transpose Convolution | ❌ |
| Concat | ❌ |
| Stack | ❌ |

## Architecture

[architecture](ARCHITECTURE.md)