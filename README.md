# SafeGraph

> [!WARNING]
> This is a work in progress

## Example

```
model = Model()
example_input = torch.rand((10, 20, 30, 40))

traced_model = torch.jit.trace(model, example_input)
safe_dag = SafeDAG.from_torchscript(traced_model)

print(safe_dag.graph.total_bytes())
print(safe_dag.graph.total_flops())
print(safe_dag.graph.total_memory_reads())
print(safe_dag.graph.total_memory_writes())

reconstructed_model = safe_dag.to_torchscript()
assert torch.allclose(traced_model(example_input), reconstructed_model(example_input))
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