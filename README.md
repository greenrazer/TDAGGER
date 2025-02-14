# SafeGraph

Transforming neural networks into portable, analyzable tensor graphs.

> [!WARNING]
> This is an early work in progress.

## Why This Exists

Working with neural network architectures across different frameworks and platforms presents several challenges:
- Analyzing model characteristics (FLOPs, memory usage) is often difficult and framework-specific
- Removing batch dimensions and other model transformations can be hard, and usually involves changing the base model code directly.
- Existing intermediate representations (IRs) typically prioritize performance over portability.

This framework takes a different approach by decomposing neural networks into their fundamental building blocks: Tensor Directed Acyclic Graphs (TDAGs) with clean, top-level control flow.

## Key Features

- **Simple Core Operations**: Reduces complex neural operations to a minimal set of tensor operations based off [einops](https://github.com/arogozhnikov/einops).
- **Framework Agnostic**: Design focuses on portability rather than framework-specific optimizations
- **Easy Analysis**: Clear graph structure makes it simpler to analyze model characteristics
- **Modular Architecture**: Split models into independent Tensor Directed Acyclic Graphs(TDAGs) that can be easily reused and modified
- **Clean Control Flow**: Handles branching and dynamic behavior at the top level, keeping tensor operations pure

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