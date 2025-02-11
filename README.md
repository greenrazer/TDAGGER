# SafeGraph

> [!WARNING]
> This is a work in progress

## Architecture

[architecture](ARCHITECTURE.md)

## Example

```
self.model = Model()
self.example_input = torch.rand((10, 20, 30, 40))

self.traced_model = torch.jit.trace(self.model, self.example_input)
self.safe_dag = SafeDAG.from_torchscript(self.traced_model)
reconstructed_model = self.safe_dag.to_torchscript()
assert torch.allclose(self.traced_model(self.example_input), reconstructed_model(self.example_input))
```