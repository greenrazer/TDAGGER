import torch

from .graph import DAGGraphBuilder, DAGGraph

from .converters.reifiers import TorchReifier
from .converters.canonicalizers import TorchCanonicalizer


class SafeDAG:
    def __init__(self, graph: DAGGraph):
        self.graph = graph

    @staticmethod
    def from_torchscript(model: torch.jit.ScriptModule) -> "SafeDAG":
        canonicalizer = TorchCanonicalizer(model)
        graph_builder = DAGGraphBuilder()
        canonicalizer.build_graph(graph_builder)
        graph = graph_builder.build()
        return SafeDAG(graph)

    def to_torchscript(self) -> torch.nn.Module:
        reifier = TorchReifier(self.graph)
        return reifier.export()
