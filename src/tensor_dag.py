import torch

from .converters.canonicalizers import TorchCanonicalizer
from .converters.lowerers import TorchLowerer
from .converters.reifiers import TorchReifier
from .graph import DAGGraph, DAGGraphBuilder, Refiner


class TensorDAG:
    def __init__(self, graph: DAGGraph):
        self.graph = graph

    @staticmethod
    def from_torchscript(model: torch.jit.ScriptModule) -> "TensorDAG":
        canonicalizer = TorchCanonicalizer(model)
        graph_builder = DAGGraphBuilder()
        canonicalizer.build_graph(graph_builder)
        graph = graph_builder.build()

        refiner = Refiner()
        refined_graph = refiner.refine(graph)

        return TensorDAG(refined_graph)

    def to_torchscript(self) -> torch.nn.Module:
        lowerer = TorchLowerer()
        lowered_graph = lowerer.lower(self.graph)

        reifier = TorchReifier(lowered_graph)
        return reifier.export()
