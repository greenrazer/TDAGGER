import torch

from .graph import DAGGraphBuilder, DAGGraph

from .exporters import TorchExporter
from .importers import TorchImporter


class SafeDAG:
    def __init__(self, graph: DAGGraph):
        self.graph = graph

    @staticmethod
    def from_torchscript(model: torch.jit.ScriptModule) -> "SafeDAG":
        importer = TorchImporter(model)
        graph_builder = DAGGraphBuilder()
        importer.build_graph(graph_builder)
        graph = graph_builder.build()
        return SafeDAG(graph)

    def to_torchscript(self) -> torch.nn.Module:
        exporter = TorchExporter(self.graph)
        return exporter.export()
