from abc import ABC, abstractmethod

from ..graph.dag_graph_builder import DAGGraphBuilder


class Canonicalizer(ABC):
    @abstractmethod
    def build_graph(self, graph_builder: DAGGraphBuilder):
        pass