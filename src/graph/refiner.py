from typing import Callable, List, Tuple

from .dag_graph import DAGGraph


class Refiner:
    def __init__(self, passes: List[Tuple[str, Callable[[DAGGraph], DAGGraph]]] = []):
        default_passes = [("remove unused constants", self.remove_unused_constants)]
        self.passes = passes if len(passes) > 0 else default_passes

    def refine(self, graph: DAGGraph) -> DAGGraph:
        last_graph = graph
        for name, p in self.passes:
            last_graph = p(last_graph)
        return last_graph

    def remove_unused_constants(self, graph: DAGGraph) -> DAGGraph:
        return graph
