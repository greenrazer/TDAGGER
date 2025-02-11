from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Union


class OpInput(ABC):
    inputs: Dict[str, Union[str, List[str]]]
    unique_indices: Set[str]

    def __init__(self, inputs: Dict[str, Union[str, List[str]]]):
        self.inputs = inputs
        self.unique_indices = set()
        self._process_inputs()

    def _process_inputs(self):
        for _, val in self.inputs.items():
            if isinstance(val, list):
                for index in val:
                    self.unique_indices.add(index)
            elif isinstance(val, str):
                index = val
                self.unique_indices.add(index)
            else:
                raise Exception("inputs can only be a reference or list of references")
