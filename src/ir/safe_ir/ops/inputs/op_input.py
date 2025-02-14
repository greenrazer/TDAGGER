from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Union


class OpInput(ABC):
    inputs: Dict[str, Union[str, List[str]]]
    unique_indices: Set[str]

    def __init__(self, inputs: Dict[str, Union[str, List[str]]]):
        self.inputs = inputs
        self.flattened_indices = self._flatten_inputs()
        self.unique_indices = set(self.flattened_indices)

    def _flatten_inputs(self):
        flattened = []
        for _, val in self.inputs.items():
            if isinstance(val, list):
                for index in val:
                    flattened.append(index)
            elif isinstance(val, str):
                index = val
                flattened.append(index)
            else:
                raise Exception("inputs can only be a reference or list of references")
        return flattened
