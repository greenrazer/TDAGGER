from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Union


class OpType(ABC):
    name: str

    inputs: Dict[str, Union[str, List[str]]]
    unique_indices: Set[str]

    debug_sources: List[Tuple[str, str, str]] = []

    def __init__(
        self,
        name: str,
        inputs: Dict[str, Union[str, List[str]]],
        debug_sources: List[Tuple[str, str, str]] = [],
    ):
        self.name = name
        self.inputs = inputs
        self.unique_indices = set()
        self._process_inputs()
        self.debug_sources = debug_sources
        self.assert_valid()

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

    def assert_valid(self):
        for s in self.required_input_keys:
            if s not in self.inputs:
                raise Exception(f"Validation Failed: {s} not found in inputs")

    @property
    @abstractmethod
    def type(self) -> str:
        pass
    
    @property
    @abstractmethod
    def required_input_keys(self) -> List[str]:
        pass
