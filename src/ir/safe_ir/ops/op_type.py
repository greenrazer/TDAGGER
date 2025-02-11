from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Union

from .inputs.op_input import OpInput
from .specs.op_spec import OpSpec


class OpType(ABC):
    name: str

    spec: OpSpec
    input: OpInput

    debug_sources: List[Tuple[str, str, str]] = []

    def __init__(
        self,
        name: str,
        spec: OpSpec,
        input: OpInput,
        debug_sources: List[Tuple[str, str, str]] = [],
    ):
        self.name = name
        self.spec = spec
        self.input = input
        if not isinstance(self.input, self.spec.input_type):
            raise Exception(
                f"input {type(self.input)} not valid for spec for {name}({self.spec.input_type})"
            )
        self.debug_sources = debug_sources

    def debug_sources_to_str(self) -> str:
        if len(self.debug_sources) > 0:
            return f' #{self.debug_sources[0][2]}( "{self.debug_sources[0][0]}", line {self.debug_sources[0][1]} )'
        else:
            return ""

    def __str__(self):
        return f"%{self.name}: {self.spec.format_input(self.input)}{self.debug_sources_to_str()}"
