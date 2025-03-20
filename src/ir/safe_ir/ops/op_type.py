from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Union

from sympy import Symbol

from .inputs import OpInput, UnaryTensorInput
from .specs import OpSpec
from ..safe_ir import SymbolicTensorSpec, SpecType


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

    def with_removed_dimension(
        self,
        dimension_symbol: Symbol,
        symbolic_input_specs: Dict[str, SpecType],
        symbolic_op_output_specs: Dict[str, SpecType],
    ) -> "OpType":
        if isinstance(self.input, UnaryTensorInput):
            input_name = self.input.input
            if input_name in symbolic_input_specs:
                op_input_spec = symbolic_input_specs[input_name]
            elif input_name in symbolic_op_output_specs:
                op_input_spec = symbolic_op_output_specs[input_name]
            else:
                return self

            if not isinstance(op_input_spec, SymbolicTensorSpec):
                return self
            else:
                dimensions_to_remove = op_input_spec.dimensions_to_remove(
                    dimension_symbol
                )

                return OpType(
                    name=self.name,
                    spec=self.spec.with_removed_dimensions(dimensions_to_remove),
                    input=self.input,
                    debug_sources=self.debug_sources,
                )
        else:
            return self
