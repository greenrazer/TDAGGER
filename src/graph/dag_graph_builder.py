from typing import Dict, List

from ..ir.safe_ir import SpecType, DataHolderType, OpType
from .dag_graph import DAGGraph


class DAGGraphBuilder:
    def __init__(self):
        self.inputs: Dict[str, SpecType] = {}
        self.constants: Dict[str, DataHolderType] = {}
        self.parameters: Dict[str, DataHolderType] = {}
        self.buffers: Dict[str, DataHolderType] = {}
        self.ops: Dict[str, OpType] = {}
        self.outputs: List[str] = []

    def add_input(self, name: str, value: SpecType):
        if name in self.inputs:
            raise Exception(f"Input name is not unique: {name}.")

        self.inputs[name] = value

    def add_constant(self, name: str, value: DataHolderType):
        if name in self.constants:
            raise Exception(f"Constant name is not unique: {name}.")

        self.constants[name] = value

    def add_parameter(self, name: str, value: DataHolderType):
        if name in self.parameters:
            raise Exception(f"Parameter name is not unique: {name}.")

        self.parameters[name] = value

    def add_buffer(self, name: str, value: DataHolderType):
        if name in self.buffers:
            raise Exception(f"Buffer name is not unique: {name}.")

        self.buffers[name] = value

    def add_op(self, name: str, op: OpType):
        if name in self.ops:
            raise Exception(f"Op name is not unique: {name}.")
        self.ops[name] = op

    def add_output(self, name: str):
        if (
            name not in self.inputs
            and name not in self.constants
            and name not in self.parameters
            and name not in self.buffers
            and name not in self.ops
        ):
            raise Exception(f"Output name not in graph: {name}.")

        self.outputs.append(name)

    def build(self) -> DAGGraph:
        return DAGGraph(
            self.inputs,
            self.constants,
            self.parameters,
            self.buffers,
            self.ops,
            self.outputs,
        )
