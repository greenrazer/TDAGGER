from enum import Enum, auto
from typing import Dict, List, Tuple, Union
from .op_type import OpType

class ELUType(OpType):
    def __str__(self) -> str:
        inp_name = self.inputs["input"]
        out = f"%{self.name}: {self.type}(%{inp_name})"
        if len(self.debug_sources) > 0:
            out += f' #{self.debug_sources[0][2]}( "{self.debug_sources[0][0]}", line {self.debug_sources[0][1]} )'
        return out
    
    @property
    def type(self) -> str:
        return "elu"

    @property
    def required_input_keys(self) -> List[str]:
        return ["input", "alpha"]
