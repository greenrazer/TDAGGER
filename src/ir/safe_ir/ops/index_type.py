from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple, Union, Set

from .op_type import OpType
    

@dataclass
class IndexSpec:
    index: Dict[int, Union[int, Tuple[int, int, int]]] # slice end indexes are inclusive

    def __str__(self):
        pos_dims = sorted([d for d in self.index if d >= 0])
        neg_dims = sorted([d for d in self.index if d < 0])
        
        parts = []
        prev_included = False
        prev_num = None
        
        # Process positive dimensions first
        if pos_dims:
            for i in range(pos_dims[0], pos_dims[-1] + 1):
                if i in pos_dims:
                    if prev_included and parts[-1] != "..." and i != prev_num + 1:
                        parts.append("...")
                    index_val = self.index[i]
                    if isinstance(index_val, tuple):
                        match index_val:
                            case (0, -1, 1):
                                index_str = "[:]"
                            case (begin, -1, 1):
                                index_str = f"[{begin}:]"
                            case (0, end, 1):
                                index_str = f"[:{end}]"
                            case (begin, -1, step):
                                index_str = f"[{begin}::{step}]"
                            case (0, end, step):
                                index_str = f"[:{end}:{step}]"
                            case (0, -1, step):
                                index_str = f"[::{step}]"
                            case (begin, end, step):
                                index_str = f"[{begin}:{end}:{step}]"
                        parts.append(f"{i}{index_str}")
                    else:
                        parts.append(f"{i}[{index_val}]")
                    prev_included = True
                    prev_num = i
                elif prev_included:
                    parts.append("...")
                    prev_included = False
        
        # Then process negative dimensions
        if neg_dims:
            prev_included = False
            prev_num = None
            if parts and parts[-1] != "...":
                parts.append("...")
                
            for i in range(neg_dims[0], neg_dims[-1] - 1, -1):
                if i in neg_dims:
                    if prev_included and parts[-1] != "..." and i != prev_num - 1:
                        parts.append("...")
                    if isinstance(index_val, tuple):
                        match index_val:
                            case (0, -1, 1):
                                index_str = "[:]"
                            case (begin, -1, 1):
                                index_str = f"[{begin}:]"
                            case (0, end, 1):
                                index_str = f"[:{end}]"
                            case (begin, -1, step):
                                index_str = f"[{begin}::{step}]"
                            case (0, end, step):
                                index_str = f"[:{end}:{step}]"
                            case (0, -1, step):
                                index_str = f"[::{step}]"
                            case (begin, end, step):
                                index_str = f"[{begin}:{end}:{step}]"
                        parts.append(f"{i}{index_str}")
                    else:
                        parts.append(f"{i}[{index_val}]")
                    prev_included = True
                    prev_num = i
                elif prev_included:
                    parts.append("...")
                    prev_included = False
        
        # Add leading ellipsis if not starting at 0
        if not pos_dims or pos_dims[0] != 0:
            parts.insert(0, "...")
        
        # Add trailing ellipsis if not ending in -1
        if not neg_dims or neg_dims[-1] != -1:
            parts.append("...")
        
        return " ".join(parts)

class IndexType(OpType):
    spec: IndexSpec

    def __init__(self, name, inputs, spec: IndexSpec, debug_sources=[]):
        super().__init__(name, inputs, debug_sources=debug_sources)
        self.spec = spec

    def __str__(self) -> str:
        inp_name = self.inputs["input"]
        out = f"%{self.name}: %{inp_name}[{self.spec}]{self.debug_sources_to_str()}"
        return out

    @property
    def type(self) -> str:
        return "index"

    @property
    def required_input_keys(self) -> List[str]:
        return ["input"]