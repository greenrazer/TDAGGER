from .op_type import OpType

class FoldSpec:
    pass


class FoldType(OpType):
    spec: FoldSpec

    def type(self) -> str:
        return "Fold"
    
    def assert_valid(self):
        pass