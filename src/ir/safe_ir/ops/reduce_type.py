from .op_type import OpType

class ReduceSpec:
    pass


class ReduceType(OpType):
    spec: ReduceSpec

    def type(self) -> str:
        return "Reduce"
    
    def assert_valid(self):
        pass