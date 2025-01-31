from .op_type import OpType

class RepeatSpec:
    pass


class RepeatType(OpType):
    spec: RepeatSpec

    def type(self) -> str:
        return "Repeat"
    
    def assert_valid(self):
        pass