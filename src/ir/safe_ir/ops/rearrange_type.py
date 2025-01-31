from .op_type import OpType

class RearrangeSpec:
    pass


class RearrangeType(OpType):
    spec: RearrangeSpec

    def type(self) -> str:
        return "Rearrange"
    
    def assert_valid(self):
        pass